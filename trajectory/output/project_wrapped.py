"""Project Wrapped — narrative insight cards for a single project.

Mines the trajectory DB for surprising/interesting facts about a project's
development history and renders them as a shareable HTML page.
No LLM calls — pure SQL aggregation + template rendering.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)


# --- Insight data ---

@dataclass
class InsightCard:
    """One insight card."""
    title: str  # e.g. "Your Most Intense Day"
    stat: str  # e.g. "47 events"
    detail: str  # e.g. "Feb 10 — you introduced 55 new concepts..."
    icon: str = ""  # emoji for visual interest
    color: str = "#58a6ff"  # accent color


@dataclass
class WrappedData:
    """All data for a project wrapped page."""
    project_name: str
    date_range: str
    total_events: int
    total_concepts: int
    total_decisions: int
    total_sessions: int
    span_days: int
    cards: list[InsightCard] = field(default_factory=list)
    # Mini heatmap data (reuses concept_heatmap data model)
    heatmap_html: str = ""


def compute_wrapped_data(db: TrajectoryDB, project_id: int) -> WrappedData:
    """Mine the DB for interesting insights about a project."""
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    cards: list[InsightCard] = []

    # --- Basic stats ---
    total_events = db.conn.execute(
        "SELECT COUNT(*) as c FROM events WHERE project_id=?", (project_id,),
    ).fetchone()["c"]

    unique_concepts = db.conn.execute(
        "SELECT COUNT(DISTINCT ce.concept_id) as c FROM concept_events ce "
        "JOIN events e ON ce.event_id=e.id WHERE e.project_id=?", (project_id,),
    ).fetchone()["c"]

    total_decisions = db.conn.execute(
        "SELECT COUNT(*) as c FROM decisions WHERE project_id=?", (project_id,),
    ).fetchone()["c"]

    sessions = db.get_sessions(project_id=project_id, limit=10000)

    # Date range
    ts_range = db.conn.execute(
        "SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM events WHERE project_id=?",
        (project_id,),
    ).fetchone()
    first_date = ts_range["first"][:10] if ts_range["first"] else "?"
    last_date = ts_range["last"][:10] if ts_range["last"] else "?"
    try:
        span_days = (datetime.fromisoformat(ts_range["last"]) - datetime.fromisoformat(ts_range["first"])).days
    except (ValueError, TypeError):
        span_days = 0

    # --- Card 1: Most Intense Day ---
    busiest = db.conn.execute("""
        SELECT DATE(timestamp) as day, COUNT(*) as cnt
        FROM events WHERE project_id=?
        GROUP BY DATE(timestamp) ORDER BY cnt DESC LIMIT 1
    """, (project_id,)).fetchone()
    if busiest:
        new_on_day = db.conn.execute("""
            SELECT COUNT(*) as c FROM concepts
            WHERE id IN (SELECT DISTINCT ce.concept_id FROM concept_events ce
                         JOIN events e ON ce.event_id=e.id WHERE e.project_id=?)
            AND DATE(first_seen) = ?
        """, (project_id, busiest["day"])).fetchone()["c"]
        try:
            day_label = datetime.strptime(busiest["day"], "%Y-%m-%d").strftime("%b %-d")
        except ValueError:
            day_label = busiest["day"]
        cards.append(InsightCard(
            title="Most Intense Day",
            stat=f"{busiest['cnt']} events",
            detail=f"{day_label} — {new_on_day} new concepts introduced in a single day",
            icon="🔥",
            color="#f78166",
        ))

    # --- Card 2: Signature Concept ---
    top_concept = db.conn.execute("""
        SELECT c.name, c.level, COUNT(*) as cnt,
               COUNT(DISTINCT DATE(e.timestamp)) as days_active
        FROM concept_events ce
        JOIN concepts c ON ce.concept_id = c.id
        JOIN events e ON ce.event_id = e.id
        WHERE e.project_id=?
        GROUP BY c.id ORDER BY cnt DESC LIMIT 1
    """, (project_id,)).fetchone()
    if top_concept:
        level_label = {"theme": "theme", "design_bet": "design bet", "technique": "technique"}.get(
            top_concept["level"], top_concept["level"]
        )
        cards.append(InsightCard(
            title="Signature Concept",
            stat=top_concept["name"].replace("_", " "),
            detail=f"A {level_label} that appeared in {top_concept['cnt']} events across {top_concept['days_active']} days",
            icon="⭐",
            color="#d2a8ff",
        ))

    # --- Card 3: Thinking Style ---
    levels = db.conn.execute("""
        SELECT c.level, COUNT(DISTINCT c.id) as unique_count
        FROM concept_events ce
        JOIN concepts c ON ce.concept_id = c.id
        JOIN events e ON ce.event_id = e.id
        WHERE e.project_id=?
        GROUP BY c.level
    """, (project_id,)).fetchall()
    level_map = {r["level"]: r["unique_count"] for r in levels}
    themes = level_map.get("theme", 0)
    bets = level_map.get("design_bet", 0)
    techniques = level_map.get("technique", 0)
    total_concepts_counted = themes + bets + techniques
    if total_concepts_counted > 0:
        bet_pct = round(100 * bets / total_concepts_counted)
        tech_pct = round(100 * techniques / total_concepts_counted)
        theme_pct = round(100 * themes / total_concepts_counted)
        if bet_pct >= tech_pct and bet_pct >= theme_pct:
            style = "Architect"
            style_detail = f"{bet_pct}% of your concepts are design bets — you think in systems"
        elif tech_pct >= bet_pct and tech_pct >= theme_pct:
            style = "Builder"
            style_detail = f"{tech_pct}% of your concepts are techniques — you think in implementations"
        else:
            style = "Visionary"
            style_detail = f"{theme_pct}% of your concepts are themes — you think in narratives"
        cards.append(InsightCard(
            title="Your Thinking Style",
            stat=style,
            detail=f"{style_detail}. {themes} themes, {bets} design bets, {techniques} techniques.",
            icon="🧠",
            color="#7ee787",
        ))

    # --- Card 4: Longest-Running Idea ---
    longest = db.conn.execute("""
        SELECT c.name, c.level,
               MIN(DATE(e.timestamp)) as first_day, MAX(DATE(e.timestamp)) as last_day,
               JULIANDAY(MAX(e.timestamp)) - JULIANDAY(MIN(e.timestamp)) as span,
               COUNT(DISTINCT DATE(e.timestamp)) as days_active
        FROM concept_events ce
        JOIN concepts c ON ce.concept_id = c.id
        JOIN events e ON ce.event_id = e.id
        WHERE e.project_id=?
        GROUP BY c.id HAVING span > 0
        ORDER BY span DESC LIMIT 1
    """, (project_id,)).fetchone()
    if longest:
        cards.append(InsightCard(
            title="Longest-Running Idea",
            stat=longest["name"].replace("_", " "),
            detail=f"Spanned {int(longest['span'])} days ({longest['first_day']} → {longest['last_day']}), active on {longest['days_active']} of them",
            icon="🏃",
            color="#58a6ff",
        ))

    # --- Card 5: One-Hit Wonders ---
    one_hits = db.conn.execute("""
        SELECT COUNT(*) as c FROM (
            SELECT ce.concept_id FROM concept_events ce
            JOIN events e ON ce.event_id = e.id
            WHERE e.project_id=?
            GROUP BY ce.concept_id HAVING COUNT(*) = 1
        )
    """, (project_id,)).fetchone()["c"]
    if one_hits > 0 and unique_concepts > 0:
        pct = round(100 * one_hits / unique_concepts)
        cards.append(InsightCard(
            title="One-Hit Wonders",
            stat=f"{one_hits} concepts",
            detail=f"{pct}% of your concepts appeared exactly once and were never revisited",
            icon="💫",
            color="#f0883e",
        ))

    # --- Card 6: Decision Velocity ---
    if total_decisions > 0 and span_days > 0:
        velocity = round(total_decisions / max(span_days, 1), 1)
        top_decision = db.conn.execute("""
            SELECT d.title FROM decisions d
            JOIN events e ON d.event_id = e.id
            WHERE e.project_id=?
            ORDER BY e.timestamp LIMIT 1
        """, (project_id,)).fetchone()
        first_decision = top_decision["title"][:80] if top_decision else ""
        cards.append(InsightCard(
            title="Decision Velocity",
            stat=f"{velocity}/day",
            detail=f"{total_decisions} architectural decisions in {span_days} days. First: \"{first_decision}\"",
            icon="⚡",
            color="#d2a8ff",
        ))

    # --- Card 7: Concept Density ---
    if total_events > 0 and unique_concepts > 0:
        density = round(unique_concepts / max(total_events, 1), 1)
        cards.append(InsightCard(
            title="Concept Density",
            stat=f"{density} per event",
            detail=f"{unique_concepts} unique concepts extracted from {total_events} events — every commit carried ideas",
            icon="🔬",
            color="#7ee787",
        ))

    # --- Card 8: Bold Decision Spotlight ---
    bold = db.conn.execute("""
        SELECT d.title, d.reasoning, d.alternatives, DATE(e.timestamp) as day
        FROM decisions d JOIN events e ON d.event_id = e.id
        WHERE e.project_id=? AND d.alternatives IS NOT NULL
        ORDER BY e.significance DESC LIMIT 1
    """, (project_id,)).fetchone()
    if bold:
        alts = json.loads(bold["alternatives"]) if bold["alternatives"] else []
        alt_text = f" (over: {alts[0][:60]})" if alts else ""
        cards.append(InsightCard(
            title="Boldest Decision",
            stat=bold["title"][:60],
            detail=f"{bold['reasoning'][:120]}{alt_text}",
            icon="🎯",
            color="#f78166",
        ))

    return WrappedData(
        project_name=project.name,
        date_range=f"{first_date} → {last_date}",
        total_events=total_events,
        total_concepts=unique_concepts,
        total_decisions=total_decisions,
        total_sessions=len(sessions),
        span_days=span_days,
        cards=cards,
    )


# --- HTML Rendering ---

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_wrapped_html(data: WrappedData, output_path: Path) -> Path:
    """Render the wrapped page as a self-contained HTML file."""

    cards_html = ""
    for card in data.cards:
        cards_html += f'''
        <div class="card" style="--accent: {card.color}">
            <div class="card-icon">{card.icon}</div>
            <div class="card-title">{_esc(card.title)}</div>
            <div class="card-stat">{_esc(card.stat)}</div>
            <div class="card-detail">{_esc(card.detail)}</div>
        </div>
        '''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(data.project_name)} — Wrapped</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    min-height: 100vh;
}}

/* --- Hero --- */
.hero {{
    text-align: center;
    padding: 60px 24px 40px;
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #21262d;
}}
.hero h1 {{
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #58a6ff, #d2a8ff, #7ee787);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero .date-range {{
    font-size: 16px;
    color: #8b949e;
    margin-bottom: 24px;
}}
.hero-stats {{
    display: flex;
    justify-content: center;
    gap: 40px;
    flex-wrap: wrap;
}}
.hero-stat {{
    text-align: center;
}}
.hero-stat .num {{
    font-size: 36px;
    font-weight: 800;
    color: #e6edf3;
    line-height: 1;
}}
.hero-stat .label {{
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}}

/* --- Cards --- */
.cards {{
    max-width: 720px;
    margin: 0 auto;
    padding: 40px 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}}
.card {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}}
.card:hover {{
    transform: translateY(-2px);
    border-color: var(--accent);
}}
.card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--accent);
    opacity: 0.6;
}}
.card-icon {{
    font-size: 28px;
    margin-bottom: 8px;
}}
.card-title {{
    font-size: 13px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}}
.card-stat {{
    font-size: 28px;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 8px;
    line-height: 1.2;
}}
.card-detail {{
    font-size: 14px;
    color: #8b949e;
    line-height: 1.5;
}}

/* --- Footer --- */
.footer {{
    text-align: center;
    padding: 40px 24px;
    color: #484f58;
    font-size: 12px;
    border-top: 1px solid #21262d;
}}
.footer a {{
    color: #58a6ff;
    text-decoration: none;
}}

/* --- Responsive --- */
@media (max-width: 480px) {{
    .hero h1 {{ font-size: 28px; }}
    .hero-stat .num {{ font-size: 24px; }}
    .hero-stats {{ gap: 20px; }}
    .card-stat {{ font-size: 22px; }}
}}
</style>
</head>
<body>

<div class="hero">
    <h1>{_esc(data.project_name)}</h1>
    <div class="date-range">{_esc(data.date_range)} · {data.span_days} days</div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="num">{data.total_events}</div>
            <div class="label">events</div>
        </div>
        <div class="hero-stat">
            <div class="num">{data.total_concepts}</div>
            <div class="label">concepts</div>
        </div>
        <div class="hero-stat">
            <div class="num">{data.total_decisions}</div>
            <div class="label">decisions</div>
        </div>
        <div class="hero-stat">
            <div class="num">{data.total_sessions}</div>
            <div class="label">sessions</div>
        </div>
    </div>
</div>

<div class="cards">
    {cards_html}
</div>

<div class="footer">
    Generated by <a href="#">trajectory</a> · {datetime.now().strftime("%b %-d, %Y")}
</div>

</body>
</html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("Wrapped page saved to %s", output_path)
    return output_path


def generate_project_wrapped(
    db: TrajectoryDB,
    project_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Generate a Wrapped page for a project. Returns output path."""
    project = db.get_project_by_name(project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found")

    data = compute_wrapped_data(db, project.id)

    if output_dir is None:
        output_dir = Path("data/wrapped")
    output_path = output_dir / f"{project_name}_wrapped.html"

    return render_wrapped_html(data, output_path)
