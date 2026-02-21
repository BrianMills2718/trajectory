"""Project Narrative — scrollytelling visualization of a project's intellectual journey.

Gathers chronological data, asks an LLM to synthesize a structured narrative (phases,
key decisions, concept mentions), then renders a scrollytelling HTML page where:
- Left side: narrative text that fades in paragraph by paragraph
- Right side: Mermaid diagram with progressive reveal synced to scroll position
- Phase transitions shift colors, concept names glow, decisions appear as pull quotes
"""

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from llm_client import call_llm, render_prompt

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _gather_narrative_data(db: TrajectoryDB, project_id: int) -> dict:
    """Gather all data needed for the narrative prompt."""
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    events = db.conn.execute(
        """SELECT e.id, e.timestamp, e.title, e.event_type, e.llm_summary,
                  e.llm_intent, e.significance
           FROM events e WHERE e.project_id = ? ORDER BY e.timestamp""",
        (project_id,),
    ).fetchall()

    concept_events = db.conn.execute(
        """SELECT ce.event_id, c.name, c.level
           FROM concept_events ce
           JOIN concepts c ON ce.concept_id = c.id
           JOIN events e ON ce.event_id = e.id
           WHERE e.project_id = ?""",
        (project_id,),
    ).fetchall()

    event_concepts: dict[int, list[str]] = defaultdict(list)
    concept_levels: dict[str, str] = {}
    for row in concept_events:
        event_concepts[row["event_id"]].append(row["name"])
        concept_levels[row["name"]] = row["level"] or "technique"

    decisions = db.conn.execute(
        """SELECT d.title, d.reasoning, d.alternatives, e.timestamp
           FROM decisions d
           JOIN events e ON d.event_id = e.id
           WHERE e.project_id = ?
           ORDER BY e.timestamp""",
        (project_id,),
    ).fetchall()

    decision_by_day: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        day = d["timestamp"][:10]
        alts = d["alternatives"] or ""
        if alts.startswith("["):
            try:
                alts = "; ".join(json.loads(alts))
            except (json.JSONDecodeError, TypeError):
                pass
        decision_by_day[day].append({
            "title": d["title"],
            "reasoning": d["reasoning"],
            "alternatives": alts,
        })

    concepts = db.conn.execute(
        """SELECT c.name, c.level, c.first_seen, c.last_seen, c.importance, c.lifecycle
           FROM concepts c
           WHERE c.id IN (
               SELECT DISTINCT ce.concept_id FROM concept_events ce
               JOIN events e ON ce.event_id = e.id
               WHERE e.project_id = ?
           )
           ORDER BY c.importance DESC""",
        (project_id,),
    ).fetchall()

    mention_counts: Counter[str] = Counter()
    for eid_concepts in event_concepts.values():
        for c in eid_concepts:
            mention_counts[c] += 1

    one_hit_wonders = sorted(c for c, n in mention_counts.items() if n == 1)

    # Concept links
    project_concept_names = {c["name"] for c in concepts}
    all_links = db.conn.execute(
        """SELECT c1.name as a, c2.name as b, cl.relationship, cl.evidence
           FROM concept_links cl
           JOIN concepts c1 ON cl.concept_a_id = c1.id
           JOIN concepts c2 ON cl.concept_b_id = c2.id""",
    ).fetchall()
    relevant_links = [
        {"a": l["a"], "b": l["b"], "relationship": l["relationship"], "evidence": l["evidence"]}
        for l in all_links
        if l["a"] in project_concept_names or l["b"] in project_concept_names
    ]

    # Group events by day
    days_data: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        day = event["timestamp"][:10]
        days_data[day].append({
            "type": event["event_type"] or "unknown",
            "title": event["title"] or "(untitled)",
            "summary": event["llm_summary"],
            "concepts": event_concepts.get(event["id"], []),
        })

    days = []
    seen_concepts: set[str] = set()
    for day in sorted(days_data):
        day_events = days_data[day]
        day_concepts = set()
        for e in day_events:
            day_concepts.update(e["concepts"])
        new_concepts = sorted(day_concepts - seen_concepts)
        seen_concepts.update(day_concepts)
        days.append({
            "date": day,
            "event_count": len(day_events),
            "events": day_events,
            "decisions": decision_by_day.get(day, []),
            "new_concepts": new_concepts,
        })

    first_date = events[0]["timestamp"][:10] if events else "?"
    last_date = events[-1]["timestamp"][:10] if events else "?"
    total_days = 0
    if events:
        t0 = datetime.fromisoformat(events[0]["timestamp"][:10])
        t1 = datetime.fromisoformat(events[-1]["timestamp"][:10])
        total_days = (t1 - t0).days + 1

    sessions = db.get_sessions(project_id=project_id, limit=10000)

    return {
        "project_name": project.name,
        "first_date": first_date,
        "last_date": last_date,
        "total_days": total_days,
        "total_events": len(events),
        "total_concepts": len(concepts),
        "total_decisions": len(decisions),
        "total_sessions": len(sessions),
        "top_concepts": [
            {
                "name": c["name"],
                "level": c["level"] or "technique",
                "importance": round(c["importance"] or 0, 1),
                "lifecycle": c["lifecycle"] or "unknown",
                "first_seen": c["first_seen"] or "?",
                "last_seen": c["last_seen"] or "?",
            }
            for c in concepts[:25]
        ],
        "days": days,
        "one_hit_wonders": one_hit_wonders,
        "concept_links": relevant_links,
        "concept_levels": {c: l for c, l in concept_levels.items() if mention_counts[c] >= 2},
    }


def generate_narrative(
    db: TrajectoryDB,
    project_name: str,
    model: str = "gemini/gemini-2.5-flash",
) -> Path:
    """Generate a scrollytelling narrative HTML page for a project."""
    project = db.conn.execute(
        "SELECT id, name FROM projects WHERE name = ?", (project_name,),
    ).fetchone()
    if not project:
        raise ValueError(f"Project '{project_name}' not found in trajectory DB")

    logger.info("Gathering narrative data for %s...", project_name)
    data = _gather_narrative_data(db, project["id"])

    logger.info(
        "Data: %d events, %d concepts, %d decisions across %d days",
        data["total_events"], data["total_concepts"],
        data["total_decisions"], data["total_days"],
    )

    messages = render_prompt(PROMPTS_DIR / "project_narrative.yaml", **data)

    logger.info("Calling LLM for narrative synthesis (%s)...", model)
    result = call_llm(
        model,
        messages,
        task="project_narrative",
        trace_id=f"trajectory.narrative.{project_name}",
        max_budget=0,
    )

    raw = result.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    narrative = json.loads(raw)
    logger.info(
        "Narrative: %d phases, cost: $%.4f",
        len(narrative.get("phases", [])),
        result.cost or 0,
    )

    html = _render_scrollytelling(project_name, data, narrative)

    output_dir = Path(__file__).parent.parent.parent / "data" / "narratives"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{project_name}_narrative.html"
    output_path.write_text(html, encoding="utf-8")

    logger.info("Output: %s", output_path)
    return output_path


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _inline_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _sanitize_mermaid(code: str) -> str:
    """Fix common LLM mistakes in Mermaid flowchart syntax."""
    lines = code.split("\n")
    sanitized = []
    for line in lines:
        # Fix: `A --> (some_node)` or `A --> ([some_node])` — strip parens from edge targets
        # These cause parse errors because Mermaid sees `(` as unexpected after `-->`
        line = re.sub(
            r"(-->|-.->)(\|[^|]*\|)?\s*\(\[?([a-zA-Z0-9_]+)\]?\)",
            r"\1\2 \3",
            line,
        )
        # Fix: node IDs with hyphens — replace with underscores
        # (only in node positions, not inside labels/strings)
        # Fix: bare `(node_name)` at start of line (shape def without ID prefix)
        line = re.sub(r"^\s+\(([a-zA-Z0-9_]+)\)", r"  \1", line)
        sanitized.append(line)
    return "\n".join(sanitized)


def _render_scrollytelling(project_name: str, data: dict, narrative: dict) -> str:
    """Render the full scrollytelling HTML page."""

    # Build phase sections HTML
    phase_sections = []

    for i, phase in enumerate(narrative.get("phases", [])):
        color = phase.get("color", "#58a6ff")
        paragraphs_html = []

        for j, para in enumerate(phase.get("paragraphs", [])):
            text = _inline_md(_esc(para.get("text", "")))
            # Highlight concept names in text
            for c in para.get("concepts_active", []):
                level = data["concept_levels"].get(c, "technique")
                css_class = f"concept-chip concept-{level}"
                display = c.replace("_", " ")
                text = re.sub(
                    re.escape(c),
                    f'<span class="{css_class}" data-concept="{_esc(c)}">{display}</span>',
                    text,
                    flags=re.IGNORECASE,
                )
            paragraphs_html.append(
                f'<p class="reveal" data-phase="{i}" data-para="{j}">{text}</p>'
            )

        # Key decision pull quote
        decision = phase.get("key_decision", {})
        decision_html = ""
        if decision.get("title"):
            decision_html = f"""
            <div class="pull-quote reveal" data-phase="{i}" style="--phase-color: {color}">
                <div class="pull-quote-mark">&ldquo;</div>
                <div class="pull-quote-text">{_esc(decision['title'])}</div>
                <div class="pull-quote-tension">{_esc(decision.get('tension', ''))}</div>
            </div>"""

        new_concepts = phase.get("new_concept_count", 0)
        event_count = phase.get("event_count", 0)

        phase_sections.append(f"""
        <section class="phase" data-phase="{i}" data-color="{color}">
            <div class="phase-header reveal">
                <div class="phase-number" style="color: {color}">Phase {i + 1}</div>
                <h2 class="phase-title" style="--phase-color: {color}">{_esc(phase.get('name', ''))}</h2>
                <div class="phase-date">{_esc(phase.get('date_range', ''))}</div>
                <div class="phase-stats">
                    <span class="phase-stat"><strong>{event_count}</strong> events</span>
                    <span class="phase-stat"><strong>{new_concepts}</strong> new concepts</span>
                </div>
            </div>
            {''.join(paragraphs_html)}
            {decision_html}
        </section>""")

    # One-hit wonders
    ohw = data.get("one_hit_wonders", [])
    ohw_tags = "".join(
        f'<span class="ohw-tag">{_esc(c.replace("_", " "))}</span>' for c in ohw[:30]
    )

    # Mermaid diagram from LLM — sanitize common mistakes
    mermaid_code = narrative.get("diagram", "graph TD\n  A[No diagram generated]")
    mermaid_code = _sanitize_mermaid(mermaid_code)
    # Escape for embedding in HTML
    mermaid_json = json.dumps(mermaid_code)

    # Phase names for highlighting
    phase_names = [p.get("name", f"Phase {i+1}") for i, p in enumerate(narrative.get("phases", []))]

    stats = data

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(project_name)} — Narrative</title>
<style>
:root {{
    --bg: #0d1117;
    --bg-card: #161b22;
    --border: #21262d;
    --text: #c9d1d9;
    --text-bright: #e6edf3;
    --text-dim: #8b949e;
    --text-dimmer: #484f58;
    --blue: #58a6ff;
    --purple: #d2a8ff;
    --green: #7ee787;
    --orange: #f78166;
    --yellow: #f0883e;
    --red: #ff7b72;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Georgia', 'Times New Roman', serif;
    min-height: 100vh;
    overflow-x: hidden;
}}

/* --- Hero --- */
.hero {{
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    position: relative;
    padding: 40px 24px;
}}
.hero-title {{
    font-size: 64px;
    font-weight: 200;
    color: var(--text-bright);
    letter-spacing: -2px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    margin-bottom: 12px;
    opacity: 0;
    animation: fadeUp 1s 0.2s forwards;
}}
.hero-arc {{
    font-size: 22px;
    color: var(--purple);
    font-style: italic;
    max-width: 600px;
    line-height: 1.6;
    margin-bottom: 32px;
    opacity: 0;
    animation: fadeUp 1s 0.6s forwards;
}}
.hero-stats {{
    display: flex;
    gap: 48px;
    opacity: 0;
    animation: fadeUp 1s 1.0s forwards;
}}
.hero-stat {{
    text-align: center;
}}
.hero-stat .num {{
    font-size: 40px;
    font-weight: 800;
    color: var(--text-bright);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.hero-stat .label {{
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 2px;
}}
.hero-scroll {{
    position: absolute;
    bottom: 40px;
    color: var(--text-dimmer);
    font-size: 13px;
    opacity: 0;
    animation: fadeUp 1s 1.4s forwards;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.hero-scroll::after {{
    content: '';
    display: block;
    width: 1px;
    height: 40px;
    background: linear-gradient(to bottom, var(--text-dimmer), transparent);
    margin: 12px auto 0;
}}
.hero-daterange {{
    font-size: 14px;
    color: var(--text-dimmer);
    margin-bottom: 24px;
    opacity: 0;
    animation: fadeUp 1s 0.4s forwards;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}

/* --- Layout --- */
.scrolly-container {{
    display: flex;
    position: relative;
}}
.narrative-track {{
    width: 55%;
    padding: 80px 60px 200px 80px;
}}
.graph-track {{
    width: 45%;
    position: sticky;
    top: 0;
    height: 100vh;
    border-left: 1px solid var(--border);
}}
.diagram-container {{
    width: 100%;
    height: 100%;
    overflow: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}}
#mermaid-diagram {{
    width: 100%;
}}
#mermaid-diagram svg {{
    max-width: 100%;
    height: auto;
}}
/* Mermaid progressive reveal */
#mermaid-diagram .cluster,
#mermaid-diagram .edgePath,
#mermaid-diagram .edgeLabel {{
    transition: opacity 0.6s ease, filter 0.6s ease;
}}

/* --- Phases --- */
.phase {{
    margin-bottom: 120px;
}}
.phase-header {{
    margin-bottom: 32px;
}}
.phase-number {{
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    margin-bottom: 8px;
}}
.phase-title {{
    font-size: 36px;
    font-weight: 300;
    color: var(--text-bright);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    margin-bottom: 4px;
    border-left: 4px solid var(--phase-color);
    padding-left: 16px;
}}
.phase-date {{
    font-size: 14px;
    color: var(--text-dim);
    padding-left: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.phase-stats {{
    display: flex;
    gap: 20px;
    padding-left: 20px;
    margin-top: 8px;
    font-size: 13px;
    color: var(--text-dimmer);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.phase-stat strong {{
    color: var(--text-dim);
}}

/* --- Paragraphs --- */
.narrative-track p {{
    font-size: 18px;
    line-height: 1.9;
    margin-bottom: 24px;
    color: var(--text);
}}
.narrative-track strong {{ color: var(--text-bright); }}
.narrative-track em {{ color: var(--text-dim); font-style: italic; }}

/* --- Concept chips --- */
.concept-chip {{
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 14px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    white-space: nowrap;
    transition: all 0.3s;
}}
.concept-theme {{
    background: rgba(88, 166, 255, 0.12);
    color: var(--blue);
    border-bottom: 1px solid rgba(88, 166, 255, 0.3);
}}
.concept-design_bet {{
    background: rgba(210, 168, 255, 0.12);
    color: var(--purple);
    border-bottom: 1px solid rgba(210, 168, 255, 0.3);
}}
.concept-technique {{
    background: rgba(126, 231, 135, 0.12);
    color: var(--green);
    border-bottom: 1px solid rgba(126, 231, 135, 0.3);
}}
.concept-chip.glow {{
    box-shadow: 0 0 12px currentColor;
    transform: scale(1.05);
}}

/* --- Pull quotes --- */
.pull-quote {{
    margin: 40px 0;
    padding: 24px 28px;
    border-left: 3px solid var(--phase-color);
    background: var(--bg-card);
    border-radius: 0 8px 8px 0;
    position: relative;
}}
.pull-quote-mark {{
    font-size: 48px;
    color: var(--phase-color);
    opacity: 0.3;
    position: absolute;
    top: 8px;
    left: 12px;
    line-height: 1;
    font-family: Georgia, serif;
}}
.pull-quote-text {{
    font-size: 20px;
    font-weight: 400;
    color: var(--text-bright);
    line-height: 1.5;
    padding-left: 24px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.pull-quote-tension {{
    font-size: 14px;
    color: var(--text-dim);
    margin-top: 12px;
    padding-left: 24px;
    font-style: italic;
}}

/* --- Reveal animation --- */
.reveal {{
    opacity: 0;
    transform: translateY(24px);
    transition: opacity 0.6s ease, transform 0.6s ease;
}}
.reveal.visible {{
    opacity: 1;
    transform: translateY(0);
}}

/* --- Arc summary --- */
.arc-section {{
    padding: 120px 80px;
    text-align: center;
    border-top: 1px solid var(--border);
}}
.arc-epitaph {{
    font-size: 28px;
    font-weight: 300;
    color: var(--purple);
    max-width: 640px;
    margin: 0 auto 40px;
    line-height: 1.5;
    font-style: italic;
}}
.arc-summary {{
    font-size: 17px;
    color: var(--text-dim);
    max-width: 560px;
    margin: 0 auto 48px;
    line-height: 1.8;
}}

/* --- One-hit wonders --- */
.ohw-section {{
    padding: 60px 80px;
    text-align: center;
    border-top: 1px solid var(--border);
}}
.ohw-title {{
    font-size: 14px;
    color: var(--text-dimmer);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.ohw-subtitle {{
    font-size: 14px;
    color: var(--text-dimmer);
    margin-bottom: 20px;
    font-style: italic;
}}
.ohw-tags {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    max-width: 700px;
    margin: 0 auto;
}}
.ohw-tag {{
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-dimmer);
    border: 1px solid rgba(255, 255, 255, 0.06);
    font-family: 'SF Mono', 'Fira Code', monospace;
    animation: ohwFade 3s ease-in-out infinite alternate;
}}
.ohw-tag:nth-child(odd) {{ animation-delay: 0.5s; }}
.ohw-tag:nth-child(3n) {{ animation-delay: 1s; }}

/* --- Footer --- */
.footer {{
    text-align: center;
    padding: 40px 24px;
    color: var(--text-dimmer);
    font-size: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}

/* --- Animations --- */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes ohwFade {{
    from {{ opacity: 0.3; }}
    to {{ opacity: 0.7; }}
}}

/* --- Responsive --- */
@media (max-width: 900px) {{
    .scrolly-container {{ flex-direction: column; }}
    .narrative-track {{ width: 100%; padding: 40px 24px 120px; }}
    .graph-track {{ width: 100%; height: 50vh; position: relative; border-left: none; border-top: 1px solid var(--border); }}
    .hero-title {{ font-size: 36px; }}
    .hero-stats {{ gap: 24px; }}
    .phase-title {{ font-size: 28px; }}
    .arc-section, .ohw-section {{ padding: 60px 24px; }}
}}
</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
    <div class="hero-title">{_esc(project_name)}</div>
    <div class="hero-daterange">{_esc(stats['first_date'])} &mdash; {_esc(stats['last_date'])} &middot; {stats['total_days']} days</div>
    <div class="hero-arc">{_esc(narrative.get('epitaph', ''))}</div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="num" data-count="{stats['total_events']}">0</div>
            <div class="label">events</div>
        </div>
        <div class="hero-stat">
            <div class="num" data-count="{stats['total_concepts']}">0</div>
            <div class="label">concepts</div>
        </div>
        <div class="hero-stat">
            <div class="num" data-count="{stats['total_decisions']}">0</div>
            <div class="label">decisions</div>
        </div>
    </div>
    <div class="hero-scroll">scroll to explore</div>
</div>

<!-- Scrollytelling -->
<div class="scrolly-container">
    <div class="narrative-track">
        {''.join(phase_sections)}
    </div>
    <div class="graph-track">
        <div class="diagram-container">
            <div id="mermaid-diagram"></div>
        </div>
    </div>
</div>

<!-- Arc -->
<div class="arc-section">
    <div class="arc-epitaph reveal">{_esc(narrative.get('epitaph', ''))}</div>
    <div class="arc-summary reveal">{_inline_md(_esc(narrative.get('arc_summary', '')))}</div>
</div>

<!-- One-hit wonders -->
<div class="ohw-section">
    <div class="ohw-title">Ideas That Flickered Once</div>
    <div class="ohw-subtitle">{len(ohw)} concepts appeared exactly once and were never revisited</div>
    <div class="ohw-tags">{ohw_tags}</div>
</div>

<div class="footer">Generated by trajectory &middot; {datetime.now().strftime('%b %d, %Y')}</div>

<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

const MERMAID_CODE = {mermaid_json};
const PHASE_NAMES = {json.dumps(phase_names)};

// --- Mermaid init ---
mermaid.initialize({{
    startOnLoad: false,
    theme: 'dark',
    themeVariables: {{
        primaryColor: '#1f2937',
        primaryTextColor: '#e6edf3',
        primaryBorderColor: '#58a6ff',
        lineColor: '#8b949e',
        secondaryColor: '#161b22',
        tertiaryColor: '#0d1117',
        background: '#0d1117',
        mainBkg: '#161b22',
        nodeBorder: '#58a6ff',
        clusterBkg: 'rgba(88, 166, 255, 0.06)',
        clusterBorder: 'rgba(88, 166, 255, 0.2)',
        titleColor: '#e6edf3',
        edgeLabelBackground: '#161b22',
        fontSize: '13px',
    }},
    flowchart: {{
        curve: 'basis',
        padding: 16,
        htmlLabels: true,
        useMaxWidth: true,
    }},
}});

async function renderDiagram() {{
    const container = document.getElementById('mermaid-diagram');
    try {{
        const {{ svg }} = await mermaid.render('mermaid-svg', MERMAID_CODE);
        container.innerHTML = svg;

        // Tag subgraphs with phase indices for highlighting
        const clusters = container.querySelectorAll('.cluster');
        clusters.forEach((cluster, idx) => {{
            cluster.setAttribute('data-phase', idx);
        }});
    }} catch (e) {{
        console.error('Mermaid render error:', e);
        container.innerHTML = '<pre style="color:#ff7b72;padding:20px;font-size:12px;">' +
            'Diagram rendering failed.\\n' + e.message + '</pre>';
    }}
}}
renderDiagram();

// --- Counter animation ---
document.querySelectorAll('.hero-stat .num').forEach(el => {{
    const target = parseInt(el.dataset.count);
    const duration = 1500;
    const start = performance.now();
    function tick(now) {{
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(tick);
    }}
    setTimeout(() => requestAnimationFrame(tick), 1200);
}});

// --- Reveal on scroll ---
const observer = new IntersectionObserver((entries) => {{
    entries.forEach(e => {{
        if (e.isIntersecting) {{
            e.target.classList.add('visible');
            e.target.querySelectorAll('.concept-chip').forEach((chip, i) => {{
                setTimeout(() => {{
                    chip.classList.add('glow');
                    setTimeout(() => chip.classList.remove('glow'), 1200);
                }}, i * 150);
            }});
        }}
    }});
}}, {{ threshold: 0.15, rootMargin: '0px 0px -60px 0px' }});

document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// --- Progressive diagram reveal ---
let maxRevealedPhase = -1;

// Build node→cluster mapping after render
function buildClusterMap() {{
    const diagram = document.getElementById('mermaid-diagram');
    const clusters = diagram.querySelectorAll('.cluster');
    const nodeToPhase = {{}};

    clusters.forEach((cluster, idx) => {{
        // Hide all clusters initially
        cluster.style.opacity = '0';
        cluster.setAttribute('data-phase', idx);

        // Map node IDs within this cluster to its phase index
        cluster.querySelectorAll('.node').forEach(node => {{
            if (node.id) nodeToPhase[node.id] = idx;
        }});
    }});

    // Hide all edges and edge labels initially
    diagram.querySelectorAll('.edgePath, .edgeLabel').forEach(el => {{
        el.style.opacity = '0';
    }});

    return {{ clusters, nodeToPhase }};
}}

// Determine which phase an edge belongs to (max phase of its endpoints)
function getEdgePhase(edgeEl, nodeToPhase) {{
    const id = edgeEl.id || '';
    let maxPhase = -1;
    for (const [nodeId, phase] of Object.entries(nodeToPhase)) {{
        if (id.includes(nodeId)) {{
            maxPhase = Math.max(maxPhase, phase);
        }}
    }}
    return maxPhase;
}}

let clusterMap = null;

// Wait for mermaid to finish rendering, then build map
const waitForRender = setInterval(() => {{
    const svg = document.querySelector('#mermaid-diagram svg');
    if (svg) {{
        clearInterval(waitForRender);
        clusterMap = buildClusterMap();
    }}
}}, 200);

function revealUpToPhase(phaseIdx) {{
    if (!clusterMap) return;
    const {{ clusters, nodeToPhase }} = clusterMap;
    const diagram = document.getElementById('mermaid-diagram');

    clusters.forEach((cluster, idx) => {{
        if (idx <= phaseIdx) {{
            // Current phase: full opacity. Previous phases: dimmed.
            cluster.style.opacity = (idx === phaseIdx) ? '1' : '0.35';
            cluster.style.filter = (idx === phaseIdx) ? 'none' : 'saturate(0.4)';
        }} else {{
            // Future phases: hidden
            cluster.style.opacity = '0';
            cluster.style.filter = 'none';
        }}
    }});

    // Reveal edges whose endpoints are both visible (phase <= phaseIdx)
    diagram.querySelectorAll('.edgePath').forEach(edge => {{
        const edgePhase = getEdgePhase(edge, nodeToPhase);
        if (edgePhase >= 0 && edgePhase <= phaseIdx) {{
            edge.style.opacity = (edgePhase === phaseIdx) ? '1' : '0.35';
        }} else {{
            edge.style.opacity = '0';
        }}
    }});
    diagram.querySelectorAll('.edgeLabel').forEach(label => {{
        // Edge labels follow same pattern — match by sibling index
        const id = label.id || '';
        let labelPhase = -1;
        for (const [nodeId, phase] of Object.entries(nodeToPhase)) {{
            if (id.includes(nodeId)) {{
                labelPhase = Math.max(labelPhase, phase);
            }}
        }}
        if (labelPhase >= 0 && labelPhase <= phaseIdx) {{
            label.style.opacity = (labelPhase === phaseIdx) ? '1' : '0.3';
        }} else {{
            label.style.opacity = '0';
        }}
    }});

    // Scroll diagram to keep active cluster in view
    if (phaseIdx >= 0 && phaseIdx < clusters.length) {{
        const rect = clusters[phaseIdx].querySelector('rect');
        if (rect) {{
            const y = parseFloat(rect.getAttribute('y') || 0);
            const container = diagram.parentElement;
            container.scrollTo({{
                top: Math.max(0, y - container.clientHeight / 3),
                behavior: 'smooth'
            }});
        }}
    }}
}}

const phaseObserver = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
        if (entry.isIntersecting) {{
            const phase = parseInt(entry.target.dataset.phase);
            if (phase > maxRevealedPhase) {{
                maxRevealedPhase = phase;
            }}
            revealUpToPhase(phase);
        }}
    }});
}}, {{ threshold: 0.3 }});

document.querySelectorAll('.phase').forEach(el => phaseObserver.observe(el));
</script>
</body>
</html>"""
