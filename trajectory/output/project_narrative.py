"""Project Narrative — LLM-synthesized story of a project's intellectual journey.

Gathers all chronological data (events, concepts, decisions, sessions) for a project,
feeds it to an LLM via a Jinja2 prompt template, and renders the result as a
self-contained HTML page.
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

    # All events, ordered
    events = db.conn.execute(
        """SELECT e.id, e.timestamp, e.title, e.event_type, e.llm_summary,
                  e.llm_intent, e.significance
           FROM events e WHERE e.project_id = ? ORDER BY e.timestamp""",
        (project_id,),
    ).fetchall()

    # Concept ↔ event links
    concept_events = db.conn.execute(
        """SELECT ce.event_id, c.name, c.level
           FROM concept_events ce
           JOIN concepts c ON ce.concept_id = c.id
           JOIN events e ON ce.event_id = e.id
           WHERE e.project_id = ?""",
        (project_id,),
    ).fetchall()

    event_concepts: dict[int, list[str]] = defaultdict(list)
    for row in concept_events:
        event_concepts[row["event_id"]].append(row["name"])

    # Decisions
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
        # alternatives might be JSON list or plain text
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

    # Concepts with importance
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

    # Concept mention counts
    mention_counts: Counter[str] = Counter()
    for eid_concepts in event_concepts.values():
        for c in eid_concepts:
            mention_counts[c] += 1

    one_hit_wonders = sorted(c for c, n in mention_counts.items() if n == 1)

    # Concept links involving this project's concepts
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

    # Build day objects
    days = []
    seen_concepts: set[str] = set()
    for day in sorted(days_data):
        day_events = days_data[day]
        # New concepts introduced this day
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
    }


def generate_narrative(
    db: TrajectoryDB,
    project_name: str,
    model: str = "gemini/gemini-2.5-flash",
) -> Path:
    """Generate a narrative HTML page for a project."""
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

    # Render prompt
    messages = render_prompt(PROMPTS_DIR / "project_narrative.yaml", **data)

    logger.info("Calling LLM for narrative synthesis (%s)...", model)
    result = call_llm(
        model,
        messages,
        task="project_narrative",
        trace_id=f"trajectory.narrative.{project_name}",
        max_budget=0,
    )

    narrative_text = result.content
    logger.info("Narrative: %d chars, cost: $%.4f", len(narrative_text), result.cost or 0)

    # Render HTML
    html = _render_narrative_html(project_name, data, narrative_text)

    output_dir = Path(__file__).parent.parent.parent / "data" / "narratives"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{project_name}_narrative.html"
    output_path.write_text(html, encoding="utf-8")

    logger.info("Output: %s", output_path)
    return output_path


def _render_narrative_html(project_name: str, data: dict, narrative: str) -> str:
    """Render narrative as a self-contained HTML page."""
    # Convert markdown-ish narrative to HTML paragraphs
    # Handle ## headers, **bold**, *italic*, and paragraph breaks
    html_body = _markdown_to_html(narrative)

    top_themes = [c for c in data["top_concepts"] if c["level"] == "theme"][:5]
    top_bets = [c for c in data["top_concepts"] if c["level"] == "design_bet"][:5]

    theme_tags = "".join(
        f'<span class="tag tag-theme">{_esc(t["name"])}</span>' for t in top_themes
    )
    bet_tags = "".join(
        f'<span class="tag tag-bet">{_esc(b["name"])}</span>' for b in top_bets
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(project_name)} — Narrative</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'Georgia', 'Times New Roman', serif;
    min-height: 100vh;
    line-height: 1.8;
}}

/* --- Header --- */
.header {{
    text-align: center;
    padding: 80px 24px 48px;
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #21262d;
}}
.header h1 {{
    font-size: 48px;
    font-weight: 300;
    letter-spacing: -1px;
    color: #e6edf3;
    margin-bottom: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.header .subtitle {{
    font-size: 16px;
    color: #8b949e;
    margin-bottom: 24px;
    font-style: italic;
}}
.header .meta {{
    display: flex;
    justify-content: center;
    gap: 32px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}}
.header .meta-item {{
    text-align: center;
}}
.header .meta-item .num {{
    font-size: 28px;
    font-weight: 700;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.header .meta-item .label {{
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}}
.tags {{
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 16px;
}}
.tag {{
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 16px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.tag-theme {{
    background: rgba(88, 166, 255, 0.15);
    color: #58a6ff;
    border: 1px solid rgba(88, 166, 255, 0.3);
}}
.tag-bet {{
    background: rgba(210, 168, 255, 0.15);
    color: #d2a8ff;
    border: 1px solid rgba(210, 168, 255, 0.3);
}}

/* --- Narrative --- */
.narrative {{
    max-width: 680px;
    margin: 0 auto;
    padding: 64px 24px 80px;
}}
.narrative h2 {{
    font-size: 28px;
    font-weight: 400;
    color: #e6edf3;
    margin-top: 48px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.narrative h3 {{
    font-size: 20px;
    font-weight: 600;
    color: #d2a8ff;
    margin-top: 32px;
    margin-bottom: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.narrative p {{
    margin-bottom: 20px;
    font-size: 17px;
    color: #c9d1d9;
}}
.narrative strong {{
    color: #e6edf3;
    font-weight: 600;
}}
.narrative em {{
    color: #8b949e;
    font-style: italic;
}}
.narrative blockquote {{
    border-left: 3px solid #d2a8ff;
    padding-left: 20px;
    margin: 24px 0;
    color: #8b949e;
    font-style: italic;
}}
.narrative ul, .narrative ol {{
    margin: 16px 0;
    padding-left: 24px;
}}
.narrative li {{
    margin-bottom: 8px;
    font-size: 17px;
}}

/* --- Footer --- */
.footer {{
    text-align: center;
    padding: 40px 24px;
    color: #484f58;
    font-size: 12px;
    border-top: 1px solid #21262d;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}

@media (max-width: 480px) {{
    .header h1 {{ font-size: 32px; }}
    .narrative {{ padding: 32px 16px 48px; }}
    .narrative p {{ font-size: 16px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>{_esc(project_name)}</h1>
    <div class="subtitle">{_esc(data['first_date'])} &mdash; {_esc(data['last_date'])} &middot; {data['total_days']} days</div>
    <div class="meta">
        <div class="meta-item">
            <div class="num">{data['total_events']}</div>
            <div class="label">events</div>
        </div>
        <div class="meta-item">
            <div class="num">{data['total_concepts']}</div>
            <div class="label">concepts</div>
        </div>
        <div class="meta-item">
            <div class="num">{data['total_decisions']}</div>
            <div class="label">decisions</div>
        </div>
        <div class="meta-item">
            <div class="num">{data['total_sessions']}</div>
            <div class="label">sessions</div>
        </div>
    </div>
    <div class="tags">
        {theme_tags}
        {bet_tags}
    </div>
</div>

<div class="narrative">
{html_body}
</div>

<div class="footer">
    Generated by trajectory &middot; {datetime.now().strftime('%b %d, %Y')}
</div>

</body>
</html>"""


def _markdown_to_html(text: str) -> str:
    """Convert simple markdown to HTML."""
    import re

    lines = text.split("\n")
    html_parts: list[str] = []
    in_list = False
    list_type = ""

    for line in lines:
        stripped = line.strip()

        # Empty line — close any open list, add paragraph break
        if not stripped:
            if in_list:
                html_parts.append(f"</{list_type}>")
                in_list = False
            html_parts.append("")
            continue

        # Headers
        if stripped.startswith("### "):
            if in_list:
                html_parts.append(f"</{list_type}>")
                in_list = False
            html_parts.append(f"<h3>{_inline_md(stripped[4:])}</h3>")
            continue
        if stripped.startswith("## "):
            if in_list:
                html_parts.append(f"</{list_type}>")
                in_list = False
            html_parts.append(f"<h2>{_inline_md(stripped[3:])}</h2>")
            continue

        # Blockquote
        if stripped.startswith("> "):
            if in_list:
                html_parts.append(f"</{list_type}>")
                in_list = False
            html_parts.append(f"<blockquote><p>{_inline_md(stripped[2:])}</p></blockquote>")
            continue

        # Unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list or list_type != "ul":
                if in_list:
                    html_parts.append(f"</{list_type}>")
                html_parts.append("<ul>")
                in_list = True
                list_type = "ul"
            html_parts.append(f"<li>{_inline_md(stripped[2:])}</li>")
            continue

        # Ordered list
        m = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if m:
            if not in_list or list_type != "ol":
                if in_list:
                    html_parts.append(f"</{list_type}>")
                html_parts.append("<ol>")
                in_list = True
                list_type = "ol"
            html_parts.append(f"<li>{_inline_md(m.group(2))}</li>")
            continue

        # Regular paragraph
        if in_list:
            html_parts.append(f"</{list_type}>")
            in_list = False
        html_parts.append(f"<p>{_inline_md(stripped)}</p>")

    if in_list:
        html_parts.append(f"</{list_type}>")

    return "\n".join(html_parts)


def _inline_md(text: str) -> str:
    """Convert inline markdown (bold, italic) to HTML."""
    # Bold: **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic: *text*
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code: `text`
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _esc(s: str) -> str:
    """HTML-escape a string."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
