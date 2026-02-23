"""Project Narrative — scrollytelling visualization of a project's intellectual journey.

Two-pass LLM generation:
1. Narrative text: phases, key decisions, concept mentions
2. Journey-map graph: story beats, decisions, dead ends, breakthroughs (JSON for D3+dagre)

Renders a scrollytelling HTML page:
- Left: narrative text with typewriter reveal and concept chips
- Right: D3+dagre journey map with node-by-node animation synced to scroll
- Cinema mode: autoplay with speed control, ambient particles, and Web Audio synth
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


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw


def _parse_versioned_doc(doc_path: str | Path) -> list[dict]:
    """Parse a versioned markdown document into a list of version dicts.

    Handles multiple header formats:
    - '# V<number>' (CEO Cloning, Elite Cyborg)
    - '# <number> <text>' or '# <number><text>' (Rewiring Brain)
    - '# <Named Version>' (e.g. 'Linkedin Version 2')

    Skips fragment sections (notes, aspects, thesis fragments) that aren't
    full document versions (< 200 words).
    """
    doc_path = Path(doc_path)
    text = doc_path.read_text(encoding="utf-8")

    # Try V-numbered headers first
    v_pattern = re.compile(r"^# (V\d+.*?)$", re.MULTILINE)
    v_splits = list(v_pattern.finditer(text))

    # Try number-prefixed headers (e.g. "# 10 short - ..." or "# 5Rewirin...")
    num_pattern = re.compile(r"^# (\d+\s*.*?)$", re.MULTILINE)
    num_splits = list(num_pattern.finditer(text))

    # Also catch named versions like "# Linkedin Version 2"
    named_pattern = re.compile(r"^# ((?:Linkedin|Draft|Final|Version)\s+.*?)$", re.MULTILINE | re.IGNORECASE)
    named_splits = list(named_pattern.finditer(text))

    # Use whichever pattern found the most matches, combining named with best
    if len(v_splits) >= len(num_splits):
        splits = v_splits
    else:
        splits = num_splits
    # Add named splits that don't overlap with primary
    primary_positions = {m.start() for m in splits}
    for m in named_splits:
        if m.start() not in primary_positions:
            splits.append(m)
    splits.sort(key=lambda m: m.start())

    if not splits:
        raise ValueError(f"No version headers found in {doc_path.name}")

    versions = []
    for i, match in enumerate(splits):
        header = match.group(1).strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        body = text[start:end].strip()
        word_count = len(body.split())

        # Skip fragments (< 200 words) — these are notes, not full versions
        if word_count < 200:
            continue

        # Extract version number
        vnum_match = re.match(r"V(\d+)", header)
        if not vnum_match:
            vnum_match = re.match(r"(\d+)", header)
        if vnum_match:
            vnum = int(vnum_match.group(1))
        else:
            # Named version — assign high number (latest)
            vnum = 100 + i

        # Extract editorial note
        if header.startswith(f"V{vnum}"):
            note = header[len(f"V{vnum}"):].strip().lstrip("-").strip()
        elif header.startswith(str(vnum)):
            note = header[len(str(vnum)):].strip().lstrip("-").strip()
        else:
            note = header

        # Section headers
        sections = re.findall(r"^#{2,4}\s+\*?\*?(.+?)\*?\*?\s*$", body, re.MULTILINE)

        versions.append({
            "number": vnum,
            "header": header,
            "note": note,
            "body": body,
            "word_count": word_count,
            "sections": sections[:8],
        })

    # Sort by version number (oldest first)
    versions.sort(key=lambda v: v["number"])

    # Deduplicate: when same version number appears twice, keep the larger one
    deduped: list[dict] = []
    for v in versions:
        if deduped and deduped[-1]["number"] == v["number"]:
            if v["word_count"] > deduped[-1]["word_count"]:
                deduped[-1] = v
        else:
            deduped.append(v)
    versions = deduped

    # Compute diffs between consecutive versions
    for i, v in enumerate(versions):
        if i == 0:
            v["added_sections"] = v["sections"][:]
            v["removed_sections"] = []
        else:
            prev = versions[i - 1]
            prev_set = set(s.lower().strip("* ") for s in prev["sections"])
            curr_set = set(s.lower().strip("* ") for s in v["sections"])
            v["added_sections"] = [s for s in v["sections"] if s.lower().strip("* ") not in prev_set]
            v["removed_sections"] = [s for s in prev["sections"] if s.lower().strip("* ") not in curr_set]
            v["word_delta"] = v["word_count"] - prev["word_count"]

    return versions


def _extract_doc_concepts(
    versions: list[dict], doc_name: str, model: str,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Run LLM concept extraction across document versions.

    Returns (concepts, version_concepts) where:
    - concepts: list of concept dicts with name, status, description, etc.
    - version_concepts: mapping of version number str -> list of concept names
    """
    concept_prompt = [
        {"role": "system", "content": """You are analyzing how a document's IDEAS evolve across versions.

For each version, identify 2-5 key concepts/ideas/framings present in that version.
Track how concepts appear, evolve, merge, split, or get abandoned across versions.

Return a JSON object:
{
  "concepts": [
    {"name": "concept_name", "first_version": 1, "last_version": 31, "status": "survived|evolved|abandoned|merged",
     "evolved_into": "other_concept_name or null", "description": "What this concept/framing is about"}
  ],
  "version_concepts": {
    "1": ["concept_a", "concept_b"],
    "5": ["concept_a", "concept_c"]
  }
}

Guidelines:
- Concepts are IDEAS and FRAMINGS, not section headers
- Track when a concept gets REFRAMED (old concept abandoned, new one born from it)
- 10-20 total concepts across all versions. Quality over quantity.
- Use snake_case names that are descriptive"""},
        {"role": "user", "content": _build_version_content_for_llm(versions)},
    ]
    result = call_llm(
        model, concept_prompt,
        task="doc_concept_extraction",
        trace_id=f"trajectory.doc_concepts.{doc_name}",
        max_budget=0,
    )
    data = json.loads(_strip_json_fences(result.content))
    concepts = data.get("concepts", [])
    version_concepts = data.get("version_concepts", {})
    logger.info("Extracted %d concepts from %s across %d versions", len(concepts), doc_name, len(version_concepts))
    return concepts, version_concepts


def _build_version_content_for_llm(versions: list[dict]) -> str:
    """Build a condensed summary of all versions for the concept extraction prompt.

    Includes version number, word count, section headers, and a content excerpt.
    Keeps total under ~8000 tokens by scaling excerpt length to version count.
    """
    # Scale excerpt size: fewer versions → more content per version
    max_chars_per_version = min(1500, 8000 // max(len(versions), 1))

    parts = []
    for v in versions:
        part = f"## V{v['number']}"
        if v.get("note"):
            part += f" — {v['note']}"
        part += f"\n{v['word_count']} words"

        delta = v.get("word_delta")
        if delta:
            part += f" ({'+' if delta > 0 else ''}{delta} from previous)"

        if v.get("sections"):
            part += f"\nSections: {', '.join(v['sections'][:8])}"

        if v.get("added_sections"):
            part += f"\nNew sections: {', '.join(v['added_sections'][:4])}"
        if v.get("removed_sections"):
            part += f"\nRemoved sections: {', '.join(v['removed_sections'][:4])}"

        # Content excerpt — take from beginning and end for coverage
        body = v["body"]
        if len(body) > max_chars_per_version:
            half = max_chars_per_version // 2
            excerpt = body[:half] + "\n[...]\n" + body[-half:]
        else:
            excerpt = body
        part += f"\nContent:\n{excerpt}"
        parts.append(part)

    return "\n\n---\n\n".join(parts)


def generate_narrative_from_doc(
    doc_path: str | Path,
    name: str | None = None,
    model: str = "gemini/gemini-2.5-flash",
) -> Path:
    """Generate a narrative from a versioned markdown document (no DB needed).

    Three-pass pipeline:
    0. Parse versions + LLM concept extraction
    1. LLM narrative synthesis with concept evolution context
    2. LLM journey diagram generation
    """
    doc_path = Path(doc_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    project_name = name or doc_path.stem.replace(" ", "_")
    versions = _parse_versioned_doc(doc_path)

    logger.info(
        "Parsed %d versions from %s (V%d → V%d, %d → %d words)",
        len(versions), doc_path.name,
        versions[0]["number"], versions[-1]["number"],
        versions[0]["word_count"], versions[-1]["word_count"],
    )

    # --- Pass 0: Concept extraction ---
    logger.info("Extracting concepts across versions...")
    concepts, version_concepts = _extract_doc_concepts(versions, project_name, model)

    # Build concept tracking data
    concept_levels: dict[str, str] = {}
    concept_mentions: Counter[str] = Counter()
    for c in concepts:
        status = c.get("status", "survived")
        concept_levels[c["name"]] = "theme" if status in ("survived", "evolved") else "technique"
        # Count mentions across versions
        for vnum, vconcepts in version_concepts.items():
            if c["name"] in vconcepts:
                concept_mentions[c["name"]] += 1

    one_hit_wonders = sorted(c for c, n in concept_mentions.items() if n == 1)

    # Build "days" data with concepts
    days = []
    seen_concepts: set[str] = set()
    for v in versions:
        vnum_str = str(v["number"])
        v_concepts = version_concepts.get(vnum_str, [])
        new_concepts = [c for c in v_concepts if c not in seen_concepts]
        seen_concepts.update(v_concepts)

        events = [{
            "type": "revision",
            "title": f"Version {v['number']}" + (f": {v['note']}" if v['note'] else ""),
            "summary": f"{v['word_count']} words. Sections: {', '.join(v['sections'][:5]) if v['sections'] else 'no headers'}",
            "concepts": v_concepts,
        }]
        days.append({
            "date": f"V{v['number']}",
            "event_count": 1,
            "events": events,
            "decisions": [],
            "new_concepts": new_concepts,
        })

    top_concepts = sorted(concepts, key=lambda c: concept_mentions.get(c["name"], 0), reverse=True)

    data = {
        "project_name": project_name,
        "first_date": f"V{versions[0]['number']}",
        "last_date": f"V{versions[-1]['number']}",
        "total_days": len(versions),
        "total_events": len(versions),
        "total_concepts": len(concepts),
        "total_decisions": 0,
        "total_sessions": 0,
        "top_concepts": [
            {"name": c["name"], "level": concept_levels.get(c["name"], "theme"),
             "importance": concept_mentions.get(c["name"], 1), "lifecycle": c.get("status", "unknown"),
             "first_seen": f"V{c.get('first_version', '?')}", "last_seen": f"V{c.get('last_version', '?')}"}
            for c in top_concepts[:25]
        ],
        "days": days,
        "one_hit_wonders": one_hit_wonders,
        "concept_links": [],
        "concept_levels": concept_levels,
    }

    # Build rich version summaries with concept + diff data
    version_summaries = []
    for v in versions:
        vnum_str = str(v["number"])
        v_concepts = version_concepts.get(vnum_str, [])
        preview = v["body"][:800].replace("\n", " ").strip()
        added = v.get("added_sections", [])
        removed = v.get("removed_sections", [])
        delta = v.get("word_delta", 0)

        summary = f"### V{v['number']}" + (f" ({v['note']})" if v['note'] else "")
        summary += f"\n{v['word_count']} words"
        if delta:
            summary += f" ({'+' if delta > 0 else ''}{delta})"
        summary += f". Sections: {', '.join(v['sections'][:6])}"
        if added:
            summary += f"\nAdded sections: {', '.join(added[:4])}"
        if removed:
            summary += f"\nRemoved sections: {', '.join(removed[:4])}"
        if v_concepts:
            summary += f"\nActive concepts: {', '.join(v_concepts)}"
        summary += f"\nContent: {preview}..."
        version_summaries.append(summary)

    # --- Pass 1: Narrative with rich context ---
    messages = [
        {"role": "system", "content": """You are narrating the intellectual evolution of a written document across multiple versions.

You have concept data showing how ideas appear, evolve, merge, and get abandoned across versions.
Your job is to tell the STORY of this evolution — not as a changelog, but as an intellectual journey.

Return a JSON object with this exact structure:
{
  "phases": [
    {
      "name": "Phase title",
      "date_range": "V1-V5",
      "color": "#58a6ff",
      "event_count": 5,
      "new_concept_count": 3,
      "paragraphs": [
        {"text": "Narrative paragraph...", "concepts_active": ["concept_name_1", "concept_name_2"]}
      ],
      "key_decision": {"title": "Key editorial decision", "tension": "What was at stake"}
    }
  ],
  "arc_summary": "One paragraph summarizing the full evolutionary arc",
  "epitaph": "One punchy sentence capturing what this document became"
}

Guidelines:
- 3-6 phases grouping versions by major shifts in direction, framing, or audience
- Each phase: 2-4 paragraphs telling the story. Reference specific version numbers.
- concepts_active: list concept names that are relevant to each paragraph (these get highlighted as chips)
- Focus on: framing shifts, audience changes, what got killed, what survived, breakthrough moments
- Mention concepts by their exact names from the concept data
- Phase colors: use visually distinct hex colors"""},
        {"role": "user", "content": f"# Document: {project_name}\n\n"
            + f"Versions: {len(versions)} (V{versions[0]['number']} → V{versions[-1]['number']})\n"
            + f"Words: {versions[0]['word_count']} → {versions[-1]['word_count']}\n\n"
            + "## Concept evolution\n"
            + "\n".join(f"- **{c['name']}** (V{c.get('first_version','?')}→V{c.get('last_version','?')}, {c.get('status','?')}): {c.get('description','')}"
                        + (f" → evolved into {c['evolved_into']}" if c.get('evolved_into') else "")
                        for c in concepts)
            + "\n\n## Version details\n\n"
            + "\n\n".join(version_summaries)},
    ]

    logger.info("Calling LLM for document narrative (%s)...", model)
    result = call_llm(
        model, messages,
        task="doc_narrative",
        trace_id=f"trajectory.doc_narrative.{project_name}",
        max_budget=0,
    )

    narrative = json.loads(_strip_json_fences(result.content))
    logger.info("Narrative: %d phases, cost: $%.4f", len(narrative.get("phases", [])), result.cost or 0)

    # Pass 2: Journey diagram
    logger.info("Generating journey diagram...")
    diagram_messages = render_prompt(
        PROMPTS_DIR / "narrative_diagram.yaml",
        project_name=project_name,
        phases=narrative.get("phases", []),
        arc_summary=narrative.get("arc_summary", ""),
        epitaph=narrative.get("epitaph", ""),
        top_concepts=data.get("top_concepts", []),
    )
    diagram_result = call_llm(
        model, diagram_messages,
        task="doc_narrative_diagram",
        trace_id=f"trajectory.doc_diagram.{project_name}",
        max_budget=0,
    )
    graph_data = json.loads(_strip_json_fences(diagram_result.content))
    narrative["graph"] = graph_data
    logger.info("Diagram: %d nodes, %d edges", len(graph_data.get("nodes", [])), len(graph_data.get("edges", [])))

    html = _render_scrollytelling(project_name, data, narrative)

    output_dir = Path(__file__).parent.parent.parent / "data" / "narratives"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{project_name}_narrative.html"
    output_path.write_text(html, encoding="utf-8")

    logger.info("Output: %s", output_path)
    return output_path


def generate_cross_doc_narrative(
    doc_paths: list[str | Path],
    name: str = "cross_doc",
    model: str = "gemini/gemini-2.5-flash",
) -> Path:
    """Generate a unified narrative showing concept migration across multiple documents.

    Pipeline:
    1. Parse each document independently
    2. Extract concepts from each document
    3. LLM merge pass: find shared/migrated concepts across documents
    4. Unified narrative synthesis
    5. Journey diagram showing cross-document concept flows
    """
    # --- Step 1+2: Parse and extract concepts from each document ---
    doc_data: list[dict] = []
    for dp in doc_paths:
        dp = Path(dp)
        if not dp.exists():
            raise FileNotFoundError(f"Document not found: {dp}")
        doc_name = dp.stem.replace(" ", "_")
        versions = _parse_versioned_doc(dp)
        logger.info(
            "Parsed %s: %d versions (V%d→V%d, %d→%d words)",
            doc_name, len(versions),
            versions[0]["number"], versions[-1]["number"],
            versions[0]["word_count"], versions[-1]["word_count"],
        )
        concepts, version_concepts = _extract_doc_concepts(versions, doc_name, model)
        doc_data.append({
            "name": doc_name,
            "path": dp,
            "versions": versions,
            "concepts": concepts,
            "version_concepts": version_concepts,
        })

    # --- Step 3: Cross-document concept merge ---
    logger.info("Merging concepts across %d documents...", len(doc_data))
    merge_input = []
    for dd in doc_data:
        merge_input.append(
            f"## {dd['name']} ({len(dd['versions'])} versions)\n"
            + "\n".join(
                f"- {c['name']} ({c.get('status', '?')}): {c.get('description', '')}"
                for c in dd["concepts"]
            )
        )

    merge_prompt = [
        {"role": "system", "content": """You are analyzing concept migration across multiple related documents by the same author.

These documents were written in parallel or sequentially. Ideas often appear in one document, get refined, and migrate to another.

Your job: identify which concepts are SHARED across documents (same idea, possibly different names) and which concepts MIGRATED (born in one doc, moved to another).

Return a JSON object:
{
  "shared_concepts": [
    {"canonical_name": "name", "description": "what this concept is about",
     "appearances": [{"doc": "doc_name", "local_name": "concept_name_in_that_doc", "status": "survived|evolved|abandoned"}]}
  ],
  "migrations": [
    {"concept": "name", "from_doc": "doc_name", "to_doc": "doc_name",
     "description": "How the concept changed when it migrated"}
  ],
  "doc_unique_concepts": [
    {"doc": "doc_name", "concepts": ["concept_a", "concept_b"],
     "description": "What makes this document's focus distinct"}
  ]
}

Guidelines:
- Be aggressive about matching — same idea with different names IS a shared concept
- A concept that appears in doc A as "personal_data_moat" and doc B as "data_as_competitive_advantage" is the SAME concept
- Migrations show intellectual development across documents
- 5-15 shared concepts, 3-10 migrations. Quality over quantity."""},
        {"role": "user", "content": "\n\n".join(merge_input)},
    ]

    merge_result = call_llm(
        model, merge_prompt,
        task="cross_doc_concept_merge",
        trace_id=f"trajectory.cross_doc_merge.{name}",
        max_budget=0,
    )
    merged = json.loads(_strip_json_fences(merge_result.content))
    shared = merged.get("shared_concepts", [])
    migrations = merged.get("migrations", [])
    doc_unique = merged.get("doc_unique_concepts", [])
    logger.info(
        "Merged: %d shared concepts, %d migrations, %d doc-unique sets",
        len(shared), len(migrations), len(doc_unique),
    )

    # --- Step 4: Build unified data + narrative ---
    # Build combined concept tracking
    all_concepts: list[dict] = []
    concept_levels: dict[str, str] = {}
    concept_mentions: Counter[str] = Counter()

    for sc in shared:
        cname = sc["canonical_name"]
        concept_levels[cname] = "theme"
        concept_mentions[cname] = len(sc.get("appearances", []))
        all_concepts.append({
            "name": cname,
            "status": "survived",
            "description": sc.get("description", ""),
            "cross_doc": True,
            "doc_count": len(sc.get("appearances", [])),
        })

    for dd in doc_data:
        for c in dd["concepts"]:
            # Skip if already covered by a shared concept
            is_shared = any(
                any(a.get("local_name") == c["name"] and a.get("doc") == dd["name"]
                    for a in sc.get("appearances", []))
                for sc in shared
            )
            if not is_shared:
                prefixed = f"{dd['name']}:{c['name']}"
                concept_levels[prefixed] = "technique"
                concept_mentions[prefixed] = 1
                all_concepts.append({
                    "name": prefixed,
                    "status": c.get("status", "unknown"),
                    "description": c.get("description", ""),
                    "cross_doc": False,
                    "doc_count": 1,
                })

    # Build timeline "days" — interleave versions from all docs
    days = []
    seen_concepts: set[str] = set()
    for dd in doc_data:
        for v in dd["versions"]:
            vnum_str = str(v["number"])
            v_concepts = dd["version_concepts"].get(vnum_str, [])
            # Map local concept names to canonical shared names where applicable
            mapped_concepts = []
            for vc in v_concepts:
                canonical = None
                for sc in shared:
                    for a in sc.get("appearances", []):
                        if a.get("local_name") == vc and a.get("doc") == dd["name"]:
                            canonical = sc["canonical_name"]
                            break
                    if canonical:
                        break
                mapped_concepts.append(canonical or f"{dd['name']}:{vc}")

            new_concepts = [c for c in mapped_concepts if c not in seen_concepts]
            seen_concepts.update(mapped_concepts)

            days.append({
                "date": f"{dd['name']}:V{v['number']}",
                "event_count": 1,
                "events": [{
                    "type": "revision",
                    "title": f"{dd['name']} V{v['number']}" + (f": {v['note']}" if v['note'] else ""),
                    "summary": f"{v['word_count']} words",
                    "concepts": mapped_concepts,
                }],
                "decisions": [],
                "new_concepts": new_concepts,
            })

    one_hit_wonders = sorted(c for c, n in concept_mentions.items() if n == 1)
    top_concepts = sorted(all_concepts, key=lambda c: concept_mentions.get(c["name"], 0), reverse=True)

    data = {
        "project_name": name,
        "first_date": days[0]["date"] if days else "?",
        "last_date": days[-1]["date"] if days else "?",
        "total_days": len(days),
        "total_events": len(days),
        "total_concepts": len(all_concepts),
        "total_decisions": 0,
        "total_sessions": 0,
        "top_concepts": [
            {"name": c["name"], "level": concept_levels.get(c["name"], "theme"),
             "importance": concept_mentions.get(c["name"], 1),
             "lifecycle": c.get("status", "unknown"),
             "first_seen": "cross-doc" if c.get("cross_doc") else "single-doc",
             "last_seen": f"{c.get('doc_count', 1)} docs"}
            for c in top_concepts[:30]
        ],
        "days": days,
        "one_hit_wonders": one_hit_wonders,
        "concept_links": [],
        "concept_levels": concept_levels,
    }

    # Build narrative prompt with cross-doc context
    doc_summaries = []
    for dd in doc_data:
        vs = dd["versions"]
        doc_summaries.append(
            f"### {dd['name']}\n"
            + f"{len(vs)} versions, {vs[0]['word_count']}→{vs[-1]['word_count']} words\n"
            + f"Concepts: {', '.join(c['name'] for c in dd['concepts'][:10])}"
        )

    messages = [
        {"role": "system", "content": """You are narrating the intellectual evolution across MULTIPLE related documents by the same author.

These documents share ideas that migrated between them. Tell the story of how the author's thinking evolved ACROSS documents — not document by document, but thematically.

Return a JSON object with this exact structure:
{
  "phases": [
    {
      "name": "Phase title",
      "date_range": "Description of what docs/versions this covers",
      "color": "#58a6ff",
      "event_count": 5,
      "new_concept_count": 3,
      "paragraphs": [
        {"text": "Narrative paragraph...", "concepts_active": ["concept_name_1", "concept_name_2"]}
      ],
      "key_decision": {"title": "Key intellectual shift", "tension": "What was at stake"}
    }
  ],
  "arc_summary": "One paragraph summarizing the full intellectual arc across all documents",
  "epitaph": "One punchy sentence capturing the author's journey"
}

Guidelines:
- 4-7 phases organized THEMATICALLY, not per-document
- Reference specific documents and version numbers
- Highlight concept MIGRATIONS: when an idea born in one doc appears in another
- Show how the author's voice and framing evolved
- Use the shared concept canonical names (not doc-local names)
- Phase colors: use visually distinct hex colors"""},
        {"role": "user", "content": f"# Cross-Document Analysis: {name}\n\n"
            + f"Documents: {len(doc_data)}\n\n"
            + "## Documents\n\n" + "\n\n".join(doc_summaries)
            + "\n\n## Shared concepts (appear in 2+ docs)\n"
            + "\n".join(
                f"- **{sc['canonical_name']}**: {sc.get('description', '')} "
                + f"(in: {', '.join(a['doc'] for a in sc.get('appearances', []))})"
                for sc in shared)
            + "\n\n## Migrations\n"
            + "\n".join(
                f"- **{m['concept']}**: {m['from_doc']} → {m['to_doc']}: {m.get('description', '')}"
                for m in migrations)
            + "\n\n## Document-unique concepts\n"
            + "\n".join(
                f"- **{du['doc']}**: {', '.join(du.get('concepts', [])[:5])} — {du.get('description', '')}"
                for du in doc_unique)},
    ]

    logger.info("Generating cross-document narrative (%s)...", model)
    result = call_llm(
        model, messages,
        task="cross_doc_narrative",
        trace_id=f"trajectory.cross_doc_narrative.{name}",
        max_budget=0,
    )
    narrative = json.loads(_strip_json_fences(result.content))
    logger.info("Narrative: %d phases, cost: $%.4f", len(narrative.get("phases", [])), result.cost or 0)

    # --- Step 5: Journey diagram ---
    logger.info("Generating cross-document journey diagram...")
    diagram_messages = render_prompt(
        PROMPTS_DIR / "narrative_diagram.yaml",
        project_name=name,
        phases=narrative.get("phases", []),
        arc_summary=narrative.get("arc_summary", ""),
        epitaph=narrative.get("epitaph", ""),
        top_concepts=data.get("top_concepts", []),
    )
    diagram_result = call_llm(
        model, diagram_messages,
        task="cross_doc_diagram",
        trace_id=f"trajectory.cross_doc_diagram.{name}",
        max_budget=0,
    )
    graph_data = json.loads(_strip_json_fences(diagram_result.content))
    narrative["graph"] = graph_data
    logger.info("Diagram: %d nodes, %d edges", len(graph_data.get("nodes", [])), len(graph_data.get("edges", [])))

    html = _render_scrollytelling(name, data, narrative)

    output_dir = Path(__file__).parent.parent.parent / "data" / "narratives"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_narrative.html"
    output_path.write_text(html, encoding="utf-8")

    logger.info("Output: %s", output_path)
    return output_path


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

    # --- Pass 2: Journey diagram derived from the narrative ---
    logger.info("Generating journey diagram from narrative...")
    diagram_messages = render_prompt(
        PROMPTS_DIR / "narrative_diagram.yaml",
        project_name=project_name,
        phases=narrative.get("phases", []),
        arc_summary=narrative.get("arc_summary", ""),
        epitaph=narrative.get("epitaph", ""),
        top_concepts=[c["name"] for c in data["top_concepts"][:20]],
    )
    diagram_result = call_llm(
        model,
        diagram_messages,
        task="narrative_diagram",
        trace_id=f"trajectory.narrative_diagram.{project_name}",
        max_budget=0,
    )
    diagram_raw = diagram_result.content.strip()
    if diagram_raw.startswith("```"):
        diagram_raw = re.sub(r"^```(?:json)?\s*", "", diagram_raw)
        diagram_raw = re.sub(r"\s*```$", "", diagram_raw)
    graph_data = json.loads(diagram_raw)
    narrative["graph"] = graph_data
    logger.info(
        "Diagram: %d nodes, %d edges, cost: $%.4f",
        len(graph_data.get("nodes", [])),
        len(graph_data.get("edges", [])),
        diagram_result.cost or 0,
    )

    html = _render_scrollytelling(project_name, data, narrative)

    output_dir = Path(__file__).parent.parent.parent / "data" / "narratives"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{project_name}_narrative.html"
    output_path.write_text(html, encoding="utf-8")

    logger.info("Output: %s", output_path)
    return output_path


def record_video(
    html_path: Path,
    output_path: Path | None = None,
    width: int = 1920,
    height: int = 1080,
    speed: float = 2.0,
) -> Path:
    """Record the narrative cinema mode as a WebM video using Playwright.

    Opens the HTML in a headless browser with ?autoplay, records until
    cinema mode completes, then saves the recording.
    """
    import shutil

    from playwright.sync_api import sync_playwright

    html_path = Path(html_path).resolve()
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    if output_path is None:
        output_path = html_path.with_suffix(".webm")
    output_path = Path(output_path)

    # Use a temp dir for recording so we don't pollute the output dir
    tmp_dir = output_path.parent / ".video_tmp"
    tmp_dir.mkdir(exist_ok=True)

    logger.info(
        "Recording video: %s → %s (%dx%d, speed=%sx)",
        html_path.name, output_path.name, width, height, speed,
    )

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(tmp_dir),
            record_video_size={"width": width, "height": height},
        )
        page = context.new_page()

        # Load the narrative with autoplay
        url = f"file://{html_path}?autoplay"
        page.goto(url, wait_until="networkidle")

        # Set playback speed via JS
        speed_val = float(speed)
        page.evaluate(f"""() => {{
            const speeds = [0.5, 1, 2, 4];
            const targetIdx = speeds.indexOf({speed_val});
            const speedBtn = document.getElementById('cinema-speed');
            if (speedBtn && targetIdx > 1) {{
                for (let i = 1; i < targetIdx; i++) speedBtn.click();
            }}
        }}""")

        # Wait for cinema to finish — poll the progress bar
        logger.info("Cinema playing... waiting for completion")
        page.wait_for_function(
            """() => {
                const progress = document.getElementById('cinema-progress');
                return progress && progress.style.width === '100%';
            }""",
            timeout=600000,  # 10 min max
        )

        logger.info("Cinema completed. Finalizing video...")
        page.wait_for_timeout(3000)

        # Must save video path before closing page
        video = page.video
        if video:
            video_path = video.path()
        else:
            video_path = None

        page.close()
        context.close()
        browser.close()

    # Move recorded file to desired output path
    if video_path and Path(video_path).exists():
        shutil.move(str(video_path), str(output_path))
    else:
        # Fallback: find the most recent webm in tmp dir
        webms = sorted(tmp_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime, reverse=True)
        if webms:
            shutil.move(str(webms[0]), str(output_path))
        else:
            raise RuntimeError("Video recording failed — no output file found")

    # Clean up tmp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Video saved: %s (%.1f MB)", output_path, size_mb)
    return output_path


def generate_supercut(
    db: TrajectoryDB,
    project_names: list[str],
    model: str = "gemini/gemini-2.5-flash",
    speed: float = 4.0,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Generate narratives for multiple projects and stitch into one supercut video.

    Each project gets its own narrative + video recording, then all are
    concatenated with ffmpeg (with fade transitions) into a single MP4.
    """
    import shutil
    import subprocess

    output_dir = Path(__file__).parent.parent.parent / "data" / "narratives"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate narratives and record videos for each project
    video_paths: list[Path] = []
    for name in project_names:
        logger.info("=== Generating narrative for %s ===", name)
        html_path = generate_narrative(db, project_name=name, model=model)
        logger.info("=== Recording video for %s ===", name)
        video_path = record_video(html_path, speed=speed, width=width, height=height)
        video_paths.append(video_path)

    if len(video_paths) < 2:
        logger.info("Only one project — no supercut needed")
        return video_paths[0]

    # Step 2: Check for ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # Try conda path
        conda_ffmpeg = Path.home() / "miniconda3" / "bin" / "ffmpeg"
        if conda_ffmpeg.exists():
            ffmpeg = str(conda_ffmpeg)

    if not ffmpeg:
        logger.warning("ffmpeg not found — returning individual videos without stitching")
        return video_paths[0]

    # Step 3: Convert each WebM to MP4 for reliable concatenation
    mp4_paths: list[Path] = []
    for vp in video_paths:
        mp4 = vp.with_suffix(".mp4")
        logger.info("Converting %s → %s", vp.name, mp4.name)
        subprocess.run(
            [ffmpeg, "-y", "-i", str(vp), "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-pix_fmt", "yuv420p",
             "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
             str(mp4)],
            check=True, capture_output=True,
        )
        mp4_paths.append(mp4)

    # Step 4: Build concat file for ffmpeg
    concat_file = output_dir / ".concat_list.txt"
    with open(concat_file, "w") as f:
        for mp4 in mp4_paths:
            f.write(f"file '{mp4}'\n")

    # Step 5: Concatenate into supercut
    supercut_path = output_dir / "supercut.mp4"
    logger.info("Stitching %d videos into supercut...", len(mp4_paths))
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
         "-c", "copy", str(supercut_path)],
        check=True, capture_output=True,
    )

    # Clean up
    concat_file.unlink(missing_ok=True)

    size_mb = supercut_path.stat().st_size / 1e6
    logger.info("Supercut saved: %s (%.1f MB, %d projects)", supercut_path, size_mb, len(project_names))
    return supercut_path


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


def _render_scrollytelling(
    project_name: str, data: dict, narrative: dict, *, integrated: bool = False,
) -> str:
    """Render the full scrollytelling HTML page.

    Args:
        integrated: If True, graph fills viewport and text floats over it as cards.
    """

    # Build phase sections HTML
    phase_sections = []

    for i, phase in enumerate(narrative.get("phases", [])):
        color = phase.get("color", "#58a6ff")
        paragraphs_html = []

        for j, para in enumerate(phase.get("paragraphs", [])):
            raw_text = para.get("text", "")
            # Strip HTML tags the LLM may have injected (e.g. <mark>, <em>)
            raw_text = re.sub(r"</?(?:mark|em|strong|b|i|span)[^>]*>", "", raw_text)
            text = _inline_md(_esc(raw_text))
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
            <div class="phase-header reveal" data-phase="{i}">
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

    # Graph data from second LLM pass (D3 + dagre rendering)
    graph_data = narrative.get("graph", {"nodes": [], "edges": []})
    graph_json = json.dumps(graph_data)

    # Phase names for labels
    phase_names = [p.get("name", f"Phase {i+1}") for i, p in enumerate(narrative.get("phases", []))]

    # Build timeline metadata for autoplay: [{phase, step, date_range, para_count}]
    timeline_steps = []
    for i, phase in enumerate(narrative.get("phases", [])):
        n_paras = len(phase.get("paragraphs", []))
        for j in range(n_paras):
            timeline_steps.append({
                "phase": i,
                "step": j,
                "date_range": phase.get("date_range", ""),
                "phase_name": phase.get("name", f"Phase {i+1}"),
            })
    timeline_json = json.dumps(timeline_steps)

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

/* --- Layout: split mode (default) --- */
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
    height: calc(100% - 48px);
    overflow: auto;
    position: relative;
    cursor: grab;
}}

/* --- Layout: integrated mode --- */
.scrolly-container.integrated {{
    display: block;
}}
.scrolly-container.integrated .graph-track {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 1;
    border-left: none;
}}
.scrolly-container.integrated .graph-track .zoom-controls {{
    position: fixed;
    top: auto;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    border-radius: 8px;
    border: 1px solid var(--border);
    background: rgba(22, 27, 34, 0.85);
    backdrop-filter: blur(8px);
    z-index: 20;
}}
.scrolly-container.integrated .diagram-container {{
    height: 100vh;
}}
.scrolly-container.integrated .narrative-track {{
    position: relative;
    z-index: 10;
    width: 44%;
    max-width: 520px;
    padding: 100vh 40px 100vh 48px;
}}
.scrolly-container.integrated .narrative-track .phase {{
    background: rgba(13, 17, 23, 0.82);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px 36px;
    margin-bottom: 80vh;
}}
.scrolly-container.integrated .narrative-track .phase:last-child {{
    margin-bottom: 40vh;
}}
.scrolly-container.integrated .narrative-track .phase-header {{
    margin-bottom: 20px;
}}
.scrolly-container.integrated .narrative-track p {{
    font-size: 16px;
    line-height: 1.8;
}}
.scrolly-container.integrated .pull-quote {{
    border-radius: 8px;
}}
.diagram-container:active {{ cursor: grabbing; }}
#diagram-svg {{
    transform-origin: 0 0;
    transition: transform 0.15s ease;
    display: block;
}}
/* D3 graph node styles */
.graph-node rect {{
    fill: var(--bg-card);
    stroke: var(--blue);
    stroke-width: 1.5;
}}
.graph-node.type-decision polygon {{
    fill: var(--bg-card);
    stroke: var(--orange);
    stroke-width: 2;
}}
.graph-node.type-dead rect {{
    fill: var(--bg-card);
    stroke: var(--red);
    stroke-width: 1.5;
    stroke-dasharray: 6 3;
}}
.graph-node.type-win rect {{
    fill: var(--bg-card);
    stroke: var(--green);
    stroke-width: 3;
    filter: drop-shadow(0 0 8px rgba(126, 231, 135, 0.4));
}}
/* Node popover */
.node-popover {{
    position: absolute;
    z-index: 20;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    max-width: 320px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    pointer-events: auto;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.2s, transform 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.node-popover.visible {{
    opacity: 1;
    transform: translateY(0);
}}
.node-popover-title {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text-bright);
    margin-bottom: 6px;
}}
.node-popover-type {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
}}
.node-popover-type.type-beat {{ background: rgba(88,166,255,0.15); color: var(--blue); }}
.node-popover-type.type-decision {{ background: rgba(247,129,102,0.15); color: var(--orange); }}
.node-popover-type.type-dead {{ background: rgba(255,123,114,0.15); color: var(--red); }}
.node-popover-type.type-win {{ background: rgba(126,231,135,0.15); color: var(--green); }}
.node-popover-phase {{
    font-size: 11px;
    color: var(--text-dimmer);
    margin-bottom: 10px;
}}
.node-popover-connections {{
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.6;
}}
.node-popover-connections .edge-in {{
    color: var(--text-dim);
}}
.node-popover-connections .edge-out {{
    color: var(--text);
}}
.node-popover-connections .edge-verb {{
    color: var(--purple);
    font-style: italic;
}}
.node-popover-close {{
    position: absolute;
    top: 8px;
    right: 12px;
    background: none;
    border: none;
    color: var(--text-dimmer);
    font-size: 16px;
    cursor: pointer;
    line-height: 1;
}}
.node-popover-close:hover {{ color: var(--text-dim); }}

/* Node breathing pulse when revealed */
.graph-node.revealed rect,
.graph-node.revealed polygon {{
    animation: nodePulse 3s ease-in-out infinite;
}}
.graph-node.type-win.revealed rect {{
    animation: winPulse 2s ease-in-out infinite;
}}
@keyframes nodePulse {{
    0%, 100% {{ filter: drop-shadow(0 0 0px transparent); }}
    50% {{ filter: drop-shadow(0 0 6px rgba(88, 166, 255, 0.25)); }}
}}
@keyframes winPulse {{
    0%, 100% {{ filter: drop-shadow(0 0 8px rgba(126, 231, 135, 0.4)); }}
    50% {{ filter: drop-shadow(0 0 18px rgba(126, 231, 135, 0.7)); }}
}}
.graph-node text {{
    fill: var(--text);
    font-size: 11px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.graph-node.type-decision text {{ fill: var(--orange); }}
.graph-node.type-dead text {{ fill: var(--red); opacity: 0.8; }}
.graph-node.type-win text {{ fill: var(--green); }}
.graph-edge path {{
    fill: none;
    stroke: var(--text-dimmer);
    stroke-width: 1.5;
}}
.graph-edge text {{
    fill: var(--text-dim);
    font-size: 9px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.phase-bg {{
    fill: rgba(88, 166, 255, 0.03);
    stroke: rgba(88, 166, 255, 0.08);
    stroke-width: 1;
    rx: 8;
}}
.phase-label-text {{
    fill: var(--text-dimmer);
    font-size: 10px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
/* Zoom controls */
.zoom-controls {{
    display: flex;
    gap: 4px;
    padding: 8px 12px;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    z-index: 10;
}}
.zoom-btn {{
    width: 32px;
    height: 32px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg);
    color: var(--text-dim);
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s;
}}
.zoom-btn:hover {{ background: var(--bg-card); color: var(--text-bright); }}
/* Fullscreen graph */
.graph-track.fullscreen {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 9999;
    background: var(--bg);
    border-left: none;
}}
.graph-track.fullscreen .diagram-container {{
    height: calc(100vh - 48px);
}}
.zoom-level {{
    font-size: 11px;
    color: var(--text-dimmer);
    align-self: center;
    margin-left: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.phase-indicator {{
    font-size: 11px;
    color: var(--text-dim);
    align-self: center;
    margin-left: auto;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
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
.concept-chip.hover-linked {{
    box-shadow: 0 0 10px currentColor;
    transform: scale(1.04);
    transition: all 0.15s;
}}
.graph-node.hover-linked rect,
.graph-node.hover-linked polygon {{
    filter: drop-shadow(0 0 12px rgba(255, 255, 255, 0.6));
    stroke-width: 3;
    transition: all 0.15s;
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

/* --- Cinema bar (autoplay) --- */
.cinema-bar {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 48px;
    background: rgba(13, 17, 23, 0.95);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 16px;
    z-index: 100;
    backdrop-filter: blur(8px);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.cinema-play {{
    background: none;
    border: 1px solid var(--blue);
    color: var(--blue);
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s;
}}
.cinema-play:hover {{ background: rgba(88, 166, 255, 0.1); }}
.cinema-play.playing {{
    border-color: var(--orange);
    color: var(--orange);
}}
.cinema-timeline {{
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    cursor: pointer;
}}
.cinema-progress {{
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--blue), var(--purple));
    border-radius: 2px;
    transition: width 0.3s ease;
}}
.cinema-date {{
    color: var(--text-dim);
    font-size: 12px;
    min-width: 80px;
    text-align: right;
}}
.cinema-time {{
    color: var(--text-dimmer);
    font-size: 11px;
    min-width: 40px;
}}

/* Typewriter mode */
.reveal.typewriter {{
    opacity: 1 !important;
    transform: none !important;
}}
.reveal.typewriter .typewriter-cursor {{
    display: inline-block;
    width: 2px;
    height: 1em;
    background: var(--blue);
    margin-left: 2px;
    animation: blink 0.6s step-end infinite;
    vertical-align: text-bottom;
}}
@keyframes blink {{
    50% {{ opacity: 0; }}
}}

/* --- Phase flash overlay --- */
.phase-flash {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 50;
    opacity: 0;
}}
.phase-flash.active {{
    animation: phaseFlash 0.8s ease-out forwards;
}}
@keyframes phaseFlash {{
    0% {{ opacity: 0.15; }}
    100% {{ opacity: 0; }}
}}

/* --- Particle canvas --- */
#particle-canvas {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -1;
    pointer-events: none;
}}

/* --- Sound toggle --- */
.sound-toggle {{
    background: none;
    border: 1px solid var(--border);
    color: var(--text-dimmer);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
.sound-toggle:hover {{ color: var(--text-dim); border-color: var(--text-dim); }}
.sound-toggle.active {{ color: var(--green); border-color: var(--green); }}

/* --- Speed control --- */
.cinema-speed {{
    background: none;
    border: 1px solid var(--border);
    color: var(--text-dimmer);
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    min-width: 36px;
    text-align: center;
}}
.cinema-speed:hover {{ color: var(--text-dim); border-color: var(--text-dim); }}

/* --- Footer --- */
.footer {{
    text-align: center;
    padding: 40px 24px 80px;
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
<canvas id="particle-canvas"></canvas>
<div class="phase-flash" id="phase-flash"></div>

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
<div class="scrolly-container{' integrated' if integrated else ''}">
    <div class="narrative-track">
        {''.join(phase_sections)}
    </div>
    <div class="graph-track">
        <div class="zoom-controls">
            <button class="zoom-btn" id="zoom-in" title="Zoom in">+</button>
            <button class="zoom-btn" id="zoom-out" title="Zoom out">&minus;</button>
            <button class="zoom-btn" id="zoom-reset" title="Reset zoom">&#x21bb;</button>
            <span class="zoom-level" id="zoom-level">100%</span>
            <span class="phase-indicator" id="phase-indicator"></span>
            <button class="zoom-btn" id="fullscreen-toggle" title="Toggle fullscreen" style="margin-left:4px">&#x26F6;</button>
        </div>
        <div class="diagram-container">
            <svg id="diagram-svg"></svg>
            <div class="node-popover" id="node-popover">
                <button class="node-popover-close" id="popover-close">&times;</button>
                <div class="node-popover-title" id="popover-title"></div>
                <span class="node-popover-type" id="popover-type"></span>
                <div class="node-popover-phase" id="popover-phase"></div>
                <div class="node-popover-connections" id="popover-connections"></div>
            </div>
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

<!-- Autoplay controls -->
<div class="cinema-bar" id="cinema-bar">
    <button class="cinema-play" id="cinema-play" title="Play cinematic mode">&#9654; Play</button>
    <button class="cinema-speed" id="cinema-speed" title="Playback speed">1x</button>
    <div class="cinema-timeline">
        <div class="cinema-progress" id="cinema-progress"></div>
    </div>
    <span class="cinema-date" id="cinema-date">{_esc(stats['first_date'])}</span>
    <span class="cinema-time" id="cinema-time"></span>
    <button class="sound-toggle" id="sound-toggle" title="Toggle sound">Sound Off</button>
</div>

<div class="footer">Generated by trajectory &middot; {datetime.now().strftime('%b %d, %Y')}</div>

<!-- Particle system -->
<script>
(function() {{
    const canvas = document.getElementById('particle-canvas');
    const ctx = canvas.getContext('2d');
    const particles = [];
    const MAX_AMBIENT = 60;

    function resize() {{
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }}
    window.addEventListener('resize', resize);
    resize();

    const COLORS = {{
        beat: [88, 166, 255],
        decision: [247, 129, 102],
        dead: [255, 123, 114],
        win: [126, 231, 135],
        ambient: [139, 148, 158],
    }};

    function Particle(x, y, color, opts) {{
        this.x = x;
        this.y = y;
        this.r = opts.r || (1 + Math.random() * 2);
        this.vx = opts.vx || (Math.random() - 0.5) * 0.3;
        this.vy = opts.vy || -0.2 - Math.random() * 0.3;
        this.life = opts.life || (120 + Math.random() * 180);
        this.maxLife = this.life;
        this.color = color;
        this.ambient = opts.ambient || false;
    }}

    // Seed ambient particles
    for (let i = 0; i < MAX_AMBIENT; i++) {{
        particles.push(new Particle(
            Math.random() * window.innerWidth,
            Math.random() * window.innerHeight,
            COLORS.ambient,
            {{ r: 0.5 + Math.random() * 1.5, life: 300 + Math.random() * 300, ambient: true,
               vx: (Math.random() - 0.5) * 0.15, vy: -0.05 - Math.random() * 0.1 }}
        ));
    }}

    function tick() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = particles.length - 1; i >= 0; i--) {{
            const p = particles[i];

            // Edge-traveling particles follow SVG path
            if (p._traveler) {{
                p._traveler.t += 0.025;
                if (p._traveler.t >= 1) {{ p.life = 0; }}
                else {{
                    try {{
                        const pt = p._traveler.pathEl.getPointAtLength(
                            p._traveler.t * p._traveler.totalLen);
                        p.x = p._traveler.rect.left + (pt.x * p._traveler.z) - p._traveler.sx;
                        p.y = p._traveler.rect.top + (pt.y * p._traveler.z) - p._traveler.sy;
                    }} catch(e) {{ p.life = 0; }}
                }}
            }} else {{
                p.x += p.vx;
                p.y += p.vy;
            }}
            p.life--;

            if (p.ambient) {{
                if (p.y < -10) {{ p.y = canvas.height + 10; p.x = Math.random() * canvas.width; }}
                if (p.x < -10) p.x = canvas.width + 10;
                if (p.x > canvas.width + 10) p.x = -10;
                if (p.life <= 0) {{ p.life = p.maxLife; }}
            }}

            const alpha = p.ambient
                ? 0.15 + 0.1 * Math.sin(p.life * 0.02)
                : p._traveler ? 0.85
                : Math.max(0, p.life / p.maxLife) * 0.7;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(' + p.color[0] + ',' + p.color[1] + ',' + p.color[2] + ',' + alpha + ')';
            ctx.fill();

            // Glow halo
            if (!p.ambient && p.r > 1) {{
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r * (p._traveler ? 5 : 3), 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(' + p.color[0] + ',' + p.color[1] + ',' + p.color[2] + ',' + (alpha * 0.2) + ')';
                ctx.fill();
            }}

            if (!p.ambient && p.life <= 0) {{
                particles.splice(i, 1);
            }}
        }}
        requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);

    // Burst particles from a screen position
    window._particleBurst = function(screenX, screenY, type) {{
        const color = COLORS[type] || COLORS.beat;
        const count = type === 'win' ? 30 : type === 'decision' ? 20 : 12;
        for (let i = 0; i < count; i++) {{
            const angle = (Math.PI * 2 / count) * i + (Math.random() - 0.5) * 0.5;
            const speed = 1 + Math.random() * 2;
            particles.push(new Particle(screenX, screenY, color, {{
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                r: 1 + Math.random() * (type === 'win' ? 3 : 2),
                life: 60 + Math.random() * 60,
            }}));
        }}
    }};

    // Edge-traveling particles: spawn particles that flow along an SVG path
    window._edgeParticles = function(pathEl, containerRect, scrollLeft, scrollTop, zoomLvl) {{
        if (!pathEl) return;
        const totalLen = pathEl.getTotalLength();
        if (totalLen < 10) return;
        const color = COLORS.beat;
        const count = Math.min(8, Math.max(3, Math.floor(totalLen / 30)));
        for (let i = 0; i < count; i++) {{
            setTimeout(function() {{
                var p = new Particle(0, 0, color, {{
                    r: 1.5 + Math.random(), life: 80, vx: 0, vy: 0 }});
                p._traveler = {{ pathEl: pathEl, totalLen: totalLen, t: 0,
                    rect: containerRect, sx: scrollLeft, sy: scrollTop, z: zoomLvl }};
                particles.push(p);
            }}, i * 60);
        }}
    }};
}})();
</script>

<!-- Web Audio synth -->
<script>
(function() {{
    var audioCtx = null;
    var soundEnabled = false;
    var masterGain = null;
    var droneOsc = null;
    var droneGain = null;

    // Note frequencies (C major pentatonic across octaves)
    var NOTES = {{
        beat: [523.25, 587.33, 659.25, 783.99],     // C5-G5
        decision: [392.00, 440.00, 493.88],           // G4-B4
        dead: [196.00, 220.00],                        // G3-A3
        win: [783.99, 880.00, 987.77, 1046.50],       // G5-C6
    }};

    function initAudio() {{
        if (audioCtx) return;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        masterGain = audioCtx.createGain();
        masterGain.gain.value = 0.3;
        masterGain.connect(audioCtx.destination);

        // Ambient drone — two detuned sine waves for warmth
        droneGain = audioCtx.createGain();
        droneGain.gain.value = 0;
        droneGain.connect(masterGain);

        droneOsc = audioCtx.createOscillator();
        droneOsc.type = 'sine';
        droneOsc.frequency.value = 110; // A2
        droneOsc.connect(droneGain);
        droneOsc.start();

        var drone2 = audioCtx.createOscillator();
        drone2.type = 'sine';
        drone2.frequency.value = 110.5; // Slightly detuned for beating
        drone2.connect(droneGain);
        drone2.start();

        // Reverb via convolver (simple impulse)
        window._reverbGain = audioCtx.createGain();
        window._reverbGain.gain.value = 0.2;
        window._reverbGain.connect(masterGain);
    }}

    function startDrone() {{
        if (!droneGain) return;
        droneGain.gain.cancelScheduledValues(audioCtx.currentTime);
        droneGain.gain.setValueAtTime(droneGain.gain.value, audioCtx.currentTime);
        droneGain.gain.linearRampToValueAtTime(0.06, audioCtx.currentTime + 2);
    }}

    function stopDrone() {{
        if (!droneGain) return;
        droneGain.gain.cancelScheduledValues(audioCtx.currentTime);
        droneGain.gain.setValueAtTime(droneGain.gain.value, audioCtx.currentTime);
        droneGain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 1);
    }}

    function playNodeSound(type) {{
        if (!audioCtx || !soundEnabled) return;
        var notes = NOTES[type] || NOTES.beat;
        var freq = notes[Math.floor(Math.random() * notes.length)];
        var now = audioCtx.currentTime;

        if (type === 'win') {{
            // Arpeggio for breakthroughs
            notes.forEach(function(f, i) {{
                var osc = audioCtx.createOscillator();
                var gain = audioCtx.createGain();
                osc.type = 'triangle';
                osc.frequency.value = f;
                gain.gain.setValueAtTime(0, now + i * 0.1);
                gain.gain.linearRampToValueAtTime(0.2, now + i * 0.1 + 0.05);
                gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.1 + 0.8);
                osc.connect(gain);
                gain.connect(masterGain);
                osc.start(now + i * 0.1);
                osc.stop(now + i * 0.1 + 0.8);
            }});
        }} else if (type === 'dead') {{
            // Low thud for dead ends
            var osc = audioCtx.createOscillator();
            var gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now);
            osc.frequency.exponentialRampToValueAtTime(80, now + 0.3);
            gain.gain.setValueAtTime(0.25, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
            osc.connect(gain);
            gain.connect(masterGain);
            osc.start(now);
            osc.stop(now + 0.5);
        }} else if (type === 'decision') {{
            // Two-note chord
            for (var di = 0; di < 2; di++) {{
                var osc = audioCtx.createOscillator();
                var gain = audioCtx.createGain();
                osc.type = 'triangle';
                osc.frequency.value = notes[di];
                gain.gain.setValueAtTime(0.15, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.6);
                osc.connect(gain);
                gain.connect(masterGain);
                osc.start(now);
                osc.stop(now + 0.6);
            }}
        }} else {{
            // Bell ping for beats
            var osc = audioCtx.createOscillator();
            var gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
            osc.connect(gain);
            gain.connect(masterGain);
            osc.start(now);
            osc.stop(now + 0.4);
        }}
    }}

    function playEdgeSound() {{
        if (!audioCtx || !soundEnabled) return;
        // Soft whoosh — filtered noise burst
        var bufferSize = audioCtx.sampleRate * 0.15;
        var buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
        var data = buffer.getChannelData(0);
        for (var i = 0; i < bufferSize; i++) data[i] = (Math.random() * 2 - 1) * 0.1;
        var noise = audioCtx.createBufferSource();
        noise.buffer = buffer;
        var filter = audioCtx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.value = 800;
        filter.Q.value = 0.5;
        var gain = audioCtx.createGain();
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.15);
        noise.connect(filter);
        filter.connect(gain);
        gain.connect(masterGain);
        noise.start();
        noise.stop(audioCtx.currentTime + 0.15);
    }}

    // Sound toggle button
    var btn = document.getElementById('sound-toggle');
    btn.addEventListener('click', function() {{
        if (!soundEnabled) {{
            initAudio();
            soundEnabled = true;
            btn.textContent = 'Sound On';
            btn.classList.add('active');
            if (audioCtx.state === 'suspended') audioCtx.resume();
            startDrone();
        }} else {{
            soundEnabled = false;
            btn.textContent = 'Sound Off';
            btn.classList.remove('active');
            stopDrone();
        }}
    }});

    // Expose to other scripts
    window._playNodeSound = playNodeSound;
    window._playEdgeSound = playEdgeSound;
    window._startDrone = function() {{ if (soundEnabled) startDrone(); }};
    window._stopDrone = function() {{ stopDrone(); }};
}})();
</script>

<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script type="module">
import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm';

const GRAPH_DATA = {graph_json};
const PHASE_NAMES = {json.dumps(phase_names)};
const NODE_W = 170, NODE_H = 48, PAD = 40;

// --- Build dagre layout ---
const g = new dagre.graphlib.Graph({{ compound: true }});
g.setGraph({{ rankdir: 'TB', ranksep: 70, nodesep: 35, marginx: PAD, marginy: PAD }});
g.setDefaultEdgeLabel(() => ({{}}));

PHASE_NAMES.forEach((name, i) => {{
    g.setNode('_phase_' + i, {{ label: name, clusterLabelPos: 'top' }});
}});

GRAPH_DATA.nodes.forEach(n => {{
    const w = n.type === 'decision' ? NODE_W + 30 : NODE_W;
    const h = n.type === 'decision' ? NODE_H + 16 : NODE_H;
    g.setNode(n.id, {{ width: w, height: h, label: n.label }});
    g.setParent(n.id, '_phase_' + n.phase);
}});

GRAPH_DATA.edges.forEach(e => {{
    g.setEdge(e.from, e.to, {{ label: e.label || '', style: e.style || 'solid' }});
}});

dagre.layout(g);

// --- Render with D3 ---
const graphW = g.graph().width || 600;
const graphH = g.graph().height || 400;

const svg = d3.select('#diagram-svg')
    .attr('width', graphW)
    .attr('height', graphH);

// Arrowhead marker
svg.append('defs').append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '0 0 10 10')
    .attr('refX', 10).attr('refY', 5)
    .attr('markerWidth', 7).attr('markerHeight', 7)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', '#8b949e');

const root = svg.append('g');

// Phase backgrounds
PHASE_NAMES.forEach((name, i) => {{
    const pn = g.node('_phase_' + i);
    if (!pn) return;
    root.append('rect')
        .attr('class', 'phase-bg')
        .attr('data-phase', i)
        .attr('x', pn.x - pn.width / 2 - 12)
        .attr('y', pn.y - pn.height / 2 - 24)
        .attr('width', pn.width + 24)
        .attr('height', pn.height + 36)
        .attr('rx', 8)
        .style('opacity', 0);
    root.append('text')
        .attr('class', 'phase-label-text')
        .attr('data-phase', i)
        .attr('x', pn.x - pn.width / 2 - 4)
        .attr('y', pn.y - pn.height / 2 - 30)
        .text(name)
        .style('opacity', 0);
}});

// Edges
const edgesG = root.append('g');
const line = d3.line().x(p => p.x).y(p => p.y).curve(d3.curveBasis);

GRAPH_DATA.edges.forEach(e => {{
    const ed = g.edge(e.from, e.to);
    if (!ed || !ed.points) return;

    const eg = edgesG.append('g')
        .attr('class', 'graph-edge')
        .attr('data-from', e.from)
        .attr('data-to', e.to)
        .style('opacity', 0);

    const path = eg.append('path')
        .attr('d', line(ed.points))
        .attr('marker-end', 'url(#arrowhead)');

    if (e.style === 'dotted') path.attr('stroke-dasharray', '5 3');

    if (e.label) {{
        const mid = ed.points[Math.floor(ed.points.length / 2)];
        eg.append('text')
            .attr('x', mid.x)
            .attr('y', mid.y - 8)
            .attr('text-anchor', 'middle')
            .text(e.label);
    }}
}});

// Nodes
const nodesG = root.append('g');

function addLabel(group, label) {{
    const words = label.split(/\\s+/);
    const t = group.append('text').attr('text-anchor', 'middle');
    if (words.length <= 4) {{
        t.append('tspan').attr('x', 0).attr('dominant-baseline', 'central').text(label);
    }} else {{
        const mid = Math.ceil(words.length / 2);
        t.append('tspan').attr('x', 0).attr('dy', '-0.4em').text(words.slice(0, mid).join(' '));
        t.append('tspan').attr('x', 0).attr('dy', '1.2em').text(words.slice(mid).join(' '));
    }}
}}

GRAPH_DATA.nodes.forEach(n => {{
    const nd = g.node(n.id);
    if (!nd) return;

    const ng = nodesG.append('g')
        .attr('class', 'graph-node type-' + n.type)
        .attr('data-id', n.id)
        .attr('data-concept', n.id)
        .attr('data-phase', n.phase)
        .attr('data-step', n.step)
        .attr('transform', 'translate(' + nd.x + ',' + nd.y + ')')
        .style('opacity', 0);

    if (n.type === 'decision') {{
        const hw = (NODE_W + 30) / 2, hh = (NODE_H + 16) / 2;
        ng.append('polygon')
            .attr('points', '0,' + (-hh) + ' ' + hw + ',0 0,' + hh + ' ' + (-hw) + ',0');
    }} else {{
        ng.append('rect')
            .attr('x', -NODE_W / 2).attr('y', -NODE_H / 2)
            .attr('width', NODE_W).attr('height', NODE_H)
            .attr('rx', 6);
    }}

    addLabel(ng, n.label);

    // Click handler for popover
    ng.style('cursor', 'pointer');
    ng.on('click', (event) => {{
        event.stopPropagation();
        const popover = document.getElementById('node-popover');
        const container = document.querySelector('.diagram-container');

        // Build connection info
        const inEdges = GRAPH_DATA.edges.filter(e => e.to === n.id);
        const outEdges = GRAPH_DATA.edges.filter(e => e.from === n.id);

        let connectionsHtml = '';
        if (inEdges.length > 0) {{
            connectionsHtml += '<div style="margin-bottom:6px"><strong style="color:var(--text-dimmer);font-size:10px;text-transform:uppercase;letter-spacing:1px">Caused by</strong></div>';
            inEdges.forEach(e => {{
                const sourceNode = nodeById[e.from];
                const label = sourceNode ? sourceNode.label : e.from;
                connectionsHtml += '<div class="edge-in">' + label +
                    ' <span class="edge-verb">' + (e.label || '→') + '</span> this</div>';
            }});
        }}
        if (outEdges.length > 0) {{
            connectionsHtml += '<div style="margin-top:8px;margin-bottom:6px"><strong style="color:var(--text-dimmer);font-size:10px;text-transform:uppercase;letter-spacing:1px">Led to</strong></div>';
            outEdges.forEach(e => {{
                const targetNode = nodeById[e.to];
                const label = targetNode ? targetNode.label : e.to;
                connectionsHtml += '<div class="edge-out">this <span class="edge-verb">' +
                    (e.label || '→') + '</span> ' + label + '</div>';
            }});
        }}
        if (!inEdges.length && !outEdges.length) {{
            connectionsHtml = '<div style="color:var(--text-dimmer);font-style:italic">No connections</div>';
        }}

        document.getElementById('popover-title').textContent = n.label;
        const typeEl = document.getElementById('popover-type');
        typeEl.textContent = n.type;
        typeEl.className = 'node-popover-type type-' + n.type;
        document.getElementById('popover-phase').textContent =
            'Phase ' + (n.phase + 1) + ': ' + PHASE_NAMES[n.phase];
        document.getElementById('popover-connections').innerHTML = connectionsHtml;

        // Position near the clicked node
        const nodeScreenX = nd.x * zoomLevel - container.scrollLeft;
        const nodeScreenY = nd.y * zoomLevel - container.scrollTop;
        popover.style.left = Math.min(nodeScreenX + 20, container.clientWidth - 340) + 'px';
        popover.style.top = Math.max(10, nodeScreenY - 60) + 'px';
        popover.classList.add('visible');
    }});
}});

// Close popover
document.getElementById('popover-close').addEventListener('click', () => {{
    document.getElementById('node-popover').classList.remove('visible');
}});
document.querySelector('.diagram-container').addEventListener('click', (e) => {{
    if (!e.target.closest('.graph-node') && !e.target.closest('.node-popover')) {{
        document.getElementById('node-popover').classList.remove('visible');
    }}
}});

// --- Scroll-triggered reveal ---
const visibleNodes = new Set();
const nodeById = {{}};
GRAPH_DATA.nodes.forEach(n => {{ nodeById[n.id] = n; }});
let lastFlashedPhase = -1;

// Phase color map for flash effect
const PHASE_COLORS = {json.dumps([p.get('color', '#58a6ff') for p in narrative.get('phases', [])])};

function revealUpTo(phase, step) {{
    let stagger = 0;

    // Phase flash on new phase entry
    if (phase > lastFlashedPhase) {{
        lastFlashedPhase = phase;
        const flash = document.getElementById('phase-flash');
        if (flash) {{
            flash.style.background = PHASE_COLORS[phase] || '#58a6ff';
            flash.classList.remove('active');
            void flash.offsetWidth; // reflow to restart animation
            flash.classList.add('active');
        }}
    }}

    // Reveal phase backgrounds
    for (let p = 0; p <= phase; p++) {{
        root.selectAll('.phase-bg[data-phase="' + p + '"], .phase-label-text[data-phase="' + p + '"]')
            .transition().duration(400).style('opacity', p === phase ? 1 : 0.5);
    }}

    // Reveal nodes
    GRAPH_DATA.nodes.forEach(n => {{
        const shouldShow = n.phase < phase || (n.phase === phase && n.step <= step);
        if (shouldShow && !visibleNodes.has(n.id)) {{
            visibleNodes.add(n.id);
            const nodeEl = d3.select('.graph-node[data-id="' + n.id + '"]');
            nodeEl.transition()
                .delay(stagger * 80)
                .duration(400)
                .ease(d3.easeBackOut.overshoot(1.1))
                .style('opacity', 1)
                .on('end', () => {{
                    // Add breathing pulse class
                    nodeEl.classed('revealed', true);
                    // Particle burst from node position on screen
                    const nd = g.node(n.id);
                    if (nd && window._particleBurst) {{
                        const container = document.querySelector('.diagram-container');
                        const svgEl = document.getElementById('diagram-svg');
                        if (container && svgEl) {{
                            const rect = container.getBoundingClientRect();
                            const sx = rect.left + (nd.x * zoomLevel) - container.scrollLeft;
                            const sy = rect.top + (nd.y * zoomLevel) - container.scrollTop;
                            window._particleBurst(sx, sy, n.type);
                        }}
                    }}
                    // Sound
                    if (window._playNodeSound) window._playNodeSound(n.type);
                }});
            stagger++;
        }}
    }});

    // Reveal edges where both endpoints are visible
    GRAPH_DATA.edges.forEach(e => {{
        if (visibleNodes.has(e.from) && visibleNodes.has(e.to)) {{
            const sel = edgesG.select('.graph-edge[data-from="' + e.from + '"][data-to="' + e.to + '"]');
            if (sel.style('opacity') === '0') {{
                // Animate the edge drawing itself
                const path = sel.select('path');
                const len = path.node() ? path.node().getTotalLength() : 0;
                if (len > 0 && e.style !== 'dotted') {{
                    path.attr('stroke-dasharray', len + ' ' + len)
                        .attr('stroke-dashoffset', len);
                    sel.style('opacity', 1);
                    path.transition().duration(600).ease(d3.easeLinear)
                        .attr('stroke-dashoffset', 0);
                    sel.select('text').style('opacity', 0)
                        .transition().delay(400).duration(300).style('opacity', 1);
                    if (window._playEdgeSound) window._playEdgeSound();
                    // Edge-traveling particles
                    if (window._edgeParticles) {{
                        const container = document.querySelector('.diagram-container');
                        if (container) {{
                            window._edgeParticles(
                                path.node(), container.getBoundingClientRect(),
                                container.scrollLeft, container.scrollTop, zoomLevel);
                        }}
                    }}
                }} else {{
                    sel.transition().duration(500).style('opacity', 1);
                    if (window._playEdgeSound) window._playEdgeSound();
                }}
            }}
        }}
    }});

    // Auto-scroll diagram to latest revealed area (skip if user recently dragged)
    if (!userPannedRecently) {{
        const lastNode = [...visibleNodes].pop();
        if (lastNode) {{
            const nd = g.node(lastNode);
            if (nd) {{
                const container = document.querySelector('.diagram-container');
                const targetY = Math.max(0, nd.y * zoomLevel - container.clientHeight / 2);
                container.scrollTo({{ top: targetY, behavior: 'smooth' }});
            }}
        }}
    }}

    // Update phase indicator
    const indicator = document.getElementById('phase-indicator');
    if (indicator && PHASE_NAMES[phase]) {{
        indicator.textContent = 'Phase ' + (phase + 1) + ': ' + PHASE_NAMES[phase];
    }}
}}

// --- Zoom ---
let zoomLevel = 1;
const ZOOM_MIN = 0.5, ZOOM_MAX = 3, ZOOM_STEP = 0.25;

function applyZoom() {{
    const svgEl = document.getElementById('diagram-svg');
    svgEl.style.transform = 'scale(' + zoomLevel + ')';
    svgEl.style.transformOrigin = '0 0';
    document.getElementById('zoom-level').textContent = Math.round(zoomLevel * 100) + '%';
}}

document.getElementById('zoom-in').addEventListener('click', () => {{
    zoomLevel = Math.min(ZOOM_MAX, zoomLevel + ZOOM_STEP); applyZoom();
}});
document.getElementById('zoom-out').addEventListener('click', () => {{
    zoomLevel = Math.max(ZOOM_MIN, zoomLevel - ZOOM_STEP); applyZoom();
}});
document.getElementById('zoom-reset').addEventListener('click', () => {{
    zoomLevel = 1; applyZoom();
    document.querySelector('.diagram-container').scrollTo({{ top: 0, left: 0 }});
}});
// --- Fullscreen toggle ---
document.getElementById('fullscreen-toggle').addEventListener('click', () => {{
    const track = document.querySelector('.graph-track');
    const btn = document.getElementById('fullscreen-toggle');
    track.classList.toggle('fullscreen');
    btn.innerHTML = track.classList.contains('fullscreen') ? '&#x2716;' : '&#x26F6;';
    btn.title = track.classList.contains('fullscreen') ? 'Exit fullscreen' : 'Toggle fullscreen';
    // Re-fit graph after layout change
    setTimeout(() => {{ zoomLevel = 1; applyZoom(); }}, 50);
}});
document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') {{
        const track = document.querySelector('.graph-track');
        if (track.classList.contains('fullscreen')) {{
            track.classList.remove('fullscreen');
            document.getElementById('fullscreen-toggle').innerHTML = '&#x26F6;';
            document.getElementById('fullscreen-toggle').title = 'Toggle fullscreen';
        }}
    }}
}});

document.querySelector('.diagram-container').addEventListener('wheel', (e) => {{
    if (e.ctrlKey || e.metaKey) {{
        e.preventDefault();
        zoomLevel = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoomLevel - e.deltaY * 0.002));
        applyZoom();
    }}
}}, {{ passive: false }});

// --- Drag to pan ---
const dc = document.querySelector('.diagram-container');
let isDragging = false, dragStartX = 0, dragStartY = 0, scrollStartX = 0, scrollStartY = 0;
let userPannedRecently = false, panTimer = null;

dc.addEventListener('mousedown', (e) => {{
    // Only start drag on left button, not on zoom controls
    if (e.button !== 0 || e.target.closest('.zoom-controls')) return;
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    scrollStartX = dc.scrollLeft;
    scrollStartY = dc.scrollTop;
    dc.style.cursor = 'grabbing';
    e.preventDefault();
}});

window.addEventListener('mousemove', (e) => {{
    if (!isDragging) return;
    dc.scrollLeft = scrollStartX - (e.clientX - dragStartX);
    dc.scrollTop = scrollStartY - (e.clientY - dragStartY);
}});

window.addEventListener('mouseup', () => {{
    if (isDragging) {{
        isDragging = false;
        dc.style.cursor = 'grab';
        userPannedRecently = true;
        clearTimeout(panTimer);
        panTimer = setTimeout(() => {{ userPannedRecently = false; }}, 5000);
    }}
}});

// Prevent SVG native drag behavior
dc.addEventListener('dragstart', (e) => e.preventDefault());

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

// --- Reveal on scroll (narrative paragraphs) ---
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
            // Trigger graph reveal for this paragraph
            const phase = parseInt(e.target.dataset.phase);
            const step = parseInt(e.target.dataset.para);
            if (!isNaN(phase) && !isNaN(step)) {{
                revealUpTo(phase, step);
            }}
        }}
    }});
}}, {{ threshold: 0.15, rootMargin: '0px 0px -60px 0px' }});

document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// Also trigger on phase headers
const phaseObserver = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
        if (entry.isIntersecting) {{
            const phase = parseInt(entry.target.dataset.phase);
            revealUpTo(phase, 0);
        }}
    }});
}}, {{ threshold: 0.3 }});

document.querySelectorAll('.phase-header').forEach(el => phaseObserver.observe(el));

// Expose revealUpTo to the cinema script
window._revealUpTo = revealUpTo;
window._userPannedRecently = (v) => {{ userPannedRecently = v; }};

// --- Bidirectional hover: text chips ↔ graph nodes ---
document.addEventListener('mouseover', (e) => {{
    const chip = e.target.closest('.concept-chip[data-concept]');
    if (chip) {{
        const cid = chip.dataset.concept;
        // Highlight matching graph node
        document.querySelectorAll('.graph-node[data-concept="' + cid + '"]')
            .forEach(n => n.classList.add('hover-linked'));
        // Highlight all matching text chips
        document.querySelectorAll('.concept-chip[data-concept="' + cid + '"]')
            .forEach(c => c.classList.add('hover-linked'));
    }}
}});
document.addEventListener('mouseout', (e) => {{
    const chip = e.target.closest('.concept-chip[data-concept]');
    if (chip) {{
        document.querySelectorAll('.hover-linked')
            .forEach(el => el.classList.remove('hover-linked'));
    }}
}});
// Graph node hover → highlight text chips
document.querySelectorAll('.graph-node[data-concept]').forEach(node => {{
    node.addEventListener('mouseenter', () => {{
        const cid = node.dataset.concept;
        node.classList.add('hover-linked');
        document.querySelectorAll('.concept-chip[data-concept="' + cid + '"]')
            .forEach(c => c.classList.add('hover-linked'));
    }});
    node.addEventListener('mouseleave', () => {{
        document.querySelectorAll('.hover-linked')
            .forEach(el => el.classList.remove('hover-linked'));
    }});
}});
</script>

<!-- Cinema mode — separate script so it works even if D3/dagre CDN fails -->
<script>
const TIMELINE = {timeline_json};
const BASE_TYPEWRITER_SPEED = 20;
const SPEED_OPTIONS = [0.5, 1, 2, 4];

let cinemaPlaying = false;
let cinemaStep = -1;
let cinemaTimer = null;
let typewriterTimer = null;
let speedIdx = 1; // index into SPEED_OPTIONS
let playbackSpeed = 1;

const playBtn = document.getElementById('cinema-play');
const speedBtn = document.getElementById('cinema-speed');
const progressBar = document.getElementById('cinema-progress');
const dateDisplay = document.getElementById('cinema-date');
const timeDisplay = document.getElementById('cinema-time');

playBtn.addEventListener('click', function() {{
    if (cinemaPlaying) stopCinema();
    else startCinema();
}});

speedBtn.addEventListener('click', function() {{
    speedIdx = (speedIdx + 1) % SPEED_OPTIONS.length;
    playbackSpeed = SPEED_OPTIONS[speedIdx];
    speedBtn.textContent = playbackSpeed + 'x';
}});

function startCinema() {{
    cinemaPlaying = true;
    playBtn.textContent = '\u275A\u275A Pause';
    playBtn.classList.add('playing');
    if (window._userPannedRecently) window._userPannedRecently(false);
    if (window._startDrone) window._startDrone();
    if (cinemaStep >= TIMELINE.length - 1) cinemaStep = -1;
    advanceCinema();
}}

function stopCinema() {{
    cinemaPlaying = false;
    playBtn.textContent = '\u25B6 Play';
    playBtn.classList.remove('playing');
    clearTimeout(cinemaTimer);
    clearTimeout(typewriterTimer);
    if (window._stopDrone) window._stopDrone();
}}

function advanceCinema() {{
    if (!cinemaPlaying) return;
    cinemaStep++;

    if (cinemaStep === 0) {{
        window.scrollTo({{ top: window.innerHeight - 100, behavior: 'smooth' }});
        cinemaTimer = setTimeout(advanceCinema, 1500 / playbackSpeed);
        return;
    }}

    var idx = cinemaStep - 1;
    if (idx >= TIMELINE.length) {{
        var arc = document.querySelector('.arc-section');
        if (arc) arc.scrollIntoView({{ behavior: 'smooth' }});
        progressBar.style.width = '100%';
        stopCinema();
        return;
    }}

    var step = TIMELINE[idx];
    progressBar.style.width = (((idx + 1) / TIMELINE.length) * 100) + '%';
    dateDisplay.textContent = step.date_range;

    var paraEl = document.querySelector(
        '.reveal[data-phase="' + step.phase + '"][data-para="' + step.step + '"]'
    );

    if (step.step === 0) {{
        var header = document.querySelector('.phase-header[data-phase="' + step.phase + '"]');
        if (header) {{
            header.classList.add('visible');
            header.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}
        cinemaTimer = setTimeout(function() {{
            if (!cinemaPlaying) return;
            typewriteParagraph(paraEl, step);
        }}, 800 / playbackSpeed);
        return;
    }}

    typewriteParagraph(paraEl, step);
}}

function typewriteParagraph(paraEl, step) {{
    if (!paraEl || !cinemaPlaying) {{ advanceAfterPause(); return; }}

    paraEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

    if (window._revealUpTo) window._revealUpTo(step.phase, step.step);

    var fullHTML = paraEl.innerHTML;
    var fullText = paraEl.textContent;
    paraEl.classList.add('visible', 'typewriter');
    paraEl.innerHTML = '<span class="typewriter-cursor"></span>';

    var charIdx = 0;
    function typeChar() {{
        if (!cinemaPlaying) {{ paraEl.innerHTML = fullHTML; return; }}
        charIdx++;
        if (charIdx <= fullText.length) {{
            paraEl.innerHTML = fullText.slice(0, charIdx) + '<span class="typewriter-cursor"></span>';
            typewriterTimer = setTimeout(typeChar, BASE_TYPEWRITER_SPEED / playbackSpeed);
        }} else {{
            paraEl.innerHTML = fullHTML;
            paraEl.querySelectorAll('.concept-chip').forEach(function(chip, i) {{
                setTimeout(function() {{
                    chip.classList.add('glow');
                    setTimeout(function() {{ chip.classList.remove('glow'); }}, 1200);
                }}, i * 150);
            }});
            advanceAfterPause();
        }}
    }}
    setTimeout(typeChar, 400 / playbackSpeed);
}}

function advanceAfterPause() {{
    cinemaTimer = setTimeout(advanceCinema, 1200 / playbackSpeed);
}}

document.querySelector('.cinema-timeline').addEventListener('click', function(e) {{
    var rect = e.currentTarget.getBoundingClientRect();
    var pct = (e.clientX - rect.left) / rect.width;
    cinemaStep = Math.floor(pct * TIMELINE.length);
    if (!cinemaPlaying) startCinema();
    else advanceCinema();
}});

var cinemaStartTime = null;
setInterval(function() {{
    if (cinemaPlaying) {{
        if (!cinemaStartTime) cinemaStartTime = Date.now();
        var elapsed = Math.floor((Date.now() - cinemaStartTime) / 1000);
        var m = Math.floor(elapsed / 60);
        var s = elapsed % 60;
        timeDisplay.textContent = m + ':' + (s < 10 ? '0' : '') + s;
    }} else {{
        cinemaStartTime = null;
    }}
}}, 1000);

if (new URLSearchParams(window.location.search).has('autoplay')) {{
    setTimeout(startCinema, 2000);
}}
</script>
</body>
</html>"""
