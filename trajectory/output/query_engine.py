"""Query engine — NL question → SQL retrieval → LLM synthesis."""

import hashlib
import logging
import re
import sys
from pathlib import Path

_llm_client_path = str(Path.home() / "projects" / "llm_client")
if _llm_client_path not in sys.path:
    sys.path.insert(0, _llm_client_path)

from llm_client import call_llm, render_prompt

from trajectory.config import Config
from trajectory.db import TrajectoryDB
from trajectory.ingest import ingest_project
from trajectory.models import QueryResult

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "after", "before", "during", "and", "or", "but", "not",
    "if", "then", "than", "so", "no", "it", "its", "this", "that",
    "what", "which", "who", "how", "when", "where", "why", "all", "each",
    "every", "both", "few", "more", "most", "some", "any", "my", "your",
    "i", "me", "he", "she", "we", "they", "you",
})


def _extract_keywords(question: str) -> list[str]:
    """Tokenize question into search keywords, removing stop words and punctuation."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", question.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def query_trajectory(
    question: str,
    db: TrajectoryDB,
    config: Config,
) -> QueryResult:
    """Answer a natural language question about project evolution."""
    keywords = _extract_keywords(question)
    logger.info("Query keywords: %s", keywords)

    # Search for matching concepts
    concepts = db.search_concepts(keywords) if keywords else []
    logger.info("Found %d matching concepts", len(concepts))

    # Get events for matching concepts
    events: list[dict[str, object]] = []
    if concepts:
        events = db.get_events_for_concepts(
            [c.id for c in concepts], limit=100
        )
    else:
        # Fallback: LIKE search on event titles
        if keywords:
            clauses = ["LOWER(title) LIKE ?"] * len(keywords)
            params = [f"%{kw.lower()}%" for kw in keywords]
            rows = db.conn.execute(
                f"""SELECT * FROM events
                    WHERE {' OR '.join(clauses)}
                    ORDER BY timestamp DESC LIMIT 50""",
                params,
            ).fetchall()
            events = [dict(r) for r in rows]

    # Collect project names from events
    project_ids = {e.get("project_id") for e in events if e.get("project_id")}
    project_names: list[str] = []
    for pid in project_ids:
        proj = db.get_project(int(pid))  # type: ignore[arg-type]
        if proj:
            project_names.append(proj.name)

    # Build data gaps
    total_events = db.conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
    analyzed_events = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM events WHERE analysis_run_id IS NOT NULL"
    ).fetchone()
    data_gaps: list[str] = []
    total = total_events["cnt"] if total_events else 0
    analyzed = analyzed_events["cnt"] if analyzed_events else 0
    if analyzed < total:
        data_gaps.append(
            f"Only {analyzed} of {total} events have been analyzed by LLM"
        )
    if not concepts and keywords:
        data_gaps.append(
            f"No concepts matched keywords: {', '.join(keywords)}"
        )

    # Add project_name to event dicts for the prompt
    project_cache: dict[int, str] = {}
    for e in events:
        pid = e.get("project_id")
        if pid and pid not in project_cache:
            proj = db.get_project(int(pid))  # type: ignore[arg-type]
            project_cache[int(pid)] = proj.name if proj else "unknown"  # type: ignore[arg-type]
        if pid:
            e["project_name"] = project_cache.get(int(pid), "unknown")  # type: ignore[arg-type]

    # Render prompt and call LLM
    messages = render_prompt(
        PROMPTS_DIR / "query_synthesis.yaml",
        question=question,
        concepts=[
            {
                "name": c.name,
                "status": c.status,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "description": c.description,
            }
            for c in concepts
        ],
        events=events,
        data_gaps=data_gaps,
    )

    question_hash = hashlib.md5(question.encode()).hexdigest()[:12]
    result = call_llm(
        config.llm.quality_model,
        messages,
        task="trajectory.query",
        trace_id=f"trajectory.query.{question_hash}",
        max_budget=0,
    )

    return QueryResult(
        answer=result.content,
        concepts_found=[c.name for c in concepts],
        events_used=len(events),
        projects_involved=sorted(set(project_names)),
        data_gaps=data_gaps,
    )


def get_timeline(
    project_name: str,
    db: TrajectoryDB,
    since: str | None = None,
    until: str | None = None,
    min_significance: float | None = None,
) -> list[dict[str, object]]:
    """Get chronological events for a project."""
    project = db.get_project_by_name(project_name)
    if not project:
        raise ValueError(f"Project not found: {project_name}")

    events = db.get_timeline(
        project.id,
        since=since,
        until=until,
        min_significance=min_significance,
    )
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "timestamp": e.timestamp,
            "title": e.title,
            "author": e.author,
            "significance": e.significance,
            "llm_summary": e.llm_summary,
            "llm_intent": e.llm_intent,
        }
        for e in events
    ]


def get_concept_history(
    concept_name: str,
    db: TrajectoryDB,
) -> dict[str, object]:
    """Get a concept's full history grouped by project."""
    concept = db.get_concept_by_name(concept_name)
    if not concept:
        raise ValueError(f"Concept not found: {concept_name}")

    events = db.get_concept_events(concept.id)

    # Group by project
    by_project: dict[str, list[dict[str, object]]] = {}
    for e in events:
        pid = e.get("project_id")
        if pid:
            proj = db.get_project(int(pid))  # type: ignore[arg-type]
            pname = proj.name if proj else "unknown"
        else:
            pname = "unknown"
        by_project.setdefault(pname, []).append(e)

    return {
        "concept": {
            "name": concept.name,
            "status": concept.status,
            "description": concept.description,
            "first_seen": concept.first_seen,
            "last_seen": concept.last_seen,
        },
        "timeline": events,
        "projects": list(by_project.keys()),
    }


def list_tracked_projects(db: TrajectoryDB) -> list[dict[str, object]]:
    """List all tracked projects with summary stats."""
    projects = db.list_projects()
    result: list[dict[str, object]] = []
    for p in projects:
        total = db.count_events(p.id)
        analyzed = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE project_id = ? AND analysis_run_id IS NOT NULL",
            (p.id,),
        ).fetchone()
        concept_count = db.conn.execute(
            """SELECT COUNT(DISTINCT ce.concept_id) as cnt
               FROM concept_events ce
               JOIN events e ON ce.event_id = e.id
               WHERE e.project_id = ?""",
            (p.id,),
        ).fetchone()
        result.append({
            "name": p.name,
            "path": p.path,
            "total_events": total,
            "analyzed_events": analyzed["cnt"] if analyzed else 0,
            "concepts": concept_count["cnt"] if concept_count else 0,
            "last_ingested": p.last_ingested,
        })
    return result


def list_concepts(
    db: TrajectoryDB,
    status: str | None = None,
    project: str | None = None,
    level: str | None = None,
) -> list[dict[str, object]]:
    """List concepts with optional filters and event counts."""
    project_id: int | None = None
    if project:
        p = db.get_project_by_name(project)
        if not p:
            raise ValueError(f"Project not found: {project}")
        project_id = p.id

    concepts = db.list_concepts(status=status, project_id=project_id, level=level)
    result: list[dict[str, object]] = []
    for c in concepts:
        event_count = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM concept_events WHERE concept_id = ?",
            (c.id,),
        ).fetchone()
        result.append({
            "name": c.name,
            "level": c.level,
            "status": c.status,
            "description": c.description,
            "first_seen": c.first_seen,
            "last_seen": c.last_seen,
            "event_count": event_count["cnt"] if event_count else 0,
        })
    return result


def correct_concept(
    concept_name: str,
    action: str,
    db: TrajectoryDB,
    new_name: str | None = None,
    merge_into: str | None = None,
    new_status: str | None = None,
) -> dict[str, str]:
    """Apply a correction to a concept (rename, merge, or status change)."""
    concept = db.get_concept_by_name(concept_name)
    if not concept:
        raise ValueError(f"Concept not found: {concept_name}")

    if action == "rename":
        if not new_name:
            raise ValueError("new_name required for rename action")
        old_name = concept.name
        db.rename_concept(concept.id, new_name)
        db.insert_correction(
            correction_type="rename",
            target_type="concept",
            target_id=concept.id,
            old_value=old_name,
            new_value=new_name,
            source_command=f"correct_concept({concept_name!r}, 'rename', new_name={new_name!r})",
        )
        return {"status": "ok", "message": f"Renamed '{old_name}' to '{new_name}'"}

    elif action == "merge":
        if not merge_into:
            raise ValueError("merge_into required for merge action")
        target = db.get_concept_by_name(merge_into)
        if not target:
            raise ValueError(f"Merge target concept not found: {merge_into}")
        moved = db.merge_concepts(concept.id, target.id)
        db.insert_correction(
            correction_type="merge",
            target_type="concept",
            target_id=concept.id,
            old_value=concept.name,
            new_value=target.name,
            source_command=f"correct_concept({concept_name!r}, 'merge', merge_into={merge_into!r})",
        )
        return {"status": "ok", "message": f"Merged '{concept.name}' into '{target.name}' ({moved} events moved)"}

    elif action == "status_change":
        if not new_status:
            raise ValueError("new_status required for status_change action")
        old_status = concept.status
        db.update_concept_status(concept.id, new_status)
        db.insert_correction(
            correction_type="status_change",
            target_type="concept",
            target_id=concept.id,
            old_value=old_status,
            new_value=new_status,
            source_command=f"correct_concept({concept_name!r}, 'status_change', new_status={new_status!r})",
        )
        return {"status": "ok", "message": f"Changed '{concept.name}' status from '{old_status}' to '{new_status}'"}

    else:
        raise ValueError(f"Invalid action: {action}. Must be 'rename', 'merge', or 'status_change'")


def ingest_project_from_path(
    project_path: str,
    db: TrajectoryDB,
    config: Config,
) -> dict[str, object]:
    """Ingest a project from a filesystem path."""
    path = Path(project_path)
    if not path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")

    result = ingest_project(path, db, config)
    return {
        "project": path.name,
        "total_extracted": result.total_extracted,
        "total_new": result.total_new,
    }
