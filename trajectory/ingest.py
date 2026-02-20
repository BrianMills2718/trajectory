"""Ingestion orchestrator — runs all extractors for a project and stores results."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trajectory.config import Config
from trajectory.db import TrajectoryDB
from trajectory.extractors.claude_log_extractor import ClaudeLogExtractor
from trajectory.extractors.doc_extractor import DocExtractor
from trajectory.extractors.git_extractor import GitExtractor
from trajectory.extractors.session_builder import build_sessions
from trajectory.models import EventInsert

logger = logging.getLogger(__name__)


class IngestionResult:
    """Summary of an ingestion run."""

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        self.commits_extracted = 0
        self.commits_new = 0
        self.conversations_extracted = 0
        self.conversations_new = 0
        self.docs_extracted = 0
        self.docs_new = 0
        self.events_backfilled = 0
        self.sessions_created = 0

    @property
    def total_extracted(self) -> int:
        return self.commits_extracted + self.conversations_extracted + self.docs_extracted

    @property
    def total_new(self) -> int:
        return self.commits_new + self.conversations_new + self.docs_new

    def __repr__(self) -> str:
        parts = [
            f"IngestionResult({self.project_name}: ",
            f"{self.total_new} new / {self.total_extracted} extracted ",
            f"[commits={self.commits_new}/{self.commits_extracted}, ",
            f"conversations={self.conversations_new}/{self.conversations_extracted}, ",
            f"docs={self.docs_new}/{self.docs_extracted}]",
        ]
        if self.events_backfilled:
            parts.append(f", backfilled={self.events_backfilled}")
        if self.sessions_created:
            parts.append(f", sessions={self.sessions_created}")
        parts.append(")")
        return "".join(parts)


def ingest_project(
    project_path: Path,
    db: TrajectoryDB,
    config: Config,
    git_remote: str | None = None,
    description: str | None = None,
    backfill: bool = False,
) -> IngestionResult:
    """Ingest all events from a single project.

    Runs git, Claude log, and doc extractors. Deduplicates by source_id.
    Updates project stats and last_ingested timestamp.

    If backfill=True, re-extracts and updates enrichment columns (diff_summary,
    change_types) on existing events.
    """
    project_path = project_path.resolve()
    project_name = project_path.name
    result = IngestionResult(project_name)

    logger.info("Starting ingestion for %s at %s", project_name, project_path)

    # Upsert project
    project_id = db.upsert_project(
        name=project_name,
        path=str(project_path),
        git_remote=git_remote,
        description=description,
    )

    # Get last_ingested for incremental processing
    project = db.get_project(project_id)
    since = project.last_ingested if project else None

    # For backfill, extract everything regardless of last_ingested
    extract_since = None if backfill else since

    # --- Git commits ---
    git_extractor = GitExtractor(
        project_path,
        max_diff_lines=config.extraction.max_diff_lines,
    )
    git_events = git_extractor.extract(since=extract_since)
    result.commits_extracted = len(git_events)

    if backfill:
        # Update enrichment columns on existing events + insert new ones
        for e in git_events:
            inserted = db.insert_event(db.extracted_to_insert(e, project_id))
            if inserted is not None:
                result.commits_new += 1
            elif e.diff_summary or e.change_types:
                # Event exists — backfill enrichment columns
                updated = db.backfill_event(
                    source_id=e.source_id,
                    diff_summary=e.diff_summary,
                    change_types=json.dumps(e.change_types) if e.change_types else None,
                )
                if updated:
                    result.events_backfilled += 1
        db.conn.commit()
    else:
        git_inserts = [db.extracted_to_insert(e, project_id) for e in git_events]
        result.commits_new = db.insert_events_batch(git_inserts)

    # --- Claude conversations ---
    claude_extractor = ClaudeLogExtractor(
        project_path,
        claude_logs_dir=config.resolved_claude_logs_dir,
        max_reasoning_chars=config.extraction.max_reasoning_chars,
    )
    claude_events = claude_extractor.extract(since=extract_since)
    result.conversations_extracted = len(claude_events)

    claude_inserts = [db.extracted_to_insert(e, project_id) for e in claude_events]
    result.conversations_new = db.insert_events_batch(claude_inserts)

    # --- Documentation ---
    doc_extractor = DocExtractor(project_path, config=config.extraction)
    doc_events = doc_extractor.extract(since=extract_since)
    result.docs_extracted = len(doc_events)

    doc_inserts = [db.extracted_to_insert(e, project_id) for e in doc_events]
    result.docs_new = db.insert_events_batch(doc_inserts)

    # --- Build sessions ---
    session_result = build_sessions(db, project_id=project_id)
    result.sessions_created = session_result.sessions_created

    # Update project stats
    total_commits = db.count_events(project_id, "commit")
    total_conversations = db.count_events(project_id, "conversation")
    db.update_project_stats(
        project_id,
        total_commits=total_commits,
        total_conversations=total_conversations,
    )

    # Update last_ingested
    now = datetime.now(timezone.utc).isoformat()
    db.update_last_ingested(project_id, now)

    logger.info("Ingestion complete: %s", result)
    return result


def ingest_all_projects(
    db: TrajectoryDB,
    config: Config,
    backfill: bool = False,
) -> list[IngestionResult]:
    """Ingest all git repositories found under projects_dir."""
    projects_dir = config.resolved_projects_dir
    results: list[IngestionResult] = []

    if not projects_dir.is_dir():
        logger.error("Projects directory not found: %s", projects_dir)
        return results

    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / ".git").exists():
            continue
        # Skip hidden directories
        if entry.name.startswith("."):
            continue

        try:
            result = ingest_project(entry, db, config, backfill=backfill)
            results.append(result)
        except Exception:
            logger.exception("Failed to ingest %s", entry.name)

    logger.info(
        "Ingested %d projects. Total new events: %d",
        len(results),
        sum(r.total_new for r in results),
    )
    return results
