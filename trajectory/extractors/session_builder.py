"""Build work sessions by linking conversations to the commits they produced."""

import json
import logging
from collections import defaultdict
from datetime import datetime

from trajectory.db import TrajectoryDB
from trajectory.models import EventRow

logger = logging.getLogger(__name__)


class SessionBuildResult:
    """Summary of session building."""

    def __init__(self) -> None:
        self.sessions_created: int = 0
        self.conversation_sessions: int = 0
        self.orphan_sessions: int = 0
        self.commits_linked: int = 0
        self.events_linked: int = 0

    def __repr__(self) -> str:
        return (
            f"SessionBuildResult({self.sessions_created} sessions: "
            f"{self.conversation_sessions} from conversations, "
            f"{self.orphan_sessions} orphan groups, "
            f"{self.commits_linked} commits linked)"
        )


def build_sessions(
    db: TrajectoryDB,
    project_id: int | None = None,
) -> SessionBuildResult:
    """Build work sessions for all (or one) project.

    For each conversation event, extract commit_hashes from raw_data,
    match to commit events via source_id prefix, and create work_session rows.
    Orphan commits (no conversation) are grouped by day.
    """
    result = SessionBuildResult()

    if project_id is not None:
        project_ids = [project_id]
    else:
        projects = db.list_projects()
        project_ids = [p.id for p in projects]

    for pid in project_ids:
        _build_sessions_for_project(db, pid, result)

    db.conn.commit()
    logger.info("Session build complete: %s", result)
    return result


def _build_sessions_for_project(
    db: TrajectoryDB,
    project_id: int,
    result: SessionBuildResult,
) -> None:
    """Build sessions for a single project."""
    # Get all conversation events
    conversations = db.get_events(
        project_id=project_id, event_type="conversation", limit=10000,
    )

    # Get all commit events
    commits = db.get_events(
        project_id=project_id, event_type="commit", limit=50000,
    )

    # Build hash→event lookup (full hash and short hash prefix)
    hash_to_commit: dict[str, EventRow] = {}
    for c in commits:
        # source_id is "git:<full_hash>"
        full_hash = c.source_id.removeprefix("git:")
        hash_to_commit[full_hash] = c
        # Also index 7-char prefix for short hash matching
        if len(full_hash) >= 7:
            hash_to_commit[full_hash[:7]] = c

    # Track which commit event IDs have been claimed by a session
    claimed_commit_ids: set[int] = set()

    # 1. Create sessions from conversations
    for conv in conversations:
        raw = _parse_raw_data(conv.raw_data)
        commit_hashes = raw.get("commit_hashes", [])

        # Match short hashes to full commit events
        matched_commits: list[EventRow] = []
        for h in commit_hashes:
            matched = hash_to_commit.get(h)
            if matched and matched.id not in claimed_commit_ids:
                matched_commits.append(matched)
                claimed_commit_ids.add(matched.id)

        # Determine session time range
        all_timestamps = [conv.timestamp]
        for mc in matched_commits:
            all_timestamps.append(mc.timestamp)
        session_start = min(all_timestamps)
        session_end = max(all_timestamps)

        # Build aggregated diff summary from commits
        diff_parts: list[str] = []
        for mc in matched_commits:
            if mc.diff_summary:
                header = f"[{mc.source_id.removeprefix('git:')[:8]}] {mc.title}"
                diff_parts.append(f"{header}\n{mc.diff_summary}")

        session_id = db.insert_work_session(
            project_id=project_id,
            conversation_event_id=conv.id,
            session_start=session_start,
            session_end=session_end,
            user_goal=conv.title,  # first user message
            tool_sequence=json.dumps(raw.get("tool_sequence")) if raw.get("tool_sequence") else None,
            files_modified=json.dumps(raw.get("files_modified")) if raw.get("files_modified") else None,
            commit_hashes=json.dumps([mc.source_id.removeprefix("git:") for mc in matched_commits]) if matched_commits else None,
            assistant_reasoning=raw.get("assistant_reasoning"),
            diff_summary="\n\n".join(diff_parts) if diff_parts else None,
        )

        # Link events
        db.insert_session_event(session_id, conv.id, "conversation")
        db.set_event_session(conv.id, session_id)
        result.events_linked += 1

        for mc in matched_commits:
            db.insert_session_event(session_id, mc.id, "commit")
            db.set_event_session(mc.id, session_id)
            result.commits_linked += 1
            result.events_linked += 1

        result.sessions_created += 1
        result.conversation_sessions += 1

    # 2. Group orphan commits by day into implicit sessions
    orphan_commits = [c for c in commits if c.id not in claimed_commit_ids]
    if orphan_commits:
        by_day: dict[str, list[EventRow]] = defaultdict(list)
        for c in orphan_commits:
            day = c.timestamp[:10]  # YYYY-MM-DD
            by_day[day].append(c)

        for day, day_commits in sorted(by_day.items()):
            timestamps = [c.timestamp for c in day_commits]
            session_start = min(timestamps)
            session_end = max(timestamps)

            diff_parts = []
            for dc in day_commits:
                if dc.diff_summary:
                    header = f"[{dc.source_id.removeprefix('git:')[:8]}] {dc.title}"
                    diff_parts.append(f"{header}\n{dc.diff_summary}")

            session_id = db.insert_work_session(
                project_id=project_id,
                session_start=session_start,
                session_end=session_end,
                commit_hashes=json.dumps([c.source_id.removeprefix("git:") for c in day_commits]),
                diff_summary="\n\n".join(diff_parts) if diff_parts else None,
            )

            for dc in day_commits:
                db.insert_session_event(session_id, dc.id, "commit")
                db.set_event_session(dc.id, session_id)
                claimed_commit_ids.add(dc.id)
                result.commits_linked += 1
                result.events_linked += 1

            result.sessions_created += 1
            result.orphan_sessions += 1


def _parse_raw_data(raw_data: str | None) -> dict:
    """Safely parse raw_data JSON string."""
    if not raw_data:
        return {}
    try:
        return json.loads(raw_data)
    except (json.JSONDecodeError, TypeError):
        return {}
