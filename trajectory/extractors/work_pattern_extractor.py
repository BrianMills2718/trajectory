"""Deterministic work pattern extractor — materializes session stats from event raw_data.

No LLM calls. Parses raw_data JSON from conversation events to count tools, tokens, timing.
Populates the work_patterns table.
"""

import json
import logging
from datetime import datetime

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)

# Core Claude Code tool names (normalized from various formats)
TOOL_CATEGORIES: dict[str, str] = {
    "read": "read",
    "write": "write",
    "edit": "edit",
    "bash": "bash",
    "glob": "glob",
    "grep": "grep",
    "task": "task",
    "notebookedit": "edit",
    "webfetch": "bash",
    "websearch": "bash",
    "askuserquestion": "read",
    "taskoutput": "task",
    "taskstop": "task",
    "taskcreate": "task",
    "taskupdate": "task",
    "tasklist": "task",
    "enterplanmode": "read",
    "exitplanmode": "write",
    "enterworktree": "bash",
    "skill": "bash",
}


def _categorize_tool(tool_name: str) -> str | None:
    """Map a tool name to one of the tracked categories. Returns None if unrecognized."""
    lower = tool_name.lower()

    # Direct match
    if lower in TOOL_CATEGORIES:
        return TOOL_CATEGORIES[lower]

    # MCP tools count as bash (external calls)
    if lower.startswith("mcp__"):
        return "bash"

    return None


class WorkPatternResult:
    """Summary of work pattern extraction for a project."""

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        self.events_processed = 0
        self.patterns_created = 0
        self.total_messages = 0
        self.total_tokens = 0

    def __repr__(self) -> str:
        return (
            f"WorkPatternResult({self.project_name}: {self.patterns_created} patterns from "
            f"{self.events_processed} events, {self.total_messages} msgs, {self.total_tokens} tokens)"
        )


def extract_work_patterns(
    db: TrajectoryDB,
    project_id: int,
) -> WorkPatternResult:
    """Extract work patterns from conversation events for a project.

    Materializes: message counts, token counts, tool usage counts,
    file counts, hour of day, day of week, model.

    Idempotent: uses upsert on event_id.
    """
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"Project ID {project_id} not found")

    result = WorkPatternResult(project.name)

    # Get all conversation events with raw_data
    rows = db.conn.execute(
        "SELECT id, project_id, timestamp, raw_data, files_changed FROM events "
        "WHERE project_id = ? AND event_type = 'conversation' AND raw_data IS NOT NULL",
        (project_id,),
    ).fetchall()

    for row in rows:
        result.events_processed += 1

        try:
            rd = json.loads(row["raw_data"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Parse timestamp for hour/day
        hour_of_day = None
        day_of_week = None
        try:
            ts = datetime.fromisoformat(row["timestamp"])
            hour_of_day = ts.hour
            day_of_week = ts.weekday()  # 0=Monday
        except (ValueError, TypeError):
            pass

        # Message counts
        message_count = rd.get("message_count", 0) or 0
        user_message_count = rd.get("user_message_count", 0) or 0

        # Token counts
        total_input_tokens = rd.get("total_input_tokens", 0) or 0
        total_output_tokens = rd.get("total_output_tokens", 0) or 0

        # Tool usage — count by category from tools_used list
        tools_used = rd.get("tools_used", [])
        tool_counts: dict[str, int] = {
            "read": 0, "write": 0, "edit": 0, "bash": 0,
            "glob": 0, "grep": 0, "task": 0,
        }
        for tool_name in tools_used:
            cat = _categorize_tool(tool_name)
            if cat and cat in tool_counts:
                tool_counts[cat] += 1

        # If we have tool_sequence (enriched data), count actual occurrences
        tool_sequence = rd.get("tool_sequence", [])
        if tool_sequence:
            # Reset and count from actual sequence
            tool_counts = {k: 0 for k in tool_counts}
            for entry in tool_sequence:
                tool_name = entry[0] if isinstance(entry, (list, tuple)) else entry
                cat = _categorize_tool(str(tool_name))
                if cat and cat in tool_counts:
                    tool_counts[cat] += 1

        # File counts
        files_modified = rd.get("files_modified", [])
        files_examined = rd.get("files_examined", [])
        files_modified_count = len(files_modified) if files_modified else 0
        files_examined_count = len(files_examined) if files_examined else 0

        # If no enriched data, estimate from files_changed
        if not files_modified and row["files_changed"]:
            try:
                fc = json.loads(row["files_changed"])
                files_modified_count = len(fc) if fc else 0
            except (json.JSONDecodeError, TypeError):
                pass

        # Model
        model = rd.get("model")

        # Duration — not available in current data, leave as None
        duration_minutes = None

        db.upsert_work_pattern(
            event_id=row["id"],
            project_id=project_id,
            duration_minutes=duration_minutes,
            message_count=message_count,
            user_message_count=user_message_count,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            tool_read_count=tool_counts["read"],
            tool_write_count=tool_counts["write"],
            tool_edit_count=tool_counts["edit"],
            tool_bash_count=tool_counts["bash"],
            tool_glob_count=tool_counts["glob"],
            tool_grep_count=tool_counts["grep"],
            tool_task_count=tool_counts["task"],
            files_examined_count=files_examined_count,
            files_modified_count=files_modified_count,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            model=model,
        )
        result.patterns_created += 1
        result.total_messages += message_count
        result.total_tokens += total_input_tokens + total_output_tokens

    db.conn.commit()
    logger.info(
        "Extracted %d work patterns for %s",
        result.patterns_created, project.name,
    )
    return result


def extract_all_work_patterns(db: TrajectoryDB) -> list[WorkPatternResult]:
    """Extract work patterns for all projects."""
    projects = db.list_projects()
    results: list[WorkPatternResult] = []

    for project in projects:
        try:
            result = extract_work_patterns(db, project.id)
            if result.patterns_created > 0:
                results.append(result)
        except Exception:
            logger.exception("Failed to extract work patterns for %s", project.name)

    return results
