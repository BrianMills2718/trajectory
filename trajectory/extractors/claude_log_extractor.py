"""Extract events from Claude Code conversation logs (JSONL)."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from trajectory.extractors.base import BaseExtractor
from trajectory.models import EventType, ExtractedEvent

logger = logging.getLogger(__name__)


def project_path_to_log_key(project_path: Path) -> str:
    """Convert a project path to Claude Code's log directory key.

    /home/brian/sam_gov -> -home-brian-sam-gov
    Claude Code replaces / with - and _ with -, keeping the leading -.
    """
    return str(project_path).replace("/", "-").replace("_", "-")


def find_log_dir(project_path: Path, claude_logs_dir: Path) -> Path | None:
    """Find the Claude Code log directory for a project.

    Tries exact key match first, then falls back to scanning all log dirs
    for ones ending with the project name (handles path aliases/symlinks).
    """
    # Try exact match
    log_key = project_path_to_log_key(project_path)
    exact = claude_logs_dir / log_key
    if exact.exists():
        return exact

    # Fallback: find dirs ending with the project name (hyphenated)
    project_suffix = project_path.name.replace("_", "-")
    matches: list[Path] = []
    if claude_logs_dir.exists():
        for d in claude_logs_dir.iterdir():
            if d.is_dir() and d.name.endswith(f"-{project_suffix}"):
                matches.append(d)

    if len(matches) == 1:
        logger.info("Matched log dir by suffix: %s -> %s", project_path.name, matches[0].name)
        return matches[0]
    if len(matches) > 1:
        # Pick the one with the most JSONL files (most active)
        best = max(matches, key=lambda d: len(list(d.glob("*.jsonl"))))
        logger.info(
            "Multiple log dir matches for %s, using %s (%d candidates)",
            project_path.name, best.name, len(matches),
        )
        return best

    return None


class ClaudeLogExtractor(BaseExtractor):
    """Extract conversation events from Claude Code JSONL logs."""

    def __init__(self, project_path: Path, claude_logs_dir: Path | None = None) -> None:
        super().__init__(project_path)
        self.claude_logs_dir = claude_logs_dir or Path.home() / ".claude" / "projects"

    def extract(self, since: str | None = None) -> list[ExtractedEvent]:
        log_dir = find_log_dir(self.project_path, self.claude_logs_dir)

        if log_dir is None:
            logger.warning("No Claude logs found for %s", self.project_path)
            return []

        since_dt = datetime.fromisoformat(since) if since else None
        if since_dt and since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)

        events: list[ExtractedEvent] = []
        jsonl_files = sorted(log_dir.glob("*.jsonl"))

        for jsonl_path in jsonl_files:
            session_id = jsonl_path.stem
            event = self._parse_session(jsonl_path, session_id, since_dt)
            if event:
                events.append(event)

        logger.info(
            "Extracted %d conversations from %s (%d log files)",
            len(events), self.project_name, len(jsonl_files),
        )
        return events

    def _parse_session(
        self,
        jsonl_path: Path,
        session_id: str,
        since_dt: datetime | None,
    ) -> ExtractedEvent | None:
        """Parse a single JSONL session file into a conversation event."""
        messages = self._read_jsonl(jsonl_path)
        if not messages:
            return None

        # Find first user message for title and timestamp
        first_user = None
        session_timestamp = None
        user_messages: list[str] = []
        tools_used: set[str] = set()
        files_touched: set[str] = set()
        total_input_tokens = 0
        total_output_tokens = 0
        model_used = None
        git_branch = None
        message_count = 0

        for msg in messages:
            msg_type = msg.get("type")
            timestamp_str = msg.get("timestamp")

            if msg_type == "user":
                message_count += 1
                content = msg.get("message", {}).get("content", "")
                if isinstance(content, str) and content.strip():
                    user_messages.append(content.strip())
                    if first_user is None:
                        first_user = content.strip()
                        session_timestamp = timestamp_str
                if not git_branch:
                    git_branch = msg.get("gitBranch")

            elif msg_type == "assistant":
                message_count += 1
                inner = msg.get("message", {})
                if not model_used:
                    model_used = inner.get("model")
                usage = inner.get("usage", {})
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)

                for block in inner.get("content", []):
                    if block.get("type") == "tool_use":
                        tools_used.add(block.get("name", "unknown"))

        if not first_user or not session_timestamp:
            return None

        ts = datetime.fromisoformat(session_timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if since_dt and ts < since_dt:
            return None

        # Truncate title to first 200 chars
        title = first_user[:200]
        if len(first_user) > 200:
            title += "..."

        # Build body from all user messages
        body = "\n\n---\n\n".join(user_messages[:20])  # cap at 20 messages

        raw_data = {
            "session_id": session_id,
            "model": model_used,
            "message_count": message_count,
            "user_message_count": len(user_messages),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tools_used": sorted(tools_used),
        }

        return ExtractedEvent(
            event_type=EventType.CONVERSATION,
            source_id=f"claude:{session_id}",
            timestamp=ts,
            author="brian",
            title=title,
            body=body,
            raw_data=raw_data,
            files_changed=sorted(files_touched) if files_touched else None,
            git_branch=git_branch,
        )

    def _read_jsonl(self, path: Path) -> list[dict]:
        """Read JSONL file, skipping malformed lines."""
        messages: list[dict] = []
        try:
            with path.open() as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.debug(
                            "Skipping malformed JSON at %s:%d", path.name, line_num
                        )
        except Exception:
            logger.exception("Error reading JSONL file %s", path)
            raise
        return messages
