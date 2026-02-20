"""Extract events from Claude Code conversation logs (JSONL)."""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from trajectory.extractors.base import BaseExtractor
from trajectory.models import EventType, ExtractedEvent

logger = logging.getLogger(__name__)

# Matches git commit output like "[main abc1234] Fix bug"
COMMIT_HASH_RE = re.compile(r"\[[\w./-]+ ([a-f0-9]{7,})\]")


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

    def __init__(
        self,
        project_path: Path,
        claude_logs_dir: Path | None = None,
        max_reasoning_chars: int = 3000,
    ) -> None:
        super().__init__(project_path)
        self.claude_logs_dir = claude_logs_dir or Path.home() / ".claude" / "projects"
        self.max_reasoning_chars = max_reasoning_chars

    def extract(self, since: str | None = None) -> list[ExtractedEvent]:
        since_dt = datetime.fromisoformat(since) if since else None
        if since_dt and since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)

        events: list[ExtractedEvent] = []
        seen_session_ids: set[str] = set()

        # 1. Project-specific log directory (exact match or suffix match)
        log_dir = find_log_dir(self.project_path, self.claude_logs_dir)
        if log_dir is not None:
            for jsonl_path in sorted(log_dir.glob("*.jsonl")):
                session_id = jsonl_path.stem
                seen_session_ids.add(session_id)
                event = self._parse_session(jsonl_path, session_id, since_dt)
                if event:
                    events.append(event)
            logger.info(
                "Found %d conversations in project dir for %s",
                len(events), self.project_name,
            )

        # 2. Catch-all parent directory (sessions launched from parent dir)
        catchall_files = self._find_catchall_sessions()
        catchall_count = 0
        for jsonl_path in catchall_files:
            session_id = jsonl_path.stem
            if session_id in seen_session_ids:
                continue  # already found in project-specific dir
            seen_session_ids.add(session_id)
            event = self._parse_session(jsonl_path, session_id, since_dt)
            if event:
                events.append(event)
                catchall_count += 1

        if catchall_count > 0:
            logger.info(
                "Found %d additional conversations in catch-all dir for %s",
                catchall_count, self.project_name,
            )

        if not events:
            logger.warning("No Claude logs found for %s", self.project_path)

        logger.info(
            "Extracted %d total conversations from %s",
            len(events), self.project_name,
        )
        return events

    def _find_catchall_sessions(self) -> list[Path]:
        """Find sessions in catch-all parent directories that reference this project.

        When Claude Code is launched from a parent dir (e.g., ~/projects),
        sessions land in a catch-all dir (e.g., -home-brian-projects) regardless
        of which sub-project was worked on. We scan these files for the project
        path string using streaming 1MB-chunk binary search with early exit.
        """
        # Walk up from project_path to find parent dirs that have catch-all log dirs
        catchall_dirs: list[Path] = []
        parent = self.project_path.parent
        while parent != parent.parent:  # stop at filesystem root
            catchall_key = project_path_to_log_key(parent)
            catchall_dir = self.claude_logs_dir / catchall_key
            if catchall_dir.is_dir():
                catchall_dirs.append(catchall_dir)
            parent = parent.parent

        if not catchall_dirs:
            return []

        # Binary search target: the project path as it appears in file_path args
        # e.g., "/projects/agent_ontology/" — must include trailing slash to avoid
        # matching "agent_ontology_v2" etc.
        target = str(self.project_path).encode() + b"/"
        # Also match without trailing slash for project name mentions in text
        target_bare = str(self.project_path).encode()

        BUF_SIZE = 1024 * 1024  # 1MB chunks
        matching_files: list[Path] = []

        for catchall_dir in catchall_dirs:
            jsonl_files = sorted(catchall_dir.glob("*.jsonl"))
            logger.debug(
                "Scanning %d catch-all files in %s for %s",
                len(jsonl_files), catchall_dir.name, self.project_path.name,
            )
            for jsonl_path in jsonl_files:
                try:
                    with jsonl_path.open("rb") as fh:
                        while True:
                            chunk = fh.read(BUF_SIZE)
                            if not chunk:
                                break
                            if target in chunk or target_bare in chunk:
                                matching_files.append(jsonl_path)
                                break
                except OSError:
                    logger.debug("Could not read %s", jsonl_path)

        if matching_files:
            logger.info(
                "Catch-all scan: %d/%d files match %s",
                len(matching_files),
                sum(len(list(d.glob("*.jsonl"))) for d in catchall_dirs),
                self.project_path.name,
            )
        return matching_files

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
        files_modified: set[str] = set()
        files_examined: set[str] = set()
        tool_sequence: list[tuple[str, str]] = []
        assistant_reasoning_parts: list[str] = []
        commit_hashes: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        model_used = None
        git_branch = None
        message_count = 0

        # Build a map of tool_use_id -> tool_use block for commit hash extraction
        tool_use_map: dict[str, dict] = {}

        for msg in messages:
            msg_type = msg.get("type")

            if msg_type == "user":
                message_count += 1
                content = msg.get("message", {}).get("content", "")

                # Check for tool_result blocks (response to tool_use)
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            tool_use_id = block.get("tool_use_id", "")
                            result_content = block.get("content", "")
                            if isinstance(result_content, str) and tool_use_id in tool_use_map:
                                tool_info = tool_use_map[tool_use_id]
                                if tool_info.get("name") == "Bash":
                                    # Look for git commit output
                                    for match in COMMIT_HASH_RE.finditer(result_content):
                                        h = match.group(1)
                                        if h not in commit_hashes:
                                            commit_hashes.append(h)
                    # Also extract text content from user message blocks
                    text_parts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
                    ]
                    if text_parts:
                        user_text = "\n".join(text_parts).strip()
                        user_messages.append(user_text)
                        if first_user is None:
                            first_user = user_text
                            session_timestamp = msg.get("timestamp")
                elif isinstance(content, str) and content.strip():
                    user_messages.append(content.strip())
                    if first_user is None:
                        first_user = content.strip()
                        session_timestamp = msg.get("timestamp")

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
                    block_type = block.get("type")

                    if block_type == "text":
                        text = block.get("text", "").strip()
                        if text:
                            assistant_reasoning_parts.append(text)

                    elif block_type == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tools_used.add(tool_name)
                        tool_input = block.get("input", {})
                        tool_use_id = block.get("id", "")

                        # Store for commit hash extraction from tool results
                        if tool_use_id:
                            tool_use_map[tool_use_id] = block

                        # Track files modified (Edit, Write, NotebookEdit)
                        if tool_name in ("Edit", "Write", "NotebookEdit"):
                            fp = tool_input.get("file_path") or tool_input.get("notebook_path", "")
                            if fp:
                                files_modified.add(fp)

                        # Track files examined (Read, Glob, Grep)
                        if tool_name == "Read":
                            fp = tool_input.get("file_path", "")
                            if fp:
                                files_examined.add(fp)
                        elif tool_name in ("Glob", "Grep"):
                            fp = tool_input.get("path", "")
                            if fp:
                                files_examined.add(fp)

                        # Build brief context for tool sequence
                        brief = _tool_brief(tool_name, tool_input)
                        tool_sequence.append((tool_name, brief))

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

        # Truncate assistant reasoning
        full_reasoning = "\n\n".join(assistant_reasoning_parts)
        if len(full_reasoning) > self.max_reasoning_chars:
            assistant_reasoning = full_reasoning[:self.max_reasoning_chars] + "\n... [truncated]"
        else:
            assistant_reasoning = full_reasoning if full_reasoning else None

        raw_data: dict = {
            "session_id": session_id,
            "model": model_used,
            "message_count": message_count,
            "user_message_count": len(user_messages),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "tools_used": sorted(tools_used),
            "commit_hashes": commit_hashes,
            "files_modified": sorted(files_modified),
            "files_examined": sorted(files_examined),
            "tool_sequence": [(t, b) for t, b in tool_sequence[:100]],  # cap at 100 entries
            "assistant_reasoning": assistant_reasoning,
        }

        # files_changed = files the conversation actually modified
        all_files = sorted(files_modified)

        return ExtractedEvent(
            event_type=EventType.CONVERSATION,
            source_id=f"claude:{session_id}",
            timestamp=ts,
            author="brian",
            title=title,
            body=body,
            raw_data=raw_data,
            files_changed=all_files if all_files else None,
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


def _tool_brief(tool_name: str, tool_input: dict) -> str:
    """Create a brief context string for a tool call."""
    if tool_name in ("Edit", "Write"):
        fp = tool_input.get("file_path", "")
        return Path(fp).name if fp else ""
    if tool_name == "Read":
        fp = tool_input.get("file_path", "")
        return Path(fp).name if fp else ""
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return cmd[:60] if cmd else ""
    if tool_name == "Glob":
        return tool_input.get("pattern", "")[:40]
    if tool_name == "Grep":
        return tool_input.get("pattern", "")[:40]
    if tool_name == "Task":
        return tool_input.get("description", "")[:40]
    return ""
