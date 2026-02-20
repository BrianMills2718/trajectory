"""Extract events from git commit history using PyDriller."""

import logging
import signal
from datetime import datetime, timezone
from pathlib import Path

from pydriller import Repository

from trajectory.extractors.base import BaseExtractor
from trajectory.models import EventType, ExtractedEvent

logger = logging.getLogger(__name__)

REPO_TIMEOUT_SECONDS = 120
FILE_TIMEOUT_SECONDS = 10


class _RepoTimeout(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise _RepoTimeout("Repository extraction timed out")


def _build_diff_summary(
    commit: object,
    max_lines: int = 5,
) -> tuple[str, dict[str, str]]:
    """Build a per-commit diff summary and change_types dict.

    Returns (diff_summary_string, change_types_dict).
    Per-file errors are logged and skipped — we don't lose the whole commit.
    """
    change_types: dict[str, str] = {}
    lines: list[str] = []

    for mf in commit.modified_files:  # type: ignore[attr-defined]
        try:
            filepath = mf.new_path or mf.old_path or "unknown"
            ct = mf.change_type.name if mf.change_type else "UNKNOWN"
            change_types[filepath] = ct

            added = mf.added_lines
            deleted = mf.deleted_lines

            summary = f"  {ct} {filepath} (+{added}/-{deleted})"

            # Include first N added lines for context
            if max_lines > 0 and mf.diff_parsed:
                added_lines = mf.diff_parsed.get("added", [])
                if added_lines:
                    preview = [
                        line[1].rstrip()
                        for line in added_lines[:max_lines]
                        if isinstance(line, tuple) and len(line) >= 2
                    ]
                    if preview:
                        summary += "\n" + "\n".join(f"    + {l}" for l in preview)
                    if len(added_lines) > max_lines:
                        summary += f"\n    ... ({len(added_lines) - max_lines} more lines)"

            lines.append(summary)
        except _RepoTimeout:
            raise
        except Exception:
            logger.debug(
                "Skipping diff for file in commit (file-level error)",
                exc_info=True,
            )

    return "\n".join(lines), change_types


class GitExtractor(BaseExtractor):
    """Extract commit events from a git repository."""

    def __init__(self, project_path: Path, max_diff_lines: int = 5) -> None:
        super().__init__(project_path)
        self.max_diff_lines = max_diff_lines

    def extract(self, since: str | None = None) -> list[ExtractedEvent]:
        git_dir = self.project_path / ".git"
        if not git_dir.exists():
            logger.warning("No .git directory found at %s", self.project_path)
            return []

        kwargs: dict[str, object] = {"path_to_repo": str(self.project_path)}
        if since:
            kwargs["since"] = datetime.fromisoformat(since)

        events: list[ExtractedEvent] = []
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(REPO_TIMEOUT_SECONDS)
        try:
            for commit in Repository(**kwargs).traverse_commits():
                title_line = commit.msg.split("\n", 1)[0].strip()
                body = commit.msg if "\n" in commit.msg else None

                # Build diff summary and change types
                diff_summary = None
                change_types = None
                try:
                    files_changed = [m.new_path or m.old_path for m in commit.modified_files]
                    diff_summary, change_types = _build_diff_summary(
                        commit, max_lines=self.max_diff_lines
                    )
                except _RepoTimeout:
                    raise
                except Exception:
                    logger.warning(
                        "Could not get modified files for %s in %s",
                        commit.hash[:8], self.project_name,
                    )
                    files_changed = []

                raw_data = {
                    "hash": commit.hash,
                    "insertions": commit.insertions,
                    "deletions": commit.deletions,
                    "files_count": commit.files,
                    "merge": commit.merge,
                }

                # PyDriller returns timezone-aware datetimes
                ts = commit.committer_date
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                events.append(ExtractedEvent(
                    event_type=EventType.COMMIT,
                    source_id=f"git:{commit.hash}",
                    timestamp=ts,
                    author=commit.author.name,
                    title=title_line,
                    body=body,
                    raw_data=raw_data,
                    files_changed=files_changed,
                    git_branch=None,  # branch info not reliable from traversal
                    diff_summary=diff_summary,
                    change_types=change_types if change_types else None,
                ))
        except _RepoTimeout:
            logger.warning(
                "Timed out after %ds extracting %s (got %d commits so far)",
                REPO_TIMEOUT_SECONDS, self.project_name, len(events),
            )
        except Exception:
            logger.exception("Error extracting git commits from %s", self.project_path)
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        logger.info("Extracted %d commits from %s", len(events), self.project_name)
        return events
