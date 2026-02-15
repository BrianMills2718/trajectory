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


class _RepoTimeout(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise _RepoTimeout("Repository extraction timed out")


class GitExtractor(BaseExtractor):
    """Extract commit events from a git repository."""

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

                # Get file list from modified_files — skip if single commit takes too long
                try:
                    files_changed = [m.new_path or m.old_path for m in commit.modified_files]
                except _RepoTimeout:
                    raise
                except Exception:
                    logger.warning("Could not get modified files for %s in %s", commit.hash[:8], self.project_name)
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
