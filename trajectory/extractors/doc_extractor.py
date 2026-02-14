"""Extract events from project documentation files."""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from trajectory.config import ExtractionConfig
from trajectory.extractors.base import BaseExtractor
from trajectory.models import EventType, ExtractedEvent

logger = logging.getLogger(__name__)

# Match date patterns in directory names or filenames
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
DATE_PATTERN_COMPACT = re.compile(r"(\d{8})")


class DocExtractor(BaseExtractor):
    """Extract events from project docs, archives, and status files."""

    def __init__(
        self,
        project_path: Path,
        config: ExtractionConfig | None = None,
    ) -> None:
        super().__init__(project_path)
        self.config = config or ExtractionConfig()

    def extract(self, since: str | None = None) -> list[ExtractedEvent]:
        since_dt = None
        if since:
            since_dt = datetime.fromisoformat(since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)

        events: list[ExtractedEvent] = []

        # 1. Root-level doc files (CLAUDE.md, STATUS.md, README.md, etc.)
        events.extend(self._extract_root_docs(since_dt))

        # 2. Archive directories (docs/archive/YYYY-MM-DD/*, archive/YYYY-MM-DD/*)
        events.extend(self._extract_archive_docs(since_dt))

        logger.info("Extracted %d doc events from %s", len(events), self.project_name)
        return events

    def _extract_root_docs(self, since_dt: datetime | None) -> list[ExtractedEvent]:
        """Extract events from root-level documentation files."""
        events: list[ExtractedEvent] = []

        for pattern in self.config.doc_patterns:
            doc_path = self.project_path / pattern
            if not doc_path.exists():
                continue
            if doc_path.stat().st_size > self.config.max_file_size:
                logger.info("Skipping large file %s (%d bytes)", doc_path, doc_path.stat().st_size)
                continue

            mtime = datetime.fromtimestamp(doc_path.stat().st_mtime, tz=timezone.utc)
            if since_dt and mtime < since_dt:
                continue

            content = doc_path.read_text(errors="replace")
            title = self._extract_title(content, doc_path.name)

            events.append(ExtractedEvent(
                event_type=EventType.DOC_CHANGE,
                source_id=f"doc:{self.project_name}/{pattern}",
                timestamp=mtime,
                title=title,
                body=content,
                raw_data={
                    "file": pattern,
                    "size_bytes": doc_path.stat().st_size,
                    "doc_type": "root",
                },
            ))

        return events

    def _extract_archive_docs(self, since_dt: datetime | None) -> list[ExtractedEvent]:
        """Extract events from archive directories with date-stamped subdirs."""
        events: list[ExtractedEvent] = []

        for archive_pattern in self.config.archive_patterns:
            archive_dir = self.project_path / archive_pattern
            if not archive_dir.is_dir():
                continue

            for entry in sorted(archive_dir.iterdir()):
                if not entry.is_dir():
                    # Top-level files in archive (e.g., *_SUMMARY.md)
                    if entry.suffix == ".md":
                        event = self._extract_single_doc(
                            entry, archive_pattern, since_dt
                        )
                        if event:
                            events.append(event)
                    continue

                # Date-stamped directories
                date = self._parse_date_from_name(entry.name)
                if date and since_dt and date < since_dt:
                    continue

                # Extract all .md files in the date directory
                for md_file in sorted(entry.glob("*.md")):
                    event = self._extract_single_doc(
                        md_file, archive_pattern, since_dt, dir_date=date
                    )
                    if event:
                        events.append(event)

        return events

    def _extract_single_doc(
        self,
        doc_path: Path,
        archive_pattern: str,
        since_dt: datetime | None,
        dir_date: datetime | None = None,
    ) -> ExtractedEvent | None:
        """Extract a single document as an event."""
        if doc_path.stat().st_size > self.config.max_file_size:
            return None

        # Use dir_date if available, otherwise file mtime
        if dir_date:
            ts = dir_date
        else:
            ts = datetime.fromtimestamp(doc_path.stat().st_mtime, tz=timezone.utc)

        if since_dt and ts < since_dt:
            return None

        content = doc_path.read_text(errors="replace")
        title = self._extract_title(content, doc_path.name)
        rel_path = str(doc_path.relative_to(self.project_path))

        return ExtractedEvent(
            event_type=EventType.ARCHIVE,
            source_id=f"doc:{self.project_name}/{rel_path}",
            timestamp=ts,
            title=title,
            body=content,
            raw_data={
                "file": rel_path,
                "size_bytes": doc_path.stat().st_size,
                "archive_pattern": archive_pattern,
                "doc_type": "archive",
            },
        )

    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from markdown heading or filename."""
        for line in content.split("\n", 10):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()[:200]
        # Fall back to filename without extension
        return Path(filename).stem.replace("_", " ").replace("-", " ").title()

    def _parse_date_from_name(self, name: str) -> datetime | None:
        """Try to extract a date from a directory/file name."""
        match = DATE_PATTERN.search(name)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        match = DATE_PATTERN_COMPACT.search(name)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        return None
