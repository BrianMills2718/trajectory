"""Base extractor interface."""

import abc
import logging
from pathlib import Path

from trajectory.models import ExtractedEvent

logger = logging.getLogger(__name__)


class BaseExtractor(abc.ABC):
    """Base class for all event extractors."""

    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path

    @abc.abstractmethod
    def extract(self, since: str | None = None) -> list[ExtractedEvent]:
        """Extract events from the source.

        Args:
            since: ISO 8601 timestamp. Only return events after this time.

        Returns:
            List of extracted events.
        """
        ...

    @property
    def project_name(self) -> str:
        return self.project_path.name
