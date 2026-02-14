"""Configuration loading for trajectory tracker."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str = "gemini/gemini-2.0-flash"
    quality_model: str = "anthropic/claude-sonnet-4-20250514"
    batch_size: int = 30
    max_cost_per_run: float = 1.00


class ExtractionConfig(BaseModel):
    doc_patterns: list[str] = Field(default_factory=lambda: [
        "CLAUDE.md", "STATUS.md", "README.md", "ROADMAP.md",
    ])
    archive_patterns: list[str] = Field(default_factory=lambda: [
        "docs/archive", "archive",
    ])
    max_file_size: int = 500_000


class DigestConfig(BaseModel):
    significance_threshold: float = 0.5
    daily_max_events: int = 50
    weekly_max_events: int = 200


class Config(BaseModel):
    db_path: str = "data/trajectory.db"
    projects_dir: str = "/home/brian/projects"
    claude_logs_dir: str = "~/.claude/projects"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)

    @property
    def resolved_db_path(self) -> Path:
        """Resolve db_path relative to project root."""
        p = Path(self.db_path)
        if p.is_absolute():
            return p
        return _project_root() / p

    @property
    def resolved_claude_logs_dir(self) -> Path:
        return Path(self.claude_logs_dir).expanduser()

    @property
    def resolved_projects_dir(self) -> Path:
        return Path(self.projects_dir).expanduser()


def _project_root() -> Path:
    """Return the trajectory project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Path | None = None) -> Config:
    """Load config from YAML file. Falls back to defaults if file missing."""
    if config_path is None:
        config_path = _project_root() / "config.yaml"

    if config_path.exists():
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
        return Config(**raw)

    return Config()
