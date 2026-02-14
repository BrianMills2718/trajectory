"""Pydantic models for trajectory tracker."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    COMMIT = "commit"
    CONVERSATION = "conversation"
    DOC_CHANGE = "doc_change"
    ARCHIVE = "archive"


class Intent(str, Enum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    EXPLORATION = "exploration"
    DOCUMENTATION = "documentation"
    TEST = "test"
    INFRASTRUCTURE = "infrastructure"
    CLEANUP = "cleanup"


class ConceptStatus(str, Enum):
    ACTIVE = "active"
    ABANDONED = "abandoned"
    COMPLETED = "completed"
    EVOLVED = "evolved"
    MERGED = "merged"


class ConceptRelationship(str, Enum):
    INTRODUCES = "introduces"
    DEVELOPS = "develops"
    REFACTORS = "refactors"
    ABANDONS = "abandons"
    COMPLETES = "completes"
    REFERENCES = "references"


class ConceptLinkType(str, Enum):
    DEPENDS_ON = "depends_on"
    EVOLVED_FROM = "evolved_from"
    REPLACED_BY = "replaced_by"
    RELATED_TO = "related_to"
    SPAWNED = "spawned"


class CorrectionType(str, Enum):
    RENAME = "rename"
    MERGE = "merge"
    SPLIT = "split"
    STATUS_CHANGE = "status_change"
    RECLASSIFY = "reclassify"


class DecisionType(str, Enum):
    ARCHITECTURAL = "architectural"
    TOOLING = "tooling"
    PATTERN = "pattern"
    TECHNOLOGY = "technology"
    PROCESS = "process"


# --- Row models (what comes out of the DB) ---


class ProjectRow(BaseModel):
    id: int
    name: str
    path: str
    git_remote: str | None = None
    description: str | None = None
    total_commits: int = 0
    total_conversations: int = 0
    last_ingested: str | None = None
    created_at: str


class EventRow(BaseModel):
    id: int
    project_id: int
    event_type: EventType
    source_id: str
    timestamp: str
    author: str | None = None
    title: str
    body: str | None = None
    raw_data: str | None = None
    files_changed: str | None = None
    git_branch: str | None = None
    llm_summary: str | None = None
    llm_intent: str | None = None
    significance: float | None = None
    analysis_run_id: int | None = None
    created_at: str


class ConceptRow(BaseModel):
    id: int
    name: str
    description: str | None = None
    first_seen: str | None = None
    last_seen: str | None = None
    status: ConceptStatus = ConceptStatus.ACTIVE
    parent_concept_id: int | None = None
    merged_into_id: int | None = None
    created_at: str
    updated_at: str


# --- Insert models (what goes into the DB) ---


class EventInsert(BaseModel):
    """Event data ready to insert into the database."""
    project_id: int
    event_type: EventType
    source_id: str
    timestamp: str
    author: str | None = None
    title: str
    body: str | None = None
    raw_data: str | None = None
    files_changed: str | None = None
    git_branch: str | None = None


# --- Extractor output models ---


class ExtractedEvent(BaseModel):
    """Raw event from an extractor, before DB insertion."""
    event_type: EventType
    source_id: str
    timestamp: datetime
    author: str | None = None
    title: str
    body: str | None = None
    raw_data: dict[str, Any] | None = None
    files_changed: list[str] | None = None
    git_branch: str | None = None
