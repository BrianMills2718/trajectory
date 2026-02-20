#!/usr/bin/env python3
"""Trajectory MCP Server — query project evolution."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
_llm_client_path = str(Path.home() / "projects" / "llm_client")
if _llm_client_path not in sys.path:
    sys.path.insert(0, _llm_client_path)

from mcp.server.fastmcp import FastMCP

from trajectory.config import load_config
from trajectory.db import TrajectoryDB
from trajectory.output import query_engine as qe

mcp = FastMCP("trajectory")
logger = logging.getLogger(__name__)

# Redirect all logging to stderr so stdout stays clean for MCP stdio transport
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_db: TrajectoryDB | None = None
_config = None


def _get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_db() -> TrajectoryDB:
    global _db
    if _db is None:
        _db = TrajectoryDB(_get_config())
        _db.init_db()
    return _db


@mcp.tool()
def query_trajectory(question: str) -> str:
    """Ask a natural language question about project evolution. Uses LLM synthesis."""
    try:
        result = qe.query_trajectory(question, _get_db(), _get_config())
        return json.dumps(result.model_dump())
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_timeline(
    project_name: str,
    since: Optional[str] = None,
    until: Optional[str] = None,
    min_significance: Optional[float] = None,
) -> str:
    """Get chronological events for a project. Dates as ISO strings (YYYY-MM-DD)."""
    try:
        result = qe.get_timeline(
            project_name, _get_db(),
            since=since, until=until, min_significance=min_significance,
        )
        return json.dumps(result, default=str)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_concept_history(concept_name: str) -> str:
    """Get a concept's full history across all projects."""
    try:
        result = qe.get_concept_history(concept_name, _get_db())
        return json.dumps(result, default=str)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_tracked_projects() -> str:
    """List all tracked projects with event counts and analysis status."""
    result = qe.list_tracked_projects(_get_db())
    return json.dumps(result, default=str)


@mcp.tool()
def list_concepts(
    status: Optional[str] = None,
    project: Optional[str] = None,
    level: Optional[str] = None,
) -> str:
    """List concepts with optional filters by status, project name, or level (theme/design_bet/technique)."""
    try:
        result = qe.list_concepts(_get_db(), status=status, project=project, level=level)
        return json.dumps(result, default=str)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def ingest_project(project_path: str) -> str:
    """Ingest events from a project directory (git commits, Claude logs, docs)."""
    try:
        result = qe.ingest_project_from_path(project_path, _get_db(), _get_config())
        return json.dumps(result, default=str)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_concept_links(
    concept_name: Optional[str] = None,
    relationship: Optional[str] = None,
    min_strength: Optional[float] = None,
) -> str:
    """Get cross-project concept links. Optionally filter by concept name, relationship type, or minimum strength."""
    try:
        result = qe.get_concept_links(
            _get_db(),
            concept_name=concept_name,
            relationship=relationship,
            min_strength=min_strength,
        )
        return json.dumps(result, default=str)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def correct_concept(
    concept_name: str,
    action: str,
    new_name: Optional[str] = None,
    merge_into: Optional[str] = None,
    new_status: Optional[str] = None,
) -> str:
    """Correct a concept: rename, merge into another, or change status."""
    try:
        result = qe.correct_concept(
            concept_name, action, _get_db(),
            new_name=new_name, merge_into=merge_into, new_status=new_status,
        )
        return json.dumps(result)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
