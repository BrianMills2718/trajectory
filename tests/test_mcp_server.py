"""Tests for trajectory MCP server tool registration and basic returns."""

import json

from trajectory.mcp_server import mcp, _get_db, _get_config
import trajectory.mcp_server as mcp_mod


EXPECTED_TOOLS = {
    "query_trajectory",
    "get_timeline",
    "get_concept_history",
    "list_tracked_projects",
    "list_concepts",
    "ingest_project",
    "correct_concept",
}


class TestMCPToolRegistration:
    def test_all_tools_registered(self):
        """All 7 expected tools are registered on the mcp object."""
        # FastMCP stores tools in _tool_manager._tools dict
        registered = set(mcp._tool_manager._tools.keys())
        assert EXPECTED_TOOLS.issubset(registered), (
            f"Missing tools: {EXPECTED_TOOLS - registered}"
        )

    def test_tool_count(self):
        registered = set(mcp._tool_manager._tools.keys())
        assert len(registered & EXPECTED_TOOLS) == 7


class TestMCPToolReturns:
    def test_list_tracked_projects_returns_json(self, populated_db, monkeypatch):
        """list_tracked_projects returns valid JSON."""
        monkeypatch.setattr(mcp_mod, "_db", populated_db)
        result = mcp_mod.list_tracked_projects()
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_error_returns_json_error(self, populated_db, monkeypatch):
        """Tools return {"error": ...} on ValueError."""
        monkeypatch.setattr(mcp_mod, "_db", populated_db)
        result = mcp_mod.get_timeline(project_name="nonexistent_project")
        data = json.loads(result)
        assert "error" in data
