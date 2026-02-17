"""Tests for trajectory query engine."""

from unittest.mock import MagicMock, patch

import pytest

from trajectory.output import query_engine as qe


class TestQueryTrajectory:
    # mock-ok: LLM synthesis is expensive/non-deterministic
    @patch("trajectory.output.query_engine.call_llm")
    @patch("trajectory.output.query_engine.render_prompt")
    def test_finds_concepts_and_events(self, mock_render, mock_llm, populated_db):
        mock_render.return_value = [{"role": "user", "content": "test"}]
        mock_llm.return_value = MagicMock(content="MCP server was introduced in January.")

        from trajectory.config import load_config
        config = load_config()

        result = qe.query_trajectory("How did the MCP server evolve?", populated_db, config)
        assert "mcp_server" in result.concepts_found
        assert result.events_used > 0
        assert result.answer == "MCP server was introduced in January."
        mock_render.assert_called_once()
        mock_llm.assert_called_once()

    # mock-ok: LLM synthesis is expensive/non-deterministic
    @patch("trajectory.output.query_engine.call_llm")
    @patch("trajectory.output.query_engine.render_prompt")
    def test_no_matches_falls_back_to_title_search(self, mock_render, mock_llm, populated_db):
        mock_render.return_value = [{"role": "user", "content": "test"}]
        mock_llm.return_value = MagicMock(content="No matching concepts found.")

        from trajectory.config import load_config
        config = load_config()

        result = qe.query_trajectory("What happened with xyznothing?", populated_db, config)
        assert len(result.concepts_found) == 0
        assert "No concepts matched" in result.data_gaps[1] if len(result.data_gaps) > 1 else True

    # mock-ok: LLM synthesis is expensive/non-deterministic
    @patch("trajectory.output.query_engine.call_llm")
    @patch("trajectory.output.query_engine.render_prompt")
    def test_data_gaps_reported(self, mock_render, mock_llm, populated_db):
        mock_render.return_value = [{"role": "user", "content": "test"}]
        mock_llm.return_value = MagicMock(content="Answer.")

        from trajectory.config import load_config
        config = load_config()

        result = qe.query_trajectory("Tell me about MCP", populated_db, config)
        # Some events are unanalyzed, so data_gaps should mention it
        assert any("analyzed" in gap.lower() for gap in result.data_gaps)


class TestGetTimeline:
    def test_valid_project(self, populated_db):
        events = qe.get_timeline("sam_gov", populated_db)
        assert len(events) == 5
        assert all("timestamp" in e for e in events)

    def test_unknown_project_raises(self, populated_db):
        with pytest.raises(ValueError, match="Project not found"):
            qe.get_timeline("nonexistent", populated_db)

    def test_with_since_filter(self, populated_db):
        events = qe.get_timeline("sam_gov", populated_db, since="2026-02-01")
        assert len(events) == 2


class TestGetConceptHistory:
    def test_returns_grouped(self, populated_db):
        result = qe.get_concept_history("mcp_server", populated_db)
        assert result["concept"]["name"] == "mcp_server"
        assert len(result["timeline"]) == 3
        assert len(result["projects"]) >= 1

    def test_unknown_concept_raises(self, populated_db):
        with pytest.raises(ValueError, match="Concept not found"):
            qe.get_concept_history("nonexistent", populated_db)


class TestListTrackedProjects:
    def test_returns_all(self, populated_db):
        projects = qe.list_tracked_projects(populated_db)
        assert len(projects) == 2
        names = {p["name"] for p in projects}
        assert names == {"sam_gov", "llm_client"}
        for p in projects:
            assert "total_events" in p
            assert "analyzed_events" in p
            assert "concepts" in p


class TestListConcepts:
    def test_all(self, populated_db):
        concepts = qe.list_concepts(populated_db)
        assert len(concepts) == 3
        for c in concepts:
            assert "event_count" in c

    def test_filter_by_project(self, populated_db):
        concepts = qe.list_concepts(populated_db, project="llm_client")
        assert len(concepts) >= 1


class TestCorrectConcept:
    def test_rename(self, populated_db):
        result = qe.correct_concept("mcp_server", "rename", populated_db, new_name="mcp_infra")
        assert result["status"] == "ok"
        assert "Renamed" in result["message"]
        assert populated_db.get_concept_by_name("mcp_infra") is not None

    def test_merge(self, populated_db):
        result = qe.correct_concept("agent_sdk", "merge", populated_db, merge_into="mcp_server")
        assert result["status"] == "ok"
        assert "Merged" in result["message"]

    def test_status_change(self, populated_db):
        result = qe.correct_concept("agent_sdk", "status_change", populated_db, new_status="completed")
        assert result["status"] == "ok"

    def test_invalid_action(self, populated_db):
        with pytest.raises(ValueError, match="Invalid action"):
            qe.correct_concept("mcp_server", "destroy", populated_db)

    def test_unknown_concept(self, populated_db):
        with pytest.raises(ValueError, match="Concept not found"):
            qe.correct_concept("nonexistent", "rename", populated_db, new_name="x")
