"""Tests for concept linking — DB methods, query engine, and MCP tool."""

from unittest.mock import patch

import pytest

from trajectory.models import ConceptLinkRow


# --- DB method tests ---


class TestConceptLinkDB:
    def test_upsert_concept_link(self, populated_db):
        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        link_id = db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.8, "MCP uses structured output")
        assert link_id > 0

        links = db.get_concept_links()
        assert len(links) == 1
        assert links[0].concept_a_id == c1.id
        assert links[0].concept_b_id == c2.id
        assert links[0].relationship == "depends_on"
        assert links[0].strength == 0.8

    def test_upsert_concept_link_updates_existing(self, populated_db):
        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.5, "initial evidence")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.9, "updated evidence")

        links = db.get_concept_links()
        assert len(links) == 1
        assert links[0].strength == 0.9
        assert links[0].evidence == "updated evidence"

    def test_delete_all_concept_links(self, populated_db):
        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        c3 = db.get_concept_by_name("agent_sdk")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.8)
        db.upsert_concept_link(c1.id, c3.id, "related_to", 0.6)
        db.conn.commit()

        deleted = db.delete_all_concept_links()
        assert deleted == 2
        assert db.get_concept_links() == []

    def test_get_concept_links_filter_by_concept(self, populated_db):
        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        c3 = db.get_concept_by_name("agent_sdk")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.8)
        db.upsert_concept_link(c2.id, c3.id, "related_to", 0.6)
        db.conn.commit()

        links = db.get_concept_links(concept_id=c1.id)
        assert len(links) == 1
        assert links[0].relationship == "depends_on"

        # c2 appears in both links
        links_c2 = db.get_concept_links(concept_id=c2.id)
        assert len(links_c2) == 2

    def test_get_concept_links_filter_by_relationship(self, populated_db):
        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        c3 = db.get_concept_by_name("agent_sdk")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.8)
        db.upsert_concept_link(c1.id, c3.id, "related_to", 0.6)
        db.conn.commit()

        links = db.get_concept_links(relationship="depends_on")
        assert len(links) == 1

    def test_get_concept_links_filter_by_min_strength(self, populated_db):
        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        c3 = db.get_concept_by_name("agent_sdk")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.8)
        db.upsert_concept_link(c1.id, c3.id, "related_to", 0.3)
        db.conn.commit()

        links = db.get_concept_links(min_strength=0.5)
        assert len(links) == 1
        assert links[0].strength == 0.8


# --- Query engine tests ---


class TestConceptLinksQueryEngine:
    def test_get_concept_links_resolves_names(self, populated_db):
        from trajectory.output.query_engine import get_concept_links

        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.8, "MCP needs structured output")
        db.conn.commit()

        result = get_concept_links(db)
        assert len(result) == 1
        assert result[0]["concept_a"] == "mcp_server"
        assert result[0]["concept_b"] == "structured_output"
        assert result[0]["relationship"] == "depends_on"
        assert result[0]["evidence"] == "MCP needs structured output"

    def test_get_concept_links_filter_by_name(self, populated_db):
        from trajectory.output.query_engine import get_concept_links

        db = populated_db
        c1 = db.get_concept_by_name("mcp_server")
        c2 = db.get_concept_by_name("structured_output")
        c3 = db.get_concept_by_name("agent_sdk")
        db.upsert_concept_link(c1.id, c2.id, "depends_on", 0.8)
        db.upsert_concept_link(c2.id, c3.id, "related_to", 0.6)
        db.conn.commit()

        result = get_concept_links(db, concept_name="mcp_server")
        assert len(result) == 1

    def test_get_concept_links_unknown_concept_raises(self, populated_db):
        from trajectory.output.query_engine import get_concept_links

        with pytest.raises(ValueError, match="Concept not found"):
            get_concept_links(populated_db, concept_name="nonexistent")

    def test_get_concept_links_empty(self, populated_db):
        from trajectory.output.query_engine import get_concept_links

        result = get_concept_links(populated_db)
        assert result == []


# --- MCP tool test ---


class TestConceptLinksMCP:
    def test_get_concept_links_tool_registered(self):
        from trajectory.mcp_server import mcp

        tool_names = list(mcp._tool_manager._tools.keys())
        assert "get_concept_links" in tool_names
