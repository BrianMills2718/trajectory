"""Shared test fixtures for trajectory tests."""

import pytest

from trajectory.config import Config, DigestConfig, ExtractionConfig, LLMConfig
from trajectory.db import TrajectoryDB
from trajectory.models import EventInsert, EventType


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a TrajectoryDB backed by a temp file."""
    config = Config(
        db_path=str(tmp_path / "test.db"),
        projects_dir="/tmp/projects",
        claude_logs_dir="/tmp/claude_logs",
        llm=LLMConfig(),
        extraction=ExtractionConfig(),
        digest=DigestConfig(),
    )
    db = TrajectoryDB(config)
    db.init_db()
    yield db
    db.close()


@pytest.fixture()
def populated_db(tmp_db):
    """DB pre-loaded with 2 projects, ~10 events, 3 concepts, 5 concept_events."""
    db = tmp_db

    # Projects
    p1_id = db.upsert_project("sam_gov", "/home/brian/projects/sam_gov")
    p2_id = db.upsert_project("llm_client", "/home/brian/projects/llm_client")

    # Events for project 1
    events_p1 = [
        EventInsert(
            project_id=p1_id, event_type=EventType.COMMIT, source_id="git:aaa111",
            timestamp="2026-01-15T10:00:00", author="brian", title="Add MCP server",
        ),
        EventInsert(
            project_id=p1_id, event_type=EventType.COMMIT, source_id="git:aaa222",
            timestamp="2026-01-20T10:00:00", author="brian", title="Fix search query builder",
        ),
        EventInsert(
            project_id=p1_id, event_type=EventType.CONVERSATION, source_id="claude:sess1",
            timestamp="2026-01-25T10:00:00", author="brian", title="Discuss MCP architecture",
        ),
        EventInsert(
            project_id=p1_id, event_type=EventType.DOC_CHANGE, source_id="doc:sam_gov/CLAUDE.md",
            timestamp="2026-02-01T10:00:00", author="brian", title="Update CLAUDE.md",
        ),
        EventInsert(
            project_id=p1_id, event_type=EventType.COMMIT, source_id="git:aaa333",
            timestamp="2026-02-05T10:00:00", author="brian", title="Deprecate search_sam",
            body="Rate limits too aggressive",
        ),
    ]
    for e in events_p1:
        db.insert_event(e)

    # Events for project 2
    events_p2 = [
        EventInsert(
            project_id=p2_id, event_type=EventType.COMMIT, source_id="git:bbb111",
            timestamp="2026-01-10T10:00:00", author="brian", title="Add retry logic",
        ),
        EventInsert(
            project_id=p2_id, event_type=EventType.COMMIT, source_id="git:bbb222",
            timestamp="2026-01-18T10:00:00", author="brian", title="Add structured output",
        ),
        EventInsert(
            project_id=p2_id, event_type=EventType.COMMIT, source_id="git:bbb333",
            timestamp="2026-02-01T10:00:00", author="brian", title="Add agent SDK support",
        ),
        EventInsert(
            project_id=p2_id, event_type=EventType.COMMIT, source_id="git:bbb444",
            timestamp="2026-02-10T10:00:00", author="brian", title="Add MCP server control",
        ),
        EventInsert(
            project_id=p2_id, event_type=EventType.DOC_CHANGE, source_id="doc:llm_client/README.md",
            timestamp="2026-02-12T10:00:00", author="brian", title="Update README with agent docs",
        ),
    ]
    for e in events_p2:
        db.insert_event(e)

    # Mark some events as analyzed
    run_id = db.create_analysis_run(
        model="gemini-flash", prompt_version="v1", project_id=p1_id,
        started_at="2026-02-10T10:00:00",
    )
    # Analyze first 3 events of p1
    for i, src in enumerate(["git:aaa111", "git:aaa222", "claude:sess1"]):
        event = db.conn.execute(
            "SELECT id FROM events WHERE source_id = ?", (src,)
        ).fetchone()
        db.update_event_analysis(
            event["id"],
            llm_summary=f"Summary for event {i}",
            llm_intent="feature",
            significance=0.5 + i * 0.1,
            analysis_run_id=run_id,
        )
    db.conn.commit()

    # Concepts
    c1_id = db.upsert_concept("mcp_server", "MCP server infrastructure", "2026-01-15", "2026-02-10")
    c2_id = db.upsert_concept("structured_output", "Structured LLM output", "2026-01-18", "2026-02-01")
    c3_id = db.upsert_concept("agent_sdk", "Agent SDK integration", "2026-02-01", "2026-02-10")

    # Concept-event links (5 total)
    e_mcp1 = db.conn.execute("SELECT id FROM events WHERE source_id = 'git:aaa111'").fetchone()["id"]
    e_mcp2 = db.conn.execute("SELECT id FROM events WHERE source_id = 'git:bbb444'").fetchone()["id"]
    e_struct = db.conn.execute("SELECT id FROM events WHERE source_id = 'git:bbb222'").fetchone()["id"]
    e_agent1 = db.conn.execute("SELECT id FROM events WHERE source_id = 'git:bbb333'").fetchone()["id"]
    e_discuss = db.conn.execute("SELECT id FROM events WHERE source_id = 'claude:sess1'").fetchone()["id"]

    db.link_concept_event(c1_id, e_mcp1, "introduces", 0.9, "Initial MCP server commit", run_id)
    db.link_concept_event(c1_id, e_mcp2, "develops", 0.8, "MCP server control feature", run_id)
    db.link_concept_event(c2_id, e_struct, "introduces", 0.95, "First structured output", run_id)
    db.link_concept_event(c3_id, e_agent1, "introduces", 0.85, "Agent SDK support added", run_id)
    db.link_concept_event(c1_id, e_discuss, "references", 0.7, "Discussion about MCP", run_id)
    db.conn.commit()

    return db
