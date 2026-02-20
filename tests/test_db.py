"""Tests for new TrajectoryDB query and correction methods."""

import pytest

from trajectory.models import EventInsert, EventType


class TestGetProjectByName:
    def test_found(self, populated_db):
        p = populated_db.get_project_by_name("sam_gov")
        assert p is not None
        assert p.name == "sam_gov"

    def test_not_found(self, populated_db):
        assert populated_db.get_project_by_name("nonexistent") is None

    def test_case_insensitive(self, populated_db):
        p = populated_db.get_project_by_name("SAM_GOV")
        assert p is not None
        assert p.name == "sam_gov"


class TestGetConceptByName:
    def test_found(self, populated_db):
        c = populated_db.get_concept_by_name("mcp_server")
        assert c is not None
        assert c.name == "mcp_server"

    def test_not_found(self, populated_db):
        assert populated_db.get_concept_by_name("nonexistent") is None

    def test_case_insensitive(self, populated_db):
        c = populated_db.get_concept_by_name("MCP_SERVER")
        assert c is not None
        assert c.name == "mcp_server"


class TestListConcepts:
    def test_all(self, populated_db):
        concepts = populated_db.list_concepts()
        assert len(concepts) == 3
        names = {c.name for c in concepts}
        assert names == {"mcp_server", "structured_output", "agent_sdk"}

    def test_by_status(self, populated_db):
        concepts = populated_db.list_concepts(status="active")
        assert len(concepts) == 3  # all are active by default

    def test_by_status_no_match(self, populated_db):
        concepts = populated_db.list_concepts(status="abandoned")
        assert len(concepts) == 0

    def test_by_project(self, populated_db):
        p = populated_db.get_project_by_name("llm_client")
        concepts = populated_db.list_concepts(project_id=p.id)
        names = {c.name for c in concepts}
        # mcp_server (via git:bbb444), structured_output (via git:bbb222), agent_sdk (via git:bbb333)
        assert names == {"mcp_server", "structured_output", "agent_sdk"}


class TestSearchConcepts:
    def test_matching(self, populated_db):
        results = populated_db.search_concepts(["mcp"])
        assert len(results) == 1
        assert results[0].name == "mcp_server"

    def test_multiple_keywords(self, populated_db):
        results = populated_db.search_concepts(["agent", "server"])
        names = {c.name for c in results}
        assert "mcp_server" in names
        assert "agent_sdk" in names

    def test_no_match(self, populated_db):
        results = populated_db.search_concepts(["xyznothing"])
        assert len(results) == 0

    def test_empty_keywords(self, populated_db):
        results = populated_db.search_concepts([])
        assert len(results) == 0


class TestGetConceptEvents:
    def test_returns_joined_data(self, populated_db):
        c = populated_db.get_concept_by_name("mcp_server")
        events = populated_db.get_concept_events(c.id)
        assert len(events) == 3  # 2 commits + 1 conversation
        # Check joined fields present
        assert "relationship" in events[0]
        assert "confidence" in events[0]
        assert "title" in events[0]

    def test_respects_limit(self, populated_db):
        c = populated_db.get_concept_by_name("mcp_server")
        events = populated_db.get_concept_events(c.id, limit=1)
        assert len(events) == 1


class TestGetEventsForConcepts:
    def test_multi_concept(self, populated_db):
        c1 = populated_db.get_concept_by_name("mcp_server")
        c2 = populated_db.get_concept_by_name("agent_sdk")
        events = populated_db.get_events_for_concepts([c1.id, c2.id])
        assert len(events) >= 3
        # Check concept_name is included
        concept_names = {e["concept_name"] for e in events}
        assert "mcp_server" in concept_names

    def test_empty_ids(self, populated_db):
        events = populated_db.get_events_for_concepts([])
        assert events == []


class TestGetTimeline:
    def test_basic(self, populated_db):
        p = populated_db.get_project_by_name("sam_gov")
        events = populated_db.get_timeline(p.id)
        assert len(events) == 5
        # Should be chronological (ASC)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_since_filter(self, populated_db):
        p = populated_db.get_project_by_name("sam_gov")
        events = populated_db.get_timeline(p.id, since="2026-02-01")
        assert len(events) == 2

    def test_until_filter(self, populated_db):
        p = populated_db.get_project_by_name("sam_gov")
        events = populated_db.get_timeline(p.id, until="2026-01-20T23:59:59")
        assert len(events) == 2

    def test_min_significance(self, populated_db):
        p = populated_db.get_project_by_name("sam_gov")
        # Only 3 events are analyzed, with significance 0.5, 0.6, 0.7
        events = populated_db.get_timeline(p.id, min_significance=0.6)
        assert len(events) == 2


class TestRenameConcept:
    def test_success(self, populated_db):
        c = populated_db.get_concept_by_name("mcp_server")
        populated_db.rename_concept(c.id, "mcp_infrastructure")
        renamed = populated_db.get_concept_by_name("mcp_infrastructure")
        assert renamed is not None
        assert renamed.id == c.id

    def test_conflict_raises(self, populated_db):
        c = populated_db.get_concept_by_name("mcp_server")
        with pytest.raises(ValueError, match="already exists"):
            populated_db.rename_concept(c.id, "agent_sdk")


class TestMergeConcepts:
    def test_events_moved(self, populated_db):
        source = populated_db.get_concept_by_name("agent_sdk")
        target = populated_db.get_concept_by_name("mcp_server")
        moved = populated_db.merge_concepts(source.id, target.id)
        assert moved == 1
        # Source is now merged
        merged = populated_db.get_concept_by_name("agent_sdk")
        assert merged.status == "merged"
        assert merged.merged_into_id == target.id
        # Target now has more events
        target_events = populated_db.get_concept_events(target.id)
        assert len(target_events) == 4  # was 3, +1 from agent_sdk


class TestInsertCorrection:
    def test_audit_record(self, populated_db):
        cid = populated_db.insert_correction(
            correction_type="rename",
            target_type="concept",
            target_id=1,
            old_value="old_name",
            new_value="new_name",
            source_command="test",
        )
        assert cid > 0
        row = populated_db.conn.execute(
            "SELECT * FROM corrections WHERE id = ?", (cid,)
        ).fetchone()
        assert row["correction_type"] == "rename"
        assert row["old_value"] == "old_name"


class TestWorkSessions:
    def test_insert_and_get(self, tmp_db):
        db = tmp_db
        pid = db.upsert_project("proj", "/tmp/proj")
        sid = db.insert_work_session(
            project_id=pid,
            session_start="2026-01-15T10:00:00",
            session_end="2026-01-15T11:00:00",
            user_goal="Fix timeout bug",
            commit_hashes='["abc1234"]',
        )
        assert sid > 0

        sessions = db.get_sessions(project_id=pid)
        assert len(sessions) == 1
        assert sessions[0].user_goal == "Fix timeout bug"

    def test_insert_session_event(self, tmp_db):
        db = tmp_db
        pid = db.upsert_project("proj", "/tmp/proj")
        eid = db.insert_event(EventInsert(
            project_id=pid, event_type=EventType.COMMIT,
            source_id="git:aaa", timestamp="2026-01-15T10:00:00",
            author="brian", title="Test commit",
        ))
        sid = db.insert_work_session(
            project_id=pid,
            session_start="2026-01-15T10:00:00",
            session_end="2026-01-15T11:00:00",
        )
        db.insert_session_event(sid, eid, "commit")
        events = db.get_session_events(sid)
        assert len(events) == 1
        assert events[0].role == "commit"

    def test_session_event_dedup(self, tmp_db):
        db = tmp_db
        pid = db.upsert_project("proj", "/tmp/proj")
        eid = db.insert_event(EventInsert(
            project_id=pid, event_type=EventType.COMMIT,
            source_id="git:aaa", timestamp="2026-01-15T10:00:00",
            author="brian", title="Test commit",
        ))
        sid = db.insert_work_session(
            project_id=pid,
            session_start="2026-01-15T10:00:00",
            session_end="2026-01-15T11:00:00",
        )
        db.insert_session_event(sid, eid, "commit")
        db.insert_session_event(sid, eid, "commit")  # duplicate — should not raise
        events = db.get_session_events(sid)
        assert len(events) == 1

    def test_unanalyzed_sessions(self, tmp_db):
        db = tmp_db
        pid = db.upsert_project("proj", "/tmp/proj")
        db.insert_work_session(
            project_id=pid,
            session_start="2026-01-15T10:00:00",
            session_end="2026-01-15T11:00:00",
        )
        unanalyzed = db.get_unanalyzed_sessions(pid)
        assert len(unanalyzed) == 1

    def test_update_session_analysis(self, tmp_db):
        db = tmp_db
        pid = db.upsert_project("proj", "/tmp/proj")
        sid = db.insert_work_session(
            project_id=pid,
            session_start="2026-01-15T10:00:00",
            session_end="2026-01-15T11:00:00",
        )
        run_id = db.create_analysis_run(
            model="test", prompt_version="v1", project_id=pid,
            started_at="2026-01-15T12:00:00",
        )
        db.update_session_analysis(sid, "Summary", "feature", 0.8, run_id)
        db.conn.commit()

        unanalyzed = db.get_unanalyzed_sessions(pid)
        assert len(unanalyzed) == 0

        sessions = db.get_sessions(project_id=pid)
        assert sessions[0].llm_summary == "Summary"
        assert sessions[0].significance == 0.8


class TestBackfillEvent:
    def test_backfill_updates_existing(self, tmp_db):
        db = tmp_db
        pid = db.upsert_project("proj", "/tmp/proj")
        db.insert_event(EventInsert(
            project_id=pid, event_type=EventType.COMMIT,
            source_id="git:abc123", timestamp="2026-01-15T10:00:00",
            author="brian", title="Test commit",
        ))
        updated = db.backfill_event(
            "git:abc123",
            diff_summary="  MODIFY foo.py (+5/-2)",
            change_types='{"foo.py": "MODIFY"}',
        )
        assert updated is True
        db.conn.commit()

        row = db.conn.execute(
            "SELECT diff_summary, change_types FROM events WHERE source_id = 'git:abc123'"
        ).fetchone()
        assert row["diff_summary"] == "  MODIFY foo.py (+5/-2)"
        assert row["change_types"] == '{"foo.py": "MODIFY"}'

    def test_backfill_nonexistent_returns_false(self, tmp_db):
        updated = tmp_db.backfill_event("git:nonexistent", diff_summary="x")
        assert updated is False


class TestSetEventSession:
    def test_sets_session_id(self, tmp_db):
        db = tmp_db
        pid = db.upsert_project("proj", "/tmp/proj")
        eid = db.insert_event(EventInsert(
            project_id=pid, event_type=EventType.COMMIT,
            source_id="git:aaa", timestamp="2026-01-15T10:00:00",
            author="brian", title="Test",
        ))
        sid = db.insert_work_session(
            project_id=pid,
            session_start="2026-01-15T10:00:00",
            session_end="2026-01-15T11:00:00",
        )
        db.set_event_session(eid, sid)
        db.conn.commit()

        row = db.conn.execute(
            "SELECT session_id FROM events WHERE id = ?", (eid,)
        ).fetchone()
        assert row["session_id"] == sid
