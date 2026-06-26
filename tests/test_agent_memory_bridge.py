"""Tests for the trajectory → agent_memory bridge."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from trajectory.analysis.agent_memory_bridge import AgentMemoryBridge, BridgeSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bridge(tmp_path):
    """AgentMemoryBridge backed by a temp state DB."""
    return AgentMemoryBridge(tmp_path / "bridge_state.db")


def make_session(source_id: str = "claude:test-sess-001", significance: float = 0.85) -> dict:
    return {
        "source_id": source_id,
        "project": "sam_gov",
        "user_goal": "Investigate SAM.gov endpoint patterns",
        "avg_significance": significance,
        "timestamp": "2026-01-01T12:00:00",
        "decisions": ["Use REST over GraphQL for simplicity"],
        "concepts": ["entity-extraction", "search-pagination"],
    }


# ---------------------------------------------------------------------------
# is_synced / mark_synced
# ---------------------------------------------------------------------------


def test_is_synced_false_initially(bridge):
    assert not bridge.is_synced("nonexistent-session")


def test_mark_and_check_synced(bridge):
    bridge.mark_synced("sess-001", 0.85, "trajectory-session")
    assert bridge.is_synced("sess-001")
    assert not bridge.is_synced("sess-002")


def test_mark_synced_is_idempotent(bridge):
    bridge.mark_synced("sess-001", 0.85, "trajectory-session")
    bridge.mark_synced("sess-001", 0.90, "trajectory-session")  # should not raise
    assert bridge.is_synced("sess-001")


# ---------------------------------------------------------------------------
# sync_session
# ---------------------------------------------------------------------------


def test_sync_session_dry_run(bridge, capsys):
    sess = make_session()
    result = bridge.sync_session(sess, dry_run=True)
    assert result is True
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "sam_gov" in out
    # dry-run must NOT mark as synced
    assert not bridge.is_synced(sess["source_id"])


def test_sync_session_writes_and_marks_synced(bridge):
    sess = make_session()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = bridge.sync_session(sess)
    assert result is True
    assert bridge.is_synced(sess["source_id"])


def test_sync_session_skips_already_synced(bridge):
    sess = make_session()
    bridge.mark_synced(sess["source_id"], 0.85, "trajectory-session")
    with patch("subprocess.run") as mock_run:
        result = bridge.sync_session(sess)
    assert result is False
    mock_run.assert_not_called()


def test_sync_session_fails_loud_on_nonzero_exit(bridge):
    sess = make_session()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="agent-memory error")
        with pytest.raises(RuntimeError, match="agent-memory error"):
            bridge.sync_session(sess)


def test_sync_session_fails_loud_on_missing_binary(bridge):
    sess = make_session()
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(RuntimeError, match="agent-memory not found on PATH"):
            bridge.sync_session(sess)


def test_sync_session_body_format(bridge, capsys):
    sess = make_session()
    bridge.sync_session(sess, dry_run=True)
    out = capsys.readouterr().out
    assert "sam_gov" in out
    assert "2026-01-01" in out
    assert "Investigate SAM.gov" in out


# ---------------------------------------------------------------------------
# sync_all
# ---------------------------------------------------------------------------


def test_sync_all_writes_all(bridge):
    sessions = [make_session(f"claude:sess-{i}") for i in range(3)]
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        summary = bridge.sync_all(sessions)
    assert summary.written == 3
    assert summary.skipped == 0
    assert summary.failed == 0


def test_sync_all_skips_already_synced(bridge):
    sess = make_session()
    bridge.mark_synced(sess["source_id"], 0.85, "trajectory-session")
    summary = bridge.sync_all([sess])
    assert summary.skipped == 1
    assert summary.written == 0


def test_sync_all_respects_max_sessions(bridge):
    sessions = [make_session(f"claude:sess-{i}") for i in range(10)]
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        summary = bridge.sync_all(sessions, max_sessions=3)
    assert summary.written == 3


def test_sync_all_continues_on_failure(bridge):
    sessions = [make_session(f"claude:sess-{i}") for i in range(3)]
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return MagicMock(returncode=1, stderr="transient error")
        return MagicMock(returncode=0, stderr="")

    with patch("subprocess.run", side_effect=side_effect):
        summary = bridge.sync_all(sessions)

    assert summary.written == 2
    assert summary.failed == 1
    assert len(summary.errors) == 1


def test_sync_all_dry_run_writes_none(bridge, capsys):
    sessions = [make_session(f"claude:sess-{i}") for i in range(3)]
    summary = bridge.sync_all(sessions, dry_run=True)
    assert summary.written == 3  # "would write"
    # Verify none are actually marked synced
    for sess in sessions:
        assert not bridge.is_synced(sess["source_id"])


# ---------------------------------------------------------------------------
# DB integration — get_sessions_for_bridge (uses populated_db fixture)
# ---------------------------------------------------------------------------


def test_get_sessions_for_bridge_threshold(populated_db):
    """Sessions below threshold must not appear."""
    sessions = populated_db.get_sessions_for_bridge(min_significance=0.7)
    for s in sessions:
        assert s["avg_significance"] >= 0.7
        assert "source_id" in s
        assert "project" in s
        assert "user_goal" in s
        assert isinstance(s["decisions"], list)
        assert isinstance(s["concepts"], list)


def test_get_sessions_for_bridge_empty_when_none_analyzed(tmp_db):
    """Returns empty list when no sessions have significance set."""
    sessions = tmp_db.get_sessions_for_bridge(min_significance=0.5)
    assert sessions == []
