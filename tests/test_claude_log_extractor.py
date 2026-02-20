"""Tests for enriched Claude log extractor — commit hash extraction, file tracking."""

import json
import pytest
from pathlib import Path

from trajectory.extractors.claude_log_extractor import ClaudeLogExtractor, _tool_brief


@pytest.fixture()
def session_jsonl(tmp_path):
    """Create a minimal JSONL session file with tool calls and git commits."""
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    (project_dir / ".git").mkdir()

    # Build the log dir key from the actual project path (matches what find_log_dir expects)
    from trajectory.extractors.claude_log_extractor import project_path_to_log_key
    log_key = project_path_to_log_key(project_dir)
    log_dir = tmp_path / "claude_logs" / log_key
    log_dir.mkdir(parents=True)

    messages = [
        # User message
        {
            "type": "user",
            "timestamp": "2026-01-15T10:00:00Z",
            "message": {"content": "Fix the MCP server timeout bug"},
            "gitBranch": "main",
        },
        # Assistant with tool calls
        {
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "text", "text": "I'll fix the timeout by adding retry logic."},
                    {
                        "type": "tool_use",
                        "id": "tu_read1",
                        "name": "Read",
                        "input": {"file_path": "/tmp/myproject/server.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_edit1",
                        "name": "Edit",
                        "input": {"file_path": "/tmp/myproject/server.py", "old_string": "x", "new_string": "y"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_write1",
                        "name": "Write",
                        "input": {"file_path": "/tmp/myproject/retry.py", "content": "..."},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_bash1",
                        "name": "Bash",
                        "input": {"command": "git add -A && git commit -m 'Fix timeout'"},
                    },
                ],
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
        },
        # User message with tool results
        {
            "type": "user",
            "timestamp": "2026-01-15T10:01:00Z",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_read1", "content": "file content..."},
                    {"type": "tool_result", "tool_use_id": "tu_edit1", "content": "edited"},
                    {"type": "tool_result", "tool_use_id": "tu_write1", "content": "written"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_bash1",
                        "content": "[main abc1234] Fix timeout\n 1 file changed, 5 insertions(+)",
                    },
                ],
            },
        },
        # Second assistant turn with more text
        {
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "text", "text": "The fix is applied. Let me also add retry."},
                    {
                        "type": "tool_use",
                        "id": "tu_bash2",
                        "name": "Bash",
                        "input": {"command": "git commit -m 'Add retry'"},
                    },
                ],
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
        },
        # Tool result with second commit
        {
            "type": "user",
            "timestamp": "2026-01-15T10:02:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_bash2",
                        "content": "[main def5678] Add retry\n 1 file changed, 30 insertions(+)",
                    },
                ],
            },
        },
    ]

    jsonl_path = log_dir / "test-session-001.jsonl"
    with jsonl_path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")

    claude_logs_dir = tmp_path / "claude_logs"
    return project_dir, claude_logs_dir


class TestCommitHashExtraction:
    def test_extracts_commit_hashes(self, session_jsonl):
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()
        assert len(events) == 1

        raw = events[0].raw_data
        assert raw is not None
        assert "abc1234" in raw["commit_hashes"]
        assert "def5678" in raw["commit_hashes"]

    def test_no_duplicate_hashes(self, session_jsonl):
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()
        hashes = events[0].raw_data["commit_hashes"]
        assert len(hashes) == len(set(hashes))


class TestFileTracking:
    def test_tracks_modified_files(self, session_jsonl):
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()
        raw = events[0].raw_data

        modified = raw["files_modified"]
        assert "/tmp/myproject/server.py" in modified
        assert "/tmp/myproject/retry.py" in modified

    def test_tracks_examined_files(self, session_jsonl):
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()
        raw = events[0].raw_data

        examined = raw["files_examined"]
        assert "/tmp/myproject/server.py" in examined

    def test_files_changed_includes_modified(self, session_jsonl):
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()

        # files_changed should be the modified files (Edit/Write targets)
        assert events[0].files_changed is not None
        assert "/tmp/myproject/server.py" in events[0].files_changed
        assert "/tmp/myproject/retry.py" in events[0].files_changed


class TestToolSequence:
    def test_captures_tool_sequence(self, session_jsonl):
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()
        raw = events[0].raw_data

        seq = raw["tool_sequence"]
        tool_names = [t[0] for t in seq]
        assert "Read" in tool_names
        assert "Edit" in tool_names
        assert "Write" in tool_names
        assert "Bash" in tool_names


class TestAssistantReasoning:
    def test_captures_reasoning(self, session_jsonl):
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()
        raw = events[0].raw_data

        reasoning = raw["assistant_reasoning"]
        assert "timeout" in reasoning.lower()
        assert "retry" in reasoning.lower()

    def test_reasoning_truncated(self, tmp_path):
        """Reasoning exceeding max_reasoning_chars should be truncated."""
        project_dir = tmp_path / "proj2"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        from trajectory.extractors.claude_log_extractor import project_path_to_log_key
        log_key = project_path_to_log_key(project_dir)
        log_dir = tmp_path / "claude_logs" / log_key
        log_dir.mkdir(parents=True)

        long_text = "x" * 5000
        messages = [
            {
                "type": "user",
                "timestamp": "2026-01-15T10:00:00Z",
                "message": {"content": "Do something"},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": long_text}],
                    "usage": {"input_tokens": 100, "output_tokens": 100},
                },
            },
        ]
        jsonl_path = log_dir / "long-session.jsonl"
        with jsonl_path.open("w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        claude_logs_dir = tmp_path / "claude_logs"
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir, max_reasoning_chars=100)
        events = extractor.extract()
        assert len(events) == 1
        reasoning = events[0].raw_data["assistant_reasoning"]
        assert len(reasoning) < 200  # truncated + marker
        assert "[truncated]" in reasoning


class TestCatchallDirectory:
    """Test extraction from catch-all parent directories."""

    def test_finds_sessions_in_catchall(self, tmp_path):
        """Sessions in parent catch-all dir that reference the project should be found."""
        from trajectory.extractors.claude_log_extractor import project_path_to_log_key

        # Create project at /tmp/.../parent/myproject
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        project_dir = parent_dir / "myproject"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        claude_logs_dir = tmp_path / "claude_logs"

        # Create catch-all dir for parent (NOT the project-specific one)
        catchall_key = project_path_to_log_key(parent_dir)
        catchall_dir = claude_logs_dir / catchall_key
        catchall_dir.mkdir(parents=True)

        # Session that references the project path in a tool call
        messages = [
            {
                "type": "user",
                "timestamp": "2026-01-15T10:00:00Z",
                "message": {"content": "Fix the bug"},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "I'll fix it."},
                        {
                            "type": "tool_use",
                            "id": "tu_read1",
                            "name": "Read",
                            "input": {"file_path": f"{project_dir}/server.py"},
                        },
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        ]
        jsonl_path = catchall_dir / "catchall-session-001.jsonl"
        with jsonl_path.open("w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")

        # Session that does NOT reference the project
        other_messages = [
            {
                "type": "user",
                "timestamp": "2026-01-15T11:00:00Z",
                "message": {"content": "Other work"},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Done."}],
                    "usage": {"input_tokens": 50, "output_tokens": 25},
                },
            },
        ]
        other_path = catchall_dir / "catchall-session-002.jsonl"
        with other_path.open("w") as f:
            for msg in other_messages:
                f.write(json.dumps(msg) + "\n")

        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()

        # Should find the session that references our project, not the other one
        assert len(events) == 1
        assert events[0].raw_data["session_id"] == "catchall-session-001"

    def test_deduplicates_across_project_and_catchall(self, tmp_path):
        """Same session in both project dir and catch-all should only appear once."""
        from trajectory.extractors.claude_log_extractor import project_path_to_log_key

        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        project_dir = parent_dir / "myproject"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        claude_logs_dir = tmp_path / "claude_logs"

        messages = [
            {
                "type": "user",
                "timestamp": "2026-01-15T10:00:00Z",
                "message": {"content": "Fix the bug"},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Fixed."},
                        {
                            "type": "tool_use",
                            "id": "tu_r1",
                            "name": "Read",
                            "input": {"file_path": f"{project_dir}/main.py"},
                        },
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        ]

        # Put same session in BOTH project-specific dir AND catch-all
        project_log_key = project_path_to_log_key(project_dir)
        project_log_dir = claude_logs_dir / project_log_key
        project_log_dir.mkdir(parents=True)

        catchall_key = project_path_to_log_key(parent_dir)
        catchall_dir = claude_logs_dir / catchall_key
        catchall_dir.mkdir(parents=True)

        for d in (project_log_dir, catchall_dir):
            jsonl_path = d / "shared-session.jsonl"
            with jsonl_path.open("w") as f:
                for msg in messages:
                    f.write(json.dumps(msg) + "\n")

        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()

        # Should appear exactly once despite being in both dirs
        assert len(events) == 1
        assert events[0].raw_data["session_id"] == "shared-session"

    def test_no_catchall_when_no_parent_dir_exists(self, session_jsonl):
        """When no catch-all parent dir exists, should still work normally."""
        project_dir, claude_logs_dir = session_jsonl
        extractor = ClaudeLogExtractor(project_dir, claude_logs_dir=claude_logs_dir)
        events = extractor.extract()
        # Should still find the project-specific session
        assert len(events) == 1


class TestToolBrief:
    def test_edit(self):
        assert _tool_brief("Edit", {"file_path": "/foo/bar/baz.py"}) == "baz.py"

    def test_bash(self):
        assert _tool_brief("Bash", {"command": "git status"}) == "git status"

    def test_glob(self):
        assert _tool_brief("Glob", {"pattern": "**/*.py"}) == "**/*.py"

    def test_unknown(self):
        assert _tool_brief("CustomTool", {}) == ""
