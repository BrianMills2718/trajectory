"""Tests for enriched git extractor — diff summary and change type generation."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from trajectory.extractors.git_extractor import _build_diff_summary


class FakeChangeType:
    def __init__(self, name: str):
        self.name = name


class FakeModifiedFile:
    def __init__(
        self,
        new_path: str,
        old_path: str | None = None,
        change_type_name: str = "MODIFY",
        added_lines: int = 5,
        deleted_lines: int = 2,
        diff_added: list | None = None,
    ):
        self.new_path = new_path
        self.old_path = old_path or new_path
        self.change_type = FakeChangeType(change_type_name)
        self.added_lines = added_lines
        self.deleted_lines = deleted_lines
        self.diff_parsed = {
            "added": diff_added if diff_added is not None else [
                (1, "line one"),
                (2, "line two"),
                (3, "line three"),
            ],
            "deleted": [],
        }


class TestBuildDiffSummary:
    def test_basic_modify(self):
        commit = SimpleNamespace(modified_files=[
            FakeModifiedFile("src/main.py", change_type_name="MODIFY", added_lines=10, deleted_lines=3),
        ])
        summary, types = _build_diff_summary(commit, max_lines=5)
        assert "MODIFY src/main.py (+10/-3)" in summary
        assert types == {"src/main.py": "MODIFY"}

    def test_add_file(self):
        commit = SimpleNamespace(modified_files=[
            FakeModifiedFile(
                "src/new_file.py", change_type_name="ADD", added_lines=30, deleted_lines=0,
                diff_added=[(1, "import os"), (2, "def hello():"), (3, "    pass")],
            ),
        ])
        summary, types = _build_diff_summary(commit, max_lines=2)
        assert "ADD src/new_file.py (+30/-0)" in summary
        assert "+ import os" in summary
        assert "+ def hello():" in summary
        # Line 3 should NOT be included (max_lines=2)
        assert "+ pass" not in summary.split("... (")[0]  # before the "more" marker
        assert types == {"src/new_file.py": "ADD"}

    def test_multiple_files(self):
        commit = SimpleNamespace(modified_files=[
            FakeModifiedFile("a.py", change_type_name="MODIFY", added_lines=5, deleted_lines=1),
            FakeModifiedFile("b.py", change_type_name="DELETE", added_lines=0, deleted_lines=20),
        ])
        summary, types = _build_diff_summary(commit, max_lines=5)
        assert "a.py" in summary
        assert "b.py" in summary
        assert types == {"a.py": "MODIFY", "b.py": "DELETE"}

    def test_empty_commit(self):
        commit = SimpleNamespace(modified_files=[])
        summary, types = _build_diff_summary(commit, max_lines=5)
        assert summary == ""
        assert types == {}

    def test_max_lines_zero_skips_preview(self):
        commit = SimpleNamespace(modified_files=[
            FakeModifiedFile("x.py", diff_added=[(1, "code")]),
        ])
        summary, types = _build_diff_summary(commit, max_lines=0)
        assert "MODIFY x.py" in summary
        assert "+ code" not in summary

    def test_file_error_skipped(self):
        """A file that raises during diff parsing shouldn't kill the whole commit."""
        good_file = FakeModifiedFile("good.py")
        bad_file = FakeModifiedFile("bad.py")
        # Make bad_file.change_type raise
        bad_file.change_type = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        # Since change_type is accessed as attribute, make it raise via descriptor
        class Exploder:
            @property
            def name(self):
                raise RuntimeError("boom")
        bad_file.change_type = Exploder()

        commit = SimpleNamespace(modified_files=[bad_file, good_file])
        summary, types = _build_diff_summary(commit, max_lines=5)
        # good_file should still be in the results
        assert "good.py" in summary
        assert "good.py" in types
