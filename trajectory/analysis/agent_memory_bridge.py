"""Bridge that syncs high-significance trajectory sessions into agent_memory."""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BridgeSummary:
    written: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class AgentMemoryBridge:
    """Syncs trajectory sessions to agent_memory. Idempotent via bridge_state.db."""

    def __init__(self, state_db_path: Path) -> None:
        self.state_db_path = state_db_path
        self._state_conn = self._init_state_db()

    def _init_state_db(self) -> sqlite3.Connection:
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.state_db_path))
        conn.execute(
            """CREATE TABLE IF NOT EXISTS synced_sessions (
               source_id TEXT PRIMARY KEY,
               synced_at TEXT NOT NULL,
               significance REAL,
               category TEXT
            )"""
        )
        conn.commit()
        return conn

    def is_synced(self, source_id: str) -> bool:
        """Return True if this source_id was already synced."""
        row = self._state_conn.execute(
            "SELECT 1 FROM synced_sessions WHERE source_id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def mark_synced(self, source_id: str, significance: float, category: str) -> None:
        """Record a successful sync so re-runs skip it."""
        from datetime import datetime, timezone

        self._state_conn.execute(
            "INSERT OR REPLACE INTO synced_sessions (source_id, synced_at, significance, category) VALUES (?, ?, ?, ?)",
            (source_id, datetime.now(timezone.utc).isoformat(), significance, category),
        )
        self._state_conn.commit()

    def sync_session(self, session: dict, dry_run: bool = False) -> bool:
        """Sync one session. Returns True if written (or would write in dry-run).

        Raises RuntimeError on subprocess failure — caller decides whether to abort.
        """
        source_id = session["source_id"]
        if self.is_synced(source_id):
            return False

        project = session["project"]
        goal = session["user_goal"]
        significance = session["avg_significance"]
        timestamp = session["timestamp"]
        decisions = session["decisions"]
        concepts = session["concepts"]

        decisions_str = "; ".join(decisions[:5]) if decisions else "none recorded"
        concepts_str = ", ".join(concepts[:10]) if concepts else "none"
        body = (
            f"Session [{project}] {timestamp[:10]}: {goal}. "
            f"Decisions: {decisions_str}. "
            f"Concepts: {concepts_str}."
        )

        if dry_run:
            print(f"[dry-run] Would write: {body[:120]}...")
            return True

        try:
            result = subprocess.run(
                [
                    "agent-memory", "store-finding", body,
                    "--category", "trajectory-session",
                    "--tags", f"{project},trajectory",
                    "--project", project,
                    "--agent", "trajectory-bridge",
                    "--active",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "agent-memory returned non-zero exit")
            self.mark_synced(source_id, significance, "trajectory-session")
            return True
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"agent-memory timed out for session {source_id}") from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "agent-memory not found on PATH. Install agent_memory or run: pip install agent-memory"
            ) from exc

    def sync_all(
        self,
        sessions: list[dict],
        max_sessions: int = 50,
        dry_run: bool = False,
    ) -> BridgeSummary:
        """Sync up to max_sessions sessions. Returns a summary of results."""
        summary = BridgeSummary()
        for session in sessions[:max_sessions]:
            source_id = session["source_id"]
            if self.is_synced(source_id):
                summary.skipped += 1
                continue
            try:
                wrote = self.sync_session(session, dry_run=dry_run)
                if wrote:
                    summary.written += 1
                else:
                    summary.skipped += 1
            except Exception as exc:
                summary.failed += 1
                summary.errors.append(str(exc))
                print(f"ERROR syncing {source_id}: {exc}", file=sys.stderr)
        return summary
