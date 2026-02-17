"""SQLite database setup and operations for trajectory tracker."""

import json
import logging
import sqlite3
from pathlib import Path

from trajectory.config import Config
from trajectory.models import (
    ConceptRow,
    DecisionRow,
    EventInsert,
    EventRow,
    ExtractedEvent,
    ProjectRow,
)

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    git_remote TEXT,
    description TEXT,
    total_commits INTEGER DEFAULT 0,
    total_conversations INTEGER DEFAULT 0,
    last_ingested TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    event_type TEXT NOT NULL,
    source_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    author TEXT,
    title TEXT NOT NULL,
    body TEXT,
    raw_data TEXT,
    files_changed TEXT,
    git_branch TEXT,
    llm_summary TEXT,
    llm_intent TEXT,
    significance REAL,
    analysis_run_id INTEGER REFERENCES analysis_runs(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    first_seen TEXT,
    last_seen TEXT,
    status TEXT DEFAULT 'active',
    parent_concept_id INTEGER REFERENCES concepts(id),
    merged_into_id INTEGER REFERENCES concepts(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concept_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL REFERENCES concepts(id),
    event_id INTEGER NOT NULL REFERENCES events(id),
    relationship TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    reasoning TEXT,
    analysis_run_id INTEGER REFERENCES analysis_runs(id),
    UNIQUE(concept_id, event_id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES events(id),
    project_id INTEGER NOT NULL REFERENCES projects(id),
    title TEXT NOT NULL,
    reasoning TEXT,
    alternatives TEXT,
    outcome TEXT,
    decision_type TEXT,
    analysis_run_id INTEGER REFERENCES analysis_runs(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concept_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_a_id INTEGER NOT NULL REFERENCES concepts(id),
    concept_b_id INTEGER NOT NULL REFERENCES concepts(id),
    relationship TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    evidence TEXT,
    UNIQUE(concept_a_id, concept_b_id, relationship)
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    prompt_version TEXT,
    project_id INTEGER REFERENCES projects(id),
    events_processed INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correction_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id INTEGER NOT NULL,
    old_value TEXT,
    new_value TEXT,
    source_command TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_type TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    content TEXT NOT NULL,
    delivered_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_project_ts ON events(project_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_significance ON events(significance);
CREATE INDEX IF NOT EXISTS idx_concept_events_concept ON concept_events(concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_events_event ON concept_events(event_id);
CREATE INDEX IF NOT EXISTS idx_concepts_status ON concepts(status);
CREATE INDEX IF NOT EXISTS idx_corrections_target ON corrections(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_digests_type_period ON digests(digest_type, period_start);
"""


class TrajectoryDB:
    """SQLite database wrapper for trajectory tracker."""

    def __init__(self, config: Config) -> None:
        self.db_path = config.resolved_db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        return self._conn

    def init_db(self) -> None:
        """Create database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        logger.info("Database initialized at %s", self.db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Project operations ---

    def upsert_project(
        self,
        name: str,
        path: str,
        git_remote: str | None = None,
        description: str | None = None,
    ) -> int:
        """Insert or update a project. Returns project ID."""
        row = self.conn.execute(
            "SELECT id FROM projects WHERE path = ?", (path,)
        ).fetchone()
        if row:
            self.conn.execute(
                "UPDATE projects SET name = ?, git_remote = ?, description = ? WHERE id = ?",
                (name, git_remote, description, row["id"]),
            )
            self.conn.commit()
            return row["id"]

        cursor = self.conn.execute(
            "INSERT INTO projects (name, path, git_remote, description) VALUES (?, ?, ?, ?)",
            (name, path, git_remote, description),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_project(self, project_id: int) -> ProjectRow | None:
        row = self.conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row:
            return ProjectRow(**dict(row))
        return None

    def get_project_by_path(self, path: str) -> ProjectRow | None:
        row = self.conn.execute(
            "SELECT * FROM projects WHERE path = ?", (path,)
        ).fetchone()
        if row:
            return ProjectRow(**dict(row))
        return None

    def list_projects(self) -> list[ProjectRow]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY name").fetchall()
        return [ProjectRow(**dict(r)) for r in rows]

    def update_project_stats(
        self,
        project_id: int,
        total_commits: int | None = None,
        total_conversations: int | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[int] = []
        if total_commits is not None:
            updates.append("total_commits = ?")
            params.append(total_commits)
        if total_conversations is not None:
            updates.append("total_conversations = ?")
            params.append(total_conversations)
        if not updates:
            return
        params.append(project_id)
        self.conn.execute(
            f"UPDATE projects SET {', '.join(updates)} WHERE id = ?", params
        )
        self.conn.commit()

    def update_last_ingested(self, project_id: int, timestamp: str) -> None:
        self.conn.execute(
            "UPDATE projects SET last_ingested = ? WHERE id = ?",
            (timestamp, project_id),
        )
        self.conn.commit()

    # --- Event operations ---

    def insert_event(self, event: EventInsert) -> int | None:
        """Insert event, skip if source_id already exists. Returns event ID or None."""
        try:
            cursor = self.conn.execute(
                """INSERT INTO events
                   (project_id, event_type, source_id, timestamp, author, title, body, raw_data, files_changed, git_branch)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.project_id,
                    event.event_type,
                    event.source_id,
                    event.timestamp,
                    event.author,
                    event.title,
                    event.body,
                    event.raw_data,
                    event.files_changed,
                    event.git_branch,
                ),
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # source_id already exists — dedup
            return None

    def insert_events_batch(self, events: list[EventInsert]) -> int:
        """Insert batch of events, skipping duplicates. Returns count of new events."""
        new_count = 0
        for event in events:
            if self.insert_event(event) is not None:
                new_count += 1
        self.conn.commit()
        return new_count

    def get_events(
        self,
        project_id: int | None = None,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[EventRow]:
        clauses: list[str] = []
        params: list[str | int] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [EventRow(**dict(r)) for r in rows]

    def count_events(self, project_id: int, event_type: str | None = None) -> int:
        if event_type:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE project_id = ? AND event_type = ?",
                (project_id, event_type),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def get_unanalyzed_events(
        self,
        project_id: int,
        limit: int = 1000,
    ) -> list[EventRow]:
        """Get events that haven't been analyzed yet."""
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id = ? AND analysis_run_id IS NULL ORDER BY timestamp LIMIT ?",
            (project_id, limit),
        ).fetchall()
        return [EventRow(**dict(r)) for r in rows]

    # --- Analysis operations ---

    def create_analysis_run(
        self,
        model: str,
        prompt_version: str,
        project_id: int,
        started_at: str,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO analysis_runs (model, prompt_version, project_id, started_at) VALUES (?, ?, ?, ?)",
            (model, prompt_version, project_id, started_at),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def update_analysis_run(
        self,
        run_id: int,
        *,
        events_processed: int | None = None,
        cost_usd: float | None = None,
        completed_at: str | None = None,
        status: str | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[str | int | float] = []
        if events_processed is not None:
            updates.append("events_processed = ?")
            params.append(events_processed)
        if cost_usd is not None:
            updates.append("cost_usd = ?")
            params.append(cost_usd)
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if not updates:
            return
        params.append(run_id)
        self.conn.execute(
            f"UPDATE analysis_runs SET {', '.join(updates)} WHERE id = ?", params
        )
        self.conn.commit()

    def update_event_analysis(
        self,
        event_id: int,
        llm_summary: str,
        llm_intent: str,
        significance: float,
        analysis_run_id: int,
    ) -> None:
        self.conn.execute(
            """UPDATE events SET llm_summary = ?, llm_intent = ?, significance = ?, analysis_run_id = ?
               WHERE id = ?""",
            (llm_summary, llm_intent, significance, analysis_run_id, event_id),
        )

    def upsert_concept(
        self,
        name: str,
        description: str | None = None,
        first_seen: str | None = None,
        last_seen: str | None = None,
    ) -> int:
        """Insert or update a concept. Returns concept ID."""
        row = self.conn.execute(
            "SELECT id, first_seen FROM concepts WHERE name = ?", (name,)
        ).fetchone()
        if row:
            updates = ["updated_at = datetime('now')"]
            params: list[str | int] = []
            if description:
                updates.append("description = ?")
                params.append(description)
            if last_seen:
                updates.append("last_seen = ?")
                params.append(last_seen)
            # Only update first_seen if earlier
            if first_seen and (not row["first_seen"] or first_seen < row["first_seen"]):
                updates.append("first_seen = ?")
                params.append(first_seen)
            params.append(row["id"])
            self.conn.execute(
                f"UPDATE concepts SET {', '.join(updates)} WHERE id = ?", params
            )
            return row["id"]

        cursor = self.conn.execute(
            "INSERT INTO concepts (name, description, first_seen, last_seen) VALUES (?, ?, ?, ?)",
            (name, description, first_seen, last_seen),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def link_concept_event(
        self,
        concept_id: int,
        event_id: int,
        relationship: str,
        confidence: float,
        reasoning: str | None = None,
        analysis_run_id: int | None = None,
    ) -> None:
        """Link a concept to an event. Skips if already linked."""
        try:
            self.conn.execute(
                """INSERT INTO concept_events (concept_id, event_id, relationship, confidence, reasoning, analysis_run_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (concept_id, event_id, relationship, confidence, reasoning, analysis_run_id),
            )
        except sqlite3.IntegrityError:
            pass  # already linked

    def insert_decision(
        self,
        event_id: int,
        project_id: int,
        title: str,
        reasoning: str | None = None,
        alternatives: str | None = None,
        decision_type: str | None = None,
        analysis_run_id: int | None = None,
    ) -> int:
        cursor = self.conn.execute(
            """INSERT INTO decisions (event_id, project_id, title, reasoning, alternatives, decision_type, analysis_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event_id, project_id, title, reasoning, alternatives, decision_type, analysis_run_id),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    # --- Query operations ---

    def get_project_by_name(self, name: str) -> ProjectRow | None:
        """Case-insensitive project lookup by name."""
        row = self.conn.execute(
            "SELECT * FROM projects WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        if row:
            return ProjectRow(**dict(row))
        return None

    def get_concept_by_name(self, name: str) -> ConceptRow | None:
        """Case-insensitive concept lookup by name."""
        row = self.conn.execute(
            "SELECT * FROM concepts WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        if row:
            return ConceptRow(**dict(row))
        return None

    def list_concepts(
        self,
        status: str | None = None,
        project_id: int | None = None,
    ) -> list[ConceptRow]:
        """List concepts with optional filters."""
        if project_id is not None:
            # Join through concept_events → events to find concepts for a project
            clauses = ["e.project_id = ?"]
            params: list[str | int] = [project_id]
            if status is not None:
                clauses.append("c.status = ?")
                params.append(status)
            rows = self.conn.execute(
                f"""SELECT DISTINCT c.* FROM concepts c
                    JOIN concept_events ce ON c.id = ce.concept_id
                    JOIN events e ON ce.event_id = e.id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY c.name""",
                params,
            ).fetchall()
        else:
            if status is not None:
                rows = self.conn.execute(
                    "SELECT * FROM concepts WHERE status = ? ORDER BY name",
                    (status,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM concepts ORDER BY name"
                ).fetchall()
        return [ConceptRow(**dict(r)) for r in rows]

    def search_concepts(self, keywords: list[str]) -> list[ConceptRow]:
        """Search concepts by LIKE matching on names."""
        if not keywords:
            return []
        clauses = ["LOWER(name) LIKE ?"] * len(keywords)
        params = [f"%{kw.lower()}%" for kw in keywords]
        rows = self.conn.execute(
            f"SELECT * FROM concepts WHERE {' OR '.join(clauses)} ORDER BY name",
            params,
        ).fetchall()
        return [ConceptRow(**dict(r)) for r in rows]

    def get_concept_events(
        self, concept_id: int, limit: int = 50
    ) -> list[dict[str, object]]:
        """Get events linked to a concept with relationship metadata."""
        rows = self.conn.execute(
            """SELECT ce.relationship, ce.confidence, ce.reasoning AS ce_reasoning,
                      e.*
               FROM concept_events ce
               JOIN events e ON ce.event_id = e.id
               WHERE ce.concept_id = ?
               ORDER BY e.timestamp DESC
               LIMIT ?""",
            (concept_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_events_for_concepts(
        self, concept_ids: list[int], limit: int = 100
    ) -> list[dict[str, object]]:
        """Get events for multiple concepts, including concept name."""
        if not concept_ids:
            return []
        placeholders = ",".join("?" * len(concept_ids))
        rows = self.conn.execute(
            f"""SELECT c.name AS concept_name, ce.relationship, ce.confidence,
                       e.*
                FROM concept_events ce
                JOIN events e ON ce.event_id = e.id
                JOIN concepts c ON ce.concept_id = c.id
                WHERE ce.concept_id IN ({placeholders})
                ORDER BY e.timestamp DESC
                LIMIT ?""",
            [*concept_ids, limit],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_timeline(
        self,
        project_id: int,
        since: str | None = None,
        until: str | None = None,
        min_significance: float | None = None,
        limit: int = 200,
    ) -> list[EventRow]:
        """Get chronological events for a project with optional filters."""
        clauses = ["project_id = ?"]
        params: list[str | int | float] = [project_id]
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        if min_significance is not None:
            clauses.append("significance >= ?")
            params.append(min_significance)
        rows = self.conn.execute(
            f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY timestamp ASC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [EventRow(**dict(r)) for r in rows]

    def get_decisions(
        self, project_id: int | None = None, limit: int = 50
    ) -> list[DecisionRow]:
        """Get decisions, optionally filtered by project."""
        if project_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM decisions WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [DecisionRow(**dict(r)) for r in rows]

    # --- Correction operations ---

    def rename_concept(self, concept_id: int, new_name: str) -> None:
        """Rename a concept. Raises ValueError if new name already exists."""
        existing = self.conn.execute(
            "SELECT id FROM concepts WHERE LOWER(name) = LOWER(?) AND id != ?",
            (new_name, concept_id),
        ).fetchone()
        if existing:
            raise ValueError(f"Concept name '{new_name}' already exists")
        self.conn.execute(
            "UPDATE concepts SET name = ?, updated_at = datetime('now') WHERE id = ?",
            (new_name, concept_id),
        )
        self.conn.commit()

    def merge_concepts(self, source_id: int, target_id: int) -> int:
        """Merge source concept into target. Returns count of events moved."""
        # Move concept_events from source to target, skip duplicates
        rows = self.conn.execute(
            "SELECT event_id FROM concept_events WHERE concept_id = ?",
            (source_id,),
        ).fetchall()
        moved = 0
        for row in rows:
            try:
                self.conn.execute(
                    "UPDATE concept_events SET concept_id = ? WHERE concept_id = ? AND event_id = ?",
                    (target_id, source_id, row["event_id"]),
                )
                moved += 1
            except sqlite3.IntegrityError:
                # Target already has this event — delete the source link
                self.conn.execute(
                    "DELETE FROM concept_events WHERE concept_id = ? AND event_id = ?",
                    (source_id, row["event_id"]),
                )
        # Mark source as merged
        self.conn.execute(
            "UPDATE concepts SET status = 'merged', merged_into_id = ?, updated_at = datetime('now') WHERE id = ?",
            (target_id, source_id),
        )
        self.conn.commit()
        return moved

    def update_concept_status(self, concept_id: int, status: str) -> None:
        """Update concept lifecycle status."""
        self.conn.execute(
            "UPDATE concepts SET status = ?, updated_at = datetime('now') WHERE id = ?",
            (status, concept_id),
        )
        self.conn.commit()

    def insert_correction(
        self,
        correction_type: str,
        target_type: str,
        target_id: int,
        old_value: str | None = None,
        new_value: str | None = None,
        source_command: str | None = None,
    ) -> int:
        """Insert a correction audit record. Returns correction ID."""
        cursor = self.conn.execute(
            """INSERT INTO corrections (correction_type, target_type, target_id, old_value, new_value, source_command)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (correction_type, target_type, target_id, old_value, new_value, source_command),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # --- Helpers ---

    def event_exists(self, source_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM events WHERE source_id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def extracted_to_insert(
        self, extracted: ExtractedEvent, project_id: int
    ) -> EventInsert:
        """Convert an ExtractedEvent to an EventInsert."""
        return EventInsert(
            project_id=project_id,
            event_type=extracted.event_type,
            source_id=extracted.source_id,
            timestamp=extracted.timestamp.isoformat(),
            author=extracted.author,
            title=extracted.title,
            body=extracted.body,
            raw_data=json.dumps(extracted.raw_data) if extracted.raw_data else None,
            files_changed=json.dumps(extracted.files_changed) if extracted.files_changed else None,
            git_branch=extracted.git_branch,
        )
