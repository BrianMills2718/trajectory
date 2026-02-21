"""Deterministic concept activity materializer — rollup tables, importance, lifecycle.

No LLM calls. Aggregates from concept_events + events tables.
Populates: concept_activity, concepts.importance, concepts.lifecycle.
"""

import logging
import math
from datetime import datetime

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)


class ConceptRollupResult:
    """Summary of concept rollup."""

    def __init__(self) -> None:
        self.activity_rows = 0
        self.concepts_scored = 0
        self.lifecycle_stages: dict[str, int] = {}

    def __repr__(self) -> str:
        stages = ", ".join(f"{k}={v}" for k, v in sorted(self.lifecycle_stages.items()))
        return (
            f"ConceptRollupResult({self.activity_rows} activity rows, "
            f"{self.concepts_scored} concepts scored, stages: [{stages}])"
        )


def _compute_lifecycle(
    first_seen: str | None,
    last_seen: str | None,
    event_count: int,
    months_active: int,
    now: datetime,
) -> str:
    """Determine lifecycle stage from activity pattern.

    Stages:
    - emerging: first seen within last 2 months, low event count
    - growing: active in recent months, increasing activity
    - stable: active for 3+ months, consistent activity
    - declining: last seen > 2 months ago but was once active
    - dormant: last seen > 4 months ago
    """
    if not first_seen or not last_seen:
        return "emerging"

    try:
        first = datetime.fromisoformat(first_seen)
        last = datetime.fromisoformat(last_seen)
    except (ValueError, TypeError):
        return "emerging"

    months_since_first = (now.year - first.year) * 12 + (now.month - first.month)
    months_since_last = (now.year - last.year) * 12 + (now.month - last.month)

    if months_since_last > 4:
        return "dormant"
    if months_since_last > 2:
        return "declining"
    if months_since_first <= 2 and event_count < 10:
        return "emerging"
    if months_active >= 3:
        return "stable"
    return "growing"


def rollup_concept_activity(db: TrajectoryDB) -> ConceptRollupResult:
    """Materialize concept_activity table + importance scores + lifecycle stages.

    concept_activity: per-concept, per-month aggregation of event_count,
    avg_significance, and project_count.

    importance: composite score = event_count × avg_significance × project_span × recency_boost

    lifecycle: derived from activity curve shape.
    """
    result = ConceptRollupResult()
    now = datetime.now()

    # --- Phase 1: Materialize concept_activity ---
    # Get per-concept, per-month aggregates
    rows = db.conn.execute("""
        SELECT
            ce.concept_id,
            strftime('%Y-%m', e.timestamp) AS period,
            COUNT(*) AS event_count,
            AVG(e.significance) AS avg_significance,
            COUNT(DISTINCT e.project_id) AS projects_active
        FROM concept_events ce
        JOIN events e ON ce.event_id = e.id
        WHERE e.timestamp IS NOT NULL
        GROUP BY ce.concept_id, strftime('%Y-%m', e.timestamp)
    """).fetchall()

    for row in rows:
        db.upsert_concept_activity(
            concept_id=row["concept_id"],
            period=row["period"],
            event_count=row["event_count"],
            avg_significance=row["avg_significance"],
            projects_active=row["projects_active"],
        )
        result.activity_rows += 1

    db.conn.commit()

    # --- Phase 2: Compute importance and lifecycle per concept ---
    concepts = db.conn.execute("""
        SELECT
            c.id,
            c.first_seen,
            c.last_seen,
            COUNT(ce.id) AS event_count,
            AVG(e.significance) AS avg_significance,
            COUNT(DISTINCT e.project_id) AS project_span,
            COUNT(DISTINCT strftime('%Y-%m', e.timestamp)) AS months_active
        FROM concepts c
        LEFT JOIN concept_events ce ON c.id = ce.concept_id
        LEFT JOIN events e ON ce.event_id = e.id
        WHERE c.status = 'active'
        GROUP BY c.id
    """).fetchall()

    for concept in concepts:
        event_count = concept["event_count"] or 0
        avg_sig = concept["avg_significance"] or 0.5
        project_span = concept["project_span"] or 1
        months_active = concept["months_active"] or 1

        # Recency boost: concepts active in last 2 months get 1.5x
        recency_boost = 1.0
        if concept["last_seen"]:
            try:
                last = datetime.fromisoformat(concept["last_seen"])
                months_since = (now.year - last.year) * 12 + (now.month - last.month)
                if months_since <= 2:
                    recency_boost = 1.5
                elif months_since <= 4:
                    recency_boost = 1.2
            except (ValueError, TypeError):
                pass

        # Importance = event_count × avg_significance × project_span × recency
        # Normalize event_count with log-ish scaling to prevent huge concepts from dominating
        count_factor = math.log2(event_count + 1)  # 0 events → 0, 10 → 3.5, 100 → 6.7
        importance = count_factor * avg_sig * project_span * recency_boost

        # Lifecycle
        lifecycle = _compute_lifecycle(
            first_seen=concept["first_seen"],
            last_seen=concept["last_seen"],
            event_count=event_count,
            months_active=months_active,
            now=now,
        )

        db.update_concept_importance(concept["id"], round(importance, 3))
        db.update_concept_lifecycle(concept["id"], lifecycle)

        result.concepts_scored += 1
        result.lifecycle_stages[lifecycle] = result.lifecycle_stages.get(lifecycle, 0) + 1

    db.conn.commit()
    logger.info("Concept rollup: %s", result)
    return result
