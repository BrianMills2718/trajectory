"""LLM-based event classification — extracts intent, concepts, significance, decisions."""

import json
import logging
from datetime import datetime, timezone

from llm_client import LLMCallResult, call_llm_structured
from pydantic import BaseModel, Field

from trajectory.config import Config
from trajectory.db import TrajectoryDB
from trajectory.models import EventRow

logger = logging.getLogger(__name__)

PROMPT_VERSION = "event_classification_v1"


# --- Pydantic response models ---


class ConceptMention(BaseModel):
    name: str = Field(description="Short concept name, lowercase, underscore-separated (e.g. 'recursive_agent', 'evidence_architecture')")
    relationship: str = Field(description="One of: introduces, develops, refactors, abandons, completes, references")
    confidence: float = Field(ge=0.0, le=1.0, description="How confident this concept is relevant")


class DecisionFound(BaseModel):
    title: str = Field(description="Short title of the decision")
    reasoning: str = Field(description="Why this decision was made")
    alternatives: list[str] = Field(default_factory=list, description="Alternatives that were considered")
    decision_type: str = Field(description="One of: architectural, tooling, pattern, technology, process")


class EventAnalysis(BaseModel):
    summary: str = Field(description="1-2 sentence summary of what this event accomplished")
    intent: str = Field(description="One of: feature, bugfix, refactor, exploration, documentation, test, infrastructure, cleanup")
    significance: float = Field(ge=0.0, le=1.0, description="How significant is this event? 0.0=noise, 0.5=noteworthy, 1.0=major milestone")
    concepts: list[ConceptMention] = Field(default_factory=list, description="Concepts/ideas this event relates to")
    decisions: list[DecisionFound] = Field(default_factory=list, description="Architectural or design decisions found")


class BatchAnalysisResult(BaseModel):
    analyses: list[EventAnalysis] = Field(description="Analysis for each event, in the same order as input")


# --- Classifier ---


def _build_batch_prompt(events: list[EventRow]) -> str:
    """Build a prompt for batch event classification."""
    event_descriptions: list[str] = []
    for i, event in enumerate(events):
        parts = [
            f"[Event {i}]",
            f"Type: {event.event_type}",
            f"Timestamp: {event.timestamp}",
            f"Title: {event.title}",
        ]
        if event.author:
            parts.append(f"Author: {event.author}")
        if event.git_branch:
            parts.append(f"Branch: {event.git_branch}")
        if event.files_changed:
            try:
                files = json.loads(event.files_changed)
                if files:
                    parts.append(f"Files changed: {', '.join(files[:15])}")
                    if len(files) > 15:
                        parts.append(f"  ... and {len(files) - 15} more")
            except json.JSONDecodeError:
                pass
        if event.body:
            # Truncate body to avoid token explosion
            body = event.body[:1500]
            if len(event.body) > 1500:
                body += "\n... [truncated]"
            parts.append(f"Body:\n{body}")
        event_descriptions.append("\n".join(parts))

    events_text = "\n\n---\n\n".join(event_descriptions)

    return f"""Analyze each event from a software project's history. For each event, determine:

1. **summary**: 1-2 sentence summary of what happened
2. **intent**: The primary intent (feature, bugfix, refactor, exploration, documentation, test, infrastructure, cleanup)
3. **significance**: How significant is this event? (0.0=routine/noise, 0.3=minor, 0.5=noteworthy, 0.7=important, 1.0=major milestone)
4. **concepts**: Key technical concepts, patterns, or architectural ideas mentioned or affected. Use lowercase underscore-separated names. Each concept should have a relationship type (introduces/develops/refactors/abandons/completes/references).
5. **decisions**: Any architectural or design decisions made (if any).

Guidelines for significance:
- 0.0-0.2: Routine commits, typo fixes, dependency bumps, trivial changes
- 0.3-0.4: Minor features, small bug fixes, test additions
- 0.5-0.6: Notable features, significant refactors, important bug fixes
- 0.7-0.8: Major features, architectural changes, critical bug fixes
- 0.9-1.0: Project milestones, fundamental architecture changes, major releases

Guidelines for concepts:
- Be specific: "recursive_agent" not "agent"
- Reuse concept names across events when they refer to the same thing
- Common patterns: error_handling, mcp_server, evidence_architecture, clean_architecture, etc.

Return exactly {len(events)} analyses in the 'analyses' array, one per event, in the same order.

Events:

{events_text}"""


class AnalysisResult:
    """Summary of an analysis run."""

    def __init__(self) -> None:
        self.events_processed: int = 0
        self.concepts_found: int = 0
        self.decisions_found: int = 0
        self.total_cost: float = 0.0
        self.status: str = "running"

    def __repr__(self) -> str:
        return (
            f"AnalysisResult({self.events_processed} events, "
            f"{self.concepts_found} concepts, {self.decisions_found} decisions, "
            f"${self.total_cost:.4f}, {self.status})"
        )


def analyze_project(
    project_id: int,
    db: TrajectoryDB,
    config: Config,
) -> AnalysisResult:
    """Run LLM analysis on all unanalyzed events for a project."""
    result = AnalysisResult()
    model = config.llm.model
    batch_size = config.llm.batch_size
    max_cost = config.llm.max_cost_per_run

    # Get unanalyzed events
    events = db.get_unanalyzed_events(project_id, limit=5000)
    if not events:
        logger.info("No unanalyzed events for project %d", project_id)
        result.status = "completed"
        return result

    logger.info("Analyzing %d events for project %d with %s", len(events), project_id, model)

    # Create analysis run for provenance
    now = datetime.now(timezone.utc).isoformat()
    run_id = db.create_analysis_run(
        model=model,
        prompt_version=PROMPT_VERSION,
        project_id=project_id,
        started_at=now,
    )

    # Process in batches
    for batch_start in range(0, len(events), batch_size):
        batch = events[batch_start:batch_start + batch_size]
        logger.info(
            "Processing batch %d-%d of %d",
            batch_start, batch_start + len(batch), len(events),
        )

        # Check cost budget
        if result.total_cost >= max_cost:
            logger.warning(
                "Cost budget exceeded: $%.4f >= $%.2f. Stopping.",
                result.total_cost, max_cost,
            )
            result.status = "budget_exceeded"
            break

        try:
            batch_analysis, llm_result = _analyze_batch(batch, model)
            result.total_cost += llm_result.cost

            # Store results
            _store_batch_results(
                db, batch, batch_analysis, run_id, project_id, result
            )

        except Exception:
            logger.exception("Error analyzing batch starting at %d", batch_start)
            result.status = "failed"
            break
    else:
        result.status = "completed"

    # Update analysis run
    db.update_analysis_run(
        run_id,
        events_processed=result.events_processed,
        cost_usd=result.total_cost,
        completed_at=datetime.now(timezone.utc).isoformat(),
        status=result.status,
    )
    db.conn.commit()

    logger.info("Analysis complete: %s", result)
    return result


def _analyze_batch(
    events: list[EventRow],
    model: str,
) -> tuple[BatchAnalysisResult, LLMCallResult]:
    """Send a batch of events to the LLM for analysis."""
    prompt = _build_batch_prompt(events)
    messages = [{"role": "user", "content": prompt}]

    parsed, meta = call_llm_structured(
        model,
        messages,
        response_model=BatchAnalysisResult,
        timeout=120,
        num_retries=2,
    )

    # Validate count matches
    if len(parsed.analyses) != len(events):
        logger.warning(
            "LLM returned %d analyses for %d events. Truncating/padding.",
            len(parsed.analyses), len(events),
        )
        # Pad with minimal analyses if needed
        while len(parsed.analyses) < len(events):
            parsed.analyses.append(EventAnalysis(
                summary="Analysis not available",
                intent="cleanup",
                significance=0.1,
            ))

    return parsed, meta


def _store_batch_results(
    db: TrajectoryDB,
    events: list[EventRow],
    batch_analysis: BatchAnalysisResult,
    run_id: int,
    project_id: int,
    result: AnalysisResult,
) -> None:
    """Store analysis results in the database."""
    for event, analysis in zip(events, batch_analysis.analyses):
        # Update event with analysis
        db.update_event_analysis(
            event_id=event.id,
            llm_summary=analysis.summary,
            llm_intent=analysis.intent,
            significance=analysis.significance,
            analysis_run_id=run_id,
        )
        result.events_processed += 1

        # Store concepts and links
        for concept_mention in analysis.concepts:
            concept_id = db.upsert_concept(
                name=concept_mention.name,
                first_seen=event.timestamp,
                last_seen=event.timestamp,
            )
            db.link_concept_event(
                concept_id=concept_id,
                event_id=event.id,
                relationship=concept_mention.relationship,
                confidence=concept_mention.confidence,
                analysis_run_id=run_id,
            )
            result.concepts_found += 1

        # Store decisions
        for decision in analysis.decisions:
            db.insert_decision(
                event_id=event.id,
                project_id=project_id,
                title=decision.title,
                reasoning=decision.reasoning,
                alternatives=json.dumps(decision.alternatives) if decision.alternatives else None,
                decision_type=decision.decision_type,
                analysis_run_id=run_id,
            )
            result.decisions_found += 1

    db.conn.commit()
