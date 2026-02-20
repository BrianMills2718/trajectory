"""LLM-based event classification — extracts intent, concepts, significance, decisions.

Supports two analysis modes:
1. Session-level: Analyzes work sessions (conversation + commits) for richer context.
2. Event-level: Analyzes remaining orphan events (docs, archives, unlinked commits).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from llm_client import LLMCallResult, call_llm_structured, render_prompt
from pydantic import BaseModel, Field

from trajectory.config import Config
from trajectory.db import TrajectoryDB
from trajectory.models import EventRow, WorkSessionRow

logger = logging.getLogger(__name__)

PROMPT_VERSION = "event_classification_v4"
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


# --- Pydantic response models ---


class ConceptMention(BaseModel):
    name: str = Field(description="Concept name, lowercase_underscore. Reuse an existing name from the vocabulary when the idea is the same.")
    level: str = Field(description="One of: theme, design_bet, technique. Use a different label if none of the three fit — but then you must fill in level_rationale.")
    level_rationale: str | None = Field(default=None, description="Required when level is not theme/design_bet/technique. Explain what this level captures and why the standard three don't fit.")
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


# --- Analysis result tracking ---


class AnalysisResult:
    """Summary of an analysis run."""

    def __init__(self) -> None:
        self.events_processed: int = 0
        self.sessions_processed: int = 0
        self.concepts_found: int = 0
        self.decisions_found: int = 0
        self.total_cost: float = 0.0
        self.status: str = "running"

    def __repr__(self) -> str:
        return (
            f"AnalysisResult({self.events_processed} events, "
            f"{self.sessions_processed} sessions, "
            f"{self.concepts_found} concepts, {self.decisions_found} decisions, "
            f"${self.total_cost:.4f}, {self.status})"
        )


# --- Main entry point ---


def analyze_project(
    project_id: int,
    db: TrajectoryDB,
    config: Config,
    force_reanalyze: bool = False,
) -> AnalysisResult:
    """Run LLM analysis on a project.

    1. Analyze unanalyzed sessions (richer context)
    2. Analyze remaining unanalyzed events (docs, archives, orphan commits)
    """
    result = AnalysisResult()
    model = config.llm.model
    max_cost = config.llm.max_cost_per_run

    if force_reanalyze:
        _clear_analysis(db, project_id)

    # Gather existing concept vocabulary for this project
    existing_concepts = [
        c.name for c in db.list_concepts(project_id=project_id)
    ]

    # Create analysis run for provenance
    now = datetime.now(timezone.utc).isoformat()
    run_id = db.create_analysis_run(
        model=model,
        prompt_version=PROMPT_VERSION,
        project_id=project_id,
        started_at=now,
    )

    # Phase 1: Session-level analysis
    sessions = db.get_unanalyzed_sessions(project_id)
    if sessions:
        logger.info(
            "Analyzing %d sessions for project %d with %s",
            len(sessions), project_id, model,
        )
        new_concepts = _analyze_sessions(
            sessions, db, config, run_id, project_id, existing_concepts, result,
        )
        existing_concepts.extend(c for c in new_concepts if c not in existing_concepts)

    # Phase 2: Event-level analysis for remaining events
    events = db.get_unanalyzed_events(project_id, limit=5000)
    if events:
        logger.info(
            "Analyzing %d remaining events for project %d with %s (%d existing concepts)",
            len(events), project_id, model, len(existing_concepts),
        )
        _analyze_events(
            events, db, config, run_id, project_id, existing_concepts, result,
        )

    if not sessions and not events:
        logger.info("No unanalyzed sessions or events for project %d", project_id)
        result.status = "completed"

    # Check final status
    if result.status == "running":
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


# --- Session analysis ---


def _analyze_sessions(
    sessions: list[WorkSessionRow],
    db: TrajectoryDB,
    config: Config,
    run_id: int,
    project_id: int,
    existing_concepts: list[str],
    result: AnalysisResult,
) -> list[str]:
    """Analyze sessions in batches. Returns list of new concept names."""
    model = config.llm.model
    batch_size = config.llm.session_batch_size
    max_cost = config.llm.max_cost_per_run
    all_new_concepts: list[str] = []

    for batch_start in range(0, len(sessions), batch_size):
        batch = sessions[batch_start:batch_start + batch_size]

        if result.total_cost >= max_cost:
            logger.warning(
                "Cost budget exceeded: $%.4f >= $%.2f. Stopping.",
                result.total_cost, max_cost,
            )
            result.status = "budget_exceeded"
            break

        try:
            # Build session context for prompt
            session_contexts = []
            for s in batch:
                ctx: dict = {}
                if s.user_goal:
                    ctx["user_goal"] = s.user_goal
                if s.assistant_reasoning:
                    ctx["assistant_reasoning"] = s.assistant_reasoning[:config.extraction.max_reasoning_chars]
                if s.files_modified:
                    ctx["files_modified"] = s.files_modified
                if s.tool_sequence:
                    ctx["tool_sequence"] = s.tool_sequence
                if s.diff_summary:
                    ctx["diff_summary"] = s.diff_summary

                # Get commit info
                session_events = db.get_session_events(s.id)
                commit_info = []
                for se in session_events:
                    if se.role == "commit":
                        event = db.conn.execute(
                            "SELECT source_id, title, diff_summary FROM events WHERE id = ?",
                            (se.event_id,),
                        ).fetchone()
                        if event:
                            ci: dict = {
                                "hash": event["source_id"].removeprefix("git:")[:8],
                                "title": event["title"],
                            }
                            if event["diff_summary"]:
                                ci["diff_summary"] = event["diff_summary"]
                            commit_info.append(ci)
                if commit_info:
                    ctx["commit_info"] = commit_info

                session_contexts.append(ctx)

            messages = render_prompt(
                PROMPTS_DIR / "session_classification.yaml",
                sessions=session_contexts,
                session_count=len(batch),
                existing_concepts=existing_concepts if existing_concepts else None,
            )

            parsed, meta = call_llm_structured(
                model,
                messages,
                response_model=BatchAnalysisResult,
                timeout=300,
                num_retries=2,
                task="trajectory.classify_sessions",
                trace_id=f"trajectory.classify_sessions.proj{project_id}.batch{batch_start}",
                max_budget=0,
            )
            result.total_cost += meta.cost

            # Pad if needed
            while len(parsed.analyses) < len(batch):
                parsed.analyses.append(EventAnalysis(
                    summary="Analysis not available",
                    intent="cleanup",
                    significance=0.1,
                ))

            # Store results: update session, propagate to constituent events
            for session, analysis in zip(batch, parsed.analyses):
                db.update_session_analysis(
                    session_id=session.id,
                    llm_summary=analysis.summary,
                    llm_intent=analysis.intent,
                    significance=analysis.significance,
                    analysis_run_id=run_id,
                )
                result.sessions_processed += 1

                # Propagate to all events in this session
                session_events = db.get_session_events(session.id)
                for se in session_events:
                    db.update_event_analysis(
                        event_id=se.event_id,
                        llm_summary=analysis.summary,
                        llm_intent=analysis.intent,
                        significance=analysis.significance,
                        analysis_run_id=run_id,
                    )
                    result.events_processed += 1

                # Store concepts and decisions (linked to conversation event or first commit)
                anchor_event_id = session.conversation_event_id
                if not anchor_event_id and session_events:
                    anchor_event_id = session_events[0].event_id

                if anchor_event_id:
                    new_concepts = _store_concepts_and_decisions(
                        db, anchor_event_id, analysis, run_id, project_id, result,
                        event_timestamp=session.session_start,
                    )
                    all_new_concepts.extend(new_concepts)
                    existing_concepts.extend(
                        c for c in new_concepts if c not in existing_concepts
                    )

            db.conn.commit()

        except Exception:
            logger.exception("Error analyzing session batch starting at %d", batch_start)
            result.status = "failed"
            break

    return all_new_concepts


# --- Event analysis ---


def _analyze_events(
    events: list[EventRow],
    db: TrajectoryDB,
    config: Config,
    run_id: int,
    project_id: int,
    existing_concepts: list[str],
    result: AnalysisResult,
) -> None:
    """Analyze individual events in batches."""
    model = config.llm.model
    batch_size = config.llm.batch_size
    max_cost = config.llm.max_cost_per_run

    for batch_start in range(0, len(events), batch_size):
        batch = events[batch_start:batch_start + batch_size]

        if result.total_cost >= max_cost:
            logger.warning(
                "Cost budget exceeded: $%.4f >= $%.2f. Stopping.",
                result.total_cost, max_cost,
            )
            result.status = "budget_exceeded"
            break

        try:
            # Build event context for prompt
            event_contexts = []
            for e in batch:
                ctx: dict = {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "title": e.title,
                }
                if e.author:
                    ctx["author"] = e.author
                if e.git_branch:
                    ctx["git_branch"] = e.git_branch
                if e.diff_summary:
                    ctx["diff_summary"] = e.diff_summary
                if e.change_types:
                    ctx["change_types"] = e.change_types
                if e.files_changed:
                    try:
                        files = json.loads(e.files_changed)
                        if files:
                            shown = files[:15]
                            ctx["files_changed"] = ", ".join(shown)
                            if len(files) > 15:
                                ctx["files_changed"] += f" ... and {len(files) - 15} more"
                    except json.JSONDecodeError:
                        pass
                if e.body:
                    body = e.body[:1500]
                    if len(e.body) > 1500:
                        body += "\n... [truncated]"
                    ctx["body"] = body
                event_contexts.append(ctx)

            messages = render_prompt(
                PROMPTS_DIR / "event_classification.yaml",
                events=event_contexts,
                event_count=len(batch),
                existing_concepts=existing_concepts if existing_concepts else None,
            )

            parsed, meta = call_llm_structured(
                model,
                messages,
                response_model=BatchAnalysisResult,
                timeout=300,
                num_retries=2,
                task="trajectory.classify_events",
                trace_id=f"trajectory.classify_events.proj{project_id}.batch{batch_start}",
                max_budget=0,
            )
            result.total_cost += meta.cost

            # Pad if needed
            while len(parsed.analyses) < len(batch):
                parsed.analyses.append(EventAnalysis(
                    summary="Analysis not available",
                    intent="cleanup",
                    significance=0.1,
                ))

            # Store results
            for event, analysis in zip(batch, parsed.analyses):
                db.update_event_analysis(
                    event_id=event.id,
                    llm_summary=analysis.summary,
                    llm_intent=analysis.intent,
                    significance=analysis.significance,
                    analysis_run_id=run_id,
                )
                result.events_processed += 1

                new_concepts = _store_concepts_and_decisions(
                    db, event.id, analysis, run_id, project_id, result,
                    event_timestamp=event.timestamp,
                )
                existing_concepts.extend(
                    c for c in new_concepts if c not in existing_concepts
                )

            db.conn.commit()

        except Exception:
            logger.exception("Error analyzing event batch starting at %d", batch_start)
            result.status = "failed"
            break


# --- Shared helpers ---


def _store_concepts_and_decisions(
    db: TrajectoryDB,
    event_id: int,
    analysis: EventAnalysis,
    run_id: int,
    project_id: int,
    result: AnalysisResult,
    event_timestamp: str,
) -> list[str]:
    """Store concepts and decisions from an analysis. Returns new concept names."""
    new_concepts: list[str] = []

    for concept_mention in analysis.concepts:
        concept_id = db.upsert_concept(
            name=concept_mention.name,
            first_seen=event_timestamp,
            last_seen=event_timestamp,
            level=concept_mention.level,
            level_rationale=concept_mention.level_rationale,
        )
        db.link_concept_event(
            concept_id=concept_id,
            event_id=event_id,
            relationship=concept_mention.relationship,
            confidence=concept_mention.confidence,
            analysis_run_id=run_id,
        )
        result.concepts_found += 1
        new_concepts.append(concept_mention.name)

    for decision in analysis.decisions:
        db.insert_decision(
            event_id=event_id,
            project_id=project_id,
            title=decision.title,
            reasoning=decision.reasoning,
            alternatives=json.dumps(decision.alternatives) if decision.alternatives else None,
            decision_type=decision.decision_type,
            analysis_run_id=run_id,
        )
        result.decisions_found += 1

    return new_concepts


def _clear_analysis(db: TrajectoryDB, project_id: int) -> None:
    """Clear all analysis results for a project (for force-reanalyze)."""
    db.conn.execute(
        "UPDATE events SET llm_summary = NULL, llm_intent = NULL, significance = NULL, analysis_run_id = NULL WHERE project_id = ?",
        (project_id,),
    )
    db.conn.execute(
        "UPDATE work_sessions SET llm_summary = NULL, llm_intent = NULL, significance = NULL, analysis_run_id = NULL WHERE project_id = ?",
        (project_id,),
    )
    db.conn.commit()
    logger.info("Cleared analysis for project %d", project_id)
