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

PROMPT_VERSION = "event_classification_v3"


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


# --- Classifier ---


def _format_events(events: list[EventRow]) -> str:
    """Format events for the prompt."""
    descriptions: list[str] = []
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
            body = event.body[:1500]
            if len(event.body) > 1500:
                body += "\n... [truncated]"
            parts.append(f"Body:\n{body}")
        descriptions.append("\n".join(parts))
    return "\n\n---\n\n".join(descriptions)


def _build_batch_prompt(events: list[EventRow], existing_concepts: list[str]) -> str:
    """Build a prompt for batch event classification."""
    events_text = _format_events(events)

    vocab_section = ""
    if existing_concepts:
        vocab_section = f"""
## Existing concept vocabulary

These concepts have already been identified in this project. Reuse these names when an event
relates to the same idea rather than inventing a synonym:

{', '.join(existing_concepts)}

You may introduce new concepts when an event genuinely brings a new idea not captured above.
"""

    return f"""You are helping build a cross-project timeline that tracks how ideas emerge, spread, and
evolve across 60+ software projects over months. This data feeds three downstream uses:

1. **Cross-project linking**: Concepts get matched across repositories to show how an idea (like
   "mcp_server" or "knowledge_graph") spread from one project to others.
2. **Timeline narrative**: Concepts cluster events into coherent storylines so someone can ask
   "what happened with the ontology idea?" and get a meaningful arc, not a list of commits.
3. **Query answering**: Users search for concepts by keyword to find relevant events.

Think about it this way: if someone asked "what is this project about and where is it headed?",
the concepts you extract should be the ideas you'd mention in your answer. They're the design bets,
architectural choices, and recurring themes that define the project's identity and direction —
not the implementation mechanisms used within individual commits. Many routine commits (dependency
bumps, typo fixes, minor bug fixes) have no concepts at all, and that's correct.

For each event below, extract:

1. **summary**: 1-2 sentence summary of what happened.
2. **intent**: One of: feature, bugfix, refactor, exploration, documentation, test, infrastructure, cleanup.
3. **significance**: How much did this event shape the project's direction?
   0.0=routine noise, 0.3=minor, 0.5=noteworthy, 0.7=important, 0.9+=milestone.
4. **concepts**: Ideas this event relates to, at the appropriate level of abstraction:
   - **theme**: What is this project about? The big ideas and directions that define its identity.
     Examples: "knowledge_graph", "agent_architecture", "osint", "epistemic_reasoning"
   - **design_bet**: Architectural choices the developer is actively pursuing within a theme.
     Examples: "langgraph_backend", "spec_driven_generation", "owl_dl_translation"
   - **technique**: Specific implementation mechanisms. Useful for search, not for narrative.
     Examples: "bfs_detection", "fan_out_fix", "pagerank_scoring"

   Not every event has concepts at every level. A routine bugfix may only have a technique.
   A vision document may only have themes. Use lowercase_underscore names.

   If a concept genuinely doesn't fit any of the three levels, use a different label and explain
   why in level_rationale. This helps us discover categories the taxonomy is missing.
5. **decisions**: Architectural or design decisions — moments where the developer chose one approach
   over alternatives. Most events don't contain decisions; only extract them when genuinely present.
{vocab_section}
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

    # Gather existing concept vocabulary for this project
    existing_concepts = [
        c.name for c in db.list_concepts(project_id=project_id)
    ]
    logger.info(
        "Analyzing %d events for project %d with %s (%d existing concepts)",
        len(events), project_id, model, len(existing_concepts),
    )

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
            batch_analysis, llm_result = _analyze_batch(
                batch, model, existing_concepts,
                project_id=project_id, batch_start=batch_start,
            )
            result.total_cost += llm_result.cost

            # Store results and collect new concept names
            new_concepts = _store_batch_results(
                db, batch, batch_analysis, run_id, project_id, result
            )
            existing_concepts.extend(
                c for c in new_concepts if c not in existing_concepts
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
    existing_concepts: list[str],
    project_id: int,
    batch_start: int,
) -> tuple[BatchAnalysisResult, LLMCallResult]:
    """Send a batch of events to the LLM for analysis."""
    prompt = _build_batch_prompt(events, existing_concepts)
    messages = [{"role": "user", "content": prompt}]

    parsed, meta = call_llm_structured(
        model,
        messages,
        response_model=BatchAnalysisResult,
        timeout=120,
        num_retries=2,
        task="trajectory.classify_events",
        trace_id=f"trajectory.classify.proj{project_id}.batch{batch_start}",
        max_budget=0,
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
) -> list[str]:
    """Store analysis results in the database. Returns new concept names."""
    new_concepts: list[str] = []
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
                level=concept_mention.level,
                level_rationale=concept_mention.level_rationale,
            )
            db.link_concept_event(
                concept_id=concept_id,
                event_id=event.id,
                relationship=concept_mention.relationship,
                confidence=concept_mention.confidence,
                analysis_run_id=run_id,
            )
            result.concepts_found += 1
            new_concepts.append(concept_mention.name)

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
    return new_concepts
