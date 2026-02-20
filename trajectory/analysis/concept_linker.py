"""Cross-project concept linking — matches theme-level concepts via LLM."""

import logging
from pathlib import Path

from llm_client import call_llm_structured, render_prompt
from pydantic import BaseModel, Field

from trajectory.config import Config
from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


# --- LLM response models ---


class ConceptLinkFound(BaseModel):
    concept_a: str = Field(description="Name of the first concept (exact match from input list)")
    concept_b: str = Field(description="Name of the second concept (exact match from input list)")
    relationship: str = Field(description="One of: depends_on, evolved_from, replaced_by, related_to, spawned")
    strength: float = Field(ge=0.0, le=1.0, description="How strong is this relationship")
    evidence: str = Field(description="One sentence explaining why this link exists")


class ConceptLinkingResult(BaseModel):
    links: list[ConceptLinkFound] = Field(default_factory=list, description="Cross-project concept links found")


# --- Linking result ---


class LinkingResult:
    """Summary of a linking run."""

    def __init__(self) -> None:
        self.themes_input: int = 0
        self.links_found: int = 0
        self.links_stored: int = 0
        self.links_rejected: int = 0
        self.old_links_deleted: int = 0
        self.cost: float = 0.0

    def __repr__(self) -> str:
        return (
            f"LinkingResult({self.themes_input} themes → "
            f"{self.links_found} found, {self.links_stored} stored, "
            f"{self.links_rejected} rejected, ${self.cost:.4f})"
        )


VALID_RELATIONSHIPS = {"depends_on", "evolved_from", "replaced_by", "related_to", "spawned"}


def link_concepts(db: TrajectoryDB, config: Config) -> LinkingResult:
    """Find and store cross-project links between theme-level concepts.

    Full-replacement strategy: deletes all existing links and rewrites them.
    """
    result = LinkingResult()

    # Load all theme-level concepts
    themes = db.list_concepts(level="theme")
    result.themes_input = len(themes)
    if len(themes) < 2:
        logger.info("Only %d themes — nothing to link", len(themes))
        return result

    # Build per-theme context: which projects, event counts
    theme_contexts: list[dict[str, object]] = []
    for t in themes:
        event_count_row = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM concept_events WHERE concept_id = ?",
            (t.id,),
        ).fetchone()
        event_count = event_count_row["cnt"] if event_count_row else 0

        # Find projects this concept appears in
        project_rows = db.conn.execute(
            """SELECT DISTINCT p.name FROM projects p
               JOIN events e ON e.project_id = p.id
               JOIN concept_events ce ON ce.event_id = e.id
               WHERE ce.concept_id = ?
               ORDER BY p.name""",
            (t.id,),
        ).fetchall()
        projects = [r["name"] for r in project_rows]

        theme_contexts.append({
            "name": t.name,
            "status": t.status,
            "description": t.description,
            "first_seen": t.first_seen[:10] if t.first_seen else None,
            "last_seen": t.last_seen[:10] if t.last_seen else None,
            "event_count": event_count,
            "projects": projects,
        })

    logger.info("Linking %d themes across projects", len(themes))

    # Render prompt and call LLM
    messages = render_prompt(
        PROMPTS_DIR / "concept_linking.yaml",
        themes=theme_contexts,
    )

    parsed, meta = call_llm_structured(
        config.llm.model,
        messages,
        response_model=ConceptLinkingResult,
        timeout=120,
        num_retries=2,
        task="trajectory.link_concepts",
        trace_id="trajectory.link_concepts",
        max_budget=0,
    )
    result.cost = meta.cost

    # Build name→id lookup
    name_to_id = {t.name.lower(): t.id for t in themes}

    # Validate and filter links
    valid_links: list[ConceptLinkFound] = []
    for link in parsed.links:
        a_id = name_to_id.get(link.concept_a.lower())
        b_id = name_to_id.get(link.concept_b.lower())
        if a_id is None:
            logger.warning("Rejecting link: unknown concept_a %r", link.concept_a)
            result.links_rejected += 1
            continue
        if b_id is None:
            logger.warning("Rejecting link: unknown concept_b %r", link.concept_b)
            result.links_rejected += 1
            continue
        if a_id == b_id:
            logger.warning("Rejecting self-link: %r", link.concept_a)
            result.links_rejected += 1
            continue
        if link.relationship not in VALID_RELATIONSHIPS:
            logger.warning("Rejecting link: invalid relationship %r", link.relationship)
            result.links_rejected += 1
            continue
        valid_links.append(link)

    result.links_found = len(parsed.links)

    # Full replacement: delete old, insert new
    result.old_links_deleted = db.delete_all_concept_links()
    for link in valid_links:
        a_id = name_to_id[link.concept_a.lower()]
        b_id = name_to_id[link.concept_b.lower()]
        db.upsert_concept_link(
            concept_a_id=a_id,
            concept_b_id=b_id,
            relationship=link.relationship,
            strength=link.strength,
            evidence=link.evidence,
        )
        result.links_stored += 1
    db.conn.commit()

    logger.info("Linking complete: %s", result)
    return result
