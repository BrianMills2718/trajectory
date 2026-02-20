"""Mural generator — continuous AI art mural of the conceptual landscape.

X-axis = time (months). Y-axis = conceptual space — each tile is a
theme, positioned by PCA of its associated techniques/bets. Themes
with similar sub-concepts cluster together. The mural shows the
intellectual landscape evolving over time, independent of projects.
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from llm_client import call_llm, render_prompt
from openai import OpenAI

from trajectory.config import Config, MuralConfig
from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


# --- Data structures ---


@dataclass
class ThemeTile:
    """A theme's activity in a single month."""

    theme: str
    month: str
    activity: str  # "active" or "dormant"
    techniques: list[str]
    design_bets: list[str]
    decisions: list[str]
    event_count: int
    # All sub-concepts (techniques + bets) for embedding
    sub_concepts: list[str] = field(default_factory=list)


@dataclass
class PlacedTile:
    """A tile with its canvas position."""

    data: ThemeTile
    x: int
    y: int
    y_norm: float  # [0, 1]
    art_prompt: str = ""
    image: Image.Image | None = None


@dataclass
class MuralResult:
    """Final mural output."""

    tiles: list[PlacedTile]
    themes: list[str]
    months: list[str]
    canvas_size: tuple[int, int]
    output_dir: Path
    total_cost: float = 0.0
    errors: list[str] = field(default_factory=list)


# --- Theme discovery ---


def discover_themes(
    db: TrajectoryDB, min_events: int = 3
) -> list[dict[str, object]]:
    """Find all theme-level concepts with enough activity.

    Returns dicts with: id, name, event_count, month_count.
    """
    rows = db.conn.execute(
        """SELECT c.id, c.name,
                  COUNT(DISTINCT ce.event_id) as event_count,
                  COUNT(DISTINCT strftime('%Y-%m', e.timestamp)) as month_count
           FROM concepts c
           JOIN concept_events ce ON c.id = ce.concept_id
           JOIN events e ON ce.event_id = e.id
           WHERE c.level = 'theme'
             AND e.analysis_run_id IS NOT NULL
           GROUP BY c.id
           HAVING event_count >= ?
           ORDER BY event_count DESC""",
        (min_events,),
    ).fetchall()
    return [dict(r) for r in rows]


def select_months_global(db: TrajectoryDB, n: int) -> list[str]:
    """Select N contiguous months with the most analyzed theme activity."""
    rows = db.conn.execute(
        """SELECT strftime('%Y-%m', e.timestamp) as month,
                  COUNT(DISTINCT c.id) as theme_count
           FROM events e
           JOIN concept_events ce ON ce.event_id = e.id
           JOIN concepts c ON c.id = ce.concept_id
           WHERE c.level = 'theme' AND e.analysis_run_id IS NOT NULL
           GROUP BY month
           ORDER BY month"""
    ).fetchall()

    if not rows:
        return []

    months = [r["month"] for r in rows]
    counts = [r["theme_count"] for r in rows]

    if len(months) <= n:
        return months

    best_start = 0
    best_score = 0
    for i in range(len(months) - n + 1):
        score = sum(counts[i : i + n])
        if score > best_score:
            best_score = score
            best_start = i

    return months[best_start : best_start + n]


# --- Tile data queries ---


def query_theme_tile(
    db: TrajectoryDB, theme_id: int, theme_name: str, month: str
) -> ThemeTile:
    """Get a theme's activity for one month: techniques, bets, decisions."""
    month_start = f"{month}-01"
    month_end = f"{month}-31"

    # Events for this theme in this month
    event_rows = db.conn.execute(
        """SELECT DISTINCT e.id FROM events e
           JOIN concept_events ce ON ce.event_id = e.id
           WHERE ce.concept_id = ? AND e.timestamp >= ? AND e.timestamp <= ?""",
        (theme_id, month_start, month_end),
    ).fetchall()
    event_ids = [r["id"] for r in event_rows]
    event_count = len(event_ids)

    if not event_ids:
        return ThemeTile(
            theme=theme_name, month=month, activity="dormant",
            techniques=[], design_bets=[], decisions=[],
            event_count=0, sub_concepts=[],
        )

    # Find co-occurring concepts (techniques and design_bets) on same events
    placeholders = ",".join("?" * len(event_ids))
    concept_rows = db.conn.execute(
        f"""SELECT DISTINCT c.name, c.level FROM concepts c
            JOIN concept_events ce ON c.id = ce.concept_id
            WHERE ce.event_id IN ({placeholders})
              AND c.level IN ('technique', 'design_bet')
            ORDER BY c.level, c.name""",
        event_ids,
    ).fetchall()

    techniques = [r["name"] for r in concept_rows if r["level"] == "technique"]
    design_bets = [r["name"] for r in concept_rows if r["level"] == "design_bet"]

    # Decisions on these events
    decision_rows = db.conn.execute(
        f"""SELECT DISTINCT d.title FROM decisions d
            WHERE d.event_id IN ({placeholders})
            LIMIT 5""",
        event_ids,
    ).fetchall()
    decisions = [r["title"] for r in decision_rows]

    return ThemeTile(
        theme=theme_name,
        month=month,
        activity="active",
        techniques=techniques,
        design_bets=design_bets,
        decisions=decisions,
        event_count=event_count,
        sub_concepts=techniques + design_bets,
    )


# --- Conceptual Y-axis via PCA ---


def compute_y_positions(tiles: list[ThemeTile]) -> list[float]:
    """Compute Y positions [0, 1] for each tile based on sub-concepts.

    Builds binary vectors from each tile's techniques + design_bets,
    then PCA first component gives a 1D conceptual axis.
    Tiles with no sub-concepts get y=0.5.
    """
    vocab: dict[str, int] = {}
    for tile in tiles:
        for concept in tile.sub_concepts:
            if concept not in vocab:
                vocab[concept] = len(vocab)

    if not vocab:
        return [0.5] * len(tiles)

    n_tiles = len(tiles)
    n_concepts = len(vocab)
    matrix = np.zeros((n_tiles, n_concepts), dtype=np.float32)
    for i, tile in enumerate(tiles):
        for concept in tile.sub_concepts:
            matrix[i, vocab[concept]] = 1.0

    has_concepts = matrix.sum(axis=1) > 0

    if has_concepts.sum() < 2:
        return [0.5] * n_tiles

    # TF-IDF weighting: downweight concepts that appear in many tiles
    doc_freq = (matrix > 0).sum(axis=0) + 1  # +1 smoothing
    idf = np.log(n_tiles / doc_freq)
    matrix = matrix * idf

    active = matrix[has_concepts]
    centered = active - active.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]

    projections = matrix @ pc1

    active_proj = projections[has_concepts]
    p_min = active_proj.min()
    p_max = active_proj.max()

    if p_max - p_min < 1e-8:
        return [0.5] * n_tiles

    positions: list[float] = []
    for i in range(n_tiles):
        if has_concepts[i]:
            positions.append(float((projections[i] - p_min) / (p_max - p_min)))
        else:
            positions.append(0.5)

    return positions


# --- Prompt generation ---


def generate_tile_prompt(
    tile: ThemeTile,
    config: MuralConfig,
    neighbor_prompts: dict[str, str] | None = None,
) -> str:
    """Use LLM to convert theme tile data into an art prompt."""
    messages = render_prompt(
        PROMPTS_DIR / "mural_tile.yaml",
        theme=tile.theme,
        activity=tile.activity,
        techniques=tile.techniques[:8],
        design_bets=tile.design_bets[:8],
        decisions=tile.decisions[:5],
        neighbor_prompts=neighbor_prompts or {},
    )

    result = call_llm(
        config.prompt_model,
        messages,
        task="trajectory.mural.prompt",
        trace_id=f"trajectory.mural.prompt.{tile.theme}.{tile.month}",
        max_budget=0,
    )

    art_prompt = result.content.strip()
    return f"{art_prompt} Style: {config.style_suffix}"


# --- Image generation ---


def generate_tile_image(
    prompt: str, config: MuralConfig, client: OpenAI
) -> Image.Image:
    """Generate a single tile via gpt-image-1."""
    size = f"{config.tile_size}x{config.tile_size}"
    response = client.images.generate(
        model=config.image_model,
        prompt=prompt,
        n=1,
        size=size,
        quality=config.quality,
    )

    image_b64 = response.data[0].b64_json
    if not image_b64:
        raise RuntimeError("No b64_json in image response")

    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA")


# --- Alpha blending assembly ---


def create_radial_fade_mask(size: int, fade_px: int) -> Image.Image:
    """Create an alpha mask with gradient fade on all four edges."""
    mask = Image.new("L", (size, size), 255)
    pixels = mask.load()

    for edge_offset in range(fade_px):
        alpha = int(255 * edge_offset / fade_px)
        for x in range(size):
            # Top/bottom
            pixels[x, edge_offset] = min(alpha, pixels[x, edge_offset])
            pixels[x, size - 1 - edge_offset] = min(alpha, pixels[x, size - 1 - edge_offset])
        for y in range(size):
            # Left/right
            pixels[edge_offset, y] = min(alpha, pixels[edge_offset, y])
            pixels[size - 1 - edge_offset, y] = min(alpha, pixels[size - 1 - edge_offset, y])

    return mask


def assemble_mural(
    placed_tiles: list[PlacedTile],
    canvas_w: int,
    canvas_h: int,
    tile_size: int,
    fade_px: int,
) -> Image.Image:
    """Composite placed tiles onto canvas with alpha blending."""
    mural = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    mask = create_radial_fade_mask(tile_size, fade_px)

    for pt in sorted(placed_tiles, key=lambda t: t.y):
        if pt.image is None:
            continue

        blended = pt.image.copy()
        blended.putalpha(mask)

        x = max(0, min(pt.x, canvas_w - tile_size))
        y = max(0, min(pt.y, canvas_h - tile_size))
        mural.alpha_composite(blended, (x, y))

    return mural


# --- Main entry point ---


def generate_mural(
    db: TrajectoryDB,
    config: Config,
    themes: list[str] | None = None,
    months: list[str] | None = None,
    dry_run: bool = False,
) -> MuralResult:
    """Generate a conceptual landscape mural.

    Each tile = a theme in a month. Y position = PCA of sub-concepts.
    Themes with similar techniques/bets cluster together.
    """
    mc = config.mural

    # Discover themes
    if themes:
        theme_info: list[dict[str, object]] = []
        for name in themes:
            c = db.get_concept_by_name(name)
            if not c:
                raise ValueError(f"Theme not found: {name}")
            theme_info.append({"id": c.id, "name": c.name})
    else:
        theme_info = discover_themes(db, min_events=3)
        if not theme_info:
            raise ValueError("No themes with sufficient activity found")
        # Limit to top N by activity
        theme_info = theme_info[: mc.max_projects]

    theme_names = [str(t["name"]) for t in theme_info]
    logger.info("Themes (%d): %s", len(theme_info), theme_names)

    # Resolve months
    if not months:
        months = select_months_global(db, mc.max_months)
        if not months:
            raise ValueError("No analyzed events found")

    n_themes = len(theme_info)
    n_months = len(months)

    # Phase 1: Query tile data for every theme × month
    all_tiles: list[ThemeTile] = []
    tile_month_idx: list[int] = []
    for ti in theme_info:
        for mi, month in enumerate(months):
            td = query_theme_tile(db, int(ti["id"]), str(ti["name"]), month)
            all_tiles.append(td)
            tile_month_idx.append(mi)

    active_count = sum(1 for t in all_tiles if t.activity == "active")
    logger.info(
        "%d tiles total, %d active, %d dormant",
        len(all_tiles), active_count, len(all_tiles) - active_count,
    )

    # Skip dormant tiles entirely — only generate art for active ones
    active_indices = [i for i, t in enumerate(all_tiles) if t.activity == "active"]
    active_tiles = [all_tiles[i] for i in active_indices]
    active_month_idxs = [tile_month_idx[i] for i in active_indices]

    if not active_tiles:
        raise ValueError("No active theme tiles found in selected months")

    # Phase 2: Compute Y positions via PCA on sub-concept vectors
    y_positions = compute_y_positions(active_tiles)

    # Canvas dimensions
    tile_size = mc.tile_size
    fade_px = int(tile_size * mc.vertical_overlap)
    canvas_w = n_months * tile_size
    # Height: scale based on Y spread needed
    n_active_per_month = max(
        sum(1 for mi in active_month_idxs if mi == m) for m in range(n_months)
    )
    canvas_h = max(tile_size * 2, int(tile_size * (n_active_per_month * 0.7 + 0.3)))
    usable_h = canvas_h - tile_size

    # Place tiles
    placed: list[PlacedTile] = []
    for i, (td, mi) in enumerate(zip(active_tiles, active_month_idxs)):
        x = mi * tile_size
        y = int(y_positions[i] * usable_h)
        pt = PlacedTile(data=td, x=x, y=y, y_norm=y_positions[i])
        placed.append(pt)

    logger.info("Canvas: %d x %d px, %d active tiles placed", canvas_w, canvas_h, len(placed))

    for pt in placed:
        logger.info(
            "  %s / %s → y=%.2f, %d techniques, %d bets",
            pt.data.theme, pt.data.month, pt.y_norm,
            len(pt.data.techniques), len(pt.data.design_bets),
        )

    output_dir = config.resolved_db_path.parent / "mural"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 3: Generate art prompts (sorted by month then y for neighbor context)
    prompt_order = sorted(range(len(placed)), key=lambda i: (active_month_idxs[i], placed[i].y))

    # Index for neighbor lookup
    by_month: dict[int, list[int]] = {}
    for i, mi in enumerate(active_month_idxs):
        by_month.setdefault(mi, []).append(i)
    for mi in by_month:
        by_month[mi].sort(key=lambda i: placed[i].y)

    generated_prompts: dict[int, str] = {}

    for idx in prompt_order:
        pt = placed[idx]
        mi = active_month_idxs[idx]

        neighbor_prompts: dict[str, str] = {}

        # Left: same theme, previous month
        if mi > 0:
            for j in range(len(placed)):
                if (placed[j].data.theme == pt.data.theme
                        and active_month_idxs[j] == mi - 1
                        and j in generated_prompts):
                    neighbor_prompts["left"] = generated_prompts[j]
                    break

        # Nearest above in same month
        if mi in by_month:
            for j in reversed(by_month[mi]):
                if j != idx and placed[j].y < pt.y and j in generated_prompts:
                    neighbor_prompts["above"] = generated_prompts[j]
                    break

        art_prompt = generate_tile_prompt(pt.data, mc, neighbor_prompts or None)
        generated_prompts[idx] = art_prompt
        pt.art_prompt = art_prompt

    result = MuralResult(
        tiles=placed,
        themes=theme_names,
        months=months,
        canvas_size=(canvas_w, canvas_h),
        output_dir=output_dir,
    )

    if dry_run:
        print(f"\nMural: {n_themes} themes x {n_months} months → {len(placed)} active tiles")
        print(f"Canvas: {canvas_w} x {canvas_h} px")
        print(f"Edge fade: {fade_px} px")
        print(f"Themes: {', '.join(theme_names)}")
        print(f"Months: {', '.join(months)}")
        print(f"Output: {output_dir}")
        print()

        for mi, month in enumerate(months):
            month_tiles = [(i, placed[i]) for i in range(len(placed)) if active_month_idxs[i] == mi]
            if not month_tiles:
                print(f"--- {month} --- (no active themes)")
                continue
            month_tiles.sort(key=lambda t: t[1].y)
            print(f"--- {month} ({len(month_tiles)} active) ---")
            for _, pt in month_tiles:
                td = pt.data
                print(f"  y={pt.y_norm:.2f} {td.theme}")
                if td.techniques:
                    print(f"    Techniques: {', '.join(td.techniques[:5])}")
                if td.design_bets:
                    print(f"    Bets: {', '.join(td.design_bets[:5])}")
                print(f"    Prompt: {pt.art_prompt[:120]}...")
            print()
        return result

    # Phase 4: Generate images
    client = OpenAI()
    estimated_cost_per_tile = 0.04
    cost_estimate = len(placed) * estimated_cost_per_tile

    if cost_estimate > mc.max_cost:
        raise ValueError(
            f"Estimated cost ${cost_estimate:.2f} exceeds max_cost ${mc.max_cost:.2f}. "
            f"Reduce tile count or increase max_cost."
        )

    for pt in placed:
        logger.info("Generating image %s/%s y=%.2f", pt.data.theme, pt.data.month, pt.y_norm)
        try:
            img = generate_tile_image(pt.art_prompt, mc, client)
            pt.image = img

            safe_name = pt.data.theme.replace("/", "_").replace(" ", "_")
            tile_path = output_dir / f"tile_{safe_name}_{pt.data.month}.png"
            img.save(str(tile_path))

        except Exception as e:
            error_msg = f"{pt.data.theme}/{pt.data.month} failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    # Phase 5: Assemble
    mural_img = assemble_mural(placed, canvas_w, canvas_h, tile_size, fade_px)
    mural_path = output_dir / "mural.png"
    mural_img.save(str(mural_path))

    generated_count = sum(1 for pt in placed if pt.image is not None)
    result.total_cost = generated_count * estimated_cost_per_tile

    print(f"\nMural generated: {mural_path}")
    print(f"Tiles: {generated_count}/{len(placed)} generated")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")
    print(f"Estimated cost: ${result.total_cost:.2f}")

    return result
