"""Mural generator v2 — LLM-planned layout, corner-out generation.

The LLM arranges themes into a conceptual vertical order based on their
relationships. Tiles are generated in scanline order (top-left → right
→ next row) so each tile's prompt can reference already-generated
neighbors, producing a coherent flowing mural.
"""

import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from PIL import Image, ImageDraw

from llm_client import call_llm, render_prompt

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


@dataclass
class GridCell:
    """A tile placed in the grid."""

    data: ThemeTile
    row: int
    col: int
    art_prompt: str = ""
    image: Image.Image | None = None


@dataclass
class MuralResult:
    """Final mural output."""

    grid: list[GridCell]
    themes: list[str]
    months: list[str]
    n_rows: int
    n_cols: int
    canvas_size: tuple[int, int]
    output_dir: Path
    total_cost: float = 0.0
    errors: list[str] = field(default_factory=list)


# --- Theme discovery ---


def discover_themes(
    db: TrajectoryDB, min_events: int = 3
) -> list[dict[str, object]]:
    """Find theme-level concepts with enough activity."""
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
    """Select N contiguous months with the most theme activity."""
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
    """Get a theme's activity for one month."""
    month_start = f"{month}-01"
    month_end = f"{month}-31"

    event_rows = db.conn.execute(
        """SELECT DISTINCT e.id FROM events e
           JOIN concept_events ce ON ce.event_id = e.id
           WHERE ce.concept_id = ? AND e.timestamp >= ? AND e.timestamp <= ?""",
        (theme_id, month_start, month_end),
    ).fetchall()
    event_ids = [r["id"] for r in event_rows]

    if not event_ids:
        return ThemeTile(
            theme=theme_name, month=month, activity="dormant",
            techniques=[], design_bets=[], decisions=[], event_count=0,
        )

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

    decision_rows = db.conn.execute(
        f"""SELECT DISTINCT d.title FROM decisions d
            WHERE d.event_id IN ({placeholders}) LIMIT 5""",
        event_ids,
    ).fetchall()

    return ThemeTile(
        theme=theme_name, month=month, activity="active",
        techniques=techniques, design_bets=design_bets,
        decisions=[r["title"] for r in decision_rows],
        event_count=len(event_ids),
    )


# --- LLM-planned layout ---


def plan_layout(
    theme_tiles: dict[str, ThemeTile],
    db: TrajectoryDB,
    config: MuralConfig,
) -> dict[str, int]:
    """Ask LLM to arrange themes into rows based on conceptual relationships.

    Returns {theme_name: row_index}.
    """
    # Gather relationship data
    links = db.get_concept_links()
    id_to_name: dict[int, str] = {}
    for link in links:
        for cid in (link.concept_a_id, link.concept_b_id):
            if cid not in id_to_name:
                row = db.conn.execute(
                    "SELECT name FROM concepts WHERE id = ?", (cid,)
                ).fetchone()
                id_to_name[cid] = row["name"] if row else f"?{cid}"

    link_data = []
    theme_set = set(theme_tiles.keys())
    for link in links:
        a = id_to_name.get(link.concept_a_id, "")
        b = id_to_name.get(link.concept_b_id, "")
        # Only include links where at least one end is in our theme set
        if a in theme_set or b in theme_set:
            link_data.append({"a": a, "rel": link.relationship, "b": b})

    # Build theme summaries for the LLM
    theme_summaries = []
    for name, tile in theme_tiles.items():
        theme_summaries.append({
            "name": name,
            "techniques": tile.techniques[:5],
            "design_bets": tile.design_bets[:5],
        })

    messages = render_prompt(
        PROMPTS_DIR / "mural_layout.yaml",
        themes=theme_summaries,
        theme_count=len(theme_summaries),
        links=link_data,
    )

    result = call_llm(
        config.prompt_model,
        messages,
        task="trajectory.mural.layout",
        trace_id="trajectory.mural.layout",
        max_budget=0,
    )

    # Parse JSON response
    content = result.content.strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]

    layout = json.loads(content)
    return {item["theme"]: item["row"] for item in layout}


# --- Prompt generation ---


def generate_tile_prompt(
    tile: ThemeTile,
    config: MuralConfig,
    neighbors: dict[str, str] | None = None,
) -> str:
    """Use LLM to create an art prompt for this tile.

    neighbors: {"above": art_prompt, "left": art_prompt} of already-generated tiles.
    """
    messages = render_prompt(
        PROMPTS_DIR / "mural_tile.yaml",
        theme=tile.theme,
        activity=tile.activity,
        techniques=tile.techniques[:8],
        design_bets=tile.design_bets[:8],
        decisions=tile.decisions[:5],
        neighbors=neighbors or {},
    )

    result = call_llm(
        config.prompt_model,
        messages,
        task="trajectory.mural.prompt",
        trace_id=f"trajectory.mural.prompt.{tile.theme}.{tile.month}",
        max_budget=0,
    )

    return f"{result.content.strip()} Style: {config.style_suffix}"


# --- Image generation ---


def generate_tile_image(
    prompt: str, config: MuralConfig, client: genai.Client
) -> Image.Image:
    """Generate a single tile image via Gemini (Nano Banana)."""
    full_prompt = (
        f"Generate an image: {prompt} "
        f"The image should be square, detailed, and artistic."
    )
    response = client.models.generate_content(
        model=config.image_model,
        contents=[full_prompt],
    )

    # Extract image from response — inline_data contains raw bytes
    for part in response.parts:
        if part.inline_data is not None:
            img = Image.open(io.BytesIO(part.inline_data.data))
            # Resize to tile_size if needed
            if img.size != (config.tile_size, config.tile_size):
                img = img.resize(
                    (config.tile_size, config.tile_size), Image.LANCZOS
                )
            return img.convert("RGBA")

    text = ""
    for part in response.parts:
        if part.text:
            text = part.text[:200]
    raise RuntimeError(f"No image in Gemini response. Text: {text}")


# --- Assembly ---


def create_fade_mask(size: int, fade_px: int) -> Image.Image:
    """Create an alpha mask with gradient fade on all four edges."""
    mask = Image.new("L", (size, size), 255)
    pixels = mask.load()

    for offset in range(fade_px):
        alpha = int(255 * offset / fade_px)
        for x in range(size):
            pixels[x, offset] = min(alpha, pixels[x, offset])
            pixels[x, size - 1 - offset] = min(alpha, pixels[x, size - 1 - offset])
        for y in range(size):
            pixels[offset, y] = min(alpha, pixels[offset, y])
            pixels[size - 1 - offset, y] = min(alpha, pixels[size - 1 - offset, y])

    return mask


def assemble_mural(
    grid: list[GridCell],
    n_rows: int,
    n_cols: int,
    tile_size: int,
    overlap: int,
) -> Image.Image:
    """Composite grid cells into a mural with alpha blending."""
    step = tile_size - overlap
    canvas_w = tile_size + (n_cols - 1) * step
    canvas_h = tile_size + (n_rows - 1) * step

    mural = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))
    mask = create_fade_mask(tile_size, overlap)

    # Composite in scanline order (same as generation order)
    for cell in sorted(grid, key=lambda c: (c.row, c.col)):
        if cell.image is None:
            continue

        blended = cell.image.copy()
        blended.putalpha(mask)

        x = cell.col * step
        y = cell.row * step
        mural.alpha_composite(blended, (x, y))

    return mural, canvas_w, canvas_h


def add_timeline_bar(
    mural: Image.Image,
    months: list[str],
    tile_size: int,
    overlap: int,
    bar_height: int = 48,
) -> Image.Image:
    """Add month labels across the bottom."""
    w, h = mural.size
    final = Image.new("RGBA", (w, h + bar_height), (0, 0, 0, 255))
    final.paste(mural, (0, 0))

    draw = ImageDraw.Draw(final)
    draw.rectangle([(0, h), (w, h + bar_height)], fill=(15, 15, 20, 255))

    step = tile_size - overlap
    for i, month in enumerate(months):
        cx = i * step + tile_size // 2
        bbox = draw.textbbox((0, 0), month)
        tw = bbox[2] - bbox[0]
        draw.text(
            (cx - tw // 2, h + bar_height // 2 - 6),
            month,
            fill=(140, 160, 180, 255),
        )

    return final


# --- Main entry point ---


def generate_mural(
    db: TrajectoryDB,
    config: Config,
    themes: list[str] | None = None,
    months: list[str] | None = None,
    dry_run: bool = False,
) -> MuralResult:
    """Generate a mural with LLM-planned layout and corner-out generation."""
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
        theme_info = theme_info[: mc.max_projects]

    theme_names = [str(t["name"]) for t in theme_info]
    logger.info("Themes (%d): %s", len(theme_info), theme_names)

    # Resolve months
    if not months:
        months = select_months_global(db, mc.max_months)
        if not months:
            raise ValueError("No analyzed events found")

    n_months = len(months)

    # Query tile data for every theme × month (keep only active)
    theme_tiles: dict[str, list[ThemeTile]] = {}  # theme -> [tiles by month]
    for ti in theme_info:
        tiles_for_theme = []
        for month in months:
            td = query_theme_tile(db, int(ti["id"]), str(ti["name"]), month)
            tiles_for_theme.append(td)
        # Only keep themes that have at least one active month
        if any(t.activity == "active" for t in tiles_for_theme):
            theme_tiles[str(ti["name"])] = tiles_for_theme

    if not theme_tiles:
        raise ValueError("No active themes found in selected months")

    # Pick the best representative tile per theme for layout planning
    # (the one with most techniques/bets)
    best_tile: dict[str, ThemeTile] = {}
    for name, tiles in theme_tiles.items():
        active = [t for t in tiles if t.activity == "active"]
        best_tile[name] = max(active, key=lambda t: len(t.techniques) + len(t.design_bets))

    # Phase 1: LLM plans the vertical arrangement
    logger.info("Planning layout for %d themes...", len(theme_tiles))
    row_assignments = plan_layout(best_tile, db, mc)

    # Sort themes by assigned row
    sorted_themes = sorted(theme_tiles.keys(), key=lambda t: row_assignments.get(t, 99))
    n_rows = len(sorted_themes)
    theme_to_row = {name: i for i, name in enumerate(sorted_themes)}

    # Build grid — all active cells
    grid: list[GridCell] = []
    grid_lookup: dict[tuple[int, int], GridCell] = {}

    for theme_name in sorted_themes:
        row = theme_to_row[theme_name]
        for col, tile in enumerate(theme_tiles[theme_name]):
            if tile.activity == "active":
                cell = GridCell(data=tile, row=row, col=col)
                grid.append(cell)
                grid_lookup[(row, col)] = cell

    logger.info("Grid: %d rows x %d cols, %d active cells", n_rows, n_months, len(grid))
    for cell in sorted(grid, key=lambda c: (c.row, c.col)):
        logger.info(
            "  [%d,%d] %s/%s — %d techniques, %d bets",
            cell.row, cell.col, cell.data.theme, cell.data.month,
            len(cell.data.techniques), len(cell.data.design_bets),
        )

    output_dir = config.resolved_db_path.parent / "mural"
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = mc.tile_size
    overlap = int(tile_size * mc.vertical_overlap)

    # Phase 2: Generate art prompts in scanline order (corner-out)
    # Each tile knows its already-generated neighbors
    scanline = sorted(grid, key=lambda c: (c.row, c.col))

    for cell in scanline:
        neighbors: dict[str, str] = {}

        # Left neighbor
        left = grid_lookup.get((cell.row, cell.col - 1))
        if left and left.art_prompt:
            neighbors["left"] = left.art_prompt

        # Above neighbor
        above = grid_lookup.get((cell.row - 1, cell.col))
        if above and above.art_prompt:
            neighbors["above"] = above.art_prompt

        cell.art_prompt = generate_tile_prompt(cell.data, mc, neighbors or None)
        logger.info(
            "  Prompt [%d,%d] %s: %s...",
            cell.row, cell.col, cell.data.theme, cell.art_prompt[:80],
        )

    # Compute canvas size
    step = tile_size - overlap
    canvas_w = tile_size + (n_months - 1) * step
    canvas_h = tile_size + (n_rows - 1) * step

    result = MuralResult(
        grid=grid,
        themes=list(sorted_themes),
        months=months,
        n_rows=n_rows,
        n_cols=n_months,
        canvas_size=(canvas_w, canvas_h),
        output_dir=output_dir,
    )

    if dry_run:
        print(f"\nMural: {n_rows} themes x {n_months} months → {len(grid)} active cells")
        print(f"Canvas: {canvas_w} x {canvas_h} px")
        print(f"Overlap: {overlap} px")
        print(f"Theme order (LLM-arranged): {', '.join(sorted_themes)}")
        print(f"Months: {', '.join(months)}")
        print(f"Est. cost: ${len(grid) * 0.04:.2f}")
        print(f"Output: {output_dir}")
        print()

        for row_idx, theme_name in enumerate(sorted_themes):
            row_cells = [c for c in scanline if c.row == row_idx]
            active_months = [months[c.col] for c in row_cells]
            print(f"Row {row_idx}: {theme_name} ({len(row_cells)} active: {', '.join(active_months)})")
            for c in row_cells:
                print(f"  [{c.row},{c.col}] {c.data.month}: {c.art_prompt[:100]}...")
            print()
        return result

    # Phase 3: Generate images in scanline order
    # llm_client import (at top) sets GEMINI_API_KEY/GOOGLE_API_KEY in env
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Ensure llm_client is imported or set the env var."
        )
    client = genai.Client(api_key=api_key)
    cost_per_tile = 0.04  # ~$0.039 for gemini-2.5-flash-image, free on AI Studio tier
    cost_estimate = len(grid) * cost_per_tile

    if cost_estimate > mc.max_cost:
        raise ValueError(
            f"Estimated cost ${cost_estimate:.2f} exceeds max_cost ${mc.max_cost:.2f}. "
            f"Reduce tile count or increase max_cost."
        )

    for cell in scanline:
        logger.info(
            "Generating [%d,%d] %s/%s",
            cell.row, cell.col, cell.data.theme, cell.data.month,
        )
        try:
            img = generate_tile_image(cell.art_prompt, mc, client)
            cell.image = img

            safe_name = cell.data.theme.replace("/", "_").replace(" ", "_")
            tile_path = output_dir / f"tile_{cell.row}_{cell.col}_{safe_name}.png"
            img.save(str(tile_path))

        except Exception as e:
            error_msg = f"[{cell.row},{cell.col}] {cell.data.theme}/{cell.data.month}: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    # Phase 4: Assemble
    mural_img, canvas_w, canvas_h = assemble_mural(grid, n_rows, n_months, tile_size, overlap)
    mural_img = add_timeline_bar(mural_img, months, tile_size, overlap)
    result.canvas_size = (canvas_w, canvas_h)

    mural_path = output_dir / "mural.png"
    mural_img.save(str(mural_path))

    generated_count = sum(1 for c in grid if c.image is not None)
    result.total_cost = generated_count * cost_per_tile

    print(f"\nMural generated: {mural_path}")
    print(f"Tiles: {generated_count}/{len(grid)} generated")
    print(f"Theme order: {', '.join(sorted_themes)}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")
    print(f"Estimated cost: ${result.total_cost:.2f}")

    return result
