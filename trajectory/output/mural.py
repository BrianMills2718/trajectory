"""Mural generator — two modes:

1. Multi-project theme×month grid (generate_mural)
2. Single-project dataflow diagram (generate_project_mural)

Both use LLM-planned layout and corner-out scanline generation.
"""

import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from PIL import Image, ImageDraw, ImageFilter

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


def smooth_seams(
    mural: Image.Image,
    n_rows: int,
    n_cols: int,
    tile_size: int,
    overlap: int,
    blur_radius: int = 12,
) -> Image.Image:
    """Apply Gaussian blur along seam lines to smooth tile boundaries.

    Creates a blurred copy of the full mural, then composites the blurred version
    only in narrow strips along the seam center lines, with a feathered mask so the
    blur fades into the sharp original.
    """
    step = tile_size - overlap
    w, h = mural.size

    blurred = mural.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Build a mask: white = use blurred, black = use original
    seam_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(seam_mask)

    seam_width = overlap // 2  # width of the blurred seam strip

    # Vertical seams (between columns)
    for col in range(1, n_cols):
        cx = col * step
        draw.rectangle(
            [(cx - seam_width // 2, 0), (cx + seam_width // 2, h)],
            fill=255,
        )

    # Horizontal seams (between rows)
    for row in range(1, n_rows):
        cy = row * step
        draw.rectangle(
            [(0, cy - seam_width // 2), (w, cy + seam_width // 2)],
            fill=255,
        )

    # Feather the seam mask so blur fades in smoothly
    seam_mask = seam_mask.filter(ImageFilter.GaussianBlur(radius=seam_width // 2))

    # Composite: original where mask is black, blurred where mask is white
    result = Image.composite(blurred, mural, seam_mask)
    return result


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


# ============================================================
# Single-project dataflow mural
# ============================================================


@dataclass
class DataflowNode:
    """A component in the dataflow graph."""

    id: str
    label: str
    x: int  # grid column
    y: int  # grid row
    description: str
    techniques: list[str] = field(default_factory=list)
    art_prompt: str = ""
    image: Image.Image | None = None


@dataclass
class DataflowEdge:
    """A directed edge in the dataflow graph."""

    from_id: str
    to_id: str
    label: str


@dataclass
class DataflowResult:
    """Output of a single-project dataflow mural."""

    project: str
    nodes: list[DataflowNode]
    edges: list[DataflowEdge]
    n_rows: int
    n_cols: int
    canvas_size: tuple[int, int]
    output_dir: Path
    total_cost: float = 0.0
    errors: list[str] = field(default_factory=list)


def query_project_data(db: TrajectoryDB, project_name: str) -> dict:
    """Gather all concept data for a single project."""
    proj = db.conn.execute(
        "SELECT id FROM projects WHERE name = ?", (project_name,)
    ).fetchone()
    if not proj:
        raise ValueError(f"Project not found: {project_name}")
    pid = proj["id"]

    result: dict = {"themes": [], "design_bets": [], "techniques": [], "decisions": [], "concept_links": []}

    for level in ("theme", "design_bet", "technique"):
        rows = db.conn.execute(
            """SELECT c.id, c.name, COUNT(ce.event_id) as event_count
               FROM concepts c
               JOIN concept_events ce ON c.id = ce.concept_id
               JOIN events e ON ce.event_id = e.id
               WHERE c.level = ? AND e.project_id = ?
               GROUP BY c.id ORDER BY event_count DESC""",
            (level, pid),
        ).fetchall()
        key = level + "s" if level != "technique" else "techniques"
        result[key] = [{"name": r["name"], "event_count": r["event_count"]} for r in rows]

    # Decisions
    decs = db.conn.execute(
        """SELECT DISTINCT d.title FROM decisions d
           JOIN events e ON d.event_id = e.id
           WHERE e.project_id = ? LIMIT 20""",
        (pid,),
    ).fetchall()
    result["decisions"] = [d["title"] for d in decs]

    # Concept links involving this project's concepts
    links = db.conn.execute(
        """SELECT c1.name as a_name, cl.relationship, c2.name as b_name
           FROM concept_links cl
           JOIN concepts c1 ON cl.concept_a_id = c1.id
           JOIN concepts c2 ON cl.concept_b_id = c2.id
           WHERE c1.id IN (SELECT concept_id FROM concept_events ce JOIN events e ON ce.event_id=e.id WHERE e.project_id=?)
              OR c2.id IN (SELECT concept_id FROM concept_events ce JOIN events e ON ce.event_id=e.id WHERE e.project_id=?)""",
        (pid, pid),
    ).fetchall()
    result["concept_links"] = [{"a": l["a_name"], "rel": l["relationship"], "b": l["b_name"]} for l in links]

    return result


def plan_dataflow_layout(
    project_name: str,
    project_data: dict,
    config: MuralConfig,
) -> tuple[list[DataflowNode], list[DataflowEdge]]:
    """Ask LLM to create a dataflow graph with spatial positions."""
    messages = render_prompt(
        PROMPTS_DIR / "dataflow_layout.yaml",
        project_name=project_name,
        themes=project_data["themes"],
        design_bets=project_data["design_bets"],
        techniques=project_data["techniques"],
        decisions=project_data["decisions"],
        concept_links=project_data["concept_links"],
    )

    result = call_llm(
        config.prompt_model,
        messages,
        task="trajectory.mural.dataflow_layout",
        trace_id=f"trajectory.mural.dataflow_layout.{project_name}",
        max_budget=0,
    )

    content = result.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[:-3]

    graph = json.loads(content)

    # Build technique lookup by concept name
    technique_by_name: dict[str, list[str]] = {}
    for t in project_data["themes"]:
        technique_by_name[t["name"]] = [
            tech["name"] for tech in project_data["techniques"][:5]
        ]

    nodes = []
    for n in graph["nodes"]:
        node = DataflowNode(
            id=n["id"],
            label=n["label"],
            x=n["x"],
            y=n["y"],
            description=n["description"],
            techniques=technique_by_name.get(n["id"], []),
        )
        nodes.append(node)

    edges = []
    for e in graph["edges"]:
        edges.append(DataflowEdge(from_id=e["from"], to_id=e["to"], label=e["label"]))

    return nodes, edges


def generate_dataflow_tile_prompt(
    node: DataflowNode,
    edges: list[DataflowEdge],
    all_nodes: dict[str, DataflowNode],
    config: MuralConfig,
    neighbors: dict[str, str] | None = None,
) -> str:
    """Generate an art prompt for a dataflow node tile."""
    # Find incoming/outgoing edges
    edges_in = []
    for e in edges:
        if e.to_id == node.id and e.from_id in all_nodes:
            edges_in.append({
                "from_label": all_nodes[e.from_id].label,
                "label": e.label,
            })

    edges_out = []
    for e in edges:
        if e.from_id == node.id and e.to_id in all_nodes:
            edges_out.append({
                "to_label": all_nodes[e.to_id].label,
                "label": e.label,
            })

    messages = render_prompt(
        PROMPTS_DIR / "dataflow_tile.yaml",
        node_label=node.label,
        node_description=node.description,
        techniques=node.techniques,
        edges_in=edges_in,
        edges_out=edges_out,
        neighbors=neighbors or {},
    )

    result = call_llm(
        config.prompt_model,
        messages,
        task="trajectory.mural.dataflow_tile",
        trace_id=f"trajectory.mural.dataflow_tile.{node.id}",
        max_budget=0,
    )

    return f"{result.content.strip()} Style: {config.style_suffix}"


def generate_project_mural(
    db: TrajectoryDB,
    config: Config,
    project_name: str,
    dry_run: bool = False,
) -> DataflowResult:
    """Generate a single-project dataflow mural."""
    mc = config.mural

    # Step 1: Query project data
    logger.info("Querying data for %s...", project_name)
    project_data = query_project_data(db, project_name)

    print(f"Project: {project_name}")
    print(f"  Themes: {len(project_data['themes'])}")
    print(f"  Design bets: {len(project_data['design_bets'])}")
    print(f"  Techniques: {len(project_data['techniques'])}")
    print(f"  Decisions: {len(project_data['decisions'])}")
    print(f"  Concept links: {len(project_data['concept_links'])}")

    # Step 2: LLM generates dataflow layout
    logger.info("Generating dataflow layout...")
    nodes, edges = plan_dataflow_layout(project_name, project_data, mc)

    # Build spatial index
    node_by_id = {n.id: n for n in nodes}
    node_at: dict[tuple[int, int], DataflowNode] = {(n.x, n.y): n for n in nodes}

    n_cols = max(n.x for n in nodes) + 1
    n_rows = max(n.y for n in nodes) + 1

    print(f"\nDataflow graph: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Grid: {n_cols} cols x {n_rows} rows")
    for node in sorted(nodes, key=lambda n: (n.y, n.x)):
        print(f"  [{node.x},{node.y}] {node.label}: {node.description[:60]}...")

    print(f"\nEdges:")
    for edge in edges:
        from_label = node_by_id[edge.from_id].label if edge.from_id in node_by_id else edge.from_id
        to_label = node_by_id[edge.to_id].label if edge.to_id in node_by_id else edge.to_id
        print(f"  {from_label} --[{edge.label}]--> {to_label}")

    output_dir = config.resolved_db_path.parent / "mural"
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = mc.tile_size
    overlap = int(tile_size * mc.vertical_overlap)

    # Step 3: Generate art prompts in scanline order (top-left → right → down)
    scanline = sorted(nodes, key=lambda n: (n.y, n.x))

    for node in scanline:
        neighbors: dict[str, str] = {}
        left = node_at.get((node.x - 1, node.y))
        if left and left.art_prompt:
            neighbors["left"] = left.art_prompt
        above = node_at.get((node.x, node.y - 1))
        if above and above.art_prompt:
            neighbors["above"] = above.art_prompt

        node.art_prompt = generate_dataflow_tile_prompt(
            node, edges, node_by_id, mc, neighbors or None
        )
        logger.info("  Prompt [%d,%d] %s: %s...", node.x, node.y, node.label, node.art_prompt[:80])

    # Compute canvas size
    step = tile_size - overlap
    canvas_w = tile_size + (n_cols - 1) * step
    canvas_h = tile_size + (n_rows - 1) * step

    result = DataflowResult(
        project=project_name,
        nodes=nodes,
        edges=edges,
        n_rows=n_rows,
        n_cols=n_cols,
        canvas_size=(canvas_w, canvas_h),
        output_dir=output_dir,
    )

    if dry_run:
        print(f"\nCanvas: {canvas_w} x {canvas_h} px")
        print(f"Overlap: {overlap} px")
        print()
        for node in scanline:
            print(f"[{node.x},{node.y}] {node.label}:")
            print(f"  {node.art_prompt[:120]}...")
            print()
        return result

    # Step 4: Generate images
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Ensure llm_client is imported or set the env var."
        )
    client = genai.Client(api_key=api_key)

    for node in scanline:
        logger.info("Generating [%d,%d] %s", node.x, node.y, node.label)
        try:
            img = generate_tile_image(node.art_prompt, mc, client)
            node.image = img

            safe_name = node.id.replace("/", "_").replace(" ", "_")
            tile_path = output_dir / f"df_{node.x}_{node.y}_{safe_name}.png"
            img.save(str(tile_path))
            print(f"  Generated: {tile_path}")

        except Exception as e:
            error_msg = f"[{node.x},{node.y}] {node.label}: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

    # Step 5: Assemble — convert nodes to GridCell for reuse
    grid_cells = []
    for node in nodes:
        if node.image is not None:
            cell = GridCell(
                data=ThemeTile(
                    theme=node.label, month="", activity="active",
                    techniques=[], design_bets=[], decisions=[], event_count=0,
                ),
                row=node.y,
                col=node.x,
                image=node.image,
            )
            grid_cells.append(cell)

    mural_img, canvas_w, canvas_h = assemble_mural(
        grid_cells, n_rows, n_cols, tile_size, overlap
    )
    mural_img = smooth_seams(mural_img, n_rows, n_cols, tile_size, overlap)
    result.canvas_size = (canvas_w, canvas_h)

    mural_path = output_dir / f"dataflow_{project_name}.png"
    mural_img.save(str(mural_path))

    generated_count = sum(1 for n in nodes if n.image is not None)
    print(f"\nDataflow mural: {mural_path}")
    print(f"Tiles: {generated_count}/{len(nodes)} generated")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")

    return result
