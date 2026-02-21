"""Developer Wrapped — shareable visualization of project activity.

Generates a single image card with:
- Treemap of themes (sized by event count)
- Project × month heatmap grid
- Bold stats
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from trajectory.config import Config
from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)

# --- Fonts ---

_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_FONT_MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        path = _FONT_MONO_BOLD if bold else _FONT_MONO
    else:
        path = _FONT_BOLD if bold else _FONT_REGULAR
    return ImageFont.truetype(path, size)


# --- Colors ---

BG = (13, 17, 23)
CARD_BG = (22, 27, 34)
TEXT = (230, 237, 243)
TEXT_DIM = (110, 118, 129)
DIVIDER = (48, 54, 61)

# Palette — muted but distinct, good on dark bg
PALETTE = [
    (88, 166, 255),    # blue
    (63, 185, 80),     # green
    (210, 153, 34),    # gold
    (201, 97, 152),    # pink
    (174, 124, 255),   # purple
    (255, 123, 114),   # coral
    (121, 192, 255),   # light blue
    (87, 171, 90),     # dark green
    (219, 171, 9),     # amber
    (255, 166, 87),    # orange
    (163, 113, 247),   # violet
    (255, 203, 107),   # yellow
]

# Heatmap color ramp (0 → max)
HEAT_EMPTY = (22, 27, 34)
HEAT_RAMP = [
    (14, 68, 41),
    (0, 109, 50),
    (38, 166, 65),
    (63, 185, 80),
    (116, 236, 120),
]


def _heat_color(value: float) -> tuple[int, int, int]:
    """Map 0.0-1.0 to heatmap color."""
    if value <= 0:
        return HEAT_EMPTY
    idx = min(value * len(HEAT_RAMP), len(HEAT_RAMP) - 1)
    return HEAT_RAMP[int(idx)]


# --- Data ---


@dataclass
class WrappedData:
    total_events: int = 0
    total_projects: int = 0
    analyzed_projects: int = 0
    total_concepts: int = 0
    total_decisions: int = 0
    date_first: str = ""
    date_last: str = ""
    top_themes: list[dict] = field(default_factory=list)  # name, events
    all_themes: list[dict] = field(default_factory=list)   # for treemap
    project_month_grid: dict = field(default_factory=dict)  # {project: {month: count}}
    months: list[str] = field(default_factory=list)
    projects_ranked: list[str] = field(default_factory=list)
    level_counts: dict[str, int] = field(default_factory=dict)
    concept_links: int = 0


def query_wrapped_data(db: TrajectoryDB) -> WrappedData:
    d = WrappedData()

    projects = db.list_projects()
    d.total_projects = len(projects)
    d.total_events = sum(db.count_events(p.id) for p in projects)

    d.analyzed_projects = db.conn.execute(
        "SELECT COUNT(DISTINCT e.project_id) FROM events e WHERE e.analysis_run_id IS NOT NULL"
    ).fetchone()[0]

    levels = db.conn.execute("SELECT level, COUNT(*) as c FROM concepts GROUP BY level").fetchall()
    d.level_counts = {r["level"]: r["c"] for r in levels}
    d.total_concepts = sum(d.level_counts.values())

    d.total_decisions = db.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    dates = db.conn.execute("SELECT MIN(timestamp) as first, MAX(timestamp) as last FROM events").fetchone()
    d.date_first = (dates["first"] or "")[:10]
    d.date_last = (dates["last"] or "")[:10]

    # All themes for treemap
    d.all_themes = [
        dict(r) for r in db.conn.execute(
            """SELECT c.name, COUNT(DISTINCT ce.event_id) as events,
                      COUNT(DISTINCT e.project_id) as projects
               FROM concepts c
               JOIN concept_events ce ON c.id = ce.concept_id
               JOIN events e ON ce.event_id = e.id
               WHERE c.level = 'theme'
               GROUP BY c.id ORDER BY events DESC"""
        ).fetchall()
    ]
    d.top_themes = d.all_themes[:6]

    # Project × month grid
    rows = db.conn.execute(
        """SELECT p.name, strftime('%Y-%m', e.timestamp) as month, COUNT(*) as events
           FROM events e JOIN projects p ON e.project_id = p.id
           WHERE e.analysis_run_id IS NOT NULL
           GROUP BY p.name, month ORDER BY p.name, month"""
    ).fetchall()

    grid: dict[str, dict[str, int]] = defaultdict(dict)
    all_months: set[str] = set()
    for r in rows:
        grid[r["name"]][r["month"]] = r["events"]
        all_months.add(r["month"])

    d.months = sorted(all_months)
    d.projects_ranked = sorted(grid.keys(), key=lambda p: sum(grid[p].values()), reverse=True)
    d.project_month_grid = dict(grid)

    d.concept_links = db.conn.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0]

    return d


# --- Treemap layout (squarified) ---


def _squarify(values: list[float], x: float, y: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Simple squarified treemap layout. Returns list of (x, y, w, h) rects."""
    if not values:
        return []
    if len(values) == 1:
        return [(x, y, w, h)]

    total = sum(values)
    if total <= 0:
        return [(x, y, w, h)] * len(values)

    rects: list[tuple[float, float, float, float]] = []

    # Split into two groups to maintain good aspect ratios
    if w >= h:
        # Split vertically
        left_sum = 0.0
        split = 0
        for i, v in enumerate(values):
            left_sum += v
            if left_sum >= total / 2:
                split = i + 1
                break
        if split == 0:
            split = 1

        left_w = w * (left_sum / total) if total > 0 else w / 2
        rects.extend(_squarify(values[:split], x, y, left_w, h))
        rects.extend(_squarify(values[split:], x + left_w, y, w - left_w, h))
    else:
        # Split horizontally
        top_sum = 0.0
        split = 0
        for i, v in enumerate(values):
            top_sum += v
            if top_sum >= total / 2:
                split = i + 1
                break
        if split == 0:
            split = 1

        top_h = h * (top_sum / total) if total > 0 else h / 2
        rects.extend(_squarify(values[:split], x, y, w, top_h))
        rects.extend(_squarify(values[split:], x, y + top_h, w, h - top_h))

    return rects


def _draw_treemap(
    draw: ImageDraw.ImageDraw,
    themes: list[dict],
    x0: int, y0: int, w: int, h: int,
    gap: int = 3,
) -> None:
    """Draw a treemap of themes."""
    if not themes:
        return

    values = [float(t["events"]) for t in themes]
    rects = _squarify(values, x0, y0, w, h)

    for i, ((rx, ry, rw, rh), theme) in enumerate(zip(rects, themes)):
        color = PALETTE[i % len(PALETTE)]
        # Muted fill
        fill = (color[0] // 3, color[1] // 3, color[2] // 3)

        irx, iry = int(rx) + gap, int(ry) + gap
        irw, irh = int(rw) - gap * 2, int(rh) - gap * 2
        if irw < 4 or irh < 4:
            continue

        draw.rounded_rectangle(
            (irx, iry, irx + irw, iry + irh),
            radius=6, fill=fill, outline=color, width=2,
        )

        # Label
        name = theme["name"].replace("_", " ")
        count = str(theme["events"])

        # Pick font size based on rect area
        area = irw * irh
        if area > 40000:
            name_size, count_size = 18, 13
        elif area > 15000:
            name_size, count_size = 14, 11
        elif area > 5000:
            name_size, count_size = 11, 9
        else:
            name_size, count_size = 9, 0

        # Truncate name to fit
        name_font = _font(name_size, bold=True)
        while name and draw.textbbox((0, 0), name, font=name_font)[2] > irw - 12:
            name = name[:-1]

        if name:
            draw.text((irx + 8, iry + 6), name, font=name_font, fill=color)

        if count_size > 0 and irh > name_size + count_size + 16:
            draw.text(
                (irx + 8, iry + 6 + name_size + 4),
                count, font=_font(count_size), fill=TEXT_DIM,
            )


def _draw_heatmap(
    draw: ImageDraw.ImageDraw,
    data: WrappedData,
    x0: int, y0: int, w: int, h: int,
    max_projects: int = 10,
) -> int:
    """Draw project × month heatmap grid. Returns y after drawing."""
    projects = data.projects_ranked[:max_projects]
    months = data.months
    if not projects or not months:
        return y0

    # Layout
    label_w = 180  # space for project names
    grid_w = w - label_w
    cell_w = grid_w // len(months)
    cell_h = min(22, (h - 30) // len(projects))  # leave room for month labels
    gap = 2

    # Find global max for color scaling
    max_val = 1
    for p in projects:
        for m in months:
            max_val = max(max_val, data.project_month_grid.get(p, {}).get(m, 0))

    y = y0
    for pi, proj in enumerate(projects):
        # Project label
        name = proj
        if len(name) > 20:
            name = name[:18] + ".."
        draw.text(
            (x0, y + 3), name,
            font=_font(11, mono=True), fill=TEXT_DIM,
        )

        # Cells
        for mi, month in enumerate(months):
            cx = x0 + label_w + mi * cell_w
            val = data.project_month_grid.get(proj, {}).get(month, 0)
            intensity = val / max_val if max_val > 0 else 0

            color = _heat_color(intensity)
            draw.rounded_rectangle(
                (cx, y, cx + cell_w - gap, y + cell_h - gap),
                radius=3, fill=color,
            )

            # Show count in hot cells
            if val > 0 and cell_w > 24:
                count_str = str(val)
                bbox = draw.textbbox((0, 0), count_str, font=_font(8))
                tw = bbox[2] - bbox[0]
                if tw < cell_w - 6:
                    text_color = TEXT if intensity > 0.3 else TEXT_DIM
                    draw.text(
                        (cx + (cell_w - gap - tw) // 2, y + 3),
                        count_str, font=_font(8), fill=text_color,
                    )

        y += cell_h

    # Month labels at bottom
    y += 4
    for mi, month in enumerate(months):
        cx = x0 + label_w + mi * cell_w
        # Only show every Nth label to avoid crowding
        if len(months) > 8 and mi % 2 != 0 and mi != len(months) - 1:
            continue
        label = month[2:]  # "24-11" instead of "2024-11"
        draw.text((cx, y), label, font=_font(8), fill=TEXT_DIM)

    return y + 16


# --- Main ---


def generate_wrapped(
    db: TrajectoryDB,
    config: Config,
    output_path: Path | None = None,
) -> Path:
    data = query_wrapped_data(db)

    W, H = 1080, 1920  # Full phone screen
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    PAD = 40
    y = PAD

    # === TITLE ===
    draw.text((PAD, y), "TRAJECTORY", font=_font(42, bold=True), fill=TEXT)
    bbox = draw.textbbox((0, 0), data.date_first[:4] + "–" + data.date_last[:4], font=_font(42, bold=True))
    year_str = data.date_first[:4] + "–" + data.date_last[:4]
    draw.text(
        (W - PAD - (bbox[2] - bbox[0]), y),
        year_str, font=_font(42, bold=True), fill=TEXT_DIM,
    )
    y += 56

    # === STATS BAR ===
    stats = [
        (f"{data.total_events:,}", "events"),
        (str(data.total_projects), "projects"),
        (str(data.total_concepts), "concepts"),
    ]
    stat_w = (W - 2 * PAD) // len(stats)
    for i, (num, label) in enumerate(stats):
        sx = PAD + i * stat_w
        draw.text((sx, y), num, font=_font(32, bold=True, mono=True), fill=PALETTE[i])
        draw.text((sx, y + 38), label, font=_font(12), fill=TEXT_DIM)
    y += 68

    # === TREEMAP ===
    draw.line([(PAD, y), (W - PAD, y)], fill=DIVIDER, width=1)
    y += 8
    draw.text((PAD, y), "THEMES", font=_font(11, bold=True), fill=TEXT_DIM)
    y += 20

    treemap_h = 340
    _draw_treemap(draw, data.all_themes[:16], PAD, y, W - 2 * PAD, treemap_h)
    y += treemap_h + 16

    # === HEATMAP ===
    draw.line([(PAD, y), (W - PAD, y)], fill=DIVIDER, width=1)
    y += 8
    draw.text((PAD, y), "PROJECT ACTIVITY", font=_font(11, bold=True), fill=TEXT_DIM)
    y += 20

    heatmap_h = 320
    y = _draw_heatmap(draw, data, PAD, y, W - 2 * PAD, heatmap_h, max_projects=12)
    y += 16

    # === CONCEPT BREAKDOWN ===
    draw.line([(PAD, y), (W - PAD, y)], fill=DIVIDER, width=1)
    y += 12

    # Three big numbers in boxes
    box_w = (W - 2 * PAD - 20) // 3
    box_h = 80
    for i, (count, label, color) in enumerate([
        (data.level_counts.get("theme", 0), "themes", PALETTE[0]),
        (data.level_counts.get("design_bet", 0), "design\nbets", PALETTE[1]),
        (data.level_counts.get("technique", 0), "tech-\nniques", PALETTE[2]),
    ]):
        bx = PAD + i * (box_w + 10)
        draw.rounded_rectangle(
            (bx, y, bx + box_w, y + box_h),
            radius=8, fill=CARD_BG,
        )
        draw.text((bx + 14, y + 8), str(count), font=_font(30, bold=True, mono=True), fill=color)
        draw.text((bx + 14, y + 46), label.replace("\n", " "), font=_font(11), fill=TEXT_DIM)
    y += box_h + 12

    # Decisions count
    draw.text(
        (PAD, y),
        f"{data.total_decisions} architectural decisions tracked",
        font=_font(13), fill=TEXT_DIM,
    )
    y += 20
    if data.concept_links:
        draw.text(
            (PAD, y),
            f"{data.concept_links} cross-project concept links",
            font=_font(13), fill=PALETTE[4],
        )
        y += 20

    # === TOP PROJECTS RANKING ===
    y += 8
    draw.line([(PAD, y), (W - PAD, y)], fill=DIVIDER, width=1)
    y += 8
    draw.text((PAD, y), "TOP PROJECTS", font=_font(11, bold=True), fill=TEXT_DIM)
    y += 22

    bar_area = W - 2 * PAD - 200
    if data.projects_ranked:
        max_proj = max(
            sum(data.project_month_grid.get(p, {}).values())
            for p in data.projects_ranked[:6]
        )
        for i, proj in enumerate(data.projects_ranked[:6]):
            total = sum(data.project_month_grid.get(proj, {}).values())
            color = PALETTE[(i + 3) % len(PALETTE)]

            name = proj
            if len(name) > 22:
                name = name[:20] + ".."

            # Rank number
            draw.text((PAD, y + 1), f"{i+1}.", font=_font(13, bold=True, mono=True), fill=TEXT_DIM)

            # Name
            draw.text((PAD + 30, y + 1), name, font=_font(13, mono=True), fill=TEXT)

            # Bar
            bar_x = PAD + 200
            bar_frac = total / max_proj if max_proj > 0 else 0
            bar_px = max(6, int(bar_area * bar_frac))
            draw.rounded_rectangle(
                (bar_x, y + 2, bar_x + bar_px, y + 18),
                radius=4, fill=color,
            )

            # Count
            count_str = f"{total}"
            bbox = draw.textbbox((0, 0), count_str, font=_font(10, bold=True))
            cw = bbox[2] - bbox[0]
            if bar_px > cw + 16:
                draw.text((bar_x + 6, y + 4), count_str, font=_font(10, bold=True), fill=BG)
            else:
                draw.text((bar_x + bar_px + 6, y + 4), count_str, font=_font(10, bold=True), fill=color)

            y += 26

    # === FOOTER ===
    y = H - 50
    draw.line([(PAD, y), (W - PAD, y)], fill=DIVIDER, width=1)
    y += 14
    draw.text((PAD, y), "trajectory", font=_font(16, bold=True), fill=TEXT_DIM)
    draw.text(
        (PAD + 110, y + 3),
        "// how ideas emerge, evolve, and spread across your code",
        font=_font(11), fill=DIVIDER,
    )

    # Save
    output_dir = config.resolved_db_path.parent / "mural"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / "wrapped.png"

    img.save(str(output_path), quality=95)
    print(f"Wrapped card: {output_path}")
    return output_path
