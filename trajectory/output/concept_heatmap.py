"""Concept heatmap — GitHub green-squares style grid for concept activity.

Rows = concepts (grouped by level, sorted by importance).
Columns = time buckets (auto-granularity: days/weeks/months).
Cell intensity = event count in that bucket.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)

# --- Fonts ---

_FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        path = _FONT_MONO
    else:
        path = _FONT_BOLD if bold else _FONT_REGULAR
    return ImageFont.truetype(path, size)


# --- Colors ---

BG = (13, 17, 23)
TEXT = (230, 237, 243)
TEXT_DIM = (110, 118, 129)
DIVIDER = (48, 54, 61)
CELL_EMPTY = (22, 27, 34)

# Per-level color ramps (empty → low → high), 5 steps each
LEVEL_RAMPS: dict[str, list[tuple[int, int, int]]] = {
    "theme": [
        (14, 41, 68),
        (0, 70, 130),
        (30, 120, 200),
        (88, 166, 255),
        (150, 210, 255),
    ],
    "design_bet": [
        (45, 25, 55),
        (80, 50, 110),
        (130, 80, 180),
        (174, 124, 255),
        (210, 180, 255),
    ],
    "technique": [
        (14, 68, 41),
        (0, 109, 50),
        (38, 166, 65),
        (63, 185, 80),
        (116, 236, 120),
    ],
}

# Fallback for unknown levels
DEFAULT_RAMP = LEVEL_RAMPS["technique"]


def _heat_color(count: int, max_count: int, level: str) -> tuple[int, int, int]:
    """Map event count to color using per-level ramp."""
    if count == 0:
        return CELL_EMPTY
    ramp = LEVEL_RAMPS.get(level, DEFAULT_RAMP)
    # Clamp to [1, max_count], map to ramp index
    t = min(count / max(max_count, 1), 1.0)
    idx = min(int(t * (len(ramp) - 1)), len(ramp) - 1)
    return ramp[idx]


# --- Granularity ---

def _pick_granularity(first: str, last: str) -> str:
    """Auto-pick time granularity based on span.

    < 60 days → 'day'
    < 365 days → 'week'
    else → 'month'
    """
    try:
        d0 = datetime.fromisoformat(first)
        d1 = datetime.fromisoformat(last)
        span_days = (d1 - d0).days
    except (ValueError, TypeError):
        return "month"

    if span_days < 60:
        return "day"
    elif span_days < 365:
        return "week"
    return "month"


def _bucket_key(timestamp: str, granularity: str) -> str:
    """Convert a timestamp to a bucket key."""
    try:
        dt = datetime.fromisoformat(timestamp)
    except (ValueError, TypeError):
        return "?"

    if granularity == "day":
        return dt.strftime("%Y-%m-%d")
    elif granularity == "week":
        # ISO week: Monday-start
        return dt.strftime("%Y-W%W")
    else:
        return dt.strftime("%Y-%m")


def _bucket_label(key: str, granularity: str, include_year: bool = False) -> str:
    """Short label for a time bucket."""
    if granularity == "day":
        # "2026-02-09" → "Feb 9"
        try:
            dt = datetime.strptime(key, "%Y-%m-%d")
            return dt.strftime("%b %-d")
        except ValueError:
            return key
    elif granularity == "week":
        # "2026-W06" → "W6"
        return key.split("-")[1] if "-" in key else key
    else:
        # "2026-02" → "Feb '26" or "Feb"
        try:
            dt = datetime.strptime(key, "%Y-%m")
            if include_year:
                return dt.strftime("%b '%y")
            return dt.strftime("%b")
        except ValueError:
            return key


def _generate_all_buckets(first: str, last: str, granularity: str) -> list[str]:
    """Generate all bucket keys between first and last, inclusive."""
    try:
        d0 = datetime.fromisoformat(first)
        d1 = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return []

    buckets: list[str] = []
    seen: set[str] = set()

    if granularity == "day":
        current = d0
        while current <= d1:
            key = current.strftime("%Y-%m-%d")
            if key not in seen:
                buckets.append(key)
                seen.add(key)
            current += timedelta(days=1)
    elif granularity == "week":
        current = d0
        while current <= d1 + timedelta(days=6):
            key = current.strftime("%Y-W%W")
            if key not in seen:
                buckets.append(key)
                seen.add(key)
            current += timedelta(days=7)
    else:
        current = d0.replace(day=1)
        while current <= d1:
            key = current.strftime("%Y-%m")
            if key not in seen:
                buckets.append(key)
                seen.add(key)
            # Next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

    return buckets


# --- Data ---

@dataclass
class ConceptHeatmapData:
    """All data needed to render a concept heatmap."""
    project_name: str
    granularity: str
    buckets: list[str]  # time bucket keys in order
    bucket_labels: list[str]  # display labels for buckets
    # Grouped by level, each entry: (concept_name, event_count, {bucket: count})
    groups: dict[str, list[tuple[str, float, dict[str, int]]]] = field(default_factory=dict)
    # Total events per bucket (for summary row)
    bucket_totals: dict[str, int] = field(default_factory=dict)
    max_count: int = 1
    total_events: int = 0
    total_concepts: int = 0
    span_days: int = 0


def query_heatmap_data(
    db: TrajectoryDB,
    project_id: int,
    max_concepts: int = 60,
) -> ConceptHeatmapData:
    """Query concept activity data at appropriate granularity."""
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    # Get time span — use p5/p95 only for long spans to avoid outliers stretching the grid.
    # For short spans (<90 days raw), use full min/max.
    all_ts = db.conn.execute(
        "SELECT timestamp FROM events WHERE project_id = ? ORDER BY timestamp",
        (project_id,),
    ).fetchall()

    if not all_ts:
        raise ValueError(f"No events for project {project_id}")

    raw_first = all_ts[0]["timestamp"]
    raw_last = all_ts[-1]["timestamp"]
    try:
        raw_span_days = (datetime.fromisoformat(raw_last) - datetime.fromisoformat(raw_first)).days
    except (ValueError, TypeError):
        raw_span_days = 0

    if raw_span_days > 90:
        # Trim outliers for long spans
        n = len(all_ts)
        p5_idx = max(0, int(n * 0.05))
        p95_idx = min(n - 1, int(n * 0.95))
        span = {
            "first": all_ts[p5_idx]["timestamp"],
            "last": all_ts[p95_idx]["timestamp"],
        }
    else:
        span = {"first": raw_first, "last": raw_last}

    granularity = _pick_granularity(span["first"], span["last"])
    buckets = _generate_all_buckets(span["first"], span["last"], granularity)
    # Include year in month labels when span crosses year boundary
    include_year = (granularity == "month" and span["first"][:4] != span["last"][:4])
    bucket_labels = [_bucket_label(b, granularity, include_year=include_year) for b in buckets]

    # Get per-concept, per-timestamp data
    rows = db.conn.execute("""
        SELECT c.name, c.level, c.importance,
               e.timestamp, COUNT(*) as cnt
        FROM concept_events ce
        JOIN concepts c ON ce.concept_id = c.id
        JOIN events e ON ce.event_id = e.id
        WHERE e.project_id = ?
        GROUP BY c.id, e.timestamp
    """, (project_id,)).fetchall()

    # Aggregate into buckets
    concept_data: dict[str, dict[str, object]] = {}  # name → {level, total_count, buckets: {key: count}}
    for row in rows:
        name = row["name"]
        if name not in concept_data:
            concept_data[name] = {
                "level": row["level"] or "technique",
                "total_count": 0,
                "buckets": defaultdict(int),
            }
        bkey = _bucket_key(row["timestamp"], granularity)
        concept_data[name]["buckets"][bkey] += row["cnt"]  # type: ignore[index]
        concept_data[name]["total_count"] = int(concept_data[name]["total_count"]) + row["cnt"]  # type: ignore[arg-type]

    # Sort by project-local event count (not cross-project importance), take top N.
    # Exclude concepts with only 1 event — not meaningful for a heatmap.
    sorted_concepts = sorted(
        ((name, cd) for name, cd in concept_data.items() if int(cd["total_count"]) >= 2),  # type: ignore[arg-type]
        key=lambda x: -int(x[1]["total_count"]),  # type: ignore[arg-type]
    )[:max_concepts]

    # Group by level
    groups: dict[str, list[tuple[str, float, dict[str, int]]]] = {
        "theme": [],
        "design_bet": [],
        "technique": [],
    }
    max_count = 1
    total_events = 0

    for name, cdata in sorted_concepts:
        level = str(cdata["level"])
        total_count = float(cdata["total_count"])  # type: ignore[arg-type]
        bucket_counts = dict(cdata["buckets"])  # type: ignore[arg-type]
        if level not in groups:
            groups[level] = []
        groups[level].append((name, total_count, bucket_counts))
        for c in bucket_counts.values():
            max_count = max(max_count, c)
            total_events += c

    # Compute bucket totals (all events per bucket, not just top concepts)
    total_rows = db.conn.execute("""
        SELECT e.timestamp, COUNT(*) as cnt
        FROM events e
        WHERE e.project_id = ?
        GROUP BY e.timestamp
    """, (project_id,)).fetchall()
    bucket_totals: dict[str, int] = defaultdict(int)
    for row in total_rows:
        bkey = _bucket_key(row["timestamp"], granularity)
        bucket_totals[bkey] += row["cnt"]

    # Compute span
    try:
        d0 = datetime.fromisoformat(span["first"])
        d1 = datetime.fromisoformat(span["last"])
        span_days = (d1 - d0).days
    except (ValueError, TypeError):
        span_days = 0

    return ConceptHeatmapData(
        project_name=project.name,
        granularity=granularity,
        buckets=buckets,
        bucket_labels=bucket_labels,
        groups=groups,
        max_count=max_count,
        total_events=total_events,
        bucket_totals=dict(bucket_totals),
        total_concepts=len(sorted_concepts),
        span_days=span_days,
    )


# --- Rendering ---

CELL_SIZE = 16
CELL_GAP = 3
LABEL_WIDTH = 260
HEADER_HEIGHT = 80
GROUP_HEADER_HEIGHT = 24
COL_HEADER_HEIGHT = 50
PADDING = 20
LEGEND_HEIGHT = 40


def render_heatmap(data: ConceptHeatmapData, output_path: Path) -> Path:
    """Render the concept heatmap as a PNG image."""
    n_cols = len(data.buckets)
    cell_step = CELL_SIZE + CELL_GAP

    # Count total rows across groups (only non-empty groups)
    active_groups = [(level, concepts) for level, concepts in data.groups.items() if concepts]
    n_rows = sum(len(concepts) for _, concepts in active_groups)
    n_group_headers = len(active_groups)

    # Calculate dimensions
    grid_width = n_cols * cell_step
    width = PADDING + LABEL_WIDTH + grid_width + PADDING
    summary_row_height = cell_step + 6  # activity summary row
    height = (
        PADDING + HEADER_HEIGHT + COL_HEADER_HEIGHT
        + summary_row_height
        + n_group_headers * GROUP_HEADER_HEIGHT
        + n_rows * cell_step
        + LEGEND_HEIGHT + PADDING
    )

    # Ensure minimum width
    width = max(width, 600)

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # --- Header ---
    y = PADDING
    draw.text(
        (PADDING, y),
        data.project_name,
        font=_font(22, bold=True),
        fill=TEXT,
    )
    y += 28

    gran_label = {"day": "daily", "week": "weekly", "month": "monthly"}[data.granularity]
    subtitle = (
        f"{data.total_concepts} concepts · {data.total_events} events · "
        f"{data.span_days} days · {gran_label}"
    )
    draw.text(
        (PADDING, y),
        subtitle,
        font=_font(12),
        fill=TEXT_DIM,
    )
    y += 20

    # Thin divider
    draw.line([(PADDING, y + 6), (width - PADDING, y + 6)], fill=DIVIDER, width=1)
    y = PADDING + HEADER_HEIGHT

    # --- Column headers (time bucket labels) ---
    col_font = _font(9, mono=True)
    grid_x0 = PADDING + LABEL_WIDTH

    for i, label in enumerate(data.bucket_labels):
        cx = grid_x0 + i * cell_step + CELL_SIZE // 2
        # Rotate text — show every Nth to avoid overlap
        # For ≤20 columns, show all; otherwise skip based on cell width
        skip = 1 if n_cols <= 20 else max(1, int(math.ceil(36 / cell_step)))
        if i % skip == 0 or i == n_cols - 1:
            txt_img = Image.new("RGBA", (60, 14), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((0, 0), label, font=col_font, fill=TEXT_DIM)
            rotated = txt_img.rotate(55, expand=True, resample=Image.BICUBIC)
            img.paste(rotated, (cx - rotated.width // 2, y), rotated)

    y += COL_HEADER_HEIGHT

    # --- Activity summary row ---
    draw.text(
        (PADDING + 4, y + 1),
        "ALL EVENTS",
        font=_font(10, bold=True),
        fill=TEXT_DIM,
    )
    max_total = max(data.bucket_totals.values()) if data.bucket_totals else 1
    for j, bkey in enumerate(data.buckets):
        count = data.bucket_totals.get(bkey, 0)
        # Use a white/gray ramp for the summary row
        if count == 0:
            color = CELL_EMPTY
        else:
            t = min(count / max(max_total, 1), 1.0)
            gray = int(40 + t * 160)
            color = (gray, gray, gray)
        x0 = grid_x0 + j * cell_step
        draw.rounded_rectangle(
            [x0, y, x0 + CELL_SIZE, y + CELL_SIZE],
            radius=2,
            fill=color,
        )
    y += cell_step + 6

    # --- Grid ---
    level_labels = {"theme": "THEMES", "design_bet": "DESIGN BETS", "technique": "TECHNIQUES"}
    label_font = _font(10)
    group_font = _font(10, bold=True)
    level_order = ["theme", "design_bet", "technique"]

    for level in level_order:
        concepts = data.groups.get(level, [])
        if not concepts:
            continue

        # Group header
        level_label = level_labels.get(level, level.upper())
        color = LEVEL_RAMPS.get(level, DEFAULT_RAMP)[3]  # bright color from ramp
        draw.text(
            (PADDING, y + 4),
            f"▎ {level_label} ({len(concepts)})",
            font=group_font,
            fill=color,
        )
        y += GROUP_HEADER_HEIGHT

        for concept_name, importance, bucket_counts in concepts:
            # Concept label (truncated)
            display_name = concept_name[:35]
            draw.text(
                (PADDING + 4, y + 1),
                display_name,
                font=label_font,
                fill=TEXT_DIM,
            )

            # Event count badge
            imp_str = str(int(importance)) if importance == int(importance) else f"{importance:.1f}"
            imp_w = draw.textlength(imp_str, font=_font(8, mono=True))
            draw.text(
                (PADDING + LABEL_WIDTH - imp_w - 8, y + 3),
                imp_str,
                font=_font(8, mono=True),
                fill=TEXT_DIM,
            )

            # Cells
            for j, bkey in enumerate(data.buckets):
                count = bucket_counts.get(bkey, 0)
                color = _heat_color(count, data.max_count, level)
                x0 = grid_x0 + j * cell_step
                y0 = y
                draw.rounded_rectangle(
                    [x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE],
                    radius=2,
                    fill=color,
                )

            y += cell_step

    # --- Legend ---
    y += 8
    draw.line([(PADDING, y), (width - PADDING, y)], fill=DIVIDER, width=1)
    y += 8

    legend_font = _font(9)
    level_short = {"theme": "Themes", "design_bet": "Bets", "technique": "Techniques"}
    lx = PADDING
    draw.text((lx, y), "Less", font=legend_font, fill=TEXT_DIM)
    lx += 32

    # Show labeled ramp for each level
    for level in level_order:
        ramp = LEVEL_RAMPS.get(level, DEFAULT_RAMP)
        label = level_short.get(level, level)
        draw.text((lx, y), label, font=_font(8), fill=TEXT_DIM)
        lx += int(draw.textlength(label, font=_font(8))) + 4
        draw.rounded_rectangle([lx, y, lx + 10, y + 10], radius=2, fill=CELL_EMPTY)
        lx += 13
        for c in ramp:
            draw.rounded_rectangle([lx, y, lx + 10, y + 10], radius=2, fill=c)
            lx += 13
        lx += 10

    draw.text((lx, y), "More", font=legend_font, fill=TEXT_DIM)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG")
    logger.info("Heatmap saved to %s (%dx%d)", output_path, width, height)
    return output_path


def generate_concept_heatmap(
    db: TrajectoryDB,
    project_name: str,
    output_dir: Path | None = None,
    max_concepts: int = 60,
) -> Path:
    """Generate a concept heatmap for a project. Returns output path."""
    project = db.get_project_by_name(project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found")

    data = query_heatmap_data(db, project.id, max_concepts=max_concepts)

    if output_dir is None:
        output_dir = Path("data/heatmap")
    output_path = output_dir / f"{project_name}_heatmap.png"

    return render_heatmap(data, output_path)
