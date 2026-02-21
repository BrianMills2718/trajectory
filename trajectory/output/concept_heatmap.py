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
class CellEvent:
    """Single event for a tooltip."""
    title: str
    event_type: str
    significance: float


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
    # Per-cell event details for tooltips: (concept_name, bucket_key) → list of events
    cell_details: dict[tuple[str, str], list[CellEvent]] = field(default_factory=dict)
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

    # Get per-concept, per-event data (including event details for tooltips)
    rows = db.conn.execute("""
        SELECT c.name, c.level, c.importance,
               e.timestamp, e.title, e.event_type, e.significance
        FROM concept_events ce
        JOIN concepts c ON ce.concept_id = c.id
        JOIN events e ON ce.event_id = e.id
        WHERE e.project_id = ?
        ORDER BY c.name, e.timestamp
    """, (project_id,)).fetchall()

    # Aggregate into buckets and collect cell details
    concept_data: dict[str, dict[str, object]] = {}  # name → {level, total_count, buckets: {key: count}}
    cell_details: dict[tuple[str, str], list[CellEvent]] = defaultdict(list)
    for row in rows:
        name = row["name"]
        if name not in concept_data:
            concept_data[name] = {
                "level": row["level"] or "technique",
                "total_count": 0,
                "buckets": defaultdict(int),
            }
        bkey = _bucket_key(row["timestamp"], granularity)
        concept_data[name]["buckets"][bkey] += 1  # type: ignore[index]
        concept_data[name]["total_count"] = int(concept_data[name]["total_count"]) + 1  # type: ignore[arg-type]
        cell_details[(name, bkey)].append(CellEvent(
            title=row["title"] or "",
            event_type=row["event_type"] or "",
            significance=row["significance"] or 0,
        ))

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

    # Filter cell_details to only include concepts that made the cut
    selected_names = {name for name, _ in sorted_concepts}
    filtered_details = {
        k: v for k, v in cell_details.items() if k[0] in selected_names
    }

    return ConceptHeatmapData(
        project_name=project.name,
        granularity=granularity,
        buckets=buckets,
        bucket_labels=bucket_labels,
        groups=groups,
        max_count=max_count,
        total_events=total_events,
        bucket_totals=dict(bucket_totals),
        cell_details=filtered_details,
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


def _html_color(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_heatmap_html(data: ConceptHeatmapData, output_path: Path) -> Path:
    """Render the concept heatmap as an interactive HTML page."""
    cell_size = 20
    cell_gap = 3
    cell_step = cell_size + cell_gap

    level_order = ["theme", "design_bet", "technique"]
    level_labels = {"theme": "THEMES", "design_bet": "DESIGN BETS", "technique": "TECHNIQUES"}
    level_css_colors = {
        "theme": _html_color(*LEVEL_RAMPS["theme"][3]),
        "design_bet": _html_color(*LEVEL_RAMPS["design_bet"][3]),
        "technique": _html_color(*LEVEL_RAMPS["technique"][3]),
    }

    # Build JSON data for tooltips
    import json
    tooltip_data: dict[str, object] = {}
    for (concept_name, bkey), events in data.cell_details.items():
        key = f"{concept_name}|{bkey}"
        tooltip_data[key] = [
            {"title": _escape_html(e.title[:80]), "type": e.event_type, "sig": round(e.significance, 2)}
            for e in events
        ]

    # Build CSS color ramps as functions
    ramp_css: dict[str, list[str]] = {}
    for level in level_order:
        ramp = LEVEL_RAMPS.get(level, DEFAULT_RAMP)
        ramp_css[level] = [_html_color(*c) for c in ramp]

    gran_label = {"day": "daily", "week": "weekly", "month": "monthly"}[data.granularity]

    # Build rows HTML
    rows_html = []

    # Summary row
    max_total = max(data.bucket_totals.values()) if data.bucket_totals else 1
    summary_cells = []
    for bkey in data.buckets:
        count = data.bucket_totals.get(bkey, 0)
        if count == 0:
            bg = _html_color(*CELL_EMPTY)
        else:
            t = min(count / max(max_total, 1), 1.0)
            gray = int(40 + t * 160)
            bg = _html_color(gray, gray, gray)
        summary_cells.append(
            f'<div class="cell" style="background:{bg}" '
            f'title="{data.bucket_labels[data.buckets.index(bkey)]}: {count} events"></div>'
        )
    rows_html.append(f'''
        <div class="row summary-row">
            <div class="label bold">ALL EVENTS</div>
            <div class="badge"></div>
            <div class="cells">{"".join(summary_cells)}</div>
        </div>
    ''')

    for level in level_order:
        concepts = data.groups.get(level, [])
        if not concepts:
            continue

        color = level_css_colors.get(level, "#6e7681")
        rows_html.append(
            f'<div class="group-header" style="color:{color}">'
            f'▎ {level_labels.get(level, level.upper())} ({len(concepts)})</div>'
        )

        for concept_name, event_count, bucket_counts in concepts:
            cells = []
            for j, bkey in enumerate(data.buckets):
                count = bucket_counts.get(bkey, 0)
                color_tuple = _heat_color(count, data.max_count, level)
                bg = _html_color(*color_tuple)
                tooltip_key = f"{concept_name}|{bkey}"
                has_detail = tooltip_key in tooltip_data
                cells.append(
                    f'<div class="cell{" has-events" if has_detail else ""}" '
                    f'style="background:{bg}" '
                    f'data-concept="{_escape_html(concept_name)}" '
                    f'data-bucket="{bkey}" '
                    f'data-label="{data.bucket_labels[j]}" '
                    f'data-count="{count}"></div>'
                )
            badge = str(int(event_count)) if event_count == int(event_count) else f"{event_count:.1f}"
            rows_html.append(f'''
                <div class="row" data-level="{level}">
                    <div class="label" title="{_escape_html(concept_name)}">{_escape_html(concept_name)}</div>
                    <div class="badge">{badge}</div>
                    <div class="cells">{"".join(cells)}</div>
                </div>
            ''')

    # Column headers
    col_headers = []
    for label in data.bucket_labels:
        col_headers.append(f'<div class="col-header"><span>{_escape_html(label)}</span></div>')

    # Legend
    legend_parts = []
    legend_parts.append('<span class="legend-label">Less</span>')
    for level in level_order:
        short = {"theme": "Themes", "design_bet": "Bets", "technique": "Techniques"}[level]
        legend_parts.append(f'<span class="legend-level-label">{short}</span>')
        legend_parts.append(f'<div class="legend-cell" style="background:{_html_color(*CELL_EMPTY)}"></div>')
        for c in ramp_css[level]:
            legend_parts.append(f'<div class="legend-cell" style="background:{c}"></div>')
    legend_parts.append('<span class="legend-label">More</span>')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape_html(data.project_name)} — Concept Heatmap</title>
<style>
:root {{
    --bg: {_html_color(*BG)};
    --text: {_html_color(*TEXT)};
    --text-dim: {_html_color(*TEXT_DIM)};
    --divider: {_html_color(*DIVIDER)};
    --cell-empty: {_html_color(*CELL_EMPTY)};
    --cell-size: {cell_size}px;
    --cell-gap: {cell_gap}px;
    --cell-step: {cell_step}px;
    --label-width: 260px;
    --badge-width: 36px;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    padding: 24px;
    min-width: 600px;
}}
h1 {{
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 4px;
}}
.subtitle {{
    color: var(--text-dim);
    font-size: 13px;
    margin-bottom: 16px;
}}
.divider {{
    border-top: 1px solid var(--divider);
    margin: 12px 0;
}}
.col-headers {{
    display: flex;
    margin-left: calc(var(--label-width) + var(--badge-width));
    margin-bottom: 4px;
}}
.col-header {{
    width: var(--cell-step);
    flex-shrink: 0;
    text-align: center;
    position: relative;
    height: 50px;
}}
.col-header span {{
    display: block;
    font-size: 10px;
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
    color: var(--text-dim);
    transform: rotate(-55deg);
    transform-origin: bottom left;
    position: absolute;
    bottom: 0;
    left: 50%;
    white-space: nowrap;
}}
.group-header {{
    font-size: 12px;
    font-weight: 700;
    padding: 8px 0 4px 0;
}}
.row {{
    display: flex;
    align-items: center;
    height: var(--cell-step);
}}
.row.summary-row {{
    margin-bottom: 6px;
}}
.label {{
    width: var(--label-width);
    flex-shrink: 0;
    font-size: 12px;
    color: var(--text-dim);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding-left: 4px;
}}
.label.bold {{
    font-weight: 700;
}}
.badge {{
    width: var(--badge-width);
    flex-shrink: 0;
    font-size: 10px;
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
    color: var(--text-dim);
    text-align: right;
    padding-right: 6px;
}}
.cells {{
    display: flex;
    gap: var(--cell-gap);
}}
.cell {{
    width: var(--cell-size);
    height: var(--cell-size);
    border-radius: 3px;
    flex-shrink: 0;
    cursor: default;
    transition: outline 0.1s;
}}
.cell.has-events {{
    cursor: pointer;
}}
.cell:hover {{
    outline: 2px solid var(--text-dim);
    outline-offset: 1px;
}}
.legend {{
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--divider);
}}
.legend-label {{
    font-size: 11px;
    color: var(--text-dim);
    margin: 0 4px;
}}
.legend-level-label {{
    font-size: 10px;
    color: var(--text-dim);
    margin-left: 8px;
}}
.legend-cell {{
    width: 12px;
    height: 12px;
    border-radius: 2px;
}}
/* Tooltip */
#tooltip {{
    display: none;
    position: fixed;
    background: #1c2128;
    border: 1px solid var(--divider);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    max-width: 360px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    pointer-events: none;
}}
#tooltip .tt-header {{
    font-weight: 700;
    margin-bottom: 6px;
    color: var(--text);
}}
#tooltip .tt-event {{
    color: var(--text-dim);
    margin-bottom: 3px;
    line-height: 1.4;
}}
#tooltip .tt-event .tt-type {{
    display: inline-block;
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 3px;
    background: #2d333b;
    margin-right: 4px;
}}
</style>
</head>
<body>

<h1>{_escape_html(data.project_name)}</h1>
<p class="subtitle">{data.total_concepts} concepts · {data.total_events} events · {data.span_days} days · {gran_label}</p>
<div class="divider"></div>

<div class="col-headers">
    {"".join(col_headers)}
</div>

{"".join(rows_html)}

<div class="legend">
    {"".join(legend_parts)}
</div>

<div id="tooltip">
    <div class="tt-header"></div>
    <div class="tt-body"></div>
</div>

<script>
const tooltipData = {json.dumps(tooltip_data)};
const tooltip = document.getElementById('tooltip');
const ttHeader = tooltip.querySelector('.tt-header');
const ttBody = tooltip.querySelector('.tt-body');

document.querySelectorAll('.cell[data-concept]').forEach(cell => {{
    cell.addEventListener('mouseenter', (e) => {{
        const concept = cell.dataset.concept;
        const bucket = cell.dataset.bucket;
        const label = cell.dataset.label;
        const count = cell.dataset.count;
        const key = concept + '|' + bucket;
        const events = tooltipData[key];

        ttHeader.textContent = concept + ' — ' + label + ' (' + count + ' events)';

        if (events && events.length > 0) {{
            ttBody.innerHTML = events.map(ev =>
                '<div class="tt-event"><span class="tt-type">' + ev.type + '</span>' + ev.title + '</div>'
            ).join('');
        }} else if (parseInt(count) > 0) {{
            ttBody.innerHTML = '<div class="tt-event">Events linked to this concept</div>';
        }} else {{
            ttBody.innerHTML = '';
        }}

        tooltip.style.display = 'block';
        const rect = cell.getBoundingClientRect();
        let left = rect.right + 8;
        let top = rect.top - 10;

        // Keep tooltip on screen
        if (left + 360 > window.innerWidth) {{
            left = rect.left - 368;
        }}
        if (top + tooltip.offsetHeight > window.innerHeight) {{
            top = window.innerHeight - tooltip.offsetHeight - 8;
        }}
        tooltip.style.left = left + 'px';
        tooltip.style.top = Math.max(8, top) + 'px';
    }});

    cell.addEventListener('mouseleave', () => {{
        tooltip.style.display = 'none';
    }});
}});
</script>

</body>
</html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("HTML heatmap saved to %s", output_path)
    return output_path


def generate_concept_heatmap(
    db: TrajectoryDB,
    project_name: str,
    output_dir: Path | None = None,
    max_concepts: int = 60,
    html: bool = False,
) -> Path:
    """Generate a concept heatmap for a project. Returns output path."""
    project = db.get_project_by_name(project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found")

    data = query_heatmap_data(db, project.id, max_concepts=max_concepts)

    if output_dir is None:
        output_dir = Path("data/heatmap")

    if html:
        output_path = output_dir / f"{project_name}_heatmap.html"
        return render_heatmap_html(data, output_path)
    else:
        output_path = output_dir / f"{project_name}_heatmap.png"
        return render_heatmap(data, output_path)
