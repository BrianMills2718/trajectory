"""Concept evolution — cinematic animated graph of how a project's ideas emerge and connect.

Events are grouped into "beats" (time-clustered bursts). Key moments are detected
and narrated. One-hit-wonder concepts are filtered out. The result is a self-contained
HTML page with D3.js force graph, auto-play, and narrative overlays.
"""

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


def _extract_timeline(db: TrajectoryDB, project_id: int) -> dict:
    """Extract event data, group into beats, detect key moments."""
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    events = db.conn.execute("""
        SELECT e.id, e.timestamp, e.title, e.event_type
        FROM events e WHERE e.project_id = ? ORDER BY e.timestamp
    """, (project_id,)).fetchall()

    concept_events = db.conn.execute("""
        SELECT ce.event_id, c.name, c.level
        FROM concept_events ce
        JOIN concepts c ON ce.concept_id = c.id
        JOIN events e ON ce.event_id = e.id
        WHERE e.project_id = ?
    """, (project_id,)).fetchall()

    # event_id → concepts
    event_concepts: dict[int, list[str]] = defaultdict(list)
    concept_levels: dict[str, str] = {}
    for row in concept_events:
        event_concepts[row["event_id"]].append(row["name"])
        concept_levels[row["name"]] = row["level"] or "technique"

    # Count total mentions per concept across all events
    concept_total_mentions: Counter[str] = Counter()
    for concepts in event_concepts.values():
        for c in concepts:
            concept_total_mentions[c] += 1

    # Filter: only concepts with 2+ mentions (kill one-hit wonders)
    significant = {c for c, n in concept_total_mentions.items() if n >= 2}

    # Build raw frames (filtered)
    raw_frames = []
    for event in events:
        concepts = [c for c in event_concepts.get(event["id"], []) if c in significant]
        if not concepts:
            continue
        unique = sorted(set(concepts))
        edges = []
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                edges.append([unique[i], unique[j]])
        raw_frames.append({
            "timestamp": event["timestamp"],
            "title": event["title"] or "",
            "type": event["event_type"] or "",
            "concepts": unique,
            "edges": edges,
        })

    # --- Group into beats ---
    # A beat = a cluster of events within 2 hours of each other
    beats: list[dict] = []
    current_beat_frames: list[dict] = []
    last_ts: datetime | None = None

    for frame in raw_frames:
        try:
            ts = datetime.fromisoformat(frame["timestamp"])
        except (ValueError, TypeError):
            continue
        if last_ts and (ts - last_ts) > timedelta(hours=2):
            if current_beat_frames:
                beats.append(_merge_beat(current_beat_frames))
                current_beat_frames = []
        current_beat_frames.append(frame)
        last_ts = ts

    if current_beat_frames:
        beats.append(_merge_beat(current_beat_frames))

    # --- Detect key moments ---
    moments = _detect_moments(beats, concept_levels)

    # Filter concept_levels to only significant concepts
    filtered_levels = {c: l for c, l in concept_levels.items() if c in significant}

    return {
        "project": project.name,
        "beats": beats,
        "concept_levels": filtered_levels,
        "moments": moments,
        "total_events": len(events),
        "total_concepts": len(significant),
        "filtered_out": len(concept_total_mentions) - len(significant),
    }


def _merge_beat(frames: list[dict]) -> dict:
    """Merge multiple events into a single beat."""
    all_concepts: list[str] = []
    all_edges: list[list[str]] = []
    titles: list[str] = []
    types: set[str] = set()

    for f in frames:
        all_concepts.extend(f["concepts"])
        all_edges.extend(f["edges"])
        if f["title"]:
            titles.append(f["title"])
        types.add(f["type"])

    # Deduplicate edges
    edge_set = set()
    unique_edges = []
    for e in all_edges:
        key = tuple(sorted(e))
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(list(key))

    first_ts = frames[0]["timestamp"]
    try:
        dt = datetime.fromisoformat(first_ts)
        label = dt.strftime("%b %-d, %H:%M")
    except (ValueError, TypeError):
        label = first_ts[:16]

    return {
        "timestamp": first_ts,
        "label": label,
        "event_count": len(frames),
        "concepts": sorted(set(all_concepts)),
        "edges": unique_edges,
        "titles": titles[:3],  # top 3 titles for display
        "types": sorted(types),
    }


def _detect_moments(beats: list[dict], concept_levels: dict[str, str]) -> list[dict]:
    """Detect narrative key moments across the beat timeline.

    At most one moment per beat — picks the most dramatic one.
    """
    moments: list[dict] = []
    seen_concepts: set[str] = set()
    concept_counts: Counter[str] = Counter()
    beats_with_moments: set[int] = set()

    for i, beat in enumerate(beats):
        new_concepts = [c for c in beat["concepts"] if c not in seen_concepts]
        new_themes = [c for c in new_concepts if concept_levels.get(c) == "theme"]
        new_bets = [c for c in new_concepts if concept_levels.get(c) == "design_bet"]

        seen_concepts.update(beat["concepts"])
        for c in beat["concepts"]:
            concept_counts[c] += 1

        # Collect candidates for this beat, pick the best one
        candidates: list[tuple[int, dict]] = []  # (priority, moment)

        if i == 0:
            candidates.append((100, {
                "beat": i, "type": "origin", "title": "The Beginning",
                "text": f"{len(new_concepts)} concepts emerge in the first burst",
                "color": "#58a6ff",
            }))
        elif len(new_concepts) >= 12:
            candidates.append((90, {
                "beat": i, "type": "burst", "title": "Concept Explosion",
                "text": f"{len(new_concepts)} new ideas land at once — {beat['event_count']} events",
                "color": "#f78166",
            }))

        if new_themes and i > 0:
            names = ", ".join(t.replace("_", " ") for t in new_themes[:2])
            candidates.append((70, {
                "beat": i, "type": "theme", "title": "New Direction",
                "text": f"Theme emerges: {names}",
                "color": "#58a6ff",
            }))

        if len(new_bets) >= 4:
            names = ", ".join(b.replace("_", " ") for b in new_bets[:3])
            candidates.append((60, {
                "beat": i, "type": "bets", "title": "Architecture Sprint",
                "text": f"{len(new_bets)} design bets placed: {names}",
                "color": "#d2a8ff",
            }))

        if len(beat["edges"]) >= 40:
            candidates.append((50, {
                "beat": i, "type": "dense", "title": "Ideas Converging",
                "text": f"{len(beat['edges'])} connections form — the graph thickens",
                "color": "#7ee787",
            }))

        if candidates:
            # Pick highest priority
            candidates.sort(key=lambda x: -x[0])
            moments.append(candidates[0][1])
            beats_with_moments.add(i)

    # Final moment (always)
    if beats:
        top = concept_counts.most_common(3)
        top_str = ", ".join(c.replace("_", " ") for c, _ in top)
        moments.append({
            "beat": len(beats) - 1, "type": "finale", "title": "The Story So Far",
            "text": f"{len(seen_concepts)} concepts, anchored by {top_str}",
            "color": "#e6edf3",
        })

    return moments


def generate_concept_evolution(
    db: TrajectoryDB,
    project_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Generate an animated concept evolution page. Returns output path."""
    project = db.get_project_by_name(project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found")

    timeline = _extract_timeline(db, project.id)

    if output_dir is None:
        output_dir = Path("data/evolution")
    output_path = output_dir / f"{project_name}_evolution.html"

    html = _render_html(timeline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("Evolution page saved to %s (%d beats, %d moments)",
                output_path, len(timeline["beats"]), len(timeline["moments"]))
    return output_path


def _render_html(timeline: dict) -> str:
    data_json = json.dumps(timeline, ensure_ascii=False)
    project_name = _esc(timeline["project"])
    total_events = timeline["total_events"]
    total_concepts = timeline["total_concepts"]

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{project_name} — Concept Evolution</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    overflow: hidden;
    height: 100vh;
    display: flex;
    flex-direction: column;
}}

/* --- Top bar --- */
.topbar {{
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    border-bottom: 1px solid #21262d;
    flex-shrink: 0;
    background: #161b22;
    z-index: 10;
}}
.topbar h1 {{
    font-size: 16px;
    font-weight: 700;
    white-space: nowrap;
}}
.topbar .beat-info {{
    font-size: 12px;
    color: #8b949e;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.topbar .counter {{
    font-size: 12px;
    color: #58a6ff;
    font-family: monospace;
    white-space: nowrap;
}}

/* --- Canvas --- */
#graph {{
    flex: 1;
    position: relative;
    overflow: hidden;
}}
#graph svg {{
    width: 100%;
    height: 100%;
}}

/* --- Narrative overlay --- */
#narrative {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    pointer-events: none;
    z-index: 20;
    opacity: 0;
    transition: opacity 0.6s;
}}
#narrative.visible {{
    opacity: 1;
}}
#narrative .n-title {{
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.8);
}}
#narrative .n-text {{
    font-size: 16px;
    color: #8b949e;
    text-shadow: 0 2px 10px rgba(0,0,0,0.8);
}}

/* --- Beat indicator --- */
#beatIndicator {{
    position: absolute;
    bottom: 70px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 13px;
    color: #484f58;
    font-family: monospace;
    pointer-events: none;
    z-index: 10;
    text-align: center;
}}

/* --- Legend --- */
.legend {{
    position: absolute;
    top: 12px;
    right: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 11px;
    color: #8b949e;
    pointer-events: none;
    z-index: 10;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
}}

/* --- Controls --- */
.controls {{
    padding: 10px 20px;
    border-top: 1px solid #21262d;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
    background: #161b22;
    z-index: 10;
}}
.controls button {{
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 5px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-family: inherit;
    transition: background 0.15s;
}}
.controls button:hover {{ background: #30363d; }}
.controls button.active {{ background: #1f6feb; border-color: #58a6ff; }}
#timeline {{
    flex: 1;
    height: 6px;
    -webkit-appearance: none;
    appearance: none;
    background: #21262d;
    border-radius: 3px;
    outline: none;
    cursor: pointer;
}}
#timeline::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #58a6ff;
    cursor: pointer;
}}
#timeline::-moz-range-thumb {{
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #58a6ff;
    cursor: pointer;
    border: none;
}}
.speed-label {{
    font-size: 11px;
    color: #8b949e;
    min-width: 30px;
    text-align: center;
}}

/* --- Tooltip --- */
#tooltip {{
    display: none;
    position: fixed;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    max-width: 280px;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    pointer-events: none;
}}
#tooltip .tt-name {{
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 4px;
}}
#tooltip .tt-level {{
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 3px;
    display: inline-block;
    margin-bottom: 4px;
}}
#tooltip .tt-stats {{
    color: #8b949e;
}}

/* Glow filter for active nodes */
svg .glow {{
    filter: url(#glow);
}}
/* Edge pulse animation */
@keyframes edgePulse {{
    0% {{ stroke-opacity: 0.9; stroke-width: 4; }}
    100% {{ stroke-opacity: 0; stroke-width: 8; }}
}}
.edge-pulse {{
    animation: edgePulse 0.8s ease-out forwards;
    pointer-events: none;
}}
/* Node entrance pop */
@keyframes nodePop {{
    0% {{ r: 0; opacity: 0; }}
    50% {{ opacity: 1; }}
    100% {{ opacity: 1; }}
}}
/* Finale overlay */
#finale {{
    position: absolute;
    inset: 0;
    background: rgba(13,17,23,0.85);
    display: none;
    z-index: 25;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    text-align: center;
    gap: 16px;
}}
#finale.visible {{ display: flex; }}
#finale .f-title {{
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff, #d2a8ff, #7ee787);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
#finale .f-stats {{
    display: flex;
    gap: 32px;
    justify-content: center;
}}
#finale .f-stat {{
    text-align: center;
}}
#finale .f-num {{
    font-size: 36px;
    font-weight: 800;
}}
#finale .f-label {{
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
#finale .f-anchors {{
    color: #8b949e;
    font-size: 14px;
    margin-top: 8px;
}}
</style>
</head>
<body>

<div class="topbar">
    <h1>{project_name}</h1>
    <div class="beat-info" id="beatInfo">Press play to watch this project think</div>
    <div class="counter" id="counter">{total_concepts} concepts · {total_events} events</div>
</div>

<div id="graph">
    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#58a6ff"></div>theme</div>
        <div class="legend-item"><div class="legend-dot" style="background:#d2a8ff"></div>design bet</div>
        <div class="legend-item"><div class="legend-dot" style="background:#7ee787"></div>technique</div>
    </div>
    <div id="narrative">
        <div class="n-title"></div>
        <div class="n-text"></div>
    </div>
    <div id="beatIndicator"></div>
    <div id="finale">
        <div class="f-title">{project_name}</div>
        <div class="f-stats">
            <div class="f-stat"><div class="f-num" id="fConcepts">0</div><div class="f-label">concepts</div></div>
            <div class="f-stat"><div class="f-num" id="fConnections">0</div><div class="f-label">connections</div></div>
            <div class="f-stat"><div class="f-num" id="fBeats">0</div><div class="f-label">beats</div></div>
        </div>
        <div class="f-anchors" id="fAnchors"></div>
    </div>
</div>

<div class="controls">
    <button id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button onclick="stepBack()">◀</button>
    <button onclick="stepForward()">▶</button>
    <input type="range" id="timeline" min="0" max="1" value="0" step="1">
    <button onclick="cycleSpeed()" id="speedBtn" class="speed-label">1×</button>
    <button onclick="resetGraph()">↺</button>
</div>

<div id="tooltip">
    <div class="tt-name"></div>
    <div class="tt-level"></div>
    <div class="tt-stats"></div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = {data_json};
const beats = DATA.beats;
const conceptLevels = DATA.concept_levels;
const moments = DATA.moments;

const COLORS = {{
    theme: "#58a6ff",
    design_bet: "#d2a8ff",
    technique: "#7ee787",
}};

// State
let currentBeat = -1;
let playing = false;
let playTimer = null;
let speed = 1;
const speeds = [0.5, 1, 2, 4];
let speedIdx = 1;
let narrativeTimer = null;

// Graph state
const nodes = new Map();
const edges = new Map();

// D3
const graphEl = document.getElementById("graph");
const W = graphEl.clientWidth;
const H = graphEl.clientHeight;

const svg = d3.select("#graph").append("svg")
    .attr("viewBox", [0, 0, W, H]);

// Glow filter
const defs = svg.append("defs");
const filter = defs.append("filter").attr("id", "glow");
filter.append("feGaussianBlur").attr("stdDeviation", 4).attr("result", "blur");
const merge = filter.append("feMerge");
merge.append("feMergeNode").attr("in", "blur");
merge.append("feMergeNode").attr("in", "SourceGraphic");

const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.2, 6]).on("zoom", e => g.attr("transform", e.transform)));

const linkGroup = g.append("g");
const nodeGroup = g.append("g");
const labelGroup = g.append("g");

const simulation = d3.forceSimulation()
    .force("charge", d3.forceManyBody().strength(-160).distanceMax(300))
    .force("center", d3.forceCenter(W / 2, H / 2).strength(0.05))
    .force("collision", d3.forceCollide().radius(d => R(d) + 6))
    .force("link", d3.forceLink().id(d => d.name).distance(70).strength(0.4))
    .force("x", d3.forceX(W / 2).strength(0.02))
    .force("y", d3.forceY(H / 2).strength(0.02))
    .on("tick", ticked)
    .alphaDecay(0.03);
simulation.stop();

const slider = document.getElementById("timeline");
slider.max = Math.max(0, beats.length - 1);
slider.addEventListener("input", () => {{
    const target = parseInt(slider.value);
    jumpTo(target);
}});

function R(d) {{
    const base = Math.min(5 + Math.pow(d.count, 0.6) * 4, 32);
    // Dormant nodes shrink
    if (currentBeat >= 0) {{
        const age = currentBeat - d.lastSeen;
        if (age > 6) return Math.max(base * 0.3, 3);
        if (age > 3) return base * 0.6;
    }}
    return base;
}}

function opacity(d) {{
    if (currentBeat < 0) return 0.3;
    const age = currentBeat - d.lastSeen;
    if (age <= 0) return 1.0;
    if (age <= 1) return 0.85;
    if (age <= 3) return 0.45;
    if (age <= 6) return 0.2;
    return 0.08; // nearly invisible
}}

// Camera: smoothly pan/zoom to center on active nodes
let userHasPanned = false;
svg.on("mousedown.userpan", () => {{ userHasPanned = true; }});

function autoCam(activeNames) {{
    if (userHasPanned || activeNames.size === 0) return;
    const activeNodes = Array.from(nodes.values()).filter(n => activeNames.has(n.name));
    if (activeNodes.length === 0) return;

    let cx = 0, cy = 0;
    for (const n of activeNodes) {{ cx += n.x; cy += n.y; }}
    cx /= activeNodes.length;
    cy /= activeNodes.length;

    // Compute bounding box of active nodes
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of activeNodes) {{
        minX = Math.min(minX, n.x); maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y); maxY = Math.max(maxY, n.y);
    }}
    const spread = Math.max(maxX - minX, maxY - minY, 100);
    const scale = Math.min(W / (spread + 200), H / (spread + 200), 2.5);

    const tx = W / 2 - cx * scale;
    const ty = H / 2 - cy * scale;
    const transform = d3.zoomIdentity.translate(tx, ty).scale(scale);

    svg.transition().duration(800).ease(d3.easeCubicOut)
        .call(d3.zoom().scaleExtent([0.2, 6]).on("zoom", e => g.attr("transform", e.transform)).transform, transform);
}}

function applyBeat(idx) {{
    const beat = beats[idx];
    const justActivated = new Set();
    const newEdgeKeys = [];
    for (const name of beat.concepts) {{
        if (nodes.has(name)) {{
            const n = nodes.get(name);
            n.count++;
            n.lastSeen = idx;
        }} else {{
            nodes.set(name, {{
                name,
                level: conceptLevels[name] || "technique",
                count: 1,
                lastSeen: idx,
                x: W/2 + (Math.random() - 0.5) * 300,
                y: H/2 + (Math.random() - 0.5) * 300,
            }});
        }}
        justActivated.add(name);
    }}
    for (const [a, b] of beat.edges) {{
        const key = a < b ? a + "|" + b : b + "|" + a;
        if (edges.has(key)) {{
            edges.get(key).weight++;
        }} else {{
            edges.set(key, {{ source: a, target: b, weight: 1 }});
            newEdgeKeys.push(key);
        }}
    }}
    return {{ justActivated, newEdgeKeys }};
}}

function jumpTo(target) {{
    nodes.clear();
    edges.clear();
    currentBeat = -1;
    for (let i = 0; i <= target; i++) applyBeat(i);
    currentBeat = target;
    updateUI(new Set());
    render();
    // Reset camera on jump
    userHasPanned = false;
    const allActive = new Set(beats[target] ? beats[target].concepts : []);
    autoCam(allActive);
}}

function updateUI(justActivated) {{
    const beat = currentBeat >= 0 ? beats[currentBeat] : null;
    const info = document.getElementById("beatInfo");
    const bi = document.getElementById("beatIndicator");

    if (beat) {{
        const titles = beat.titles.map(t => t.substring(0, 60)).join(" · ");
        info.textContent = `${{beat.label}} — ${{beat.event_count}} event${{beat.event_count > 1 ? "s" : ""}} — ${{titles}}`;
        bi.textContent = `${{nodes.size}} concepts · ${{edges.size}} connections`;
    }} else {{
        info.textContent = "Press play to watch this project think";
        bi.textContent = "";
    }}
    slider.value = Math.max(0, currentBeat);

    // Check for narrative moments
    const moment = moments.find(m => m.beat === currentBeat);
    showNarrative(moment);
}}

function showNarrative(moment) {{
    const el = document.getElementById("narrative");
    if (narrativeTimer) clearTimeout(narrativeTimer);

    if (moment) {{
        el.querySelector(".n-title").textContent = moment.title;
        el.querySelector(".n-title").style.color = moment.color;
        el.querySelector(".n-text").textContent = moment.text;
        el.classList.add("visible");
        narrativeTimer = setTimeout(() => el.classList.remove("visible"), 3000);
    }} else {{
        el.classList.remove("visible");
    }}
}}

function render() {{
    const nodeArr = Array.from(nodes.values());
    const edgeArr = Array.from(edges.values()).map(e => ({{
        source: e.source, target: e.target, weight: e.weight,
    }}));

    // Links
    const links = linkGroup.selectAll("line").data(edgeArr, d => {{
        const s = typeof d.source === "string" ? d.source : d.source.name;
        const t = typeof d.target === "string" ? d.target : d.target.name;
        return s + "|" + t;
    }});
    links.exit().transition().duration(200).attr("stroke-opacity", 0).remove();
    const linksEnter = links.enter().append("line")
        .attr("stroke", "#30363d")
        .attr("stroke-opacity", 0);
    linksEnter.merge(links)
        .transition().duration(400)
        .attr("stroke-opacity", d => Math.min(0.15 + d.weight * 0.12, 0.7))
        .attr("stroke-width", d => Math.min(0.5 + d.weight * 0.6, 5))
        .attr("stroke", d => {{
            // Color edges by the dominant node's level
            const sName = typeof d.source === "string" ? d.source : d.source.name;
            const level = conceptLevels[sName] || "technique";
            const c = COLORS[level] || COLORS.technique;
            return c + "40"; // with alpha
        }});

    // Nodes
    const circles = nodeGroup.selectAll("circle").data(nodeArr, d => d.name);
    circles.exit().transition().duration(300).attr("r", 0).remove();
    const circlesEnter = circles.enter().append("circle")
        .attr("r", 0)
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("cursor", "pointer")
        .call(d3.drag().on("start", dragStart).on("drag", dragging).on("end", dragEnd))
        .on("mouseenter", showTooltip)
        .on("mouseleave", hideTooltip);
    circlesEnter.merge(circles)
        .attr("fill", d => COLORS[d.level] || COLORS.technique)
        .classed("glow", d => d.lastSeen === currentBeat)
        .transition().duration(400)
        .attr("r", d => R(d))
        .attr("opacity", d => opacity(d));

    // Labels — show for concepts with 3+ mentions or just activated
    const labelData = nodeArr.filter(d => d.count >= 3 || d.lastSeen >= currentBeat - 1);
    const labels = labelGroup.selectAll("text").data(labelData, d => d.name);
    labels.exit().transition().duration(200).attr("opacity", 0).remove();
    labels.enter().append("text")
        .attr("text-anchor", "middle")
        .attr("pointer-events", "none")
        .attr("opacity", 0)
        .merge(labels)
        .text(d => d.name.replace(/_/g, " "))
        .attr("font-size", d => d.count >= 6 ? 11 : d.count >= 3 ? 10 : 9)
        .attr("font-weight", d => d.lastSeen === currentBeat ? 700 : 400)
        .attr("fill", d => d.lastSeen === currentBeat ? "#e6edf3" : "#6e7681")
        .transition().duration(400)
        .attr("opacity", d => opacity(d) * 0.9);

    simulation.nodes(nodeArr);
    simulation.force("link").links(edgeArr);
    simulation.alpha(0.4).restart();
}}

function ticked() {{
    linkGroup.selectAll("line")
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeGroup.selectAll("circle")
        .attr("cx", d => d.x).attr("cy", d => d.y);
    labelGroup.selectAll("text")
        .attr("x", d => d.x).attr("y", d => d.y - R(d) - 4);
}}

function dragStart(e, d) {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
function dragging(e, d) {{ d.fx = e.x; d.fy = e.y; }}
function dragEnd(e, d) {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}

const tooltip = document.getElementById("tooltip");
function showTooltip(event, d) {{
    tooltip.querySelector(".tt-name").textContent = d.name.replace(/_/g, " ");
    const ttLevel = tooltip.querySelector(".tt-level");
    ttLevel.textContent = d.level.replace("_", " ");
    ttLevel.style.background = (COLORS[d.level] || COLORS.technique) + "20";
    ttLevel.style.color = COLORS[d.level] || COLORS.technique;
    const conns = Array.from(edges.values()).filter(e => {{
        const s = typeof e.source === "string" ? e.source : e.source.name;
        const t = typeof e.target === "string" ? e.target : e.target.name;
        return s === d.name || t === d.name;
    }}).length;
    tooltip.querySelector(".tt-stats").textContent = `${{d.count}} mentions · ${{conns}} connections · last beat ${{d.lastSeen + 1}}`;
    tooltip.style.display = "block";
    tooltip.style.left = (event.pageX + 12) + "px";
    tooltip.style.top = (event.pageY - 10) + "px";
}}
function hideTooltip() {{ tooltip.style.display = "none"; }}

// --- Playback ---
function togglePlay() {{
    playing = !playing;
    const btn = document.getElementById("playBtn");
    btn.textContent = playing ? "⏸" : "▶ Play";
    btn.classList.toggle("active", playing);
    if (playing) {{
        if (currentBeat >= beats.length - 1) resetGraph();
        scheduleNext();
    }} else {{
        clearTimeout(playTimer);
    }}
}}

function scheduleNext() {{
    if (!playing) return;
    if (currentBeat >= beats.length - 1) {{ togglePlay(); return; }}

    // Check if next beat has a narrative moment — pause longer
    const nextMoment = moments.find(m => m.beat === currentBeat + 1);
    const delay = nextMoment ? 2500 / speed : 1000 / speed;

    playTimer = setTimeout(() => {{
        stepForward();
        scheduleNext();
    }}, delay);
}}

function stepForward() {{
    if (currentBeat >= beats.length - 1) return;
    currentBeat++;
    const {{ justActivated, newEdgeKeys }} = applyBeat(currentBeat);
    updateUI(justActivated);
    render();

    // Camera follows the action
    autoCam(justActivated);

    // Pulse new edges
    if (newEdgeKeys.length > 0 && newEdgeKeys.length < 30) {{
        pulseEdges(newEdgeKeys);
    }}

    // Finale
    const moment = moments.find(m => m.beat === currentBeat);
    if (moment && moment.type === "finale") {{
        setTimeout(showFinale, 1500);
    }}
}}

function pulseEdges(edgeKeys) {{
    // Add temporary bright lines that animate and disappear
    const pulseGroup = g.append("g").attr("class", "pulse-group");
    for (const key of edgeKeys) {{
        const edge = edges.get(key);
        if (!edge) continue;
        const s = typeof edge.source === "string" ? nodes.get(edge.source) : edge.source;
        const t = typeof edge.target === "string" ? nodes.get(edge.target) : edge.target;
        if (!s || !t) continue;
        const level = conceptLevels[s.name || s] || "technique";
        pulseGroup.append("line")
            .attr("x1", s.x).attr("y1", s.y)
            .attr("x2", t.x).attr("y2", t.y)
            .attr("stroke", COLORS[level] || COLORS.technique)
            .attr("class", "edge-pulse");
    }}
    setTimeout(() => pulseGroup.remove(), 900);
}}

function showFinale() {{
    document.getElementById("fConcepts").textContent = nodes.size;
    document.getElementById("fConnections").textContent = edges.size;
    document.getElementById("fBeats").textContent = beats.length;
    // Top 3 concepts
    const sorted = Array.from(nodes.values()).sort((a, b) => b.count - a.count);
    const top3 = sorted.slice(0, 3).map(n => n.name.replace(/_/g, " ")).join(" · ");
    document.getElementById("fAnchors").textContent = "anchored by " + top3;
    document.getElementById("finale").classList.add("visible");
    // Auto-hide after 5s
    setTimeout(() => document.getElementById("finale").classList.remove("visible"), 6000);
}}

function stepBack() {{
    if (currentBeat <= 0) return;
    jumpTo(currentBeat - 1);
}}

function resetGraph() {{
    nodes.clear();
    edges.clear();
    currentBeat = -1;
    linkGroup.selectAll("*").remove();
    nodeGroup.selectAll("*").remove();
    labelGroup.selectAll("*").remove();
    g.selectAll(".pulse-group").remove();
    document.getElementById("finale").classList.remove("visible");
    userHasPanned = false;
    updateUI(new Set());
}}

function cycleSpeed() {{
    speedIdx = (speedIdx + 1) % speeds.length;
    speed = speeds[speedIdx];
    document.getElementById("speedBtn").textContent = speed + "×";
}}

updateUI(new Set());
</script>
</body>
</html>'''
