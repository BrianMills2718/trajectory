# Trajectory Visualization Experiments

## Goal
A shareable visualization of everything a developer is working on — concepts, decisions, evolution across projects. "Spotify Wrapped for developers" but for *ideas*, not just commit counts.

**Differentiator**: Every existing tool (GitHub Unwrapped, Gource, Skyline, etc.) visualizes commits/files/contributions. Trajectory visualizes *concepts and ideas* — what you're actually thinking about, what architectural bets you're making, how themes evolve and connect across projects. Nobody else does this.

## What We've Tried

### 1. AI Art Mural (theme × month grid) — EXPLORED, DEAD END
- **Approach**: Generate tiles per theme/month with image AI, alpha-blend into mural
- **Models tried**: gpt-image-1 ($0.04/tile), Gemini gemini-2.5-flash-image (free)
- **Styles tried**: Abstract/fantasy → concrete rooms → literal software depictions
- **Seam blending**: Alpha fade, Gaussian blur on seams, Stability AI inpainting
- **Result**: Individual tiles can look decent. Stitched mural never looks coherent. Alpha blending = visible seams. Outpainting doesn't work (Gemini doesn't preserve pixels, Stability generates nonsense on diagrams).
- **One-shot panorama** worked better (one prompt → one image) but capped at 1024x1024.
- **Verdict**: Tile-stitching approach has a fundamental ceiling. Not shareable quality.

### 2. Dataflow Diagram Mural — EXPLORED, PARTIAL
- **Approach**: LLM generates dependency/dataflow graph → positions nodes on grid → generate tile per node → assemble
- **Result**: Layout logic works well (LLM picks good components, sensible positions). Image quality same problem as #1.
- **Useful pieces**: The dataflow layout generation is solid — reusable for other formats.

### 3. Developer Wrapped Card (static infographic) — CURRENT, v2
- **Approach**: Pillow-rendered card with treemap, heatmap, stats
- **Result**: Functional, data-rich, readable. Not visually exciting enough to share.
- **Useful pieces**: Data query layer, treemap layout, heatmap grid — all reusable.

## Ideas Backlog

### Tier 1: High priority — likely viral, feasible

#### A. Animated Video (Remotion-style)
- **Inspiration**: GitHub Unwrapped rendered 1TB of video in 72h. Most viral dev viz format.
- **Concept**: 30-60s video walking through your project evolution. Concepts appear, grow, connect. Stats fly in. Music optional.
- **Tech**: Remotion (React → MP4), or programmatic ffmpeg frame generation with Pillow/Cairo
- **Data advantage**: We have temporal data — concepts emerging over months, decisions, cross-project links. This is a STORY, not a snapshot. Video is the natural format.
- **Dependencies**: Need good frame rendering first (Spike B or C)
- **Effort**: Large. Remotion is a whole React project. ffmpeg approach is simpler but less polished.
- **Priority**: HIGH — best viral potential, but most work

#### B. Interactive Force Graph (D3.js)
- **Inspiration**: GitDiagram, but for ideas not files. repo-visualizer circle packing.
- **Concept**: Browser-based force-directed graph. Concepts = nodes (sized by activity), edges = relationships, clusters = projects. Hoverable, zoomable. Shareable via URL.
- **Tech**: D3.js, host on GitHub Pages or embed in repo
- **Data advantage**: We have concept links, co-occurrence, project membership. Natural graph structure.
- **Dependencies**: Need to generate the graph JSON from trajectory DB
- **Effort**: Medium. D3 force layout is well-documented. HTML template + JSON data.
- **Priority**: HIGH — good balance of impact vs. effort

#### C. Contribution-Style Grid (concept heatmap)
- **Inspiration**: GitHub green squares, but for concepts. Isometric contributions.
- **Concept**: Grid where rows = themes/projects, columns = weeks, cell intensity = activity. Could be flat (SVG) or isometric 3D.
- **Tech**: SVG generation (Pillow or direct SVG), or Three.js for isometric
- **Data advantage**: We have per-concept, per-month activity data already.
- **Dependencies**: None — can spike immediately
- **Effort**: Small for flat, medium for isometric
- **Priority**: MEDIUM — recognizable format, easy to build, but less novel

#### D. "Code DNA" Strip
- **Concept**: Horizontal strip encoding your project evolution as a barcode/genome. Each band = a concept, color = level (theme/bet/technique), width = activity period, brightness = intensity. Unique per developer like a fingerprint.
- **Tech**: Pillow or SVG
- **Dependencies**: None
- **Effort**: Small
- **Priority**: MEDIUM — visually distinctive, very compact, could go on profile README

### Tier 2: Medium priority — worth exploring

#### E. 3D Concept City
- **Inspiration**: GitHub City, GitHub Skyline
- **Concept**: 3D city where buildings = projects, height = activity, neighborhoods = theme clusters. Orbitable in browser.
- **Tech**: Three.js
- **Dependencies**: Spike B (force graph) informs layout
- **Effort**: Large
- **Priority**: LOW — impressive but Three.js is heavy to build well

#### F. Animated SVG (Contribution Snake variant)
- **Inspiration**: Platane/snk contribution snake
- **Concept**: Animated SVG showing concepts growing and connecting over time. Embeddable in GitHub README. Auto-updates via GitHub Action.
- **Tech**: SVG animation, GitHub Action for generation
- **Dependencies**: Need static SVG first (Spike C or D)
- **Effort**: Medium
- **Priority**: MEDIUM — great distribution (GitHub READMEs) but less wow-factor than video

#### G. Git-of-Theseus for Concepts
- **Inspiration**: git-of-theseus "half-life of code" charts
- **Concept**: Stacked area chart showing concept age cohorts — what % of your current conceptual landscape was born in each month. "Half-life of ideas" stat.
- **Tech**: matplotlib or Pillow
- **Dependencies**: Need temporal concept data (already have)
- **Effort**: Small
- **Priority**: MEDIUM — the narrative ("half-life of ideas is X months") is very shareable even if the chart itself is simple

### Tier 3: Future / speculative

#### H. Physical Artifact (3D print)
- Like GitHub Skyline but for concepts. Export STL.
- **Dependencies**: Need 3D representation first (E)

#### I. LLM-Narrated Video
- AI voice narrates your project evolution over the animated visualization
- **Dependencies**: Need video generation first (A)

#### J. Multi-Developer Comparison
- Side-by-side wrapped cards or overlaid graphs for teams
- **Dependencies**: Need single-developer version solid first

## Experiment Dependency Graph

```
                    ┌─────────────┐
                    │ Data Layer  │ ← DONE (trajectory DB, queries)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼────┐  ┌───▼───┐  ┌────▼────┐
         │ Spike C │  │Spike D│  │ Spike G │
         │ Concept │  │ Code  │  │ Concept │
         │ Heatmap │  │  DNA  │  │ Half-   │
         │  Grid   │  │ Strip │  │  Life   │
         └────┬────┘  └───┬───┘  └─────────┘
              │           │
         ┌────▼────┐  ┌───▼───┐
         │ Spike F │  │Spike B│
         │Animated │  │ D3.js │
         │  SVG    │  │ Force │
         └─────────┘  │ Graph │
                      └───┬───┘
                          │
                     ┌────▼────┐
                     │ Spike A │
                     │ Animated│
                     │  Video  │
                     └────┬────┘
                          │
                ┌─────────▼─────────┐
                │ Full Product:     │
                │ Zero-friction     │
                │ "enter username,  │
                │  get your wrapped"│
                └───────────────────┘
```

## Recommended Experiment Order

| # | Experiment | Type | Why now |
|---|-----------|------|---------|
| 1 | **D. Code DNA strip** | Spike (1 session) | Cheapest to test. Unique visual. If it looks cool, validates the "fingerprint" concept. |
| 2 | **G. Concept half-life** | Spike (1 session) | One chart, one stat. If "half-life of ideas = X months" resonates, that's a viral hook by itself. |
| 3 | **C. Concept heatmap grid** | Spike (1 session) | GitHub green squares are the most recognized dev viz. Concept version is a clear pitch. |
| 4 | **B. D3.js force graph** | PoC (2-3 sessions) | First interactive output. Shareable URL. Tests whether concept graphs look compelling in browser. |
| 5 | **F. Animated SVG** | PoC (1-2 sessions) | GitHub README distribution. Builds on C. |
| 6 | **A. Animated video** | PoC → Product (3-5 sessions) | The big bet. Needs good frame rendering from earlier spikes. Remotion or ffmpeg. |
| 7 | **Full product** | Product | Web app: enter GitHub username → analyze repos → generate wrapped video/card → shareable URL |

## What Exists That We Should Use (not rebuild)

- **Remotion** (21k stars) — React → MP4 video framework. If we go video route, use this.
- **D3.js** — force-directed graphs, treemaps, circle packing. Don't hand-roll layouts.
- **Vega-Lite** — declarative grammar for interactive charts. Alternative to raw D3.
- **lowlighter/metrics** — study their plugin architecture if we want embeddable SVG cards.
- **Pillow** — already using for static image generation. Good enough for spikes.
- **matplotlib** — for quick analytical charts (concept half-life). Don't over-engineer.

## Schema Redesign (top-down)

### Step 1: Requirements — what does the system need to show?

A compelling developer portrait answers:
- **What** are they building? (concepts, architecture)
- **How** do they build? (tools, languages, workflow patterns)
- **When** do they build? (rhythms, bursts, streaks)
- **Where** does the work live? (files, projects, technologies)
- **Why** do they build it? (decisions, goals, intent)
- **How does it connect?** (cross-project deps, shared patterns)

Current extraction only covers "what" (concepts) and partially "why" (decisions). Everything else is unextracted or buried in raw_data JSON.

### Step 2: Domain model

```
DEVELOPER
  ├── PROJECTS ──────────── depends_on ──── other PROJECTS
  │     ├── TECHNOLOGIES ── (languages, frameworks, tools)
  │     ├── FILES ───────── implements ──── CONCEPTS
  │     └── MILESTONES ──── (phase transitions, releases)
  │
  ├── WORK SESSIONS ─────── (conversation + commits as a unit)
  │     ├── timing ──────── (duration, time-of-day, depth)
  │     ├── tool_usage ──── (Read vs Write vs Bash patterns)
  │     └── outcome ─────── (files changed, concepts touched)
  │
  ├── CONCEPTS ──────────── parent/child hierarchy
  │     ├── lifecycle ───── (emerging → growing → stable → declining)
  │     ├── co-occurrence ─ (which concepts travel together)
  │     └── cross-project ─ links
  │
  └── DECISIONS ─────────── linked to CONCEPTS + PROJECTS
```

### Step 3: Gap analysis

| Entity | Currently extracted? | What's missing |
|--------|---------------------|----------------|
| **Projects** | Name, path, remote | No inter-project deps. No language/tech profile. |
| **Technologies** | NO | File extensions exist in events but never aggregated. No framework detection. |
| **Files** | Partial (files_changed per event) | No file→concept map. No language tagging. No importance score. |
| **Work patterns** | Raw in JSON blobs | Token counts, tool sequences, durations never materialized. |
| **Concepts** | LLM extracted | No hierarchy (parent empty). No lifecycle. No importance score. Sparse links. |
| **Decisions** | LLM extracted | No direct decision→concept link. No outcome tracking. |
| **Milestones** | NO | No phase transition detection. |
| **Dependencies** | NO | No pyproject.toml parsing. No import graph. |

### Step 4: New tables

```sql
-- Deterministic extraction (no LLM)
project_technologies    (project_id, technology, category, file_count, first_seen, last_seen)
project_dependencies    (project_id, depends_on_project_id, dep_type, evidence)
work_patterns           (session_id, duration_minutes, message_count, token_count,
                         tool_read_count, tool_write_count, tool_bash_count,
                         files_examined_count, files_modified_count, hour_of_day)
file_profiles           (project_id, file_path, language, event_count, last_modified)

-- Materialized rollups
concept_activity        (concept_id, period, event_count, avg_significance, projects_active)

-- Computed columns on existing tables
concepts.importance     (composite: event_count × avg_significance × project_span × recency)
concepts.lifecycle      (emerging/growing/stable/declining/dormant from activity curve)
concepts.parent_concept_id  (populate: technique → parent theme)
```

### Step 5: New extractors (deterministic, no LLM)

| Extractor | Source | Output | Cost |
|-----------|--------|--------|------|
| **TechExtractor** | File extensions + pyproject.toml | project_technologies | Free |
| **DependencyExtractor** | pyproject.toml + import grep | project_dependencies | Free |
| **WorkPatternExtractor** | Claude session raw_data JSON | work_patterns | Free |
| **FileProfiler** | Aggregate file mentions from events | file_profiles | Free |
| **ConceptRollup** | concept_events + events | concept_activity + importance + lifecycle | Free |

### What each visualization gains from the new data

| Visualization | Before (concepts only) | After (full schema) |
|---|---|---|
| **Wrapped Card** | Top themes, project bars, heatmap | + "80% Python, 15% YAML" + "avg session: 45 msgs" + "most productive hour: 11pm" + project dependency tree |
| **Code DNA** | Concept bands over time | + technology bands + work intensity bands + milestone markers |
| **Force Graph** | 6 sparse concept links | + co-occurrence edges + project dependency edges + technology clustering |
| **Animated Video** | Concepts appearing over time | + projects spawning from dependencies + technology adoption arcs + work rhythm pulse |
| **Concept Half-Life** | Birth/death from first/last_seen | + proper lifecycle stages + activity curves + cohort analysis |

## Design Principles (learned from exploration)

1. **Data encodes the visual** — the visualization should look different for every developer because the DATA is different, not because of random AI art.
2. **Video > interactive > animated SVG > static image** for virality.
3. **Zero-friction matters more than visual quality** — "enter username, get result" beats "run this CLI tool."
4. **One compelling stat/narrative** beats a dashboard of numbers — "half-life of ideas" or "your #1 theme appeared in 5 projects" is more shareable than a treemap.
5. **Don't tile-stitch AI art** — it never looks coherent. One-shot or programmatic.
6. **Structural data > LLM-extracted data** for reliability and visualization. Technologies, dependencies, work patterns are deterministic and more trustworthy than concept extraction.
7. **Schema from the top down** — define requirements → domain model → schema. Don't let the extraction pipeline dictate what you can visualize.
