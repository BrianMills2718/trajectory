# Trajectory Tracker

## What This Is

Project trajectory tracker — tracks how ideas emerge, evolve, and spread across Brian's 60+ projects. Ingests git commits, Claude Code conversation logs, and documentation into SQLite, then uses LLM analysis to extract concepts, significance scores, decisions, and intent from each event. Exposed as an MCP server for both Claude Code and OpenClaw/moltbot.

## Current State (2026-02-19)

### What's Built and Working (Phase 1 + Phase 2 + Phase 3)

**Phase 1: Extractors + SQLite** — COMPLETE
- `trajectory/config.py` + `config.yaml` — Pydantic config loaded from YAML
- `trajectory/models.py` — All Pydantic models (EventType, Intent, ConceptStatus, QueryResult, WorkSessionRow, etc.)
- `trajectory/db.py` — SQLite with WAL mode, full schema (11 tables), all CRUD + query + correction + session operations
- `trajectory/extractors/git_extractor.py` — PyDriller-based commit extraction with diff summaries and change types
- `trajectory/extractors/claude_log_extractor.py` — JSONL parser with enriched extraction (commit hashes, files modified/examined, tool sequence, assistant reasoning). Scans both project-specific AND catch-all parent log directories
- `trajectory/extractors/doc_extractor.py` — Extracts from CLAUDE.md, STATUS.md, archive dirs, docs/archive dirs
- `trajectory/extractors/session_builder.py` — Links conversations to commits via hash matching, groups orphan commits by day
- `trajectory/ingest.py` — Orchestrator that runs all 3 extractors with dedup by source_id, supports `--backfill`
- `trajectory/cli.py` — CLI with `ingest`, `analyze`, `build-sessions`, `stats`, `query`, `link` commands

**Phase 2: LLM Analysis** — COMPLETE
- `trajectory/analysis/event_classifier.py` — Two-phase: session-level analysis first (richer context), then remaining orphan events. Prompts loaded from YAML templates via `render_prompt()`
- `prompts/session_classification.yaml` — Jinja2 template for session-level analysis (user goal + reasoning + files + commits)
- `prompts/event_classification.yaml` — Jinja2 template for event-level analysis (migrated from inline f-string)
- Uses Pydantic response models: `EventAnalysis`, `ConceptMention`, `DecisionFound`, `BatchAnalysisResult`
- Provenance tracking via `analysis_runs` table (model, prompt version, cost)
- Cost budget enforcement (stops if exceeds `max_cost_per_run`)
- Results stored: events updated with llm_summary/llm_intent/significance, concepts table populated, concept_events linked, decisions stored

**Phase 3: Query Engine + MCP Server** — COMPLETE
- `trajectory/output/query_engine.py` — 8 functions: NL question → SQL retrieval → LLM synthesis, timelines, concept history, project listing, concept listing, concept links, concept corrections, project ingestion
- `trajectory/mcp_server.py` — 8 MCP tools wrapping query engine (query_trajectory, get_timeline, get_concept_history, list_tracked_projects, list_concepts, get_concept_links, ingest_project, correct_concept)
- `prompts/query_synthesis.yaml` — Jinja2 template for NL synthesis via `llm_client.render_prompt()`
- MCP server wired into both Claude Code (`~/.config/claude-cli-nodejs/mcp.json`) and Codex CLI (`~/.codex/config.toml`)
- Keyword-based concept search (SQL LIKE on ~100 concepts — fast, no vector DB needed)
- Explicit data gaps in QueryResult (reports unanalyzed event counts, missing concepts)
- Concept corrections with audit trail: rename, merge, status change → `corrections` table

**Tested on sam_gov:**
- 595 commits + 1 conversation + 68 docs = 664 events ingested
- All 664 events analyzed by LLM
- Concepts, decisions, significance scores all stored in SQLite

**Test Suite: 96 tests passing**
- `tests/test_db.py` — 34 tests for DB query, correction, session, and backfill methods
- `tests/test_query_engine.py` — 16 tests for query engine functions (LLM calls mocked)
- `tests/test_mcp_server.py` — 4 tests for tool registration and JSON returns
- `tests/test_concept_linker.py` — 11 tests for concept link DB methods, query engine, and MCP tool
- `tests/test_claude_log_extractor.py` — 15 tests for enriched extraction (commit hashes, files, reasoning, catch-all dirs)
- `tests/test_git_extractor.py` — 6 tests for diff summary generation
- `tests/test_session_builder.py` — 10 tests for session creation, orphan grouping, dedup

**Multi-Level Concept Extraction (2026-02-18):**
- Concepts now carry a `level`: theme (project identity), design_bet (architectural choice), technique (implementation mechanism)
- Open vocabulary — LLM can invent new levels with `level_rationale` explaining why
- Prompt v3 (`event_classification_v3`) guides extraction with examples per level
- `list_concepts` MCP tool + query engine accept `level` filter
- DB migration adds `level` + `level_rationale` columns (idempotent)
- First-extraction-wins: if a concept already has a level, later extractions don't overwrite
- Verified on agent_ontology: 18 themes, 38 design_bets, 69 techniques (1 off-label "process" fixed)

### Validation Results (2026-02-19) — GATE PASSED

Multi-level extraction validated across 3 diverse project types:

| Project | Events | Themes | Design Bets | Techniques | Cost |
|---------|--------|--------|-------------|------------|------|
| agent_ontology | 79 | 18 | 38 | 69 | $0.05 |
| llm_client | 310 | 11 | 23 | 25 | $0.13 |
| sam_gov | 672 | 29 | 131 | 190 | $0.26 |

**Cross-project consistency**: 4 shared concepts, all with consistent levels across projects.
**Off-label concepts**: Zero. Open vocabulary produced no noise — all concepts fit theme/design_bet/technique.
**NULL-level concepts**: 15 (pre-v3 leftovers) → bulk-fixed to technique.

**Issues found**:
- sam_gov has 29 themes — too many. ~10 are really techniques (repository_maintenance, tag_formatting, etc.). File names leaked in as themes (TODO_ARCHITECTURE.md, PATTERNS.md). May need prompt tuning or post-hoc correction.
- llm_client `prompt_templates` tagged 187 events — inflated by Claude Code conversations. Not a level problem, but a concept-granularity one.
- `gemini-2.5-flash-lite` can't handle the nested Pydantic schema (nesting depth limit). Must use `gpt-5-mini` or larger.
- Timeout needed increase from 120s → 300s for projects with 200+ existing concepts.

**Verdict**: Themes are coherent across projects. Proceed to cross-project concept linking.

**Cross-Project Concept Linking (2026-02-19):**
- `trajectory/analysis/concept_linker.py` — `link_concepts()` loads all theme-level concepts, calls LLM structured output to find cross-project links, full-replacement strategy (delete all + rewrite)
- `prompts/concept_linking.yaml` — Goal-framed Jinja2 template: themes with project membership, event counts, dates. 5 relationship types: depends_on, evolved_from, replaced_by, related_to, spawned
- DB methods: `upsert_concept_link()`, `delete_all_concept_links()`, `get_concept_links()` with concept/relationship/min_strength filters
- Query engine: `get_concept_links()` resolves concept IDs to names
- CLI: `python -m trajectory.cli link` — runs linker, prints all links with evidence
- MCP tool: `get_concept_links(concept_name?, relationship?, min_strength?)`
- LLM response models: `ConceptLinkFound`, `ConceptLinkingResult`
- Validation: rejects unknown concept names, self-links, invalid relationship types

**Phase 4: Mural Visualization** — IN PROGRESS
- `trajectory/output/mural.py` — AI art mural generator (v2). LLM-planned vertical layout (themes arranged by conceptual relationships), corner-out scanline generation (each tile prompt references already-generated neighbors for coherence). Alpha-blended edges for compositing.
- `prompts/mural_tile.yaml` — v4.0 Jinja2 template. Concrete/literal style — physical workspaces, not abstract imagery. Theme name appears as physical object in scene.
- `prompts/mural_layout.yaml` — LLM arranges themes into vertical order based on concept links and co-occurrence.
- `MuralConfig` in config.py — tile_size, vertical_overlap, style_suffix, prompt_model, image_model, max_cost, max_projects, max_months
- Image generation via Gemini `gemini-2.5-flash-image` (free tier, google-genai SDK)
- Prompt generation via `gemini/gemini-2.5-flash-lite` through `llm_client`
- Seam blending: Stability AI inpainting tested for seamless tile transitions ($0.01/credit)
- CLI: `python -m trajectory.cli mural [--themes ...] [--months ...] [--dry-run]`
- Dependencies: Pillow, google-genai

**Deterministic Extractors (2026-02-20):** — COMPLETE
- `trajectory/extractors/tech_extractor.py` — Languages, frameworks, tools from file extensions + pyproject.toml/package.json. Populates `project_technologies` table. 44 projects × ~10 techs each.
- `trajectory/extractors/work_pattern_extractor.py` — Materializes session stats (message counts, token counts, tool usage, hour of day, day of week) from conversation event raw_data. Populates `work_patterns` table. 811 conversation events → 811 patterns.
- `trajectory/extractors/dep_extractor.py` — Cross-project dependencies from pyproject.toml deps + `pip install -e` commands. Populates `project_dependencies` table. Found 4 cross-project links.
- `trajectory/extractors/concept_rollup.py` — Materializes `concept_activity` (per-concept, per-month event counts), computes `concepts.importance` (log-scaled event count × significance × project span × recency), assigns `concepts.lifecycle` (emerging/growing/stable/declining/dormant). 1133 concepts scored.
- CLI: `extract-tech`, `extract-patterns`, `extract-deps`, `rollup` subcommands.
- All deterministic — zero LLM calls, zero cost.

**Dataflow Mural + Wrapped Card** — EXPLORATORY
- `trajectory/output/mural.py` — Extended with dataflow mode: LLM generates 3×3 dependency graph layout, generates tile per node.
- `trajectory/output/wrapped.py` — Developer Wrapped card (1080×1920): treemap of themes, project×month heatmap, bold stats.
- `prompts/dataflow_layout.yaml` — LLM generates dataflow graph as JSON (9 nodes, 3×3 grid).
- `prompts/dataflow_tile.yaml` — Literal software-process image prompts.
- CLI: `dataflow <project>` subcommand.
- See `docs/VISUALIZATION_EXPERIMENTS.md` for full experiment tracking and backlog.

### What's NOT Built Yet

**Phase 3 Remaining:**
- Digest generation for daily/weekly summaries (deferred)

**Phase 4 Visualization Spikes (next):**
- Concept heatmap grid — DONE (`trajectory/output/concept_heatmap.py`). Auto-granularity, per-level color ramps, interactive HTML mode. CLI: `heatmap <project> [--max-concepts 40] [--html]`
- Project wrapped — DONE (`trajectory/output/project_wrapped.py`). Spotify Wrapped-style insight cards. CLI: `wrapped <project>`
- Concept evolution — DONE (`trajectory/output/concept_evolution.py`). Animated D3.js force graph with beats, narrative moments, auto-camera. CLI: `evolution <project>`
- Project narrative — DONE (`trajectory/output/project_narrative.py`). LLM-synthesized story of a project's intellectual journey. Gathers events/concepts/decisions, feeds to LLM via Jinja2 prompt, renders dark-themed HTML. CLI: `narrative <project> [--model ...]`
- Doc narrative — DONE (`trajectory/output/project_narrative.py`). Analyzes versioned markdown documents (no DB needed). 3-pass pipeline: parse versions → LLM concept extraction → narrative + journey diagram. Handles V-numbered, number-prefixed, and named version headers. CLI: `doc-narrative <path> [--name ...]`
- Cross-doc narrative — DONE (`trajectory/output/project_narrative.py`). Analyzes concept migration across multiple versioned documents. 5-step pipeline: parse each doc → extract concepts → LLM merge (shared/migrated concepts) → unified narrative → journey diagram. CLI: `cross-doc <paths...> [--name ...]`
- Code DNA strip (deferred)
- Concept half-life chart (deferred)
- See `docs/VISUALIZATION_EXPERIMENTS.md` for full backlog + dependency graph

## Architecture

```
trajectory/
├── config.yaml                    # All configuration
├── pyproject.toml                 # Dependencies: pydriller, pydantic, pyyaml, python-dotenv
├── trajectory/
│   ├── config.py                  # Pydantic config from YAML
│   ├── models.py                  # All Pydantic models
│   ├── db.py                      # TrajectoryDB class — SQLite wrapper with all operations
│   ├── ingest.py                  # Orchestrator: runs extractors, dedup, stores events
│   ├── cli.py                     # CLI: ingest, analyze, stats, query, extract-tech, extract-patterns, extract-deps, rollup, heatmap, wrapped, evolution, narrative, doc-narrative, cross-doc, mural, dataflow
│   ├── mcp_server.py              # MCP server — 8 tools
│   ├── extractors/
│   │   ├── base.py                # BaseExtractor ABC
│   │   ├── git_extractor.py       # PyDriller commit extraction + diff summaries
│   │   ├── claude_log_extractor.py # JSONL parser — enriched extraction + catch-all scanning
│   │   ├── doc_extractor.py       # CLAUDE.md/STATUS.md/archive parser
│   │   ├── session_builder.py     # Links conversations to commits → work_sessions
│   │   ├── tech_extractor.py      # Deterministic: file extensions + pyproject.toml → technologies
│   │   ├── work_pattern_extractor.py  # Deterministic: conversation raw_data → work patterns
│   │   ├── dep_extractor.py       # Deterministic: pyproject.toml + pip install -e → project deps
│   │   └── concept_rollup.py      # Deterministic: concept_activity + importance + lifecycle
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── event_classifier.py    # Session-level + event-level LLM classification
│   │   └── concept_linker.py      # Cross-project concept linking
│   └── output/
│       ├── query_engine.py        # NL query → SQL → LLM synthesis
│       ├── concept_heatmap.py     # Concept heatmap grid (GitHub-squares style, auto-granularity)
│       ├── concept_evolution.py   # Animated D3.js force graph with beats + narrative moments
│       ├── project_wrapped.py     # Spotify Wrapped-style insight cards
│       ├── project_narrative.py   # LLM-synthesized project story → HTML
│       ├── mural.py               # AI art mural generator (themes × months + dataflow)
│       └── wrapped.py             # Developer Wrapped card (treemap + heatmap)
├── data/
│   └── trajectory.db              # SQLite database (gitignored)
├── prompts/
│   ├── query_synthesis.yaml       # Jinja2 template for query synthesis
│   ├── concept_linking.yaml       # Jinja2 template for concept linking
│   ├── session_classification.yaml # Jinja2 template for session-level analysis
│   ├── event_classification.yaml  # Jinja2 template for event-level analysis
│   ├── project_narrative.yaml      # Jinja2 template for project narrative synthesis
│   ├── mural_tile.yaml            # Jinja2 template for mural art prompts
│   ├── dataflow_layout.yaml       # LLM generates dataflow graph as JSON
│   └── dataflow_tile.yaml         # Literal software-process tile prompts
├── templates/                     # Empty — for D3.js HTML later
└── tests/
    ├── conftest.py                # Shared fixtures (tmp_db, populated_db)
    ├── test_db.py                 # 34 DB method tests
    ├── test_query_engine.py       # 16 query engine tests
    ├── test_mcp_server.py         # 4 MCP server tests
    ├── test_concept_linker.py     # 11 concept link tests
    ├── test_claude_log_extractor.py # 15 enriched extraction tests
    ├── test_git_extractor.py      # 6 diff summary tests
    └── test_session_builder.py    # 10 session builder tests
```

## Database Schema (15 tables)

- `projects` — tracked repos (name, path, git_remote, stats, last_ingested)
- `events` — unified timeline (commit/conversation/doc_change/archive events with LLM analysis fields, diff_summary, change_types, session_id)
- `work_sessions` — conversations linked to their commits (user_goal, tool_sequence, files_modified, commit_hashes, assistant_reasoning, diff_summary)
- `session_events` — junction table linking sessions to events with role (conversation/commit)
- `concepts` — ideas/patterns that emerge and evolve (name, level, level_rationale, status, first_seen, last_seen, importance, lifecycle)
- `concept_events` — links events to concepts with relationship type and confidence
- `concept_activity` — per-concept, per-month event counts + avg significance + project count (materialized)
- `decisions` — architectural/design decisions extracted from events
- `concept_links` — cross-concept relationships (depends_on, evolved_from, etc.)
- `project_technologies` — languages, frameworks, tools per project (deterministic, from file extensions + deps)
- `project_dependencies` — cross-project + external deps (from pyproject.toml + pip install -e)
- `work_patterns` — per-conversation materialized stats (message counts, token counts, tool usage, timing)
- `analysis_runs` — provenance (model, prompt_version, cost, status)
- `corrections` — user correction audit trail
- `digests` — generated daily/weekly summaries

## Key Design Decisions

- **llm_client**: Uses `/home/brian/projects/llm_client` (pip installed in .venv). Default model: gemini-flash for bulk work, quality_model (claude-sonnet) for synthesis.
- **Dedup**: Events deduplicated by `source_id` (UNIQUE constraint). Format: `git:{hash}`, `claude:{session_id}`, `doc:{project}/{path}`
- **Claude log matching**: Three-tier lookup: (1) exact path key match, (2) suffix match for aliases, (3) catch-all parent directory scan. Catch-all scan uses streaming 1MB-chunk binary search with early exit — handles 5.3GB / 145 files in ~1 second.
- **Work sessions**: A conversation + the commits it produced. Session builder matches short commit hashes from Claude logs to git events via `source_id` prefix. Orphan commits (no conversation) grouped by day.
- **Incremental processing**: Tracks `last_ingested` per project. Extractors accept `since` parameter. Re-runs only process new events.
- **Batch analysis**: 30 events per LLM call. Cost budget per run (default $1.00). Provenance tracked per analysis run.
- **Flattened Pydantic models**: Gemini has a nesting depth limit for structured output. The classifier uses `FlatEventAnalysis` with `concepts_json`/`decisions_json` as JSON strings, then `_unflatten_analysis()` converts back to rich typed models after the LLM call.
- **Keyword search, not embeddings**: `search_concepts` uses SQL LIKE on ~100 concepts. Fast, debuggable, no vector DB.
- **Absolute prompt path**: Query engine uses `Path(__file__).parent.parent / "prompts"` to avoid cwd sensitivity.

## How to Run

```bash
cd /home/brian/projects/trajectory
source .venv/bin/activate

# Ingest a single project
python -m trajectory.cli ingest /home/brian/projects/sam_gov

# Ingest with backfill (re-enrich existing events with diff summaries)
python -m trajectory.cli ingest --backfill /home/brian/projects/sam_gov

# Ingest all projects under ~/projects
python -m trajectory.cli ingest

# Build work sessions (link conversations to commits)
python -m trajectory.cli build-sessions

# Run LLM analysis on ingested events
python -m trajectory.cli analyze /home/brian/projects/sam_gov

# Force re-analyze all events (useful after prompt changes)
python -m trajectory.cli analyze --force-reanalyze /home/brian/projects/sam_gov

# Find cross-project concept links (LLM-powered, theme-level only)
python -m trajectory.cli link

# Ask about project evolution (NL query → LLM synthesis)
python -m trajectory.cli query "How has the ontology idea evolved?"

# Show stats
python -m trajectory.cli stats

# Deterministic extractors (no LLM, zero cost)
python -m trajectory.cli extract-tech              # all projects
python -m trajectory.cli extract-tech /home/brian/projects/sam_gov
python -m trajectory.cli extract-patterns          # all projects
python -m trajectory.cli extract-deps              # cross-project deps
python -m trajectory.cli rollup                    # concept activity + importance + lifecycle

# Concept heatmap (GitHub-squares style)
python -m trajectory.cli heatmap agent_ontology    # single project
python -m trajectory.cli heatmap agent_ontology --max-concepts 60 --html

# Project narrative (LLM-synthesized story)
python -m trajectory.cli narrative agent_ontology
python -m trajectory.cli narrative agent_ontology --model gemini/gemini-2.5-flash

# Other visualizations
python -m trajectory.cli wrapped agent_ontology    # Wrapped-style insight cards
python -m trajectory.cli evolution agent_ontology  # Animated concept graph

# Document narrative (no DB — analyzes versioned markdown directly)
python -m trajectory.cli doc-narrative "/path/to/Versioned Doc.md"
python -m trajectory.cli doc-narrative "/path/to/doc.md" --name my_analysis

# Cross-document concept migration analysis
python -m trajectory.cli cross-doc "/path/to/doc1.md" "/path/to/doc2.md" "/path/to/doc3.md"
python -m trajectory.cli cross-doc "/path/to/doc1.md" "/path/to/doc2.md" --name writing_evolution

# Generate AI art mural (themes × months, PCA-positioned)
python -m trajectory.cli mural --dry-run          # preview layout + prompts
python -m trajectory.cli mural                      # auto-select themes/months
python -m trajectory.cli mural --themes graph_rag,prompt_templates --months 2026-02

# Run tests
python -m pytest tests/ -v
```

## MCP Server

8 tools available via both Claude Code and Codex CLI:

| Tool | Purpose |
|------|---------|
| `query_trajectory` | NL question → concept search → event retrieval → LLM synthesis |
| `get_timeline` | Chronological events for a project with date/significance filters |
| `get_concept_history` | Full history of a concept across all projects |
| `list_tracked_projects` | All projects with event counts and analysis status |
| `list_concepts` | Concepts with optional status/project/level filters |
| `get_concept_links` | Cross-project concept links with optional concept/relationship/strength filters |
| `ingest_project` | Ingest events from a project directory |
| `correct_concept` | Rename, merge, or change status of a concept |

## Venv Dependencies

```bash
pip install pydriller pydantic pyyaml python-dotenv litellm instructor jinja2
pip install -e ~/projects/llm_client
pip install "mcp>=1.0"
pip install Pillow google-genai  # for mural generation
```

## Full Plan

The complete phased plan is at `/home/brian/.claude/plans/twinkling-dancing-turtle.md` — includes PRD, operators/policies, full schema, MCP tool specs, and Phase 4 visualization details.


## Multi-Agent Coordination

This repo uses worktree-based isolation for concurrent AI instances.

**Before starting work:**
1. Check existing claims: `python scripts/meta/worktree-coordination/check_claims.py --list`
2. Claim your work: `python scripts/meta/worktree-coordination/check_claims.py --claim --feature <name> --task "description"`
3. Create a worktree: `make worktree` (or `git worktree add worktrees/plan-N-desc`)
4. Work in the worktree, not the main directory

**Before committing:**
- Commits must use prefixes: `[Plan #N]`, `[Trivial]`, or `[Unplanned]`
- Release claims when done: `python scripts/meta/worktree-coordination/check_claims.py --release`

**Check for messages from other instances:**
`python scripts/meta/worktree-coordination/check_messages.py`

