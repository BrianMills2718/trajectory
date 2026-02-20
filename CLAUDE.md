# Trajectory Tracker

## What This Is

Project trajectory tracker — tracks how ideas emerge, evolve, and spread across Brian's 60+ projects. Ingests git commits, Claude Code conversation logs, and documentation into SQLite, then uses LLM analysis to extract concepts, significance scores, decisions, and intent from each event. Exposed as an MCP server for both Claude Code and OpenClaw/moltbot.

## Current State (2026-02-19)

### What's Built and Working (Phase 1 + Phase 2 + Phase 3)

**Phase 1: Extractors + SQLite** — COMPLETE
- `trajectory/config.py` + `config.yaml` — Pydantic config loaded from YAML
- `trajectory/models.py` — All Pydantic models (EventType, Intent, ConceptStatus, QueryResult, etc.)
- `trajectory/db.py` — SQLite with WAL mode, full schema (9 tables), all CRUD + query + correction operations
- `trajectory/extractors/git_extractor.py` — PyDriller-based commit extraction
- `trajectory/extractors/claude_log_extractor.py` — JSONL parser for Claude Code logs with smart log dir matching (handles path aliases)
- `trajectory/extractors/doc_extractor.py` — Extracts from CLAUDE.md, STATUS.md, archive dirs, docs/archive dirs
- `trajectory/ingest.py` — Orchestrator that runs all 3 extractors with dedup by source_id
- `trajectory/cli.py` — CLI with `ingest`, `analyze`, `stats`, `query`, `link` commands

**Phase 2: LLM Analysis** — COMPLETE
- `trajectory/analysis/event_classifier.py` — Batches events (30 at a time), sends to LLM via `call_llm_structured()`, extracts intent, summary, concepts, significance (0.0-1.0), and decisions
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

**Test Suite: 57 tests passing**
- `tests/test_db.py` — 26 tests for DB query and correction methods
- `tests/test_query_engine.py` — 16 tests for query engine functions (LLM calls mocked)
- `tests/test_mcp_server.py` — 4 tests for tool registration and JSON returns
- `tests/test_concept_linker.py` — 11 tests for concept link DB methods, query engine, and MCP tool

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

### What's NOT Built Yet

**Phase 3 Remaining:**
- Digest generation for daily/weekly summaries (deferred)

**Phase 4: Visualization + Narrative Export**
- `trajectory/output/html_timeline.py` — D3.js interactive timeline
- `trajectory/output/markdown_exporter.py` — Narrative markdown reports

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
│   ├── cli.py                     # CLI: ingest, analyze, stats, query
│   ├── mcp_server.py              # MCP server — 7 tools
│   ├── extractors/
│   │   ├── base.py                # BaseExtractor ABC
│   │   ├── git_extractor.py       # PyDriller commit extraction
│   │   ├── claude_log_extractor.py # JSONL conversation log parser
│   │   └── doc_extractor.py       # CLAUDE.md/STATUS.md/archive parser
│   ├── analysis/
│   │   ├── __init__.py            # Adds llm_client to sys.path
│   │   ├── event_classifier.py    # LLM batch classification
│   │   └── concept_linker.py      # Cross-project concept linking
│   └── output/
│       └── query_engine.py        # NL query → SQL → LLM synthesis
├── data/
│   └── trajectory.db              # SQLite database (gitignored)
├── prompts/
│   ├── query_synthesis.yaml       # Jinja2 template for query synthesis
│   └── concept_linking.yaml       # Jinja2 template for concept linking
├── templates/                     # Empty — for D3.js HTML later
└── tests/
    ├── conftest.py                # Shared fixtures (tmp_db, populated_db)
    ├── test_db.py                 # 26 DB method tests
    ├── test_query_engine.py       # 16 query engine tests
    └── test_mcp_server.py         # 4 MCP server tests
```

## Database Schema (9 tables)

- `projects` — tracked repos (name, path, git_remote, stats, last_ingested)
- `events` — unified timeline (commit/conversation/doc_change/archive events with LLM analysis fields)
- `concepts` — ideas/patterns that emerge and evolve (name, level, level_rationale, status, first_seen, last_seen)
- `concept_events` — links events to concepts with relationship type and confidence
- `decisions` — architectural/design decisions extracted from events
- `concept_links` — cross-concept relationships (depends_on, evolved_from, etc.)
- `analysis_runs` — provenance (model, prompt_version, cost, status)
- `corrections` — user correction audit trail
- `digests` — generated daily/weekly summaries

## Key Design Decisions

- **llm_client**: Uses `/home/brian/projects/llm_client` (pip installed in .venv). Default model: gemini-flash for bulk work, quality_model (claude-sonnet) for synthesis.
- **Dedup**: Events deduplicated by `source_id` (UNIQUE constraint). Format: `git:{hash}`, `claude:{session_id}`, `doc:{project}/{path}`
- **Claude log key matching**: Claude Code log dirs use hyphenated paths (`-home-brian-sam-gov`). The extractor tries exact key match first, then falls back to suffix matching for path aliases.
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

# Ingest all projects under ~/projects
python -m trajectory.cli ingest

# Run LLM analysis on ingested events
python -m trajectory.cli analyze /home/brian/projects/sam_gov

# Find cross-project concept links (LLM-powered, theme-level only)
python -m trajectory.cli link

# Ask about project evolution (NL query → LLM synthesis)
python -m trajectory.cli query "How has the ontology idea evolved?"

# Show stats
python -m trajectory.cli stats

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
```

## Full Plan

The complete phased plan is at `/home/brian/.claude/plans/twinkling-dancing-turtle.md` — includes PRD, operators/policies, full schema, MCP tool specs, and Phase 4 visualization details.
