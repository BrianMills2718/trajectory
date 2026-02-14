# Trajectory Tracker

## What This Is

Project trajectory tracker — tracks how ideas emerge, evolve, and spread across Brian's 60+ projects. Ingests git commits, Claude Code conversation logs, and documentation into SQLite, then uses LLM analysis to extract concepts, significance scores, decisions, and intent from each event. Exposed as an MCP server for both Claude Code and OpenClaw/moltbot.

## Current State (2026-02-14)

### What's Built and Working (Phase 1 + Phase 2)

**Phase 1: Extractors + SQLite** — COMPLETE
- `trajectory/config.py` + `config.yaml` — Pydantic config loaded from YAML
- `trajectory/models.py` — All Pydantic models (EventType, Intent, ConceptStatus, ExtractedEvent, etc.)
- `trajectory/db.py` — SQLite with WAL mode, full schema (9 tables), all CRUD operations including analysis methods
- `trajectory/extractors/git_extractor.py` — PyDriller-based commit extraction
- `trajectory/extractors/claude_log_extractor.py` — JSONL parser for Claude Code logs with smart log dir matching (handles path aliases)
- `trajectory/extractors/doc_extractor.py` — Extracts from CLAUDE.md, STATUS.md, archive dirs, docs/archive dirs
- `trajectory/ingest.py` — Orchestrator that runs all 3 extractors with dedup by source_id
- `trajectory/cli.py` — CLI with `ingest`, `analyze`, `stats` commands

**Phase 2: LLM Analysis** — COMPLETE
- `trajectory/analysis/event_classifier.py` — Batches events (30 at a time), sends to LLM via `call_llm_structured()`, extracts intent, summary, concepts, significance (0.0-1.0), and decisions
- Uses Pydantic response models: `EventAnalysis`, `ConceptMention`, `DecisionFound`, `BatchAnalysisResult`
- Provenance tracking via `analysis_runs` table (model, prompt version, cost)
- Cost budget enforcement (stops if exceeds `max_cost_per_run`)
- Results stored: events updated with llm_summary/llm_intent/significance, concepts table populated, concept_events linked, decisions stored

**Tested on sam_gov:**
- 595 commits + 1 conversation + 68 docs = 664 events ingested
- All 664 events analyzed by LLM
- Concepts, decisions, significance scores all stored in SQLite

### What's NOT Built Yet

**Phase 3: MCP Server + Queries + Digests + Corrections**
- `trajectory/mcp_server.py` — 9 MCP tools (query_trajectory, get_timeline, get_concept_history, generate_narrative, generate_digest, ingest_project, list_tracked_projects, list_concepts, correct_concept)
- `trajectory/output/query_engine.py` — NL question → SQL retrieval → LLM synthesis
- Digest generation for daily/weekly summaries
- User corrections via NL commands
- MCP config for Claude Code and OpenClaw

**Phase 4: Visualization + Narrative Export**
- `trajectory/output/html_timeline.py` — D3.js interactive timeline
- `trajectory/output/markdown_exporter.py` — Narrative markdown reports

### Not Yet Done (Housekeeping)
- No entry in `/home/brian/projects/PROJECT_GRAPH.json` yet
- No initial git commit yet
- No tests written yet

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
│   ├── cli.py                     # CLI: ingest, analyze, stats
│   ├── extractors/
│   │   ├── base.py                # BaseExtractor ABC
│   │   ├── git_extractor.py       # PyDriller commit extraction
│   │   ├── claude_log_extractor.py # JSONL conversation log parser
│   │   └── doc_extractor.py       # CLAUDE.md/STATUS.md/archive parser
│   ├── analysis/
│   │   ├── __init__.py            # Adds llm_client to sys.path
│   │   └── event_classifier.py    # LLM batch classification
│   └── output/                    # Empty — Phase 3+4
├── data/
│   └── trajectory.db              # SQLite database (gitignored)
├── prompts/                       # Empty — for Jinja2 templates later
├── templates/                     # Empty — for D3.js HTML later
└── tests/                         # Empty — no tests yet
```

## Database Schema (9 tables)

- `projects` — tracked repos (name, path, git_remote, stats, last_ingested)
- `events` — unified timeline (commit/conversation/doc_change/archive events with LLM analysis fields)
- `concepts` — ideas/patterns that emerge and evolve (name, status, first_seen, last_seen)
- `concept_events` — links events to concepts with relationship type and confidence
- `decisions` — architectural/design decisions extracted from events
- `concept_links` — cross-concept relationships (depends_on, evolved_from, etc.)
- `analysis_runs` — provenance (model, prompt_version, cost, status)
- `corrections` — user correction audit trail
- `digests` — generated daily/weekly summaries

## Key Design Decisions

- **llm_client**: Uses `/home/brian/projects/llm_client` (added to sys.path in analysis/__init__.py). Default model: gemini-flash for bulk work.
- **Dedup**: Events deduplicated by `source_id` (UNIQUE constraint). Format: `git:{hash}`, `claude:{session_id}`, `doc:{project}/{path}`
- **Claude log key matching**: Claude Code log dirs use hyphenated paths (`-home-brian-sam-gov`). The extractor tries exact key match first, then falls back to suffix matching for path aliases.
- **Incremental processing**: Tracks `last_ingested` per project. Extractors accept `since` parameter. Re-runs only process new events.
- **Batch analysis**: 30 events per LLM call. Cost budget per run (default $1.00). Provenance tracked per analysis run.
- **Flattened Pydantic models**: Gemini has a nesting depth limit for structured output. The classifier uses `FlatEventAnalysis` with `concepts_json`/`decisions_json` as JSON strings, then `_unflatten_analysis()` converts back to rich typed models after the LLM call.

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

# Show stats
python -m trajectory.cli stats

# Query the DB directly
python3 -c "
import sqlite3
db = sqlite3.connect('data/trajectory.db')
db.row_factory = sqlite3.Row
for r in db.execute('SELECT name FROM concepts ORDER BY name'):
    print(r['name'])
"
```

## Venv Dependencies

```bash
pip install pydriller pydantic pyyaml python-dotenv litellm instructor
```

## What To Build Next (Phase 3)

Priority order:
1. `trajectory/output/query_engine.py` — NL query → SQL → LLM synthesis (this is the core user-facing feature)
2. `trajectory/mcp_server.py` — Thin MCP adapter exposing 9 tools
3. Wire MCP server into Claude Code config (`.mcp.json`)
4. Digest generation
5. Concept corrections

## Full Plan

The complete phased plan is at `/home/brian/.claude/plans/twinkling-dancing-turtle.md` — includes PRD, operators/policies, full schema, MCP tool specs, and Phase 4 visualization details.
