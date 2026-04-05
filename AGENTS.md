# Trajectory Tracker

<!-- GENERATED FILE: DO NOT EDIT DIRECTLY -->
<!-- generated_by: scripts/meta/render_agents_md.py -->
<!-- canonical_claude: CLAUDE.md -->
<!-- canonical_relationships: scripts/relationships.yaml -->
<!-- canonical_relationships_sha256: c597a2e79109 -->
<!-- sync_check: python scripts/meta/check_agents_sync.py --check -->

This file is a generated Codex-oriented projection of repo governance.
Edit the canonical sources instead of editing this file directly.

Canonical governance sources:
- `CLAUDE.md` — human-readable project rules, workflow, and references
- `scripts/relationships.yaml` — machine-readable ADR, coupling, and required-reading graph

## Purpose

Trajectory Tracker uses `CLAUDE.md` as canonical repo governance and workflow policy.

## Commands

```bash
# Ingest data (git commits, Claude logs, docs)
python -m trajectory.cli ingest

# Run LLM analysis
python -m trajectory.cli analyze

# Query concepts
python -m trajectory.cli query "search query"

# Start MCP server
python trajectory_mcp_server.py

# Build sessions
python -m trajectory.cli sessions
```

## Operating Rules

This projection keeps the highest-signal rules in always-on Codex context.
For full project structure, detailed terminology, and any rule omitted here,
read `CLAUDE.md` directly.

### Principles

- Fail loud: all LLM calls use structured Pydantic output; validation errors surface, not silence
- Observability: `analysis_runs` table tracks every LLM call (model, prompt version, cost, provenance)
- Budget-aware: cost caps on analysis runs enforced by config
- Schema-first: 15-table SQLite schema defined in `trajectory/db.py`

### Workflow

1. Ingest: `python -m trajectory.cli ingest` (git + Claude logs + docs → SQLite)
2. Analyze: `python -m trajectory.cli analyze` (LLM extracts concepts/decisions)
3. Query: `python -m trajectory.cli query "..."` (semantic concept search)
4. Access via MCP server for Claude Code / OpenClaw integration

## Machine-Readable Governance

`scripts/relationships.yaml` is the source of truth for machine-readable governance in this repo: ADR coupling, required-reading edges, and doc-code linkage. This generated file does not inline that graph; it records the canonical path and sync marker, then points operators and validators back to the source graph. Prefer deterministic validators over prompt-only memory when those scripts are available.

## References

- `CLAUDE.md` — This file (canonical operating guidance)
- `AGENTS.md` — Generated mirror for non-Claude agents
- `config.yaml` — Configuration (DB path, LLM model, cost limits)
- `docs/` — Design docs and plans
- `trajectory/models.py` — All Pydantic models
