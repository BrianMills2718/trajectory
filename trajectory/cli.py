"""CLI entry point for trajectory tracker."""

import argparse
import logging
import sys
from pathlib import Path

from trajectory.analysis.concept_linker import link_concepts
from trajectory.analysis.event_classifier import analyze_project
from trajectory.config import load_config
from trajectory.db import TrajectoryDB
from trajectory.extractors.session_builder import build_sessions
from trajectory.ingest import ingest_all_projects, ingest_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory Tracker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command")

    # ingest command
    ingest_parser = sub.add_parser("ingest", help="Ingest project events")
    ingest_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    ingest_parser.add_argument(
        "--backfill", action="store_true",
        help="Re-extract all events and update enrichment columns (diff_summary, change_types)",
    )
    ingest_parser.add_argument(
        "project_path",
        nargs="?",
        help="Path to a specific project. If omitted, ingests all projects.",
    )

    # analyze command
    analyze_parser = sub.add_parser("analyze", help="Run LLM analysis on events")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    analyze_parser.add_argument(
        "--force-reanalyze", action="store_true",
        help="Clear existing analysis and re-analyze all events/sessions",
    )
    analyze_parser.add_argument(
        "project_path",
        help="Path to the project to analyze",
    )

    # build-sessions command
    sessions_parser = sub.add_parser("build-sessions", help="Build work sessions linking conversations to commits")
    sessions_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sessions_parser.add_argument(
        "project_path",
        nargs="?",
        help="Path to a specific project. If omitted, builds for all projects.",
    )

    # query command
    query_parser = sub.add_parser("query", help="Ask about project evolution")
    query_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    query_parser.add_argument("question", help="Natural language question")

    # link command
    link_parser = sub.add_parser("link", help="Find cross-project concept links")
    link_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    # stats command
    stats_parser = sub.add_parser("stats", help="Show ingestion stats")
    stats_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    # mural command
    mural_parser = sub.add_parser("mural", help="Generate AI art mural from trajectory data")
    mural_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    mural_parser.add_argument(
        "--themes", type=str, default=None,
        help="Comma-separated theme names (auto-selects if omitted)",
    )
    mural_parser.add_argument(
        "--months", type=str, default=None,
        help="Comma-separated YYYY-MM months (auto-selects if omitted)",
    )
    mural_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show layout and prompts without generating images",
    )

    # extract-tech command
    tech_parser = sub.add_parser("extract-tech", help="Extract technologies from file extensions + deps")
    tech_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    tech_parser.add_argument(
        "project_path",
        nargs="?",
        help="Path to a specific project. If omitted, extracts for all projects.",
    )

    # extract-patterns command
    wp_parser = sub.add_parser("extract-patterns", help="Extract work patterns from conversation events")
    wp_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    wp_parser.add_argument(
        "project_path",
        nargs="?",
        help="Path to a specific project. If omitted, extracts for all projects.",
    )

    # extract-deps command
    dep_parser = sub.add_parser("extract-deps", help="Extract cross-project dependencies")
    dep_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    # rollup command
    rollup_parser = sub.add_parser("rollup", help="Materialize concept activity + importance + lifecycle")
    rollup_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    # heatmap command
    hm_parser = sub.add_parser("heatmap", help="Generate concept heatmap for a project")
    hm_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    hm_parser.add_argument("project", help="Project name (as stored in trajectory DB)")
    hm_parser.add_argument("--max-concepts", type=int, default=40, help="Max concept rows")
    hm_parser.add_argument("--html", action="store_true", help="Output interactive HTML instead of PNG")

    # wrapped command — narrative insight cards
    wrap_parser = sub.add_parser("wrapped", help="Generate project Wrapped page (narrative insights)")
    wrap_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    wrap_parser.add_argument("project", help="Project name (as stored in trajectory DB)")

    # evolution command — animated concept graph
    evo_parser = sub.add_parser("evolution", help="Animated concept evolution timeline")
    evo_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    evo_parser.add_argument("project", help="Project name (as stored in trajectory DB)")

    # dataflow command — single-project dataflow mural
    df_parser = sub.add_parser("dataflow", help="Generate single-project dataflow mural")
    df_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    df_parser.add_argument("project", help="Project name (as stored in trajectory DB)")
    df_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show layout and prompts without generating images",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config()
    db = TrajectoryDB(config)
    db.init_db()

    try:
        if args.command == "ingest":
            if args.project_path:
                result = ingest_project(
                    Path(args.project_path), db, config, backfill=args.backfill,
                )
                print(result)
            else:
                results = ingest_all_projects(db, config, backfill=args.backfill)
                for r in results:
                    print(r)
                print(f"\nTotal: {len(results)} projects, {sum(r.total_new for r in results)} new events")

        elif args.command == "analyze":
            project = db.get_project_by_path(str(Path(args.project_path).resolve()))
            if not project:
                print(f"Project not found. Run 'ingest' first for {args.project_path}")
                return
            result = analyze_project(
                project.id, db, config,
                force_reanalyze=args.force_reanalyze,
            )
            print(result)

            # Show concepts found
            rows = db.conn.execute(
                "SELECT name, first_seen, last_seen FROM concepts ORDER BY name"
            ).fetchall()
            if rows:
                print(f"\nConcepts ({len(rows)}):")
                for r in rows:
                    print(f"  {r['name']} (first: {r['first_seen'][:10] if r['first_seen'] else '?'}, last: {r['last_seen'][:10] if r['last_seen'] else '?'})")

        elif args.command == "build-sessions":
            if args.project_path:
                project = db.get_project_by_path(str(Path(args.project_path).resolve()))
                if not project:
                    print(f"Project not found. Run 'ingest' first for {args.project_path}")
                    return
                result = build_sessions(db, project_id=project.id)
            else:
                result = build_sessions(db)
            print(result)

        elif args.command == "link":
            result = link_concepts(db, config)
            print(result)
            # Print links
            links = db.get_concept_links()
            if links:
                # Resolve names
                id_to_name: dict[int, str] = {}
                for link in links:
                    for cid in (link.concept_a_id, link.concept_b_id):
                        if cid not in id_to_name:
                            row = db.conn.execute(
                                "SELECT name FROM concepts WHERE id = ?", (cid,)
                            ).fetchone()
                            id_to_name[cid] = row["name"] if row else f"?{cid}"
                print(f"\nLinks ({len(links)}):")
                for link in links:
                    a = id_to_name[link.concept_a_id]
                    b = id_to_name[link.concept_b_id]
                    print(f"  {a} --[{link.relationship} {link.strength:.1f}]--> {b}")
                    if link.evidence:
                        print(f"    {link.evidence}")

        elif args.command == "query":
            from trajectory.output.query_engine import query_trajectory
            result = query_trajectory(args.question, db, config)
            print(result.answer)
            if result.data_gaps:
                print("\nData gaps:")
                for gap in result.data_gaps:
                    print(f"  - {gap}")

        elif args.command == "stats":
            projects = db.list_projects()
            if not projects:
                print("No projects ingested yet.")
                return
            for p in projects:
                total = db.count_events(p.id)
                sessions = db.get_sessions(project_id=p.id, limit=10000)
                print(
                    f"  {p.name}: {total} events, {len(sessions)} sessions "
                    f"({p.total_commits} commits, {p.total_conversations} conversations), "
                    f"last ingested: {p.last_ingested}"
                )

        elif args.command == "extract-tech":
            from trajectory.extractors.tech_extractor import (
                extract_all_technologies,
                extract_technologies,
            )

            if args.project_path:
                project = db.get_project_by_path(str(Path(args.project_path).resolve()))
                if not project:
                    print(f"Project not found. Run 'ingest' first for {args.project_path}")
                    return
                result = extract_technologies(db, project.id, Path(args.project_path).resolve())
                print(result)
                # Show breakdown
                techs = db.get_technologies(project.id)
                for t in techs:
                    print(f"  {t['category']:10s} {t['technology']:20s} files={t['file_count']}")
            else:
                results = extract_all_technologies(db)
                for r in results:
                    print(r)
                print(f"\nTotal: {len(results)} projects")

        elif args.command == "extract-patterns":
            from trajectory.extractors.work_pattern_extractor import (
                extract_all_work_patterns,
                extract_work_patterns,
            )

            if args.project_path:
                project = db.get_project_by_path(str(Path(args.project_path).resolve()))
                if not project:
                    print(f"Project not found. Run 'ingest' first for {args.project_path}")
                    return
                result = extract_work_patterns(db, project.id)
                print(result)
            else:
                results = extract_all_work_patterns(db)
                for r in results:
                    print(r)
                total_msgs = sum(r.total_messages for r in results)
                total_toks = sum(r.total_tokens for r in results)
                print(f"\nTotal: {len(results)} projects, {total_msgs} messages, {total_toks} tokens")

        elif args.command == "extract-deps":
            from trajectory.extractors.dep_extractor import extract_dependencies

            result = extract_dependencies(db)
            print(result)

            # Show cross-project links
            rows = db.conn.execute(
                """SELECT p1.name AS from_proj, p2.name AS to_proj, pd.dep_type, pd.evidence
                   FROM project_dependencies pd
                   JOIN projects p1 ON pd.project_id = p1.id
                   LEFT JOIN projects p2 ON pd.depends_on_project_id = p2.id
                   WHERE pd.depends_on_project_id IS NOT NULL
                   ORDER BY p1.name"""
            ).fetchall()
            if rows:
                print(f"\nCross-project links ({len(rows)}):")
                for r in rows:
                    print(f"  {r['from_proj']} → {r['to_proj']} ({r['dep_type']})")

        elif args.command == "rollup":
            from trajectory.extractors.concept_rollup import rollup_concept_activity

            result = rollup_concept_activity(db)
            print(result)

            # Show top concepts by importance
            top = db.conn.execute(
                "SELECT name, importance, lifecycle, level FROM concepts "
                "WHERE importance IS NOT NULL ORDER BY importance DESC LIMIT 15"
            ).fetchall()
            if top:
                print("\nTop 15 concepts by importance:")
                for c in top:
                    print(f"  {c['importance']:6.1f}  {c['lifecycle']:10s}  [{c['level'] or '?':12s}]  {c['name']}")

        elif args.command == "heatmap":
            from trajectory.output.concept_heatmap import generate_concept_heatmap

            path = generate_concept_heatmap(
                db,
                project_name=args.project,
                max_concepts=args.max_concepts,
                html=args.html,
            )
            print(f"Output: {path}")

        elif args.command == "wrapped":
            from trajectory.output.project_wrapped import generate_project_wrapped

            path = generate_project_wrapped(db, project_name=args.project)
            print(f"Output: {path}")

        elif args.command == "evolution":
            from trajectory.output.concept_evolution import generate_concept_evolution

            path = generate_concept_evolution(db, project_name=args.project)
            print(f"Output: {path}")

        elif args.command == "mural":
            from trajectory.output.mural import generate_mural

            theme_list = args.themes.split(",") if args.themes else None
            month_list = args.months.split(",") if args.months else None

            result = generate_mural(
                db, config,
                themes=theme_list,
                months=month_list,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                print(f"Output: {result.output_dir}")

        elif args.command == "dataflow":
            from trajectory.output.mural import generate_project_mural

            result = generate_project_mural(
                db, config,
                project_name=args.project,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                print(f"Output: {result.output_dir}")

        else:
            parser.print_help()
    finally:
        db.close()


if __name__ == "__main__":
    main()
