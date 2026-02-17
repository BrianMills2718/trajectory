"""CLI entry point for trajectory tracker."""

import argparse
import logging
import sys
from pathlib import Path

from trajectory.analysis.event_classifier import analyze_project
from trajectory.config import load_config
from trajectory.db import TrajectoryDB
from trajectory.ingest import ingest_all_projects, ingest_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory Tracker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command")

    # ingest command
    ingest_parser = sub.add_parser("ingest", help="Ingest project events")
    ingest_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    ingest_parser.add_argument(
        "project_path",
        nargs="?",
        help="Path to a specific project. If omitted, ingests all projects.",
    )

    # analyze command
    analyze_parser = sub.add_parser("analyze", help="Run LLM analysis on events")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    analyze_parser.add_argument(
        "project_path",
        help="Path to the project to analyze",
    )

    # query command
    query_parser = sub.add_parser("query", help="Ask about project evolution")
    query_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    query_parser.add_argument("question", help="Natural language question")

    # stats command
    stats_parser = sub.add_parser("stats", help="Show ingestion stats")
    stats_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

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
                result = ingest_project(Path(args.project_path), db, config)
                print(result)
            else:
                results = ingest_all_projects(db, config)
                for r in results:
                    print(r)
                print(f"\nTotal: {len(results)} projects, {sum(r.total_new for r in results)} new events")

        elif args.command == "analyze":
            project = db.get_project_by_path(str(Path(args.project_path).resolve()))
            if not project:
                print(f"Project not found. Run 'ingest' first for {args.project_path}")
                return
            result = analyze_project(project.id, db, config)
            print(result)

            # Show concepts found
            rows = db.conn.execute(
                "SELECT name, first_seen, last_seen FROM concepts ORDER BY name"
            ).fetchall()
            if rows:
                print(f"\nConcepts ({len(rows)}):")
                for r in rows:
                    print(f"  {r['name']} (first: {r['first_seen'][:10] if r['first_seen'] else '?'}, last: {r['last_seen'][:10] if r['last_seen'] else '?'})")

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
                print(f"  {p.name}: {total} events ({p.total_commits} commits, {p.total_conversations} conversations), last ingested: {p.last_ingested}")

        else:
            parser.print_help()
    finally:
        db.close()


if __name__ == "__main__":
    main()
