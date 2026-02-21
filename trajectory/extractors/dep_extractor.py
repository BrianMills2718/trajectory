"""Deterministic dependency extractor — cross-project deps from pyproject.toml + pip install -e.

No LLM calls. Scans pyproject.toml dependencies + pip install commands in events
to find cross-project links.
Populates the project_dependencies table.
"""

import json
import logging
import re
from pathlib import Path

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)


class DepResult:
    """Summary of dependency extraction."""

    def __init__(self) -> None:
        self.deps_found = 0
        self.cross_project_links = 0

    def __repr__(self) -> str:
        return (
            f"DepResult({self.deps_found} dependencies found, "
            f"{self.cross_project_links} cross-project links)"
        )


def _parse_pyproject_all_deps(project_path: Path) -> list[str]:
    """Parse ALL dependency names from pyproject.toml. Returns lowercase package names."""
    pyproject = project_path / "pyproject.toml"
    if not pyproject.exists():
        return []

    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.warning("Could not parse %s", pyproject)
        return []

    deps: list[str] = []

    # PEP 621 format
    for dep in data.get("project", {}).get("dependencies", []):
        name = dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split("!")[0].split("~")[0].strip()
        if name:
            deps.append(name.lower())

    # Optional
    for group_deps in data.get("project", {}).get("optional-dependencies", {}).values():
        for dep in group_deps:
            name = dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split("!")[0].split("~")[0].strip()
            if name:
                deps.append(name.lower())

    # Poetry
    for name in data.get("tool", {}).get("poetry", {}).get("dependencies", {}):
        if name.lower() != "python":
            deps.append(name.lower())
    for name in data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {}):
        deps.append(name.lower())

    return list(set(deps))


# Regex for `pip install -e /path/to/project` or `pip install -e ~/projects/X`
PIP_EDITABLE_RE = re.compile(
    r"pip\s+install\s+(?:-[a-zA-Z]*e\s+|--editable\s+)"
    r"([~/][\w/.+-]+)",
)


def _find_editable_installs(db: TrajectoryDB, project_id: int) -> list[str]:
    """Scan event body/raw_data for `pip install -e` commands pointing to other projects."""
    paths: list[str] = []

    # Search in commit messages and conversation bodies
    rows = db.conn.execute(
        "SELECT body, raw_data FROM events WHERE project_id = ? AND (body IS NOT NULL OR raw_data IS NOT NULL)",
        (project_id,),
    ).fetchall()

    for row in rows:
        for field in [row["body"], row["raw_data"]]:
            if not field:
                continue
            for match in PIP_EDITABLE_RE.finditer(str(field)):
                path_str = match.group(1)
                # Expand ~
                if path_str.startswith("~"):
                    path_str = str(Path.home()) + path_str[1:]
                paths.append(path_str)

    return list(set(paths))


def extract_dependencies(db: TrajectoryDB) -> DepResult:
    """Extract cross-project dependencies for all projects.

    Sources:
    1. pyproject.toml deps that match other tracked project names
    2. `pip install -e /path/to/project` commands in events

    Idempotent: clears all deps and rewrites.
    """
    result = DepResult()

    projects = db.list_projects()
    if not projects:
        return result

    # Build lookup: project_name → project_id (lowercase)
    name_to_id: dict[str, int] = {}
    path_to_id: dict[str, int] = {}
    for p in projects:
        name_to_id[p.name.lower()] = p.id
        # Also match by directory name from path
        path_to_id[p.path] = p.id
        path_to_id[Path(p.path).name.lower()] = p.id

    # Common package name → project name aliases
    # (when the pip package name differs from the directory name)
    PACKAGE_ALIASES: dict[str, str] = {
        "llm-client": "llm_client",
        "llm_client": "llm_client",
        "fastmcp": "mcp-servers",
    }

    # Clear existing deps
    db.conn.execute("DELETE FROM project_dependencies")
    db.conn.commit()

    for project in projects:
        project_path = Path(project.path)

        # --- Source 1: pyproject.toml dependencies ---
        pyproject_deps = _parse_pyproject_all_deps(project_path)
        for dep_name in pyproject_deps:
            # Check if dep matches another tracked project
            target_name = PACKAGE_ALIASES.get(dep_name, dep_name)
            target_id = name_to_id.get(target_name.lower().replace("-", "_"))
            if not target_id:
                target_id = name_to_id.get(target_name.lower().replace("_", "-"))
            if not target_id:
                target_id = name_to_id.get(target_name.lower())

            if target_id and target_id != project.id:
                db.conn.execute(
                    "INSERT OR IGNORE INTO project_dependencies "
                    "(project_id, depends_on_project_id, depends_on_name, dep_type, evidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (project.id, target_id, dep_name, "pyproject", f"pyproject.toml dependency: {dep_name}"),
                )
                result.cross_project_links += 1
                result.deps_found += 1

        # --- Source 2: pip install -e commands ---
        editable_paths = _find_editable_installs(db, project.id)
        for path_str in editable_paths:
            # Try to resolve to a tracked project
            resolved = Path(path_str).resolve() if Path(path_str).exists() else None
            target_id = None
            if resolved:
                target_id = path_to_id.get(str(resolved))
            if not target_id:
                # Try matching by directory name
                dir_name = Path(path_str).name.lower()
                target_id = path_to_id.get(dir_name)

            if target_id and target_id != project.id:
                dep_name = Path(path_str).name
                db.conn.execute(
                    "INSERT OR IGNORE INTO project_dependencies "
                    "(project_id, depends_on_project_id, depends_on_name, dep_type, evidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (project.id, target_id, dep_name, "editable_install",
                     f"pip install -e {path_str}"),
                )
                result.cross_project_links += 1
                result.deps_found += 1

        # --- Source 3: pyproject.toml non-cross-project deps (for reference) ---
        for dep_name in pyproject_deps:
            target_name = PACKAGE_ALIASES.get(dep_name, dep_name)
            if not name_to_id.get(target_name.lower().replace("-", "_")) and \
               not name_to_id.get(target_name.lower().replace("_", "-")) and \
               not name_to_id.get(target_name.lower()):
                db.conn.execute(
                    "INSERT OR IGNORE INTO project_dependencies "
                    "(project_id, depends_on_project_id, depends_on_name, dep_type, evidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (project.id, None, dep_name, "external", f"pyproject.toml: {dep_name}"),
                )
                result.deps_found += 1

    db.conn.commit()
    logger.info("Dependency extraction: %s", result)
    return result
