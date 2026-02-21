"""Deterministic technology extractor — languages, frameworks, tools from file extensions + pyproject.toml.

No LLM calls. Scans files_changed from existing events + parses pyproject.toml on disk.
Populates the project_technologies table.
"""

import json
import logging
from pathlib import Path

from trajectory.db import TrajectoryDB

logger = logging.getLogger(__name__)

# File extension → (technology_name, category)
# category: "language", "config", "data", "docs", "build"
EXTENSION_MAP: dict[str, tuple[str, str]] = {
    ".py": ("Python", "language"),
    ".js": ("JavaScript", "language"),
    ".ts": ("TypeScript", "language"),
    ".tsx": ("TypeScript", "language"),
    ".jsx": ("JavaScript", "language"),
    ".rs": ("Rust", "language"),
    ".go": ("Go", "language"),
    ".java": ("Java", "language"),
    ".kt": ("Kotlin", "language"),
    ".rb": ("Ruby", "language"),
    ".php": ("PHP", "language"),
    ".c": ("C", "language"),
    ".h": ("C", "language"),
    ".cpp": ("C++", "language"),
    ".hpp": ("C++", "language"),
    ".cs": ("C#", "language"),
    ".swift": ("Swift", "language"),
    ".sh": ("Shell", "language"),
    ".bash": ("Shell", "language"),
    ".zsh": ("Shell", "language"),
    ".sql": ("SQL", "language"),
    ".lua": ("Lua", "language"),
    ".r": ("R", "language"),
    ".R": ("R", "language"),
    ".scala": ("Scala", "language"),
    ".ex": ("Elixir", "language"),
    ".exs": ("Elixir", "language"),
    ".zig": ("Zig", "language"),
    ".html": ("HTML", "language"),
    ".css": ("CSS", "language"),
    ".scss": ("SCSS", "language"),
    ".yaml": ("YAML", "config"),
    ".yml": ("YAML", "config"),
    ".toml": ("TOML", "config"),
    ".ini": ("INI", "config"),
    ".cfg": ("INI", "config"),
    ".json": ("JSON", "data"),
    ".jsonl": ("JSONL", "data"),
    ".csv": ("CSV", "data"),
    ".xml": ("XML", "data"),
    ".md": ("Markdown", "docs"),
    ".rst": ("reStructuredText", "docs"),
    ".txt": ("Text", "docs"),
    ".ipynb": ("Jupyter", "language"),
    ".proto": ("Protobuf", "data"),
    ".graphql": ("GraphQL", "language"),
    ".gql": ("GraphQL", "language"),
    ".dockerfile": ("Docker", "build"),
    ".tf": ("Terraform", "config"),
    ".hcl": ("HCL", "config"),
}

# Exact filename matches (case-insensitive)
FILENAME_MAP: dict[str, tuple[str, str]] = {
    "dockerfile": ("Docker", "build"),
    "docker-compose.yml": ("Docker Compose", "build"),
    "docker-compose.yaml": ("Docker Compose", "build"),
    "compose.yml": ("Docker Compose", "build"),
    "compose.yaml": ("Docker Compose", "build"),
    "makefile": ("Make", "build"),
    "cmakelists.txt": ("CMake", "build"),
    "pyproject.toml": ("pyproject.toml", "build"),
    "setup.py": ("setuptools", "build"),
    "setup.cfg": ("setuptools", "build"),
    "package.json": ("npm", "build"),
    "cargo.toml": ("Cargo", "build"),
    "go.mod": ("Go Modules", "build"),
    "gemfile": ("Bundler", "build"),
    "requirements.txt": ("pip", "build"),
    ".gitignore": ("Git", "build"),
    ".env": ("dotenv", "config"),
    "justfile": ("Just", "build"),
}

# pyproject.toml dependency → (technology_name, category)
# Maps common package names to frameworks/tools
DEPENDENCY_MAP: dict[str, tuple[str, str]] = {
    "fastapi": ("FastAPI", "framework"),
    "flask": ("Flask", "framework"),
    "django": ("Django", "framework"),
    "starlette": ("Starlette", "framework"),
    "pydantic": ("Pydantic", "framework"),
    "sqlalchemy": ("SQLAlchemy", "framework"),
    "pytest": ("pytest", "tool"),
    "mypy": ("mypy", "tool"),
    "ruff": ("ruff", "tool"),
    "black": ("Black", "tool"),
    "isort": ("isort", "tool"),
    "flake8": ("flake8", "tool"),
    "pylint": ("pylint", "tool"),
    "litellm": ("LiteLLM", "framework"),
    "openai": ("OpenAI SDK", "framework"),
    "anthropic": ("Anthropic SDK", "framework"),
    "instructor": ("Instructor", "framework"),
    "langchain": ("LangChain", "framework"),
    "llama-index": ("LlamaIndex", "framework"),
    "transformers": ("Transformers", "framework"),
    "torch": ("PyTorch", "framework"),
    "tensorflow": ("TensorFlow", "framework"),
    "numpy": ("NumPy", "framework"),
    "pandas": ("pandas", "framework"),
    "scipy": ("SciPy", "framework"),
    "scikit-learn": ("scikit-learn", "framework"),
    "matplotlib": ("matplotlib", "framework"),
    "plotly": ("Plotly", "framework"),
    "pillow": ("Pillow", "framework"),
    "requests": ("requests", "framework"),
    "httpx": ("httpx", "framework"),
    "aiohttp": ("aiohttp", "framework"),
    "uvicorn": ("uvicorn", "tool"),
    "gunicorn": ("gunicorn", "tool"),
    "celery": ("Celery", "framework"),
    "redis": ("Redis (Python)", "framework"),
    "psycopg2": ("PostgreSQL (Python)", "framework"),
    "asyncpg": ("PostgreSQL (Python)", "framework"),
    "pymongo": ("MongoDB (Python)", "framework"),
    "mcp": ("MCP SDK", "framework"),
    "fastmcp": ("FastMCP", "framework"),
    "pydriller": ("PyDriller", "framework"),
    "jinja2": ("Jinja2", "framework"),
    "pyyaml": ("PyYAML", "framework"),
    "click": ("Click", "framework"),
    "typer": ("Typer", "framework"),
    "rich": ("Rich", "framework"),
    "streamlit": ("Streamlit", "framework"),
    "gradio": ("Gradio", "framework"),
    "docker": ("Docker SDK", "framework"),
    "boto3": ("AWS SDK", "framework"),
    "google-cloud-storage": ("GCS SDK", "framework"),
    "networkx": ("NetworkX", "framework"),
    "faiss-cpu": ("FAISS", "framework"),
    "faiss-gpu": ("FAISS", "framework"),
    "chromadb": ("ChromaDB", "framework"),
    "weaviate-client": ("Weaviate", "framework"),
    "react": ("React", "framework"),
    "next": ("Next.js", "framework"),
    "vue": ("Vue.js", "framework"),
    "svelte": ("Svelte", "framework"),
    "tailwindcss": ("Tailwind CSS", "framework"),
    "express": ("Express", "framework"),
    "d3": ("D3.js", "framework"),
    "remotion": ("Remotion", "framework"),
}


def _parse_pyproject_deps(project_path: Path) -> list[str]:
    """Parse dependency names from pyproject.toml. Returns lowercase package names."""
    pyproject = project_path / "pyproject.toml"
    if not pyproject.exists():
        return []

    try:
        # Use tomllib (Python 3.11+) or tomli
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

    # PEP 621 format: [project.dependencies]
    project_deps = data.get("project", {}).get("dependencies", [])
    for dep in project_deps:
        # "pydantic>=2.0" → "pydantic"
        name = dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split("!")[0].split("~")[0].strip()
        if name:
            deps.append(name.lower())

    # Optional dependencies
    optional = data.get("project", {}).get("optional-dependencies", {})
    for group_deps in optional.values():
        for dep in group_deps:
            name = dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split("!")[0].split("~")[0].strip()
            if name:
                deps.append(name.lower())

    # Poetry format: [tool.poetry.dependencies]
    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for name in poetry_deps:
        if name.lower() != "python":
            deps.append(name.lower())

    # Poetry dev dependencies
    poetry_dev = data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {})
    for name in poetry_dev:
        deps.append(name.lower())

    return deps


def _parse_package_json_deps(project_path: Path) -> list[str]:
    """Parse dependency names from package.json. Returns lowercase package names."""
    pkg_json = project_path / "package.json"
    if not pkg_json.exists():
        return []

    try:
        with open(pkg_json) as f:
            data = json.load(f)
    except Exception:
        logger.warning("Could not parse %s", pkg_json)
        return []

    deps: list[str] = []
    for section in ("dependencies", "devDependencies", "peerDependencies"):
        for name in data.get(section, {}):
            deps.append(name.lower())

    return deps


class TechResult:
    """Summary of technology extraction for a project."""

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        self.languages: dict[str, int] = {}  # tech_name → file_count
        self.frameworks: list[str] = []
        self.tools: list[str] = []
        self.build: list[str] = []
        self.config: list[str] = []
        self.total: int = 0

    def __repr__(self) -> str:
        langs = ", ".join(f"{k}({v})" for k, v in sorted(self.languages.items(), key=lambda x: -x[1])[:5])
        return (
            f"TechResult({self.project_name}: {self.total} technologies — "
            f"languages=[{langs}], "
            f"frameworks={len(self.frameworks)}, tools={len(self.tools)})"
        )


def extract_technologies(
    db: TrajectoryDB,
    project_id: int,
    project_path: Path | None = None,
) -> TechResult:
    """Extract technologies for a project from events + project files.

    1. Scans files_changed across all events → file extensions → languages
    2. Parses pyproject.toml/package.json → frameworks/tools
    3. Detects build system files from events

    Idempotent: clears existing rows for the project first.
    """
    project = db.get_project(project_id)
    if not project:
        raise ValueError(f"Project ID {project_id} not found")

    result = TechResult(project.name)

    if project_path is None:
        project_path = Path(project.path)

    # --- Phase 1: Scan file extensions from events ---
    rows = db.conn.execute(
        "SELECT files_changed, timestamp FROM events WHERE project_id = ? AND files_changed IS NOT NULL",
        (project_id,),
    ).fetchall()

    # Track per-technology: file_count, first_seen, last_seen
    tech_data: dict[str, dict[str, object]] = {}  # tech_name → {category, files: set, first_seen, last_seen}

    for row in rows:
        try:
            files = json.loads(row["files_changed"])
        except (json.JSONDecodeError, TypeError):
            continue

        ts = row["timestamp"]

        for filepath in files:
            if not filepath:
                continue

            p = Path(filepath)
            ext = p.suffix.lower()
            fname = p.name.lower()

            # Check exact filename first
            match = FILENAME_MAP.get(fname)
            if match:
                tech_name, category = match
                if tech_name not in tech_data:
                    tech_data[tech_name] = {"category": category, "files": set(), "first_seen": ts, "last_seen": ts}
                tech_data[tech_name]["files"].add(filepath)  # type: ignore[union-attr]
                if ts and (not tech_data[tech_name]["first_seen"] or ts < tech_data[tech_name]["first_seen"]):  # type: ignore[operator]
                    tech_data[tech_name]["first_seen"] = ts
                if ts and (not tech_data[tech_name]["last_seen"] or ts > tech_data[tech_name]["last_seen"]):  # type: ignore[operator]
                    tech_data[tech_name]["last_seen"] = ts

            # Check extension
            if ext and ext in EXTENSION_MAP:
                tech_name, category = EXTENSION_MAP[ext]
                if tech_name not in tech_data:
                    tech_data[tech_name] = {"category": category, "files": set(), "first_seen": ts, "last_seen": ts}
                tech_data[tech_name]["files"].add(filepath)  # type: ignore[union-attr]
                if ts and (not tech_data[tech_name]["first_seen"] or ts < tech_data[tech_name]["first_seen"]):  # type: ignore[operator]
                    tech_data[tech_name]["first_seen"] = ts
                if ts and (not tech_data[tech_name]["last_seen"] or ts > tech_data[tech_name]["last_seen"]):  # type: ignore[operator]
                    tech_data[tech_name]["last_seen"] = ts

    # --- Phase 2: Parse dependency files ---
    pyproject_deps = _parse_pyproject_deps(project_path)
    for dep_name in pyproject_deps:
        if dep_name in DEPENDENCY_MAP:
            tech_name, category = DEPENDENCY_MAP[dep_name]
            if tech_name not in tech_data:
                tech_data[tech_name] = {"category": category, "files": set(), "first_seen": None, "last_seen": None}

    pkg_deps = _parse_package_json_deps(project_path)
    for dep_name in pkg_deps:
        if dep_name in DEPENDENCY_MAP:
            tech_name, category = DEPENDENCY_MAP[dep_name]
            if tech_name not in tech_data:
                tech_data[tech_name] = {"category": category, "files": set(), "first_seen": None, "last_seen": None}

    # --- Phase 3: Write to DB ---
    db.clear_technologies(project_id)

    for tech_name, data in tech_data.items():
        file_count = len(data["files"])  # type: ignore[arg-type]
        category = str(data["category"])
        first_seen = str(data["first_seen"]) if data["first_seen"] else None
        last_seen = str(data["last_seen"]) if data["last_seen"] else None

        db.upsert_technology(
            project_id=project_id,
            technology=tech_name,
            category=category,
            file_count=file_count,
            first_seen=first_seen,
            last_seen=last_seen,
        )

        # Build result summary
        if category == "language":
            result.languages[tech_name] = file_count
        elif category == "framework":
            result.frameworks.append(tech_name)
        elif category == "tool":
            result.tools.append(tech_name)
        elif category == "build":
            result.build.append(tech_name)
        elif category == "config":
            result.config.append(tech_name)

    db.conn.commit()
    result.total = len(tech_data)

    logger.info("Extracted %d technologies for %s", result.total, project.name)
    return result


def extract_all_technologies(db: TrajectoryDB) -> list[TechResult]:
    """Extract technologies for all projects."""
    projects = db.list_projects()
    results: list[TechResult] = []

    for project in projects:
        project_path = Path(project.path)
        try:
            result = extract_technologies(db, project.id, project_path)
            results.append(result)
        except Exception:
            logger.exception("Failed to extract technologies for %s", project.name)

    return results
