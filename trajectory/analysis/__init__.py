"""Analysis modules — LLM-based event classification and concept tracking."""

import sys
from pathlib import Path

# Ensure llm_client is importable
_llm_client_path = str(Path.home() / "projects" / "llm_client")
if _llm_client_path not in sys.path:
    sys.path.insert(0, _llm_client_path)
