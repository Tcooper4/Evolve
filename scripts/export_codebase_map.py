"""
Export a JSON map of Python files for graph visualization.
Run from repo root: .\\evolve_venv\\Scripts\\python.exe scripts/export_codebase_map.py
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

EXCLUDE_DIRS = {
    "evolve_venv",
    "venv",
    "__pycache__",
    ".git",
    ".cache",
    "_archive",
    "node_modules",
}

LAYER_MAP = (
    ("pages", "ui"),
    ("components", "ui"),
    ("trading/models", "model"),
    ("trading/analysis", "service"),
    ("trading/data", "data"),
    ("trading/services", "service"),
    ("trading/validation", "service"),
    ("agents", "agent"),
    ("trading/agents", "agent"),
    ("utils", "util"),
    ("config", "config"),
    ("tests", "test"),
    ("scripts", "util"),
)


def get_layer(filepath: str) -> str:
    for prefix, layer in LAYER_MAP:
        if filepath.startswith(prefix + "/") or filepath == prefix:
            return layer
    return "other"


def get_imports(filepath: Path) -> list:
    """Extract project-internal imports (best-effort)."""
    skip_prefixes = (
        "streamlit",
        "pandas",
        "numpy",
        "torch",
        "sklearn",
        "plotly",
        "yfinance",
        "anthropic",
        "openai",
        "scipy",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "joblib",
        "requests",
        "httpx",
        "pydantic",
    )
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if not any(mod.startswith(p) for p in skip_prefixes):
                    imports.append(mod)
        return sorted(set(imports))
    except Exception:
        return []


def get_line_count(filepath: Path) -> int:
    try:
        return len(filepath.read_text(encoding="utf-8", errors="replace").splitlines())
    except Exception:
        return 0


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    nodes: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.endswith("venv")
        ]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            py_file = Path(dirpath) / fname
            rel = str(py_file.relative_to(root)).replace("\\", "/")
            nodes.append({
                "id": rel.replace("/", "_").replace(".py", ""),
                "label": py_file.stem,
                "file": rel,
                "layer": get_layer(rel),
                "lines": get_line_count(py_file),
                "imports_from": get_imports(py_file),
            })

    nodes.sort(key=lambda x: x["file"])
    output = {"nodes": nodes, "total": len(nodes)}
    out_path = root / "scripts" / "codebase_map.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Exported {len(nodes)} nodes to {out_path.relative_to(root)}")


if __name__ == "__main__":
    main()
