#!/usr/bin/env python3
"""Reachability analysis for the Evolve live tree.

Classifies every application .py file as REACHABLE (transitively imported
from an entry point) or a candidate for archive, with a safety layer: any
file whose module name appears as a *string* anywhere in the live tree
(dynamic import, importlib, config reference, docs) is flagged
DYNAMIC-REF and never recommended for removal.

Entry points: app.py, pages/*.py, trading/services/mcp_server.py,
scripts referenced from docs, and conftest/test files count as
"referenced-by-tests" (kept, flagged separately).
"""

import ast
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(".").resolve()
SKIP_DIRS = {"_archive", "__pycache__", ".git", ".venv", "node_modules"}

def all_py():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(".py"):
                out.append(Path(dirpath, f).relative_to(ROOT))
    return sorted(out)

def module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

FILES = all_py()
MODLOOKUP = {}
for p in FILES:
    MODLOOKUP[module_name(p)] = p

def imports_of(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    mods = set()
    pkg_parts = list(path.parent.parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            if mod:
                mods.add(mod)
                # also add mod.attr for each name (from pkg import submodule)
                for a in node.names:
                    mods.add(f"{mod}.{a.name}")
    return mods

def resolve(mod: str):
    """Map an import string to a repo file, walking up dotted parents."""
    parts = mod.split(".")
    while parts:
        cand = ".".join(parts)
        if cand in MODLOOKUP:
            return MODLOOKUP[cand]
        # package __init__
        parts = parts[:-1]
    return None

# Build graph
GRAPH = defaultdict(set)
for p in FILES:
    for m in imports_of(p):
        t = resolve(m)
        if t is not None and t != p:
            GRAPH[p].add(t)
        # package import reaches package __init__
        pkg = resolve(m.split(".")[0]) if "." in m else None
    # importing a module inside a package executes the package __init__s
    parts = p.parts
    for i in range(1, len(parts)):
        init = Path(*parts[:i], "__init__.py")
        if (ROOT / init).exists():
            GRAPH[p].add(init)

ENTRY = [Path("app.py")]
ENTRY += sorted(Path("pages").glob("*.py"))
ENTRY += [Path("trading/services/mcp_server.py")]
ENTRY = [e for e in ENTRY if (ROOT / e).exists()]

reach = set()
stack = list(ENTRY)
while stack:
    cur = stack.pop()
    if cur in reach:
        continue
    reach.add(cur)
    stack.extend(GRAPH.get(cur, ()))

# String/dynamic reference scan across the whole live tree (py + md + yaml + toml + json)
TEXT = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for f in filenames:
        if f.endswith((".py", ".md", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".sh")):
            p = Path(dirpath, f).relative_to(ROOT)
            try:
                TEXT.append((p, (ROOT / p).read_text(encoding="utf-8", errors="replace")))
            except Exception:
                pass

def string_refs(path: Path):
    stem = path.stem
    if stem == "__init__":
        stem = path.parent.name
    if len(stem) < 4:  # too generic to search safely
        return ["<name-too-generic>"]
    pat = re.compile(rf"\b{re.escape(stem)}\b")
    hits = []
    for p, txt in TEXT:
        if p == path:
            continue
        if pat.search(txt):
            hits.append(str(p))
    return hits

def is_test(p: Path):
    return p.parts[0] == "tests" or p.name.startswith("test_")

APP_FILES = [p for p in FILES if p.parts[0] not in ("tests", "scripts", "docs")]

unreachable = [p for p in APP_FILES if p not in reach and not is_test(p)]
report = {"REACHABLE": [], "TEST_ONLY": [], "DYNAMIC_REF": [], "ORPHAN": []}
for p in sorted(unreachable):
    refs = string_refs(p)
    test_refs = [r for r in refs if r.startswith("tests/") or Path(r).name.startswith("test_")]
    nontest_refs = [r for r in refs if r not in test_refs]
    if nontest_refs:
        report["DYNAMIC_REF"].append((p, nontest_refs[:4]))
    elif test_refs:
        report["TEST_ONLY"].append((p, test_refs[:3]))
    else:
        report["ORPHAN"].append((p, []))

print(f"entry points: {len(ENTRY)} | app files: {len(APP_FILES)} | reachable: {len([p for p in APP_FILES if p in reach])}")
print(f"\nORPHAN (zero imports, zero string refs anywhere) — {len(report['ORPHAN'])}:")
for p, _ in report["ORPHAN"]:
    print(f"  {p}")
print(f"\nTEST_ONLY (only tests reference) — {len(report['TEST_ONLY'])}:")
for p, r in report["TEST_ONLY"]:
    print(f"  {p}  <- {r}")
print(f"\nDYNAMIC_REF (string-referenced by live code/docs — NEVER auto-remove) — {len(report['DYNAMIC_REF'])}:")
for p, r in report["DYNAMIC_REF"][:40]:
    print(f"  {p}  <- {r}")
