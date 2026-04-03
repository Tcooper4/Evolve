"""
Move ARCHIVE-verdict files from scripts/dead_code_audit.json into _archive/<same path>.

Skips:
- __init__.py (barrel / package roots; audit often mislabels these)
- Any module still referenced by another .py (substring scan in pass 1; AST in analysis)
- Paths in NEVER_MOVE

CLI:
  python scripts/run_archive_dead_code.py [--dry-run] [--second-pass] [--report FILE.md]

  --dry-run       Print tables only; no moves. Shows skipped-ref files + importers + live/dead.
  --second-pass   After pass-1 logic, also move ARCHIVE files where ALL AST-detected importers
                  are non-live (reachable-from-pages closure does not include importer).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "dead_code_audit.json"
ARCHIVE_ROOT = ROOT / "_archive"

SKIP_DIRS = frozenset(
    {
        "_archive",
        "evolve_venv",
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "dist",
        "build",
    }
)

NEVER_MOVE = frozenset()

def _iter_py_files() -> list[Path]:
    """Walk repo without descending into heavy dirs (e.g. evolve_venv)."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(Path(dirpath) / fn)
    return out


def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _build_index(cache: dict[Path, str]) -> tuple[str, dict[Path, str]]:
    """Full-repo blob for absolute imports; per-parent-dir blob for relative same-dir."""
    all_blob = "\n".join(cache.values())
    by_parent: dict[Path, list[str]] = {}
    for p, t in cache.items():
        by_parent.setdefault(p.parent.resolve(), []).append(t)
    dir_blob = {d: "\n".join(parts) for d, parts in by_parent.items()}
    return all_blob, dir_blob


def is_referenced(rel: str, all_blob: str, dir_blob: dict[Path, str]) -> bool:
    rel = rel.replace("\\", "/")
    if not rel.endswith(".py"):
        return False
    mod = rel[:-3].replace("/", ".")
    name = Path(rel).stem
    parent = (ROOT / rel).parent.resolve()

    if f"from {mod} import" in all_blob:
        return True
    if f"import {mod} " in all_blob or f"import {mod}\n" in all_blob or f"import {mod}\r" in all_blob:
        return True
    if f"import {mod} as" in all_blob:
        return True
    if f"import {mod}," in all_blob:
        return True

    pb = dir_blob.get(parent)
    if pb:
        if f"from .{name} import" in pb:
            return True
        if f"from . import {name}" in pb:
            return True
    return False


def path_to_dotted_module(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def get_package_for_module(path: Path) -> str:
    """Package used for relative imports (PEP 328)."""
    mod = path_to_dotted_module(path)
    if path.name == "__init__.py":
        return mod
    segs = mod.split(".")
    if len(segs) <= 1:
        return ""
    return ".".join(segs[:-1])


def import_from_module_strings(file_path: Path, node: ast.ImportFrom) -> list[str]:
    """Dotted module names that an ImportFrom may load (for edge resolution)."""
    if node.level == 0:
        if not node.module:
            return []
        cands = [node.module]
        for a in node.names:
            if a.name == "*":
                continue
            cands.append(f"{node.module}.{a.name}")
        return cands
    pkg_parts = get_package_for_module(file_path).split(".")
    if not pkg_parts:
        return []
    up = node.level - 1
    if up > len(pkg_parts):
        return []
    base = pkg_parts[:-up] if up else pkg_parts
    if node.module:
        return [".".join(base + node.module.split("."))]
    out = []
    for a in node.names:
        if a.name == "*":
            continue
        out.append(".".join(base + [a.name]))
    return out


def build_module_to_paths(py_files: list[Path]) -> dict[str, list[Path]]:
    m: dict[str, list[Path]] = defaultdict(list)
    for p in py_files:
        try:
            p.relative_to(ROOT)
        except ValueError:
            continue
        dotted = path_to_dotted_module(p)
        m[dotted].append(p.resolve())
    return dict(m)


def parse_import_edges(
    py_files: list[Path], cache: dict[Path, str]
) -> tuple[dict[Path, set[Path]], dict[Path, set[Path]]]:
    """
    importer -> set of imported file paths (resolved).
    imported_path -> set of importers (reverse).
    """
    m2p = build_module_to_paths(py_files)
    forward: dict[Path, set[Path]] = defaultdict(set)
    reverse: dict[Path, set[Path]] = defaultdict(set)

    def add_edge(frm: Path, to: Path) -> None:
        frm_r = frm.resolve()
        to_r = to.resolve()
        forward[frm_r].add(to_r)
        reverse[to_r].add(frm_r)

    def paths_for_mod(mod: str) -> list[Path]:
        return list(m2p.get(mod, ()))

    for p in py_files:
        pr = p.resolve()
        if pr not in cache:
            continue
        try:
            tree = ast.parse(cache[pr])
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    seen: set[Path] = set()
                    parts = mod.split(".")
                    for k in range(len(parts), 0, -1):
                        sub = ".".join(parts[:k])
                        for tp in paths_for_mod(sub):
                            if tp not in seen:
                                seen.add(tp)
                                add_edge(pr, tp)
            elif isinstance(node, ast.ImportFrom):
                for ms in import_from_module_strings(p, node):
                    if ms in m2p:
                        for tp in m2p[ms]:
                            add_edge(pr, tp)
    return forward, reverse


def compute_live_files(
    forward: dict[Path, set[Path]], cache_paths: set[Path]
) -> set[Path]:
    """BFS from pages/*.py and app.py following import edges."""
    seeds: list[Path] = []
    pages_dir = ROOT / "pages"
    if pages_dir.is_dir():
        seeds.extend(p.resolve() for p in pages_dir.glob("*.py") if p.is_file())
    app_p = ROOT / "app.py"
    if app_p.is_file():
        seeds.append(app_p.resolve())
    live: set[Path] = set()
    q: deque[Path] = deque()
    for s in seeds:
        if s in cache_paths:
            q.append(s)
            live.add(s)
    while q:
        cur = q.popleft()
        for nxt in forward.get(cur, ()):
            if nxt in cache_paths and nxt not in live:
                live.add(nxt)
                q.append(nxt)
    return live


def expand_live_with_package_inits(live: set[Path], cache_paths: set[Path]) -> set[Path]:
    """Mark package __init__.py files on the path to any live module as live (import loader behavior)."""
    out = set(live)
    for p in live:
        cur = p.parent
        while True:
            try:
                cur.relative_to(ROOT)
            except ValueError:
                break
            if cur == ROOT:
                break
            init = (cur / "__init__.py").resolve()
            if init in cache_paths:
                out.add(init)
            cur = cur.parent
    return out


def rel_str(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _maybe_stub_after_moved_init(moved_src: Path) -> None:
    if moved_src.name != "__init__.py":
        return
    parent = moved_src.parent
    if not parent.is_dir():
        return
    others = [p for p in parent.glob("*.py") if p.name != "__init__.py"]
    if not others:
        return
    stub = parent / "__init__.py"
    if stub.exists():
        return
    stub.write_text(
        '"""Package stub; original __init__.py moved to _archive/ (dead code round)."""\n',
        encoding="utf-8",
        errors="replace",
    )
    print("stub", stub.relative_to(ROOT))


def run_dry_run(
    archive_rels: list[str],
    all_blob: str,
    dir_blob: dict[Path, str],
    cache: dict[Path, str],
    py_files: list[Path],
    forward: dict[Path, set[Path]],
    reverse: dict[Path, set[Path]],
    live: set[Path],
    report_path: Path | None,
) -> None:
    live_x = expand_live_with_package_inits(live, set(cache.keys()))
    lines: list[str] = []
    skipped_ref: list[str] = []
    skipped_ref_missing = 0
    for rel in archive_rels:
        if rel in NEVER_MOVE or rel.endswith("__init__.py"):
            continue
        src = ROOT / rel
        if not src.is_file():
            if is_referenced(rel, all_blob, dir_blob):
                skipped_ref_missing += 1
            continue
        if not is_referenced(rel, all_blob, dir_blob):
            continue
        skipped_ref.append(rel)

    header = (
        "| File skipped (still referenced) | Imported by | Importer live/dead |\n"
        "|---|---|---|\n"
    )
    lines.append("# Dry-run: ARCHIVE files skipped - reference scan\n\n")
    lines.append(
        f"On-disk referenced (table below): **{len(skipped_ref)}**. "
        f"Referenced but already absent (e.g. prior archive): **{skipped_ref_missing}**.\n\n"
    )
    lines.append(header)

    for rel in sorted(skipped_ref):
        target = (ROOT / rel).resolve()
        importers = sorted(reverse.get(target, ()), key=lambda p: rel_str(p))
        if not importers:
            row = f"| `{rel}` | *(substring ref only — no AST edge)* | — |\n"
            lines.append(row)
            continue
        for imp in importers:
            status = "live" if imp in live_x else "dead"
            lines.append(f"| `{rel}` | `{rel_str(imp)}` | **{status}** |\n")

    text = "".join(lines)
    print(text)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8", errors="replace")
        print(f"Wrote {report_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive dead code from dead_code_audit.json")
    ap.add_argument("--dry-run", action="store_true", help="Print skipped-ref table; no moves")
    ap.add_argument(
        "--second-pass",
        action="store_true",
        help="Move ARCHIVE .py files where all AST importers are non-live",
    )
    ap.add_argument(
        "--report",
        type=str,
        default="",
        help="Write dry-run markdown table to this path (e.g. scripts/archive_dry_run.md)",
    )
    args = ap.parse_args()

    data = json.loads(AUDIT.read_text(encoding="utf-8", errors="replace"))
    files = data.get("files", [])
    archive_rels = sorted(
        {e["file"].replace("\\", "/") for e in files if e.get("verdict") == "ARCHIVE"}
    )

    cache: dict[Path, str] = {}
    py_files = _iter_py_files()
    for p in py_files:
        try:
            cache[p.resolve()] = _load_text(p)
        except OSError:
            continue

    all_blob, dir_blob = _build_index(cache)
    forward, reverse = parse_import_edges(py_files, cache)
    live = compute_live_files(forward, set(cache.keys()))
    live_x = expand_live_with_package_inits(live, set(cache.keys()))

    report_path = Path(args.report) if args.report else None
    if args.dry_run:
        run_dry_run(
            archive_rels,
            all_blob,
            dir_blob,
            cache,
            py_files,
            forward,
            reverse,
            live,
            report_path,
        )
        if not args.second_pass:
            return 0

    moved = 0
    skipped_missing = 0
    skipped_block = 0
    skipped_exists_dst = 0
    skipped_init = 0
    skipped_ref = 0
    moved_second = 0

    for rel in archive_rels:
        if rel in NEVER_MOVE:
            skipped_block += 1
            continue
        if rel.endswith("__init__.py"):
            skipped_init += 1
            continue
        src = ROOT / rel
        if not src.is_file():
            skipped_missing += 1
            print("missing", rel)
            continue

        ref = is_referenced(rel, all_blob, dir_blob)
        if args.second_pass:
            if not ref:
                continue
            tgt = src.resolve()
            importers = reverse.get(tgt, set())
            if importers:
                if any(imp in live_x for imp in importers):
                    skipped_ref += 1
                    continue
            else:
                skipped_ref += 1
                continue
        elif ref:
            skipped_ref += 1
            continue

        dst = ARCHIVE_ROOT / rel
        if dst.exists():
            skipped_exists_dst += 1
            print("skip exists", rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1
        if args.second_pass:
            moved_second += 1
            print("moved (second pass)", rel)
        else:
            print("moved", rel)
        _maybe_stub_after_moved_init(ROOT / rel)

    print(
        f"done: moved={moved} moved_second_pass={moved_second} missing={skipped_missing} "
        f"blocked={skipped_block} dst_exists={skipped_exists_dst} "
        f"skipped_init={skipped_init} skipped_ref={skipped_ref} total_archive={len(archive_rels)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
