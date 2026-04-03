"""
Move ARCHIVE-verdict files from scripts/dead_code_audit.json into _archive/<same path>.

Skips:
- __init__.py (barrel / package roots; audit often mislabels these)
- Any module still referenced by another live .py (absolute or same-package relative import)
- Paths in NEVER_MOVE
"""
from __future__ import annotations

import json
import os
import shutil
import sys
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


def _iter_py_files() -> list[Path]:
    """Walk repo without descending into heavy dirs (e.g. evolve_venv)."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(Path(dirpath) / fn)
    return out

NEVER_MOVE = frozenset()


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


def main() -> int:
    data = json.loads(AUDIT.read_text(encoding="utf-8", errors="replace"))
    files = data.get("files", [])
    archive_rels = sorted(
        {e["file"].replace("\\", "/") for e in files if e.get("verdict") == "ARCHIVE"}
    )

    cache: dict[Path, str] = {}
    for p in _iter_py_files():
        try:
            cache[p.resolve()] = _load_text(p)
        except OSError:
            continue

    all_blob, dir_blob = _build_index(cache)

    moved = 0
    skipped_missing = 0
    skipped_block = 0
    skipped_exists_dst = 0
    skipped_init = 0
    skipped_ref = 0

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
        if is_referenced(rel, all_blob, dir_blob):
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
        print("moved", rel)
        _maybe_stub_after_moved_init(ROOT / rel)

    print(
        f"done: moved={moved} missing={skipped_missing} blocked={skipped_block} "
        f"dst_exists={skipped_exists_dst} skipped_init={skipped_init} skipped_ref={skipped_ref} "
        f"total_archive={len(archive_rels)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
