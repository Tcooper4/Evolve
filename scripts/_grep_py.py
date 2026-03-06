"""
Tiny grep-equivalent for Windows environments without grep/rg.

Usage:
  py -3.10 scripts/_grep_py.py "pattern1|pattern2" [root_dir]

Prints matches in: path:line_number:line
"""

from __future__ import annotations

import os
import re
import sys
from typing import Iterable


def iter_py_files(root: str) -> Iterable[str]:
    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        ".cache",
    }
    for cur_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(cur_root, fn)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: py -3.10 scripts/_grep_py.py \"pattern1|pattern2\" [root_dir]")
        return 2

    pattern = re.compile(sys.argv[1])
    root = sys.argv[2] if len(sys.argv) >= 3 else "."

    for path in iter_py_files(root):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pattern.search(line):
                        rel = os.path.relpath(path, root).replace("\\", "/")
                        print(f"{rel}:{i}:{line.rstrip()}")
        except OSError:
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

