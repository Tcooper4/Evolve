import io
import os
import re

EXCLUDE = {
    "evolve_venv",
    "_archive",
    "__pycache__",
    ".git",
    "scripts",
    "node_modules",
    ".venv",
    "venv",
}


def fix_file(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as f:
        original = f.read()
    updated = re.sub(
        r"use_container_width\s*=\s*True",
        "width='stretch'",
        original,
    )
    updated = re.sub(
        r"use_container_width\s*=\s*False",
        "width='content'",
        updated,
    )
    if updated == original:
        return 0
    buf = io.BytesIO()
    buf.write(updated.encode("utf-8"))
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return updated.count("width=")


total_files = 0
total_replacements = 0
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        n = fix_file(fpath)
        if n > 0:
            total_files += 1
            total_replacements += n
            print(f"Fixed {fpath}: {n} replacements")

print(
    f"\nTotal: {total_replacements} replacements in "
    f"{total_files} files"
)
