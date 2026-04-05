import ast
import os

python = r".\evolve_venv\Scripts\python.exe"
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Check each migrated file
migrated = [
    "agents/llm/agent.py",
    "trading/services/agent_tools.py",
    "trading/validation/walk_forward_utils.py",
]
for path in migrated:
    text = open(path, encoding="utf-8", errors="replace").read()
    has_singleton = "get_router_singleton" in text
    raw_count = sum(
        1
        for line in text.splitlines()
        if "ForecastRouter()" in line
        and "return ForecastRouter()" not in line
        and not line.strip().startswith("#")
    )
    if has_singleton:
        ok(f"{path}: uses get_router_singleton")
    else:
        fail(f"{path}: missing get_router_singleton")
    if raw_count == 0:
        ok(f"{path}: 0 raw ForecastRouter() calls")
    else:
        fail(
            f"{path}: still has {raw_count} raw ForecastRouter() calls"
        )

# Syntax check
print()
for f in migrated:
    try:
        ast.parse(open(f, encoding="utf-8", errors="replace").read())
        ok(f"Syntax valid: {f}")
    except SyntaxError as e:
        fail(f"Syntax error in {f}: {e}")

# Full remaining scan
print("\n--- All remaining ForecastRouter() sites ---")
remaining = []
for d in ["components", "pages", "agents", "trading"]:
    for root, dirs, files in os.walk(d):
        dirs[:] = [
            x
            for x in dirs
            if x not in ("__pycache__", "evolve_venv", ".git")
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            lines = open(
                fpath, encoding="utf-8", errors="replace"
            ).readlines()
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if "ForecastRouter()" not in s:
                    continue
                if "return ForecastRouter()" in s:
                    continue
                if s.startswith("#"):
                    continue
                remaining.append(f"  {fpath}:{i}  {s}")
for r in remaining:
    print(r)
print(f"\n  Total remaining: {len(remaining)}")

# Summary
print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    for f in FAIL:
        print(f"  FAIL  {f}")
