# scripts/verify_s22.py
import ast, os, subprocess, sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 22 Verification ===\n")

PASS, FAIL = [], []
def ok(msg):   PASS.append(msg); print(f"OK    {msg}")
def fail(msg): FAIL.append(msg); print(f"FAIL  {msg}")

# 1. Check each migrated file has singleton and no raw calls
migrated = [
    "components/analyze_ai_score.py",
    "components/analyze_forecast.py",
    "components/tabs/tab_ai_model_selection.py",
    "components/tabs/tab_backtester.py",
    "agents/briefing/morning_briefing.py",
]
for path in migrated:
    text = open(path, encoding="utf-8", errors="replace").read()
    has_singleton = "get_router_singleton" in text
    raw_count = sum(
        1 for line in text.splitlines()
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
        fail(f"{path}: still has {raw_count} raw ForecastRouter() calls")

# 2. Syntax check
print()
for f in migrated:
    try:
        ast.parse(open(f, encoding="utf-8", errors="replace").read())
        ok(f"Syntax valid: {f}")
    except SyntaxError as e:
        fail(f"Syntax error in {f}: {e}")

# 3. Remaining sites summary (informational)
print("\n--- Still unmigrated ForecastRouter() sites ---")
remaining = []
for d in ["components", "pages", "agents", "trading"]:
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x not in
                   ("__pycache__", "evolve_venv", ".git")]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            lines = open(fpath, encoding="utf-8",
                         errors="replace").readlines()
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
print(f"  Total remaining: {len(remaining)}")

# 4. Smoke test
print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True, text=True
)
print((result.stdout + result.stderr)[-2000:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

# Summary
print(f"\n=== Summary: {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    for f in FAIL:
        print(f"  FAIL  {f}")
    sys.exit(1)
else:
    print("All checks passed. Ready for Session 23.")
    sys.exit(0)