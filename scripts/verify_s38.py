import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 38 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Overextension penalty
t = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error ai_score.py: {e}")

if "Overextended" in t:
    ok("Overextension penalty present")
else:
    fail("Overextension penalty missing")

if "analyst_signals" in t:
    ok("analyst_signals in SIGNAL_SOURCES")
else:
    fail("analyst_signals missing from SIGNAL_SOURCES")

if "_fetch_analyst_safe" in t:
    ok("Analyst parallel worker present")
else:
    fail("Analyst parallel worker missing")

# Analyst signals module
t2 = ""
try:
    t2 = open(
        "trading/data/analyst_signals.py",
        encoding="utf-8",
        errors="replace",
    ).read()
    ast.parse(t2)
    ok("Syntax valid: analyst_signals.py")
except FileNotFoundError:
    fail("analyst_signals.py not created")
except SyntaxError as e:
    fail(f"Syntax error analyst_signals: {e}")

if t2 and "get_analyst_signals" in t2:
    ok("get_analyst_signals function present")
else:
    fail("get_analyst_signals missing")

# max_workers increased
mw = re.search(r"max_workers\s*=\s*(\d+)", t)
if mw and int(mw.group(1)) >= 7:
    ok(f"max_workers={mw.group(1)} (>=7)")
else:
    fail("max_workers not updated to 7")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-1500:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.2.8.")
    sys.exit(0)
