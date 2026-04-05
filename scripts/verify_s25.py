# -*- coding: utf-8 -*-
import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 25 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


path = "pages/2_Analyze.py"
text = open(path, encoding="utf-8", errors="replace").read()

if "_needs_load" in text:
    ok("2_Analyze.py: _needs_load guard present")
else:
    fail("2_Analyze.py: _needs_load guard missing")

if "auto_forecast_key" in text:
    ok("2_Analyze.py: auto-forecast block present")
else:
    fail("2_Analyze.py: auto-forecast block missing")

if "Loading forecast data" in text:
    ok("2_Analyze.py: spinner present")
else:
    fail("2_Analyze.py: spinner missing")

for f in [
    "pages/2_Analyze.py",
    "components/tabs/tab_quick_forecast.py",
]:
    try:
        ast.parse(open(f, encoding="utf-8", errors="replace").read())
        ok(f"Syntax valid: {f}")
    except SyntaxError as e:
        fail(f"Syntax error in {f}: {e}")

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
    for f in FAIL:
        print(f"  FAIL  {f}")
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.1.2.")
    sys.exit(0)
