# -*- coding: utf-8 -*-
import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 26 Verification ===\n")
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
    ok("_needs_load guard present")
else:
    fail("_needs_load guard missing")

if "auto_forecast_key" not in text:
    ok("auto-forecast block removed")
else:
    fail("auto-forecast block still present")

if "st.rerun()" in text:
    ok("st.rerun() present after data load")
else:
    fail("st.rerun() missing")

if "Auto-run consensus forecast" not in text:
    ok("auto-forecast comment removed")
else:
    fail("auto-forecast comment still present")

try:
    ast.parse(text)
    ok("Syntax valid: pages/2_Analyze.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

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
    print("All checks passed. Ready to commit v4.1.3.")
    sys.exit(0)
