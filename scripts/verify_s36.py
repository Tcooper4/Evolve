import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 36 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


t = open(
    "components/tabs/tab_quick_forecast.py",
    encoding="utf-8",
    errors="replace",
).read()

try:
    ast.parse(t)
    ok("Syntax valid: tab_quick_forecast.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

# Automatic consensus should not call
# get_consensus_forecast outside button
# Count occurrences of get_consensus_forecast
matches = [
    m.start() for m in re.finditer(
        r"get_consensus_forecast",
        t,
    )
]
# Find position of forecast_button
btn_pos = t.find("forecast_button = st.button")
# All get_consensus_forecast calls should
# be after the button definition
auto_calls = [
    p for p in matches
    if p < btn_pos
]
if not auto_calls:
    ok("No automatic consensus calls before forecast button")
else:
    fail(f"Automatic consensus calls found before button: {len(auto_calls)}")

# Check display-only fallback exists
if "Run **Generate Forecast**" in t:
    ok("Display-only fallback present")
else:
    fail("Display-only fallback missing")

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
    print("All checks passed. Ready to commit v4.2.6.")
    sys.exit(0)
