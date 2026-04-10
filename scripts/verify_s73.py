import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 73 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — skip_forecast param
tb = open(
    "agents/briefing/morning_briefing.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tb)
    ok("Syntax valid: morning_briefing.py")
except SyntaxError as e:
    fail(f"Syntax error briefing: {e}")

if "skip_forecast" in tb:
    ok("skip_forecast param present")
else:
    fail("skip_forecast missing")

if "include_forecasts" in tb:
    ok("include_forecasts pref read")
else:
    fail("include_forecasts missing")

_default = re.search(
    r"include_forecasts.*False",
    tb,
)
if _default:
    ok("include_forecasts defaults False")
else:
    fail("include_forecasts default wrong")

# Fix 2 — Settings toggle
ts = open(
    "pages/7_Settings.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ts)
    ok("Syntax valid: 7_Settings.py")
except SyntaxError as e:
    fail(f"Syntax error Settings: {e}")

if "include_forecasts" in ts:
    ok("include_forecasts in Settings")
else:
    fail("include_forecasts missing from Settings")

if (
    ("2" in ts and "3 min" in ts.lower())
    or "2–3" in ts
    or "2-3 min" in ts.lower()
):
    ok("Speed warning in Settings toggle")
else:
    fail("Speed warning missing")

# Fix 3 — error handling
td = open(
    "pages/1_Dashboard.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(td)
    ok("Syntax valid: 1_Dashboard.py")
except SyntaxError as e:
    fail(f"Syntax error Dashboard: {e}")

if "st.error" in td and "Briefing" in td:
    ok("Visible error handling present")
else:
    fail("Error handling not visible")

if "report.get(\"error\")" in td or "report.get('error')" in td:
    ok("Report validity check present")
else:
    fail("Report validity check missing")

# Fix 4 — portfolio optimizer
tp = open(
    "trading/optimization/portfolio_optimizer.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tp)
    ok("Syntax valid: portfolio_optimizer")
except SyntaxError as e:
    fail(f"Syntax error optimizer: {e}")

if "equal_weight" in tp or "equal weight" in tp.lower():
    ok("Equal weight fallback present")
else:
    fail("Equal weight fallback missing")

# Fix 5 — forecast progress
if "forecast_progress_callback" in tb:
    ok("Forecast progress callback in briefing")
else:
    fail("Forecast progress callback missing")

if "forecast_progress_callback" in td:
    ok("Forecast progress wired in Dashboard")
else:
    fail("Forecast progress not wired")

# Smoke test
print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
_out = result.stdout + result.stderr
_passes = _out.count("PASS:")
_fails = _out.count("FAIL:")
print(f"Smoke: {_passes} PASS, {_fails} FAIL")
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.6.4.")
    sys.exit(0)
