# -*- coding: utf-8 -*-
"""Session 78 verification — briefing prefs ID + bearish scan filter."""
import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 78 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


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

if (
    "prefs: Optional[Dict[str, Any]] = None" in tb
    or "prefs: dict = None" in tb
    or "prefs=None" in tb
):
    ok("prefs parameter in __init__")
else:
    fail("prefs parameter missing")

if "_is_bearish_only" in tb:
    ok("Direction-aware scan present")
else:
    fail("Direction-aware scan missing")

if "high_short_score" in tb and "_scan_filter" in tb:
    ok("high_short_score used for bearish scan")
else:
    fail("high_short_score not used for bearish scan")

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

if "prefs=_prefs" in td or "prefs=_load_briefing_prefs" in td:
    ok("Dashboard passes prefs to MorningBriefing")
else:
    fail("Dashboard not passing prefs")

if (
    "briefing_universe" in tb
    and "Morning briefing prefs:" in tb
    and "opportunity_direction" in tb
):
    ok("Prefs log line present")
else:
    fail("Prefs log line missing")

_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from agents.briefing.morning_briefing import MorningBriefing;"
        "mb = MorningBriefing("
        "universe='sp100',"
        "prefs={"
        "'opportunity_direction':"
        "'Bearish only (short signals)',"
        "'briefing_universe': 'SP500',"
        "}"
        ");"
        "assert mb._briefing_prefs.get("
        "'opportunity_direction'"
        ") == 'Bearish only (short signals)';"
        "print('Prefs passthrough OK')",
    ],
    capture_output=True,
    text=True,
    timeout=30,
)
print((_r.stdout + _r.stderr)[:200])
if _r.returncode == 0 and "OK" in _r.stdout:
    ok("Prefs passthrough functional")
else:
    fail(f"Prefs test failed: {(_r.stdout + _r.stderr)[:200]}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
    timeout=600,
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
    print("All checks passed. Ready to commit v4.6.9.")
    sys.exit(0)
