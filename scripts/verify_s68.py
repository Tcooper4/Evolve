import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 68 Verification ===\n")
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

# Fix 1 — two-phase scan
if "quick_technical" in tb and "_pre_tickers" in tb:
    ok("Two-phase scan present")
else:
    fail("Two-phase scan missing")

if "min_quick_score" in tb:
    ok("min_quick_score in briefing scan")
else:
    fail("min_quick_score missing")

# Fallback guard present
if (
    "_pre_tickers" in tb
    and ("Fallback" in tb or "fallback" in tb.lower())
):
    ok("Pre-filter fallback guard present")
else:
    fail("Pre-filter fallback missing")

# Fix 2 — single prefs load
_prefs_loads = len(re.findall(r"load_user_preferences", tb))
if _prefs_loads <= 2:
    ok(
        f"Preferences loaded {_prefs_loads}x "
        f"(consolidated)"
    )
else:
    fail(
        f"Still {_prefs_loads} load_user_preferences calls"
    )

# _briefing_prefs set before _scan_universe call
_gen_block = tb[
    tb.find("def generate"):
    tb.find("def generate") + 3000
]
_prefs_pos = _gen_block.find("_briefing_prefs")
_scan_pos = _gen_block.find("_scan_universe")
if (
    _prefs_pos > 0
    and _scan_pos > 0
    and _prefs_pos < _scan_pos
):
    ok("_briefing_prefs set before _scan_universe()")
else:
    fail("_briefing_prefs not set before _scan_universe()")

# Fix 3 — batch sector
if (
    "yf.Tickers" in tb
    and "sector" in tb
    and "_sector_map" in tb
):
    ok("Batch sector fetch present")
else:
    fail("Batch sector fetch missing")

# Fix 4 — Monte Carlo not in briefing
_analyze_fn = tb[
    tb.find("def _analyze_opportunity"):
    tb.find("def _analyze_opportunity") + 4000
]
if (
    "if briefing:" in _analyze_fn
    and "MonteCarloSimulator" in _analyze_fn
):
    fail(
        "Monte Carlo still runs when briefing=True"
    )
else:
    ok("Monte Carlo skipped in briefing mode")

# Dashboard wiring
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

print(
    f"\n=== {len(PASS)} passed, {len(FAIL)} failed ==="
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. Ready to commit v4.6.1."
    )
    sys.exit(0)
