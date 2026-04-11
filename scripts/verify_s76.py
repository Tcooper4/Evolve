# -*- coding: utf-8 -*-
"""Session 76 verification — short score, briefing, dashboard, settings."""
import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 76 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — Short Score
ta = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ta)
    ok("Syntax valid: ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error ai_score: {e}")

if "compute_short_score" in ta:
    ok("compute_short_score defined")
else:
    fail("compute_short_score missing")

if "_neutral_short" in ta:
    ok("_neutral_short defined")
else:
    fail("_neutral_short missing")

_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.analysis.ai_score import compute_short_score;"
        "print('Import OK')",
    ],
    capture_output=True,
    text=True,
    timeout=60,
)
if _r.returncode == 0 and "OK" in _r.stdout:
    ok("compute_short_score importable")
else:
    fail(
        "Short score import failed: "
        f"{(_r.stdout + _r.stderr)[:200]}"
    )

tm = open(
    "trading/analysis/market_scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tm)
    ok("Syntax valid: market_scanner.py")
except SyntaxError as e:
    fail(f"Syntax error scanner: {e}")

if "_short_quick_score" in tm:
    ok("_short_quick_score defined")
else:
    fail("_short_quick_score missing")

if "high_short_score" in tm:
    ok("high_short_score filter present")
else:
    fail("high_short_score missing")

if "short_quick_score" in tm:
    ok("short_quick_score in results")
else:
    fail("short_quick_score missing from results")

# Fix 2 — risk commentary
tc = open(
    "components/analyze_diagnostics.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tc)
    ok("Syntax valid: analyze_diagnostics.py")
except SyntaxError as e:
    fail(f"Syntax error diagnostics: {e}")

if "_render_risk_summary" in tc:
    ok("_render_risk_summary present")
else:
    fail("_render_risk_summary missing")

# Fix 3 — briefing shorts
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

if "_scan_shorts" in tb:
    ok("_scan_shorts method present")
else:
    fail("_scan_shorts missing")

if "short_opportunities" in tb:
    ok("short_opportunities in report")
else:
    fail("short_opportunities missing")

# Fix 4 — deep dive short score
td = open(
    "components/deep_dive.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(td)
    ok("Syntax valid: deep_dive.py")
except SyntaxError as e:
    fail(f"Syntax error deep_dive: {e}")

if "compute_short_score" in td:
    ok("Short score in deep dive")
else:
    fail("Short score missing from deep dive")

if "_show_short" in td:
    ok("Direction pref gate present")
else:
    fail("Direction pref gate missing")

# Fix 5 — Settings manual triggers
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

if "load_gpr_btn" in ts:
    ok("GPR trigger in Settings")
else:
    fail("GPR trigger missing")

if "compute_breadth_btn" in ts:
    ok("EPS breadth trigger in Settings")
else:
    fail("EPS breadth trigger missing")

# Dashboard no longer auto-computes
tp = open(
    "pages/1_Dashboard.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tp)
    ok("Syntax valid: 1_Dashboard.py")
except SyntaxError as e:
    fail(f"Syntax error Dashboard: {e}")

if "_rb_computing" in tp:
    fail("Auto-compute _rb_computing still in Dashboard")
else:
    ok("Auto-compute removed from Dashboard")

# Fix 6 — scanner short score column
tsc = open(
    "pages/3_Scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tsc)
    ok("Syntax valid: 3_Scanner.py")
except SyntaxError as e:
    fail(f"Syntax error Scanner: {e}")

if "short_quick_score" in tsc or "Short Score" in tsc:
    ok("Short Score in scanner display")
else:
    fail("Short Score missing from scanner")

# Smoke test
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

print(
    f"\n=== {len(PASS)} passed, {len(FAIL)} failed ==="
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. Ready to commit v4.6.7."
    )
    sys.exit(0)
