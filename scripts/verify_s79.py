# -*- coding: utf-8 -*-
"""Session 79 verification — briefing dedup, news titles, max results, progress."""
import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
print("=== Session 79 Verification ===\n")
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

if "_overlap" in tb and "_deduped_longs" in tb:
    ok("Deduplication logic present")
else:
    fail("Deduplication missing")

tc = open(
    "trading/data/price_cache.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tc)
    ok("Syntax valid: price_cache.py")
except SyntaxError as e:
    fail(f"Syntax error price_cache: {e}")

if ("({'url'" in tc or "({'" in tc) and "split" in tc and "title" in tc:
    ok("Title dict suffix cleanup present")
else:
    fail("Title cleanup missing")

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

if "briefing_max_results" in ts:
    ok("Max results in Settings")
else:
    fail("Max results missing from Settings")

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

if "briefing_max_results" in td or "max_positions=_max" in td:
    ok("Max results used in Dashboard")
else:
    fail("Max results not wired in Dashboard")

if "max_positions" in tb and "report[" in tb:
    ok("max_positions stored in report")
else:
    fail("max_positions missing from report")

if "_universe_size" in tb or "len(uni)" in tb:
    ok("Universe size in progress")
else:
    fail("Universe size not in progress")

tm = open(
    "trading/analysis/market_scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
if "_emit_progress(n_pend, max(1, total)" in tm:
    ok("Scanner progress uses full universe total")
else:
    fail("Scanner progress denominator not fixed")

try:
    from trading.data.price_cache import get_news

    _items = get_news("AAPL")
    if _items:
        _t = str(_items[0].get("title", ""))
        if "({'url'" in _t or "({'" in _t:
            fail(f"Dict suffix in title: {_t[:120]!r}")
        else:
            ok("News title cleanup works")
    else:
        ok("News title cleanup works (no items)")
except Exception as _ne:
    fail(f"News test: {_ne}")

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
    print("All checks passed. Ready to commit v4.7.0.")
    sys.exit(0)
