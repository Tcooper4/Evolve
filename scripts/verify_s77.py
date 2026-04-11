# -*- coding: utf-8 -*-
"""Session 77 verification — Home / deep dive / briefing / chart / news."""
import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 77 Verification ===\n")
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

if "short_quick_score" in tb and "_opportunity_passes" in tb:
    ok("Direction filter uses quick_score fallback")
else:
    fail("Direction filter no-forecast fix missing")

if "SP500" in tb and "[:50]" not in tb:
    ok("Hardcoded [:50] removed")
else:
    fail("Hardcoded [:50] still present")

if "_uni_cap" in tb or "_uni_cap_map" in tb:
    ok("Dynamic universe cap present")
else:
    fail("Dynamic universe cap missing")

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

_news_fn = re.search(
    r"def get_news.*?(?=\n@|\ndef |\Z)",
    tc,
    re.DOTALL,
)
if _news_fn and (
    "url" in _news_fn.group()
    or "link" in _news_fn.group()
):
    ok("get_news preserves url/link")
else:
    fail("get_news missing url/link")

ta = open(
    "components/analyze_chart.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ta)
    ok("Syntax valid: analyze_chart.py")
except SyntaxError as e:
    fail(f"Syntax error chart: {e}")

if "_is_24hr" in ta or "_24hr" in ta:
    ok("24hr asset detection present")
else:
    fail("24hr asset detection missing")

if (
    "ticker" in ta
    and "_intraday_rangebreaks" in ta
):
    ok("ticker passed to rangebreaks")
else:
    fail("ticker not passed to rangebreaks")

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

if (
    "st.metric" in td
    and "AI Score" in td
    and "Target" in td
    and "Stop Loss" in td
):
    ok("Rich recommendation formatting")
else:
    fail("Recommendation formatting not updated")

tdiag = open(
    "components/analyze_diagnostics.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tdiag)
    ok("Syntax valid: analyze_diagnostics.py")
except SyntaxError as e:
    fail(f"Syntax error diagnostics: {e}")

if (
    "moves around a lot" in tdiag
    or "yearly swings" in tdiag
    or "been going up" in tdiag
):
    ok("Plain English risk text present")
else:
    fail("Plain English risk text missing")

if (
    "Quick checks" not in tdiag
    and "ADF, Ljung-Box" not in tdiag
):
    ok("Redundant ADF section removed")
else:
    fail("ADF section still present in diagnostics")

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
        "All checks passed. Ready to commit v4.6.8."
    )
    sys.exit(0)
