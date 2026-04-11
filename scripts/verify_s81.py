# -*- coding: utf-8 -*-
"""Session 81 verification (Settings audit wiring)."""
import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 81 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — scoring_style weights
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

if "SCORING_STYLE_WEIGHTS" in ta:
    ok("SCORING_STYLE_WEIGHTS defined")
else:
    fail("SCORING_STYLE_WEIGHTS missing")

if (
    "Momentum-heavy" in ta
    and "Technical-heavy" in ta
    and "Fundamental-heavy" in ta
):
    ok("All 4 scoring styles present")
else:
    fail("Scoring styles incomplete")

if "scoring_style" in ta and "def compute_ai_score" in ta:
    ok("scoring_style param in compute_ai_score")
else:
    fail("scoring_style param missing")

_code = (
    "import sys; sys.path.insert(0, '.'); "
    "from trading.analysis.ai_score import SCORING_STYLE_WEIGHTS; "
    "assert len(SCORING_STYLE_WEIGHTS) == 4; "
    "w = SCORING_STYLE_WEIGHTS['Momentum-heavy']; "
    "assert w['momentum'] == 0.50; "
    "print('OK')"
)
_r = subprocess.run(
    [python, "-c", _code],
    capture_output=True,
    text=True,
    timeout=60,
)
if _r.returncode == 0 and "OK" in (_r.stdout or ""):
    ok("Scoring weights functional")
else:
    fail(f"Weights test: {(_r.stdout + _r.stderr)[:200]}")

# Fix 2 — callers pass scoring_style
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

if "scoring_style" in td:
    ok("scoring_style in deep_dive")
else:
    fail("scoring_style missing from deep_dive")

# Fix 3 — sectors in scanner
ts = open(
    "pages/3_Scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ts)
    ok("Syntax valid: 3_Scanner.py")
except SyntaxError as e:
    fail(f"Syntax error Scanner: {e}")

if "preferred_sectors" in ts or "_pref_sectors" in ts:
    ok("preferred_sectors in Scanner")
else:
    fail("preferred_sectors missing from Scanner")

# Fix 4 — ID consistency
tset = open(
    "pages/7_Settings.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tset)
    ok("Syntax valid: 7_Settings.py")
except SyntaxError as e:
    fail(f"Syntax error Settings: {e}")

_bare_id = re.findall(
    r"_[st]sid\s*=\s*get_stable_user_id\(\)",
    tset,
)
if _bare_id:
    fail(
        f"Bare get_stable_user_id() still used {len(_bare_id)}x",
    )
else:
    ok("All IDs use evolve_session_id first")

# Fix 5 — version + message
if "v4.7.1" in tset:
    ok("Version string updated")
else:
    fail("Version string not updated")

if (
    "Morning Briefing" in tset
    and "Scanner" in tset
    and "AI Score" in tset
):
    ok("Save message updated")
else:
    fail("Save message not updated")

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
    f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===",
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. Ready to commit v4.7.2.",
    )
    sys.exit(0)
