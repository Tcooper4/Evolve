import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 80 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — scanner
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

if "Buy Score" in ts:
    ok("Buy Score rename in scanner")
else:
    fail("Buy Score rename missing")

if "_short_mode" in ts:
    ok("Short mode detection in scanner")
else:
    fail("Short mode detection missing")

# Widget key conflict check
import re

_direct_sets = re.findall(
    r'session_state\["scanner_min_ai'
    r'_score"\]\s*=',
    ts,
)
if len(_direct_sets) == 0:
    ok("Widget key conflict resolved")
else:
    fail(
        f"Widget key still set directly {len(_direct_sets)}x",
    )

# Fix 2 — score mode toggle
ta = open(
    "pages/2_Analyze.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ta)
    ok("Syntax valid: 2_Analyze.py")
except SyntaxError as e:
    fail(f"Syntax error Analyze: {e}")

if "score_mode" in ta:
    ok("Score mode in Analyze page")
else:
    fail("Score mode missing from Analyze page")

if "analyze_score_mode" in ta:
    ok("Score mode session key present")
else:
    fail("Score mode session key missing")

if "Trading Style" in ta or "Score Mode" in ta:
    ok("Updated radio labels present")
else:
    fail("Radio labels not updated")

# Fix 3 — tabs sections
tc = open(
    "components/analyze_tabs_sections.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tc)
    ok("Syntax valid: analyze_tabs_sections.py")
except SyntaxError as e:
    fail(f"Syntax error tabs: {e}")

if "score_mode" in tc:
    ok("score_mode in tabs sections")
else:
    fail("score_mode missing from tabs")

# Fix 4 — news overlay hyperlinks
tn = open(
    "components/analyze_news_headline.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tn)
    ok("Syntax valid: analyze_news_headline.py")
except SyntaxError as e:
    fail(f"Syntax error news: {e}")

if (
    "url" in tn
    and "](" in tn
    and "markdown" in tn.lower()
):
    ok("News overlay has hyperlinks")
else:
    fail("News overlay links missing")

# Fix 5 — technical tab explanation
td = open(
    "components/tabs/tab_diagnostics.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(td)
    ok("Syntax valid: tab_diagnostics.py")
except SyntaxError as e:
    fail(f"Syntax error diagnostics: {e}")

if "_render_risk_summary" in td:
    ok("Plain-English summary in Technical tab")
else:
    fail("Plain-English summary missing")

# Fix 6 — ADF error
if (
    "truth value" not in td
    and (
        ".empty" in td
        or "is not None" in td
    )
):
    ok("DataFrame truth value fix present")
else:
    fail("DataFrame truth value fix missing or not needed")

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
        "All checks passed. Ready to commit v4.7.1.",
    )
    sys.exit(0)
