import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 74 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — no CommentaryEngine crash
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

if "CommentaryEngine" in td:
    fail("CommentaryEngine still referenced in deep_dive.py")
else:
    ok("CommentaryEngine removed")

if "call_active_llm_simple" in td and "AI Commentary" in td:
    ok("Direct LLM commentary present")
else:
    fail("Direct LLM commentary missing")

# Fix 2 — no duplicate chart patterns (import + one call = 2 name matches)
_pattern_calls = re.findall(r"ChartPatternDetector\(", td)
if len(_pattern_calls) > 1:
    fail(
        f"ChartPatternDetector( appears {len(_pattern_calls)}x — "
        f"duplicate not removed"
    )
else:
    ok("No duplicate chart patterns")

# Fix 3 — Options tab exists
if "Options" in td and "render_options" in td:
    ok("Options tab present")
else:
    fail("Options tab missing")

_risk_block = re.search(
    r"with tr:(.*?)with to:",
    td,
    re.DOTALL,
)
if _risk_block:
    if "render_options" in _risk_block.group(1):
        fail("render_options still in Risk tab")
    else:
        ok("render_options removed from Risk tab")
else:
    fail("Could not parse Risk tab block")

# Fix 4 — news links
tn = open(
    "components/analyze_news.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tn)
    ok("Syntax valid: analyze_news.py")
except SyntaxError as e:
    fail(f"Syntax error news: {e}")

if "url" in tn and "markdown" in tn.lower() and "](" in tn:
    ok("News items render as links")
else:
    fail("News items not rendering as links")

# Fix 5 — page chat guarded
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

_guard = 'if not st.session_state.get("deep_dive_ticker")'
if _guard in tp and "home_bottom_chat" in tp:
    if tp.find(_guard) < tp.find("home_bottom_chat"):
        ok("Page chat guarded by deep_dive_ticker check")
    else:
        fail("Page chat guard not before chat_input")
else:
    fail("Missing keys for chat guard check")

# Fix 6 — timeframe controls
if "_tf_options" in td:
    ok("Timeframe controls present")
else:
    fail("Timeframe controls missing")

if "dd_tf_" in td:
    ok("Timeframe session state keys")
else:
    fail("Timeframe keys missing")

# Commentary engine: market regime method exists
tc = open(
    "trading/commentary/commentary_engine.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tc)
    ok("Syntax valid: commentary_engine.py")
except SyntaxError as e:
    fail(f"Syntax error commentary_engine: {e}")
if "def _generate_market_regime_commentary" in tc:
    ok("_generate_market_regime_commentary defined")
else:
    fail("_generate_market_regime_commentary missing")

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
    print("All checks passed. Ready to commit v4.6.5.")
    sys.exit(0)
