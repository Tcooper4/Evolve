import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 67 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# market_scanner.py
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

if "quick_technical" in tm:
    ok("quick_technical in SCAN_FILTERS")
else:
    fail("quick_technical missing")

if "top_ai_score" in tm:
    fail("top_ai_score still present in market_scanner.py")
else:
    ok("top_ai_score removed from market_scanner.py")

if "_quick_score" in tm:
    ok("_quick_score() function present")
else:
    fail("_quick_score() missing")

if "min_quick_score" in tm:
    ok("min_quick_score param present")
else:
    fail("min_quick_score param missing")

# quick_score functional test
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.analysis.market_scanner import _quick_score;"
        "s1 = _quick_score(45, 8.0, 3.0, 1.8, -3.0);"
        "print(f'Strong stock: {s1}');"
        "assert s1 >= 7.0, f'Expected >=7, got {s1}';"
        "s2 = _quick_score(75, -8.0, -4.0, 0.4, -40.0);"
        "print(f'Weak stock: {s2}');"
        "assert s2 <= 4.5, f'Expected <=4.5, got {s2}';"
        "print('Quick score test OK')",
    ],
    capture_output=True,
    text=True,
    timeout=15,
)
print(_r.stdout[:200])
if _r.returncode == 0 and "OK" in _r.stdout:
    ok("_quick_score() functional test")
else:
    fail(f"Quick score test failed: {(_r.stdout + _r.stderr)[:150]}")

# 3_Scanner.py
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

if "top_ai_score" in ts:
    fail("top_ai_score still in 3_Scanner.py")
else:
    ok("top_ai_score removed from 3_Scanner.py")

if "min_quick_score" in ts:
    ok("min_quick_score passed to scan_market()")
else:
    fail("min_quick_score not passed")

if "Quick Score" in ts:
    ok("Quick Score label in Scanner UI")
else:
    fail("Quick Score label missing")

if (
    "quick score is a fast" in ts.lower()
    or "technical estimate" in ts.lower()
):
    ok("Quick Score caption present")
else:
    fail("Quick Score caption missing")

# No post-filter block remaining
if re.search(r"top_ai_score.*in.*selected_filters", ts):
    fail("Post-filter block still present")
else:
    ok("Double-filter redundancy removed")

# Spot-check
for fpath in [
    "trading/analysis/market_scanner.py",
    "pages/3_Scanner.py",
]:
    try:
        ast.parse(
            open(fpath, encoding="utf-8", errors="replace").read()
        )
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

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
    print("All checks passed. Ready to commit v4.6.0.")
    sys.exit(0)
