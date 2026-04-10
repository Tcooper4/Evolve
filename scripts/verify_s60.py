import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 60 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — no AAPL hardcoded default
t = open(
    "pages/2_Analyze.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: 2_Analyze.py")
except SyntaxError as e:
    fail(f"Syntax error Analyze: {e}")

_init_aapl = re.search(
    r'analyze_symbol.*["\']AAPL["\']',
    t,
)
_bad_ticker_default = (
    'get("analyze_ticker", "AAPL")' in t
    or "get('analyze_ticker', 'AAPL')" in t
)
if _init_aapl or _bad_ticker_default:
    fail(
        "AAPL still used as default ticker init: "
        f"{_init_aapl.group() if _init_aapl else 'analyze_ticker'}"
    )
else:
    ok("AAPL not hardcoded as default")

# Check empty state prompt exists
if (
    "Enter a ticker" in t
    or "enter a ticker" in t.lower()
):
    ok("Empty state prompt present")
else:
    fail("Empty state prompt missing")

# Fix 2 — options chain
t2 = open(
    "components/tabs/tab_options_chain.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: tab_options_chain.py")
except SyntaxError as e:
    fail(f"Syntax error options: {e}")

if "load options chain" in t2.lower():
    ok("Load Options Chain button present")
else:
    fail("Load Options Chain button missing")

# Fix 3 — chart rangebreaks
t3 = open(
    "components/analyze_chart.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t3)
    ok("Syntax valid: analyze_chart.py")
except SyntaxError as e:
    fail(f"Syntax error chart: {e}")

if (
    "5d" in t3
    and "rangebreaks" in t3
    and (
        "dtick" in t3
        or "total_seconds" in t3
        or "_dtick" in t3
    )
):
    ok(
        "5d rangebreaks with interval "
        "detection present"
    )
else:
    fail(
        "5d rangebreaks fix missing or "
        "no interval detection"
    )

# Syntax spot-check
for fpath in [
    "pages/2_Analyze.py",
    "components/analyze_chart.py",
    "components/tabs/tab_options_chain.py",
]:
    try:
        ast.parse(
            open(
                fpath,
                encoding="utf-8",
                errors="replace",
            ).read()
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
print((result.stdout + result.stderr)[-800:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(
    f"\n=== {len(PASS)} passed, "
    f"{len(FAIL)} failed ==="
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. "
        "Ready to commit v4.5.3."
    )
    sys.exit(0)
