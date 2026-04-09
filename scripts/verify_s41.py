import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 41 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Congressional trading module
try:
    t = open(
        "trading/data/congressional_trading.py",
        encoding="utf-8",
        errors="replace",
    ).read()
    ast.parse(t)
    ok("Syntax valid: congressional_trading.py")
    if "get_congressional_trades" in t:
        ok("get_congressional_trades function present")
    else:
        fail("get_congressional_trades missing")
except FileNotFoundError:
    fail("congressional_trading.py not created")

# Wired into ai_score
t2 = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error ai_score: {e}")

if "congressional_trading" in t2:
    ok("Congressional wired in SIGNAL_SOURCES")
else:
    fail("Congressional not in SIGNAL_SOURCES")

if "_fetch_congressional_safe" in t2:
    ok("Congressional parallel worker present")
else:
    fail("Congressional parallel worker missing")

# Economic calendar
t3 = open(
    "trading/data/earnings_calendar.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t3)
    ok("Syntax valid: earnings_calendar")
except SyntaxError as e:
    fail(f"Syntax error calendar: {e}")

if "get_macro_calendar" in t3:
    ok("get_macro_calendar present")
else:
    fail("get_macro_calendar missing")

# max_workers increased to 8
mw = re.search(r"max_workers\s*=\s*(\d+)", t2)
if mw and int(mw.group(1)) >= 8:
    ok(f"max_workers={mw.group(1)} (>=8)")
else:
    fail("max_workers not updated to 8")

# Functional test
print("\n--- Macro calendar test ---")
_r = subprocess.run(
    [
        python,
        "-c",
        (
            "import sys; sys.path.insert(0,'.');"
            "from trading.data.earnings_calendar "
            "import get_macro_calendar;"
            "r=get_macro_calendar();"
            "print('Events:', len(r.get('events',[])));"
            "print('Success:', r.get('success'))"
        ),
    ],
    capture_output=True,
    text=True,
)
print(_r.stdout[:300])
if _r.returncode == 0:
    ok("Macro calendar functional")
else:
    fail(f"Macro calendar error: {_r.stderr[:200]}")

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

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.3.3.")
    sys.exit(0)
