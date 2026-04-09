import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 48 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Dark pool module
try:
    t = open(
        "trading/data/dark_pool.py",
        encoding="utf-8",
        errors="replace",
    ).read()
    ast.parse(t)
    ok("Syntax valid: dark_pool.py")
    if "get_dark_pool_activity" in t:
        ok("get_dark_pool_activity present")
    else:
        fail("get_dark_pool_activity missing")
    if "finra" in t.lower():
        ok("FINRA endpoint referenced")
    else:
        fail("FINRA endpoint missing")
    if "_neutral_dp" in t:
        ok("Neutral fallback present")
    else:
        fail("Neutral fallback missing")
except FileNotFoundError:
    fail("dark_pool.py not created")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

# AI score wiring
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

if "dark_pool" in t2:
    ok("dark_pool in SIGNAL_SOURCES")
else:
    fail("dark_pool missing from SIGNAL_SOURCES")

if "get_dark_pool_activity" in t2:
    ok("dark pool wired in ai_score")
else:
    fail("dark pool not wired")

if "Dark Pool" in t2:
    ok("Dark Pool signal appended")
else:
    fail("Dark Pool signal missing")

# Functional import test
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.data.dark_pool import "
        "get_dark_pool_activity, _neutral_dp;"
        "r = _neutral_dp('TEST');"
        "assert r['success'] == False;"
        "assert r['signal'] == 'NEUTRAL';"
        "print('Import OK');"
        "print('Neutral fallback OK')",
    ],
    capture_output=True,
    text=True,
    timeout=10,
)
print(_r.stdout[:200])
if _r.returncode == 0:
    ok("Dark pool import functional")
else:
    fail(f"Import error: {_r.stderr[:200]}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-600:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.4.0.")
    sys.exit(0)
