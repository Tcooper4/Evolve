import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 49 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# earnings_quality module
try:
    t = open(
        "trading/data/earnings_quality.py",
        encoding="utf-8",
        errors="replace",
    ).read()
    ast.parse(t)
    ok("Syntax valid: earnings_quality.py")
    for check, pat in [
        ("get_earnings_quality", "get_earnings_quality"),
        ("Accruals formula", "accruals_ratio"),
        ("Beat rate", "beat_rate"),
        ("Revision signal", "revision_signal"),
        ("Neutral fallback", "_neutral_eq"),
    ]:
        if pat in t:
            ok(f"{check} present")
        else:
            fail(f"{check} missing")
except FileNotFoundError:
    fail("earnings_quality.py not created")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

# ai_score wiring
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

for check, pat in [
    ("earnings_quality in SIGNAL_SOURCES", "earnings_quality"),
    ("Parallel worker", "_fetch_earnings_quality_safe"),
    ("Merge block signal", "Earnings Quality"),
    ("Accruals signal append", '"Accruals"'),
    ("max_workers 10", "max_workers=10"),
]:
    if pat in t2:
        ok(f"{check} present")
    else:
        fail(f"{check} missing")

# Functional import test
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.data.earnings_quality"
        " import get_earnings_quality,"
        " _neutral_eq;"
        "r = _neutral_eq('TEST');"
        "assert r['success'] == False;"
        "assert r['signal'] == 'NEUTRAL';"
        "assert 'beat_rate' in r;"
        "assert 'accruals_ratio' in r;"
        "print('Import + neutral OK')",
    ],
    capture_output=True,
    text=True,
    timeout=10,
)
print(_r.stdout[:200])
if _r.returncode == 0:
    ok("earnings_quality functional")
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
    print("All checks passed. Ready to commit v4.4.2.")
    sys.exit(0)
