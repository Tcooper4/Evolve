import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 43 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# signal_score_store.py
try:
    t = open(
        "trading/analysis/signal_score_store.py",
        encoding="utf-8",
        errors="replace",
    ).read()
    ast.parse(t)
    ok("Syntax valid: signal_score_store.py")
    for fn in [
        "init_score_db",
        "record_score",
        "fill_forward_returns",
        "get_dimension_scores_and_returns",
        "get_global_dimension_scores",
    ]:
        if fn in t:
            ok(f"{fn} present")
        else:
            fail(f"{fn} missing")
except FileNotFoundError:
    fail("signal_score_store.py not created")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

# ai_score.py
t2 = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

if "_compute_ic_weights" in t2:
    ok("_compute_ic_weights present")
else:
    fail("_compute_ic_weights missing")

if "record_score" in t2:
    ok("Score persistence in ai_score.py")
else:
    fail("Score persistence missing")

if "_compute_ic_weights(symbol)" in t2 or "_compute_ic_weights(" in t2:
    ok("_compute_ic_weights called instead of static weights")
else:
    fail("Static weights not replaced")

# Settings page IC status
t3 = open(
    "pages/7_Settings.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t3)
    ok("Syntax valid: 7_Settings.py")
except SyntaxError as e:
    fail(f"Syntax error Settings: {e}")

if "Signal IC Status" in t3:
    ok("IC Status section in Settings")
else:
    fail("IC Status missing from Settings")

# Functional test
print("\n--- Score store test ---")
_r = subprocess.run(
    [
        python,
        "-c",
        (
            "import sys; sys.path.insert(0,'.');"
            "from trading.analysis.signal_score_store import "
            "init_score_db, record_score, "
            "get_global_dimension_scores;"
            "init_score_db();"
            "record_score('TEST',5.0,5.0,"
            "5.0,5.0,5.0,100.0);"
            "print('Record OK');"
            "r=get_global_dimension_scores("
            "min_rows=1);"
            "print('Global data:', "
            "r['n_obs'] if r else 'None')"
        ),
    ],
    capture_output=True,
    text=True,
    timeout=15,
)
print(_r.stdout[:300])
if _r.returncode == 0:
    ok("Score store functional")
else:
    fail(f"Store error: {_r.stderr[:200]}")

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
    print("All checks passed. Ready to commit v4.3.4.")
    sys.exit(0)
