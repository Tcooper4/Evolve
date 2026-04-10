import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 58 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Trainer importable
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.analysis"
        " import ml_score_trainer;"
        "print('OK')",
    ],
    capture_output=True,
    text=True,
    timeout=30,
)
if _r.returncode == 0 and "OK" in _r.stdout:
    ok("ml_score_trainer importable")
else:
    fail(
        f"ml_score_trainer import failed:"
        f" {_r.stderr[:200]}"
    )

# Settings page has train button
t = open(
    "pages/7_Settings.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: 7_Settings.py")
except SyntaxError as e:
    fail(f"Syntax error Settings: {e}")

if "train" in t.lower() and (
    "ml_score" in t.lower()
    or "ML Score" in t
):
    ok("Train ML Score button present")
else:
    fail("Train ML Score button missing")

# Spinner/feedback present
if "spinner" in t or "st.spinner" in t:
    ok("Spinner feedback in Settings")
else:
    fail("No spinner on train button")

# Path consistency
trainer_path = None
for root, dirs, files in os.walk("trading"):
    for f in files:
        if "ml_score_trainer" in f:
            trainer_path = os.path.join(root, f)
if trainer_path:
    tt = open(
        trainer_path,
        encoding="utf-8",
        errors="replace",
    ).read()
    # Check save path matches load path
    if ".cache/ml_score" in tt:
        ok("Trainer saves to .cache/ml_score")
    else:
        fail(
            "Trainer save path may not "
            "match ai_score.py load path"
        )
else:
    fail("ml_score_trainer.py not found")

# Check notify in analyze if untrained
_found_notice = False
for fpath in [
    "components/tabs/tab_quick_forecast.py",
    "components/analyze_ai_score.py",
]:
    if os.path.exists(fpath):
        c = open(
            fpath, encoding="utf-8",
            errors="replace",
        ).read()
        if (
            "ml_score" in c.lower()
            and (
                "not trained" in c
                or "inactive" in c
                or "train the model" in c
            )
        ):
            _found_notice = True
if _found_notice:
    ok("ML Score untrained notice present")
else:
    fail(
        "ML Score untrained notice missing"
        " in Analyze page"
    )

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
        "Ready to commit v4.5.1."
    )
    sys.exit(0)
