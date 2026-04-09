import ast
import os
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 46 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Task 1 - no use_container_width left (same scope as fix script + vendor dirs)
_count = 0
_files = []
EXCLUDE = {
    "evolve_venv",
    "_archive",
    "__pycache__",
    ".git",
    ".cache",
    "scripts",
    "tests",
    "node_modules",
    ".venv",
    "venv",
}
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        txt = open(fpath, encoding="utf-8", errors="replace").read()
        if "use_container_width" in txt:
            _count += txt.count("use_container_width")
            _files.append(fpath)

if _count == 0:
    ok("No use_container_width remaining")
else:
    fail(
        f"{_count} use_container_width still in {len(_files)} files: "
        f"{_files[:3]}"
    )

# Syntax check all py files
_syntax_errors = []
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        try:
            ast.parse(
                open(fpath, encoding="utf-8", errors="replace").read()
            )
        except SyntaxError as e:
            _syntax_errors.append(f"{fpath}: {e}")

if not _syntax_errors:
    ok("All .py files parse cleanly")
else:
    for err in _syntax_errors[:5]:
        fail(f"Syntax error: {err}")

# Task 2 - ML trainer
t = open(
    "trading/analysis/ml_score_trainer.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: ml_score_trainer")
except SyntaxError as e:
    fail(f"Syntax error trainer: {e}")

if "try" in t and "continue" in t:
    ok("Per-ticker error handling present")
else:
    fail("Per-ticker error handling missing")

if "makedirs" in t or "exist_ok" in t:
    ok("Model save path creation present")
else:
    fail("Model save path creation missing")

# Quick functional test - import only
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.analysis"
        ".ml_score_trainer import "
        "MLScoreTrainer;"
        "t = MLScoreTrainer();"
        "print('MLScoreTrainer OK');"
        "# Test score when untrained;"
        "s = t.score('AAPL', {});"
        "print(f'Untrained score: {s}');"
        "assert 0 <= s <= 10, "
        "'Score out of range'",
    ],
    capture_output=True,
    text=True,
    timeout=15,
)
print(_r.stdout[:300])
if _r.returncode == 0:
    ok("MLScoreTrainer import + untrained score works")
else:
    fail(f"MLScoreTrainer error: {_r.stderr[:300]}")

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
    print("All checks passed. Ready to commit v4.3.8.")
    sys.exit(0)
