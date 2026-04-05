import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 24 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


path = "trading/models/lstm_model.py"
text = open(path, encoding="utf-8", errors="replace").read()

# Check save includes scalers
if '"X_scaler"' in text and '"y_scaler"' in text:
    ok("lstm_model.py: scaler keys present in code")
else:
    fail("lstm_model.py: scaler keys NOT found")

# Check load handles dict format
if "isinstance(loaded, dict)" in text:
    ok("lstm_model.py: dict-aware load present")
else:
    fail("lstm_model.py: dict-aware load NOT found")

# Check legacy fallback
if "legacy format" in text or "Legacy format" in text:
    ok("lstm_model.py: legacy fallback present")
else:
    fail("lstm_model.py: legacy fallback NOT found")

# Syntax check
try:
    ast.parse(text)
    ok("Syntax valid: lstm_model.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

# Cache dir should be empty or not exist
cache_dir = ".cache/lstm"
if os.path.exists(cache_dir):
    files = [f for f in os.listdir(cache_dir) if f.endswith(".joblib")]
    if len(files) == 0:
        ok("Cache cleared: no stale .joblib files")
    else:
        fail(f"Cache NOT cleared: {len(files)} old files remain")
else:
    ok("Cache dir does not exist (clean slate)")

# Smoke test
print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-2000:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    for f in FAIL:
        print(f"  FAIL  {f}")
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.1.1.")
    sys.exit(0)
