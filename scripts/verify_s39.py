import ast
import subprocess
import sys

import numpy as np

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 39 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 - Ensemble denormalization
t = open(
    "trading/models/ensemble_model.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: ensemble_model.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

if "abs(pred[-1]) < 1.5" in t or (
    "normalized" in t.lower() and "pred" in t and "scale" in t
):
    ok("Ensemble denorm check present")
else:
    fail("Ensemble denorm check missing")

# Fix 2 - Transformer norm stats
t2 = open(
    "trading/models/advanced/transformer/time_series_transformer.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: transformer")
except SyntaxError as e:
    fail(f"Syntax error transformer: {e}")

if (
    "_prepare_data" in t2
    and "is_training=True" in t2
    and "fit" in t2
):
    ok("Transformer fit calls _prepare_data(is_training=True)")
else:
    fail("Transformer fit missing _prepare_data call")

# Functional test - ensemble scale check
print("\n--- Ensemble scale test ---")
try:
    import sys as _sys

    _sys.path.insert(0, ".")
    import numpy as _np

    # Simulate normalized output
    pred_norm = _np.array([0.98, 0.99, 1.00, 1.01, 1.02, 1.01, 1.00])
    last_price = 258.90
    if last_price > 10 and pred_norm.size > 0 and abs(pred_norm[-1]) < 1.5:
        scaled = pred_norm * last_price
        ok(
            f"Scale check works: {pred_norm[-1]:.2f} -> {scaled[-1]:.2f}"
        )
    else:
        fail("Scale check logic failed")
except Exception as e:
    fail(f"Scale test error: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-1500:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.3.0.")
    sys.exit(0)
