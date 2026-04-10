import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 56 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — LSTM pickle
t = open(
    "trading/models/lstm_model.py",
    encoding="utf-8", errors="replace"
).read()
try:
    ast.parse(t)
    ok("Syntax valid: lstm_model.py")
except SyntaxError as e:
    fail(f"Syntax error lstm: {e}")

if (
    "torch.save" in t
    or (
        "class _LSTM" in t
        and t.index("class _LSTM")
        < t.index("class LSTMForecaster")
    )
):
    ok("LSTM serialization fix present")
else:
    fail("LSTM serialization fix missing")

# Fix 2 — data validation
_found_dv = []
_EXCL = {
    "evolve_venv", ".git", "__pycache__", "_archive",
    ".venv", "venv", "node_modules",
}
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in _EXCL]
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        content = open(
            p, encoding="utf-8",
            errors="replace"
        ).read()
        _bad = "validation" + " unavailable"
        if _bad in content:
            _found_dv.append(p)
if _found_dv:
    fail(
        f"stale data-validation caption still "
        f"in: {_found_dv}"
    )
else:
    ok("data-validation caption removed")

# Fix 3 — options surface guard
t3 = open(
    "components/tabs/tab_options_chain.py",
    encoding="utf-8", errors="replace"
).read()
try:
    ast.parse(t3)
    ok("Syntax valid: tab_options_chain.py")
except SyntaxError as e:
    fail(f"Syntax error options tab: {e}")

if (
    "impliedVolatility" in t3
    and (
        "notna" in t3
        or "unavailable" in t3
        or "st.caption" in t3
    )
):
    ok("Volatility surface guard present")
else:
    fail("Volatility surface guard missing")

# Syntax spot-check
for fpath in [
    "pages/2_Analyze.py",
    "trading/models/lstm_model.py",
    "components/tabs/tab_options_chain.py",
]:
    try:
        ast.parse(open(
            fpath, encoding="utf-8",
            errors="replace"
        ).read())
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True, text=True
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
        "Ready to commit v4.4.9."
    )
    sys.exit(0)
