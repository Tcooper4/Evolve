import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 47 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


t = open(
    "components/tabs/tab_options_chain.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: tab_options_chain.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

checks = [
    ("Strategy Visualizer header", "Strategy Visualizer"),
    ("_get_premium helper", "_get_premium"),
    ("P&L computation", "tozeroy"),
    ("Strategy selectbox", "opt_strat_"),
    ("Probability of profit", "Prob. of Profit"),
    ("Breakeven detection", "Breakeven"),
    ("Long Call strategy", "Long Call"),
    ("Bull Call Spread", "Bull Call Spread"),
    ("Long Straddle", "Long Straddle"),
    ("Max profit metric", "Max Profit"),
]
for name, pattern in checks:
    if pattern in t:
        ok(f"{name} present")
    else:
        fail(f"{name} missing")

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
    print("All checks passed. Ready to commit v4.3.9.")
    sys.exit(0)
