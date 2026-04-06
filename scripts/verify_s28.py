import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 28 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Check no st.stop() in any tab file
tab_dir = "components/tabs"
for fname in os.listdir(tab_dir):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(tab_dir, fname)
    text = open(fpath, encoding="utf-8", errors="replace").read()
    if "st.stop()" in text:
        fail(f"{fname}: still contains st.stop()")
    else:
        ok(f"{fname}: no st.stop()")

# Check diagnostic reverted
print()
diag_files = [
    "components/tabs/tab_market_analysis.py",
    "components/tabs/tab_diagnostics.py",
    "components/tabs/tab_multi_asset_gnn.py",
    "components/tabs/tab_causal.py",
    "components/tabs/tab_earnings.py",
    "components/tabs/tab_monte_carlo.py",
]
for f in diag_files:
    text = open(f, encoding="utf-8", errors="replace").read()
    if 'st.error(f"Tab error:' in text:
        fail(f"{f}: diagnostic st.error still present")
    else:
        ok(f"{f}: diagnostic reverted")

# Syntax check all tab files
print()
for fname in os.listdir(tab_dir):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(tab_dir, fname)
    try:
        ast.parse(open(fpath, encoding="utf-8", errors="replace").read())
        ok(f"Syntax valid: {fname}")
    except SyntaxError as e:
        fail(f"Syntax error in {fname}: {e}")

# Smoke test
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
    for f in FAIL:
        print(f"  FAIL  {f}")
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.1.5.")
    sys.exit(0)
