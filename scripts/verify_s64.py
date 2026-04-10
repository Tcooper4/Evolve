"""Session 64 verification: GNN coercion, GNN tab bounds, causal tab, smoke."""
import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 64 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)

tg = open(
    "trading/models/advanced/gnn/gnn_model.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tg)
    ok("Syntax valid: gnn_model.py")
except SyntaxError as e:
    fail(f"Syntax error gnn: {e}")

for coerce in [
    "int(epochs",
    "int(batch_size",
    "int(horizon",
    "float(correlation_threshold",
]:
    if coerce in tg:
        ok(f"Coercion present: {coerce})")
    else:
        fail(f"Coercion missing: {coerce})")

tt = open(
    "components/tabs/tab_multi_asset_gnn.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tt)
    ok("Syntax valid: tab_multi_asset_gnn.py")
except SyntaxError as e:
    fail(f"Syntax error gnn tab: {e}")

if "_bound_col" in tt or "np.asarray" in tt:
    ok("numpy bound fix present")
else:
    fail("numpy bound fix missing")

tc = open(
    "components/tabs/tab_causal.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tc)
    ok("Syntax valid: tab_causal.py")
except SyntaxError as e:
    fail(f"Syntax error causal: {e}")

for dead in [
    "_archive",
    "CausalModel",
    "causal_model",
]:
    if dead in tc:
        fail(f"Dead reference '{dead}' still in tab_causal.py")
    else:
        ok(f"'{dead}' removed from tab_causal.py")

for expected in [
    "Correlation Analysis",
    "px.imshow",
    "pct_change",
    "rolling",
    "Load Correlation Data",
]:
    if expected in tc:
        ok(f"New content: '{expected}'")
    else:
        fail(f"Missing new content: '{expected}'")

for fpath in [
    "components/analyze_tabs_sections.py",
    "components/tabs/tab_causal.py",
    "components/tabs/tab_multi_asset_gnn.py",
    "trading/models/advanced/gnn/gnn_model.py",
]:
    try:
        ast.parse(open(fpath, encoding="utf-8", errors="replace").read())
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
    cwd=_root,
    timeout=300000,
)
_out = result.stdout + result.stderr
_passes = _out.count("PASS:")
_fails = _out.count("FAIL:")
print(f"Smoke: {_passes} PASS, {_fails} FAIL")
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.5.7.")
    sys.exit(0)
