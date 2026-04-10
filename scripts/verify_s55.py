import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 55 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — GNN scale check (nested package path)
_gnn_path = os.path.join(
    "trading", "models", "advanced", "gnn", "gnn_model.py"
)
t = open(
    _gnn_path,
    encoding="utf-8", errors="replace"
).read()
try:
    ast.parse(t)
    ok("Syntax valid: gnn_model.py")
except SyntaxError as e:
    fail(f"Syntax error gnn_model: {e}")

if "_ratio" in t:
    ok("GNN scale/ratio check present")
else:
    fail("GNN scale check missing")

# Fix 1b — GNN in skip_mape list
tr = open(
    "trading/models/forecast_router.py",
    encoding="utf-8", errors="replace"
).read()
try:
    ast.parse(tr)
    ok("Syntax valid: forecast_router.py")
except SyntaxError as e:
    fail(f"Syntax error router: {e}")

if (
    "_skip_mape" in tr
    and "gnn" in tr.lower()
):
    ok("GNN in skip_mape list")
else:
    fail("GNN not in skip_mape list")

# Fix 2 — alias dict extended
tr2 = open(
    "trading/data/ticker_resolver.py",
    encoding="utf-8", errors="replace"
).read()
try:
    ast.parse(tr2)
    ok("Syntax valid: ticker_resolver.py")
except SyntaxError as e:
    fail(f"Syntax error resolver: {e}")

for alias, target in [
    ("SPX", "^GSPC"),
    ("VIX", "^VIX"),
    ("BTC", "BTC-USD"),
    ("GOLD", "GC=F"),
]:
    if alias in tr2 and target in tr2:
        ok(f"Alias {alias} -> {target}")
    else:
        fail(
            f"Alias {alias} -> {target} "
            f"missing"
        )

# Functional test — resolver
print("\n--- Resolver test ---")
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.data.ticker_resolver import resolve_ticker;"
        "cases=[('SPX','^GSPC'),('VIX','^VIX'),('BTC','BTC-USD'),"
        "('AAPL','AAPL'),('GOLD','GC=F')];"
        "ok=all(resolve_ticker(a,validate=False)==e for a,e in cases);"
        "print('OK' if ok else 'FAIL');"
        "sys.exit(0 if ok else 1)",
    ],
    capture_output=True,
    text=True,
    timeout=60,
)
print(_r.stdout[:400])
if _r.returncode == 0:
    ok("Resolver functional test passed")
else:
    fail(f"Resolver error: {_r.stderr[:200]}")

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
        "Ready to commit v4.4.8."
    )
    sys.exit(0)
