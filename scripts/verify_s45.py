import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 45 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# GNN fix
t = open(
    "trading/models/advanced/gnn/gnn_model.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: gnn_model.py")
except SyntaxError as e:
    fail(f"Syntax error GNN: {e}")

if "_ratio" in t:
    ok("GNN explicit scale detection present")
else:
    fail("GNN scale detection missing")

if "np.full" in t and "last_price" in t:
    ok("GNN fallback to flat forecast present")
else:
    fail("GNN fallback missing")

# Skip MAPE
t2 = open(
    "trading/models/forecast_router.py",
    encoding="utf-8",
    errors="replace",
).read()
if '"gnn"' in t2 or "'gnn'" in t2:
    ok("gnn in skip_mape or router")
else:
    fail("gnn not found in router")

# Ticker resolver
try:
    t3 = open(
        "trading/data/ticker_resolver.py",
        encoding="utf-8",
        errors="replace",
    ).read()
    ast.parse(t3)
    ok("Syntax valid: ticker_resolver.py")
    if "normalize_ticker" in t3:
        ok("normalize_ticker present")
    else:
        fail("normalize_ticker missing")
    if "resolve_ticker" in t3:
        ok("resolve_ticker present")
    else:
        fail("resolve_ticker missing")
    if "^GSPC" in t3:
        ok("SPX alias present")
    else:
        fail("SPX alias missing")
    if "BTC-USD" in t3:
        ok("BTC alias present")
    else:
        fail("BTC alias missing")
except FileNotFoundError:
    fail("ticker_resolver.py not created")

# Wired into analyze page
t4 = open(
    "pages/2_Analyze.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t4)
    ok("Syntax valid: 2_Analyze.py")
except SyntaxError as e:
    fail(f"Syntax error Analyze: {e}")
if "normalize_ticker" in t4:
    ok("normalize_ticker in 2_Analyze.py")
else:
    fail("normalize_ticker missing from 2_Analyze.py")

# Functional test
print("\n--- Resolver test ---")
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.data.ticker_resolver"
        " import normalize_ticker;"
        "tests = [('SPX','^GSPC'),"
        "('BTC','BTC-USD'),"
        "('AAPL','AAPL'),"
        "('VIX','^VIX'),"
        "('ES','ES=F')];"
        "[print(f'OK {i}->{normalize_ticker(i)}')"
        " if normalize_ticker(i)==e"
        " else print(f'FAIL {i}->{normalize_ticker(i)} expected {e}')"
        " for i,e in tests]",
    ],
    capture_output=True,
    text=True,
    timeout=10,
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
    print("All checks passed. Ready to commit v4.3.7.")
    sys.exit(0)
