import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 75 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — GPR in macro_factors
tm = open(
    "trading/analysis/macro_factors.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tm)
    ok("Syntax valid: macro_factors.py")
except SyntaxError as e:
    fail(f"Syntax error macro: {e}")

for check in [
    "_get_gpr_index",
    "GPR_URL",
    "GPR_CACHE_PATH",
    "geopolitical",
    "Caldara",
    "Geopolitical Risk",
]:
    if check in tm:
        ok(f"GPR: '{check}' present")
    else:
        fail(f"GPR: '{check}' missing")

_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.'); "
        "from trading.analysis.macro_factors import MacroFactors; "
        "mf = MacroFactors(); r = mf._get_gpr_index(); "
        "print('level:', r.get('level')); "
        "print('current:', r.get('current')); "
        "assert 'level' in r; print('GPR OK')",
    ],
    capture_output=True,
    text=True,
    timeout=90,
)
print((_r.stdout + _r.stderr)[:400])
if _r.returncode == 0 and "OK" in _r.stdout:
    ok("GPR functional test passed")
else:
    fail(f"GPR test failed: {(_r.stdout + _r.stderr)[:250]}")

# Fix 2 — revision breadth
te = open(
    "trading/data/earnings_quality.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(te)
    ok("Syntax valid: earnings_quality.py")
except SyntaxError as e:
    fail(f"Syntax error earnings: {e}")

for check in [
    "get_revision_breadth",
    "_BREADTH_CACHE",
    "_BREADTH_TTL",
    "pct_up",
    "ThreadPoolExecutor",
    "_neutral_breadth",
]:
    if check in te:
        ok(f"Breadth: '{check}' present")
    else:
        fail(f"Breadth: '{check}' missing")

_r2 = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.'); "
        "from trading.data.earnings_quality import get_revision_breadth; "
        "print('Import OK')",
    ],
    capture_output=True,
    text=True,
    timeout=20,
)
if _r2.returncode == 0 and "OK" in _r2.stdout:
    ok("get_revision_breadth importable")
else:
    fail(f"Breadth import failed: {(_r2.stdout + _r2.stderr)[:150]}")

# Fix 3 — Quick Score improvements
ts = open(
    "trading/analysis/market_scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ts)
    ok("Syntax valid: market_scanner.py")
except SyntaxError as e:
    fail(f"Syntax error scanner: {e}")

for check in ["vs_sma50", "vol_expansion", "sma50"]:
    if check in ts:
        ok(f"Quick Score: '{check}' present")
    else:
        fail(f"Quick Score: '{check}' missing")

_r3 = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.'); "
        "from trading.analysis.market_scanner import _quick_score; "
        "import inspect; "
        "sig = str(inspect.signature(_quick_score)); "
        "print('sig:', sig); "
        "assert 'vs_sma50' in sig; assert 'vol_expansion' in sig; "
        "s = _quick_score(45, 8.0, 3.0, 1.8, -3.0, vs_sma50=4.0, "
        "vol_expansion=1.3); "
        "print(f'Strong: {s}'); assert s >= 7.0; print('Quick Score OK')",
    ],
    capture_output=True,
    text=True,
    timeout=20,
)
print((_r3.stdout + _r3.stderr)[:400])
if _r3.returncode == 0 and "OK" in _r3.stdout:
    ok("Quick Score 7-component test passed")
else:
    fail(f"Quick Score test failed: {(_r3.stdout + _r3.stderr)[:200]}")

# Fix 4 — Dashboard
td = open(
    "pages/1_Dashboard.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(td)
    ok("Syntax valid: 1_Dashboard.py")
except SyntaxError as e:
    fail(f"Syntax error Dashboard: {e}")

if "get_revision_breadth" in td:
    ok("Revision breadth in Dashboard")
else:
    fail("Revision breadth missing from Dashboard")

if "_get_gpr_index" in td or "Geopolitical Risk" in td:
    ok("GPR display in Dashboard")
else:
    fail("GPR missing from Dashboard")

if "_rb_computing" in td:
    ok("Non-blocking breadth compute")
else:
    fail("Breadth blocking Dashboard")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
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
    print("All checks passed. Ready to commit v4.6.6.")
    sys.exit(0)
