"""Session 66 verification: Scanner + market_scanner RSI, smoke."""
import ast
import os
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 66 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)

t = open(
    "pages/3_Scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: 3_Scanner.py")
except SyntaxError as e:
    fail(f"Syntax error Scanner: {e}")

if "streaming_pipeline" in t:
    fail("Dead streaming_pipeline still present")
else:
    ok("Dead streaming code removed")

if "_batch_short_floats" in t:
    ok("Batch short float present")
else:
    fail("Batch short float missing")

_loop_call = re.search(
    r"for r in results.*?" r"_get_short_float",
    t,
    re.DOTALL,
)
if _loop_call:
    fail("Per-row _get_short_float still in results loop")
else:
    ok("Per-row short float removed from loop")

if "scanner_streaming_mode" in t and "run_every" in t:
    ok("Fragment conditional on streaming_mode")
else:
    fail("Fragment not conditional")

tm = open(
    "trading/analysis/market_scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tm)
    ok("Syntax valid: market_scanner.py")
except SyntaxError as e:
    fail(f"Syntax error scanner: {e}")

if (
    "Wilder" in tm
    or (
        "ag * (period - 1)" in tm
        and "al * (period - 1)" in tm
    )
):
    ok("Wilder smoothing RSI present")
else:
    fail("Wilder RSI missing")

_r = subprocess.run(
    [
        python,
        "-c",
        "import sys,numpy as np;"
        "sys.path.insert(0,'.');"
        "from trading.analysis.market_scanner import _rsi;"
        "prices=np.array([float(i) for i in range(1,52)]);"
        "v=_rsi(prices,14);"
        "print(f'RSI flat up: {v:.1f}');"
        "assert v>90,'RSI flat up <90';"
        "prices2=np.array([float(50-i) for i in range(51)]);"
        "v2=_rsi(prices2,14);"
        "print(f'RSI flat down: {v2:.1f}');"
        "assert v2<10,'RSI flat down >10';"
        "print('RSI test OK')",
    ],
    capture_output=True,
    text=True,
    timeout=30,
    cwd=_root,
)
print(_r.stdout[:200])
if _r.returncode == 0 and "OK" in _r.stdout:
    ok("RSI Wilder functional test passed")
else:
    fail(f"RSI test failed: {(_r.stdout + _r.stderr)[:150]}")

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
    print("All checks passed. Ready to commit v4.5.9.")
    sys.exit(0)
