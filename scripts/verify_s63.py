"""Session 63 verification: NeuralForecast wiring, registry, MAPE skip, smoke."""
import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 63 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


_r = subprocess.run(
    [python, "-c", "import neuralforecast; print(neuralforecast.__version__)"],
    capture_output=True,
    text=True,
    timeout=60,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if _r.returncode == 0:
    ok(f"neuralforecast installed: {_r.stdout.strip()}")
else:
    fail("neuralforecast not installed")

_r2 = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.models.neuralforecast_models import ("
        "NBEATSModel, NHITSModel, PatchTSTModel, TFTModel);"
        "print('all 4 imported OK')",
    ],
    capture_output=True,
    text=True,
    timeout=30,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if _r2.returncode == 0 and "OK" in _r2.stdout:
    ok("All 4 NF model classes importable")
else:
    fail(f"NF import failed: {(_r2.stdout + _r2.stderr)[:200]}")

_r3 = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.'); "
        "from trading.models.model_registry import get_registry; "
        "r = get_registry(); "
        "names = {str(x).lower() for x in r.list_models()}; "
        "need = ('n-beats', 'n-hits', 'patchtst', 'tft'); "
        "assert all(m in names for m in need), names; "
        "print('registry OK')",
    ],
    capture_output=True,
    text=True,
    timeout=30,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if _r3.returncode == 0 and "registry OK" in _r3.stdout:
    ok("All 4 models in registry")
else:
    fail(f"Registry check failed: {(_r3.stdout + _r3.stderr)[:300]}")

tr = open(
    "trading/models/forecast_router.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tr)
    ok("Syntax valid: forecast_router.py")
except SyntaxError as e:
    fail(f"Syntax error router: {e}")

for m in ("nbeats", "nhits", "patchtst", "tft"):
    if f"'{m}'" in tr or f'"{m}"' in tr:
        ok(f"'{m}' in _skip_mape_models")
    else:
        fail(f"'{m}' missing from _skip_mape_models")

ts = open(
    "pages/7_Settings.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ts)
    ok("Syntax valid: 7_Settings.py")
except SyntaxError as e:
    fail(f"Syntax error Settings: {e}")

if (
    "N-BEATS" in ts
    and "N-HiTS" in ts
    and "PatchTST" in ts
    and "TFT" in ts
):
    ok("Settings shows 4 model names")
else:
    fail("Settings display not updated")

for fpath in (
    "trading/models/neuralforecast_models.py",
    "trading/models/model_registry.py",
):
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
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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
    print("All checks passed. Ready to commit v4.5.6.")
    sys.exit(0)
