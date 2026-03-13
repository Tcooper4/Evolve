# verify_s28b.py -- Session 28 verification
# Run: .\evolve_venv\Scripts\python.exe scripts\verify_s28b.py
import sys, os, ast, json, subprocess

# Force UTF-8 stdout to avoid cp1252 crashes on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY   = sys.executable
PASS = []
FAIL = []

def ok(msg):   PASS.append(msg); print("  PASS  " + msg)
def fail(msg): FAIL.append(msg); print("  FAIL  " + msg)

print("")
print("[1] Universe JSON files")
universes = {
    "sp500": 400,
    "nasdaq100": 90,
    "sp100": 90,
    "sp500_nasdaq100": 400,
    "russell1000": 900,
    "russell3000": 2000,
}
for name, min_count in universes.items():
    path = os.path.join(ROOT, "data", "universes", name + ".json")
    if not os.path.exists(path):
        fail("data/universes/" + name + ".json MISSING")
        continue
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= min_count:
            ok(name + ".json -- " + str(len(data)) + " tickers")
        else:
            fail(name + ".json -- only " + str(len(data)) + " tickers (want >= " + str(min_count) + ")")
    except Exception as e:
        fail(name + ".json -- load error: " + str(e))

print("")
print("[2] Forecasting.py syntax check")
fcast_path = os.path.join(ROOT, "pages", "2_Forecasting.py")
if not os.path.exists(fcast_path):
    fail("pages/2_Forecasting.py not found")
else:
    try:
        with open(fcast_path, encoding="utf-8", errors="replace") as f:
            src = f.read()
        ast.parse(src)
        ok("pages/2_Forecasting.py parses OK")
    except SyntaxError as e:
        fail("SyntaxError in 2_Forecasting.py line " + str(e.lineno) + ": " + str(e.msg))

print("")
print("[3] Forecasting.py -- no duplicate except block")
if os.path.exists(fcast_path):
    try:
        with open(fcast_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        bad = [str(i+1) for i, l in enumerate(lines)
               if l.strip().startswith("except") and "_pp_err" in l]
        if not bad:
            ok("No stray _pp_err except clause found")
        else:
            fail("Stray _pp_err except at lines: " + ", ".join(bad))
    except Exception as e:
        fail("Could not read 2_Forecasting.py: " + str(e))

print("")
print("[4] Universe loader wired in pages")
for page in ["pages/0_Home.py", "pages/13_Scanner.py"]:
    path = os.path.join(ROOT, page)
    if not os.path.exists(path):
        fail(page + " not found")
        continue
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    if "universes" in src:
        ok(page + " -- references data/universes")
    else:
        fail(page + " -- no reference to data/universes (not wired)")

print("")
print("[5] Model smoke tests")
result = subprocess.run(
    [PY, "tests/model_smoke_test.py"],
    capture_output=True, text=True, cwd=ROOT
)
if result.returncode == 0:
    ok("All models passed smoke tests")
else:
    output_lines = (result.stdout + result.stderr).strip().splitlines()
    for l in output_lines[-15:]:
        print("       " + l)
    fail("Smoke tests failed -- see above")

print("")
print("=" * 50)
print("  PASSED: " + str(len(PASS)) + "   FAILED: " + str(len(FAIL)))
if FAIL:
    print("")
    print("Failed checks:")
    for f in FAIL:
        print("  - " + f)
else:
    print("  All checks passed. Ready to tag v1.9.0.")
print("=" * 50)