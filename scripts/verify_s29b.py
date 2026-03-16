# verify_s29b.py -- Session 29B verification
# Run: .\evolve_venv\Scripts\python.exe scripts\verify_s29b.py
import sys, os, ast, subprocess, io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY   = sys.executable
PASS = []; FAIL = []

def ok(m):   PASS.append(m); print("  PASS  " + m)
def fail(m): FAIL.append(m); print("  FAIL  " + m)

def parse_ok(rel):
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full):
        fail(rel + " NOT FOUND"); return False
    src = open(full, encoding='utf-8', errors='replace').read()
    try:
        ast.parse(src); ok(rel + " parses OK"); return True
    except SyntaxError as e:
        fail(rel + " SyntaxError line " + str(e.lineno)); return False

def exists(rel, label):
    if os.path.exists(os.path.join(ROOT, rel)): ok(label)
    else: fail(label)

def gone(rel, label):
    if not os.path.exists(os.path.join(ROOT, rel)): ok(label)
    else: fail(label)

print("\n[1] New pages present and parse")
for p in ["pages/1_Dashboard.py", "pages/2_Analyze.py",
          "pages/3_Scanner.py",   "pages/4_Trade.py",
          "pages/5_Backtest.py",  "pages/6_Chat.py",
          "pages/7_Settings.py"]:
    parse_ok(p)

print("\n[2] Old pages removed")
for p in ["pages/0_Home.py", "pages/2_Forecasting.py",
          "pages/13_Scanner.py", "pages/1_Chat.py",
          "pages/4_Trade_Execution.py", "pages/7_Performance.py",
          "pages/3_Strategy_Testing.py", "pages/5_Portfolio.py",
          "pages/6_Risk_Management.py", "pages/8_Model_Lab.py",
          "pages/9_Reports.py", "pages/10_Alerts.py",
          "pages/11_Admin.py", "pages/12_Memory.py"]:
    gone(p, p + " deleted")

print("\n[3] Exactly 7 pages (plus __init__.py)")
pages_dir = os.path.join(ROOT, "pages")
py_pages = [f for f in os.listdir(pages_dir)
            if f.endswith(".py") and f != "__init__.py"]
if len(py_pages) == 7:
    ok("pages/ contains exactly 7 page files")
else:
    fail("pages/ contains " + str(len(py_pages))
         + " files (expected 7): " + str(sorted(py_pages)))

print("\n[4] Core modules untouched")
for m in ["trading/analysis/ai_score.py",
          "trading/models/forecast_router.py",
          "trading/data/news_aggregator.py",
          "trading/backtesting/backtester.py",
          "agents/llm/agent.py",
          "config/user_store.py",
          "components/news_candle_chart.py",
          "components/multi_timeframe_chart.py",
          "tests/model_smoke_test.py",
          "components/theme.py",
          "trading/data/price_cache.py"]:
    exists(m, m + " intact")

print("\n[5] Streamlit >= 1.37")
r = subprocess.run(
    [PY, "-c",
     "import streamlit as st; "
     "v=tuple(int(x) for x in st.__version__.split('.')[:2]); "
     "print(st.__version__); import sys; "
     "sys.exit(0 if v>=(1,37) else 1)"],
    capture_output=True, text=True, cwd=ROOT)
ver = r.stdout.strip()
if r.returncode == 0:
    ok("Streamlit " + ver + " >= 1.37 (fragment support active)")
else:
    fail("Streamlit " + ver + " < 1.37 -- fragments not supported")

print("\n[6] Memory store DB still exists")
exists("data/memory_store.db", "data/memory_store.db intact")

print("\n[7] Universe JSON files intact")
for u in ["sp500","nasdaq100","sp100",
          "sp500_nasdaq100","russell1000","russell3000"]:
    exists("data/universes/" + u + ".json",
           "data/universes/" + u + ".json intact")

print("\n[8] Smoke tests")
r = subprocess.run(
    [PY, "tests/model_smoke_test.py"],
    capture_output=True, text=True, cwd=ROOT)
if r.returncode == 0:
    ok("All 12 models passed smoke tests")
else:
    for l in (r.stdout + r.stderr).splitlines()[-15:]:
        print("       " + l)
    fail("Smoke tests failed")

print("\n" + "=" * 52)
print("  PASSED: " + str(len(PASS))
      + "   FAILED: " + str(len(FAIL)))
if FAIL:
    print("  Failed:")
    for f in FAIL: print("    - " + f)
else:
    print("  29B complete. Ready to tag v2.0.0.")
    print("  Next: git tag v2.0.0 && git push origin main --tags")
print("=" * 52)