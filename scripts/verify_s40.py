import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 40 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 - portfolio optimizer
t = open(
    "trading/optimization/portfolio_optimizer.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: portfolio_optimizer")
except SyntaxError as e:
    fail(f"Syntax error optimizer: {e}")
if "_w = np.array" in t or "w.value" in t:
    ok("Portfolio optimizer fix present")
else:
    fail("Portfolio optimizer fix missing")

# Fix 2 - relative valuation
t2 = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error ai_score: {e}")
if "SECTOR_VALUATION_MEDIANS" in t2:
    ok("Relative valuation medians present")
else:
    fail("SECTOR_VALUATION_MEDIANS missing")
if "forwardPE" in t2 or "forward_pe" in t2:
    ok("Forward P/E check present")
else:
    fail("Forward P/E missing")

# Fix 3 - sector rotation module
try:
    t3 = open(
        "trading/analysis/sector_rotation.py",
        encoding="utf-8",
        errors="replace",
    ).read()
    ast.parse(t3)
    ok("Syntax valid: sector_rotation.py")
except FileNotFoundError:
    fail("sector_rotation.py not created")
except SyntaxError as e:
    fail(f"Syntax error sector_rotation: {e}")

if "sector_rotation" in t2:
    ok("sector_rotation in SIGNAL_SOURCES or wired in ai_score")
else:
    fail("sector_rotation not wired")

# Functional test
print("\n--- Sector rotation test ---")
try:
    _r = subprocess.run(
        [
            python,
            "-c",
            "import sys; sys.path.insert(0,'.');"
            "from trading.analysis.sector_rotation import get_sector_rotation; "
            "r=get_sector_rotation(); "
            "print('Sectors found:', len(r.get('sectors',{})));"
            "print('Top:', r.get('top_sectors',[]))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(_r.stdout[:500])
    if _r.returncode == 0:
        ok("Sector rotation functional")
    else:
        fail(f"Sector rotation error: {_r.stderr[:200]}")
except Exception as _te:
    fail(f"Test failed: {_te}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-1000:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.3.2.")
    sys.exit(0)
