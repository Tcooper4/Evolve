import ast, subprocess, sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 31 Verification ===\n")
PASS, FAIL = [], []
def ok(msg):   PASS.append(msg); print(f"OK    {msg}")
def fail(msg): FAIL.append(msg); print(f"FAIL  {msg}")

t = open(
    "trading/analysis/market_scanner.py",
    encoding="utf-8", errors="replace"
).read()
if 'phase="filter"' in t:
    ok("filter phase progress in scanner")
else:
    fail("filter phase progress missing")
if 'phase="ai"' in t:
    ok("ai phase progress in scanner")
else:
    fail("ai phase progress missing")

t2 = open("pages/3_Scanner.py",
    encoding="utf-8", errors="replace"
).read()
if "Filtering universe" in t2:
    ok("filter phase text in Scanner page")
else:
    fail("filter phase text missing")
if "AI scoring" in t2:
    ok("AI scoring text in Scanner page")
else:
    fail("AI scoring text missing")

for fpath in [
    "trading/analysis/market_scanner.py",
    "pages/3_Scanner.py",
]:
    try:
        ast.parse(open(fpath,
            encoding="utf-8",
            errors="replace").read())
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True, text=True
)
print((result.stdout + result.stderr)[-1500:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, "
      f"{len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. "
          "Ready to commit v4.1.8.")
    sys.exit(0)
