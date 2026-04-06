import ast, subprocess, sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 33 Verification ===\n")
PASS, FAIL = [], []
def ok(msg):   PASS.append(msg); print(f"OK    {msg}")
def fail(msg): FAIL.append(msg); print(f"FAIL  {msg}")

# Fix 1 - stan_backend probe removed
t = open(
    "trading/models/prophet_model.py",
    encoding="utf-8", errors="replace"
).read()
if "stan_backend" not in t:
    ok("stan_backend probe removed")
else:
    fail("stan_backend probe still present")

if "_unavailable" in t:
    ok("_unavailable flag still present")
else:
    fail("_unavailable flag missing")

# Syntax checks
for fpath in [
    "trading/models/prophet_model.py",
    "trading/models/forecast_router.py",
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
          "Ready to commit v4.2.0.")
    sys.exit(0)
