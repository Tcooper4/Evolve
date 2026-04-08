import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 35 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


files = {
    "components/tabs/tab_market_analysis.py": [
        "get_history",
        "analyze_forecast_data",
        "self-fetch or _gh",
    ],
    "components/tabs/tab_monte_carlo.py": [
        "get_history",
        "analyze_forecast_data",
        "self-fetch or _gh",
    ],
}

for fpath, checks in files.items():
    t = open(fpath, encoding="utf-8", errors="replace").read()
    try:
        ast.parse(t)
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

    if "get_history" in t:
        ok(f"Self-fetch present: {fpath}")
    else:
        fail(f"Self-fetch missing: {fpath}")

    if "st.warning" not in t:
        ok(f"Warning replaced: {fpath}")
    else:
        fail(f"st.warning still present: {fpath}")

# Diagnostics should be unchanged
t3 = open(
    "components/tabs/tab_diagnostics.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t3)
    ok("Syntax valid: tab_diagnostics.py")
except SyntaxError as e:
    fail(f"Syntax error tab_diagnostics: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-1500:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.2.5.")
    sys.exit(0)
