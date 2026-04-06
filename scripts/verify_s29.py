import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 29 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


text = open(
    "pages/3_Scanner.py",
    encoding="utf-8",
    errors="replace",
).read()

if "_fmt_signals" in text:
    ok("signals normalization present")
else:
    fail("signals normalization missing")

if "isinstance(val, list)" in text:
    ok("list check present")
else:
    fail("list check missing")

try:
    ast.parse(text)
    ok("Syntax valid: pages/3_Scanner.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

# Check market_scanner unchanged
text2 = open(
    "trading/analysis/market_scanner.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(text2)
    ok("Syntax valid: market_scanner.py")
except SyntaxError as e:
    fail(f"Syntax error market_scanner: {e}")

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
    for f in FAIL:
        print(f"  FAIL  {f}")
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.1.6.")
    sys.exit(0)
