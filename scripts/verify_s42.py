import ast
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 42 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# 13F institutional ownership
t = open(
    "trading/data/sec_edgar.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: sec_edgar.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

if "get_institutional_ownership" in t:
    ok("get_institutional_ownership present")
else:
    fail("get_institutional_ownership missing")

if "get_earnings_transcript_sentiment" in t:
    ok("get_earnings_transcript_sentiment present")
else:
    fail("transcript sentiment missing")

if "_SEC_CACHE" in t:
    ok("_SEC_CACHE present")
else:
    fail("_SEC_CACHE missing")

# AI score wiring
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

if "institutional_ownership" in t2:
    ok("institutional_ownership in SIGNAL_SOURCES")
else:
    fail("institutional_ownership missing")

if "_fetch_institutional_safe" in t2:
    ok("Institutional parallel worker")
else:
    fail("Institutional worker missing")

mw = re.search(r"max_workers\s*=\s*(\d+)", t2)
if mw and int(mw.group(1)) >= 9:
    ok(f"max_workers={mw.group(1)}")
else:
    fail("max_workers not updated to 9")

# Functional test
print("\n--- Institutional ownership test ---")
_r = subprocess.run(
    [
        python,
        "-c",
        (
            "import sys; sys.path.insert(0,'.');"
            "from trading.data.sec_edgar import "
            "get_institutional_ownership;"
            "r=get_institutional_ownership('AAPL');"
            "print('Holders:', r.get('n_holders'));"
            "print('Inst%:', r.get('institutional_pct'));"
            "print('Signal:', r.get('signal'))"
        ),
    ],
    capture_output=True,
    text=True,
    timeout=30,
)
print(_r.stdout[:300])
if _r.returncode == 0:
    ok("Institutional ownership functional")
else:
    fail(f"Error: {_r.stderr[:200]}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-800:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.3.4.")
    sys.exit(0)
