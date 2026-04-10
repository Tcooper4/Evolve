"""Session 61 verification: LLM no-key handling, price_cache syntax, smoke test."""
import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 61 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — _NoKeyError defined
t = open(
    "config/llm_config.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: llm_config.py")
except SyntaxError as e:
    fail(f"Syntax error llm_config: {e}")

if "_NoKeyError" in t:
    ok("_NoKeyError defined in llm_config")
else:
    fail("_NoKeyError missing")

if "llm_available" in t or os.path.exists("config/llm_status.py"):
    ok("llm_available() helper exists")
else:
    fail("llm_available() missing")

# Fix 2 — active_llm_calls intercepts
t2 = open(
    "agents/llm/active_llm_calls.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: active_llm_calls.py")
except SyntaxError as e:
    fail(f"Syntax error llm_calls: {e}")

if "_NoKeyError" in t2:
    ok("_NoKeyError caught in active_llm_calls")
else:
    fail("_NoKeyError not caught in calls")

# Fix 1+2 functional — no stack trace when keys absent
_nokey_check = r"""
import os
import sys
import importlib

sys.path.insert(0, ".")
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)

import config.llm_config as lc
lc._config = None
importlib.reload(lc)

from config.llm_config import llm_available

r = llm_available()
print("llm_available():", r)
assert r is False, "Expected False with no env keys"
print("No-key check: OK")
"""
_r = subprocess.run(
    [python, "-c", _nokey_check],
    capture_output=True,
    text=True,
    timeout=30,
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if _r.returncode == 0 and "No-key check: OK" in _r.stdout:
    ok("llm_available() returns False when no keys present")
else:
    fail(
        f"llm_available() check failed: "
        f"{(_r.stdout + _r.stderr)[:400]}"
    )

# Fix 3 — require_llm exists
_found_require = (
    "require_llm" in t
    or (
        os.path.exists("config/llm_status.py")
        and "require_llm"
        in open(
            "config/llm_status.py",
            encoding="utf-8",
            errors="replace",
        ).read()
    )
)
if _found_require:
    ok("require_llm() helper exists")
else:
    fail("require_llm() missing")

# Fix 4 — syntax check price_cache
if os.path.exists("trading/data/price_cache.py"):
    try:
        ast.parse(
            open(
                "trading/data/price_cache.py",
                encoding="utf-8",
                errors="replace",
            ).read()
        )
        ok("Syntax valid: price_cache.py")
    except SyntaxError as e:
        fail(f"Syntax error price_cache: {e}")

# Spot-check key files
for fpath in [
    "config/llm_config.py",
    "agents/llm/active_llm_calls.py",
    "pages/7_Settings.py",
]:
    try:
        ast.parse(
            open(fpath, encoding="utf-8", errors="replace").read()
        )
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
)
print((result.stdout + result.stderr)[-800:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(
    f"\n=== {len(PASS)} passed, "
    f"{len(FAIL)} failed ==="
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. "
        "Ready to commit v4.5.4."
    )
    sys.exit(0)
