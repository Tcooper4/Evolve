import ast
import os
import sys

print("=== Session 37 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 + 2 - deep_dive gated
t = open(
    "components/deep_dive.py",
    encoding="utf-8",
    errors="replace",
).read()

try:
    ast.parse(t)
    ok("Syntax valid: deep_dive.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

if "dd_forecast_btn" in t:
    ok("Deep dive forecast button present")
else:
    fail("Deep dive forecast button missing")

if "deep_dive_forecast_" in t:
    ok("Deep dive forecast cache key present")
else:
    fail("Deep dive forecast cache missing")

# Fix 3 - cache file deleted
if not os.path.exists(".cache/_analyze_split_source.py"):
    ok("_analyze_split_source.py deleted")
else:
    fail("_analyze_split_source.py still exists")

# Syntax check analyze_ai_score
t2 = open(
    "components/analyze_ai_score.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: analyze_ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error analyze_ai_score: {e}")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.2.7.")
    sys.exit(0)
