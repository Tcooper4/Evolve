# Session 57 verification — briefing cache invalidation vs saved prefs
import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 57 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Check preference fingerprint exists
t = open(
    "pages/1_Dashboard.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: 1_Dashboard.py")
except SyntaxError as e:
    fail(f"Syntax error Dashboard: {e}")

if (
    "home_briefing_prefs" in t
    or "pref_hash" in t
    or "_pref_" in t
):
    ok("Preference fingerprint present")
else:
    fail("Preference fingerprint missing")

# Check morning_briefing.py
t2 = open(
    "agents/briefing/morning_briefing.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: morning_briefing.py")
except SyntaxError as e:
    fail(f"Syntax error briefing: {e}")

# Check settings keys match
t3 = open(
    "pages/7_Settings.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t3)
    ok("Syntax valid: 7_Settings.py")
except SyntaxError as e:
    fail(f"Syntax error Settings: {e}")

# Saved pref keys (user_store), not st.session_state briefing_min_score
for key in [
    "briefing_universe",
    "min_ai_score",
]:
    in_dashboard = key in t
    in_settings = key in t3
    in_briefing = key in t2
    if in_dashboard or in_briefing:
        ok(f"Pref key '{key}' referenced")
    else:
        fail(
            f"Pref key '{key}' not found "
            f"in Dashboard or briefing"
        )

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

print(
    f"\n=== {len(PASS)} passed, "
    f"{len(FAIL)} failed ==="
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. "
        "Ready to commit v4.5.0."
    )
    sys.exit(0)
