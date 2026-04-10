import ast
import os
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 72 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — agent_logger IF NOT EXISTS
ta = open(
    "trading/memory/agent_logger.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ta)
    ok("Syntax valid: agent_logger.py")
except SyntaxError as e:
    fail(f"Syntax error agent_logger: {e}")

_bare = re.findall(
    r"CREATE TABLE(?!\s+IF)",
    ta,
    re.IGNORECASE,
)
if _bare:
    fail(
        f"agent_logger still has {len(_bare)} bare CREATE TABLE"
    )
else:
    ok("agent_logger CREATE TABLE uses IF NOT EXISTS")

# Fix 2 — theme.py no components.html
tt = open(
    "components/theme.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tt)
    ok("Syntax valid: theme.py")
except SyntaxError as e:
    fail(f"Syntax error theme: {e}")

if "components.html(" in tt:
    fail("components.html still called in theme.py")
else:
    ok("components.html removed from theme.py")

# Fix 3 — app.py startup guard
tap = open(
    "app.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tap)
    ok("Syntax valid: app.py")
except SyntaxError as e:
    fail(f"Syntax error app.py: {e}")

if "_app_initialized" in tap:
    ok("Startup log guard present")
else:
    fail("Startup log guard missing")

# Fix 4a — button state pattern
td = open(
    "pages/1_Dashboard.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(td)
    ok("Syntax valid: 1_Dashboard.py")
except SyntaxError as e:
    fail(f"Syntax error Dashboard: {e}")

if "_briefing_requested" in td:
    ok("Briefing button state pattern present")
else:
    fail("Briefing button state missing")

# Fix 4b — phase 2 removed, no cloud split in briefing
tb = open(
    "agents/briefing/morning_briefing.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tb)
    ok("Syntax valid: morning_briefing.py")
except SyntaxError as e:
    fail(f"Syntax error briefing: {e}")

_phase2 = re.search(
    r"scan_market\(\s*filters=\[\].*"
    r"universe=_pre_tickers",
    tb,
    re.DOTALL,
)
if _phase2:
    fail("Phase 2 full AI Score still present in briefing")
else:
    ok("Phase 2 removed — briefing uses quick scores only")

if "quick_score" in tb:
    ok("quick_score used in briefing")
else:
    fail("quick_score missing from briefing candidates")

if "_on_cloud" in tb:
    fail("Unnecessary _on_cloud detection still present")
else:
    ok("No _on_cloud in briefing")

# Smoke test
print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
_out = result.stdout + result.stderr
_passes = _out.count("PASS:")
_fails = _out.count("FAIL:")
print(f"Smoke: {_passes} PASS, {_fails} FAIL")
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.6.3.")
    sys.exit(0)
