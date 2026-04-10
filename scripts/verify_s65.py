"""Session 65 verification: SQLite context managers, smoke test."""
import ast
import os
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 65 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)


def check_file(path, label):
    try:
        t = open(path, encoding="utf-8", errors="replace").read()
        ast.parse(t)
        ok(f"Syntax valid: {label}")
        return t
    except SyntaxError as e:
        fail(f"Syntax error {label}: {e}")
        return ""


t1 = check_file("trading/data/options_flow.py", "options_flow.py")
_raw = re.findall(
    r"(?<!with )conn\s*=\s*sqlite3" r"\.connect",
    t1,
)
if _raw:
    fail(
        f"options_flow.py still has {len(_raw)} raw connect(s) outside with"
    )
else:
    ok("options_flow.py uses context managers only")

if "conn.close()" in t1:
    fail("options_flow.py still has explicit conn.close()")
else:
    ok("No explicit conn.close() in options_flow.py")

t2 = check_file("trading/data/watchlist.py", "watchlist.py")
_raw2 = len(re.findall(r"conn\.close\(\)", t2))
if _raw2 > 0:
    fail(f"watchlist.py still has {_raw2} explicit conn.close()")
else:
    ok("No explicit conn.close() in watchlist.py")

t3 = check_file("config/user_store.py", "user_store.py")
_lines = t3.splitlines()
_ctx = "\n".join(_lines[180:215] if len(_lines) > 215 else _lines)
if "with _get_conn()" in _ctx or "finally" in _ctx:
    ok("user_store.py line ~188 uses context manager or finally")
else:
    fail("user_store.py line ~188 still has raw connection")

for fpath, label in [
    ("trading/analysis/signal_score_store.py", "signal_score_store.py"),
    ("trading/memory/agent_logger.py", "agent_logger.py"),
    ("trading/memory/persistent_memory.py", "persistent_memory.py"),
]:
    check_file(fpath, label)

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
    cwd=_root,
    timeout=300000,
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
    print("All checks passed. Ready to commit v4.5.8.")
    sys.exit(0)
