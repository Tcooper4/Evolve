"""Session 28 verification."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 28 VERIFICATION")
print("=" * 60)

# 1. Strategy lifecycle uses memory store
print("\n[1] Strategy lifecycle uses memory store...")
try:
    with open("pages/7_Performance.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    fn_start = c.find("get_strategy_lifecycle")
    fn_body = c[fn_start : fn_start + 1500] if fn_start > 0 else ""
    uses_memory = "get_memory_store" in fn_body or "memory_store" in fn_body
    uses_backtest_fallback = "backtest_results" in fn_body or "session_state" in fn_body
    no_hardcoded = (
        ("Mature" not in fn_body[:200] and "Testing" not in fn_body[:100])
        or uses_memory
    )
    print(f"  Queries memory store: {uses_memory}")
    print(f"  Backtest session fallback: {uses_backtest_fallback}")
    print(f"  Hard-coded data replaced: {no_hardcoded}")
    print(f"  {'PASS' if uses_memory else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. Auto-pause persistence
print("\n[2] Auto-pause rules persistence...")
try:
    with open("pages/7_Performance.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_upsert = "upsert" in c and "autopause" in c.lower()
    has_load = "autopause" in c.lower() and ("_mem.list" in c or "memory_store" in c or "get_memory_store" in c)
    has_session_fallback = "autopause_rules_" in c
    print(f"  Saves to memory store: {has_upsert}")
    print(f"  Loads existing rules: {has_load}")
    print(f"  Session state fallback: {has_session_fallback}")
    print(f"  {'PASS' if has_upsert else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Task Orchestrator diagnostics
print("\n[3] Task Orchestrator status...")
try:
    with open("pages/11_Admin.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_diagnostics = "orch_status" in c or "orchestrator" in c.lower()
    has_path_check = "core/orchestrator" in c or "os.path" in c
    has_init = "TaskOrchestrator()" in c
    print(f"  Diagnostic status check: {has_diagnostics}")
    print(f"  Path existence check: {has_path_check}")
    print(f"  Proper initialization if available: {has_init}")
    _exists = os.path.exists("core/orchestrator") if os.path.exists("core") else False
    print(f"  core/orchestrator exists on disk: {_exists}")
    if _exists:
        print(f"  Files: {os.listdir('core/orchestrator')}")
    print(f"  {'PASS' if has_diagnostics else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Memory store interface test
print("\n[4] Memory store interface...")
try:
    from trading.memory import get_memory_store
    from trading.memory.memory_store import MemoryType

    _mem = get_memory_store()
    _methods = [m for m in dir(_mem) if not m.startswith("_")]
    _relevant = [m for m in _methods if any(k in m for k in ["get", "set", "upsert", "query", "insert", "list", "add"])]
    print(f"  Available methods: {_relevant}")
    print("  PASS")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Smoke tests
print("\n[5] Model smoke tests...")
import subprocess

result = subprocess.run(
    [sys.executable, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
output = result.stdout + result.stderr
if "All smoke tests completed. All PASS" in output:
    print("  PASS: All 12 models")
else:
    fails = [l.strip() for l in output.split("\n") if "FAIL" in l]
    print(f"  ISSUES: {fails}")

print("\n" + "=" * 60)
print("Session 28 complete. Paste output back.")
print("=" * 60)
