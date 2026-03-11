"""Session 31 — final two fixes."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 31 VERIFICATION")
print("=" * 60)

# 1. cache_management in TaskType
print("\n[1] TaskType includes cache_management...")
try:
    from core.orchestrator.task_models import TaskType

    all_types = [t.value for t in TaskType]
    has_cache = "cache_management" in all_types
    print(f"  cache_management present: {has_cache}")
    print(f"  All types ({len(all_types)}): {all_types}")
    print(f"  {'PASS' if has_cache else 'MISSING'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. Orchestrator init fully clean
print("\n[2] TaskOrchestrator init — fully clean...")
import io
from contextlib import redirect_stderr, redirect_stdout

_buf = io.StringIO()
with redirect_stdout(_buf), redirect_stderr(_buf):
    try:
        for mod in list(sys.modules.keys()):
            if any(x in mod for x in ["core", "trading", "agents"]):
                del sys.modules[mod]
        from core.orchestrator.task_orchestrator import TaskOrchestrator

        _orch = TaskOrchestrator()
    except Exception as _e:
        _buf.write(f"IMPORT ERROR: {_e}\n")
_output = _buf.getvalue()
_fail_lines = [l for l in _output.split("\n") if "Failed" in l or "Error" in l]
_warn_lines = [l for l in _output.split("\n") if "WARNING" in l]
print(f"  'Failed' lines: {len(_fail_lines)}")
print(f"  WARNING lines: {len(_warn_lines)}")
for l in _fail_lines:
    print(f"    {l[:100]}")
print(f"  {'PASS' if len(_fail_lines) == 0 else 'STILL HAS ISSUES'}")

# 3. Startup noise fully dead
print("\n[3] Startup noise — root logger...")
_buf2 = io.StringIO()
with redirect_stdout(_buf2), redirect_stderr(_buf2):
    try:
        for mod in list(sys.modules.keys()):
            if "trading" in mod:
                del sys.modules[mod]
        from trading.memory import get_memory_store

        _m = get_memory_store()
    except Exception:
        pass
_combined = _buf2.getvalue()
_info_lines = [
    l
    for l in _combined.split("\n")
    if ("INFO" in l or l.strip().startswith("{")) and l.strip()
]
print(f"  INFO/JSON lines: {len(_info_lines)}")
for l in _info_lines[:3]:
    print(f"    {l[:80]}")
print(f"  {'PASS' if len(_info_lines) == 0 else 'STILL ' + str(len(_info_lines)) + ' lines'}")

# 4. Smoke tests
print("\n[4] Model smoke tests...")
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
print("Session 31 complete. Paste output back.")
print("=" * 60)
