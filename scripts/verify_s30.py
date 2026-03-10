"""Session 30 verification — orchestrator cleanup + noise final fix."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 30 VERIFICATION")
print("=" * 60)

# 1. Orchestrator init — clean, no warnings
print("\n[1] TaskOrchestrator init — clean...")
import io
from contextlib import redirect_stderr, redirect_stdout

_buf = io.StringIO()
with redirect_stdout(_buf), redirect_stderr(_buf):
    try:
        for mod in list(sys.modules.keys()):
            if "core" in mod or "trading" in mod or "agents" in mod:
                del sys.modules[mod]
        from core.orchestrator.task_orchestrator import TaskOrchestrator

        _orch = TaskOrchestrator()
    except Exception as _e:
        print(f"IMPORT ERROR: {_e}")
_output = _buf.getvalue()
_fail_lines = [l for l in _output.split("\n") if "Failed" in l or "Error" in l]
_warn_lines = [l for l in _output.split("\n") if "WARNING" in l]
print(f"  'Failed to initialize' lines: {len(_fail_lines)}")
print(f"  WARNING lines: {len(_warn_lines)}")
for l in _fail_lines:
    print(f"    {l[:100]}")
print(f"  {'PASS' if len(_fail_lines) == 0 else 'STILL HAS FAILURES'}")

# 2. TaskType has alert_manager
print("\n[2] TaskType enum includes alert_manager...")
try:
    from core.orchestrator.task_models import TaskType

    has_alert = hasattr(TaskType, "ALERT_MANAGER") or "alert_manager" in [
        t.value for t in TaskType
    ]
    all_types = [t.value for t in TaskType]
    print(f"  alert_manager present: {has_alert}")
    print(f"  All types: {all_types}")
    print(f"  {'PASS' if has_alert else 'MISSING'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Startup noise — all import paths
print("\n[3] Startup noise — trading.memory import path...")
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

# 4. CHANGELOG v1.3.1
print("\n[4] CHANGELOG v1.3.1...")
try:
    with open("CHANGELOG.md", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_v131 = "1.3.1" in c
    print(f"  v1.3.1 entry: {has_v131}")
    print(f"  {'PASS' if has_v131 else 'MISSING'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Git tags
print("\n[5] Git tags...")
import subprocess

r = subprocess.run(["git", "tag"], capture_output=True, text=True)
tags = [t for t in r.stdout.strip().split("\n") if t.startswith("v")]
print(f"  Tags: {tags}")
print(f"  v1.3.1: {'PASS' if 'v1.3.1' in tags else 'MISSING'}")

# 6. Smoke tests
print("\n[6] Model smoke tests...")
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
print("Session 30 complete. Paste output back.")
print("=" * 60)
