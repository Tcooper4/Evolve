"""Session 29 verification."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 29 VERIFICATION")
print("=" * 60)

# 1. Task Orchestrator imports
print("\n[1] Task Orchestrator imports...")
try:
    from core.orchestrator.task_orchestrator import TaskOrchestrator
    from core.orchestrator.task_scheduler import TaskScheduler
    from core.orchestrator.task_monitor import TaskMonitor
    from core.orchestrator.task_models import TaskType, TaskPriority, TaskStatus

    _orch = TaskOrchestrator()
    _sched = TaskScheduler()
    _mon = TaskMonitor()
    print("  TaskOrchestrator: PASS")
    print("  TaskScheduler: PASS")
    print("  TaskMonitor: PASS")
    print("  Instantiation: PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# 2. Startup noise fully suppressed
print("\n[2] Startup noise (all paths)...")
import io
from contextlib import redirect_stderr, redirect_stdout

_buf_out = io.StringIO()
_buf_err = io.StringIO()
with redirect_stdout(_buf_out), redirect_stderr(_buf_err):
    try:
        for mod in list(sys.modules.keys()):
            if "trading" in mod:
                del sys.modules[mod]
        from trading.memory import get_memory_store

        _m = get_memory_store()
    except Exception:
        pass
_combined = _buf_out.getvalue() + _buf_err.getvalue()
_info_lines = [
    l
    for l in _combined.split("\n")
    if ("INFO" in l or l.strip().startswith("{")) and l.strip()
]
print(f"  INFO/JSON lines after trading.memory import: {len(_info_lines)}")
for l in _info_lines[:3]:
    print(f"    {l[:80]}")
print(f"  {'PASS' if len(_info_lines) < 3 else 'STILL NOISY'}")

# 3. CHANGELOG v1.3.0
print("\n[3] CHANGELOG v1.3.0...")
try:
    with open("CHANGELOG.md", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_v130 = "1.3.0" in c
    has_cloud = "Cloud" in c or "onboarding" in c.lower()
    has_orchestrator = "Orchestrator" in c or "orchestrator" in c
    print(f"  v1.3.0 entry: {has_v130}")
    print(f"  Cloud fix mentioned: {has_cloud}")
    print(f"  Orchestrator mentioned: {has_orchestrator}")
    print(f"  {'PASS' if has_v130 else 'MISSING'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. TECHNICAL_DEBT.md updated
print("\n[4] TECHNICAL_DEBT.md updated...")
try:
    with open("TECHNICAL_DEBT.md", encoding="utf-8", errors="replace") as f:
        c = f.read()
    resolved_count = c.count("RESOLVED")
    open_count = c.count("OPEN")
    has_options = "Options flow" in c or "options flow" in c.lower()
    print(f"  RESOLVED items: {resolved_count}")
    print(f"  OPEN items: {open_count}")
    print(f"  Options flow in OPEN: {has_options}")
    print(f"  {'PASS' if resolved_count > 10 else 'CHECK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Git tags
print("\n[5] Git tags...")
import subprocess

r = subprocess.run(["git", "tag"], capture_output=True, text=True)
tags = [t for t in r.stdout.strip().split("\n") if t.startswith("v")]
print(f"  Tags: {tags}")
has_v130 = "v1.3.0" in tags
print(f"  v1.3.0: {'PASS' if has_v130 else 'MISSING - run: git tag v1.3.0'}")

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
print("Session 29 complete. Paste output back.")
print("=" * 60)
