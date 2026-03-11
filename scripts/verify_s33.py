"""Session 33 verification."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 33 VERIFICATION")
print("=" * 60)

# 1. Orchestrator init speed
print("\n[1] Orchestrator init speed...")
import io
import time
from contextlib import redirect_stderr, redirect_stdout

_buf = io.StringIO()
_t0 = time.time()
with redirect_stdout(_buf), redirect_stderr(_buf):
    try:
        for mod in list(sys.modules.keys()):
            if any(x in mod for x in ["core", "trading", "agents"]):
                del sys.modules[mod]
        from core.orchestrator.task_orchestrator import TaskOrchestrator

        _orch = TaskOrchestrator()
    except Exception as _e:
        print(f"  IMPORT ERROR: {_e}")
_elapsed = time.time() - _t0
_output = _buf.getvalue()
_errors = [l for l in _output.split("\n") if "Failed" in l or "Error" in l]
print(f"  Init time: {_elapsed:.2f}s")
print(f"  Errors during init: {len(_errors)}")
print(f"  Has start() method: {hasattr(_orch, 'start')}")
print(
    f"  {'PASS' if _elapsed < 2.0 else 'SLOW — ' + str(round(_elapsed, 1)) + 's (target: <2s)'}"
)

# 2. Startup noise still clean
print("\n[2] Startup noise still clean...")
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
_info_lines = [
    l
    for l in _buf2.getvalue().split("\n")
    if ("INFO" in l or l.strip().startswith("{")) and l.strip()
]
print(f"  INFO/JSON lines: {len(_info_lines)}")
print(
    f"  {'PASS' if len(_info_lines) == 0 else 'REGRESSED — ' + str(len(_info_lines)) + ' lines'}"
)

# 3. CHANGELOG v1.3.2
print("\n[3] CHANGELOG v1.3.2...")
try:
    c = open("CHANGELOG.md", encoding="utf-8", errors="replace").read()
    has_v132 = "1.3.2" in c
    has_entropy = "entropy" in c or "session_id" in c
    print(f"  v1.3.2 entry: {has_v132}")
    print(f"  entropy/session_id fix mentioned: {has_entropy}")
    print(f"  {'PASS' if has_v132 else 'MISSING'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Git tag v1.3.2
print("\n[4] Git tags...")
import subprocess

r = subprocess.run(["git", "tag"], capture_output=True, text=True)
tags = [t for t in r.stdout.strip().split("\n") if t.startswith("v")]
print(f"  Tags: {tags}")
print(f"  v1.3.2: {'PASS' if 'v1.3.2' in tags else 'MISSING'}")

# 5. Smoke tests
print("\n[5] Model smoke tests...")
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
print("Session 33 complete. Paste output back.")
print("=" * 60)
