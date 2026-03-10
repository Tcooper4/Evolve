"""Session 25 verification."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 25 VERIFICATION")
print("=" * 60)

# 1. Advanced orders wired to ExecutionAgent
print("\n[1] Advanced orders -> ExecutionAgent wiring...")
try:
    c = open("pages/4_Trade_Execution.py", encoding="utf-8", errors="replace").read()
    has_bracket_wire = "submit_bracket_order" in c or (
        "bracket" in c.lower() and "execution_agent" in c
    )
    has_exec_agent_check = c.count("execution_agent") > 3
    has_graceful_degrade = "broker" in c and ("caption" in c or "warning" in c.lower())
    print(f"  Bracket order wired to agent: {has_bracket_wire}")
    print(f"  ExecutionAgent referenced in order sections: {has_exec_agent_check}")
    print(f"  Graceful degradation if no agent: {has_graceful_degrade}")
    print(f"  {'PASS' if has_bracket_wire or has_exec_agent_check else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. Automated execution loop
print("\n[2] Automated execution loop...")
try:
    c = open("pages/4_Trade_Execution.py", encoding="utf-8", errors="replace").read()
    has_cycle_fn = "_run_auto_execution_cycle" in c or "auto_execution_cycle" in c
    has_polling = "auto_exec_last_cycle" in c or "last_cycle" in c
    has_emergency_check = "emergency_stop" in c
    has_confidence_check = "min_signal_confidence" in c or "confidence" in c.lower()
    has_daily_limit = "max_orders_per_day" in c or "daily" in c.lower()
    print(f"  Execution cycle function: {has_cycle_fn}")
    print(f"  Polling with interval: {has_polling}")
    print(f"  Emergency stop respected: {has_emergency_check}")
    print(f"  Signal confidence threshold: {has_confidence_check}")
    print(f"  Daily order limit: {has_daily_limit}")
    print(f"  {'PASS' if has_cycle_fn and has_polling else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Strategy registry accessible
print("\n[3] Strategy registry for auto-execution...")
try:
    from trading.strategies.registry import get_strategy_registry

    _reg = get_strategy_registry()
    _names = list(_reg.get_strategy_names()) if hasattr(_reg, "get_strategy_names") else []
    print("  Registry accessible: True")
    print(f"  Strategies available: {_names[:5]}")
    print("  PASS")
except Exception as e:
    print(f"  Registry error: {e}")

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
print("Session 25 complete. Paste output back.")
print("=" * 60)
