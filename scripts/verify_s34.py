"""Session 34 — HIGH severity bug fixes verification."""
import sys, os
sys.path.insert(0, '.')

print("="*60)
print("SESSION 34 VERIFICATION")
print("="*60)

# 1. CCXT flag
print("\n[1] CCXT_AVAILABLE flag on ImportError...")
try:
    c = open('trading/execution/execution_engine.py',
             encoding='utf-8', errors='replace').read()
    # Find the ImportError block
    import_block = c[c.find('ccxt'):c.find('ccxt')+300] if 'ccxt' in c else ''
    correct = 'CCXT_AVAILABLE = False' in c
    wrong = ('CCXT_AVAILABLE = True' in import_block and
             'ImportError' in import_block[:import_block.find('CCXT_AVAILABLE = True')+50])
    print(f"  CCXT_AVAILABLE = False on ImportError: {correct}")
    print(f"  Inversion bug still present: {wrong}")
    print(f"  {'PASS' if correct and not wrong else 'NEEDS FIX'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. Hardcoded secret
print("\n[2] Hardcoded secret key removed...")
try:
    c = open('trading/services/agent_api_service.py',
             encoding='utf-8', errors='replace').read()
    hardcoded = 'your-secret-key-change-in-production' in c
    uses_env = 'os.environ' in c or 'os.getenv' in c
    uses_secrets = 'secrets.token' in c or 'token_hex' in c
    print(f"  Hardcoded string still present: {hardcoded}")
    print(f"  Uses env var: {uses_env}")
    print(f"  Uses secrets module: {uses_secrets}")
    print(f"  {'PASS' if not hardcoded else 'STILL HARDCODED'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Backtester.run() exists
print("\n[3] Backtester.run() method exists...")
try:
    from trading.backtesting.backtester import Backtester
    has_run = hasattr(Backtester, 'run')
    has_execute = hasattr(Backtester, 'execute') or hasattr(Backtester, 'run_backtest')
    print(f"  Backtester.run() exists: {has_run}")
    print(f"  Other run method: {has_execute}")
    if has_run:
        import inspect
        sig = inspect.signature(Backtester.run)
        print(f"  Signature: {sig}")
    print(f"  {'PASS' if has_run else 'MISSING — backtests will fail'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. execution_engine result KeyError fixed
print("\n[4] execution_engine result key safety...")
try:
    c = open('trading/execution/execution_engine.py',
             encoding='utf-8', errors='replace').read()
    # Find get_execution_summary
    fn_start = c.find('get_execution_summary')
    fn_body = c[fn_start:fn_start+800] if fn_start > 0 else ''
    has_safe_get = '.get("result"' in fn_body or 'result = ex.get' in fn_body
    has_unsafe = '["result"]["' in fn_body
    print(f"  Safe .get() for result key: {has_safe_get}")
    print(f"  Unsafe direct access remaining: {has_unsafe}")
    print(f"  {'PASS' if has_safe_get and not has_unsafe else 'NEEDS FIX'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Division zero guards
print("\n[5] Division by zero guards...")
checks = [
    ('trading/report/report_generator.py', 'running_max',
     ['where(running_max', 'running_max != 0', 'np.where']),
    ('trading/backtesting/backtester.py', 'initial_cash',
     ['initial_cash > 0', 'if self.initial_cash']),
    ('trading/agents/strategy_selector_agent.py', 'negative_returns',
     ['negative_returns != 0', 'if negative_returns']),
]
for fpath, context, patterns in checks:
    try:
        c = open(fpath, encoding='utf-8', errors='replace').read()
        guarded = any(p in c for p in patterns)
        print(f"  {fpath.split('/')[-1]} ({context}): {'PASS' if guarded else 'MISSING GUARD'}")
    except Exception as e:
        print(f"  {fpath}: ERROR {e}")

# 6. exec() guard in safe_executor
print("\n[6] exec() call chain guard...")
try:
    c = open('trading/utils/safe_executor.py',
             encoding='utf-8', errors='replace').read()
    has_exec = 'exec(' in c
    has_guard = ('assert' in c or 'isinstance' in c) and 'model_code' in c
    has_comment = 'internal' in c.lower() or 'trusted' in c.lower() or 'pipeline' in c.lower()
    print(f"  exec() present: {has_exec}")
    print(f"  Guard/assertion added: {has_guard}")
    print(f"  Trust comment added: {has_comment}")
    print(f"  {'PASS' if has_guard or has_comment else 'NEEDS DOCUMENTATION/GUARD'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 7. SQLite context manager in history_logger
print("\n[7] SQLite context manager in history_logger...")
try:
    c = open('trading/core/history_logger.py',
             encoding='utf-8', errors='replace').read()
    with_count = c.count('with sqlite3.connect')
    raw_count = c.count('= sqlite3.connect')
    print(f"  'with sqlite3.connect' usages: {with_count}")
    print(f"  Raw 'sqlite3.connect' assignments: {raw_count}")
    print(f"  {'PASS' if with_count >= 2 and raw_count == 0 else 'CHECK — ' + str(raw_count) + ' raw connections remain'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 8. Smoke tests
print("\n[8] Model smoke tests...")
import subprocess
result = subprocess.run([sys.executable, 'tests/model_smoke_test.py'],
    capture_output=True, text=True)
output = result.stdout + result.stderr
if 'All smoke tests completed. All PASS' in output:
    print("  PASS: All 12 models")
else:
    fails = [l.strip() for l in output.split('\n') if 'FAIL' in l]
    print(f"  ISSUES: {fails}")

print("\n" + "="*60)
print("Session 34 complete. Paste output back.")
print("="*60)
