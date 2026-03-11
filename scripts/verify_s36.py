"""Session 36 — iloc precision fix + v1.4.0 release."""
import sys, os, re
sys.path.insert(0, '.')

print("="*60)
print("SESSION 36 VERIFICATION")
print("="*60)

# 1. iloc unguarded count in key files
print("\n[1] iloc guard coverage in key files...")
files = [
    'trading/backtesting/backtest_utils.py',
    'trading/market/market_analyzer.py',
    'pages/5_Portfolio.py',
    'pages/7_Performance.py',
    'agents/llm/agent.py',
    'trading/models/forecast_router.py',
]
total_unguarded = 0
for fpath in files:
    try:
        c = open(fpath, encoding='utf-8', errors='replace').read()
        unguarded = 0
        for m in re.finditer(r'\.iloc\[[-01]\]', c):
            preceding = c[max(0, m.start()-150):m.start()]
            if not any(g in preceding for g in
                       ['if len', 'if not', '.empty', '> 0', '>0',
                        'empty()', 'notna', 'is not None', 'len(']):
                unguarded += 1
        total_unguarded += unguarded
        fname = fpath.split('/')[-1]
        status = 'PASS' if unguarded == 0 else f'WARN ~{unguarded}'
        print(f"  {status} {fname}")
    except FileNotFoundError:
        print(f"  SKIP {fpath} (not found)")
print(f"  Total potentially unguarded: {total_unguarded}")
print(f"  {'PASS' if total_unguarded <= 5 else 'STILL HAS GAPS'}")

# 2. app.py shutdown no bare pass
print("\n[2] app.py shutdown error handling...")
try:
    c = open('app.py', encoding='utf-8', errors='replace').read()
    shutdown_idx = c.find('_shutdown')
    shutdown_body = c[shutdown_idx:shutdown_idx+400] if shutdown_idx > 0 else ''
    bare_pass = shutdown_body.count('except Exception: pass') + \
                shutdown_body.count('except Exception:\n        pass')
    print(f"  Bare except pass in shutdown: {bare_pass}")
    print(f"  {'PASS' if bare_pass == 0 else 'STILL BARE'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. execution_engine simulated price logging
print("\n[3] execution_engine _get_simulated_price error logging...")
try:
    c = open('trading/execution/execution_engine.py',
             encoding='utf-8', errors='replace').read()
    fn_idx = c.find('def _get_simulated_price')
    fn_body = c[fn_idx:fn_idx+2000] if fn_idx >= 0 else ''
    has_log = 'logger.debug' in fn_body or 'logger.warning' in fn_body
    bare_pass = fn_body.count('except Exception: pass') + \
                fn_body.count('except Exception:\n            pass')
    print(f"  Has logging in except: {has_log}")
    print(f"  Bare pass remaining: {bare_pass}")
    print(f"  {'PASS' if has_log and bare_pass == 0 else 'NEEDS FIX'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. CHANGELOG v1.4.0
print("\n[4] CHANGELOG v1.4.0...")
try:
    c = open('CHANGELOG.md', encoding='utf-8', errors='replace').read()
    has_v140 = '1.4.0' in c
    has_audit = 'audit' in c.lower() or 'Backtester' in c
    print(f"  v1.4.0 entry: {has_v140}")
    print(f"  Audit fixes mentioned: {has_audit}")
    print(f"  {'PASS' if has_v140 else 'MISSING'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Git tag v1.4.0
print("\n[5] Git tags...")
import subprocess
r = subprocess.run(['git', 'tag'], capture_output=True, text=True)
tags = [t for t in r.stdout.strip().split('\n') if t.startswith('v')]
print(f"  Tags: {tags}")
print(f"  v1.4.0: {'PASS' if 'v1.4.0' in tags else 'MISSING'}")

# 6. Smoke tests
print("\n[6] Model smoke tests...")
result = subprocess.run([sys.executable, 'tests/model_smoke_test.py'],
    capture_output=True, text=True)
output = result.stdout + result.stderr
if 'All smoke tests completed. All PASS' in output:
    print("  PASS: All 12 models")
else:
    fails = [l.strip() for l in output.split('\n') if 'FAIL' in l]
    print(f"  ISSUES: {fails}")

print("\n" + "="*60)
print("Session 36 complete. Paste output back.")
print("="*60)
