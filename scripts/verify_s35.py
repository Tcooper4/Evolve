"""Session 35 — MEDIUM severity sweep verification."""
import sys, os, glob
sys.path.insert(0, '.')

print("="*60)
print("SESSION 35 VERIFICATION")
print("="*60)

# 1. st.set_page_config removed from non-page modules
print("\n[1] st.set_page_config in non-page modules...")
bad_files = []
for fpath in ['trading/agents/leaderboard_dashboard.py',
              'trading/ui/institutional_dashboard.py']:
    try:
        c = open(fpath, encoding='utf-8', errors='replace').read()
        if 'st.set_page_config' in c:
            bad_files.append(fpath)
    except FileNotFoundError:
        pass
print(f"  Files still calling set_page_config: {bad_files}")
print(f"  {'PASS' if not bad_files else 'NEEDS FIX'}")

# 2. create_strategy lru_cache fixed
print("\n[2] create_strategy lru_cache with **kwargs...")
try:
    c = open('trading/strategies/__init__.py',
             encoding='utf-8', errors='replace').read()
    fn_start = c.find('def create_strategy')
    context = c[max(0, fn_start-200):fn_start+100] if fn_start > 0 else ''
    has_lru_on_kwargs = '@lru_cache' in context and '**kwargs' in c[fn_start:fn_start+100]
    has_safe_cache = '_create_strategy_cached' in c or '@lru_cache' not in context
    print(f"  Unsafe lru_cache on **kwargs: {has_lru_on_kwargs}")
    print(f"  Safe cache pattern or removed: {has_safe_cache}")
    print(f"  {'PASS' if not has_lru_on_kwargs else 'STILL UNSAFE'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. user_store decrypt safety
print("\n[3] user_store decrypt try/except...")
try:
    c = open('config/user_store.py', encoding='utf-8', errors='replace').read()
    decrypt_idx = c.find('cipher.decrypt')
    context = c[decrypt_idx:decrypt_idx+200] if decrypt_idx > 0 else ''
    has_try = 'try:' in c[max(0,decrypt_idx-100):decrypt_idx+200]
    print(f"  cipher.decrypt wrapped in try/except: {has_try}")
    print(f"  {'PASS' if has_try else 'NEEDS FIX'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Admin bare except replaced
print("\n[4] Admin bare except: pass replaced...")
try:
    c = open('pages/11_Admin.py', encoding='utf-8', errors='replace').read()
    bare_pass = c.count('except Exception: pass') + c.count('except Exception:\n            pass') + c.count('except Exception:\n        pass')
    has_logging = 'logger.debug' in c or 'logger.warning' in c
    print(f"  Bare 'except Exception: pass' remaining: {bare_pass}")
    print(f"  Has logging in except blocks: {has_logging}")
    print(f"  {'PASS' if bare_pass == 0 else str(bare_pass) + ' remaining'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. iloc safety — spot check key files
print("\n[5] iloc safety spot checks...")
spot_checks = [
    ('trading/backtesting/backtest_utils.py', ['iloc[-1]', 'iloc[0]']),
    ('trading/market/market_analyzer.py', ['iloc[-1]']),
    ('pages/5_Portfolio.py', ['iloc[-1]', 'iloc[0]']),
    ('pages/7_Performance.py', ['iloc[0]']),
]
for fpath, patterns in spot_checks:
    try:
        c = open(fpath, encoding='utf-8', errors='replace').read()
        # Count unguarded iloc (not preceded by 'if' or 'len' within 120 chars)
        import re
        unguarded = 0
        for pat in patterns:
            for m in re.finditer(re.escape(pat), c):
                preceding = c[max(0, m.start()-120):m.start()]
                if not any(g in preceding for g in
                           ['if len', 'if not', '.empty', '> 0', '>0', 'empty()']):
                    unguarded += 1
        fname = fpath.split('/')[-1]
        print(f"  {fname}: ~{unguarded} potentially unguarded iloc calls")
    except Exception as e:
        print(f"  {fpath}: ERROR {e}")

# 6. Smoke tests
print("\n[6] Model smoke tests...")
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
print("Session 35 complete. Paste output back.")
print("="*60)
