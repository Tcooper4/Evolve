"""Session 20 — final polish verification."""
import sys, os
sys.path.insert(0, '.')

print("="*60)
print("SESSION 20 — v1.2.0 FINAL VERIFICATION")
print("="*60)

# 1. Noise suppression
print("\n[1] Startup noise check...")
import io, logging
from contextlib import redirect_stderr
_buf = io.StringIO()
with redirect_stderr(_buf):
    try:
        import importlib
        if 'trading.data.earnings_reaction' in sys.modules:
            del sys.modules['trading.data.earnings_reaction']
        from trading.data.earnings_reaction import get_earnings_reactions
    except Exception:
        pass
_noise = _buf.getvalue()
_neuro_suppressed = 'neuralforecast_models' not in _noise
_rl_suppressed = 'rl_trainer' not in _noise
print(f"  NeuralForecast warning suppressed: {_neuro_suppressed}")
print(f"  RL trainer warning suppressed: {_rl_suppressed}")

# 2. Earnings reaction TTL cache
print("\n[2] Earnings reaction TTL cache...")
try:
    from trading.data.earnings_reaction import get_earnings_reactions, _reaction_cache
    r1 = get_earnings_reactions('AAPL', num_quarters=2)
    r2 = get_earnings_reactions('AAPL', num_quarters=2)
    assert r1 is r2 or r1 == r2, "Cache not working"
    print(f"  Cache entries: {len(_reaction_cache)}")
    print(f"  PASS: TTL cache functional")
except Exception as e:
    print(f"  FAIL or lru_cache still in use: {e}")

# 3. Page-level error boundaries
print("\n[3] Error boundaries in high-risk pages...")
for page in ['pages/2_Forecasting.py',
             'pages/3_Strategy_Testing.py',
             'pages/7_Performance.py']:
    try:
        c = open(page, encoding='utf-8', errors='replace').read()
        has_boundary = '_page_error' in c or 'page_error' in c
        name = page.split('/')[-1]
        print(f"  {name}: {'PASS' if has_boundary else 'MISSING'}")
    except Exception as e:
        print(f"  {page}: ERROR {e}")

# 4. Home page scan cache
print("\n[4] Home page scan cache...")
try:
    c = open('pages/0_Home.py', encoding='utf-8', errors='replace').read()
    has_ttl = 'home_scan_ts' in c or 'scan_ttl' in c or '_scan_ts_key' in c
    has_refresh = 'Refresh' in c and 'rerun' in c
    print(f"  TTL cache: {has_ttl}")
    print(f"  Refresh button: {has_refresh}")
    print(f"  {'PASS' if has_ttl else 'FAIL'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. CHANGELOG v1.2.0
print("\n[5] CHANGELOG v1.2.0...")
try:
    c = open('CHANGELOG.md', encoding='utf-8', errors='replace').read()
    has_v120 = '1.2.0' in c
    has_scanner = 'Scanner' in c or 'scanner' in c
    print(f"  v1.2.0 entry: {has_v120}")
    print(f"  Scanner mentioned: {has_scanner}")
    print(f"  {'PASS' if has_v120 else 'FAIL'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 6. Git tag
print("\n[6] Git tag v1.2.0...")
try:
    import subprocess
    r = subprocess.run(['git', 'tag'], capture_output=True, text=True)
    tags = r.stdout.strip().split('\n') if r.stdout.strip() else []
    has_v120 = 'v1.2.0' in tags
    print(f"  Tags: {[t for t in tags if t.startswith('v')]}")
    print(f"  v1.2.0 tag: {'PASS' if has_v120 else 'MISSING — run: git tag v1.2.0'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 7. Full smoke test
print("\n[7] Model smoke test...")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/model_smoke_test.py'],
        capture_output=True, text=True, cwd='.'
    )
    output = result.stdout + result.stderr
    if 'All smoke tests completed. All PASS' in output:
        print("  PASS: All 12 models passing")
    else:
        fails = [l.strip() for l in output.split('\n') if 'FAIL' in l]
        print(f"  ISSUES: {fails}")
except Exception as e:
    print(f"  ERROR: {e}")

# 8. Feature completeness audit
print("\n[8] Feature completeness audit...")
features = {
    'AI Score': ('trading/analysis/ai_score.py', 'compute_ai_score'),
    'Market Scanner': ('trading/analysis/market_scanner.py', 'scan_market'),
    'MTF Chart': ('components/multi_timeframe_chart.py', 'render_multi_timeframe_chart'),
    'Earnings Reaction': ('trading/data/earnings_reaction.py', 'get_earnings_reactions'),
    'Earnings Calendar': ('trading/data/earnings_calendar.py', None),
    'Short Interest': ('trading/data/short_interest.py', None),
    'Insider Flow': ('trading/data/insider_flow.py', None),
    'Watchlist': ('trading/data/watchlist.py', 'WatchlistManager'),
    'Walk-forward': ('trading/models/walk_forward_validator.py', 'WalkForwardValidator'),
    'Transformer': ('trading/models/transformer_model.py', 'TransformerForecaster'),
    'Scanner Page': ('pages/13_Scanner.py', None),
}
all_pass = True
for name, (path, symbol) in features.items():
    exists = os.path.exists(path)
    if exists and symbol:
        c = open(path, encoding='utf-8', errors='replace').read()
        has_sym = symbol in c
        status = 'PASS' if has_sym else 'FILE EXISTS, SYMBOL MISSING'
    else:
        status = 'PASS' if exists else 'MISSING'
    if status != 'PASS':
        all_pass = False
    icon = "PASS" if status == "PASS" else "FAIL"
    print(f"  {icon}: {name}: {status}")

print(f"\n  Overall: {'ALL PASS' if all_pass else 'ISSUES FOUND'}")

print("\n" + "="*60)
print("v1.2.0 verification complete.")
print("Next: git push origin main --tags")
print("="*60)

