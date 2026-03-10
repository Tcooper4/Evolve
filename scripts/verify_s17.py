"""Session 17 verification."""
import sys
sys.path.insert(0, '.')
import os

print("="*60)
print("SESSION 17 VERIFICATION")
print("="*60)

# 1. Multi-timeframe chart component
print("\n[1] Multi-timeframe chart component...")
if os.path.exists('components/multi_timeframe_chart.py'):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mtf", "components/multi_timeframe_chart.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        has_fn = hasattr(mod, 'render_multi_timeframe_chart')
        print(f"  File exists: True")
        print(f"  render_multi_timeframe_chart: {has_fn}")
        print(f"  {'PASS' if has_fn else 'FAIL'}")
    except Exception as e:
        print(f"  Import check failed: {e}")
else:
    print("  FAIL: components/multi_timeframe_chart.py not found")

# 2. Forecasting page — tabs now show errors
print("\n[2] Forecasting page error visibility...")
try:
    with open('pages/2_Forecasting.py', encoding='utf-8', errors='replace') as f:
        c = f.read()
    has_traceback = 'traceback.format_exc' in c or 'st.exception' in c
    has_mtf = 'multi_timeframe_chart' in c or 'render_multi_timeframe_chart' in c
    has_ai_score = 'compute_ai_score' in c or 'ai_score' in c
    print(f"  Error visibility (traceback/exception): {has_traceback}")
    print(f"  Multi-timeframe chart wired: {has_mtf}")
    print(f"  AI Score wired: {has_ai_score}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Home page scanner
print("\n[3] Home page Top Opportunities...")
try:
    with open('pages/0_Home.py', encoding='utf-8', errors='replace') as f:
        c = f.read()
    has_scan = 'scan_market' in c
    has_opps = 'Top Opportunities' in c or 'top_ai_score' in c
    print(f"  scan_market wired: {has_scan}")
    print(f"  Opportunities section: {has_opps}")
    print(f"  {'PASS' if has_scan else 'FAIL'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Scanner functional test
print("\n[4] Scanner functional test...")
try:
    from trading.analysis.market_scanner import scan_market
    r = scan_market(filters=["momentum"], universe=["AAPL", "MSFT", "NVDA"], max_results=5)
    assert r.get('error') is None
    print(f"  Scanned: {r['scanned']}, Passed: {r['passed']}, Time: {r['scan_time_s']}s")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

# 5. AI Score multi-symbol
print("\n[5] AI Score multi-symbol test...")
try:
    from trading.analysis.ai_score import compute_ai_score
    import yfinance as yf
    for sym in ['AAPL', 'SPY']:
        h = yf.Ticker(sym).history(period='3mo')
        r = compute_ai_score(sym, h)
        status = 'PASS' if r['error'] is None else f"FAIL: {r['error']}"
        print(f"  {sym}: {r['overall_score']}/10 ({r['grade']}) — {status}")
except Exception as e:
    print(f"  FAIL: {e}")

# 6. Smoke test
print("\n[6] Model smoke test...")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, 'tests/model_smoke_test.py'],
        capture_output=True, text=True, cwd='.'
    )
    output = result.stdout + result.stderr
    if 'All smoke tests completed. All PASS' in output:
        print("  PASS: All models passing")
    else:
        fails = [l for l in output.split('\n') if 'FAIL' in l]
        print(f"  Issues: {fails[:3] if fails else 'check manually'}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*60)
print("Done. Run: .\\evolve_venv\\Scripts\\python.exe scripts\\verify_s17.py")
