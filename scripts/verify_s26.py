"""
verify_s26.py — Session 26 Verification Script
Run with: .\evolve_venv\Scripts\python.exe scripts\verify_s26.py

Checks:
  Fix 1 — Wikipedia S&P 500 column resilience in market_scanner.py
  Fix 2 — Prophet stan_backend removed from prophet_model.py
  Fix 3 — forecast_explainability.py IndentationError resolved (both files)
  Fix 4 — LSTM scaler shape fix in lstm_model.py
  Runtime — LSTM end-to-end forecast produces price-space values
  Runtime — Smoke tests still pass (summary)
"""

import sys, os, re, subprocess
sys.path.insert(0, '.')

results = []

def check(name, passed, detail=''):
    results.append(passed)
    status = '[PASS]' if passed else '[FAIL]'
    msg = f"{status} {name}"
    if detail:
        msg += f"\n       {detail[:200]}"
    print(msg)

def read(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()

print("=" * 60)
print("SESSION 26 VERIFICATION")
print("=" * 60)

# ── FIX 1: Wikipedia column resilience ──────────────────────────
print("\n--- Fix 1: Universe scraping ---")
try:
    scanner = read('trading/analysis/market_scanner.py')
    # Should have resilient column lookup — either a loop or multiple column names
    has_symbol_col = "'Symbol'" in scanner or '"Symbol"' in scanner
    has_ticker_col = "'Ticker'" in scanner or '"Ticker"' in scanner
    has_fallback    = 'RUSSELL' in scanner.upper() and (
        'fallback' in scanner.lower() or 'AAPL' in scanner
    )
    check("S&P 500 scraper references 'Symbol' column",    has_symbol_col)
    check("S&P 500 scraper still handles 'Ticker' column", has_ticker_col,
          "Should support both for backward compatibility")
    check("Russell fallback list present in scanner",      has_fallback)
except FileNotFoundError:
    check("market_scanner.py readable", False, "File not found")

# ── FIX 2: Prophet stan_backend ─────────────────────────────────
print("\n--- Fix 2: Prophet stan_backend ---")
try:
    prophet = read('trading/models/prophet_model.py')
    has_stan = 'stan_backend' in prophet
    check("stan_backend removed from prophet_model.py", not has_stan,
          "Found 'stan_backend'" if has_stan else "")

    # Make sure Prophet() is still being instantiated somewhere
    has_prophet_init = 'Prophet(' in prophet
    check("Prophet() instantiation still present", has_prophet_init)
except FileNotFoundError:
    check("prophet_model.py readable", False, "File not found")

# ── FIX 3: forecast_explainability IndentationError ─────────────
print("\n--- Fix 3: forecast_explainability.py IndentationError ---")

def check_explainability_file(path):
    try:
        src = read(path)
        # The broken pattern: try: at end of line followed by non-indented import
        # Check it compiles cleanly
        try:
            compile(src, path, 'exec')
            check(f"{path} compiles without SyntaxError", True)
        except (SyntaxError, IndentationError) as e:
            check(f"{path} compiles without SyntaxError", False, str(e))

        # Should have proper try/except around shap import
        has_shap_try = bool(re.search(
            r'try\s*:\s*\n\s+import shap', src
        ))
        has_shap_available = 'SHAP_AVAILABLE' in src
        check(f"{path}: shap import in proper try block",
              has_shap_try or has_shap_available,
              "Need 'try:\\n    import shap' pattern")
    except FileNotFoundError:
        check(f"{path} readable", False, "File not found")

check_explainability_file('trading/analytics/forecast_explainability.py')
check_explainability_file('trading/models/forecast_explainability.py')

# ── FIX 4: LSTM scaler shape ─────────────────────────────────────
print("\n--- Fix 4: LSTM scaler shape fix ---")
try:
    lstm = read('trading/models/lstm_model.py')

    # Should have separate scalers for features and target
    has_feature_scaler = 'feature_scaler' in lstm
    has_target_scaler  = 'target_scaler' in lstm
    has_fit_transform_full = bool(re.search(
        r'fit_transform\(X\b', lstm
    ))

    check("LSTM has feature_scaler attribute",      has_feature_scaler)
    check("LSTM has target_scaler attribute",       has_target_scaler)
    check("LSTM scaler fit_transform called on X",  has_fit_transform_full,
          "Scaler should be applied to full X, not single column")

    # Make sure the old broken line is gone
    broken = "X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)"
    # This line itself isn't wrong if shape is correct — check the scaler setup instead
    # Just confirm no shape-1 scaler feeding into full-column DataFrame
    has_single_col_scale = bool(re.search(
        r"fit_transform\(X\[(?:target|'close'|\"close\")\]", lstm
    ))
    check("LSTM not scaling single column into multi-column DataFrame",
          not has_single_col_scale,
          "Found single-column fit_transform feeding into full DataFrame" if has_single_col_scale else "")

except FileNotFoundError:
    check("lstm_model.py readable", False, "File not found")

# ── RUNTIME: LSTM end-to-end ─────────────────────────────────────
print("\n--- Runtime: LSTM end-to-end forecast ---")
try:
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='B')
    last_close = 308.42
    data = pd.DataFrame({
        'open':   last_close + np.random.randn(300).cumsum(),
        'high':   last_close + 2 + np.random.randn(300).cumsum(),
        'low':    last_close - 2 + np.random.randn(300).cumsum(),
        'close':  last_close + np.random.randn(300).cumsum(),
        'volume': np.random.randint(1_000_000, 10_000_000, 300).astype(float),
    }, index=dates)
    last_close_actual = float(data['close'].iloc[-1])

    from trading.models.forecast_router import ForecastRouter
    router = ForecastRouter()
    result = router.get_forecast(data, model_type='lstm', horizon=7)
    vals = list(result.get('forecast', []))

    in_range = all(last_close_actual * 0.5 < v < last_close_actual * 1.5
                   for v in vals[:3]) if len(vals) >= 3 else False
    check(
        f"LSTM forecast returns price-space values (last_close={last_close_actual:.2f})",
        in_range,
        f"Got: {[round(v,2) for v in vals[:5]]}" if vals else "Empty forecast"
    )
    # Make sure it's NOT falling back to ARIMA (ARIMA result would be fine
    # but we want to confirm LSTM itself ran)
    model_used = result.get('model', '')
    check("LSTM did not fall back to ARIMA",
          'arima' not in model_used.lower(),
          f"Model used: {model_used}")

except Exception as e:
    check("LSTM runtime test", False, str(e))

# ── RUNTIME: Prophet import ──────────────────────────────────────
print("\n--- Runtime: Prophet model init ---")
try:
    from trading.models.prophet_model import ProphetModel
    import pandas as pd, numpy as np
    np.random.seed(1)
    dates = pd.date_range('2023-01-01', periods=300, freq='B')
    data  = pd.DataFrame({
        'close': 308.0 + np.random.randn(300).cumsum(),
        'volume': np.random.randint(1_000_000, 5_000_000, 300).astype(float),
    }, index=dates)
    m = ProphetModel()
    m.fit(data)
    result = m.forecast(data, horizon=7)
    vals = list(result.get('forecast', []))
    last_close = float(data['close'].iloc[-1])
    ok = all(last_close * 0.5 < v < last_close * 1.5 for v in vals[:3]) if vals else False
    check("ProphetModel initializes without stan_backend error", m.available if hasattr(m, 'available') else True)
    check("ProphetModel forecast returns price-space values", ok,
          f"Got: {[round(v,2) for v in vals[:5]]}" if vals else "Empty")
except Exception as e:
    check("ProphetModel runtime test", False, str(e))

# ── RUNTIME: Scanner universe count ─────────────────────────────
print("\n--- Runtime: Universe scraping ---")
try:
    from trading.analysis.market_scanner import MarketScanner
    scanner_obj = MarketScanner()
    # Try to get S&P 500 tickers
    get_fn = None
    for attr in ['get_sp500_tickers', '_get_sp500_tickers', 'get_universe',
                 '_get_universe_tickers', 'get_universe_tickers']:
        if hasattr(scanner_obj, attr):
            get_fn = getattr(scanner_obj, attr)
            break

    if get_fn:
        try:
            tickers = get_fn('S&P 500') if 'universe' in get_fn.__name__.lower() else get_fn()
            count = len(tickers) if tickers else 0
            check(f"S&P 500 universe returns >100 stocks (got {count})", count > 100)
        except Exception as e:
            check("S&P 500 universe fetch", False, str(e))
    else:
        check("MarketScanner has universe fetch method", False,
              f"None of the expected method names found. Available: {[a for a in dir(scanner_obj) if not a.startswith('__')][:10]}")
except Exception as e:
    check("MarketScanner import", False, str(e))

# ── SMOKE TEST SUMMARY ───────────────────────────────────────────
print("\n--- Smoke tests ---")
try:
    result = subprocess.run(
        [r'.\evolve_venv\Scripts\python.exe', 'tests/model_smoke_test.py'],
        capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace'
    )
    output = result.stdout + result.stderr
    for line in output.splitlines():
        if any(k in line.lower() for k in ['pass', 'fail', 'smoke', 'completed', 'error', 'exception']):
            print(f"  {line}")
    passed_count = output.lower().count('pass')
    check("Smoke tests: all 12 models pass", '12' in output and 'fail' not in output.lower(),
          f"Check output above")
except subprocess.TimeoutExpired:
    check("Smoke tests completed in time", False, "Timed out after 300s")
except Exception as e:
    check("Smoke tests ran", False, str(e))

# ── SUMMARY ─────────────────────────────────────────────────────
print()
print("=" * 60)
total  = len(results)
passed = sum(results)
failed = total - passed
print(f"RESULTS: {passed}/{total} PASS   {failed} FAIL")
if failed == 0:
    print("SESSION 26 COMPLETE — safe to commit and tag v1.8.0")
else:
    print("SESSION 26 INCOMPLETE — do not tag until all checks pass")
print("=" * 60)
