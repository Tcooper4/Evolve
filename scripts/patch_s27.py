"""
patch_s27.py — Direct patches for S27 remaining issues
Run: .\evolve_venv\Scripts\python.exe scripts\patch_s27.py
"""
import re, py_compile

def read(path):
    return open(path, encoding='utf-8', errors='replace').read()

def write(path, content):
    open(path, 'w', encoding='utf-8').write(content)

def check_compile(path):
    try:
        py_compile.compile(path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

results = []

# ══════════════════════════════════════════════════════════════
# FIX 1 — pages/2_Forecasting.py
# The try: block starting at line 523 has no except clause.
# The postprocessor.process() call falls outside the try block.
# Fix: indent the entire postprocess+update block inside the try,
# and add an except clause after it.
# ══════════════════════════════════════════════════════════════
path = 'pages/2_Forecasting.py'
src = read(path)

OLD = '''                                try:
                                    from trading.forecasting.forecast_postprocessor import ForecastPostprocessor

                                    postprocessor = ForecastPostprocessor()

                                    # Extract forecast values (ARIMA uses 'forecast'; some use 'predictions'/'values'/'forecast_values')
                                    if isinstance(forecast_result, dict):
                                        forecast_vals = (
                                            forecast_result.get('forecast')
                                            or forecast_result.get('predictions')
                                            or forecast_result.get('values')
                                            or forecast_result.get('forecast_values')
                                            or []
                                        )
                                        if hasattr(forecast_vals, 'tolist'):
                                            forecast_vals = forecast_vals.tolist()
                                    else:
                                        forecast_vals = forecast_result
                                
                                # Postprocess forecast
                                processed_forecast = postprocessor.process(
                                    forecast=forecast_vals,
                                    historical_data=data,
                                    apply_smoothing=True,
                                    remove_outliers=True,
                                    ensure_realistic_bounds=True
                                )
                                
                                # Update forecast_result with processed version
                                if isinstance(forecast_result, dict):
                                    forecast_result['forecast'] = processed_forecast['values']
                                    forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                                else:
                                    forecast_result = processed_forecast['values']'''

NEW = '''                                try:
                                    from trading.forecasting.forecast_postprocessor import ForecastPostprocessor

                                    postprocessor = ForecastPostprocessor()

                                    # Extract forecast values (ARIMA uses 'forecast'; some use 'predictions'/'values'/'forecast_values')
                                    if isinstance(forecast_result, dict):
                                        forecast_vals = (
                                            forecast_result.get('forecast')
                                            or forecast_result.get('predictions')
                                            or forecast_result.get('values')
                                            or forecast_result.get('forecast_values')
                                            or []
                                        )
                                        if hasattr(forecast_vals, 'tolist'):
                                            forecast_vals = forecast_vals.tolist()
                                    else:
                                        forecast_vals = forecast_result

                                    # Postprocess forecast
                                    processed_forecast = postprocessor.process(
                                        forecast=forecast_vals,
                                        historical_data=data,
                                        apply_smoothing=True,
                                        remove_outliers=True,
                                        ensure_realistic_bounds=True
                                    )

                                    # Update forecast_result with processed version
                                    if isinstance(forecast_result, dict):
                                        forecast_result['forecast'] = processed_forecast['values']
                                        forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                                    else:
                                        forecast_result = processed_forecast['values']
                                except Exception as _pp_err:
                                    import logging
                                    logging.getLogger(__name__).warning(f"Postprocessing skipped: {_pp_err}")'''

if OLD in src:
    src = src.replace(OLD, NEW)
    write(path, src)
    ok, err = check_compile(path)
    print(f"[{'PASS' if ok else 'FAIL'}] Forecasting.py try-block fix {'applied' if ok else 'SYNTAX ERROR: ' + str(err)}")
    results.append(ok)
else:
    # The exact whitespace may differ — try a regex approach
    # Find the try: block and check if it's missing except
    ok, err = check_compile(path)
    if ok:
        print(f"[PASS] Forecasting.py already compiles — no patch needed")
        results.append(True)
    else:
        print(f"[FAIL] Forecasting.py OLD pattern not found and still has error: {err}")
        print("  Showing lines 519-560:")
        for i, l in enumerate(src.splitlines()[518:560], start=519):
            print(f"    {i:5}: {l}")
        results.append(False)

# ══════════════════════════════════════════════════════════════
# CHECK forecast_router.py
# ══════════════════════════════════════════════════════════════
path2 = 'trading/models/forecast_router.py'
src2 = read(path2)
ok2, err2 = check_compile(path2)
has_cache = '_MODEL_CACHE' in src2 or '_get_cached' in src2 or 'lru_cache' in src2 or 'cache_resource' in src2
print(f"\n[{'PASS' if ok2 else 'FAIL'}] forecast_router.py compiles: {ok2}")
print(f"[{'PASS' if has_cache else 'FAIL'}] forecast_router.py has caching: {has_cache}")
if not ok2:
    print(f"  Error: {err2}")
    # Show structure
    for i, l in enumerate(src2.splitlines()[:80], start=1):
        print(f"  {i:5}: {l}")
results.append(ok2)

# ══════════════════════════════════════════════════════════════
# CHECK forecast_explainability.py
# ══════════════════════════════════════════════════════════════
path3 = 'trading/analytics/forecast_explainability.py'
src3 = read(path3)
ok3, err3 = check_compile(path3)
has_kernel = 'KernelExplainer' in src3
has_routing = 'model_class' in src3 or 'model_type' in src3 or 'xgboost' in src3.lower()
print(f"\n[{'PASS' if ok3 else 'FAIL'}] forecast_explainability.py compiles: {ok3}")
print(f"[{'PASS' if has_kernel else 'FAIL'}] forecast_explainability.py has KernelExplainer: {has_kernel}")
print(f"[{'PASS' if has_routing else 'FAIL'}] forecast_explainability.py has model routing: {has_routing}")
if not ok3:
    print(f"  Error: {err3}")
results.append(ok3)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print()
print("=" * 50)
passed = sum(results)
print(f"RESULTS: {passed}/{len(results)} PASS")
if all(results):
    print("All files compile. Run smoke tests:")
    print("  .\\evolve_venv\\Scripts\\python.exe tests\\model_smoke_test.py")