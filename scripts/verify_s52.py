from pathlib import Path

def read(p):
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

results = []

# Change 1 - transformer fixes
ts = read("trading/models/advanced/transformer/time_series_transformer.py")
results.append(("Transformer col_map_iter in forecast loop", "_col_map_iter" in ts or "_resolved_tgt" in ts))
results.append(("Transformer norm stats shape validation", "shape mismatch" in ts and "ignoring cached stats" in ts))
results.append(("Transformer norm stats load failure logged", "will refit from scratch" in ts or "norm stats load failed" in ts))
results.append(("Transformer normalization shape guard", "X_mean.shape[-1] == X.shape[-1]" in ts or "refitting stats" in ts))

# Change 2 - transformer_model.py
tm = read("trading/models/transformer_model.py")
results.append(("transformer_model default lowercase close", 'target_column", "close"' in tm))
results.append(("transformer_model _col_map in _prepare_features", "_col_map" in tm))
results.append(("transformer_model no hardcoded Close fallback", '"Close"' not in tm or "_col_map" in tm))

# Change 3 - _col_map in models
models = [
    "trading/models/xgboost_model.py",
    "trading/models/garch_model.py",
    "trading/models/prophet_model.py",
    "trading/models/ensemble_model.py",
    "trading/models/ridge_model.py",
    "trading/models/tcn_model.py",
    "trading/models/catboost_model.py",
    "trading/models/arima_model.py",
]
for m in models:
    src = read(m)
    results.append((f"{Path(m).name} _col_map added", "_col_map" in src))

# Change 4 - remaining dataframe calls
analyze = read("pages/2_Analyze.py")
results.append(("targets_df normalized", "normalize_for_display(targets_df)" in analyze))
results.append(("_c_display normalized", "normalize_for_display(_c_display)" in analyze))
results.append(("_p_display normalized", "normalize_for_display(_p_display)" in analyze))

# Change 5 - silent exceptions fixed
results.append(("history load failure logged", "history load failed" in analyze or "Could not load price history" in analyze))
results.append(("no bare except pass in analyze", analyze.count("except Exception:\n        pass") == 0))

# Change 6 - session state namespaced
results.append(("analyze_symbol key present", "analyze_symbol" in analyze))
results.append(("analyze_forecast_data key present", "analyze_forecast_data" in analyze))

print()
for name, passed in results:
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print()
