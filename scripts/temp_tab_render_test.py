# scripts/temp_tab_render_test.py
import sys
import os

# Mock streamlit minimally so we can import
print("=== Testing tab imports ===\n")

tabs_to_test = [
    ("tab_market_analysis", "render_market_analysis"),
    ("tab_diagnostics",     "render_diagnostics"),
    ("tab_multi_asset_gnn", "render_multi_asset_gnn"),
    ("tab_causal",          "render_causal"),
    ("tab_earnings",        "render_earnings"),
    ("tab_monte_carlo",     "render_monte_carlo"),
    ("tab_options_chain",   "render_options_chain"),
    ("tab_scanner_signals", "render_scanner_signals"),
]

for module_name, func_name in tabs_to_test:
    try:
        mod = __import__(
            f"components.tabs.{module_name}",
            fromlist=[func_name]
        )
        fn = getattr(mod, func_name)
        print(f"OK    {module_name}.{func_name} imports cleanly")
    except Exception as e:
        print(f"FAIL  {module_name}: {e}")

# Also check the forecasting_backend issue
print("\n=== Testing forecasting backend imports ===\n")
backend_imports = [
    ("trading.data.data_loader", ["DataLoader", "DataLoadRequest"]),
    ("trading.data.providers.yfinance_provider", ["YFinanceProvider"]),
    ("trading.models.lstm_model", ["LSTMForecaster"]),
    ("trading.models.xgboost_model", ["XGBoostModel"]),
    ("trading.models.prophet_model", ["ProphetModel"]),
    ("trading.models.arima_model", ["ARIMAModel"]),
    ("trading.data.preprocessing", ["FeatureEngineering", "DataPreprocessor"]),
    ("trading.agents.model_selector_agent", ["ModelSelectorAgent"]),
    ("trading.market.market_analyzer", ["MarketAnalyzer"]),
]

for module_path, names in backend_imports:
    try:
        mod = __import__(module_path, fromlist=names)
        for name in names:
            getattr(mod, name)
        print(f"OK    {module_path}")
    except Exception as e:
        print(f"FAIL  {module_path}: {e}")

# Check trading.ui.forecast_components
print("\n=== Testing trading.ui.forecast_components ===\n")
try:
    from trading.ui.forecast_components import (
        render_forecast_results, render_confidence_metrics
    )
    print("OK    trading.ui.forecast_components imports cleanly")
except Exception as e:
    print(f"FAIL  trading.ui.forecast_components: {e}")
    # Show what it tries to import
    try:
        lines = open(
            "trading/ui/forecast_components.py",
            encoding="utf-8", errors="replace"
        ).readlines()
        print(f"\n  forecast_components.py imports:")
        for i, line in enumerate(lines[:30], 1):
            if "import" in line:
                print(f"    {i}: {line.rstrip()}")
    except Exception as e2:
        print(f"  Could not read file: {e2}")