# scripts/temp_backend_check.py
print("=== Testing full backend import ===\n")

try:
    from trading.market.market_analyzer import MarketAnalyzer
    print("OK    MarketAnalyzer")
except Exception as e:
    print(f"FAIL  MarketAnalyzer: {e}")

try:
    from trading.agents.model_selector_agent import ModelSelectorAgent
    print("OK    ModelSelectorAgent")
except Exception as e:
    print(f"FAIL  ModelSelectorAgent: {e}")

try:
    from trading.data.preprocessing import FeatureEngineering, DataPreprocessor
    print("OK    FeatureEngineering, DataPreprocessor")
except Exception as e:
    print(f"FAIL  preprocessing: {e}")

# Check if analyze_common imports cleanly
try:
    from components.analyze_common import (
        render_ai_score_panel,
        render_earnings_panel,
        render_insider_panel,
        render_short_interest_panel,
    )
    print("OK    analyze_common")
except Exception as e:
    print(f"FAIL  analyze_common: {e}")

# Check each tab individually
tabs = [
    "tab_market_analysis",
    "tab_diagnostics", 
    "tab_multi_asset_gnn",
    "tab_causal",
    "tab_earnings",
    "tab_monte_carlo",
    "tab_options_chain",
    "tab_scanner_signals",
]

print("\n=== Testing tab imports ===\n")
import sys, os
sys.path.insert(0, os.getcwd())

for tab in tabs:
    try:
        mod = __import__(
            f"components.tabs.{tab}",
            fromlist=["render"]
        )
        print(f"OK    {tab}")
    except Exception as e:
        print(f"FAIL  {tab}: {e}")