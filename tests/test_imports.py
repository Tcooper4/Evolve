#!/usr/bin/env python3
"""
Test script to check all imports and identify any remaining issues.
"""

import sys
import traceback
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_import(module_name):
    """Test importing a module and return success status."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} failed to import: {e}")
        return None

def main():
    """Test all critical imports."""
    print("Testing critical imports...")
    print("=" * 50)
    
    modules_to_test = [
        "trading.memory.agent_memory_manager",
        "trading.portfolio.portfolio_manager",
        "trading.risk.risk_analyzer",
        "trading.agents.market_regime_agent",
        "trading.utils.logging_utils",
        "trading.strategies.bollinger_strategy",
        "trading.models.forecast_router",
        "trading.optimization.self_tuning_optimizer",
        "trading.backtesting.backtester",
        "trading.data.providers.yfinance_provider",
        "strategies.gatekeeper",
        "utils.config_loader",
        "config.app_config"
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module in modules_to_test:
        if test_import(module):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"Results: {success_count}/{total_count} modules imported successfully")
    
    if success_count == total_count:
        print("🎉 All imports successful! The app should work now.")
    else:
        print("⚠️  Some imports failed. Check the errors above.")

    return None

if __name__ == "__main__":
    main() 