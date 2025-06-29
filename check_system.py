#!/usr/bin/env python3
"""Quick System Check for Evolve Trading Platform"""

import os
import sys
import importlib
from typing import Dict, List

def check_imports() -> Dict[str, bool]:
    """Check critical imports."""
    results = {}
    
    critical_modules = [
        'streamlit', 'pandas', 'numpy', 'yfinance', 'plotly',
        'scikit-learn', 'torch', 'transformers'
    ]
    
    for module in critical_modules:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError:
            results[module] = False
    
    return results

def check_core_modules() -> Dict[str, bool]:
    """Check core Evolve modules."""
    results = {}
    
    core_modules = [
        'trading.agents.base_agent',
        'trading.models.forecast_router',
        'trading.strategies.bollinger_strategy',
        'trading.data.data_loader',
        'trading.execution.execution_engine',
        'trading.optimization.bayesian_optimizer',
        'trading.risk.risk_analyzer',
        'trading.portfolio.portfolio_manager',
        'trading.evaluation.metrics',
        'trading.feature_engineering.feature_engineer'
    ]
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError:
            results[module] = False
    
    return results

def check_advanced_modules() -> Dict[str, bool]:
    """Check advanced modules."""
    results = {}
    
    advanced_modules = [
        'causal.causal_model',
        'trading.models.advanced.tcn.tcn_model',
        'trading.models.advanced.transformer.transformer_model',
        'trading.models.advanced.lstm.lstm_model',
        'trading.models.advanced.gnn.gnn_model',
        'trading.models.advanced.rl.rl_model',
        'trading.models.advanced.ensemble.ensemble_model',
        'trading.nlp.llm_processor',
        'trading.meta_agents.agents.agent_router'
    ]
    
    for module in advanced_modules:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError:
            results[module] = False
    
    return results

def check_ui_modules() -> Dict[str, bool]:
    """Check UI modules."""
    results = {}
    
    ui_modules = [
        'pages.1_Forecast_Trade',
        'pages.2_Strategy_Backtest',
        'pages.3_Trade_Execution',
        'pages.4_Portfolio_Management',
        'pages.5_Risk_Analysis',
        'pages.6_Model_Optimization',
        'pages.7_Market_Analysis',
        'pages.8_Agent_Management',
        'pages.9_System_Monitoring',
        'pages.10_Strategy_Health_Dashboard'
    ]
    
    for module in ui_modules:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError:
            results[module] = False
    
    return results

def main():
    """Run system check."""
    print("🔍 EVOLVE TRADING PLATFORM - SYSTEM CHECK")
    print("=" * 60)
    
    # Check imports
    print("\n📦 CRITICAL IMPORTS:")
    import_results = check_imports()
    import_success = sum(import_results.values())
    import_total = len(import_results)
    
    for module, success in import_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")
    
    # Check core modules
    print("\n🏗️ CORE MODULES:")
    core_results = check_core_modules()
    core_success = sum(core_results.values())
    core_total = len(core_results)
    
    for module, success in core_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")
    
    # Check advanced modules
    print("\n🚀 ADVANCED MODULES:")
    advanced_results = check_advanced_modules()
    advanced_success = sum(advanced_results.values())
    advanced_total = len(advanced_results)
    
    for module, success in advanced_results.items():
        status = "✅" if success else "⚠️"
        print(f"  {status} {module}")
    
    # Check UI modules
    print("\n🖥️ UI MODULES:")
    ui_results = check_ui_modules()
    ui_success = sum(ui_results.values())
    ui_total = len(ui_results)
    
    for module, success in ui_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")
    
    # Calculate overall success rate
    total_success = import_success + core_success + advanced_success + ui_success
    total_checks = import_total + core_total + advanced_total + ui_total
    
    success_rate = (total_success / total_checks) * 100 if total_checks > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📊 OVERALL SUCCESS RATE: {success_rate:.1f}%")
    print(f"✅ Success: {total_success}/{total_checks}")
    print(f"❌ Failures: {total_checks - total_success}")
    print("=" * 60)
    
    return success_rate

if __name__ == "__main__":
    main() 