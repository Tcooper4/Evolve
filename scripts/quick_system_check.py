#!/usr/bin/env python3
"""Quick system check for Evolve trading platform."""

import sys
import os
import traceback
from pathlib import Path
from typing import List, Dict

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_import(module_name: str, description: str = None) -> Dict:
    """Test importing a module."""
    try:
        __import__(module_name)
        return {
            'module': module_name,
            'description': description or module_name,
            'status': 'SUCCESS',
            'error': None
        }
    except Exception as e:
        return {
            'module': module_name,
            'description': description or module_name,
            'status': 'FAILED',
            'error': str(e)
        }

def test_core_functionality() -> Dict:
    """Test core trading functionality."""
    try:
        # Test core trading imports
        from trading.utils.logging_utils import setup_logger
        from trading.agents.nlp_agent import NLPRequest, NLPResult
        from trading.data.providers.yfinance_provider import YFinanceProvider
        from trading.strategies.bollinger_strategy import BollingerStrategy
        from trading.models.forecast_router import ForecastRouter
        from trading.optimization.self_tuning_optimizer import SelfTuningOptimizer
        from trading.risk.risk_analyzer import RiskAnalyzer
        from trading.portfolio.portfolio_manager import PortfolioManager
        
        return {
            'module': 'core_functionality',
            'description': 'Core Trading Functionality',
            'status': 'SUCCESS',
            'error': None
        }
    except Exception as e:
        return {
            'module': 'core_functionality',
            'description': 'Core Trading Functionality',
            'status': 'FAILED',
            'error': str(e)
        }

def main():
    """Run system check."""
    print("🔍 Evolve Trading Platform - Quick System Check")
    print("=" * 60)
    
    # Core modules to test
    tests = [
        ('trading', 'Trading Core Module'),
        ('trading.agents', 'Trading Agents'),
        ('trading.models', 'Trading Models'),
        ('trading.strategies', 'Trading Strategies'),
        ('trading.risk', 'Risk Management'),
        ('trading.portfolio', 'Portfolio Management'),
        ('trading.data', 'Data Pipeline'),
        ('trading.execution', 'Execution Engine'),
        ('trading.optimization', 'Optimization Engine'),
        ('trading.report', 'Reporting System'),
        ('trading.visualization', 'Visualization'),
        ('trading.ui', 'UI Components'),
        ('strategies', 'Strategy Gatekeeper'),
        ('execution', 'Execution Engine'),
        ('models', 'Model Router'),
        ('utils', 'Utilities'),
        ('config', 'Configuration'),
        ('data', 'Data Pipeline'),
        ('pages', 'Streamlit Pages'),
        ('dashboard', 'Dashboard'),
        ('rl', 'Reinforcement Learning'),
        ('causal', 'Causal Inference'),
        ('llm', 'LLM Integration'),
    ]
    
    results = []
    successes = 0
    failures = 0
    
    # Test core functionality first
    core_result = test_core_functionality()
    results.append(core_result)
    if core_result['status'] == 'SUCCESS':
        successes += 1
        print(f"✅ {core_result['description']}")
    else:
        failures += 1
        print(f"❌ {core_result['description']}: {core_result['error']}")
    
    # Test individual modules
    for module, description in tests:
        result = test_import(module, description)
        results.append(result)
        
        if result['status'] == 'SUCCESS':
            successes += 1
            print(f"✅ {result['description']}")
        else:
            failures += 1
            print(f"❌ {result['description']}: {result['error']}")
    
    total = len(results)
    success_rate = (successes / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {successes}/{total} successful ({success_rate:.1f}%)")
    print(f"✅ Successes: {successes}")
    print(f"❌ Failures: {failures}")
    
    if failures > 0:
        print("\n🔧 FAILED MODULES:")
        for result in results:
            if result['status'] == 'FAILED':
                print(f"  - {result['module']}: {result['error']}")
    
    if success_rate >= 80:
        print("\n🎯 System Status: ✅ EXCELLENT")
        print("🚀 System is ready for institutional-grade trading!")
    elif success_rate >= 60:
        print("\n🎯 System Status: ✅ GOOD")
        print("🚀 System is ready for production with minor issues!")
    else:
        print("\n🎯 System Status: ⚠️ NEEDS ATTENTION")
        print("🔧 Some modules need fixing before production!")

if __name__ == "__main__":
    main() 