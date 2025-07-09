#!/usr/bin/env python3
"""Simple test for core functionality."""

import sys
import traceback
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly.graph_objects as go
        import yfinance as yf
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return None

def test_trading_core():
    """Test trading core functionality."""
    print("Testing trading core...")
    
    try:
        from trading.utils.logging_utils import setup_logger
        from trading.agents.nlp_agent import NLPRequest, NLPResult
        from trading.data.providers.yfinance_provider import YFinanceProvider
        from trading.strategies.bollinger_strategy import BollingerStrategy
        print("✅ Trading core successful")
        return True
    except ImportError as e:
        print(f"❌ Trading core failed: {e}")
        return None

def test_config_loader():
    """Test configuration loader."""
    print("Testing configuration loader...")
    
    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader()
        print("✅ Configuration loader successful")
        return True
    except ImportError as e:
        print(f"❌ Configuration loader failed: {e}")
        return None

def test_strategies():
    """Test strategy functionality."""
    print("Testing strategies...")
    
    try:
        from trading.strategies.rsi_signals import generate_rsi_signals
        from trading.strategies.macd_strategy import MACDStrategy
        from trading.strategies.sma_strategy import SMAStrategy
        from strategies.gatekeeper import StrategyGatekeeper
        print("✅ Strategies successful")
        return True
    except ImportError as e:
        print(f"❌ Strategies failed: {e}")
        return None

def test_agents():
    """Test agent functionality."""
    print("Testing agents...")
    
    try:
        from trading.agents.prompt_router_agent import PromptRouterAgent
        from trading.agents.base_agent_interface import BaseAgent, AgentConfig
        from trading.agents.registry import AgentRegistry
        print("✅ Agents successful")
        return True
    except ImportError as e:
        print(f"❌ Agents failed: {e}")
        return None

def test_backtesting():
    """Test backtesting functionality."""
    print("Testing backtesting...")
    
    try:
        from trading.backtesting.backtester import Backtester
        from trading.backtesting.enhanced_backtester import EnhancedBacktester
        from trading.backtesting.position_sizing import PositionSizingEngine
        print("✅ Backtesting successful")
        return True
    except ImportError as e:
        print(f"❌ Backtesting failed: {e}")
        return None

def test_optimization():
    """Test optimization functionality."""
    print("Testing optimization...")
    
    try:
        from trading.optimization.self_tuning_optimizer import SelfTuningOptimizer
        from trading.optimization.bayesian_optimizer import BayesianOptimizer
        from trading.optimization.genetic_optimizer import GeneticOptimizer
        print("✅ Optimization successful")
        return True
    except ImportError as e:
        print(f"❌ Optimization failed: {e}")
        return None

def main():
    """Run all tests."""
    print("🧪 Testing Core Functionality")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_trading_core,
        test_config_loader,
        test_strategies,
        test_agents,
        test_backtesting,
        test_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n📊 Summary:")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All core tests passed!")
        return True
    else:
        print("\n⚠️ Some tests failed.")
        return None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 