#!/usr/bin/env python3
"""
System Health Test for Evolve Trading System

This script performs basic health checks on the system components.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test that basic modules can be imported."""
    logger.info("Testing basic imports...")
    
    try:
        # Test core imports
        from utils.common_helpers import safe_execute, validate_data
        logger.info("✅ Core utils imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import core utils: {e}")
        return False
    
    try:
        # Test trading imports
        from trading.strategies.rsi_signals import generate_rsi_signals
        logger.info("✅ Trading strategies imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import trading strategies: {e}")
        return False
    
    try:
        # Test agent imports
        from trading.agents.base_agent_interface import BaseAgent, AgentConfig
        logger.info("✅ Agent interface imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import agent interface: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality."""
    logger.info("Testing basic functionality...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'Open': np.random.normal(100, 10, len(dates)),
        'High': np.random.normal(105, 10, len(dates)),
        'Low': np.random.normal(95, 10, len(dates)),
        'Close': np.random.normal(100, 10, len(dates)),
        'Volume': np.random.normal(1000000, 200000, len(dates))
    }, index=dates)
    
    # Add trend
    trend = np.linspace(0, 20, len(dates))
    data['Close'] += trend
    data['Open'] += trend
    data['High'] += trend
    data['Low'] += trend
    
    # Test data validation
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in data.columns:
            logger.error(f"❌ Missing required column: {col}")
            return False
    
    # Test technical indicators
    sma_20 = data['Close'].rolling(window=20).mean()
    if len(sma_20) != len(data):
        logger.error("❌ SMA calculation failed")
        return False
    
    # Test signal generation
    sma_short = data['Close'].rolling(window=10).mean()
    sma_long = data['Close'].rolling(window=30).mean()
    signals = pd.Series(0, index=data.index)
    signals[sma_short > sma_long] = 1
    signals[sma_short < sma_long] = -1
    
    if not all(signals.isin([-1, 0, 1])):
        logger.error("❌ Signal generation failed")
        return False
    
    # Test performance calculation
    returns = data['Close'].pct_change()
    strategy_returns = signals.shift(1) * returns
    total_return = (1 + strategy_returns).prod() - 1
    
    if not isinstance(total_return, float):
        logger.error("❌ Performance calculation failed")
        return False
    
    logger.info("✅ Basic functionality tests passed")
    return True

def test_rsi_strategy():
    """Test RSI strategy functionality."""
    logger.info("Testing RSI strategy...")
    
    try:
        from trading.strategies.rsi_signals import generate_rsi_signals
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'Open': np.random.normal(100, 10, len(dates)),
            'High': np.random.normal(105, 10, len(dates)),
            'Low': np.random.normal(95, 10, len(dates)),
            'Close': np.random.normal(100, 10, len(dates)),
            'Volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        # Generate RSI signals
        result = generate_rsi_signals(data, period=14, buy_threshold=30, sell_threshold=70)
        
        # Check that result has required columns
        required_columns = ['signal', 'rsi', 'returns', 'strategy_returns']
        for col in required_columns:
            if col not in result.columns:
                logger.error(f"❌ RSI result missing column: {col}")
                return False
        
        # Check that signals are valid
        if not all(result['signal'].isin([-1, 0, 1])):
            logger.error("❌ RSI signals are invalid")
            return False
        
        logger.info("✅ RSI strategy tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RSI strategy test failed: {e}")
        return False

def test_agent_functionality():
    """Test basic agent functionality."""
    logger.info("Testing agent functionality...")
    
    try:
        from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult
        
        # Create a simple test agent
        class TestAgent(BaseAgent):
            def __init__(self):
                config = AgentConfig(
                    name="test_agent",
                    enabled=True,
                    priority=1
                )
                super().__init__(config)
            
            async def execute(self, prompt: str, agents=None):
                return AgentResult(
                    success=True,
                    data={"message": "Test successful"},
                    message="Test agent executed successfully",
                    timestamp=datetime.now().isoformat()
                )
        
        # Test agent creation
        agent = TestAgent()
        if agent.config.name != "test_agent":
            logger.error("❌ Agent creation failed")
            return False
        
        logger.info("✅ Agent functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent functionality test failed: {e}")
        return False

def test_system_resilience():
    """Test system resilience features."""
    logger.info("Testing system resilience...")
    
    try:
        from trading.system_resilience import SystemResilience
        
        # Create resilience manager
        resilience = SystemResilience()
        
        # Test signal activity check
        signals = pd.Series([1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
        data = pd.DataFrame({'Close': np.random.normal(100, 10, 10)})
        
        warnings = resilience.check_signal_activity(signals, data, "test_strategy")
        
        # Test system health
        health = resilience.check_system_health()
        
        if not isinstance(health, dict):
            logger.error("❌ System health check failed")
            return False
        
        logger.info("✅ System resilience tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ System resilience test failed: {e}")
        return False

def main():
    """Run all health checks."""
    logger.info("🚀 Starting Evolve Trading System Health Check")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Functionality", test_basic_functionality),
        ("RSI Strategy", test_rsi_strategy),
        ("Agent Functionality", test_agent_functionality),
        ("System Resilience", test_system_resilience)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Health Check Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All health checks passed! System is healthy.")
        return 0
    else:
        logger.error("⚠️ Some health checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 