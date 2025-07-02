"""
Basic Import Tests

This module tests that all core modules can be imported without errors.
This is a prerequisite for running the full test suite.
"""

import unittest
import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestBasicImports(unittest.TestCase):
    """Test that all core modules can be imported."""
    
    def test_core_imports(self):
        """Test core module imports."""
        try:
            from core.agent_hub import AgentHub
            from core.capability_router import get_system_health
            from core.utils.common_helpers import safe_execute, validate_data
            from core.utils.technical_indicators import calculate_sma, calculate_rsi
            logger.info("✅ Core modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")
    
    def test_trading_imports(self):
        """Test trading module imports."""
        try:
            from trading.models.lstm_model import LSTMModel
            from trading.models.xgboost_model import XGBoostModel
            from trading.strategies.rsi_signals import generate_rsi_signals
            from trading.strategies.macd_strategy import MACDStrategy
            from trading.data.data_loader import DataLoader
            from trading.backtesting.backtester import Backtester
            logger.info("✅ Trading modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import trading modules: {e}")
    
    def test_agents_imports(self):
        """Test agent module imports."""
        try:
            from trading.agents.prompt_router_agent import PromptRouterAgent
            from trading.agents.base_agent_interface import BaseAgent, AgentConfig
            from trading.agents.registry import AgentRegistry
            logger.info("✅ Agent modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import agent modules: {e}")
    
    def test_utils_imports(self):
        """Test utility module imports."""
        try:
            from trading.utils.safe_executor import SafeExecutor
            from trading.utils.reasoning_logger import ReasoningLogger
            from trading.utils.error_handling import handle_exceptions
            logger.info("✅ Utility modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import utility modules: {e}")
    
    def test_config_imports(self):
        """Test configuration module imports."""
        try:
            from trading.config.configuration import TradingConfig
            from trading.config.settings import Settings
            logger.info("✅ Configuration modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import configuration modules: {e}")

if __name__ == '__main__':
    unittest.main() 