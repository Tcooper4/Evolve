"""
Basic Import Tests

This module tests that all core modules can be imported without errors.
This is a prerequisite for running the full test suite.
"""

import unittest
import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestBasicImports(unittest.TestCase):
    """Test that all core modules can be imported."""
    
    def test_core_imports(self):
        """Test core module imports."""
        try:
            from trading.utils.logging_utils import setup_logger
            from trading.agents.nlp_agent import NLPRequest, NLPResult
            from trading.data.providers.yfinance_provider import YFinanceProvider
            from trading.strategies.bollinger_strategy import BollingerStrategy
            from trading.models.forecast_router import ForecastRouter
            from trading.optimization.self_tuning_optimizer import SelfTuningOptimizer
            from trading.risk.risk_analyzer import RiskAnalyzer
            from trading.portfolio.portfolio_manager import PortfolioManager
            logger.info("✅ Core trading modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core trading modules: {e}")
    
    def test_strategy_imports(self):
        """Test strategy module imports."""
        try:
            from trading.strategies.rsi_signals import generate_rsi_signals
            from trading.strategies.macd_strategy import MACDStrategy
            from trading.strategies.sma_strategy import SMAStrategy
            from strategies.gatekeeper import StrategyGatekeeper
            logger.info("✅ Strategy modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import strategy modules: {e}")
    
    def test_agent_imports(self):
        """Test agent module imports."""
        try:
            from trading.agents.prompt_router_agent import PromptRouterAgent
            from trading.agents.base_agent_interface import BaseAgent, AgentConfig
            from trading.agents.registry import AgentRegistry
            from trading.agents.market_regime_agent import MarketRegimeAgent
            from trading.agents.execution_risk_agent import ExecutionRiskAgent
            logger.info("✅ Agent modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import agent modules: {e}")
    
    def test_data_imports(self):
        """Test data module imports."""
        try:
            from trading.data.providers.alpha_vantage_provider import AlphaVantageProvider
            from trading.data.providers.fallback_provider import get_fallback_provider
            from trading.data.macro_data_integration import MacroDataIntegration
            logger.info("✅ Data modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import data modules: {e}")
    
    def test_backtesting_imports(self):
        """Test backtesting module imports."""
        try:
            from trading.backtesting.backtester import Backtester
            from trading.backtesting.enhanced_backtester import EnhancedBacktester
            from trading.backtesting.position_sizing import PositionSizingEngine
            from trading.backtesting.risk_metrics import RiskMetricsEngine
            logger.info("✅ Backtesting modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import backtesting modules: {e}")
    
    def test_optimization_imports(self):
        """Test optimization module imports."""
        try:
            from trading.optimization.bayesian_optimizer import BayesianOptimizer
            from trading.optimization.genetic_optimizer import GeneticOptimizer
            from trading.optimization.core_optimizer import CoreOptimizer
            logger.info("✅ Optimization modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import optimization modules: {e}")
    
    def test_utils_imports(self):
        """Test utility module imports."""
        try:
            from trading.utils.logging_utils import setup_logger, get_logger
            from trading.utils.config_utils import load_config
            from trading.utils.data_validation import validate_data
            from utils.config_loader import ConfigLoader
            logger.info("✅ Utility modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import utility modules: {e}")
    
    def test_config_imports(self):
        """Test configuration module imports."""
        try:
            from config.app_config import load_app_config
            from trading.config.configuration import Configuration
            from trading.config.enhanced_settings import EnhancedSettings
            logger.info("✅ Configuration modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import configuration modules: {e}")

if __name__ == '__main__':
    unittest.main() 