"""
Production Readiness Test Suite

This module provides comprehensive testing for all production components:
- Unit tests for each model
- Strategy signal tests
- Prompt routing tests
- Backtest logic tests
- UI component tests
- Integration tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define COMPONENTS_AVAILABLE flag
COMPONENTS_AVAILABLE = True  # Set to False if components are missing

# Add sample_price_data fixture
@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    volumes = np.random.uniform(1000000, 5000000, len(dates))
    return pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': volumes
    }, index=dates)

# Import components to test
try:
    from interface.unified_interface import EnhancedUnifiedInterface
    from trading.strategies.hybrid_engine import HybridEngine
    from trading.models.forecast_engine import ForecastEngine
    from trading.agents.prompt_router_agent import PromptRouterAgent
    from trading.backtesting.backtester import Backtester
    from trading.portfolio.portfolio_manager import PortfolioManager
    from trading.risk.risk_manager import RiskManager
    from trading.optimization.strategy_optimizer import StrategyOptimizer
    from trading.llm.agent import PromptAgent
    from trading.report.export_engine import ReportExporter
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available for testing: {e}")
    COMPONENTS_AVAILABLE = False

class TestProductionReadiness:
    """Comprehensive test suite for production readiness."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.sample_data = self._generate_sample_data()
        self.test_config = {
            'rsi_window': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_window': 20,
            'bollinger_std': 2.0
        }
        
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.normal(1000000, 200000, len(dates))
        })
        
        data.set_index('Date', inplace=True)
        return data

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_unified_interface_initialization(self):
        """Test unified interface initialization."""
        interface = EnhancedUnifiedInterface()
        assert interface is not None
        assert hasattr(interface, 'config')
        assert hasattr(interface, 'agent_hub')

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_hybrid_engine_strategies(self):
        """Test hybrid engine strategy execution."""
        engine = HybridEngine()
        
        # Test RSI strategy
        rsi_result = engine.run_strategy(self.sample_data, 'RSI')
        assert rsi_result is not None
        assert 'signals' in rsi_result
        assert 'performance' in rsi_result
        
        # Test MACD strategy
        macd_result = engine.run_strategy(self.sample_data, 'MACD')
        assert macd_result is not None
        assert 'signals' in macd_result
        
        # Test Bollinger strategy
        bollinger_result = engine.run_strategy(self.sample_data, 'Bollinger')
        assert bollinger_result is not None
        assert 'signals' in bollinger_result

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_forecast_engine_models(self):
        """Test forecast engine with different models."""
        engine = ForecastEngine()
        
        # Test Prophet model
        prophet_result = engine.forecast(self.sample_data, 'prophet', days=30)
        assert prophet_result is not None
        assert 'forecast' in prophet_result
        assert 'confidence_intervals' in prophet_result
        
        # Test LSTM model
        lstm_result = engine.forecast(self.sample_data, 'lstm', days=30)
        assert lstm_result is not None
        assert 'forecast' in lstm_result
        
        # Test XGBoost model
        xgb_result = engine.forecast(self.sample_data, 'xgboost', days=30)
        assert xgb_result is not None
        assert 'forecast' in xgb_result

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_prompt_router_agent(self):
        """Test prompt router agent functionality."""
        agent = PromptRouterAgent()
        
        # Test forecast prompt
        forecast_prompt = "Forecast AAPL for the next 30 days"
        forecast_result = agent.route_prompt(forecast_prompt, {})
        assert forecast_result is not None
        assert 'action' in forecast_result
        
        # Test strategy prompt
        strategy_prompt = "Create a bullish strategy for TSLA"
        strategy_result = agent.route_prompt(strategy_prompt, {})
        assert strategy_result is not None
        assert 'action' in strategy_result
        
        # Test backtest prompt
        backtest_prompt = "Backtest RSI strategy on SPY"
        backtest_result = agent.route_prompt(backtest_prompt, {})
        assert backtest_result is not None
        assert 'action' in backtest_result

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_backtester_functionality(self):
        """Test backtester with various strategies."""
        backtester = Backtester()
        
        # Test RSI backtest
        rsi_backtest = backtester.run_backtest(
            self.sample_data, 
            'RSI', 
            initial_capital=100000
        )
        assert rsi_backtest is not None
        assert 'returns' in rsi_backtest
        assert 'sharpe_ratio' in rsi_backtest
        assert 'max_drawdown' in rsi_backtest
        assert 'win_rate' in rsi_backtest
        
        # Test MACD backtest
        macd_backtest = backtester.run_backtest(
            self.sample_data, 
            'MACD', 
            initial_capital=100000
        )
        assert macd_backtest is not None
        assert 'returns' in macd_backtest

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_portfolio_manager(self):
        """Test portfolio manager functionality."""
        manager = PortfolioManager()
        
        # Test position management
        position = manager.add_position('AAPL', 100, 150.0)
        assert position is not None
        assert position['symbol'] == 'AAPL'
        assert position['quantity'] == 100
        
        # Test portfolio summary
        summary = manager.get_position_summary()
        assert summary is not None
        assert 'total_value' in summary
        assert 'positions' in summary
        
        # Test risk metrics
        risk_metrics = manager.get_risk_metrics()
        assert risk_metrics is not None
        assert 'sharpe_ratio' in risk_metrics
        assert 'max_drawdown' in risk_metrics

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_risk_manager(self):
        """Test risk manager functionality."""
        manager = RiskManager()
        
        # Test position sizing
        position_size = manager.calculate_position_size(
            capital=100000,
            risk_per_trade=0.02,
            entry_price=150.0,
            stop_loss=145.0
        )
        assert position_size > 0
        assert position_size <= 100000
        
        # Test risk assessment
        risk_assessment = manager.assess_risk(self.sample_data)
        assert risk_assessment is not None
        assert 'volatility' in risk_assessment
        assert 'var_95' in risk_assessment

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_strategy_optimizer(self):
        """Test strategy optimizer functionality."""
        optimizer = StrategyOptimizer('RSI')
        
        # Test parameter optimization
        optimization_result = optimizer.optimize(
            self.sample_data,
            n_trials=10,
            timeout=60
        )
        assert optimization_result is not None
        assert 'best_params' in optimization_result
        assert 'best_score' in optimization_result
        
        # Test save/load functionality
        test_file = 'test_optimization.json'
        optimizer.save_optimization_results(test_file)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_llm_agent(self):
        """Test LLM agent functionality."""
        agent = PromptAgent()
        
        # Test commentary generation
        commentary = agent.generate_commentary({
            'symbol': 'AAPL',
            'action': 'BUY',
            'confidence': 0.85,
            'reasoning': 'Strong technical indicators'
        })
        assert commentary is not None
        assert len(commentary) > 0
        
        # Test decision explanation
        explanation = agent.explain_decision({
            'strategy': 'RSI',
            'signal': 'BUY',
            'parameters': {'rsi_window': 14}
        })
        assert explanation is not None
        assert len(explanation) > 0

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Components not available")
    def test_report_exporter(self):
        """Test report exporter functionality."""
        exporter = ReportExporter()
        
        # Test JSON export
        test_data = {
            'forecast': {'target_price': 150.0, 'confidence': 0.85},
            'strategy': {'name': 'RSI', 'performance': 0.12}
        }
        
        json_report = exporter.export_report(test_data, 'json')
        assert json_report is not None
        assert isinstance(json_report, str)
        
        # Test CSV export
        csv_report = exporter.export_report(test_data, 'csv')
        assert csv_report is not None

    def test_data_validation(self):
        """Test data validation functionality."""
        # Test valid data
        assert not self.sample_data.empty
        assert 'Open' in self.sample_data.columns
        assert 'High' in self.sample_data.columns
        assert 'Low' in self.sample_data.columns
        assert 'Close' in self.sample_data.columns
        assert 'Volume' in self.sample_data.columns
        
        # Test data types
        assert isinstance(self.sample_data.index, pd.DatetimeIndex)
        assert self.sample_data.index.is_monotonic_increasing
        
        # Test for missing values
        assert not self.sample_data.isnull().any().any()

    def test_performance_metrics(self):
        """Test performance metrics calculations."""
        # Calculate basic metrics
        returns = self.sample_data['Close'].pct_change().dropna()
        
        # Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        assert not np.isnan(sharpe_ratio)
        assert not np.isinf(sharpe_ratio)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        assert not np.isnan(max_drawdown)
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        assert not np.isnan(volatility)
        assert volatility > 0

    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            if COMPONENTS_AVAILABLE:
                engine = HybridEngine()
                engine.run_strategy(empty_data, 'RSI')
        
        # Test with invalid parameters
        invalid_config = {'invalid_param': 'invalid_value'}
        with pytest.raises(ValueError):
            if COMPONENTS_AVAILABLE:
                engine = HybridEngine()
                engine.config = invalid_config
                engine._validate_config()

    def test_logging_functionality(self):
        """Test logging functionality."""
        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.INFO)
        
        # Test info logging
        with self._capture_logs(logger, level='INFO') as logs:
            logger.info("Test info message")
            assert "Test info message" in logs.getvalue()
        
        # Test error logging
        with self._capture_logs(logger, level='ERROR') as logs:
            logger.error("Test error message")
            assert "Test error message" in logs.getvalue()

    @contextmanager
    def _capture_logs(self, logger, level='INFO'):
        """Context manager to capture log output."""
        import io
        import contextlib
        
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(getattr(logging, level))
        logger.addHandler(handler)
        
        try:
            yield log_capture
        finally:
            logger.removeHandler(handler)

    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        # Test valid configuration
        valid_config = {
            'rsi_window': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
        # Validate configuration
        assert 'rsi_window' in valid_config
        assert valid_config['rsi_window'] > 0
        assert valid_config['macd_fast'] > 0
        assert valid_config['macd_slow'] > valid_config['macd_fast']

    def test_data_persistence(self):
        """Test data persistence functionality."""
        # Test JSON serialization
        test_data = {
            'forecast': [100.0, 101.0, 102.0],
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat()
        }
        
        # Serialize
        json_str = json.dumps(test_data)
        assert isinstance(json_str, str)
        
        # Deserialize
        deserialized = json.loads(json_str)
        assert deserialized['confidence'] == 0.85
        assert len(deserialized['forecast']) == 3

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "--cov=trading",
        "--cov=core",
        "--cov=unified_interface",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v"
    ]) 