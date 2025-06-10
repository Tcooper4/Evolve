import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from trading.nlp.nl_interface import NLInterface, NLResponse

@pytest.fixture
def nl_interface():
    """Create an NLInterface instance for testing."""
    config_dir = Path(__file__).parent.parent.parent / "trading" / "nlp" / "config"
    return NLInterface(config_dir)

@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'open': np.random.randn(10),
        'high': np.random.randn(10),
        'low': np.random.randn(10),
        'close': np.random.randn(10),
        'volume': np.random.randint(1000, 2000, 10)
    }, index=dates)

@pytest.fixture
def mock_forecast():
    """Create mock forecast data for testing."""
    dates = pd.date_range(start='2024-01-11', periods=5, freq='D')
    return pd.Series(np.random.randn(5), index=dates)

def test_process_forecast_query(nl_interface, mock_market_data, mock_forecast):
    """Test processing a forecast query."""
    # Mock the market analyzer and model
    nl_interface.market_analyzer.get_historical_data = Mock(return_value=mock_market_data)
    nl_interface.model.predict = Mock(return_value=mock_forecast)
    nl_interface.model.get_confidence_intervals = Mock(return_value={
        'lower': mock_forecast - 0.1,
        'upper': mock_forecast + 0.1
    })
    nl_interface.model.get_feature_importance = Mock(return_value={
        'factor1': 0.6,
        'factor2': 0.4
    })
    
    # Process the query
    response = nl_interface.process_query("What's the price forecast for BTC in the next 5 days?")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "BTC" in response.text
    assert "forecast" in response.text.lower()

def test_process_analysis_query(nl_interface, mock_market_data):
    """Test processing an analysis query."""
    # Mock the market analyzer
    nl_interface.market_analyzer.get_market_data = Mock(return_value=mock_market_data)
    nl_interface.market_analyzer.analyze_technical = Mock(return_value={
        'RSI': 65.5,
        'MACD': 0.5,
        'BB': {'upper': 1.1, 'middle': 1.0, 'lower': 0.9}
    })
    nl_interface.market_analyzer.get_market_state = Mock(return_value="bullish")
    
    # Process the query
    response = nl_interface.process_query("Analyze the current market conditions for ETH")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "ETH" in response.text
    assert "analysis" in response.text.lower()
    assert "bullish" in response.text.lower()

def test_process_recommendation_query(nl_interface, mock_market_data):
    """Test processing a recommendation query."""
    # Mock the market analyzer and strategy manager
    nl_interface.market_analyzer.get_market_data = Mock(return_value=mock_market_data)
    nl_interface.strategy_manager.generate_signals = Mock(return_value=pd.Series([1, 1, 1, 0, 0]))
    nl_interface.strategy_manager.get_entry_level = Mock(return_value=100.0)
    nl_interface.strategy_manager.get_signal_rationale = Mock(
        return_value="Strong technical indicators and positive market sentiment"
    )
    nl_interface.risk_manager.calculate_position_size = Mock(return_value=0.1)
    nl_interface.risk_manager.calculate_stop_loss = Mock(return_value=95.0)
    nl_interface.risk_manager.calculate_take_profit = Mock(return_value=110.0)
    nl_interface.portfolio_manager.get_portfolio_value = Mock(return_value=100000.0)
    
    # Process the query
    response = nl_interface.process_query("Should I buy SOL now?")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "SOL" in response.text
    assert "buy" in response.text.lower()
    assert "recommendation" in response.text.lower()

def test_process_explanation_query(nl_interface):
    """Test processing an explanation query."""
    # Mock the market analyzer
    nl_interface.market_analyzer.explain_topic = Mock(
        return_value="Detailed explanation of the recent price movement"
    )
    nl_interface.market_analyzer.get_key_points = Mock(
        return_value=["Point 1", "Point 2", "Point 3"]
    )
    nl_interface.market_analyzer.get_explanation_source = Mock(
        return_value="Technical Analysis"
    )
    
    # Process the query
    response = nl_interface.process_query("Explain the recent price movement of BTC")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "BTC" in response.text
    assert "explanation" in response.text.lower()

def test_process_comparison_query(nl_interface, mock_market_data):
    """Test processing a comparison query."""
    # Mock the market analyzer
    nl_interface.market_analyzer.get_market_data = Mock(return_value=mock_market_data)
    nl_interface.market_analyzer.calculate_correlation = Mock(
        return_value=pd.DataFrame({
            'BTC': [1.0, 0.8],
            'ETH': [0.8, 1.0]
        }, index=['BTC', 'ETH'])
    )
    nl_interface.market_analyzer.compare_assets = Mock(
        return_value={"BTC": 0.1, "ETH": 0.05}
    )
    nl_interface.market_analyzer.get_key_differences = Mock(
        return_value=["Difference 1", "Difference 2"]
    )
    
    # Process the query
    response = nl_interface.process_query("Compare the performance of BTC and ETH")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "BTC" in response.text
    assert "ETH" in response.text
    assert "comparison" in response.text.lower()

def test_process_optimization_query(nl_interface):
    """Test processing an optimization query."""
    # Mock the strategy manager
    nl_interface.strategy_manager.get_strategy_parameters = Mock(
        return_value={"param1": [1, 2, 3], "param2": [0.1, 0.2, 0.3]}
    )
    nl_interface.strategy_manager.optimize_strategy = Mock(
        return_value=Mock(
            best_params={"param1": 2, "param2": 0.2},
            best_score=0.85,
            all_scores=[0.8, 0.85, 0.82],
            convergence_history=[0.8, 0.85],
            optimization_time=10.0,
            method="grid_search"
        )
    )
    nl_interface.strategy_manager.get_improvements = Mock(
        return_value=["Improvement 1", "Improvement 2"]
    )
    
    # Process the query
    response = nl_interface.process_query("Optimize the MACD strategy for BTC")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "MACD" in response.text
    assert "optimization" in response.text.lower()

def test_process_validation_query(nl_interface):
    """Test processing a validation query."""
    # Mock the strategy manager
    nl_interface.strategy_manager.validate_strategy = Mock(
        return_value=Mock(
            results={"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
            accuracy=0.85,
            metrics=["accuracy", "precision", "recall"],
            values=[0.85, 0.82, 0.88],
            validation_time=5.0,
            method="cross_validation"
        )
    )
    
    # Process the query
    response = nl_interface.process_query("Validate the performance of the RSI strategy")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "RSI" in response.text
    assert "validation" in response.text.lower()

def test_process_monitoring_query(nl_interface):
    """Test processing a monitoring query."""
    # Mock the portfolio manager
    nl_interface.portfolio_manager.get_portfolio_status = Mock(
        return_value={"status": "active", "positions": 3}
    )
    nl_interface.portfolio_manager.get_performance_metrics = Mock(
        return_value=pd.Series([0.1, 0.2, 0.3], index=pd.date_range(start='2024-01-01', periods=3))
    )
    nl_interface.portfolio_manager.get_alerts = Mock(
        return_value=["Alert 1", "Alert 2"]
    )
    nl_interface.portfolio_manager.get_thresholds = Mock(
        return_value={"threshold1": 0.1, "threshold2": 0.2}
    )
    
    # Process the query
    response = nl_interface.process_query("Monitor the portfolio performance")
    
    assert isinstance(response, NLResponse)
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "monitoring" in response.text.lower()
    assert "portfolio" in response.text.lower()

def test_error_handling(nl_interface):
    """Test error handling in query processing."""
    # Test with invalid query
    response = nl_interface.process_query("Invalid query")
    assert isinstance(response, NLResponse)
    assert "error" in response.text.lower()
    
    # Test with missing required information
    response = nl_interface.process_query("What's the price forecast?")
    assert isinstance(response, NLResponse)
    assert "error" in response.text.lower()
    
    # Test with component error
    nl_interface.market_analyzer.get_market_data = Mock(side_effect=Exception("Test error"))
    response = nl_interface.process_query("Analyze the current market conditions for BTC")
    assert isinstance(response, NLResponse)
    assert "error" in response.text.lower() 