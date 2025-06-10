import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from trading.nlp.nl_interface import NLInterface
from trading.nlp.prompt_processor import PromptProcessor
from trading.nlp.response_formatter import ResponseFormatter
from trading.models.base_model import BaseModel
from trading.strategies.strategy_manager import StrategyManager
from trading.risk.risk_manager import RiskManager
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.market.market_analyzer import MarketAnalyzer

@pytest.fixture
def config_dir():
    """Get the configuration directory path."""
    return Path(__file__).parent.parent.parent / "trading" / "nlp" / "config"

@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    return pd.DataFrame({
        'open': np.random.randn(20) + 100,
        'high': np.random.randn(20) + 102,
        'low': np.random.randn(20) + 98,
        'close': np.random.randn(20) + 100,
        'volume': np.random.randint(1000, 2000, 20)
    }, index=dates)

@pytest.fixture
def mock_forecast():
    """Create mock forecast data for testing."""
    dates = pd.date_range(start='2024-01-21', periods=5, freq='D')
    return pd.Series(np.random.randn(5) + 100, index=dates)

@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    market_analyzer = Mock(spec=MarketAnalyzer)
    model = Mock(spec=BaseModel)
    strategy_manager = Mock(spec=StrategyManager)
    risk_manager = Mock(spec=RiskManager)
    portfolio_manager = Mock(spec=PortfolioManager)
    
    return {
        'market_analyzer': market_analyzer,
        'model': model,
        'strategy_manager': strategy_manager,
        'risk_manager': risk_manager,
        'portfolio_manager': portfolio_manager
    }

@pytest.fixture
def nl_interface(config_dir, mock_components):
    """Create an NLInterface instance with mock components."""
    interface = NLInterface(config_dir)
    
    # Replace components with mocks
    interface.market_analyzer = mock_components['market_analyzer']
    interface.model = mock_components['model']
    interface.strategy_manager = mock_components['strategy_manager']
    interface.risk_manager = mock_components['risk_manager']
    interface.portfolio_manager = mock_components['portfolio_manager']
    
    return interface

def test_forecast_pipeline(nl_interface, mock_market_data, mock_forecast):
    """Test the complete forecast pipeline."""
    # Setup mock responses
    nl_interface.market_analyzer.get_historical_data.return_value = mock_market_data
    nl_interface.model.predict.return_value = mock_forecast
    nl_interface.model.get_confidence_intervals.return_value = {
        'lower': mock_forecast - 0.1,
        'upper': mock_forecast + 0.1
    }
    nl_interface.model.get_feature_importance.return_value = {
        'factor1': 0.6,
        'factor2': 0.4
    }
    
    # Test different forecast queries
    queries = [
        "What's the price forecast for BTC in the next 5 days?",
        "Predict BTC price for the next week",
        "Show me the market outlook for BTC"
    ]
    
    for query in queries:
        response = nl_interface.process_query(query)
        
        # Verify response structure
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
        
        # Verify content
        assert "BTC" in response.text
        assert "forecast" in response.text.lower()
        assert str(mock_forecast.mean()) in response.text
        
        # Verify component interactions
        nl_interface.market_analyzer.get_historical_data.assert_called()
        nl_interface.model.predict.assert_called()
        nl_interface.model.get_confidence_intervals.assert_called()
        nl_interface.model.get_feature_importance.assert_called()

def test_analysis_pipeline(nl_interface, mock_market_data):
    """Test the complete analysis pipeline."""
    # Setup mock responses
    nl_interface.market_analyzer.get_market_data.return_value = mock_market_data
    nl_interface.market_analyzer.analyze_technical.return_value = {
        'RSI': 65.5,
        'MACD': 0.5,
        'BB': {'upper': 1.1, 'middle': 1.0, 'lower': 0.9}
    }
    nl_interface.market_analyzer.get_market_state.return_value = "bullish"
    
    # Test different analysis queries
    queries = [
        "Analyze the current market conditions for ETH",
        "What's the technical analysis for ETH?",
        "Show me the market state of ETH"
    ]
    
    for query in queries:
        response = nl_interface.process_query(query)
        
        # Verify response structure
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
        
        # Verify content
        assert "ETH" in response.text
        assert "analysis" in response.text.lower()
        assert "bullish" in response.text.lower()
        assert "RSI" in response.text
        assert "MACD" in response.text
        
        # Verify component interactions
        nl_interface.market_analyzer.get_market_data.assert_called()
        nl_interface.market_analyzer.analyze_technical.assert_called()
        nl_interface.market_analyzer.get_market_state.assert_called()

def test_recommendation_pipeline(nl_interface, mock_market_data):
    """Test the complete recommendation pipeline."""
    # Setup mock responses
    nl_interface.market_analyzer.get_market_data.return_value = mock_market_data
    nl_interface.strategy_manager.generate_signals.return_value = pd.Series([1, 1, 1, 0, 0])
    nl_interface.strategy_manager.get_entry_level.return_value = 100.0
    nl_interface.strategy_manager.get_signal_rationale.return_value = "Strong technical indicators"
    nl_interface.risk_manager.calculate_position_size.return_value = 0.1
    nl_interface.risk_manager.calculate_stop_loss.return_value = 95.0
    nl_interface.risk_manager.calculate_take_profit.return_value = 110.0
    nl_interface.portfolio_manager.get_portfolio_value.return_value = 100000.0
    
    # Test different recommendation queries
    queries = [
        "Should I buy SOL now?",
        "What's your trading recommendation for SOL?",
        "Is it a good time to enter SOL?"
    ]
    
    for query in queries:
        response = nl_interface.process_query(query)
        
        # Verify response structure
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
        
        # Verify content
        assert "SOL" in response.text
        assert "recommendation" in response.text.lower()
        assert "buy" in response.text.lower()
        assert "100.0" in response.text  # Entry price
        assert "95.0" in response.text   # Stop loss
        assert "110.0" in response.text  # Take profit
        
        # Verify component interactions
        nl_interface.market_analyzer.get_market_data.assert_called()
        nl_interface.strategy_manager.generate_signals.assert_called()
        nl_interface.risk_manager.calculate_position_size.assert_called()
        nl_interface.risk_manager.calculate_stop_loss.assert_called()
        nl_interface.risk_manager.calculate_take_profit.assert_called()

def test_comparison_pipeline(nl_interface, mock_market_data):
    """Test the complete comparison pipeline."""
    # Setup mock responses
    nl_interface.market_analyzer.get_market_data.return_value = mock_market_data
    nl_interface.market_analyzer.calculate_correlation.return_value = pd.DataFrame({
        'BTC': [1.0, 0.8],
        'ETH': [0.8, 1.0]
    }, index=['BTC', 'ETH'])
    nl_interface.market_analyzer.compare_assets.return_value = {
        'BTC': 0.1,
        'ETH': 0.05
    }
    nl_interface.market_analyzer.get_key_differences.return_value = [
        "BTC has higher volatility",
        "ETH has better liquidity"
    ]
    
    # Test different comparison queries
    queries = [
        "Compare the performance of BTC and ETH",
        "How do BTC and ETH compare?",
        "Show me the correlation between BTC and ETH"
    ]
    
    for query in queries:
        response = nl_interface.process_query(query)
        
        # Verify response structure
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
        
        # Verify content
        assert "BTC" in response.text
        assert "ETH" in response.text
        assert "comparison" in response.text.lower()
        assert "correlation" in response.text.lower()
        
        # Verify component interactions
        nl_interface.market_analyzer.get_market_data.assert_called()
        nl_interface.market_analyzer.calculate_correlation.assert_called()
        nl_interface.market_analyzer.compare_assets.assert_called()
        nl_interface.market_analyzer.get_key_differences.assert_called()

def test_error_handling_pipeline(nl_interface):
    """Test error handling across the entire pipeline."""
    # Test invalid queries
    invalid_queries = [
        "Invalid query",
        "What's the price?",
        "Analyze the market",
        "Compare performance"
    ]
    
    for query in invalid_queries:
        response = nl_interface.process_query(query)
        assert "error" in response.text.lower()
    
    # Test component errors
    nl_interface.market_analyzer.get_market_data.side_effect = Exception("Market data error")
    response = nl_interface.process_query("Analyze BTC")
    assert "error" in response.text.lower()
    
    nl_interface.model.predict.side_effect = Exception("Prediction error")
    response = nl_interface.process_query("Forecast BTC")
    assert "error" in response.text.lower()
    
    nl_interface.strategy_manager.generate_signals.side_effect = Exception("Strategy error")
    response = nl_interface.process_query("Recommend BTC")
    assert "error" in response.text.lower()

def test_performance_pipeline(nl_interface, mock_market_data):
    """Test the performance of the pipeline."""
    import time
    
    # Setup mock responses
    nl_interface.market_analyzer.get_market_data.return_value = mock_market_data
    nl_interface.model.predict.return_value = pd.Series(np.random.randn(5) + 100)
    
    # Test response time for different query types
    query_types = {
        "forecast": "What's the price forecast for BTC?",
        "analysis": "Analyze ETH",
        "recommendation": "Should I buy SOL?",
        "comparison": "Compare BTC and ETH"
    }
    
    max_response_time = 2.0  # Maximum acceptable response time in seconds
    
    for query_type, query in query_types.items():
        start_time = time.time()
        response = nl_interface.process_query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < max_response_time, f"{query_type} query took too long: {response_time:.2f}s"
        
        # Verify response quality
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None

def test_concurrent_queries(nl_interface, mock_market_data):
    """Test handling of concurrent queries."""
    import concurrent.futures
    
    # Setup mock responses
    nl_interface.market_analyzer.get_market_data.return_value = mock_market_data
    nl_interface.model.predict.return_value = pd.Series(np.random.randn(5) + 100)
    
    # Create a list of queries to process concurrently
    queries = [
        "What's the price forecast for BTC?",
        "Analyze ETH",
        "Should I buy SOL?",
        "Compare BTC and ETH"
    ]
    
    # Process queries concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(nl_interface.process_query, query) for query in queries]
        responses = [future.result() for future in futures]
    
    # Verify all responses
    for response in responses:
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
        assert "error" not in response.text.lower()

def test_forecast_integration(nl_interface):
    """Test the forecast pipeline integration."""
    query = "What's the price forecast for BTC in the next 5 days?"
    response = nl_interface.process_query(query)
    
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "BTC" in response.text
    assert "forecast" in response.text.lower()

def test_analysis_integration(nl_interface):
    """Test the analysis pipeline integration."""
    query = "Analyze the current market conditions for ETH"
    response = nl_interface.process_query(query)
    
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "ETH" in response.text
    assert "analysis" in response.text.lower()
    assert "bullish" in response.text.lower()

def test_recommendation_integration(nl_interface):
    """Test the recommendation pipeline integration."""
    query = "Should I buy SOL now?"
    response = nl_interface.process_query(query)
    
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None
    assert "SOL" in response.text
    assert "recommendation" in response.text.lower()
    assert "buy" in response.text.lower()

def test_error_handling_integration(nl_interface):
    """Test error handling integration."""
    # Test invalid query
    response = nl_interface.process_query("Invalid query")
    assert "error" in response.text.lower()
    
    # Test missing required information
    response = nl_interface.process_query("What's the price forecast?")
    assert "error" in response.text.lower()
    
    # Test component error
    nl_interface.market_analyzer.get_market_data.side_effect = Exception("Test error")
    response = nl_interface.process_query("Analyze BTC")
    assert "error" in response.text.lower()

def test_performance_integration(nl_interface):
    """Test performance integration."""
    import time
    
    # Test response time
    start_time = time.time()
    response = nl_interface.process_query("What's the price forecast for BTC?")
    end_time = time.time()
    
    response_time = end_time - start_time
    assert response_time < 2.0  # Maximum acceptable response time in seconds
    
    # Verify response quality
    assert response.text is not None
    assert response.visualization is not None
    assert response.metadata is not None 