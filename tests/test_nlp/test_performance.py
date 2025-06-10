import pytest
import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from trading.nlp.nl_interface import NLInterface
from trading.nlp.prompt_processor import PromptProcessor
from trading.nlp.response_formatter import ResponseFormatter

@pytest.fixture
def config_dir():
    """Get the configuration directory path."""
    return Path(__file__).parent.parent.parent / "trading" / "nlp" / "config"

@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 102,
        'low': np.random.randn(100) + 98,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 2000, 100)
    }, index=dates)

@pytest.fixture
def mock_forecast():
    """Create mock forecast data for testing."""
    dates = pd.date_range(start='2024-04-10', periods=30, freq='D')
    return pd.Series(np.random.randn(30) + 100, index=dates)

@pytest.fixture
def nl_interface(config_dir):
    """Create an NLInterface instance with mocked components."""
    interface = NLInterface(config_dir)
    
    # Mock market analyzer
    interface.market_analyzer.get_historical_data = Mock(return_value=mock_market_data())
    interface.market_analyzer.get_market_data = Mock(return_value=mock_market_data())
    interface.market_analyzer.analyze_technical = Mock(return_value={
        'RSI': 65.5,
        'MACD': 0.5,
        'BB': {'upper': 1.1, 'middle': 1.0, 'lower': 0.9}
    })
    interface.market_analyzer.get_market_state = Mock(return_value="bullish")
    
    # Mock model
    interface.model.predict = Mock(return_value=mock_forecast())
    interface.model.get_confidence_intervals = Mock(return_value={
        'lower': mock_forecast() - 0.1,
        'upper': mock_forecast() + 0.1
    })
    interface.model.get_feature_importance = Mock(return_value={
        'factor1': 0.6,
        'factor2': 0.4
    })
    
    # Mock strategy manager
    interface.strategy_manager.generate_signals = Mock(return_value=pd.Series([1, 1, 1, 0, 0]))
    interface.strategy_manager.get_entry_level = Mock(return_value=100.0)
    interface.strategy_manager.get_signal_rationale = Mock(
        return_value="Strong technical indicators and positive market sentiment"
    )
    
    # Mock risk manager
    interface.risk_manager.calculate_position_size = Mock(return_value=0.1)
    interface.risk_manager.calculate_stop_loss = Mock(return_value=95.0)
    interface.risk_manager.calculate_take_profit = Mock(return_value=110.0)
    
    # Mock portfolio manager
    interface.portfolio_manager.get_portfolio_value = Mock(return_value=100000.0)
    
    return interface

def test_response_time_benchmarks(nl_interface):
    """Test response time benchmarks for different query types."""
    query_types = {
        "forecast": "What's the price forecast for BTC in the next 5 days?",
        "analysis": "Analyze the current market conditions for ETH",
        "recommendation": "Should I buy SOL now?",
        "comparison": "Compare the performance of BTC and ETH",
        "explanation": "Explain the recent price movement of BTC",
        "optimization": "Optimize the MACD strategy for BTC",
        "validation": "Validate the performance of the RSI strategy",
        "monitoring": "Monitor the portfolio performance"
    }
    
    results = {}
    max_response_time = 2.0  # Maximum acceptable response time in seconds
    
    for query_type, query in query_types.items():
        # Warm-up run
        nl_interface.process_query(query)
        
        # Measure response time
        start_time = time.time()
        response = nl_interface.process_query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        results[query_type] = response_time
        
        # Verify response quality
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
        assert response_time < max_response_time, f"{query_type} query took too long: {response_time:.2f}s"
    
    # Print results
    print("\nResponse Time Benchmarks:")
    for query_type, response_time in results.items():
        print(f"{query_type}: {response_time:.3f}s")

def test_concurrent_load(nl_interface):
    """Test system performance under concurrent load."""
    num_queries = 50
    max_workers = 10
    queries = [
        "What's the price forecast for BTC?",
        "Analyze ETH",
        "Should I buy SOL?",
        "Compare BTC and ETH"
    ] * (num_queries // 4)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(nl_interface.process_query, query) for query in queries]
        responses = [future.result() for future in as_completed(futures)]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    avg_response_time = total_time / num_queries
    queries_per_second = num_queries / total_time
    
    # Verify all responses
    for response in responses:
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
        assert "error" not in response.text.lower()
    
    # Print results
    print("\nConcurrent Load Test Results:")
    print(f"Total Queries: {num_queries}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Response Time: {avg_response_time:.3f}s")
    print(f"Queries per Second: {queries_per_second:.2f}")
    
    # Assert performance requirements
    assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time:.3f}s"
    assert queries_per_second > 5.0, f"Queries per second too low: {queries_per_second:.2f}"

def test_memory_usage(nl_interface):
    """Test memory usage under load."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate load
    num_queries = 100
    queries = ["What's the price forecast for BTC?"] * num_queries
    
    for query in queries:
        response = nl_interface.process_query(query)
        assert response.text is not None
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print("\nMemory Usage Test Results:")
    print(f"Initial Memory: {initial_memory:.2f} MB")
    print(f"Final Memory: {final_memory:.2f} MB")
    print(f"Memory Increase: {memory_increase:.2f} MB")
    
    # Assert memory requirements
    assert memory_increase < 100, f"Memory increase too high: {memory_increase:.2f} MB"

def test_error_recovery(nl_interface):
    """Test system recovery from errors."""
    # Simulate various error conditions
    error_conditions = [
        (Exception("Market data error"), "Analyze BTC"),
        (Exception("Prediction error"), "Forecast BTC"),
        (Exception("Strategy error"), "Recommend BTC"),
        (Exception("Risk calculation error"), "What's the risk for BTC?"),
        (Exception("Portfolio error"), "Monitor portfolio")
    ]
    
    for error, query in error_conditions:
        # Set up error condition
        if "Analyze" in query:
            nl_interface.market_analyzer.get_market_data.side_effect = error
        elif "Forecast" in query:
            nl_interface.model.predict.side_effect = error
        elif "Recommend" in query:
            nl_interface.strategy_manager.generate_signals.side_effect = error
        elif "risk" in query.lower():
            nl_interface.risk_manager.calculate_position_size.side_effect = error
        else:
            nl_interface.portfolio_manager.get_portfolio_status.side_effect = error
        
        # Process query and verify error handling
        response = nl_interface.process_query(query)
        assert "error" in response.text.lower()
        
        # Verify system is still functional
        nl_interface.market_analyzer.get_market_data.side_effect = None
        nl_interface.model.predict.side_effect = None
        nl_interface.strategy_manager.generate_signals.side_effect = None
        nl_interface.risk_manager.calculate_position_size.side_effect = None
        nl_interface.portfolio_manager.get_portfolio_status.side_effect = None
        
        recovery_response = nl_interface.process_query("Analyze BTC")
        assert "error" not in recovery_response.text.lower()

def test_long_running_queries(nl_interface):
    """Test handling of long-running queries."""
    # Simulate long-running operations
    def slow_market_data(*args, **kwargs):
        time.sleep(1)
        return mock_market_data()
    
    def slow_prediction(*args, **kwargs):
        time.sleep(1)
        return mock_forecast()
    
    nl_interface.market_analyzer.get_market_data = Mock(side_effect=slow_market_data)
    nl_interface.model.predict = Mock(side_effect=slow_prediction)
    
    # Process multiple queries
    queries = [
        "What's the price forecast for BTC?",
        "Analyze ETH",
        "Should I buy SOL?",
        "Compare BTC and ETH"
    ]
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(nl_interface.process_query, query) for query in queries]
        responses = [future.result() for future in as_completed(futures)]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Verify responses
    for response in responses:
        assert response.text is not None
        assert response.visualization is not None
        assert response.metadata is not None
    
    # Verify parallel execution
    assert total_time < 3.0, f"Long-running queries took too long: {total_time:.2f}s"

def test_resource_cleanup(nl_interface):
    """Test proper resource cleanup after processing queries."""
    import gc
    
    # Process multiple queries
    queries = ["What's the price forecast for BTC?"] * 10
    
    # Get initial resource counts
    initial_objects = len(gc.get_objects())
    
    for query in queries:
        response = nl_interface.process_query(query)
        assert response.text is not None
    
    # Force garbage collection
    gc.collect()
    
    # Get final resource counts
    final_objects = len(gc.get_objects())
    object_increase = final_objects - initial_objects
    
    print("\nResource Cleanup Test Results:")
    print(f"Initial Objects: {initial_objects}")
    print(f"Final Objects: {final_objects}")
    print(f"Object Increase: {object_increase}")
    
    # Assert resource cleanup
    assert object_increase < 1000, f"Too many objects not cleaned up: {object_increase}" 