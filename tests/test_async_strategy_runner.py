"""
Tests for Async Strategy Runner

This module tests the async strategy runner functionality including:
- Parallel strategy execution
- Error handling and timeouts
- Ensemble result creation
- Performance monitoring
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from trading.strategies.strategy_runner import AsyncStrategyRunner


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, name: str = "MockStrategy"):
        self.name = name
        self.description = f"Mock {name} strategy"
        self.parameters = {}
    
    def set_parameters(self, params: dict):
        self.parameters = params
    
    def get_parameters(self) -> dict:
        return self.parameters
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mock signals."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = np.random.choice([-1, 0, 1], size=len(data))
        signals['confidence'] = np.random.uniform(0.5, 1.0, size=len(data))
        return signals


def create_sample_data(rows: int = 100) -> pd.DataFrame:
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=rows, freq='D')
    prices = 100 + np.cumsum(np.random.randn(rows) * 0.5)
    
    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, rows)
    }, index=dates)
    
    return data


class TestAsyncStrategyRunner:
    """Test cases for AsyncStrategyRunner."""
    
    @pytest.fixture
    def runner(self):
        """Create a test runner instance."""
        return AsyncStrategyRunner({
            "max_concurrent_strategies": 3,
            "strategy_timeout": 5,
            "enable_ensemble": True,
            "ensemble_method": "weighted",
            "log_performance": True
        })
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_sample_data(50)
    
    @pytest.mark.asyncio
    async def test_initialization(self, runner):
        """Test runner initialization."""
        assert runner.max_concurrent == 3
        assert runner.strategy_timeout == 5
        assert runner.enable_ensemble is True
        assert runner.ensemble_method == "weighted"
        assert runner.log_performance is True
        assert len(runner.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_run_single_strategy_async(self, runner, sample_data):
        """Test running a single strategy asynchronously."""
        strategy = MockStrategy("TestStrategy")
        parameters = {"param1": 10, "param2": 20}
        
        result = await runner._run_single_strategy_async(
            strategy, sample_data, parameters, "TestStrategy"
        )
        
        assert result["success"] is True
        assert result["strategy_name"] == "TestStrategy"
        assert result["parameters_used"] == parameters
        assert "execution_time" in result
        assert "timestamp" in result
        assert "signals" in result
        assert "performance_metrics" in result
    
    @pytest.mark.asyncio
    async def test_strategy_timeout(self, runner, sample_data):
        """Test strategy timeout handling."""
        # Create a slow strategy
        slow_strategy = MockStrategy("SlowStrategy")
        
        # Mock the strategy to be slow
        async def slow_generate_signals(data):
            await asyncio.sleep(10)  # Sleep longer than timeout
            return pd.DataFrame()
        
        slow_strategy.generate_signals = slow_generate_signals
        
        # Set very short timeout
        runner.strategy_timeout = 0.1
        
        result = await runner._run_single_strategy_async(
            slow_strategy, sample_data, {}, "SlowStrategy"
        )
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_strategy_error_handling(self, runner, sample_data):
        """Test strategy error handling."""
        # Create a strategy that raises an exception
        error_strategy = MockStrategy("ErrorStrategy")
        
        def error_generate_signals(data):
            raise ValueError("Test error")
        
        error_strategy.generate_signals = error_generate_signals
        
        result = await runner._run_single_strategy_async(
            error_strategy, sample_data, {}, "ErrorStrategy"
        )
        
        assert result["success"] is False
        assert "Test error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_parallel_strategy_execution(self, runner, sample_data):
        """Test parallel execution of multiple strategies."""
        strategies = [
            MockStrategy("Strategy1"),
            MockStrategy("Strategy2"),
            MockStrategy("Strategy3")
        ]
        
        parameters = {
            "Strategy1": {"param1": 10},
            "Strategy2": {"param2": 20},
            "Strategy3": {"param3": 30}
        }
        
        result = await runner.run_strategies_parallel(
            strategies, sample_data, parameters
        )
        
        assert result["success"] is True
        assert len(result["individual_results"]) == 3
        
        # Check that all strategies completed
        for strategy_name in ["Strategy1", "Strategy2", "Strategy3"]:
            assert strategy_name in result["individual_results"]
            assert result["individual_results"][strategy_name]["success"] is True
        
        # Check execution stats
        stats = result["execution_stats"]
        assert stats["total_strategies"] == 3
        assert stats["successful_strategies"] == 3
        assert stats["failed_strategies"] == 0
        assert stats["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_failures(self, runner, sample_data):
        """Test parallel execution when some strategies fail."""
        # Create strategies with one that fails
        strategies = [
            MockStrategy("GoodStrategy1"),
            MockStrategy("GoodStrategy2")
        ]
        
        # Create a failing strategy
        failing_strategy = MockStrategy("FailingStrategy")
        def failing_generate_signals(data):
            raise RuntimeError("Simulated failure")
        failing_strategy.generate_signals = failing_generate_signals
        
        strategies.append(failing_strategy)
        
        result = await runner.run_strategies_parallel(
            strategies, sample_data, {}
        )
        
        assert result["success"] is True
        assert len(result["individual_results"]) == 3
        
        # Check results
        assert result["individual_results"]["GoodStrategy1"]["success"] is True
        assert result["individual_results"]["GoodStrategy2"]["success"] is True
        assert result["individual_results"]["FailingStrategy"]["success"] is False
        
        # Check stats
        stats = result["execution_stats"]
        assert stats["total_strategies"] == 3
        assert stats["successful_strategies"] == 2
        assert stats["failed_strategies"] == 1
    
    @pytest.mark.asyncio
    async def test_ensemble_result_creation(self, runner, sample_data):
        """Test ensemble result creation."""
        # Create successful strategy results
        strategy_results = []
        for i in range(3):
            signals = pd.DataFrame(index=sample_data.index)
            signals['signal'] = np.random.choice([-1, 0, 1], size=len(sample_data))
            signals['confidence'] = np.random.uniform(0.5, 1.0, size=len(sample_data))
            
            result = {
                "success": True,
                "signals": signals,
                "performance_metrics": {
                    "sharpe_ratio": np.random.uniform(0.5, 2.0),
                    "total_return": np.random.uniform(-0.1, 0.3)
                },
                "data": sample_data
            }
            strategy_results.append(result)
        
        ensemble_result = await runner._create_ensemble_result(
            strategy_results, {"method": "weighted"}
        )
        
        assert ensemble_result["success"] is True
        assert "combined_signals" in ensemble_result
        assert "performance_metrics" in ensemble_result
        assert "ensemble_config" in ensemble_result
        
        config = ensemble_result["ensemble_config"]
        assert config["method"] == "weighted"
        assert config["num_strategies"] == 3
        assert len(config["weights"]) == 3
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, runner, sample_data):
        """Test performance metrics calculation."""
        # Create signals
        signals = pd.DataFrame(index=sample_data.index)
        signals['signal'] = np.random.choice([-1, 0, 1], size=len(sample_data))
        signals['confidence'] = np.random.uniform(0.5, 1.0, size=len(sample_data))
        
        metrics = runner._calculate_performance_metrics(sample_data, signals)
        
        assert "total_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        
        # Check that metrics are reasonable
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["volatility"], float)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["max_drawdown"], float)
        assert isinstance(metrics["win_rate"], float)
    
    @pytest.mark.asyncio
    async def test_concurrent_limit(self, runner, sample_data):
        """Test that concurrent execution is limited."""
        # Create many strategies
        strategies = [MockStrategy(f"Strategy{i}") for i in range(10)]
        
        # Track execution order
        execution_order = []
        
        async def tracked_execution(strategy, data, params, name):
            execution_order.append(name)
            await asyncio.sleep(0.1)  # Small delay
            return await runner._execute_strategy_core(strategy, data)
        
        # Patch the execution method
        with patch.object(runner, '_execute_strategy_core', side_effect=tracked_execution):
            result = await runner.run_strategies_parallel(
                strategies[:5], sample_data, {}
            )
        
        # Check that execution was limited by semaphore
        assert result["success"] is True
        assert len(result["individual_results"]) == 5
    
    @pytest.mark.asyncio
    async def test_execution_history(self, runner, sample_data):
        """Test execution history tracking."""
        strategies = [MockStrategy("TestStrategy")]
        
        # Clear history
        runner.clear_history()
        assert len(runner.execution_history) == 0
        
        # Run strategies
        await runner.run_strategies_parallel(strategies, sample_data, {})
        
        # Check history
        history = runner.get_execution_history()
        assert len(history) == 1
        
        entry = history[0]
        assert "timestamp" in entry
        assert "total_strategies" in entry
        assert "successful_strategies" in entry
        assert "execution_time" in entry
    
    @pytest.mark.asyncio
    async def test_empty_strategies_list(self, runner, sample_data):
        """Test handling of empty strategies list."""
        result = await runner.run_strategies_parallel([], sample_data, {})
        
        assert result["success"] is False
        assert "No valid strategies" in result["error"]
    
    @pytest.mark.asyncio
    async def test_invalid_strategy_in_registry(self, runner, sample_data):
        """Test handling of invalid strategy names."""
        # Mock registry to return None for invalid strategy
        with patch.object(runner.strategy_registry, 'get_strategy', return_value=None):
            result = await runner.run_strategies_parallel(
                ["InvalidStrategy"], sample_data, {}
            )
        
        assert result["success"] is False
        assert "No valid strategies" in result["error"]


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_run_rsi_function(self):
        """Test run_rsi convenience function."""
        from trading.strategies.strategy_runner import run_rsi
        
        result = await run_rsi()
        assert result["success"] is False
        assert "Data and parameters required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_run_macd_function(self):
        """Test run_macd convenience function."""
        from trading.strategies.strategy_runner import run_macd
        
        result = await run_macd()
        assert result["success"] is False
        assert "Data and parameters required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_run_bollinger_bands_function(self):
        """Test run_bollinger_bands convenience function."""
        from trading.strategies.strategy_runner import run_bollinger_bands
        
        result = await run_bollinger_bands()
        assert result["success"] is False
        assert "Data and parameters required" in result["error"]
    
    @pytest.mark.asyncio
    async def test_run_strategies_parallel_example(self):
        """Test the example function."""
        from trading.strategies.strategy_runner import run_strategies_parallel_example
        
        data = create_sample_data(50)
        result = await run_strategies_parallel_example(data)
        
        # Should return a result structure
        assert isinstance(result, dict)
        assert "success" in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 