import pytest
import pandas as pd
import numpy as np
from trading.backtesting.backtester import Backtester

def test_backtester():
    """Test the backtester functionality."""
    data = pd.DataFrame({
        'AAPL': [150.0, 160.0, 155.0, 165.0, 170.0],
        'GOOGL': [2800.0, 2900.0, 2850.0, 2950.0, 3000.0]
    })
    
    def simple_strategy(data):
        return {'AAPL': 0.5, 'GOOGL': 0.5}
    
    bt = Backtester(data, initial_cash=100000.0)
    bt.run_backtest(simple_strategy)
    
    # Test portfolio value
    assert bt.portfolio_values[-1] > bt.initial_cash
    
    # Test performance metrics
    metrics = bt.get_performance_metrics()
    assert metrics['total_return'] > 0
    assert metrics['sharpe_ratio'] > 0
    assert metrics['max_drawdown'] >= 0 