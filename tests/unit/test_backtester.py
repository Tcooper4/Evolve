import pytest
import pandas as pd
import numpy as np
from trading.backtesting.backtester import Backtester, BacktestError

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

def test_run_backtest():
    """Test detailed backtest execution."""
    # Create test data with known patterns
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'AAPL': np.linspace(100, 200, 100),  # Steady uptrend
        'GOOGL': np.sin(np.linspace(0, 4*np.pi, 100)) * 50 + 200,  # Oscillating
        'MSFT': np.random.normal(150, 10, 100)  # Random walk
    }, index=dates)
    
    def test_strategy(data):
        # Simple momentum strategy
        returns = data.pct_change()
        return {
            'AAPL': 1.0 if returns['AAPL'].iloc[-1] > 0 else 0.0,
            'GOOGL': 1.0 if returns['GOOGL'].iloc[-1] > 0 else 0.0,
            'MSFT': 1.0 if returns['MSFT'].iloc[-1] > 0 else 0.0
        }
    
    bt = Backtester(data, initial_cash=100000.0)
    results = bt.run_backtest(test_strategy)
    
    # Test results structure
    assert isinstance(results, dict)
    assert 'portfolio_values' in results
    assert 'trades' in results
    assert 'positions' in results
    assert 'returns' in results
    
    # Test portfolio values
    assert len(results['portfolio_values']) == len(data)
    assert results['portfolio_values'][0] == bt.initial_cash
    assert all(v >= 0 for v in results['portfolio_values'])
    
    # Test trades
    assert isinstance(results['trades'], list)
    assert all(isinstance(trade, dict) for trade in results['trades'])
    if results['trades']:
        assert all('timestamp' in trade for trade in results['trades'])
        assert all('asset' in trade for trade in results['trades'])
        assert all('quantity' in trade for trade in results['trades'])
        assert all('price' in trade for trade in results['trades'])

def test_cumulative_returns():
    """Test cumulative returns calculation."""
    # Create test data with known returns
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'AAPL': [100, 110, 121, 133.1, 146.4, 161.1, 177.2, 194.9, 214.4, 235.8],  # 10% daily return
        'GOOGL': [200, 190, 180, 171, 162.5, 154.3, 146.6, 139.3, 132.3, 125.7]  # -5% daily return
    }, index=dates)
    
    def test_strategy(data):
        return {'AAPL': 1.0, 'GOOGL': 0.0}  # Only invest in AAPL
    
    bt = Backtester(data, initial_cash=100000.0)
    results = bt.run_backtest(test_strategy)
    
    # Test cumulative returns
    returns = results['returns']
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(data)
    assert returns.iloc[0] == 0.0  # First day return should be 0
    assert returns.iloc[-1] > 0.0  # Should be positive due to AAPL's uptrend
    
    # Test return calculation
    expected_return = (235.8 / 100) - 1  # Expected return for AAPL
    assert abs(returns.iloc[-1] - expected_return) < 0.01  # Allow small rounding error

def test_max_drawdown():
    """Test maximum drawdown calculation."""
    # Create test data with a drawdown pattern
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    data = pd.DataFrame({
        'AAPL': [100, 110, 120, 130, 140, 130, 120, 110, 100, 90,  # Drawdown
                 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]  # Recovery
    }, index=dates)
    
    def test_strategy(data):
        return {'AAPL': 1.0}
    
    bt = Backtester(data, initial_cash=100000.0)
    results = bt.run_backtest(test_strategy)
    
    # Test max drawdown
    metrics = bt.get_performance_metrics()
    assert 'max_drawdown' in metrics
    assert metrics['max_drawdown'] >= 0
    assert metrics['max_drawdown'] <= 1
    
    # Calculate expected drawdown
    peak = 140
    trough = 90
    expected_drawdown = (peak - trough) / peak
    assert abs(metrics['max_drawdown'] - expected_drawdown) < 0.01

def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Create test data with known volatility and return
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')  # One year of trading days
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns with 0.1% mean and 2% std
    data = pd.DataFrame({
        'AAPL': 100 * (1 + returns).cumprod()  # Price series
    }, index=dates)
    
    def test_strategy(data):
        return {'AAPL': 1.0}
    
    bt = Backtester(data, initial_cash=100000.0)
    results = bt.run_backtest(test_strategy)
    
    # Test Sharpe ratio
    metrics = bt.get_performance_metrics()
    assert 'sharpe_ratio' in metrics
    assert isinstance(metrics['sharpe_ratio'], float)
    
    # Calculate expected Sharpe ratio
    expected_sharpe = np.sqrt(252) * (0.001 / 0.02)  # Annualized Sharpe ratio
    assert abs(metrics['sharpe_ratio'] - expected_sharpe) < 0.1  # Allow some deviation

def test_trade_metrics():
    """Test trade count and win rate calculations."""
    # Create test data with alternating wins and losses
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'AAPL': [100, 110, 100, 110, 100, 110, 100, 110, 100, 110],  # Alternating +10% and -10%
        'GOOGL': [200, 190, 200, 190, 200, 190, 200, 190, 200, 190]  # Alternating -5% and +5%
    }, index=dates)
    
    def test_strategy(data):
        # Strategy that trades every day
        return {
            'AAPL': 1.0 if data['AAPL'].iloc[-1] > data['AAPL'].iloc[-2] else -1.0,
            'GOOGL': 1.0 if data['GOOGL'].iloc[-1] > data['GOOGL'].iloc[-2] else -1.0
        }
    
    bt = Backtester(data, initial_cash=100000.0)
    results = bt.run_backtest(test_strategy)
    
    # Test trade metrics
    metrics = bt.get_performance_metrics()
    assert 'total_trades' in metrics
    assert 'win_rate' in metrics
    assert 'avg_trade_return' in metrics
    
    # Verify trade count
    assert metrics['total_trades'] > 0
    assert metrics['total_trades'] == len(results['trades'])
    
    # Verify win rate
    assert 0 <= metrics['win_rate'] <= 1
    expected_win_rate = 0.5  # Strategy should win half the time
    assert abs(metrics['win_rate'] - expected_win_rate) < 0.1
    
    # Verify average trade return
    assert isinstance(metrics['avg_trade_return'], float)
    assert metrics['avg_trade_return'] != 0

def test_error_handling():
    """Test backtester error handling."""
    # Test with empty data
    with pytest.raises(BacktestError) as exc_info:
        bt = Backtester(pd.DataFrame(), initial_cash=100000.0)
    assert "Empty data" in str(exc_info.value)
    
    # Test with invalid initial cash
    with pytest.raises(BacktestError) as exc_info:
        bt = Backtester(pd.DataFrame({'AAPL': [100]}), initial_cash=-100000.0)
    assert "Invalid initial cash" in str(exc_info.value)
    
    # Test with invalid strategy
    data = pd.DataFrame({'AAPL': [100, 110, 120]})
    bt = Backtester(data, initial_cash=100000.0)
    with pytest.raises(BacktestError) as exc_info:
        bt.run_backtest(lambda x: {'INVALID': 1.0})
    assert "Invalid asset" in str(exc_info.value)
    
    # Test with missing data
    data_with_nan = data.copy()
    data_with_nan.loc[data_with_nan.index[1], 'AAPL'] = np.nan
    with pytest.raises(BacktestError) as exc_info:
        bt = Backtester(data_with_nan, initial_cash=100000.0)
    assert "Missing data" in str(exc_info.value)

if __name__ == '__main__':
    pytest.main([__file__]) 