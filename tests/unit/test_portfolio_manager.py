import pytest
from trading.portfolio.portfolio_manager import PortfolioManager

def test_portfolio_manager():
    """Test the portfolio manager functionality."""
    pm = PortfolioManager(initial_cash=100000.0)
    
    # Test adding assets
    pm.add_asset('AAPL', 10, 150.0)
    pm.add_asset('GOOGL', 5, 2800.0)
    
    # Test portfolio value
    assert pm.get_portfolio_value() == 100000.0 + (10 * 150.0) + (5 * 2800.0)
    
    # Test updating prices
    pm.update_prices({'AAPL': 160.0, 'GOOGL': 2900.0})
    assert pm.get_portfolio_value() == 100000.0 + (10 * 160.0) + (5 * 2900.0)
    
    # Test removing assets
    pm.remove_asset('AAPL', 5)
    assert pm.get_portfolio_value() == 100000.0 + (5 * 160.0) + (5 * 2900.0)
    
    # Test rebalancing
    target_weights = {'AAPL': 0.5, 'GOOGL': 0.5}
    pm.rebalance(target_weights)
    total_value = pm.get_portfolio_value()
    assert abs(pm.positions['AAPL'] * pm.prices['AAPL'] / total_value - 0.5) < 0.01
    assert abs(pm.positions['GOOGL'] * pm.prices['GOOGL'] / total_value - 0.5) < 0.01 