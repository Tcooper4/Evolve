import pytest
import pandas as pd
import numpy as np
from trading.risk.risk_manager import RiskManager

def test_risk_manager():
    """Test the risk manager functionality."""
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    rm = RiskManager(returns, confidence_level=0.95)
    
    # Test VaR calculation
    var = rm.calculate_var()
    assert var == pytest.approx(-0.02, abs=1e-2)
    
    # Test CVaR calculation
    cvar = rm.calculate_cvar()
    assert cvar == pytest.approx(-0.02, abs=1e-2)
    
    # Test Sharpe ratio calculation
    sharpe = rm.calculate_sharpe_ratio(risk_free_rate=0.0)
    expected_sharpe = returns.mean() / returns.std()
    assert sharpe == pytest.approx(expected_sharpe, abs=1e-2) 