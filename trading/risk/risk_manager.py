import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class RiskManager:
    def __init__(self, returns: pd.Series, confidence_level: float = 0.95):
        self.returns = returns
        self.confidence_level = confidence_level

    def calculate_var(self) -> float:
        """Calculate Value at Risk (VaR) for the given returns."""
        return np.percentile(self.returns, (1 - self.confidence_level) * 100)

    def calculate_cvar(self) -> float:
        """Calculate Conditional Value at Risk (CVaR) for the given returns."""
        var = self.calculate_var()
        return self.returns[self.returns <= var].mean()

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate the Sharpe ratio for the given returns."""
        excess_returns = self.returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() 