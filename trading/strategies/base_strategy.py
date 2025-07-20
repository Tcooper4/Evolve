"""
Base Strategy Module

Defines the abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, description: str = ""):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
        """
        self.name = name
        self.description = description
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate trading signals from market data.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with trading signals
        """

    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get the parameter space for optimization.
        
        Returns:
            Dictionary defining parameter ranges and types
        """

    def set_parameters(self, parameters: Dict[str, Any]):
        """Set strategy parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        self.parameters.update(parameters)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters.
        
        Returns:
            Copy of current parameters dictionary
        """
        return self.parameters.copy()

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        return all(col in data.columns for col in required_columns)

    def get_name(self) -> str:
        """Get strategy name.
        
        Returns:
            Strategy name
        """
        return self.name

    def get_description(self) -> str:
        """Get strategy description.
        
        Returns:
            Strategy description
        """
        return self.description

    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"

    def __repr__(self) -> str:
        """Detailed string representation of the strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}', parameters={self.parameters})"
