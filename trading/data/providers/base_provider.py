"""
Base Data Provider

This module defines the base interface for all data providers in the system.
All providers must implement this interface to be compatible with the data manager.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProviderConfig:
    """Configuration for a data provider."""
    name: str
    enabled: bool = True
    priority: int = 1
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    retry_attempts: int = 3
    custom_config: Optional[Dict[str, Any]] = None

@dataclass
class ProviderStatus:
    """Status information for a data provider."""
    name: str
    enabled: bool
    is_available: bool
    last_request: Optional[datetime] = None
    last_success: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    current_error: Optional[str] = None
    rate_limit_remaining: int = 0

class BaseDataProvider(ABC):
    """Base interface for all data providers."""
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self.status = ProviderStatus(
            name=config.name,
            enabled=config.enabled,
            is_available=True
        )
        self._setup()

    def _setup(self) -> None:
        """Setup method called during initialization.
        
        Override this method to perform any provider-specific setup.
        """
        pass

    @abstractmethod
    def fetch(self, symbol: str, interval: str = '1d', **kwargs) -> pd.DataFrame:
        """Fetch data for a given symbol and interval.
        
        Args:
            symbol: Stock symbol
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            Exception: If data fetching fails
        """
        pass
    
    @abstractmethod
    def fetch_multiple(self, symbols: List[str], interval: str = '1d', **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            interval: Data interval (1d, 1h, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping symbols to DataFrames
            
        Raises:
            Exception: If data fetching fails
        """
        pass
    
    def enable(self) -> None:
        """Enable the provider."""
        self.config.enabled = True
        self.status.enabled = True
        self.logger.info(f"Provider {self.config.name} enabled")

    def disable(self) -> None:
        """Disable the provider."""
        self.config.enabled = False
        self.status.enabled = False
        self.logger.info(f"Provider {self.config.name} disabled")

    def is_enabled(self) -> bool:
        """Check if the provider is enabled."""
        return self.config.enabled and self.status.enabled
    
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self.status.is_available
    
    def get_status(self) -> ProviderStatus:
        """Get the current status of the provider."""
        return self.status
    
    def get_config(self) -> ProviderConfig:
        """Get the provider configuration."""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update provider configuration.
        
        Args:
            new_config: New configuration values
        """
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif self.config.custom_config is not None:
                self.config.custom_config[key] = value
        
        self.logger.info(f"Updated configuration for provider {self.config.name}")

    def validate_symbol(self, symbol: str) -> bool:
        """Validate a symbol format.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            bool: True if symbol is valid
        """
        if not symbol or not isinstance(symbol, str):
            return False
        return len(symbol.strip()) > 0
    
    def validate_interval(self, interval: str) -> bool:
        """Validate an interval format.
        
        Args:
            interval: Interval to validate
            
        Returns:
            bool: True if interval is valid
        """
        valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1mo']
        return interval in valid_intervals
    
    def handle_error(self, error: Exception) -> None:
        """Handle errors during data fetching.
        
        Args:
            error: Exception that occurred
        """
        self.status.failed_requests += 1
        self.status.current_error = str(error)
        self.status.is_available = False
        self.logger.error(f"Provider {self.config.name} error: {error}")
    
    def _update_status_on_request(self) -> None:
        """Update status when a request is made."""
        self.status.last_request = datetime.now()
        self.status.total_requests += 1
        self.status.current_error = None

    def _update_status_on_success(self) -> None:
        """Update status when a request succeeds."""
        self.status.last_success = datetime.now()
        self.status.successful_requests += 1
        self.status.is_available = True
        self.status.current_error = None

    def _update_status_on_failure(self, error: str) -> None:
        """Update status when a request fails."""
        self.status.failed_requests += 1
        self.status.current_error = error
        self.status.is_available = False

    def get_metadata(self) -> Dict[str, Any]:
        """Get provider metadata.
        
        Returns:
            Dictionary with provider metadata
        """
        return {
            'name': self.config.name,
            'enabled': self.is_enabled(),
            'available': self.is_available(),
            'status': self.get_status(),
            'config': self.get_config()
        } 