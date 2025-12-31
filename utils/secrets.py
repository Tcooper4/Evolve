"""
Secrets management for EVOLVE trading system

Manages API keys and secrets securely from environment variables.
"""

import os
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manages API keys and secrets"""
    
    def __init__(self):
        self.secrets: Dict[str, Optional[str]] = {}
        self._load_secrets()
    
    def _load_secrets(self) -> None:
        """Load secrets from environment"""
        secret_keys = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY',
            'POLYGON_API_KEY',
            'ALPHA_VANTAGE_API_KEY',
            'FINNHUB_API_KEY',
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY',
            'IBKR_USERNAME',
            'IBKR_PASSWORD',
            'REDIS_PASSWORD',
            'DATABASE_PASSWORD',
            'FLASK_SECRET_KEY',
        ]
        
        for key in secret_keys:
            value = os.getenv(key)
            if value:
                self.secrets[key] = value
                # Log that secret was loaded (but not the value)
                logger.debug(f"Loaded secret: {key}")
            else:
                self.secrets[key] = None
                logger.debug(f"Secret not found: {key}")
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Get secret value.
        
        Args:
            key: Secret key name
            
        Returns:
            Secret value or None if not found
        """
        return self.secrets.get(key)
    
    def set_secret(self, key: str, value: str) -> None:
        """
        Set secret value (for testing or runtime updates).
        
        Args:
            key: Secret key name
            value: Secret value
        """
        self.secrets[key] = value
        logger.debug(f"Set secret: {key}")
    
    def has_secret(self, key: str) -> bool:
        """
        Check if secret exists and is not empty.
        
        Args:
            key: Secret key name
            
        Returns:
            True if secret exists and is not empty
        """
        return self.secrets.get(key) is not None and self.secrets.get(key) != ""
    
    def validate_secrets(self, required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate all secrets are present.
        
        Args:
            required_keys: List of required secret keys (default: all keys)
            
        Returns:
            Dictionary with validation results
        """
        if required_keys is None:
            required_keys = list(self.secrets.keys())
        
        missing = []
        present = []
        
        for key in required_keys:
            if self.has_secret(key):
                present.append(key)
            else:
                missing.append(key)
        
        return {
            'valid': len(missing) == 0,
            'missing': missing,
            'present': present,
            'total_checked': len(required_keys),
        }
    
    def get_secret_status(self) -> Dict[str, bool]:
        """
        Get status of all secrets.
        
        Returns:
            Dictionary mapping secret keys to their presence status
        """
        return {key: self.has_secret(key) for key in self.secrets.keys()}
    
    def reload_secrets(self) -> None:
        """Reload secrets from environment"""
        self.secrets.clear()
        self._load_secrets()
        logger.info("Secrets reloaded from environment")


# Global secrets manager instance
_secrets_manager = SecretsManager()


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance"""
    return _secrets_manager

