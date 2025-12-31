"""
Feature flags system for EVOLVE trading system

Manages feature flags to enable/disable features without code changes.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FeatureFlags:
    """Manages feature flags"""
    
    def __init__(self):
        self.flags: Dict[str, bool] = {}
        self._load_flags()
    
    def _load_flags(self) -> None:
        """Load flags from environment"""
        flag_keys = [
            'FEATURE_REAL_TIME_STREAMING',
            'FEATURE_ADVANCED_ORDERS',
            'FEATURE_RL_TRADING',
            'FEATURE_AUTO_REBALANCE',
            'FEATURE_GPU_ACCELERATION',
            'FEATURE_DISASTER_RECOVERY',
            'FEATURE_BROKER_REDUNDANCY',
        ]
        
        for key in flag_keys:
            value = os.getenv(key, 'false').lower()
            self.flags[key] = value in ('true', '1', 'yes', 'on')
            logger.debug(f"Feature flag {key}: {self.flags[key]}")
    
    def is_enabled(self, feature: str) -> bool:
        """
        Check if feature is enabled.
        
        Args:
            feature: Feature name (with or without FEATURE_ prefix)
            
        Returns:
            True if feature is enabled, False otherwise
        """
        # Normalize feature name
        if not feature.startswith('FEATURE_'):
            feature = f'FEATURE_{feature.upper()}'
        else:
            feature = feature.upper()
        
        return self.flags.get(feature, False)
    
    def set_flag(self, feature: str, enabled: bool) -> None:
        """
        Set a feature flag (runtime override).
        
        Args:
            feature: Feature name
            enabled: Whether feature is enabled
        """
        if not feature.startswith('FEATURE_'):
            feature = f'FEATURE_{feature.upper()}'
        else:
            feature = feature.upper()
        
        self.flags[feature] = enabled
        logger.info(f"Feature flag {feature} set to {enabled}")
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all feature flag statuses"""
        return self.flags.copy()
    
    def reload_flags(self) -> None:
        """Reload flags from environment"""
        self.flags.clear()
        self._load_flags()
        logger.info("Feature flags reloaded from environment")


# Global feature flags instance
_feature_flags = FeatureFlags()


def is_feature_enabled(feature: str) -> bool:
    """
    Check if a feature is enabled.
    
    Args:
        feature: Feature name (e.g., 'REAL_TIME_STREAMING' or 'FEATURE_REAL_TIME_STREAMING')
        
    Returns:
        True if feature is enabled, False otherwise
    
    Example:
        if is_feature_enabled('REAL_TIME_STREAMING'):
            enable_streaming()
    """
    return _feature_flags.is_enabled(feature)


def get_feature_flags() -> FeatureFlags:
    """Get the global feature flags instance"""
    return _feature_flags

