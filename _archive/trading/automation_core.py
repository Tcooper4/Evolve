"""
Automation Core Bridge Module

This module provides a bridge to the system.infra.agents.services.automation_core module
for backward compatibility and cleaner import paths.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from system.infra.agents.services.automation_core import (
        AutomationConfig,
        AutomationCoreService,
    )
    
    # Create alias for backward compatibility
    AutomationCore = AutomationCoreService
    
    __all__ = ["AutomationCore", "AutomationConfig", "AutomationCoreService"]
except ImportError as e:
    # Log the import error for debugging
    logger.warning(f"Could not import AutomationCore from system.infra.agents.services.automation_core: {e}")
    
    # Create stub classes that raise informative errors when instantiated
    class AutomationCore:
        """Stub AutomationCore class when import fails."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"AutomationCore not available. Original error: {e}. "
                "Please ensure system.infra.agents.services.automation_core is properly installed."
            )
    
    class AutomationConfig:
        """Stub AutomationConfig class when import fails."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"AutomationConfig not available. Original error: {e}. "
                "Please ensure system.infra.agents.services.automation_core is properly installed."
            )
    
    __all__ = ["AutomationCore", "AutomationConfig"]

