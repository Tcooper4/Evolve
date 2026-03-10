"""
Walk-forward validation for trading strategies.
Re-exports WalkForwardValidator from trading.validation.walk_forward_utils
so imports from trading.models.walk_forward_validator resolve.
"""
from trading.validation.walk_forward_utils import WalkForwardValidator

__all__ = ["WalkForwardValidator"]
