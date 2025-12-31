"""
Strategy Execution Validation Module

Provides production-grade validation for strategy execution, signals, and state.
Ensures no silent failures and validates all inputs/outputs.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StrategyExecutionValidator:
    """Validates strategy execution for production use."""

    @staticmethod
    def validate_signal(signal: Union[Dict[str, Any], Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a trading signal.

        Args:
            signal: Trading signal (dict or object with attributes)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Handle dict or object
            if isinstance(signal, dict):
                action = signal.get('action')
                symbol = signal.get('symbol')
                quantity = signal.get('quantity')
            else:
                # Try to get attributes
                action = getattr(signal, 'action', None) or getattr(signal, 'direction', None)
                symbol = getattr(signal, 'symbol', None)
                quantity = getattr(signal, 'quantity', None) or getattr(signal, 'size', None)

            # Check if signal is None
            if signal is None:
                return False, "Strategy returned None signal"

            # Validate required fields
            required_fields = ['action', 'symbol', 'quantity']
            missing_fields = []

            if action is None:
                missing_fields.append('action')
            if symbol is None:
                missing_fields.append('symbol')
            if quantity is None:
                missing_fields.append('quantity')

            if missing_fields:
                return False, f"Signal missing required fields: {', '.join(missing_fields)}"

            # Validate action values
            valid_actions = ['buy', 'sell', 'hold', 'long', 'short']
            if isinstance(action, str):
                action_lower = action.lower()
            else:
                action_lower = str(action).lower()

            if action_lower not in valid_actions:
                return False, f"Invalid action: {action}. Must be one of {valid_actions}"

            # Validate quantity
            if isinstance(quantity, (int, float)):
                if quantity < 0:
                    return False, f"Negative quantity: {quantity}"
                if np.isnan(quantity) or np.isinf(quantity):
                    return False, f"Invalid quantity value: {quantity}"
            else:
                return False, f"Invalid quantity type: {type(quantity)}. Must be numeric"

            # Validate symbol
            if not isinstance(symbol, str):
                return False, f"Invalid symbol type: {type(symbol)}. Must be string"
            if not symbol.strip():
                return False, "Symbol cannot be empty"

            # Validate price if present
            if isinstance(signal, dict):
                price = signal.get('price') or signal.get('entry_price')
            else:
                price = getattr(signal, 'price', None) or getattr(signal, 'entry_price', None)

            if price is not None:
                if isinstance(price, (int, float)):
                    if price <= 0:
                        return False, f"Invalid price: {price}. Must be positive"
                    if np.isnan(price) or np.isinf(price):
                        return False, f"Invalid price value: {price}"
                else:
                    return False, f"Invalid price type: {type(price)}. Must be numeric"

            return True, None

        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False, f"Signal validation error: {str(e)}"

    @staticmethod
    def validate_strategy_state(strategy: Any) -> tuple[bool, Optional[str]]:
        """
        Validate strategy is in valid state.

        Args:
            strategy: Strategy object

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if strategy is None:
                return False, "Strategy is None"

            # Check if strategy is initialized
            if hasattr(strategy, 'initialized'):
                if not strategy.initialized:
                    return False, "Strategy not initialized"
            elif hasattr(strategy, '_initialized'):
                if not strategy._initialized:
                    return False, "Strategy not initialized"

            # Check if strategy has data
            if hasattr(strategy, 'data'):
                if strategy.data is None:
                    return False, "Strategy has no data"
                if isinstance(strategy.data, pd.DataFrame) and strategy.data.empty:
                    return False, "Strategy data is empty"

            # Check if strategy has required methods
            if not hasattr(strategy, 'generate_signals'):
                return False, "Strategy missing generate_signals method"

            return True, None

        except Exception as e:
            logger.error(f"Error validating strategy state: {e}")
            return False, f"Strategy state validation error: {str(e)}"

    @staticmethod
    def validate_signals_dataframe(signals: pd.DataFrame, data: Optional[pd.DataFrame] = None) -> tuple[bool, Optional[str]]:
        """
        Validate signals DataFrame.

        Args:
            signals: Signals DataFrame
            data: Original data DataFrame (for index validation)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if signals is None:
                return False, "Signals DataFrame is None"

            if not isinstance(signals, pd.DataFrame):
                return False, f"Signals is not a DataFrame: {type(signals)}"

            if signals.empty:
                return False, "Signals DataFrame is empty"

            # Check for required columns
            required_columns = ['signal', 'position']
            missing_columns = [col for col in required_columns if col not in signals.columns]
            if missing_columns:
                return False, f"Signals missing required columns: {', '.join(missing_columns)}"

            # Check for NaN or infinite values
            if signals.isnull().any().any():
                return False, "Signals contain NaN values"

            if np.isinf(signals.select_dtypes(include=[np.number])).any().any():
                return False, "Signals contain infinite values"

            # Validate index alignment if data provided
            if data is not None and isinstance(data, pd.DataFrame):
                if not signals.index.equals(data.index):
                    return False, "Signals index does not match data index"

            return True, None

        except Exception as e:
            logger.error(f"Error validating signals DataFrame: {e}")
            return False, f"Signals DataFrame validation error: {str(e)}"

    @staticmethod
    def validate_execution_context(
        strategy: Any,
        market_data: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate execution context before running strategy.

        Args:
            strategy: Strategy object
            market_data: Market data DataFrame
            context: Additional context

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate strategy state
            is_valid, error = StrategyExecutionValidator.validate_strategy_state(strategy)
            if not is_valid:
                return False, error

            # Validate market data
            if market_data is not None:
                if not isinstance(market_data, pd.DataFrame):
                    return False, f"Market data is not a DataFrame: {type(market_data)}"
                if market_data.empty:
                    return False, "Market data is empty"
                if 'Close' not in market_data.columns:
                    return False, "Market data missing 'Close' column"

            return True, None

        except Exception as e:
            logger.error(f"Error validating execution context: {e}")
            return False, f"Execution context validation error: {str(e)}"

    @staticmethod
    def validate_strategy_result(result: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate strategy execution result.

        Args:
            result: Strategy execution result dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if result is None:
                return False, "Strategy result is None"

            if not isinstance(result, dict):
                return False, f"Strategy result is not a dict: {type(result)}"

            # Check for required keys
            required_keys = ['signals']
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                return False, f"Result missing required keys: {', '.join(missing_keys)}"

            # Validate signals in result
            signals = result.get('signals')
            if signals is None:
                return False, "Result signals is None"

            # Validate signals DataFrame if present
            if isinstance(signals, pd.DataFrame):
                is_valid, error = StrategyExecutionValidator.validate_signals_dataframe(signals)
                if not is_valid:
                    return False, f"Result signals validation failed: {error}"

            return True, None

        except Exception as e:
            logger.error(f"Error validating strategy result: {e}")
            return False, f"Strategy result validation error: {str(e)}"


def execute_strategy_with_validation(
    strategy: Any,
    market_data: pd.DataFrame,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute strategy with comprehensive validation.

    Args:
        strategy: Strategy object
        market_data: Market data DataFrame
        context: Additional context

    Returns:
        Strategy execution result with validation

    Raises:
        ValueError: If validation fails
        RuntimeError: If strategy execution fails
    """
    validator = StrategyExecutionValidator()

    # Validate execution context
    is_valid, error = validator.validate_execution_context(strategy, market_data, context)
    if not is_valid:
        raise ValueError(f"Execution context validation failed: {error}")

    # Execute strategy
    try:
        signals = strategy.generate_signals(market_data)
    except Exception as e:
        logger.error(f"Strategy signal generation failed: {e}")
        raise RuntimeError(f"Strategy signal generation failed: {e}")

    # Validate signals
    if isinstance(signals, pd.DataFrame):
        is_valid, error = validator.validate_signals_dataframe(signals, market_data)
        if not is_valid:
            raise ValueError(f"Signals validation failed: {error}")
    elif isinstance(signals, dict):
        # Single signal dict
        is_valid, error = validator.validate_signal(signals)
        if not is_valid:
            raise ValueError(f"Signal validation failed: {error}")
    elif signals is None:
        raise ValueError("Strategy returned None signals")
    else:
        logger.warning(f"Unknown signal type: {type(signals)}")

    # Create result
    result = {
        'signals': signals,
        'success': True,
        'validation_passed': True,
    }

    # Validate result
    is_valid, error = validator.validate_strategy_result(result)
    if not is_valid:
        raise ValueError(f"Result validation failed: {error}")

    return result

