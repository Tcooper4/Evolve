"""
Input validation for EVOLVE trading system

Provides comprehensive input validation for all external inputs
to prevent bad data from entering the system.
"""

import re
from typing import Any, Callable
from datetime import datetime


class InputInputValidationError(Exception):
    """Raised when input validation fails"""
    pass


class InputValidator:
    """Validates all external inputs"""

    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Validate stock symbol.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Validated and normalized symbol

        Raises:
            InputInputValidationError: If symbol is invalid
        """
        if not symbol:
            raise InputInputValidationError("Symbol cannot be empty")

        if not isinstance(symbol, str):
            raise InputInputValidationError(f"Symbol must be string, got {type(symbol)}")

        symbol = symbol.strip().upper()

        if len(symbol) < 1 or len(symbol) > 10:
            raise InputInputValidationError(f"Symbol length invalid: {len(symbol)} (must be 1-10 characters)")

        if not re.match(r'^[A-Z][A-Z0-9.-]*$', symbol):
            raise InputInputValidationError(f"Symbol contains invalid characters: {symbol}")

        return symbol

    @staticmethod
    def validate_quantity(quantity: float, min_qty: float = 0.0, max_qty: float = 1000000.0) -> float:
        """
        Validate order quantity.

        Args:
            quantity: Quantity to validate
            min_qty: Minimum allowed quantity
            max_qty: Maximum allowed quantity

        Returns:
            Validated quantity

        Raises:
            InputInputValidationError: If quantity is invalid
        """
        if not isinstance(quantity, (int, float)):
            raise InputInputValidationError(f"Quantity must be numeric, got {type(quantity)}")

        quantity = float(quantity)

        if quantity < min_qty:
            raise InputInputValidationError(f"Quantity {quantity} below minimum {min_qty}")

        if quantity > max_qty:
            raise InputInputValidationError(f"Quantity {quantity} exceeds maximum {max_qty}")

        if quantity <= 0:
            raise InputInputValidationError(f"Quantity must be positive, got {quantity}")

        return quantity

    @staticmethod
    def validate_price(price: float, min_price: float = 0.01, max_price: float = 1000000.0) -> float:
        """
        Validate price.

        Args:
            price: Price to validate
            min_price: Minimum allowed price
            max_price: Maximum allowed price

        Returns:
            Validated price

        Raises:
            InputValidationError: If price is invalid
        """
        if not isinstance(price, (int, float)):
            raise InputValidationError(f"Price must be numeric, got {type(price)}")

        price = float(price)

        if price < min_price:
            raise InputValidationError(f"Price {price} below minimum {min_price}")

        if price > max_price:
            raise InputValidationError(f"Price {price} exceeds maximum {max_price}")

        return price

    @staticmethod
    def validate_side(side: str) -> str:
        """
        Validate order side.

        Args:
            side: Order side to validate

        Returns:
            Validated and normalized side

        Raises:
            InputValidationError: If side is invalid
        """
        if not isinstance(side, str):
            raise InputValidationError(f"Side must be string, got {type(side)}")

        side = side.strip().lower()

        if side not in ['buy', 'sell', 'hold']:
            raise InputValidationError(f"Invalid side: {side} (must be 'buy', 'sell', or 'hold')")

        return side

    @staticmethod
    def validate_order_type(order_type: str) -> str:
        """
        Validate order type.

        Args:
            order_type: Order type to validate

        Returns:
            Validated and normalized order type

        Raises:
            InputValidationError: If order type is invalid
        """
        if not isinstance(order_type, str):
            raise InputValidationError(f"Order type must be string, got {type(order_type)}")

        order_type = order_type.strip().lower()

        valid_types = ['market', 'limit', 'stop', 'stop_limit', 'twap', 'vwap', 'iceberg']

        if order_type not in valid_types:
            raise InputValidationError(
                f"Invalid order type: {order_type} (must be one of {valid_types})"
            )

        return order_type

    @staticmethod
    def validate_timestamp(timestamp: Any) -> datetime:
        """
        Validate timestamp.

        Args:
            timestamp: Timestamp to validate (datetime, string, or float)

        Returns:
            Validated datetime object

        Raises:
            InputValidationError: If timestamp is invalid
        """
        if isinstance(timestamp, datetime):
            return timestamp

        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                raise InputValidationError(f"Invalid timestamp format: {timestamp}")

        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                raise InputValidationError(f"Invalid timestamp value: {timestamp}")

        raise InputValidationError(f"Timestamp must be datetime, string, or numeric, got {type(timestamp)}")

    @staticmethod
    def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        Validate percentage value.

        Args:
            value: Percentage to validate
            min_val: Minimum allowed percentage
            max_val: Maximum allowed percentage

        Returns:
            Validated percentage

        Raises:
            InputValidationError: If percentage is invalid
        """
        if not isinstance(value, (int, float)):
            raise InputValidationError(f"Percentage must be numeric, got {type(value)}")

        value = float(value)

        if value < min_val:
            raise InputValidationError(f"Percentage {value} below minimum {min_val}")

        if value > max_val:
            raise InputValidationError(f"Percentage {value} exceeds maximum {max_val}")

        return value

    @staticmethod
    def validate_positive_integer(value: Any, min_val: int = 1, max_val: int = 1000000) -> int:
        """
        Validate positive integer.

        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Validated integer

        Raises:
            InputValidationError: If value is invalid
        """
        if not isinstance(value, (int, float)):
            raise InputValidationError(f"Value must be numeric, got {type(value)}")

        try:
            value = int(value)
        except (ValueError, TypeError):
            raise InputValidationError(f"Cannot convert {value} to integer")

        if value < min_val:
            raise InputValidationError(f"Value {value} below minimum {min_val}")

        if value > max_val:
            raise InputValidationError(f"Value {value} exceeds maximum {max_val}")

        return value


# Decorator for automatic validation
def validate_inputs(**validators):
    """
    Decorator to automatically validate function inputs.

    Args:
        **validators: Mapping of parameter names to validator method names
                     (e.g., symbol='symbol', quantity='quantity')

    Example:
        @validate_inputs(symbol='symbol', quantity='quantity', side='side')
        def place_order(symbol, quantity, side):
            # Inputs already validated!
            pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            validator = InputValidator()
            for param_name, validator_name in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    validator_method = getattr(validator, f'validate_{validator_name}')
                    validated_value = validator_method(value)
                    bound_args.arguments[param_name] = validated_value

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper
    return decorator
