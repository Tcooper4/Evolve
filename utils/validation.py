"""
Input validation for EVOLVE trading system

Provides comprehensive input validation for all external inputs to prevent
bad data from entering the system.
"""

import re
from typing import Any, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails"""
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
            Validated and normalized symbol (uppercase, stripped)
            
        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol:
            raise ValidationError("Symbol cannot be empty")
        
        if not isinstance(symbol, str):
            raise ValidationError(f"Symbol must be string, got {type(symbol)}")
        
        symbol = symbol.strip().upper()
        
        if len(symbol) < 1 or len(symbol) > 10:
            raise ValidationError(f"Symbol length invalid: {len(symbol)} (must be 1-10 characters)")
        
        if not re.match(r'^[A-Z][A-Z0-9.-]*$', symbol):
            raise ValidationError(f"Symbol contains invalid characters: {symbol}")
        
        return symbol
    
    @staticmethod
    def validate_quantity(quantity: float, min_qty: float = 0.0, 
                         max_qty: float = 1000000.0) -> float:
        """
        Validate order quantity.
        
        Args:
            quantity: Quantity to validate
            min_qty: Minimum allowed quantity
            max_qty: Maximum allowed quantity
            
        Returns:
            Validated quantity as float
            
        Raises:
            ValidationError: If quantity is invalid
        """
        if not isinstance(quantity, (int, float)):
            raise ValidationError(f"Quantity must be numeric, got {type(quantity)}")
        
        quantity = float(quantity)
        
        if quantity < min_qty:
            raise ValidationError(f"Quantity {quantity} below minimum {min_qty}")
        
        if quantity > max_qty:
            raise ValidationError(f"Quantity {quantity} exceeds maximum {max_qty}")
        
        if quantity <= 0:
            raise ValidationError(f"Quantity must be positive, got {quantity}")
        
        return quantity
    
    @staticmethod
    def validate_price(price: float, min_price: float = 0.01,
                      max_price: float = 1000000.0) -> float:
        """
        Validate price.
        
        Args:
            price: Price to validate
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            
        Returns:
            Validated price as float
            
        Raises:
            ValidationError: If price is invalid
        """
        if not isinstance(price, (int, float)):
            raise ValidationError(f"Price must be numeric, got {type(price)}")
        
        price = float(price)
        
        if price < min_price:
            raise ValidationError(f"Price {price} below minimum {min_price}")
        
        if price > max_price:
            raise ValidationError(f"Price {price} exceeds maximum {max_price}")
        
        return price
    
    @staticmethod
    def validate_side(side: str) -> str:
        """
        Validate order side.
        
        Args:
            side: Order side to validate
            
        Returns:
            Validated and normalized side (lowercase)
            
        Raises:
            ValidationError: If side is invalid
        """
        if not isinstance(side, str):
            raise ValidationError(f"Side must be string, got {type(side)}")
        
        side = side.strip().lower()
        
        if side not in ['buy', 'sell', 'hold']:
            raise ValidationError(f"Invalid side: {side} (must be 'buy', 'sell', or 'hold')")
        
        return side
    
    @staticmethod
    def validate_order_type(order_type: str) -> str:
        """
        Validate order type.
        
        Args:
            order_type: Order type to validate
            
        Returns:
            Validated and normalized order type (lowercase)
            
        Raises:
            ValidationError: If order type is invalid
        """
        if not isinstance(order_type, str):
            raise ValidationError(f"Order type must be string, got {type(order_type)}")
        
        order_type = order_type.strip().lower()
        
        valid_types = ['market', 'limit', 'stop', 'stop_limit', 'twap', 'vwap', 'iceberg']
        
        if order_type not in valid_types:
            raise ValidationError(
                f"Invalid order type: {order_type} "
                f"(must be one of: {', '.join(valid_types)})"
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
            ValidationError: If timestamp is invalid
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                raise ValidationError(f"Invalid timestamp format: {timestamp}")
        
        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                raise ValidationError(f"Invalid timestamp value: {timestamp}")
        
        raise ValidationError(f"Timestamp must be datetime, string, or numeric, got {type(timestamp)}")
    
    @staticmethod
    def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        Validate percentage value.
        
        Args:
            value: Percentage to validate
            min_val: Minimum allowed percentage
            max_val: Maximum allowed percentage
            
        Returns:
            Validated percentage as float
            
        Raises:
            ValidationError: If percentage is invalid
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"Percentage must be numeric, got {type(value)}")
        
        value = float(value)
        
        if value < min_val:
            raise ValidationError(f"Percentage {value} below minimum {min_val}")
        
        if value > max_val:
            raise ValidationError(f"Percentage {value} exceeds maximum {max_val}")
        
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
            
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate specified parameters
            validator = InputValidator()
            for param_name, validator_name in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validator_method = getattr(validator, f'validate_{validator_name}')
                        validated_value = validator_method(value)
                        bound_args.arguments[param_name] = validated_value
                    except AttributeError:
                        logger.warning(f"Unknown validator: validate_{validator_name}")
                    except ValidationError as e:
                        logger.error(f"Validation failed for {param_name}: {e}")
                        raise
            
            return func(*bound_args.args, **bound_args.kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    return decorator

