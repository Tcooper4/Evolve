"""
Custom Exceptions for QuantGPT

Defines specific exception types for different error scenarios
with proper error codes and recovery strategies.
"""

from typing import Optional, Dict, Any

class QuantGPTException(Exception):
    """Base exception for QuantGPT system."""
    
    def __init__(self, message: str, error_code: str = None, 
                 details: Dict[str, Any] = None, recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.recoverable = recoverable

class QueryParsingError(QuantGPTException):
    """Raised when query parsing fails."""
    
    def __init__(self, message: str, query: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "QUERY_PARSING_ERROR", details, recoverable=True)
        self.query = query

class ActionExecutionError(QuantGPTException):
    """Raised when action execution fails."""
    
    def __init__(self, message: str, action: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "ACTION_EXECUTION_ERROR", details, recoverable=True)
        self.action = action

class CommentaryGenerationError(QuantGPTException):
    """Raised when commentary generation fails."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "COMMENTARY_GENERATION_ERROR", details, recoverable=True)

class ServiceConnectionError(QuantGPTException):
    """Raised when service connection fails."""
    
    def __init__(self, message: str, service: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "SERVICE_CONNECTION_ERROR", details, recoverable=True)
        self.service = service

class RateLimitExceededError(QuantGPTException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = None, details: Dict[str, Any] = None):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details, recoverable=True)
        self.retry_after = retry_after

class ValidationError(QuantGPTException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: str = None, value: Any = None, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", details, recoverable=True)
        self.field = field
        self.value = value

class ConfigurationError(QuantGPTException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details, recoverable=False)
        self.config_key = config_key

class ModelBuildError(QuantGPTException):
    """Raised when model building fails."""
    
    def __init__(self, message: str, model_type: str = None, symbol: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "MODEL_BUILD_ERROR", details, recoverable=True)
        self.model_type = model_type
        self.symbol = symbol

class ModelEvaluationError(QuantGPTException):
    """Raised when model evaluation fails."""
    
    def __init__(self, message: str, model_id: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "MODEL_EVALUATION_ERROR", details, recoverable=True)
        self.model_id = model_id

class DataProviderError(QuantGPTException):
    """Raised when data provider fails."""
    
    def __init__(self, message: str, provider: str = None, symbol: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "DATA_PROVIDER_ERROR", details, recoverable=True)
        self.provider = provider
        self.symbol = symbol 