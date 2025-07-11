"""
Custom Exceptions for QuantGPT

Defines specific exception types for different error scenarios
with proper error codes and recovery strategies.
Enhanced with agent execution error handling and context capture.
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import traceback
import json

class QuantGPTException(Exception):
    """Base exception for QuantGPT system."""
    
    def __init__(self, message: str, error_code: str = None, 
                 details: Dict[str, Any] = None, recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()

class AgentExecutionError(QuantGPTException):
    """Base exception for agent execution failures with context capture."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None,
                 execution_context: Dict[str, Any] = None, error_code: str = "AGENT_EXECUTION_ERROR",
                 details: Dict[str, Any] = None, recoverable: bool = True):
        super().__init__(message, error_code, details, recoverable)
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.execution_context = execution_context or {}
        self.execution_steps = []
        self.fallback_attempted = False
        self.recovery_suggestions = []
    
    def add_execution_step(self, step_name: str, step_result: str, step_details: Dict[str, Any] = None):
        """Add an execution step to the context."""
        self.execution_steps.append({
            'step_name': step_name,
            'step_result': step_result,
            'step_details': step_details or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def set_fallback_attempted(self, attempted: bool = True):
        """Mark that fallback was attempted."""
        self.fallback_attempted = attempted
    
    def add_recovery_suggestion(self, suggestion: str, priority: int = 1):
        """Add a recovery suggestion."""
        self.recovery_suggestions.append({
            'suggestion': suggestion,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution context."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'error_code': self.error_code,
            'recoverable': self.recoverable,
            'fallback_attempted': self.fallback_attempted,
            'execution_steps_count': len(self.execution_steps),
            'recovery_suggestions_count': len(self.recovery_suggestions),
            'timestamp': self.timestamp.isoformat(),
            'execution_context_keys': list(self.execution_context.keys()) if self.execution_context else []
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat(),
            'execution_context': self.execution_context,
            'execution_steps': self.execution_steps,
            'fallback_attempted': self.fallback_attempted,
            'recovery_suggestions': self.recovery_suggestions,
            'details': self.details,
            'traceback': self.traceback
        }

class AgentInitializationError(AgentExecutionError):
    """Raised when agent initialization fails."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None,
                 initialization_context: Dict[str, Any] = None, details: Dict[str, Any] = None):
        super().__init__(message, agent_id, agent_type, initialization_context, 
                        "AGENT_INITIALIZATION_ERROR", details, recoverable=True)

class AgentCommunicationError(AgentExecutionError):
    """Raised when agent communication fails."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None,
                 communication_context: Dict[str, Any] = None, details: Dict[str, Any] = None):
        super().__init__(message, agent_id, agent_type, communication_context,
                        "AGENT_COMMUNICATION_ERROR", details, recoverable=True)

class AgentOrchestrationError(AgentExecutionError):
    """Raised when agent orchestration fails."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None,
                 orchestration_context: Dict[str, Any] = None, details: Dict[str, Any] = None):
        super().__init__(message, agent_id, agent_type, orchestration_context,
                        "AGENT_ORCHESTRATION_ERROR", details, recoverable=True)

class AgentTimeoutError(AgentExecutionError):
    """Raised when agent execution times out."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None,
                 timeout_context: Dict[str, Any] = None, timeout_duration: float = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, agent_id, agent_type, timeout_context,
                        "AGENT_TIMEOUT_ERROR", details, recoverable=True)
        self.timeout_duration = timeout_duration

class AgentResourceError(AgentExecutionError):
    """Raised when agent resource allocation fails."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None,
                 resource_context: Dict[str, Any] = None, resource_type: str = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, agent_id, agent_type, resource_context,
                        "AGENT_RESOURCE_ERROR", details, recoverable=True)
        self.resource_type = resource_type

class AgentValidationError(AgentExecutionError):
    """Raised when agent input/output validation fails."""
    
    def __init__(self, message: str, agent_id: str = None, agent_type: str = None,
                 validation_context: Dict[str, Any] = None, validation_field: str = None,
                 details: Dict[str, Any] = None):
        super().__init__(message, agent_id, agent_type, validation_context,
                        "AGENT_VALIDATION_ERROR", details, recoverable=True)
        self.validation_field = validation_field

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

# Exception handling utilities
def capture_agent_context(func):
    """Decorator to capture agent execution context in exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AgentExecutionError as e:
            # Re-raise with additional context
            e.add_execution_step(
                step_name=func.__name__,
                step_result="failed",
                step_details={
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'function_module': func.__module__
                }
            )
            raise
        except Exception as e:
            # Convert to AgentExecutionError if possible
            agent_context = {
                'function_name': func.__name__,
                'function_module': func.__module__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            raise AgentExecutionError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                execution_context=agent_context,
                details={'original_exception': str(e)}
            ) from e
    return wrapper

def handle_agent_exception(exception: AgentExecutionError, logger=None) -> Dict[str, Any]:
    """Handle agent exception and return recovery information."""
    if logger:
        logger.error(f"Agent execution error: {exception.message}")
        logger.error(f"Context: {exception.get_context_summary()}")
    
    # Determine recovery strategy
    recovery_strategy = {
        'should_retry': exception.recoverable and not exception.fallback_attempted,
        'should_fallback': exception.recoverable and len(exception.recovery_suggestions) > 0,
        'should_escalate': not exception.recoverable or exception.fallback_attempted,
        'suggestions': [s['suggestion'] for s in sorted(exception.recovery_suggestions, 
                                                       key=lambda x: x['priority'], reverse=True)]
    }
    
    return {
        'error_summary': exception.get_context_summary(),
        'recovery_strategy': recovery_strategy,
        'full_context': exception.to_dict()
    } 