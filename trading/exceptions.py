"""
Trading System Exceptions

This module defines comprehensive custom exception types for the trading system,
including agent-specific errors, trading system errors, and data processing errors.
"""

from typing import Any, Dict, Optional


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class AgentError(TradingSystemError):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.agent_name = agent_name


class AgentInitializationError(AgentError):
    """Raised when an agent fails to initialize properly."""



class AgentExecutionError(AgentError):
    """Raised when an agent fails during execution."""



class AgentCommunicationError(AgentError):
    """Raised when agents fail to communicate with each other."""



class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""



class AgentValidationError(AgentError):
    """Raised when agent input validation fails."""



class AgentRegistryError(TradingSystemError):
    """Raised when there are issues with agent registry operations."""



class ModelError(TradingSystemError):
    """Base exception for model-related errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.model_name = model_name


class ModelInitializationError(ModelError):
    """Raised when a model fails to initialize."""



class ModelTrainingError(ModelError):
    """Raised when model training fails."""



class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""



class ModelValidationError(ModelError):
    """Raised when model validation fails."""



class ModelDeploymentError(ModelError):
    """Raised when model deployment fails."""



class DataError(TradingSystemError):
    """Base exception for data-related errors."""

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.data_source = data_source


class DataConnectionError(DataError):
    """Raised when data connection fails."""



class DataValidationError(DataError):
    """Raised when data validation fails."""



class DataProcessingError(DataError):
    """Raised when data processing fails."""



class DataNotFoundError(DataError):
    """Raised when requested data is not found."""



class DataIntegrityError(DataError):
    """Raised when data integrity issues are detected (bad or mismatched input series)."""

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        integrity_issues: Optional[list] = None,
    ):
        super().__init__(message, data_source, error_code, context)
        self.integrity_issues = integrity_issues or []

    def add_integrity_issue(self, issue: str):
        """Add a specific integrity issue to the list."""
        self.integrity_issues.append(issue)

    def get_integrity_summary(self) -> str:
        """Get a summary of all integrity issues."""
        if not self.integrity_issues:
            return "No specific integrity issues identified"

        summary = f"Found {len(self.integrity_issues)} integrity issue(s):\n"
        for i, issue in enumerate(self.integrity_issues, 1):
            summary += f"  {i}. {issue}\n"
        return summary.rstrip()


class StrategyError(TradingSystemError):
    """Base exception for strategy-related errors."""

    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.strategy_name = strategy_name


class StrategyInitializationError(StrategyError):
    """Raised when strategy initialization fails."""



class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""



class StrategyValidationError(StrategyError):
    """Raised when strategy validation fails."""



class ExecutionError(TradingSystemError):
    """Base exception for trade execution errors."""

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.symbol = symbol


class OrderExecutionError(ExecutionError):
    """Raised when order execution fails."""



class PositionError(ExecutionError):
    """Raised when position management fails."""



class RiskError(TradingSystemError):
    """Base exception for risk management errors."""

    def __init__(
        self,
        message: str,
        risk_type: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.risk_type = risk_type


class RiskLimitExceededError(RiskError):
    """Raised when risk limits are exceeded."""



class RiskValidationError(RiskError):
    """Raised when risk validation fails."""



class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid or missing."""



class AuthenticationError(TradingSystemError):
    """Raised when authentication fails."""



class AuthorizationError(TradingSystemError):
    """Raised when authorization fails."""



class NetworkError(TradingSystemError):
    """Raised when network operations fail."""



class TimeoutError(TradingSystemError):
    """Raised when operations timeout."""



class ResourceError(TradingSystemError):
    """Raised when resource allocation fails."""



class MemoryError(TradingSystemError):
    """Raised when memory operations fail."""



class FileError(TradingSystemError):
    """Raised when file operations fail."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.file_path = file_path


class DatabaseError(TradingSystemError):
    """Raised when database operations fail."""



class APIError(TradingSystemError):
    """Raised when external API calls fail."""

    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, context)
        self.api_name = api_name
        self.status_code = status_code


class ValidationError(TradingSystemError):
    """Raised when general validation fails."""



class StateError(TradingSystemError):
    """Raised when system state is invalid."""



class CircularDependencyError(TradingSystemError):
    """Raised when circular dependencies are detected."""



class VersionError(TradingSystemError):
    """Raised when version compatibility issues occur."""



# Error code constants
ERROR_CODES = {
    "AGENT_INIT_FAILED": "AGENT_001",
    "AGENT_EXECUTION_FAILED": "AGENT_002",
    "AGENT_TIMEOUT": "AGENT_003",
    "AGENT_VALIDATION_FAILED": "AGENT_004",
    "MODEL_TRAINING_FAILED": "MODEL_001",
    "MODEL_PREDICTION_FAILED": "MODEL_002",
    "MODEL_DEPLOYMENT_FAILED": "MODEL_003",
    "DATA_CONNECTION_FAILED": "DATA_001",
    "DATA_VALIDATION_FAILED": "DATA_002",
    "STRATEGY_EXECUTION_FAILED": "STRATEGY_001",
    "ORDER_EXECUTION_FAILED": "EXECUTION_001",
    "RISK_LIMIT_EXCEEDED": "RISK_001",
    "CONFIGURATION_INVALID": "CONFIG_001",
    "AUTHENTICATION_FAILED": "AUTH_001",
    "NETWORK_ERROR": "NETWORK_001",
    "TIMEOUT_ERROR": "TIMEOUT_001",
    "API_ERROR": "API_001",
    "VALIDATION_FAILED": "VALIDATION_001",
    "STATE_ERROR": "STATE_001",
}


def get_error_code(error_type: str) -> str:
    """Get error code for a given error type."""
    return ERROR_CODES.get(error_type, "UNKNOWN_ERROR")


def create_error_context(**kwargs) -> Dict[str, Any]:
    """Create error context dictionary."""
    return {
        "timestamp": kwargs.get("timestamp"),
        "user_id": kwargs.get("user_id"),
        "session_id": kwargs.get("session_id"),
        "request_id": kwargs.get("request_id"),
        "additional_info": kwargs.get("additional_info", {}),
    }
