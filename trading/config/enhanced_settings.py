"""
Enhanced configuration settings for the trading system.

This module provides specialized configuration classes for different trading components
including trading strategies, agents, risk management, and performance monitoring.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ValidationSchema:
    """Schema for validating environment variables."""

    name: str
    required: bool = True
    default: Any = None
    type: Type = str
    validator: Optional[Callable] = None
    pattern: Optional[str] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    description: str = ""


class ConfigurationValidator:
    """Validator for configuration settings."""

    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []

    def validate_env_var(self, schema: ValidationSchema, value: Any) -> bool:
        """
        Validate a single environment variable against its schema.

        Args:
            schema: Validation schema
            value: Value to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if required
            if schema.required and (value is None or value == ""):
                self.validation_errors.append(
                    f"Required environment variable '{schema.name}' is missing"
                )
                return False

            # Apply default if not required and missing
            if not schema.required and (value is None or value == ""):
                value = schema.default

            # Type conversion
            if value is not None and value != "":
                try:
                    if schema.type == bool:
                        if isinstance(value, str):
                            value = value.lower() in ("true", "1", "yes", "on")
                    elif schema.type == int:
                        value = int(value)
                    elif schema.type == float:
                        value = float(value)
                    elif schema.type == list:
                        if isinstance(value, str):
                            value = [
                                item.strip()
                                for item in value.split(",")
                                if item.strip()
                            ]
                except (ValueError, TypeError) as e:
                    self.validation_errors.append(
                        f"Invalid type for '{schema.name}': expected {schema.type.__name__}, got {type(value).__name__}"
                    )
                    return False

            # Pattern validation
            if schema.pattern and value:
                if not re.match(schema.pattern, str(value)):
                    self.validation_errors.append(
                        f"Value for '{schema.name}' does not match pattern: {schema.pattern}"
                    )
                    return False

            # Range validation
            if schema.min_value is not None and value is not None:
                if value < schema.min_value:
                    self.validation_errors.append(
                        f"Value for '{schema.name}' ({value}) is below minimum ({schema.min_value})"
                    )
                    return False

            if schema.max_value is not None and value is not None:
                if value > schema.max_value:
                    self.validation_errors.append(
                        f"Value for '{schema.name}' ({value}) is above maximum ({schema.max_value})"
                    )
                    return False

            # Allowed values validation
            if schema.allowed_values and value not in schema.allowed_values:
                self.validation_errors.append(
                    f"Value for '{schema.name}' ({value}) is not in allowed values: {schema.allowed_values}"
                )
                return False

            # Custom validator
            if schema.validator and value is not None:
                if not schema.validator(value):
                    self.validation_errors.append(
                        f"Custom validation failed for '{schema.name}'"
                    )
                    return False

            return True

        except Exception as e:
            self.validation_errors.append(
                f"Validation error for '{schema.name}': {str(e)}"
            )
            return False

    def validate_schemas(self, schemas: List[ValidationSchema]) -> Dict[str, Any]:
        """
        Validate multiple environment variables against their schemas.

        Args:
            schemas: List of validation schemas

        Returns:
            Dictionary of validated values
        """
        validated_values = {}

        for schema in schemas:
            value = os.getenv(schema.name, schema.default)
            if self.validate_env_var(schema, value):
                validated_values[schema.name] = value
            else:
                # Use default for non-required fields
                if not schema.required:
                    validated_values[schema.name] = schema.default
                    self.validation_warnings.append(
                        f"Using default value for '{schema.name}': {schema.default}"
                    )

        return validated_values

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.validation_errors

    def get_warnings(self) -> List[str]:
        """Get validation warnings."""
        return self.validation_warnings

    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.validation_errors) > 0


# Define validation schemas for environment variables
ENVIRONMENT_SCHEMAS = [
    ValidationSchema(
        name="TRADING_ENV",
        required=False,
        default="development",
        allowed_values=["development", "staging", "production"],
        description="Trading environment",
    ),
    ValidationSchema(
        name="LOG_LEVEL",
        required=False,
        default="INFO",
        allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        description="Logging level",
    ),
    ValidationSchema(
        name="LOG_DIR", required=False, default="logs", description="Log directory path"
    ),
    ValidationSchema(
        name="ALPHA_VANTAGE_API_KEY",
        required=False,
        pattern=r"^[A-Z0-9]{16}$",
        description="Alpha Vantage API key",
    ),
    ValidationSchema(
        name="POLYGON_API_KEY",
        required=False,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Polygon API key",
    ),
    ValidationSchema(
        name="OPENAI_API_KEY",
        required=False,
        pattern=r"^sk-[A-Za-z0-9]{32,}$",
        description="OpenAI API key",
    ),
    ValidationSchema(
        name="JWT_SECRET_KEY",
        required=False,
        description="JWT secret key (minimum 32 characters)",
    ),
    ValidationSchema(
        name="WEB_SECRET_KEY",
        required=False,
        description="Web secret key (minimum 32 characters)",
    ),
    ValidationSchema(
        name="DEFAULT_STRATEGY",
        required=False,
        default="momentum",
        allowed_values=["momentum", "mean_reversion", "arbitrage", "fundamental"],
        description="Default trading strategy",
    ),
    ValidationSchema(
        name="STRATEGY_DIR",
        required=False,
        default="strategies",
        description="Strategy directory path",
    ),
    ValidationSchema(
        name="BACKTEST_DAYS",
        required=False,
        default=365,
        type=int,
        min_value=30,
        max_value=3650,
        description="Number of days for backtesting",
    ),
    ValidationSchema(
        name="SLIPPAGE",
        required=False,
        default=0.001,
        type=float,
        min_value=0.0,
        max_value=0.1,
        description="Slippage percentage",
    ),
    ValidationSchema(
        name="TRANSACTION_COST",
        required=False,
        default=0.001,
        type=float,
        min_value=0.0,
        max_value=0.1,
        description="Transaction cost percentage",
    ),
    ValidationSchema(
        name="MAX_POSITION_SIZE",
        required=False,
        default=0.25,
        type=float,
        min_value=0.01,
        max_value=1.0,
        description="Maximum position size",
    ),
    ValidationSchema(
        name="MIN_POSITION_SIZE",
        required=False,
        default=0.01,
        type=float,
        min_value=0.001,
        max_value=0.5,
        description="Minimum position size",
    ),
    ValidationSchema(
        name="MAX_LEVERAGE",
        required=False,
        default=1.0,
        type=float,
        min_value=0.1,
        max_value=10.0,
        description="Maximum leverage",
    ),
    ValidationSchema(
        name="RISK_PER_TRADE",
        required=False,
        default=0.02,
        type=float,
        min_value=0.001,
        max_value=0.1,
        description="Risk per trade percentage",
    ),
    ValidationSchema(
        name="STOP_LOSS",
        required=False,
        default=0.05,
        type=float,
        min_value=0.01,
        max_value=0.5,
        description="Stop loss percentage",
    ),
    ValidationSchema(
        name="TAKE_PROFIT",
        required=False,
        default=0.10,
        type=float,
        min_value=0.01,
        max_value=1.0,
        description="Take profit percentage",
    ),
    ValidationSchema(
        name="DEFAULT_TICKERS",
        required=False,
        default="AAPL,MSFT,GOOGL",
        description="Default ticker symbols",
    ),
    ValidationSchema(
        name="DATA_PROVIDER",
        required=False,
        default="yahoo",
        allowed_values=["yahoo", "alpha_vantage", "polygon", "quandl"],
        description="Data provider",
    ),
    ValidationSchema(
        name="AGENT_TIMEOUT",
        required=False,
        default=300,
        type=int,
        min_value=30,
        max_value=3600,
        description="Agent timeout in seconds",
    ),
    ValidationSchema(
        name="MAX_CONCURRENT_AGENTS",
        required=False,
        default=5,
        type=int,
        min_value=1,
        max_value=50,
        description="Maximum concurrent agents",
    ),
    ValidationSchema(
        name="AGENT_MEMORY_SIZE",
        required=False,
        default=1000,
        type=int,
        min_value=100,
        max_value=10000,
        description="Agent memory size",
    ),
    ValidationSchema(
        name="DEFAULT_LLM_PROVIDER",
        required=False,
        default="openai",
        allowed_values=["openai", "huggingface", "anthropic"],
        description="Default LLM provider",
    ),
    ValidationSchema(
        name="HUGGINGFACE_API_KEY", required=False, description="Hugging Face API key"
    ),
    ValidationSchema(
        name="HUGGINGFACE_MODEL",
        required=False,
        default="gpt2",
        description="Hugging Face model name",
    ),
    ValidationSchema(
        name="MEMORY_DIR",
        required=False,
        default="memory",
        description="Memory directory path",
    ),
    ValidationSchema(
        name="MEMORY_BACKEND",
        required=False,
        default="json",
        allowed_values=["json", "sqlite", "redis"],
        description="Memory backend",
    ),
    ValidationSchema(
        name="PERFORMANCE_THRESHOLD",
        required=False,
        default=0.05,
        type=float,
        min_value=0.001,
        max_value=1.0,
        description="Performance threshold",
    ),
    ValidationSchema(
        name="IMPROVEMENT_THRESHOLD",
        required=False,
        default=0.02,
        type=float,
        min_value=0.001,
        max_value=1.0,
        description="Improvement threshold",
    ),
    ValidationSchema(
        name="MAX_DRAWDOWN",
        required=False,
        default=0.20,
        type=float,
        min_value=0.01,
        max_value=1.0,
        description="Maximum drawdown",
    ),
    ValidationSchema(
        name="MAX_CORRELATION",
        required=False,
        default=0.70,
        type=float,
        min_value=0.0,
        max_value=1.0,
        description="Maximum correlation",
    ),
    ValidationSchema(
        name="MAX_CONCENTRATION",
        required=False,
        default=0.30,
        type=float,
        min_value=0.01,
        max_value=1.0,
        description="Maximum concentration",
    ),
    ValidationSchema(
        name="VOLATILITY_WINDOW",
        required=False,
        default=20,
        type=int,
        min_value=5,
        max_value=252,
        description="Volatility calculation window",
    ),
    ValidationSchema(
        name="VOLATILITY_THRESHOLD",
        required=False,
        default=0.30,
        type=float,
        min_value=0.01,
        max_value=2.0,
        description="Volatility threshold",
    ),
    ValidationSchema(
        name="VAR_CONFIDENCE",
        required=False,
        default=0.95,
        type=float,
        min_value=0.90,
        max_value=0.99,
        description="Value at Risk confidence level",
    ),
    ValidationSchema(
        name="VAR_WINDOW",
        required=False,
        default=252,
        type=int,
        min_value=30,
        max_value=1000,
        description="Value at Risk calculation window",
    ),
    ValidationSchema(
        name="STRESS_TEST_ENABLED",
        required=False,
        default=True,
        type=bool,
        description="Enable stress testing",
    ),
    ValidationSchema(
        name="STRESS_SCENARIOS",
        required=False,
        default="market_crash,recession,volatility_spike",
        description="Stress test scenarios",
    ),
    ValidationSchema(
        name="METRICS_ENABLED",
        required=False,
        default=True,
        type=bool,
        description="Enable metrics collection",
    ),
    ValidationSchema(
        name="METRICS_PATH",
        required=False,
        default="logs/metrics.log",
        description="Metrics log file path",
    ),
    ValidationSchema(
        name="EVALUATION_WINDOW",
        required=False,
        default=252,
        type=int,
        min_value=30,
        max_value=1000,
        description="Evaluation window in days",
    ),
    ValidationSchema(
        name="BENCHMARK_SYMBOL",
        required=False,
        default="SPY",
        description="Benchmark symbol",
    ),
    ValidationSchema(
        name="RISK_FREE_RATE",
        required=False,
        default=0.02,
        type=float,
        min_value=0.0,
        max_value=0.1,
        description="Risk-free rate",
    ),
    ValidationSchema(
        name="REPORT_FREQUENCY",
        required=False,
        default="daily",
        allowed_values=["daily", "weekly", "monthly"],
        description="Report frequency",
    ),
    ValidationSchema(
        name="REPORT_DIR",
        required=False,
        default="reports",
        description="Report directory path",
    ),
    ValidationSchema(
        name="ALERT_ENABLED",
        required=False,
        default=True,
        type=bool,
        description="Enable alerts",
    ),
    ValidationSchema(
        name="ALERT_EMAIL",
        required=False,
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        description="Alert email address",
    ),
    ValidationSchema(
        name="ALERT_WEBHOOK",
        required=False,
        pattern=r"^https?://.+",
        description="Alert webhook URL",
    ),
    ValidationSchema(
        name="SHARPE_THRESHOLD",
        required=False,
        default=1.0,
        type=float,
        min_value=0.0,
        max_value=5.0,
        description="Sharpe ratio threshold",
    ),
    ValidationSchema(
        name="SORTINO_THRESHOLD",
        required=False,
        default=1.0,
        type=float,
        min_value=0.0,
        max_value=5.0,
        description="Sortino ratio threshold",
    ),
    ValidationSchema(
        name="MAX_DRAWDOWN_THRESHOLD",
        required=False,
        default=0.15,
        type=float,
        min_value=0.01,
        max_value=1.0,
        description="Maximum drawdown threshold",
    ),
]


@dataclass
class EnhancedSettings:
    """Enhanced settings container for the trading system."""

    def __post_init__(self):
        """Validate settings after initialization."""
        self._values = {}
        self._validate_required_env_vars()

    def _validate_required_env_vars(self):
        """Validate required environment variables."""
        validator = ConfigurationValidator()
        validated_values = validator.validate_schemas(ENVIRONMENT_SCHEMAS)

        # Log validation results
        if validator.has_errors():
            logger.error("Configuration validation errors:")
            for error in validator.get_errors():
                logger.error(f"  - {error}")
            raise ValueError("Configuration validation failed")

        if validator.get_warnings():
            logger.warning("Configuration validation warnings:")
            for warning in validator.get_warnings():
                logger.warning(f"  - {warning}")

        # Store validated values
        for name, value in validated_values.items():
            self._values[name.lower()] = value

        logger.info("Configuration validation completed successfully")

    # Environment settings
    @property
    def env(self) -> str:
        return self._values.get("trading_env", "development")

    @property
    def debug(self) -> bool:
        return self.env == "development"

    # Logging settings
    @property
    def log_dir(self) -> Path:
        return Path(self._values.get("log_dir", "logs"))

    @property
    def log_level(self) -> str:
        return self._values.get("log_level", "INFO")

    # API Keys
    @property
    def alpha_vantage_api_key(self) -> str:
        return self._values.get("alpha_vantage_api_key", "")

    @property
    def polygon_api_key(self) -> str:
        return self._values.get("polygon_api_key", "")

    @property
    def openai_api_key(self) -> str:
        return self._values.get("openai_api_key", "")

    # Security
    @property
    def jwt_secret_key(self) -> str:
        return self._values.get("jwt_secret_key", "")

    @property
    def web_secret_key(self) -> str:
        return self._values.get("web_secret_key", "")

    def validate(self) -> bool:
        """Validate enhanced settings."""
        if self.env == "production":
            if not self.alpha_vantage_api_key:
                raise ValueError("ALPHA_VANTAGE_API_KEY is required in production")
            if not self.polygon_api_key:
                raise ValueError("POLYGON_API_KEY is required in production")
            if not self.jwt_secret_key:
                raise ValueError("JWT_SECRET_KEY is required in production")
            if len(self.jwt_secret_key) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")
            if not self.web_secret_key:
                raise ValueError("WEB_SECRET_KEY is required in production")
            if len(self.web_secret_key) < 32:
                raise ValueError("WEB_SECRET_KEY must be at least 32 characters in production")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "env": self.env,
            "debug": self.debug,
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "alpha_vantage_api_key": self.alpha_vantage_api_key,
            "polygon_api_key": self.polygon_api_key,
            "openai_api_key": self.openai_api_key,
            "jwt_secret_key": self.jwt_secret_key,
            "web_secret_key": self.web_secret_key,
        }


@dataclass
class TradingConfig:
    """Configuration for trading strategies and execution."""

    # Strategy settings
    default_strategy: str = field(
        default_factory=lambda: os.getenv("DEFAULT_STRATEGY", "momentum")
    )
    strategy_dir: Path = field(
        default_factory=lambda: Path(os.getenv("STRATEGY_DIR", "strategies"))
    )
    backtest_days: int = field(
        default_factory=lambda: int(os.getenv("BACKTEST_DAYS", "365"))
    )

    # Execution settings
    slippage: float = field(
        default_factory=lambda: float(os.getenv("SLIPPAGE", "0.001"))
    )
    transaction_cost: float = field(
        default_factory=lambda: float(os.getenv("TRANSACTION_COST", "0.001"))
    )
    max_position_size: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "0.25"))
    )
    min_position_size: float = field(
        default_factory=lambda: float(os.getenv("MIN_POSITION_SIZE", "0.01"))
    )

    # Risk settings
    max_leverage: float = field(
        default_factory=lambda: float(os.getenv("MAX_LEVERAGE", "1.0"))
    )
    risk_per_trade: float = field(
        default_factory=lambda: float(os.getenv("RISK_PER_TRADE", "0.02"))
    )
    stop_loss: float = field(
        default_factory=lambda: float(os.getenv("STOP_LOSS", "0.05"))
    )
    take_profit: float = field(
        default_factory=lambda: float(os.getenv("TAKE_PROFIT", "0.10"))
    )

    # Data settings
    default_tickers: List[str] = field(
        default_factory=lambda: os.getenv("DEFAULT_TICKERS", "AAPL,MSFT,GOOGL").split(
            ","
        )
    )
    data_provider: str = field(
        default_factory=lambda: os.getenv("DATA_PROVIDER", "yahoo")
    )

    def validate(self) -> bool:
        """Validate trading configuration."""
        if self.slippage < 0 or self.slippage > 1:
            raise ValueError(f"Invalid slippage: {self.slippage}")
        if self.transaction_cost < 0 or self.transaction_cost > 1:
            raise ValueError(f"Invalid transaction_cost: {self.transaction_cost}")
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError(f"Invalid max_position_size: {self.max_position_size}")
        if (
            self.min_position_size < 0
            or self.min_position_size > self.max_position_size
        ):
            raise ValueError(f"Invalid min_position_size: {self.min_position_size}")
        if self.max_leverage <= 0:
            raise ValueError(f"Invalid max_leverage: {self.max_leverage}")
        if self.risk_per_trade <= 0 or self.risk_per_trade > 1:
            raise ValueError(f"Invalid risk_per_trade: {self.risk_per_trade}")
        if self.stop_loss <= 0 or self.stop_loss > 1:
            raise ValueError(f"Invalid stop_loss: {self.stop_loss}")
        if self.take_profit <= 0 or self.take_profit > 1:
            raise ValueError(f"Invalid take_profit: {self.take_profit}")
        if not self.default_tickers:
            raise ValueError("Default tickers cannot be empty")

        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_strategy": self.default_strategy,
            "strategy_dir": str(self.strategy_dir),
            "backtest_days": self.backtest_days,
            "slippage": self.slippage,
            "transaction_cost": self.transaction_cost,
            "max_position_size": self.max_position_size,
            "min_position_size": self.min_position_size,
            "max_leverage": self.max_leverage,
            "risk_per_trade": self.risk_per_trade,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "default_tickers": self.default_tickers,
            "data_provider": self.data_provider,
        }


@dataclass
class AgentConfig:
    """Configuration for trading agents."""

    # Agent settings
    timeout: int = field(default_factory=lambda: int(os.getenv("AGENT_TIMEOUT", "300")))
    max_concurrent_agents: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
    )
    memory_size: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MEMORY_SIZE", "1000"))
    )

    # LLM settings
    default_llm_provider: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    )
    huggingface_api_key: str = field(
        default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY", "")
    )
    huggingface_model: str = field(
        default_factory=lambda: os.getenv("HUGGINGFACE_MODEL", "gpt2")
    )

    # Memory settings
    memory_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MEMORY_DIR", "memory"))
    )
    memory_backend: str = field(
        default_factory=lambda: os.getenv("MEMORY_BACKEND", "json")
    )

    # Performance settings
    performance_threshold: float = field(
        default_factory=lambda: float(os.getenv("PERFORMANCE_THRESHOLD", "0.05"))
    )
    improvement_threshold: float = field(
        default_factory=lambda: float(os.getenv("IMPROVEMENT_THRESHOLD", "0.02"))
    )

    def validate(self) -> bool:
        """Validate agent configuration."""
        if self.timeout <= 0:
            raise ValueError(f"Invalid timeout: {self.timeout}")
        if self.max_concurrent_agents <= 0:
            raise ValueError(
                f"Invalid max_concurrent_agents: {self.max_concurrent_agents}"
            )
        if self.memory_size <= 0:
            raise ValueError(f"Invalid memory_size: {self.memory_size}")
        if self.performance_threshold <= 0 or self.performance_threshold > 1:
            raise ValueError(
                f"Invalid performance_threshold: {self.performance_threshold}"
            )
        if self.improvement_threshold <= 0 or self.improvement_threshold > 1:
            raise ValueError(
                f"Invalid improvement_threshold: {self.improvement_threshold}"
            )

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeout": self.timeout,
            "max_concurrent_agents": self.max_concurrent_agents,
            "memory_size": self.memory_size,
            "default_llm_provider": self.default_llm_provider,
            "huggingface_api_key": self.huggingface_api_key,
            "huggingface_model": self.huggingface_model,
            "memory_dir": str(self.memory_dir),
            "memory_backend": self.memory_backend,
            "performance_threshold": self.performance_threshold,
            "improvement_threshold": self.improvement_threshold,
        }


@dataclass
class RiskConfig:
    """Configuration for risk management."""

    # Risk limits
    max_drawdown: float = field(
        default_factory=lambda: float(os.getenv("MAX_DRAWDOWN", "0.20"))
    )
    max_correlation: float = field(
        default_factory=lambda: float(os.getenv("MAX_CORRELATION", "0.70"))
    )
    max_concentration: float = field(
        default_factory=lambda: float(os.getenv("MAX_CONCENTRATION", "0.30"))
    )

    # Volatility settings
    volatility_window: int = field(
        default_factory=lambda: int(os.getenv("VOLATILITY_WINDOW", "20"))
    )
    volatility_threshold: float = field(
        default_factory=lambda: float(os.getenv("VOLATILITY_THRESHOLD", "0.30"))
    )

    # VaR settings
    var_confidence: float = field(
        default_factory=lambda: float(os.getenv("VAR_CONFIDENCE", "0.95"))
    )
    var_window: int = field(default_factory=lambda: int(os.getenv("VAR_WINDOW", "252")))

    # Stress testing
    stress_test_enabled: bool = field(
        default_factory=lambda: os.getenv("STRESS_TEST_ENABLED", "true").lower()
        == "true"
    )
    stress_scenarios: List[str] = field(
        default_factory=lambda: os.getenv(
            "STRESS_SCENARIOS", "market_crash,recession,volatility_spike"
        ).split(",")
    )

    def validate(self) -> bool:
        """Validate risk configuration."""
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            raise ValueError(f"Invalid max_drawdown: {self.max_drawdown}")
        if self.max_correlation <= 0 or self.max_correlation > 1:
            raise ValueError(f"Invalid max_correlation: {self.max_correlation}")
        if self.max_concentration <= 0 or self.max_concentration > 1:
            raise ValueError(f"Invalid max_concentration: {self.max_concentration}")
        if self.volatility_window <= 0:
            raise ValueError(f"Invalid volatility_window: {self.volatility_window}")
        if self.volatility_threshold <= 0:
            raise ValueError(
                f"Invalid volatility_threshold: {self.volatility_threshold}"
            )
        if self.var_confidence <= 0 or self.var_confidence >= 1:
            raise ValueError(f"Invalid var_confidence: {self.var_confidence}")
        if self.var_window <= 0:
            raise ValueError(f"Invalid var_window: {self.var_window}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_drawdown": self.max_drawdown,
            "max_correlation": self.max_correlation,
            "max_concentration": self.max_concentration,
            "volatility_window": self.volatility_window,
            "volatility_threshold": self.volatility_threshold,
            "var_confidence": self.var_confidence,
            "var_window": self.var_window,
            "stress_test_enabled": self.stress_test_enabled,
            "stress_scenarios": self.stress_scenarios,
        }


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and evaluation."""

    # Metrics settings
    metrics_enabled: bool = field(
        default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true"
    )
    metrics_path: Path = field(
        default_factory=lambda: Path(os.getenv("METRICS_PATH", "logs/metrics.log"))
    )

    # Evaluation settings
    evaluation_window: int = field(
        default_factory=lambda: int(os.getenv("EVALUATION_WINDOW", "252"))
    )
    benchmark_symbol: str = field(
        default_factory=lambda: os.getenv("BENCHMARK_SYMBOL", "SPY")
    )
    risk_free_rate: float = field(
        default_factory=lambda: float(os.getenv("RISK_FREE_RATE", "0.02"))
    )

    # Reporting settings
    report_frequency: str = field(
        default_factory=lambda: os.getenv("REPORT_FREQUENCY", "daily")
    )
    report_dir: Path = field(
        default_factory=lambda: Path(os.getenv("REPORT_DIR", "reports"))
    )

    # Alerting settings
    alert_enabled: bool = field(
        default_factory=lambda: os.getenv("ALERT_ENABLED", "true").lower() == "true"
    )
    alert_email: str = field(default_factory=lambda: os.getenv("ALERT_EMAIL", ""))
    alert_webhook: str = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK", ""))

    # Thresholds
    sharpe_threshold: float = field(
        default_factory=lambda: float(os.getenv("SHARPE_THRESHOLD", "1.0"))
    )
    sortino_threshold: float = field(
        default_factory=lambda: float(os.getenv("SORTINO_THRESHOLD", "1.0"))
    )
    max_drawdown_threshold: float = field(
        default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_THRESHOLD", "0.15"))
    )

    def validate(self) -> bool:
        """Validate performance configuration."""
        if self.evaluation_window <= 0:
            raise ValueError(f"Invalid evaluation_window: {self.evaluation_window}")
        if self.risk_free_rate < 0 or self.risk_free_rate > 1:
            raise ValueError(f"Invalid risk_free_rate: {self.risk_free_rate}")
        if self.report_frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError(f"Invalid report_frequency: {self.report_frequency}")
        if self.sharpe_threshold < 0:
            raise ValueError(f"Invalid sharpe_threshold: {self.sharpe_threshold}")
        if self.sortino_threshold < 0:
            raise ValueError(f"Invalid sortino_threshold: {self.sortino_threshold}")
        if self.max_drawdown_threshold <= 0 or self.max_drawdown_threshold > 1:
            raise ValueError(
                f"Invalid max_drawdown_threshold: {self.max_drawdown_threshold}"
            )

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics_enabled": self.metrics_enabled,
            "metrics_path": str(self.metrics_path),
            "evaluation_window": self.evaluation_window,
            "benchmark_symbol": self.benchmark_symbol,
            "risk_free_rate": self.risk_free_rate,
            "report_frequency": self.report_frequency,
            "report_dir": str(self.report_dir),
            "alert_enabled": self.alert_enabled,
            "alert_email": self.alert_email,
            "alert_webhook": self.alert_webhook,
            "sharpe_threshold": self.sharpe_threshold,
            "sortino_threshold": self.sortino_threshold,
            "max_drawdown_threshold": self.max_drawdown_threshold,
        }


def create_enhanced_settings() -> EnhancedSettings:
    """Create enhanced settings instance."""
    settings = EnhancedSettings()
    settings.validate()
    return settings


def create_trading_config() -> TradingConfig:
    """Create trading configuration instance."""
    config = TradingConfig()
    config.validate()
    return config


def create_agent_config() -> AgentConfig:
    """Create agent configuration instance."""
    config = AgentConfig()
    config.validate()
    return config


def create_risk_config() -> RiskConfig:
    """Create risk configuration instance."""
    config = RiskConfig()
    config.validate()
    return config


def create_performance_config() -> PerformanceConfig:
    """Create performance configuration instance."""
    config = PerformanceConfig()
    config.validate()
    return config


# Create default instances
enhanced_settings = create_enhanced_settings()
trading_config = create_trading_config()
agent_config = create_agent_config()
risk_config = create_risk_config()
performance_config = create_performance_config()
