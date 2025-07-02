"""
Enhanced configuration settings for the trading system.

This module provides specialized configuration classes for different trading components
including trading strategies, agents, risk management, and performance monitoring.
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict, Union, List
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class EnhancedSettings:
    """Enhanced settings container for the trading system."""
    
    # Environment settings
    env: str = field(default_factory=lambda: os.getenv('TRADING_ENV', 'development'))
    debug: bool = field(default_factory=lambda: os.getenv('TRADING_ENV', 'development') == 'development')
    
    # Logging settings
    log_dir: Path = field(default_factory=lambda: Path(os.getenv('LOG_DIR', 'logs')))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    # API Keys
    alpha_vantage_api_key: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_API_KEY', ''))
    polygon_api_key: str = field(default_factory=lambda: os.getenv('POLYGON_API_KEY', ''))
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    
    # Security
    jwt_secret_key: str = field(default_factory=lambda: os.getenv('JWT_SECRET_KEY', ''))
    web_secret_key: str = field(default_factory=lambda: os.getenv('WEB_SECRET_KEY', ''))
    
    def validate(self) -> bool:
        """Validate enhanced settings."""
        if self.env == 'production':
            if not self.alpha_vantage_api_key:
                raise ValueError("ALPHA_VANTAGE_API_KEY is required in production")
            if not self.polygon_api_key:
                raise ValueError("POLYGON_API_KEY is required in production")
            if not self.jwt_secret_key:
                raise ValueError("JWT_SECRET_KEY is required in production")
            if not self.web_secret_key:
                raise ValueError("WEB_SECRET_KEY is required in production")
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'env': self.env,
            'debug': self.debug,
            'log_dir': str(self.log_dir),
            'log_level': self.log_level,
            'alpha_vantage_api_key': self.alpha_vantage_api_key,
            'polygon_api_key': self.polygon_api_key,
            'openai_api_key': self.openai_api_key,
            'jwt_secret_key': self.jwt_secret_key,
            'web_secret_key': self.web_secret_key
        }

@dataclass
class TradingConfig:
    """Configuration for trading strategies and execution."""
    
    # Strategy settings
    default_strategy: str = field(default_factory=lambda: os.getenv('DEFAULT_STRATEGY', 'momentum'))
    strategy_dir: Path = field(default_factory=lambda: Path(os.getenv('STRATEGY_DIR', 'strategies')))
    backtest_days: int = field(default_factory=lambda: int(os.getenv('BACKTEST_DAYS', '365')))
    
    # Execution settings
    slippage: float = field(default_factory=lambda: float(os.getenv('SLIPPAGE', '0.001')))
    transaction_cost: float = field(default_factory=lambda: float(os.getenv('TRANSACTION_COST', '0.001')))
    max_position_size: float = field(default_factory=lambda: float(os.getenv('MAX_POSITION_SIZE', '0.25')))
    min_position_size: float = field(default_factory=lambda: float(os.getenv('MIN_POSITION_SIZE', '0.01')))
    
    # Risk settings
    max_leverage: float = field(default_factory=lambda: float(os.getenv('MAX_LEVERAGE', '1.0')))
    risk_per_trade: float = field(default_factory=lambda: float(os.getenv('RISK_PER_TRADE', '0.02')))
    stop_loss: float = field(default_factory=lambda: float(os.getenv('STOP_LOSS', '0.05')))
    take_profit: float = field(default_factory=lambda: float(os.getenv('TAKE_PROFIT', '0.10')))
    
    # Data settings
    default_tickers: List[str] = field(default_factory=lambda: os.getenv('DEFAULT_TICKERS', 'AAPL,MSFT,GOOGL').split(','))
    data_provider: str = field(default_factory=lambda: os.getenv('DATA_PROVIDER', 'yahoo'))
    
    def validate(self) -> bool:
        """Validate trading configuration."""
        if self.slippage < 0 or self.slippage > 1:
            raise ValueError(f"Invalid slippage: {self.slippage}")
        if self.transaction_cost < 0 or self.transaction_cost > 1:
            raise ValueError(f"Invalid transaction_cost: {self.transaction_cost}")
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError(f"Invalid max_position_size: {self.max_position_size}")
        if self.min_position_size < 0 or self.min_position_size > self.max_position_size:
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
            'default_strategy': self.default_strategy,
            'strategy_dir': str(self.strategy_dir),
            'backtest_days': self.backtest_days,
            'slippage': self.slippage,
            'transaction_cost': self.transaction_cost,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'max_leverage': self.max_leverage,
            'risk_per_trade': self.risk_per_trade,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'default_tickers': self.default_tickers,
            'data_provider': self.data_provider
        }

@dataclass
class AgentConfig:
    """Configuration for trading agents."""
    
    # Agent settings
    timeout: int = field(default_factory=lambda: int(os.getenv('AGENT_TIMEOUT', '300')))
    max_concurrent_agents: int = field(default_factory=lambda: int(os.getenv('MAX_CONCURRENT_AGENTS', '5')))
    memory_size: int = field(default_factory=lambda: int(os.getenv('AGENT_MEMORY_SIZE', '1000')))
    
    # LLM settings
    default_llm_provider: str = field(default_factory=lambda: os.getenv('DEFAULT_LLM_PROVIDER', 'openai'))
    huggingface_api_key: str = field(default_factory=lambda: os.getenv('HUGGINGFACE_API_KEY', ''))
    huggingface_model: str = field(default_factory=lambda: os.getenv('HUGGINGFACE_MODEL', 'gpt2'))
    
    # Memory settings
    memory_dir: Path = field(default_factory=lambda: Path(os.getenv('MEMORY_DIR', 'memory')))
    memory_backend: str = field(default_factory=lambda: os.getenv('MEMORY_BACKEND', 'json'))
    
    # Performance settings
    performance_threshold: float = field(default_factory=lambda: float(os.getenv('PERFORMANCE_THRESHOLD', '0.05')))
    improvement_threshold: float = field(default_factory=lambda: float(os.getenv('IMPROVEMENT_THRESHOLD', '0.02')))
    
    def validate(self) -> bool:
        """Validate agent configuration."""
        if self.timeout <= 0:
            raise ValueError(f"Invalid timeout: {self.timeout}")
        if self.max_concurrent_agents <= 0:
            raise ValueError(f"Invalid max_concurrent_agents: {self.max_concurrent_agents}")
        if self.memory_size <= 0:
            raise ValueError(f"Invalid memory_size: {self.memory_size}")
        if self.performance_threshold <= 0 or self.performance_threshold > 1:
            raise ValueError(f"Invalid performance_threshold: {self.performance_threshold}")
        if self.improvement_threshold <= 0 or self.improvement_threshold > 1:
            raise ValueError(f"Invalid improvement_threshold: {self.improvement_threshold}")
        
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timeout': self.timeout,
            'max_concurrent_agents': self.max_concurrent_agents,
            'memory_size': self.memory_size,
            'default_llm_provider': self.default_llm_provider,
            'huggingface_api_key': self.huggingface_api_key,
            'huggingface_model': self.huggingface_model,
            'memory_dir': str(self.memory_dir),
            'memory_backend': self.memory_backend,
            'performance_threshold': self.performance_threshold,
            'improvement_threshold': self.improvement_threshold
        }

@dataclass
class RiskConfig:
    """Configuration for risk management."""
    
    # Risk limits
    max_drawdown: float = field(default_factory=lambda: float(os.getenv('MAX_DRAWDOWN', '0.20')))
    max_correlation: float = field(default_factory=lambda: float(os.getenv('MAX_CORRELATION', '0.70')))
    max_concentration: float = field(default_factory=lambda: float(os.getenv('MAX_CONCENTRATION', '0.30')))
    
    # Volatility settings
    volatility_window: int = field(default_factory=lambda: int(os.getenv('VOLATILITY_WINDOW', '20')))
    volatility_threshold: float = field(default_factory=lambda: float(os.getenv('VOLATILITY_THRESHOLD', '0.30')))
    
    # VaR settings
    var_confidence: float = field(default_factory=lambda: float(os.getenv('VAR_CONFIDENCE', '0.95')))
    var_window: int = field(default_factory=lambda: int(os.getenv('VAR_WINDOW', '252')))
    
    # Stress testing
    stress_test_enabled: bool = field(default_factory=lambda: os.getenv('STRESS_TEST_ENABLED', 'true').lower() == 'true')
    stress_scenarios: List[str] = field(default_factory=lambda: os.getenv('STRESS_SCENARIOS', 'market_crash,recession,volatility_spike').split(','))
    
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
            raise ValueError(f"Invalid volatility_threshold: {self.volatility_threshold}")
        if self.var_confidence <= 0 or self.var_confidence >= 1:
            raise ValueError(f"Invalid var_confidence: {self.var_confidence}")
        if self.var_window <= 0:
            raise ValueError(f"Invalid var_window: {self.var_window}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_drawdown': self.max_drawdown,
            'max_correlation': self.max_correlation,
            'max_concentration': self.max_concentration,
            'volatility_window': self.volatility_window,
            'volatility_threshold': self.volatility_threshold,
            'var_confidence': self.var_confidence,
            'var_window': self.var_window,
            'stress_test_enabled': self.stress_test_enabled,
            'stress_scenarios': self.stress_scenarios
        }

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and evaluation."""
    
    # Metrics settings
    metrics_enabled: bool = field(default_factory=lambda: os.getenv('METRICS_ENABLED', 'true').lower() == 'true')
    metrics_path: Path = field(default_factory=lambda: Path(os.getenv('METRICS_PATH', 'logs/metrics.log')))
    
    # Evaluation settings
    evaluation_window: int = field(default_factory=lambda: int(os.getenv('EVALUATION_WINDOW', '252')))
    benchmark_symbol: str = field(default_factory=lambda: os.getenv('BENCHMARK_SYMBOL', 'SPY'))
    risk_free_rate: float = field(default_factory=lambda: float(os.getenv('RISK_FREE_RATE', '0.02')))
    
    # Reporting settings
    report_frequency: str = field(default_factory=lambda: os.getenv('REPORT_FREQUENCY', 'daily'))
    report_dir: Path = field(default_factory=lambda: Path(os.getenv('REPORT_DIR', 'reports')))
    
    # Alerting settings
    alert_enabled: bool = field(default_factory=lambda: os.getenv('ALERT_ENABLED', 'true').lower() == 'true')
    alert_email: str = field(default_factory=lambda: os.getenv('ALERT_EMAIL', ''))
    alert_webhook: str = field(default_factory=lambda: os.getenv('ALERT_WEBHOOK', ''))
    
    # Thresholds
    sharpe_threshold: float = field(default_factory=lambda: float(os.getenv('SHARPE_THRESHOLD', '1.0')))
    sortino_threshold: float = field(default_factory=lambda: float(os.getenv('SORTINO_THRESHOLD', '1.0')))
    max_drawdown_threshold: float = field(default_factory=lambda: float(os.getenv('MAX_DRAWDOWN_THRESHOLD', '0.15')))
    
    def validate(self) -> bool:
        """Validate performance configuration."""
        if self.evaluation_window <= 0:
            raise ValueError(f"Invalid evaluation_window: {self.evaluation_window}")
        if self.risk_free_rate < 0 or self.risk_free_rate > 1:
            raise ValueError(f"Invalid risk_free_rate: {self.risk_free_rate}")
        if self.report_frequency not in ['daily', 'weekly', 'monthly']:
            raise ValueError(f"Invalid report_frequency: {self.report_frequency}")
        if self.sharpe_threshold < 0:
            raise ValueError(f"Invalid sharpe_threshold: {self.sharpe_threshold}")
        if self.sortino_threshold < 0:
            raise ValueError(f"Invalid sortino_threshold: {self.sortino_threshold}")
        if self.max_drawdown_threshold <= 0 or self.max_drawdown_threshold > 1:
            raise ValueError(f"Invalid max_drawdown_threshold: {self.max_drawdown_threshold}")
        
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metrics_enabled': self.metrics_enabled,
            'metrics_path': str(self.metrics_path),
            'evaluation_window': self.evaluation_window,
            'benchmark_symbol': self.benchmark_symbol,
            'risk_free_rate': self.risk_free_rate,
            'report_frequency': self.report_frequency,
            'report_dir': str(self.report_dir),
            'alert_enabled': self.alert_enabled,
            'alert_email': self.alert_email,
            'alert_webhook': self.alert_webhook,
            'sharpe_threshold': self.sharpe_threshold,
            'sortino_threshold': self.sortino_threshold,
            'max_drawdown_threshold': self.max_drawdown_threshold
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