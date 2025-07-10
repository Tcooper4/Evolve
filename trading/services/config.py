"""
Configuration Management for QuantGPT

Centralized configuration system with environment-based settings,
validation, and type safety.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class QuantGPTConfig:
    """Configuration for QuantGPT system."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 500
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Trading Context
    available_symbols: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'NVDA', 'TSLA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN'
    ])
    available_timeframes: List[str] = field(default_factory=lambda: [
        '1m', '5m', '15m', '1h', '4h', '1d'
    ])
    available_periods: List[str] = field(default_factory=lambda: [
        '7d', '14d', '30d', '90d', '180d', '1y'
    ])
    available_models: List[str] = field(default_factory=lambda: [
        'lstm', 'xgboost', 'ensemble', 'transformer', 'tcn'
    ])
    
    # Performance Settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_calls: int = 100
    rate_limit_period: int = 3600  # 1 hour
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Validation
    validate_inputs: bool = True
    strict_mode: bool = False
    
    @classmethod
    def from_env(cls) -> 'QuantGPTConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # OpenAI
        config.openai_api_key = os.getenv('OPENAI_API_KEY')
        config.openai_model = os.getenv('OPENAI_MODEL', config.openai_model)
        config.openai_temperature = float(os.getenv('OPENAI_TEMPERATURE', config.openai_temperature))
        config.openai_max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', config.openai_max_tokens))
        
        # Redis
        config.redis_host = os.getenv('REDIS_HOST', config.redis_host)
        config.redis_port = int(os.getenv('REDIS_PORT', config.redis_port))
        config.redis_db = int(os.getenv('REDIS_DB', config.redis_db))
        config.redis_password = os.getenv('REDIS_PASSWORD')
        
        # Performance
        config.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        config.cache_ttl = int(os.getenv('CACHE_TTL', config.cache_ttl))
        config.max_retries = int(os.getenv('MAX_RETRIES', config.max_retries))
        config.retry_delay = float(os.getenv('RETRY_DELAY', config.retry_delay))
        
        # Rate Limiting
        config.rate_limit_enabled = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        config.rate_limit_calls = int(os.getenv('RATE_LIMIT_CALLS', config.rate_limit_calls))
        config.rate_limit_period = int(os.getenv('RATE_LIMIT_PERIOD', config.rate_limit_period))
        
        # Logging
        config.log_level = os.getenv('LOG_LEVEL', config.log_level)
        
        # Validation
        config.validate_inputs = os.getenv('VALIDATE_INPUTS', 'true').lower() == 'true'
        config.strict_mode = os.getenv('STRICT_MODE', 'false').lower() == 'true'
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'QuantGPTConfig':
        """Create configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            config = cls()
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        # Validate OpenAI settings
        if self.openai_temperature < 0 or self.openai_temperature > 2:
            errors.append("openai_temperature must be between 0 and 2")
        
        if self.openai_max_tokens < 1:
            errors.append("openai_max_tokens must be positive")
        
        # Validate Redis settings
        if self.redis_port < 1 or self.redis_port > 65535:
            errors.append("redis_port must be between 1 and 65535")
        
        if self.redis_db < 0:
            errors.append("redis_db must be non-negative")
        
        # Validate performance settings
        if self.cache_ttl < 0:
            errors.append("cache_ttl must be non-negative")
        
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        
        if self.retry_delay < 0:
            errors.append("retry_delay must be non-negative")
        
        # Validate rate limiting
        if self.rate_limit_calls < 1:
            errors.append("rate_limit_calls must be positive")
        
        if self.rate_limit_period < 1:
            errors.append("rate_limit_period must be positive")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'openai_api_key': '***' if self.openai_api_key else None,
            'openai_model': self.openai_model,
            'openai_temperature': self.openai_temperature,
            'openai_max_tokens': self.openai_max_tokens,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'redis_db': self.redis_db,
            'redis_password': '***' if self.redis_password else None,
            'available_symbols': self.available_symbols,
            'available_timeframes': self.available_timeframes,
            'available_periods': self.available_periods,
            'available_models': self.available_models,
            'cache_enabled': self.cache_enabled,
            'cache_ttl': self.cache_ttl,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'rate_limit_enabled': self.rate_limit_enabled,
            'rate_limit_calls': self.rate_limit_calls,
            'rate_limit_period': self.rate_limit_period,
            'log_level': self.log_level,
            'validate_inputs': self.validate_inputs,
            'strict_mode': self.strict_mode
        } 