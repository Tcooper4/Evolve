"""
Application configuration management.

This module handles loading and managing application configuration from YAML files
and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    workers: int = 4
    timeout: int = 60
    reload: bool = False
    access_log: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/app.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    console: bool = True
    json_format: bool = False

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_pool_size: int = 10
    redis_retry_timeout: bool = True
    sqlite_path: str = "data/trading.db"
    sqlite_timeout: int = 30

@dataclass
class MarketDataConfig:
    """Market data configuration settings."""
    default_timeframe: str = "1d"
    default_assets: list = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    cache_ttl: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: int = 1
    providers: list = field(default_factory=list)

@dataclass
class ModelConfig:
    """Model configuration settings."""
    forecast_horizon: int = 30
    confidence_interval: float = 0.95
    min_training_samples: int = 1000
    update_frequency: int = 3600  # 1 hour
    ensemble_size: int = 5
    validation_split: float = 0.2
    technical_indicators: list = field(default_factory=list)

@dataclass
class StrategyConfig:
    """Strategy configuration settings."""
    position_size: float = 0.1
    stop_loss: float = 0.02
    take_profit: float = 0.04
    max_positions: int = 5
    rebalance_frequency: str = "1d"
    optimization_method: str = "bayesian"
    n_trials: int = 100
    cv_folds: int = 5
    optimization_timeout: int = 3600
    parallel_jobs: int = 4

@dataclass
class RiskConfig:
    """Risk management configuration settings."""
    max_drawdown: float = 0.2
    max_leverage: float = 3
    position_limits: Dict[str, float] = field(default_factory=dict)
    correlation_threshold: float = 0.7
    var_confidence: float = 0.95
    stress_test_scenarios: int = 10

@dataclass
class AgentConfig:
    """Agent configuration settings."""
    goal_planner_enabled: bool = True
    goal_update_frequency: int = 3600
    max_goals: int = 10
    router_enabled: bool = True
    router_confidence: float = 0.7
    fallback_agent: str = "commentary"
    self_improving_enabled: bool = True
    improvement_interval: int = 86400  # 24 hours
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class NLPConfig:
    """NLP configuration settings."""
    confidence_threshold: float = 0.7
    max_tokens: int = 1000
    temperature: float = 0.7
    cache_ttl: int = 3600  # 1 hour
    models: list = field(default_factory=list)
    templates: Dict[str, str] = field(default_factory=dict)

@dataclass
class APIConfig:
    """API configuration settings."""
    rate_limit: int = 100
    timeout: int = 30
    max_retries: int = 3
    cache_ttl: int = 300  # 5 minutes
    version: str = "v1"
    documentation: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""
    enabled: bool = True
    metrics: list = field(default_factory=list)
    alert_email: str = "alerts@example.com"
    alert_slack: Optional[str] = None
    alert_webhook: Optional[str] = None
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    dashboard_refresh: int = 30

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    ssl_enabled: bool = False
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    cors_origins: list = field(default_factory=lambda: ["*"])
    cors_methods: list = field(default_factory=lambda: ["GET", "POST"])
    cors_headers: list = field(default_factory=lambda: ["*"])
    rate_limiting_enabled: bool = True
    rate_limiting_window: int = 60
    rate_limiting_max_requests: int = 100
    auth_enabled: bool = False
    jwt_secret: Optional[str] = None
    token_expiry: int = 3600

@dataclass
class DevelopmentConfig:
    """Development configuration settings."""
    debug: bool = False
    hot_reload: bool = False
    profiling: bool = False
    test_mode: bool = False
    mock_data: bool = False
    seed_data: bool = False

@dataclass
class AppConfig:
    """Main application configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Server settings
        self.server.host = os.getenv("HOST", self.server.host)
        
        # Fix PORT parsing to handle Docker-style variable substitution
        port_env = os.getenv("PORT", str(self.server.port))
        try:
            # Handle Docker-style variable substitution like ${PORT:-8501}
            if port_env.startswith("${") and ":-" in port_env:
                # Extract the default value after :-
                default_value = port_env.split(":-")[-1].rstrip("}")
                port_env = default_value
            
            self.server.port = int(port_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid PORT value '{port_env}', using default {self.server.port}: {e}")
            # Keep the default value
        
        self.server.debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # Logging settings
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file = os.getenv("LOG_FILE", self.logging.file)
        
        # Database settings
        self.database.redis_host = os.getenv("REDIS_HOST", self.database.redis_host)
        
        # Fix Redis port parsing
        redis_port_env = os.getenv("REDIS_PORT", str(self.database.redis_port))
        try:
            if redis_port_env.startswith("${") and ":-" in redis_port_env:
                default_value = redis_port_env.split(":-")[-1].rstrip("}")
                redis_port_env = default_value
            self.database.redis_port = int(redis_port_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid REDIS_PORT value '{redis_port_env}', using default {self.database.redis_port}: {e}")
        
        self.database.redis_password = os.getenv("REDIS_PASSWORD") or None
        
        # Market data settings
        self.market_data.default_timeframe = os.getenv("DEFAULT_TIMEFRAME", self.market_data.default_timeframe)
        
        # Model settings
        forecast_horizon_env = os.getenv("FORECAST_HORIZON", str(self.models.forecast_horizon))
        try:
            if forecast_horizon_env.startswith("${") and ":-" in forecast_horizon_env:
                default_value = forecast_horizon_env.split(":-")[-1].rstrip("}")
                forecast_horizon_env = default_value
            self.models.forecast_horizon = int(forecast_horizon_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid FORECAST_HORIZON value '{forecast_horizon_env}', using default {self.models.forecast_horizon}: {e}")
        
        confidence_interval_env = os.getenv("CONFIDENCE_INTERVAL", str(self.models.confidence_interval))
        try:
            if confidence_interval_env.startswith("${") and ":-" in confidence_interval_env:
                default_value = confidence_interval_env.split(":-")[-1].rstrip("}")
                confidence_interval_env = default_value
            self.models.confidence_interval = float(confidence_interval_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid CONFIDENCE_INTERVAL value '{confidence_interval_env}', using default {self.models.confidence_interval}: {e}")
        
        # Strategy settings
        position_size_env = os.getenv("POSITION_SIZE", str(self.strategies.position_size))
        try:
            if position_size_env.startswith("${") and ":-" in position_size_env:
                default_value = position_size_env.split(":-")[-1].rstrip("}")
                position_size_env = default_value
            self.strategies.position_size = float(position_size_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid POSITION_SIZE value '{position_size_env}', using default {self.strategies.position_size}: {e}")
        
        stop_loss_env = os.getenv("STOP_LOSS", str(self.strategies.stop_loss))
        try:
            if stop_loss_env.startswith("${") and ":-" in stop_loss_env:
                default_value = stop_loss_env.split(":-")[-1].rstrip("}")
                stop_loss_env = default_value
            self.strategies.stop_loss = float(stop_loss_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid STOP_LOSS value '{stop_loss_env}', using default {self.strategies.stop_loss}: {e}")
        
        # Risk settings
        max_drawdown_env = os.getenv("MAX_DRAWDOWN", str(self.risk.max_drawdown))
        try:
            if max_drawdown_env.startswith("${") and ":-" in max_drawdown_env:
                default_value = max_drawdown_env.split(":-")[-1].rstrip("}")
                max_drawdown_env = default_value
            self.risk.max_drawdown = float(max_drawdown_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid MAX_DRAWDOWN value '{max_drawdown_env}', using default {self.risk.max_drawdown}: {e}")
        
        max_leverage_env = os.getenv("MAX_LEVERAGE", str(self.risk.max_leverage))
        try:
            if max_leverage_env.startswith("${") and ":-" in max_leverage_env:
                default_value = max_leverage_env.split(":-")[-1].rstrip("}")
                max_leverage_env = default_value
            self.risk.max_leverage = float(max_leverage_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid MAX_LEVERAGE value '{max_leverage_env}', using default {self.risk.max_leverage}: {e}")
        
        # Agent settings
        self.agents.goal_planner_enabled = os.getenv("GOAL_PLANNER_ENABLED", "true").lower() == "true"
        self.agents.router_enabled = os.getenv("ROUTER_ENABLED", "true").lower() == "true"
        
        # NLP settings
        nlp_confidence_env = os.getenv("NLP_CONFIDENCE_THRESHOLD", str(self.nlp.confidence_threshold))
        try:
            if nlp_confidence_env.startswith("${") and ":-" in nlp_confidence_env:
                default_value = nlp_confidence_env.split(":-")[-1].rstrip("}")
                nlp_confidence_env = default_value
            self.nlp.confidence_threshold = float(nlp_confidence_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid NLP_CONFIDENCE_THRESHOLD value '{nlp_confidence_env}', using default {self.nlp.confidence_threshold}: {e}")
        
        nlp_tokens_env = os.getenv("NLP_MAX_TOKENS", str(self.nlp.max_tokens))
        try:
            if nlp_tokens_env.startswith("${") and ":-" in nlp_tokens_env:
                default_value = nlp_tokens_env.split(":-")[-1].rstrip("}")
                nlp_tokens_env = default_value
            self.nlp.max_tokens = int(nlp_tokens_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid NLP_MAX_TOKENS value '{nlp_tokens_env}', using default {self.nlp.max_tokens}: {e}")
        
        # API settings
        api_rate_limit_env = os.getenv("API_RATE_LIMIT", str(self.api.rate_limit))
        try:
            if api_rate_limit_env.startswith("${") and ":-" in api_rate_limit_env:
                default_value = api_rate_limit_env.split(":-")[-1].rstrip("}")
                api_rate_limit_env = default_value
            self.api.rate_limit = int(api_rate_limit_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid API_RATE_LIMIT value '{api_rate_limit_env}', using default {self.api.rate_limit}: {e}")
        
        api_timeout_env = os.getenv("API_TIMEOUT", str(self.api.timeout))
        try:
            if api_timeout_env.startswith("${") and ":-" in api_timeout_env:
                default_value = api_timeout_env.split(":-")[-1].rstrip("}")
                api_timeout_env = default_value
            self.api.timeout = int(api_timeout_env)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Invalid API_TIMEOUT value '{api_timeout_env}', using default {self.api.timeout}: {e}")
        
        # Monitoring settings
        self.monitoring.enabled = os.getenv("MONITORING_ENABLED", "true").lower() == "true"
        self.monitoring.alert_email = os.getenv("ALERT_EMAIL", self.monitoring.alert_email)
        
        # Security settings
        self.security.ssl_enabled = os.getenv("SSL_ENABLED", "false").lower() == "true"
        self.security.auth_enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"
        
        # Development settings
        self.development.debug = os.getenv("DEV_DEBUG", "false").lower() == "true"
        self.development.test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'AppConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            AppConfig instance
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return cls()
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Create config instance
            config = cls()
            
            # Update with YAML data
            config._update_from_dict(config_data)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return cls()
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        try:
            # Server settings
            if 'server' in config_data:
                server_data = config_data['server']
                self.server.host = server_data.get('host', self.server.host)
                self.server.port = int(server_data.get('port', self.server.port))
                self.server.debug = server_data.get('debug', self.server.debug)
                self.server.workers = int(server_data.get('workers', self.server.workers))
                self.server.timeout = int(server_data.get('timeout', self.server.timeout))
                self.server.reload = server_data.get('reload', self.server.reload)
                self.server.access_log = server_data.get('access_log', self.server.access_log)
            
            # Logging settings
            if 'logging' in config_data:
                logging_data = config_data['logging']
                self.logging.level = logging_data.get('level', self.logging.level)
                self.logging.format = logging_data.get('format', self.logging.format)
                self.logging.file = logging_data.get('file', self.logging.file)
                self.logging.max_size = int(logging_data.get('max_size', self.logging.max_size))
                self.logging.backup_count = int(logging_data.get('backup_count', self.logging.backup_count))
                self.logging.console = logging_data.get('console', self.logging.console)
                self.logging.json_format = logging_data.get('json_format', self.logging.json_format)
            
            # Database settings
            if 'database' in config_data:
                db_data = config_data['database']
                if 'redis' in db_data:
                    redis_data = db_data['redis']
                    self.database.redis_host = redis_data.get('host', self.database.redis_host)
                    self.database.redis_port = int(redis_data.get('port', self.database.redis_port))
                    self.database.redis_db = int(redis_data.get('db', self.database.redis_db))
                    self.database.redis_password = redis_data.get('password') or None
                    self.database.redis_ssl = redis_data.get('ssl', self.database.redis_ssl)
                    self.database.redis_pool_size = int(redis_data.get('pool_size', self.database.redis_pool_size))
                    self.database.redis_retry_timeout = redis_data.get('retry_on_timeout', self.database.redis_retry_timeout)
                
                if 'sqlite' in db_data:
                    sqlite_data = db_data['sqlite']
                    self.database.sqlite_path = sqlite_data.get('path', self.database.sqlite_path)
                    self.database.sqlite_timeout = int(sqlite_data.get('timeout', self.database.sqlite_timeout))
            
            # Market data settings
            if 'market_data' in config_data:
                md_data = config_data['market_data']
                self.market_data.default_timeframe = md_data.get('default_timeframe', self.market_data.default_timeframe)
                self.market_data.default_assets = md_data.get('default_assets', self.market_data.default_assets)
                self.market_data.cache_ttl = int(md_data.get('cache_ttl', self.market_data.cache_ttl))
                self.market_data.max_retries = int(md_data.get('max_retries', self.market_data.max_retries))
                self.market_data.retry_delay = int(md_data.get('retry_delay', self.market_data.retry_delay))
                self.market_data.providers = md_data.get('providers', self.market_data.providers)
            
            # Model settings
            if 'models' in config_data:
                models_data = config_data['models']
                if 'forecast' in models_data:
                    forecast_data = models_data['forecast']
                    self.models.forecast_horizon = int(forecast_data.get('horizon', self.models.forecast_horizon))
                    self.models.confidence_interval = float(forecast_data.get('confidence_interval', self.models.confidence_interval))
                    self.models.min_training_samples = int(forecast_data.get('min_training_samples', self.models.min_training_samples))
                    self.models.update_frequency = int(forecast_data.get('update_frequency', self.models.update_frequency))
                    self.models.ensemble_size = int(forecast_data.get('ensemble_size', self.models.ensemble_size))
                    self.models.validation_split = float(forecast_data.get('validation_split', self.models.validation_split))
                
                if 'technical' in models_data:
                    tech_data = models_data['technical']
                    self.models.technical_indicators = tech_data.get('indicators', self.models.technical_indicators)
            
            # Strategy settings
            if 'strategies' in config_data:
                strategies_data = config_data['strategies']
                if 'default' in strategies_data:
                    default_data = strategies_data['default']
                    self.strategies.position_size = float(default_data.get('position_size', self.strategies.position_size))
                    self.strategies.stop_loss = float(default_data.get('stop_loss', self.strategies.stop_loss))
                    self.strategies.take_profit = float(default_data.get('take_profit', self.strategies.take_profit))
                    self.strategies.max_positions = int(default_data.get('max_positions', self.strategies.max_positions))
                    self.strategies.rebalance_frequency = default_data.get('rebalance_frequency', self.strategies.rebalance_frequency)
                
                if 'optimization' in strategies_data:
                    opt_data = strategies_data['optimization']
                    self.strategies.optimization_method = opt_data.get('method', self.strategies.optimization_method)
                    self.strategies.n_trials = int(opt_data.get('n_trials', self.strategies.n_trials))
                    self.strategies.cv_folds = int(opt_data.get('cv_folds', self.strategies.cv_folds))
                    self.strategies.optimization_timeout = int(opt_data.get('timeout', self.strategies.optimization_timeout))
                    self.strategies.parallel_jobs = int(opt_data.get('parallel_jobs', self.strategies.parallel_jobs))
            
            # Risk settings
            if 'risk' in config_data:
                risk_data = config_data['risk']
                self.risk.max_drawdown = float(risk_data.get('max_drawdown', self.risk.max_drawdown))
                self.risk.max_leverage = float(risk_data.get('max_leverage', self.risk.max_leverage))
                self.risk.position_limits = risk_data.get('position_limits', self.risk.position_limits)
                self.risk.correlation_threshold = float(risk_data.get('correlation_threshold', self.risk.correlation_threshold))
                self.risk.var_confidence = float(risk_data.get('var_confidence', self.risk.var_confidence))
                self.risk.stress_test_scenarios = int(risk_data.get('stress_test_scenarios', self.risk.stress_test_scenarios))
            
            # Agent settings
            if 'agents' in config_data:
                agents_data = config_data['agents']
                if 'goal_planner' in agents_data:
                    gp_data = agents_data['goal_planner']
                    self.agents.goal_planner_enabled = gp_data.get('enabled', self.agents.goal_planner_enabled)
                    self.agents.goal_update_frequency = int(gp_data.get('update_frequency', self.agents.goal_update_frequency))
                    self.agents.max_goals = int(gp_data.get('max_goals', self.agents.max_goals))
                
                if 'router' in agents_data:
                    router_data = agents_data['router']
                    self.agents.router_enabled = router_data.get('enabled', self.agents.router_enabled)
                    self.agents.router_confidence = float(router_data.get('confidence_threshold', self.agents.router_confidence))
                    self.agents.fallback_agent = router_data.get('fallback_agent', self.agents.fallback_agent)
                
                if 'self_improving' in agents_data:
                    si_data = agents_data['self_improving']
                    self.agents.self_improving_enabled = si_data.get('enabled', self.agents.self_improving_enabled)
                    self.agents.improvement_interval = int(si_data.get('improvement_interval', self.agents.improvement_interval))
                    self.agents.performance_thresholds = si_data.get('performance_thresholds', self.agents.performance_thresholds)
            
            # NLP settings
            if 'nlp' in config_data:
                nlp_data = config_data['nlp']
                self.nlp.confidence_threshold = float(nlp_data.get('confidence_threshold', self.nlp.confidence_threshold))
                self.nlp.max_tokens = int(nlp_data.get('max_tokens', self.nlp.max_tokens))
                self.nlp.temperature = float(nlp_data.get('temperature', self.nlp.temperature))
                self.nlp.cache_ttl = int(nlp_data.get('cache_ttl', self.nlp.cache_ttl))
                self.nlp.models = nlp_data.get('models', self.nlp.models)
                self.nlp.templates = nlp_data.get('templates', self.nlp.templates)
            
            # API settings
            if 'api' in config_data:
                api_data = config_data['api']
                self.api.rate_limit = int(api_data.get('rate_limit', self.api.rate_limit))
                self.api.timeout = int(api_data.get('timeout', self.api.timeout))
                self.api.max_retries = int(api_data.get('max_retries', self.api.max_retries))
                self.api.cache_ttl = int(api_data.get('cache_ttl', self.api.cache_ttl))
                self.api.version = api_data.get('version', self.api.version)
                self.api.documentation = api_data.get('documentation', self.api.documentation)
            
            # Monitoring settings
            if 'monitoring' in config_data:
                monitoring_data = config_data['monitoring']
                self.monitoring.enabled = monitoring_data.get('enabled', self.monitoring.enabled)
                self.monitoring.metrics = monitoring_data.get('metrics', self.monitoring.metrics)
                
                if 'alert' in monitoring_data:
                    alert_data = monitoring_data['alert']
                    self.monitoring.alert_email = alert_data.get('email', self.monitoring.alert_email)
                    self.monitoring.alert_slack = alert_data.get('slack')
                    self.monitoring.alert_webhook = alert_data.get('webhook')
                
                if 'dashboard' in monitoring_data:
                    dashboard_data = monitoring_data['dashboard']
                    self.monitoring.dashboard_enabled = dashboard_data.get('enabled', self.monitoring.dashboard_enabled)
                    self.monitoring.dashboard_port = int(dashboard_data.get('port', self.monitoring.dashboard_port))
                    self.monitoring.dashboard_refresh = int(dashboard_data.get('refresh_interval', self.monitoring.dashboard_refresh))
            
            # Security settings
            if 'security' in config_data:
                security_data = config_data['security']
                if 'ssl' in security_data:
                    ssl_data = security_data['ssl']
                    self.security.ssl_enabled = ssl_data.get('enabled', self.security.ssl_enabled)
                    self.security.ssl_cert_file = ssl_data.get('cert_file')
                    self.security.ssl_key_file = ssl_data.get('key_file')
                
                if 'cors' in security_data:
                    cors_data = security_data['cors']
                    self.security.cors_origins = cors_data.get('allowed_origins', self.security.cors_origins)
                    self.security.cors_methods = cors_data.get('allowed_methods', self.security.cors_methods)
                    self.security.cors_headers = cors_data.get('allowed_headers', self.security.cors_headers)
                
                if 'rate_limiting' in security_data:
                    rate_data = security_data['rate_limiting']
                    self.security.rate_limiting_enabled = rate_data.get('enabled', self.security.rate_limiting_enabled)
                    self.security.rate_limiting_window = int(rate_data.get('window', self.security.rate_limiting_window))
                    self.security.rate_limiting_max_requests = int(rate_data.get('max_requests', self.security.rate_limiting_max_requests))
                
                if 'authentication' in security_data:
                    auth_data = security_data['authentication']
                    self.security.auth_enabled = auth_data.get('enabled', self.security.auth_enabled)
                    self.security.jwt_secret = auth_data.get('jwt_secret')
                    self.security.token_expiry = int(auth_data.get('token_expiry', self.security.token_expiry))
            
            # Development settings
            if 'development' in config_data:
                dev_data = config_data['development']
                self.development.debug = dev_data.get('debug', self.development.debug)
                self.development.hot_reload = dev_data.get('hot_reload', self.development.hot_reload)
                self.development.profiling = dev_data.get('profiling', self.development.profiling)
                self.development.test_mode = dev_data.get('test_mode', self.development.test_mode)
                self.development.mock_data = dev_data.get('mock_data', self.development.mock_data)
                self.development.seed_data = dev_data.get('seed_data', self.development.seed_data)
                
        except Exception as e:
            logger.error(f"Error updating configuration from dictionary: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'server': {
                'host': self.server.host,
                'port': self.server.port,
                'debug': self.server.debug,
                'workers': self.server.workers,
                'timeout': self.server.timeout,
                'reload': self.server.reload,
                'access_log': self.server.access_log
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file': self.logging.file,
                'max_size': self.logging.max_size,
                'backup_count': self.logging.backup_count,
                'console': self.logging.console,
                'json_format': self.logging.json_format
            },
            'database': {
                'redis': {
                    'host': self.database.redis_host,
                    'port': self.database.redis_port,
                    'db': self.database.redis_db,
                    'password': self.database.redis_password,
                    'ssl': self.database.redis_ssl,
                    'pool_size': self.database.redis_pool_size,
                    'retry_on_timeout': self.database.redis_retry_timeout
                },
                'sqlite': {
                    'path': self.database.sqlite_path,
                    'timeout': self.database.sqlite_timeout
                }
            },
            'market_data': {
                'default_timeframe': self.market_data.default_timeframe,
                'default_assets': self.market_data.default_assets,
                'cache_ttl': self.market_data.cache_ttl,
                'max_retries': self.market_data.max_retries,
                'retry_delay': self.market_data.retry_delay,
                'providers': self.market_data.providers
            },
            'models': {
                'forecast': {
                    'horizon': self.models.forecast_horizon,
                    'confidence_interval': self.models.confidence_interval,
                    'min_training_samples': self.models.min_training_samples,
                    'update_frequency': self.models.update_frequency,
                    'ensemble_size': self.models.ensemble_size,
                    'validation_split': self.models.validation_split
                },
                'technical': {
                    'indicators': self.models.technical_indicators
                }
            },
            'strategies': {
                'default': {
                    'position_size': self.strategies.position_size,
                    'stop_loss': self.strategies.stop_loss,
                    'take_profit': self.strategies.take_profit,
                    'max_positions': self.strategies.max_positions,
                    'rebalance_frequency': self.strategies.rebalance_frequency
                },
                'optimization': {
                    'method': self.strategies.optimization_method,
                    'n_trials': self.strategies.n_trials,
                    'cv_folds': self.strategies.cv_folds,
                    'timeout': self.strategies.optimization_timeout,
                    'parallel_jobs': self.strategies.parallel_jobs
                }
            },
            'risk': {
                'max_drawdown': self.risk.max_drawdown,
                'max_leverage': self.risk.max_leverage,
                'position_limits': self.risk.position_limits,
                'correlation_threshold': self.risk.correlation_threshold,
                'var_confidence': self.risk.var_confidence,
                'stress_test_scenarios': self.risk.stress_test_scenarios
            },
            'agents': {
                'goal_planner': {
                    'enabled': self.agents.goal_planner_enabled,
                    'update_frequency': self.agents.goal_update_frequency,
                    'max_goals': self.agents.max_goals
                },
                'router': {
                    'enabled': self.agents.router_enabled,
                    'confidence_threshold': self.agents.router_confidence,
                    'fallback_agent': self.agents.fallback_agent
                },
                'self_improving': {
                    'enabled': self.agents.self_improving_enabled,
                    'improvement_interval': self.agents.improvement_interval,
                    'performance_thresholds': self.agents.performance_thresholds
                }
            },
            'nlp': {
                'confidence_threshold': self.nlp.confidence_threshold,
                'max_tokens': self.nlp.max_tokens,
                'temperature': self.nlp.temperature,
                'cache_ttl': self.nlp.cache_ttl,
                'models': self.nlp.models,
                'templates': self.nlp.templates
            },
            'api': {
                'rate_limit': self.api.rate_limit,
                'timeout': self.api.timeout,
                'max_retries': self.api.max_retries,
                'cache_ttl': self.api.cache_ttl,
                'version': self.api.version,
                'documentation': self.api.documentation
            },
            'monitoring': {
                'enabled': self.monitoring.enabled,
                'metrics': self.monitoring.metrics,
                'alert': {
                    'email': self.monitoring.alert_email,
                    'slack': self.monitoring.alert_slack,
                    'webhook': self.monitoring.alert_webhook
                },
                'dashboard': {
                    'enabled': self.monitoring.dashboard_enabled,
                    'port': self.monitoring.dashboard_port,
                    'refresh_interval': self.monitoring.dashboard_refresh
                }
            },
            'security': {
                'ssl': {
                    'enabled': self.security.ssl_enabled,
                    'cert_file': self.security.ssl_cert_file,
                    'key_file': self.security.ssl_key_file
                },
                'cors': {
                    'allowed_origins': self.security.cors_origins,
                    'allowed_methods': self.security.cors_methods,
                    'allowed_headers': self.security.cors_headers
                },
                'rate_limiting': {
                    'enabled': self.security.rate_limiting_enabled,
                    'window': self.security.rate_limiting_window,
                    'max_requests': self.security.rate_limiting_max_requests
                },
                'authentication': {
                    'enabled': self.security.auth_enabled,
                    'jwt_secret': self.security.jwt_secret,
                    'token_expiry': self.security.token_expiry
                }
            },
            'development': {
                'debug': self.development.debug,
                'hot_reload': self.development.hot_reload,
                'profiling': self.development.profiling,
                'test_mode': self.development.test_mode,
                'mock_data': self.development.mock_data,
                'seed_data': self.development.seed_data
            }
        }
    
    def save_to_yaml(self, config_path: Union[str, Path]) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            True if saved successfully
        """
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = self.to_dict()
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            return False
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration settings.
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Server validation
        if self.server.port < 1 or self.server.port > 65535:
            errors.append("Server port must be between 1 and 65535")
        
        if self.server.workers < 1:
            errors.append("Server workers must be at least 1")
        
        # Database validation
        if self.database.redis_port < 1 or self.database.redis_port > 65535:
            errors.append("Redis port must be between 1 and 65535")
        
        # Model validation
        if self.models.forecast_horizon < 1:
            errors.append("Forecast horizon must be at least 1")
        
        if not 0 < self.models.confidence_interval < 1:
            errors.append("Confidence interval must be between 0 and 1")
        
        # Strategy validation
        if self.strategies.position_size <= 0 or self.strategies.position_size > 1:
            errors.append("Position size must be between 0 and 1")
        
        if self.strategies.stop_loss <= 0:
            errors.append("Stop loss must be positive")
        
        if self.strategies.take_profit <= 0:
            errors.append("Take profit must be positive")
        
        # Risk validation
        if self.risk.max_drawdown <= 0 or self.risk.max_drawdown > 1:
            errors.append("Max drawdown must be between 0 and 1")
        
        if self.risk.max_leverage <= 0:
            errors.append("Max leverage must be positive")
        
        # Agent validation
        if self.agents.router_confidence < 0 or self.agents.router_confidence > 1:
            errors.append("Router confidence must be between 0 and 1")
        
        # NLP validation
        if self.nlp.confidence_threshold < 0 or self.nlp.confidence_threshold > 1:
            errors.append("NLP confidence threshold must be between 0 and 1")
        
        if self.nlp.max_tokens < 1:
            errors.append("NLP max tokens must be at least 1")
        
        # API validation
        if self.api.rate_limit < 1:
            errors.append("API rate limit must be at least 1")
        
        if self.api.timeout < 1:
            errors.append("API timeout must be at least 1")
        
        # Security validation
        if self.security.token_expiry < 1:
            errors.append("Token expiry must be at least 1 second")
        
        # Warnings
        if self.server.debug and not self.development.debug:
            warnings.append("Server debug is enabled but development debug is not")
        
        if self.security.auth_enabled and not self.security.jwt_secret:
            warnings.append("Authentication is enabled but JWT secret is not set")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

# Global configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """
    Get global configuration instance.
    
    Returns:
        AppConfig instance
    """
    global _config
    if _config is None:
        _config = AppConfig.from_yaml("config/app_config.yaml")
    return _config

def set_config(config: AppConfig) -> None:
    """
    Set global configuration instance.
    
    Args:
        config: AppConfig instance
    """
    global _config
    _config = config 