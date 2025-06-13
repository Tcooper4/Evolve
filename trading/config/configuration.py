import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class ConfigManager:
    """Manager for handling configuration settings."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config = {}

        # Automatically load settings from environment variables
        env_config = self.create_config_from_env()
        if env_config:
            self.config.update(env_config)
        
    def load_config(self, config_type: str) -> Dict[str, Any]:
        """Load configuration settings.
        
        Args:
            config_type: Type of configuration to load
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / f"{config_type}.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
        
    def save_config(self, config: Any, config_path: str) -> None:
        """Save configuration settings.
        
        Args:
            config: Configuration object or dictionary
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(config, 'to_dict'):
            config_data = config.to_dict()
        else:
            config_data = config
            
        with open(config_path, 'w') as f:
            if config_path.suffix == '.yaml':
                yaml.dump(config_data, f)
            else:
                json.dump(config_data, f, indent=4)
                
    def delete_config(self, config_path: str) -> None:
        """Delete a configuration file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        if config_path.exists():
            config_path.unlink()
            
    def create_config_from_env(self) -> Dict[str, Any]:
        """Create configuration from environment variables.
        
        Returns:
            Configuration dictionary
        """
        config = {}
        for key, value in os.environ.items():
            if key.startswith('TRADING_'):
                config_key = key[8:].lower()
                try:
                    # Try to convert to appropriate type
                    if value.lower() in ('true', 'false'):
                        config[config_key] = value.lower() == 'true'
                    elif value.isdigit():
                        config[config_key] = int(value)
                    elif value.replace('.', '').isdigit():
                        config[config_key] = float(value)
                    else:
                        config[config_key] = value
                except ValueError:
                    config[config_key] = value
        return config

class ModelConfig:
    """Configuration class for model settings."""
    
    def __init__(self, model_type: str = 'transformer', model_name: str = None,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dropout: float = 0.1, batch_size: int = 32,
                 learning_rate: float = 0.001, **kwargs):
        """Initialize model configuration.
        
        Args:
            model_type: Type of model (e.g., 'lstm', 'transformer', 'tcn')
            model_name: Name of the model
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of layers
            dropout: Dropout rate
            batch_size: Batch size
            learning_rate: Learning rate
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model_name = model_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.parameters = kwargs
        
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.model_type not in ['lstm', 'transformer', 'tcn', 'gnn']:
            raise ValueError(f"Invalid model type: {self.model_type}")
            
        if self.d_model <= 0:
            raise ValueError(f"Invalid d_model: {self.d_model}")
            
        if self.nhead <= 0:
            raise ValueError(f"Invalid nhead: {self.nhead}")
            
        if self.num_layers <= 0:
            raise ValueError(f"Invalid num_layers: {self.num_layers}")
            
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Invalid dropout: {self.dropout}")
            
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
            
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}")
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'model_name': self.model_name,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            **self.parameters
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ModelConfig instance
        """
        return cls(**config_dict)
        
    def merge(self, other: Any) -> Dict[str, Any]:
        """Merge with another configuration.
        
        Args:
            other: Another configuration object or dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if hasattr(other, 'to_dict'):
            other_dict = other.to_dict()
        else:
            other_dict = other
            
        return {**self.to_dict(), **other_dict}

class DataConfig:
    """Configuration class for data settings."""
    
    def __init__(self, data_source: str = 'yfinance', symbols: list = None,
                 start_date: str = None, end_date: str = None,
                 features: list = None, target: str = None,
                 frequency: str = '1d'):
        """Initialize data configuration.
        
        Args:
            data_source: Source of the data (e.g., 'csv', 'api', 'database')
            symbols: List of symbols to fetch data for
            start_date: Start date for data collection
            end_date: End date for data collection
            features: List of feature columns to use
            target: Target column for prediction
            frequency: Data frequency (e.g., '1d', '1h', '1m')
        """
        self.data_source = data_source
        self.symbols = symbols or []
        self.start_date = start_date
        self.end_date = end_date
        self.features = features or ['Close']
        self.target = target
        self.frequency = frequency
        
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.data_source not in ['yfinance', 'csv', 'api', 'database']:
            raise ValueError(f"Invalid data source: {self.data_source}")
            
        if not self.symbols:
            raise ValueError("Symbols list cannot be empty")
            
        if not self.features:
            raise ValueError("Features list cannot be empty")
            
        if self.start_date and self.end_date:
            try:
                start = datetime.strptime(self.start_date, '%Y-%m-%d')
                end = datetime.strptime(self.end_date, '%Y-%m-%d')
                if start > end:
                    raise ValueError("Start date must be before end date")
            except ValueError as e:
                raise ValueError(f"Invalid date format: {e}")
                
        if self.frequency not in ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1M']:
            raise ValueError(f"Invalid frequency: {self.frequency}")
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data_source': self.data_source,
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'features': self.features,
            'target': self.target,
            'frequency': self.frequency
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            DataConfig instance
        """
        return cls(**config_dict)
        
    def merge(self, other: Any) -> Dict[str, Any]:
        """Merge with another configuration.
        
        Args:
            other: Another configuration object or dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if hasattr(other, 'to_dict'):
            other_dict = other.to_dict()
        else:
            other_dict = other
            
        return {**self.to_dict(), **other_dict}

class TrainingConfig:
    """Configuration class for model training settings."""
    
    def __init__(self, epochs: int = 100, batch_size: int = 32,
                 learning_rate: float = 0.001, optimizer: str = 'adam',
                 loss_function: str = 'mse', validation_split: float = 0.2,
                 early_stopping: bool = True, early_stopping_patience: int = 10,
                 learning_rate_scheduler: bool = True, scheduler_patience: int = 5,
                 gradient_clipping: bool = True, max_grad_norm: float = 1.0):
        """Initialize training configuration.
        
        Args:
            epochs: Number of training epochs
            batch_size: Size of training batches
            learning_rate: Learning rate for optimizer
            optimizer: Name of optimizer to use
            loss_function: Name of loss function to use
            validation_split: Fraction of data to use for validation
            early_stopping: Whether to use early stopping
            early_stopping_patience: Number of epochs to wait before early stopping
            learning_rate_scheduler: Whether to use learning rate scheduler
            scheduler_patience: Number of epochs to wait before reducing learning rate
            gradient_clipping: Whether to use gradient clipping
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate_scheduler = learning_rate_scheduler
        self.scheduler_patience = scheduler_patience
        self.gradient_clipping = gradient_clipping
        self.max_grad_norm = max_grad_norm
        
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs: {self.epochs}")
            
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
            
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}")
            
        if self.optimizer not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
            
        if self.loss_function not in ['mse', 'mae', 'huber', 'smooth_l1']:
            raise ValueError(f"Invalid loss_function: {self.loss_function}")
            
        if not 0 < self.validation_split < 1:
            raise ValueError(f"Invalid validation_split: {self.validation_split}")
            
        if self.early_stopping_patience <= 0:
            raise ValueError(f"Invalid early_stopping_patience: {self.early_stopping_patience}")
            
        if self.scheduler_patience <= 0:
            raise ValueError(f"Invalid scheduler_patience: {self.scheduler_patience}")
            
        if self.max_grad_norm <= 0:
            raise ValueError(f"Invalid max_grad_norm: {self.max_grad_norm}")
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'loss_function': self.loss_function,
            'validation_split': self.validation_split,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'learning_rate_scheduler': self.learning_rate_scheduler,
            'scheduler_patience': self.scheduler_patience,
            'gradient_clipping': self.gradient_clipping,
            'max_grad_norm': self.max_grad_norm
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            TrainingConfig instance
        """
        return cls(**config_dict)
        
    def merge(self, other: Any) -> Dict[str, Any]:
        """Merge with another configuration.
        
        Args:
            other: Another configuration object or dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if hasattr(other, 'to_dict'):
            other_dict = other.to_dict()
        else:
            other_dict = other
            
        return {**self.to_dict(), **other_dict}

class WebConfig:
    """Configuration class for web interface settings."""
    
    def __init__(self, host: str = 'localhost', port: int = 5000,
                 debug: bool = False, secret_key: str = None,
                 static_folder: str = 'static', template_folder: str = 'templates',
                 ssl_cert: str = None, ssl_key: str = None):
        """Initialize web configuration.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            debug: Whether to run in debug mode
            secret_key: Secret key for session management
            static_folder: Path to static files
            template_folder: Path to template files
            ssl_cert: Path to SSL certificate
            ssl_key: Path to SSL key
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.secret_key = secret_key or os.urandom(24).hex()
        self.static_folder = static_folder
        self.template_folder = template_folder
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid port: {self.port}")
            
        if self.ssl_cert and not os.path.exists(self.ssl_cert):
            raise ValueError(f"SSL certificate not found: {self.ssl_cert}")
            
        if self.ssl_key and not os.path.exists(self.ssl_key):
            raise ValueError(f"SSL key not found: {self.ssl_key}")
            
        if self.ssl_cert and not self.ssl_key:
            raise ValueError("SSL key must be provided with SSL certificate")
            
        if self.ssl_key and not self.ssl_cert:
            raise ValueError("SSL certificate must be provided with SSL key")
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            'secret_key': self.secret_key,
            'static_folder': self.static_folder,
            'template_folder': self.template_folder,
            'ssl_cert': self.ssl_cert,
            'ssl_key': self.ssl_key
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WebConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            WebConfig instance
        """
        return cls(**config_dict)
        
    def merge(self, other: Any) -> Dict[str, Any]:
        """Merge with another configuration.
        
        Args:
            other: Another configuration object or dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if hasattr(other, 'to_dict'):
            other_dict = other.to_dict()
        else:
            other_dict = other
            
        return {**self.to_dict(), **other_dict}

class MonitoringConfig:
    """Configuration class for monitoring settings."""
    
    def __init__(self, enabled: bool = True, log_level: str = 'INFO',
                 metrics_port: int = 9090, prometheus_enabled: bool = True,
                 grafana_enabled: bool = True, alerting_enabled: bool = True,
                 alert_email: str = None, alert_webhook: str = None):
        """Initialize monitoring configuration.
        
        Args:
            enabled: Whether monitoring is enabled
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            metrics_port: Port for metrics server
            prometheus_enabled: Whether Prometheus metrics are enabled
            grafana_enabled: Whether Grafana dashboard is enabled
            alerting_enabled: Whether alerting is enabled
            alert_email: Email address for alerts
            alert_webhook: Webhook URL for alerts
        """
        self.enabled = enabled
        self.log_level = log_level.upper()
        self.metrics_port = metrics_port
        self.prometheus_enabled = prometheus_enabled
        self.grafana_enabled = grafana_enabled
        self.alerting_enabled = alerting_enabled
        self.alert_email = alert_email
        self.alert_webhook = alert_webhook
        
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError(f"Invalid log level: {self.log_level}")
            
        if not isinstance(self.metrics_port, int) or not 1 <= self.metrics_port <= 65535:
            raise ValueError(f"Invalid metrics port: {self.metrics_port}")
            
        if self.alerting_enabled and not (self.alert_email or self.alert_webhook):
            raise ValueError("Alert email or webhook must be provided when alerting is enabled")
            
        if self.alert_email and not '@' in self.alert_email:
            raise ValueError(f"Invalid alert email: {self.alert_email}")
            
        if self.alert_webhook and not self.alert_webhook.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid alert webhook URL: {self.alert_webhook}")
            
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'log_level': self.log_level,
            'metrics_port': self.metrics_port,
            'prometheus_enabled': self.prometheus_enabled,
            'grafana_enabled': self.grafana_enabled,
            'alerting_enabled': self.alerting_enabled,
            'alert_email': self.alert_email,
            'alert_webhook': self.alert_webhook
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MonitoringConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            MonitoringConfig instance
        """
        return cls(**config_dict)
        
    def merge(self, other: Any) -> Dict[str, Any]:
        """Merge with another configuration.
        
        Args:
            other: Another configuration object or dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if hasattr(other, 'to_dict'):
            other_dict = other.to_dict()
        else:
            other_dict = other
            
        return {**self.to_dict(), **other_dict} 