# -*- coding: utf-8 -*-
"""Environment variable management with secure loading and validation."""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseSettings, SecretStr

class EnvironmentSettings(BaseSettings):
    """Environment settings with validation."""
    
    # API Keys
    POLYGON_KEY: SecretStr
    OPENAI_API_KEY: SecretStr
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: Optional[SecretStr] = None
    
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "evolve_db"
    DB_USER: str
    DB_PASSWORD: SecretStr
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    
    # Agent Configuration
    DEFAULT_AGENT: str = "code_review"
    MAX_CONCURRENT_TASKS: int = 10
    TASK_TIMEOUT: int = 300
    
    # Security Configuration
    JWT_SECRET: SecretStr
    ENCRYPTION_KEY: SecretStr
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    ALERT_EMAIL: str
    
    # Development Configuration
    DEBUG: bool = False
    TEST_MODE: bool = False
    MOCK_AGENTS: bool = False
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

class EnvironmentManager:
    """Manages environment variables securely."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize environment manager.
        
        Args:
            env_file: Optional path to .env file
        """
        self.logger = logging.getLogger("EnvironmentManager")
        self.env_file = env_file
        self.settings = None
        self._load_environment()def _load_environment(self):
        """Load environment variables from file and system."""
        try:
            # Load from .env file if specified
            if self.env_file:
                load_dotenv(self.env_file)
            
            # Validate and load settings
            self.settings = EnvironmentSettings()
            self.logger.info("Environment loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading environment: {str(e)}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        if not self.settings:
            raise RuntimeError("Environment not loaded")
            
        value = getattr(self.settings, key, default)
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all environment variables.
        
        Returns:
            Dict of environment variables
        """
        if not self.settings:
            raise RuntimeError("Environment not loaded")
            
        return {
            key: value.get_secret_value() if isinstance(value, SecretStr) else value
            for key, value in self.settings.dict().items()
        }
    
    def validate(self) -> bool:
        """Validate environment configuration.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            if not self.settings:
                self._load_environment()
            return True
        except Exception as e:
            self.logger.error(f"Environment validation failed: {str(e)}")
            return False
    
    def create_template(self, output_path: str = ".env.template"):
        """Create template .env file.
        
        Args:
            output_path: Path to output template file
        """
        template = []
        for field in EnvironmentSettings.__fields__:
            field_info = EnvironmentSettings.__fields__[field]
            default = field_info.default
            if isinstance(default, SecretStr):
                default = "your_secret_here"
            template.append(f"{field}={default}")
        
        with open(output_path, "w") as f:
            f.write("# Environment Template\n")
            f.write("# Replace placeholder values with your actual secrets\n\n")
            f.write("\n".join(template))
        
        self.logger.info(f"Template created at {output_path}")

    def rotate_secret(self, key: str, new_value: str):
        """Rotate a secret value.
        
        Args:
            key: Secret key to rotate
            new_value: New secret value
        """
        if not self.settings:
            raise RuntimeError("Environment not loaded")
            
        if not hasattr(self.settings, key):
            raise ValueError(f"Unknown secret key: {key}")
            
        # Update in memory
        setattr(self.settings, key, SecretStr(new_value))
        
        # Update in .env file if it exists
        if self.env_file and Path(self.env_file).exists():
            with open(self.env_file, "r") as f:
                lines = f.readlines()
            
            with open(self.env_file, "w") as f:
                for line in lines:
                    if line.startswith(f"{key}="):
                        f.write(f"{key}={new_value}\n")
                    else:
                        f.write(line)
        
        self.logger.info(f"Rotated secret: {key}")
