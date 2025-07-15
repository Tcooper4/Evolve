"""
Unit tests for environment variable validation.
"""

import pytest
from unittest.mock import patch, MagicMock

from trading.utils.env_manager import EnvironmentManager, EnvironmentSettings


class TestEnvironmentValidation:
    """Test environment variable validation."""

    def test_missing_openai_api_key(self, monkeypatch):
        """Test that EnvironmentError is raised when OPENAI_API_KEY is missing."""
        # Remove OPENAI_API_KEY from environment
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY is not set in the environment"):
            EnvironmentManager()

    def test_missing_openai_api_key_empty_string(self, monkeypatch):
        """Test that EnvironmentError is raised when OPENAI_API_KEY is empty string."""
        monkeypatch.setenv('OPENAI_API_KEY', '')
        
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY is not set in the environment"):
            EnvironmentManager()

    def test_valid_environment_variables(self, monkeypatch):
        """Test that EnvironmentManager initializes with valid environment variables."""
        # Set required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        
        env_manager = EnvironmentManager()
        assert env_manager.settings is not None
        assert env_manager.get('OPENAI_API_KEY') == 'test_openai_key'

    def test_missing_redis_url_uses_default(self, monkeypatch):
        """Test that REDIS_URL uses default value when not set."""
        # Set required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        
        # Remove REDIS_URL to test default
        monkeypatch.delenv('REDIS_URL', raising=False)
        
        env_manager = EnvironmentManager()
        assert env_manager.get('REDIS_URL') == 'redis://localhost:6379'

    def test_custom_redis_url(self, monkeypatch):
        """Test that custom REDIS_URL is used when set."""
        # Set required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        monkeypatch.setenv('REDIS_URL', 'redis://custom:6379')
        
        env_manager = EnvironmentManager()
        assert env_manager.get('REDIS_URL') == 'redis://custom:6379'

    def test_environment_settings_validation(self, monkeypatch):
        """Test EnvironmentSettings validation."""
        # Set required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        
        settings = EnvironmentSettings()
        assert settings.OPENAI_API_KEY.get_secret_value() == 'test_openai_key'
        assert settings.POLYGON_API_KEY.get_secret_value() == 'test_polygon_key'
        assert settings.DB_USER == 'test_user'
        assert settings.DB_PASSWORD.get_secret_value() == 'test_password'

    def test_optional_environment_variables(self, monkeypatch):
        """Test that optional environment variables have correct defaults."""
        # Set only required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        
        settings = EnvironmentSettings()
        assert settings.DB_HOST == 'localhost'
        assert settings.DB_PORT == 5432
        assert settings.DB_NAME == 'evolve_db'
        assert settings.LOG_LEVEL == 'INFO'
        assert settings.DEFAULT_AGENT == 'code_review'
        assert settings.MAX_CONCURRENT_TASKS == 10
        assert settings.TASK_TIMEOUT == 300
        assert settings.ENABLE_METRICS is True
        assert settings.METRICS_PORT == 9090
        assert settings.DEBUG is False
        assert settings.TEST_MODE is False
        assert settings.MOCK_AGENTS is False

    def test_boolean_environment_variables(self, monkeypatch):
        """Test boolean environment variable parsing."""
        # Set required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        
        # Test boolean values
        monkeypatch.setenv('DEBUG', 'true')
        monkeypatch.setenv('TEST_MODE', 'false')
        monkeypatch.setenv('ENABLE_METRICS', 'false')
        
        settings = EnvironmentSettings()
        assert settings.DEBUG is True
        assert settings.TEST_MODE is False
        assert settings.ENABLE_METRICS is False

    def test_integer_environment_variables(self, monkeypatch):
        """Test integer environment variable parsing."""
        # Set required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        
        # Test integer values
        monkeypatch.setenv('DB_PORT', '5433')
        monkeypatch.setenv('MAX_CONCURRENT_TASKS', '20')
        monkeypatch.setenv('TASK_TIMEOUT', '600')
        
        settings = EnvironmentSettings()
        assert settings.DB_PORT == 5433
        assert settings.MAX_CONCURRENT_TASKS == 20
        assert settings.TASK_TIMEOUT == 600

    @patch('trading.utils.env_manager.load_dotenv')
    def test_env_file_loading(self, mock_load_dotenv, monkeypatch):
        """Test that .env file is loaded when specified."""
        # Set required environment variables
        monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_key')
        monkeypatch.setenv('POLYGON_API_KEY', 'test_polygon_key')
        monkeypatch.setenv('DB_USER', 'test_user')
        monkeypatch.setenv('DB_PASSWORD', 'test_password')
        monkeypatch.setenv('JWT_SECRET', 'test_jwt_secret')
        monkeypatch.setenv('ENCRYPTION_KEY', 'test_encryption_key')
        monkeypatch.setenv('ALERT_EMAIL', 'test@example.com')
        
        env_manager = EnvironmentManager(env_file='test.env')
        mock_load_dotenv.assert_called_with('test.env') 