"""Mock data for testing."""

import os
from datetime import datetime, timedelta

# Mock configuration
MOCK_CONFIG = {
    'email': {
        'smtp_server': 'smtp.test.com',
        'smtp_port': 587,
        'sender_email': 'test@example.com',
        'sender_password': 'test_password',
        'use_tls': True,
        'use_ssl': False
    },
    'slack': {
        'webhook_url': 'https://hooks.slack.com/services/test',
        'channel': '#test-notifications',
        'username': 'Test Bot'
    },
    'security': {
        'jwt_secret': 'test_secret',
        'jwt_algorithm': 'HS256',
        'jwt_expiry_minutes': 30
    }
}

# Mock user data
MOCK_USERS = {
    'test_user': {
        'username': 'test_user',
        'email': 'test@example.com',
        'role': 'user',
        'created_at': datetime.now().isoformat(),
        'last_login': datetime.now().isoformat()
    },
    'admin_user': {
        'username': 'admin_user',
        'email': 'admin@example.com',
        'role': 'admin',
        'created_at': datetime.now().isoformat(),
        'last_login': datetime.now().isoformat()
    }
}

# Mock API responses
MOCK_API_RESPONSES = {
    'alpha_vantage': {
        'Meta Data': {
            '1. Information': 'Daily Prices',
            '2. Symbol': 'AAPL',
            '3. Last Refreshed': datetime.now().strftime('%Y-%m-%d')
        },
        'Time Series (Daily)': {
            datetime.now().strftime('%Y-%m-%d'): {
                '1. open': '100.0',
                '2. high': '101.0',
                '3. low': '99.0',
                '4. close': '100.5',
                '5. volume': '1000000'
            }
        }
    }
}

# Mock environment variables
MOCK_ENV = {
    'ALPHA_VANTAGE_API_KEY': 'test_key',
    'EMAIL_USERNAME': 'test@example.com',
    'EMAIL_PASSWORD': 'test_password',
    'EMAIL_FROM': 'test@example.com',
    'SMTP_HOST': 'smtp.test.com',
    'SMTP_PORT': '587',
    'SLACK_WEBHOOK_URL': 'https://hooks.slack.com/services/test',
    'SLACK_DEFAULT_CHANNEL': '#test-notifications',
    'JWT_SECRET': 'test_secret',
    'ENVIRONMENT': 'test'
}

def setup_mock_env():
    """Set up mock environment variables for testing."""
    for key, value in MOCK_ENV.items():
        os.environ[key] = value

def teardown_mock_env():
    """Clean up mock environment variables after testing."""
    for key in MOCK_ENV.keys():
        if key in os.environ:
            del os.environ[key] 