"""Smoke tests for the main application."""

import pytest
import sys
from pathlib import Path

def test_app_imports():
    """Test that all required modules can be imported."""
    try:
        import app
        assert app is not None
    except ImportError as e:
        pytest.fail(f"Failed to import app: {e}")

def test_app_initialization():
    """Test that the app can be initialized without crashing."""
    try:
        from app import create_app
        app = create_app()
        assert app is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize app: {e}")

def test_app_configuration():
    """Test that the app configuration is valid."""
    try:
        from app import create_app
        app = create_app()
        assert app.config is not None
        assert 'SECRET_KEY' in app.config
        assert 'DEBUG' in app.config
    except Exception as e:
        pytest.fail(f"Failed to configure app: {e}")

def test_app_routes():
    """Test that basic app routes are registered."""
    try:
        from app import create_app
        app = create_app()
        assert app.url_map is not None
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        assert '/' in routes
        assert '/api/' in routes
    except Exception as e:
        pytest.fail(f"Failed to register app routes: {e}")

def test_app_services():
    """Test that required services are initialized."""
    try:
        from app import create_app
        app = create_app()
        assert hasattr(app, 'trading_engine')
        assert hasattr(app, 'forecast_engine')
        assert hasattr(app, 'agent_engine')
    except Exception as e:
        pytest.fail(f"Failed to initialize app services: {e}") 