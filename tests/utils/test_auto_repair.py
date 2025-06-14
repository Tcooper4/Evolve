"""Tests for the auto-repair system."""

import pytest
import sys
import os
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock

from trading.utils.auto_repair import AutoRepair

@pytest.fixture
def auto_repair():
    """Create an AutoRepair instance for testing."""
    return AutoRepair()

@pytest.fixture
def mock_subprocess():
    """Mock subprocess calls."""
    with patch('subprocess.check_call') as mock:
        yield mock

def test_check_packages(auto_repair):
    """Test package checking functionality."""
    # Test with all packages present
    with patch('pkg_resources.working_set') as mock_ws:
        mock_ws.by_key = {
            'numpy': MagicMock(version='1.24.3'),
            'pandas': MagicMock(version='2.0.3'),
            'torch': MagicMock(version='2.0.1')
        }
        is_healthy, issues = auto_repair.check_packages()
        assert is_healthy
        assert not issues
    
    # Test with missing package
    with patch('pkg_resources.working_set') as mock_ws:
        mock_ws.by_key = {
            'numpy': MagicMock(version='1.24.3'),
            'pandas': MagicMock(version='2.0.3')
        }
        is_healthy, issues = auto_repair.check_packages()
        assert not is_healthy
        assert 'torch' in issues

def test_check_dlls(auto_repair):
    """Test DLL checking functionality."""
    # Test with no DLL issues
    with patch('numpy.__version__', '1.24.3'):
        is_healthy, issues = auto_repair.check_dlls()
        assert is_healthy
        assert not issues
    
    # Test with numpy DLL issue
    with patch('numpy.__version__', side_effect=ImportError('_multiarray_umath')):
        is_healthy, issues = auto_repair.check_dlls()
        assert not is_healthy
        assert any('numpy DLL' in issue for issue in issues)

def test_check_transformers(auto_repair):
    """Test transformers checking functionality."""
    # Test with working transformers
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer.return_value = MagicMock()
        is_healthy, issues = auto_repair.check_transformers()
        assert is_healthy
        assert not issues
    
    # Test with transformers error
    with patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception('Model not found')):
        is_healthy, issues = auto_repair.check_transformers()
        assert not is_healthy
        assert any('Transformers issue' in issue for issue in issues)

def test_repair_packages(auto_repair, mock_subprocess):
    """Test package repair functionality."""
    # Test successful repair
    result = auto_repair.repair_packages()
    assert result
    assert mock_subprocess.call_count >= 2  # At least pip upgrade and one package install
    
    # Test failed repair
    mock_subprocess.side_effect = Exception('Installation failed')
    result = auto_repair.repair_packages()
    assert not result

def test_repair_dlls(auto_repair, mock_subprocess):
    """Test DLL repair functionality."""
    # Test successful repair
    result = auto_repair.repair_dlls()
    assert result
    assert mock_subprocess.call_count >= 2  # At least numpy and torch reinstall
    
    # Test failed repair
    mock_subprocess.side_effect = Exception('Reinstallation failed')
    result = auto_repair.repair_dlls()
    assert not result

def test_repair_transformers(auto_repair, mock_subprocess):
    """Test transformers repair functionality."""
    # Create test cache directory
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "test.txt").write_text("test")
    
    # Test successful repair
    result = auto_repair.repair_transformers()
    assert result
    assert not cache_dir.exists()  # Cache should be cleared
    assert mock_subprocess.call_count >= 2  # At least uninstall and install
    
    # Test failed repair
    mock_subprocess.side_effect = Exception('Reinstallation failed')
    result = auto_repair.repair_transformers()
    assert not result

def test_repair_environment(auto_repair, mock_subprocess):
    """Test environment repair functionality."""
    # Test successful repair
    result = auto_repair.repair_environment()
    assert result
    assert mock_subprocess.call_count >= 1  # At least venv creation
    
    # Test failed repair
    mock_subprocess.side_effect = Exception('Environment creation failed')
    result = auto_repair.repair_environment()
    assert not result

def test_run_repair(auto_repair):
    """Test complete repair process."""
    # Mock all repair methods to succeed
    with patch.multiple(auto_repair,
        repair_packages=MagicMock(return_value=True),
        repair_dlls=MagicMock(return_value=True),
        repair_transformers=MagicMock(return_value=True),
        repair_environment=MagicMock(return_value=True)
    ):
        results = auto_repair.run_repair()
        assert results['status'] == 'success'
        assert not results['issues_found']
        assert not results['issues_fixed']
    
    # Mock some repairs to fail
    with patch.multiple(auto_repair,
        repair_packages=MagicMock(return_value=False),
        repair_dlls=MagicMock(return_value=True),
        repair_transformers=MagicMock(return_value=False),
        repair_environment=MagicMock(return_value=True)
    ):
        results = auto_repair.run_repair()
        assert results['status'] == 'failed'
        assert results['issues_found']
        assert not results['issues_fixed']

def test_get_repair_status(auto_repair):
    """Test repair status reporting."""
    status = auto_repair.get_repair_status()
    assert 'status' in status
    assert 'system_info' in status
    assert 'log' in status
    assert isinstance(status['status'], dict)
    assert isinstance(status['system_info'], dict)
    assert isinstance(status['log'], list) 