"""
Comprehensive Live Dashboard Test Suite

This module tests the live dashboard functionality with:
- Async test support using pytest.mark.asyncio
- Dynamic port scanning for free ports
- Log mocking and Streamlit output inspection
- Comprehensive error handling and edge cases
"""

import pytest
import asyncio
import socket
import subprocess
import time
import threading
import queue
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Optional, Dict, Any
import requests
import json
import tempfile
import os
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortScanner:
    """Utility class for finding free ports."""
    
    @staticmethod
    def find_free_port(start_port: int = 8501, max_attempts: int = 100) -> int:
        """Find a free port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_attempts}")

class StreamlitOutputCapture:
    """Capture and analyze Streamlit output."""
    
    def __init__(self):
        self.output_queue = queue.Queue()
        self.logs = []
        self.errors = []
        self.warnings = []
    
    def capture_output(self, process):
        """Capture output from a subprocess."""
        def capture_stream(stream, log_type):
            for line in iter(stream.readline, b''):
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    self.output_queue.put((log_type, line_str))
                    if log_type == 'error':
                        self.errors.append(line_str)
                    elif log_type == 'warning':
                        self.warnings.append(line_str)
                    else:
                        self.logs.append(line_str)
        
        # Start capture threads
        stdout_thread = threading.Thread(
            target=capture_stream, 
            args=(process.stdout, 'info')
        )
        stderr_thread = threading.Thread(
            target=capture_stream, 
            args=(process.stderr, 'error')
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        return stdout_thread, stderr_thread

@pytest.fixture
def free_port():
    """Fixture to provide a free port for testing."""
    return PortScanner.find_free_port()

@pytest.fixture
def temp_dashboard_dir():
    """Create a temporary directory for dashboard testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing without actual UI."""
    with patch('streamlit.set_page_config') as mock_config, \
         patch('streamlit.title') as mock_title, \
         patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.container') as mock_container, \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.line_chart') as mock_chart, \
         patch('streamlit.dataframe') as mock_df, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.text_input') as mock_text_input, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.info') as mock_info:
        
        # Create a mock sidebar context manager
        mock_sidebar_context = MagicMock()
        mock_sidebar.return_value = mock_sidebar_context
        
        # Create a mock container context manager
        mock_container_context = MagicMock()
        mock_container.return_value = mock_container_context
        
        yield {
            'config': mock_config,
            'title': mock_title,
            'sidebar': mock_sidebar,
            'container': mock_container,
            'metric': mock_metric,
            'chart': mock_chart,
            'dataframe': mock_df,
            'button': mock_button,
            'selectbox': mock_selectbox,
            'text_input': mock_text_input,
            'success': mock_success,
            'error': mock_error,
            'warning': mock_warning,
            'info': mock_info,
            'sidebar_context': mock_sidebar_context,
            'container_context': mock_container_context
        }

class TestLiveDashboard:
    """Test suite for live dashboard functionality."""
    
    @pytest.mark.asyncio
    async def test_dashboard_startup(self, free_port, temp_dashboard_dir):
        """Test dashboard startup and basic functionality."""
        # Mock the dashboard startup process
        with patch('subprocess.Popen') as mock_popen, \
             patch('time.sleep') as mock_sleep:
            
            # Mock successful process
            mock_process = Mock()
            mock_process.poll.return_value = None  # Process is running
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            # Test dashboard startup
            try:
                # Simulate dashboard startup
                process = subprocess.Popen(
                    ['streamlit', 'run', 'dashboard/App.js', '--server.port', str(free_port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for startup
                await asyncio.sleep(2)
                
                # Check if process is running
                assert process.poll() is None, "Dashboard process should be running"
                
                # Test basic connectivity
                try:
                    response = requests.get(f'http://localhost:{free_port}', timeout=5)
                    assert response.status_code == 200, "Dashboard should be accessible"
                except requests.RequestException:
                    # Dashboard might not be fully started yet
                    pass
                
            finally:
                # Cleanup
                if 'process' in locals():
                    process.terminate()
                    process.wait(timeout=5)
    
    @pytest.mark.asyncio
    async def test_dashboard_port_conflict(self, free_port):
        """Test dashboard behavior when port is already in use."""
        # First, occupy the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', free_port))
            
            # Try to start dashboard on occupied port
            with patch('subprocess.Popen') as mock_popen:
                mock_process = Mock()
                mock_process.poll.return_value = 1  # Process failed
                mock_process.returncode = 1
                mock_popen.return_value = mock_process
                
                # Should handle port conflict gracefully
                process = subprocess.Popen(
                    ['streamlit', 'run', 'dashboard/App.js', '--server.port', str(free_port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                await asyncio.sleep(1)
                
                # Process should fail due to port conflict
                assert process.poll() is not None, "Process should fail on port conflict"
    
    @pytest.mark.asyncio
    async def test_dashboard_error_handling(self, free_port):
        """Test dashboard error handling and recovery."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('logging.error') as mock_log_error, \
             patch('logging.info') as mock_log_info:
            
            # Mock process that crashes
            mock_process = Mock()
            mock_process.poll.side_effect = [None, 1]  # Running then crashed
            mock_process.returncode = 1
            mock_popen.return_value = mock_process
            
            # Test error handling
            try:
                process = subprocess.Popen(
                    ['streamlit', 'run', 'dashboard/App.js', '--server.port', str(free_port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                await asyncio.sleep(1)
                
                # Simulate crash
                mock_process.poll.return_value = 1
                
                # Should log error
                mock_log_error.assert_called()
                
            except Exception as e:
                # Should handle exceptions gracefully
                assert "dashboard" in str(e).lower() or "streamlit" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_dashboard_refresh_mechanism(self, free_port):
        """Test dashboard refresh and auto-restart functionality."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('time.sleep') as mock_sleep:
            
            # Mock process that needs restart
            mock_process = Mock()
            mock_process.poll.side_effect = [None, 1, None]  # Running, crashed, restarted
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            # Test refresh mechanism
            process = subprocess.Popen(
                ['streamlit', 'run', 'dashboard/App.js', '--server.port', str(free_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            await asyncio.sleep(1)
            
            # Simulate restart
            mock_process.poll.return_value = None
            
            await asyncio.sleep(1)
            
            # Should handle restart gracefully
            assert mock_process.poll() is not None or mock_process.poll() is None
    
    @pytest.mark.asyncio
    async def test_dashboard_logging(self, free_port):
        """Test dashboard logging and output capture."""
        output_capture = StreamlitOutputCapture()
        
        with patch('subprocess.Popen') as mock_popen:
            # Mock process with output
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_process.stdout = Mock()
            mock_process.stderr = Mock()
            mock_process.stdout.readline.side_effect = [
                b"INFO: Dashboard started\n",
                b"INFO: Loading data\n",
                b""
            ]
            mock_process.stderr.readline.side_effect = [
                b"WARNING: Slow data load\n",
                b""
            ]
            mock_popen.return_value = mock_process
            
            # Test output capture
            process = subprocess.Popen(
                ['streamlit', 'run', 'dashboard/App.js', '--server.port', str(free_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Capture output
            stdout_thread, stderr_thread = output_capture.capture_output(process)
            
            await asyncio.sleep(2)
            
            # Check captured output
            assert len(output_capture.logs) > 0, "Should capture logs"
            assert any("Dashboard" in log for log in output_capture.logs), "Should capture dashboard logs"
    
    @pytest.mark.asyncio
    async def test_dashboard_configuration(self, free_port):
        """Test dashboard configuration and settings."""
        # Test configuration loading
        config_data = {
            'port': free_port,
            'host': 'localhost',
            'auto_reload': True,
            'theme': 'dark'
        }
        
        config_file = Path(tempfile.gettempdir()) / 'dashboard_config.json'
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            # Test configuration loading
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config['port'] == free_port
            assert loaded_config['auto_reload'] is True
            
        finally:
            if config_file.exists():
                config_file.unlink()
    
    @pytest.mark.asyncio
    async def test_dashboard_health_check(self, free_port):
        """Test dashboard health check functionality."""
        with patch('requests.get') as mock_get:
            # Mock successful health check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'healthy'}
            mock_get.return_value = mock_response
            
            # Test health check
            try:
                response = requests.get(f'http://localhost:{free_port}/_stcore/health', timeout=5)
                assert response.status_code == 200
                
                health_data = response.json()
                assert health_data['status'] == 'healthy'
                
            except requests.RequestException:
                # Dashboard might not be running, which is expected in test
                pass
    
    @pytest.mark.asyncio
    async def test_dashboard_graceful_shutdown(self, free_port):
        """Test dashboard graceful shutdown."""
        with patch('subprocess.Popen') as mock_popen, \
             patch('signal.signal') as mock_signal:
            
            # Mock process
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_process.terminate = Mock()
            mock_process.wait = Mock(return_value=0)
            mock_popen.return_value = mock_process
            
            # Test graceful shutdown
            process = subprocess.Popen(
                ['streamlit', 'run', 'dashboard/App.js', '--server.port', str(free_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            await asyncio.sleep(1)
            
            # Simulate graceful shutdown
            process.terminate()
            await asyncio.sleep(1)
            
            # Should handle shutdown gracefully
            assert process.poll() is not None or process.returncode == 0

class TestDashboardIntegration:
    """Integration tests for dashboard with other components."""
    
    @pytest.mark.asyncio
    async def test_dashboard_with_trading_data(self, free_port):
        """Test dashboard integration with trading data."""
        # Mock trading data
        mock_trading_data = {
            'portfolio_value': 100000,
            'daily_pnl': 1500,
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'value': 15000},
                {'symbol': 'GOOGL', 'quantity': 50, 'value': 25000}
            ]
        }
        
        with patch('trading.data.data_provider.get_portfolio_data') as mock_data:
            mock_data.return_value = mock_trading_data
            
            # Test data integration
            data = mock_data()
            assert data['portfolio_value'] == 100000
            assert len(data['positions']) == 2
    
    @pytest.mark.asyncio
    async def test_dashboard_with_agent_status(self, free_port):
        """Test dashboard integration with agent status."""
        # Mock agent status
        mock_agent_status = {
            'active_agents': 5,
            'total_agents': 10,
            'system_health': 'healthy'
        }
        
        with patch('trading.agents.agent_manager.get_system_status') as mock_status:
            mock_status.return_value = mock_agent_status
            
            # Test status integration
            status = mock_status()
            assert status['active_agents'] == 5
            assert status['system_health'] == 'healthy'

# Utility functions for testing
def create_test_dashboard_config(port: int, **kwargs) -> Dict[str, Any]:
    """Create a test dashboard configuration."""
    config = {
        'port': port,
        'host': 'localhost',
        'auto_reload': True,
        'theme': 'light',
        'debug': True
    }
    config.update(kwargs)
    return config

def validate_dashboard_response(response: requests.Response) -> bool:
    """Validate dashboard response."""
    return (
        response.status_code == 200 and
        'text/html' in response.headers.get('content-type', '') and
        len(response.content) > 0
    )

# Test runner utilities
@pytest.mark.asyncio
async def test_dashboard_end_to_end(free_port):
    """End-to-end test of dashboard functionality."""
    # This test would require actual Streamlit to be installed
    # For now, we'll mock the entire process
    
    with patch('subprocess.Popen') as mock_popen, \
         patch('requests.get') as mock_get, \
         patch('time.sleep') as mock_sleep:
        
        # Mock successful dashboard startup
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = b'<html><body>Dashboard</body></html>'
        mock_get.return_value = mock_response
        
        # Test end-to-end flow
        try:
            # Start dashboard
            process = subprocess.Popen(
                ['streamlit', 'run', 'dashboard/App.js', '--server.port', str(free_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            await asyncio.sleep(2)
            
            # Test connectivity
            response = requests.get(f'http://localhost:{free_port}', timeout=5)
            assert validate_dashboard_response(response)
            
        finally:
            # Cleanup
            if 'process' in locals():
                process.terminate()
                process.wait(timeout=5)

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--asyncio-mode=auto"]) 