#!/usr/bin/env python3
"""
Institutional-Grade Trading System Launcher

Comprehensive launcher for the institutional-grade trading system.
Provides unified interface to all strategic intelligence modules.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess
import signal
import threading
from typing import Dict, Any, Optional

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent / "trading"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/institutional_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class InstitutionalSystemLauncher:
    """Launcher for the institutional-grade trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the launcher.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/institutional_system.json"
        self.config = self._load_config()
        self.processes = {}
        self.running = False
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("Institutional System Launcher initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load launcher configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default launcher configuration."""
        return {
            'launcher': {
                'name': 'Institutional System Launcher',
                'version': '2.0.0',
                'auto_restart': True,
                'restart_delay': 30,
                'max_restarts': 3
            },
            'services': {
                'dashboard': {
                    'enabled': True,
                    'command': ['streamlit', 'run', 'trading/ui/institutional_dashboard.py'],
                    'port': 8501,
                    'host': 'localhost'
                },
                'api_server': {
                    'enabled': False,
                    'command': ['python', 'trading/api/server.py'],
                    'port': 8000,
                    'host': 'localhost'
                },
                'websocket_server': {
                    'enabled': True,
                    'command': ['python', 'trading/services/websocket_server.py'],
                    'port': 8765,
                    'host': 'localhost'
                }
            },
            'monitoring': {
                'enabled': True,
                'health_check_interval': 60,
                'log_rotation': True,
                'max_log_size': '100MB'
            }
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            'logs',
            'config',
            'reports',
            'charts',
            'cache',
            'data',
            'models'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def start(self):
        """Start the institutional system."""
        try:
            logger.info("Starting Institutional-Grade Trading System...")
            
            # Check system requirements
            self._check_requirements()
            
            # Start services
            self._start_services()
            
            # Start monitoring
            if self.config['monitoring']['enabled']:
                self._start_monitoring()
            
            self.running = True
            
            logger.info("Institutional-Grade Trading System started successfully")
            
            # Keep launcher running
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.stop()
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.stop()
    
    def stop(self):
        """Stop the institutional system."""
        try:
            logger.info("Stopping Institutional-Grade Trading System...")
            
            self.running = False
            
            # Stop all processes
            for service_name, process in self.processes.items():
                if process and process.poll() is None:
                    logger.info(f"Stopping {service_name}...")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {service_name}")
                        process.kill()
            
            self.processes.clear()
            
            logger.info("Institutional-Grade Trading System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def _check_requirements(self):
        """Check system requirements."""
        try:
            logger.info("Checking system requirements...")
            
            # Check Python version
            if sys.version_info < (3, 8):
                raise RuntimeError("Python 3.8 or higher is required")
            
            # Check required packages
            required_packages = [
                'streamlit', 'pandas', 'numpy', 'plotly',
                'yfinance', 'scikit-learn', 'scipy'
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                raise RuntimeError(f"Missing required packages: {', '.join(missing_packages)}")
            
            # Check environment variables
            required_env_vars = ['FRED_API_KEY', 'ALPHA_VANTAGE_API_KEY']
            missing_env_vars = []
            
            for var in required_env_vars:
                if not os.getenv(var):
                    missing_env_vars.append(var)
            
            if missing_env_vars:
                logger.warning(f"Missing environment variables: {', '.join(missing_env_vars)}")
                logger.warning("Some features may not work properly")
            
            logger.info("System requirements check completed")
            
        except Exception as e:
            logger.error(f"Requirements check failed: {e}")
            raise
    
    def _start_services(self):
        """Start all configured services."""
        try:
            services = self.config['services']
            
            for service_name, service_config in services.items():
                if service_config.get('enabled', False):
                    self._start_service(service_name, service_config)
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            raise
    
    def _start_service(self, service_name: str, service_config: Dict[str, Any]):
        """Start a specific service."""
        try:
            command = service_config['command']
            
            logger.info(f"Starting {service_name} with command: {' '.join(command)}")
            
            # Start process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = process
            
            # Wait a moment to see if it starts successfully
            time.sleep(2)
            
            if process.poll() is None:
                logger.info(f"{service_name} started successfully (PID: {process.pid})")
            else:
                stdout, stderr = process.communicate()
                logger.error(f"{service_name} failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                raise RuntimeError(f"Failed to start {service_name}")
            
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            raise
    
    def _start_monitoring(self):
        """Start system monitoring."""
        try:
            monitoring_thread = threading.Thread(target=self._monitoring_loop)
            monitoring_thread.daemon = True
            monitoring_thread.start()
            
            logger.info("System monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check process health
                self._check_process_health()
                
                # Check system resources
                self._check_system_resources()
                
                # Sleep for monitoring interval
                interval = self.config['monitoring']['health_check_interval']
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _check_process_health(self):
        """Check health of running processes."""
        try:
            for service_name, process in self.processes.items():
                if process and process.poll() is not None:
                    logger.warning(f"{service_name} process terminated unexpectedly")
                    
                    # Restart if auto-restart is enabled
                    if self.config['launcher']['auto_restart']:
                        self._restart_service(service_name)
            
        except Exception as e:
            logger.error(f"Error checking process health: {e}")
    
    def _restart_service(self, service_name: str):
        """Restart a failed service."""
        try:
            logger.info(f"Restarting {service_name}...")
            
            # Get service config
            service_config = self.config['services'].get(service_name, {})
            
            if service_config.get('enabled', False):
                # Stop old process
                old_process = self.processes.get(service_name)
                if old_process:
                    old_process.terminate()
                    try:
                        old_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        old_process.kill()
                
                # Wait before restart
                restart_delay = self.config['launcher']['restart_delay']
                time.sleep(restart_delay)
                
                # Start new process
                self._start_service(service_name, service_config)
            
        except Exception as e:
            logger.error(f"Error restarting {service_name}: {e}")
    
    def _check_system_resources(self):
        """Check system resource usage."""
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent:.1f}%")
            
        except ImportError:
            # psutil not available, skip resource monitoring
            pass
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        try:
            status = {
                'launcher': {
                    'running': self.running,
                    'config': self.config['launcher']
                },
                'services': {}
            }
            
            for service_name, process in self.processes.items():
                if process:
                    service_status = {
                        'pid': process.pid,
                        'running': process.poll() is None,
                        'return_code': process.returncode if process.poll() is not None else None
                    }
                    status['services'][service_name] = service_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}
    
    def show_status(self):
        """Display system status."""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("🏛️  INSTITUTIONAL-GRADE TRADING SYSTEM STATUS")
        print("="*60)
        
        # Launcher status
        launcher_status = status.get('launcher', {})
        print(f"Launcher Status: {'🟢 Running' if launcher_status.get('running') else '🔴 Stopped'}")
        print(f"Version: {launcher_status.get('config', {}).get('version', 'Unknown')}")
        
        # Services status
        print("\nServices:")
        services = status.get('services', {})
        for service_name, service_status in services.items():
            if service_status.get('running'):
                print(f"  🟢 {service_name} (PID: {service_status.get('pid', 'Unknown')})")
            else:
                print(f"  🔴 {service_name} (Stopped)")
        
        print("\n" + "="*60)
    
    def show_help(self):
        """Display help information."""
        help_text = """
🏛️  INSTITUTIONAL-GRADE TRADING SYSTEM

USAGE:
    python launch_institutional_system.py [OPTIONS]

OPTIONS:
    --config PATH       Path to configuration file
    --start            Start the system
    --stop             Stop the system
    --status           Show system status
    --restart          Restart the system
    --help             Show this help message

EXAMPLES:
    # Start the system
    python launch_institutional_system.py --start
    
    # Show status
    python launch_institutional_system.py --status
    
    # Restart the system
    python launch_institutional_system.py --restart

SERVICES:
    - Dashboard (Streamlit UI)
    - WebSocket Server (Real-time data)
    - API Server (REST API)

CONFIGURATION:
    Edit config/institutional_system.json to customize settings.

ENVIRONMENT VARIABLES:
    FRED_API_KEY           Federal Reserve Economic Data API key
    ALPHA_VANTAGE_API_KEY  Alpha Vantage API key
    OPENAI_API_KEY         OpenAI API key (for LLM features)

LOGS:
    Check logs/institutional_system.log for detailed logs.
        """
        print(help_text)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Institutional-Grade Trading System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--start',
        action='store_true',
        help='Start the system'
    )
    
    parser.add_argument(
        '--stop',
        action='store_true',
        help='Stop the system'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status'
    )
    
    parser.add_argument(
        '--restart',
        action='store_true',
        help='Restart the system'
    )
    
    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help information'
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = InstitutionalSystemLauncher(config_path=args.config)
    
    try:
        if args.help:
            launcher.show_help()
        
        elif args.status:
            launcher.show_status()
        
        elif args.stop:
            launcher.stop()
            print("System stopped")
        
        elif args.restart:
            launcher.stop()
            time.sleep(2)
            launcher.start()
        
        elif args.start:
            launcher.start()
        
        else:
            # Default: start the system
            launcher.start()
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
        launcher.stop()
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 