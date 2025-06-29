#!/usr/bin/env python3
"""
Institutional-Grade Trading System Launcher

This script launches the complete institutional trading system with all agents,
strategies, and monitoring services. It provides health monitoring, auto-restart,
and comprehensive logging.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import psutil
import requests
from dataclasses import dataclass, asdict
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceStatus:
    """Service status information."""
    name: str
    pid: Optional[int]
    status: str  # running, stopped, error, starting
    start_time: Optional[datetime]
    last_heartbeat: Optional[datetime]
    restart_count: int
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'pid': self.pid,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'restart_count': self.restart_count,
            'error_message': self.error_message
        }

class InstitutionalSystemLauncher:
    """Main launcher for the institutional trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the launcher."""
        self.config = self._load_config(config_path)
        self.services: Dict[str, ServiceStatus] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        self.monitoring_thread = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Institutional System Launcher initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "system_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'services': {
                'dashboard': {
                    'command': ['streamlit', 'run', 'unified_interface.py'],
                    'port': 8501,
                    'auto_restart': True,
                    'max_restarts': 5,
                    'health_check_interval': 30
                },
                'api_server': {
                    'command': ['python', '-m', 'uvicorn', 'trading.api.main:app', '--host', '0.0.0.0', '--port', '8000'],
                    'port': 8000,
                    'auto_restart': True,
                    'max_restarts': 5,
                    'health_check_interval': 30
                },
                'websocket_server': {
                    'command': ['python', 'trading/services/websocket_server.py'],
                    'port': 8001,
                    'auto_restart': True,
                    'max_restarts': 5,
                    'health_check_interval': 30
                },
                'signal_center': {
                    'command': ['python', 'trading/services/real_time_signal_center.py'],
                    'auto_restart': True,
                    'max_restarts': 5,
                    'health_check_interval': 60
                },
                'model_monitor': {
                    'command': ['python', 'trading/memory/model_monitor.py'],
                    'auto_restart': True,
                    'max_restarts': 3,
                    'health_check_interval': 300
                }
            },
            'monitoring': {
                'enabled': True,
                'check_interval': 30,
                'max_memory_usage': 0.8,
                'max_cpu_usage': 0.9
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/system.log'
            }
        }
    
    def start_all_services(self) -> bool:
        """Start all configured services."""
        try:
            logger.info("Starting all institutional trading system services...")
            
            services_config = self.config.get('services', {})
            
            for service_name, service_config in services_config.items():
                self._start_service(service_name, service_config)
                time.sleep(2)  # Brief delay between service starts
            
            # Start monitoring thread
            if self.config.get('monitoring', {}).get('enabled', True):
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitoring_thread.start()
                logger.info("Monitoring thread started")
            
            self.running = True
            logger.info("All services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            return False
    
    def _start_service(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Start a single service."""
        try:
            command = service_config.get('command', [])
            if not command:
                logger.warning(f"No command specified for service {service_name}")
                return False
            
            # Check if service is already running
            if service_name in self.processes and self.processes[service_name].poll() is None:
                logger.info(f"Service {service_name} is already running")
                return True
            
            # Start the service
            logger.info(f"Starting service {service_name} with command: {' '.join(command)}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Update service status
            self.services[service_name] = ServiceStatus(
                name=service_name,
                pid=process.pid,
                status="starting",
                start_time=datetime.now(),
                last_heartbeat=datetime.now(),
                restart_count=0,
                error_message=None
            )
            
            self.processes[service_name] = process
            
            # Wait a moment and check if process started successfully
            time.sleep(2)
            if process.poll() is None:
                self.services[service_name].status = "running"
                logger.info(f"Service {service_name} started successfully (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                error_msg = stderr if stderr else "Unknown error"
                self.services[service_name].status = "error"
                self.services[service_name].error_message = error_msg
                logger.error(f"Service {service_name} failed to start: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting service {service_name}: {e}")
            self.services[service_name] = ServiceStatus(
                name=service_name,
                pid=None,
                status="error",
                start_time=None,
                last_heartbeat=None,
                restart_count=0,
                error_message=str(e)
            )
            return False
    
    def stop_all_services(self) -> None:
        """Stop all running services."""
        logger.info("Stopping all services...")
        self.running = False
        
        for service_name, process in self.processes.items():
            try:
                if process.poll() is None:
                    logger.info(f"Stopping service {service_name} (PID: {process.pid})")
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Service {service_name} did not terminate gracefully, forcing kill")
                        process.kill()
                        process.wait()
                    
                    self.services[service_name].status = "stopped"
                    logger.info(f"Service {service_name} stopped")
                    
            except Exception as e:
                logger.error(f"Error stopping service {service_name}: {e}")
        
        self.processes.clear()
        logger.info("All services stopped")
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        try:
            logger.info(f"Restarting service {service_name}")
            
            # Stop the service if running
            if service_name in self.processes:
                process = self.processes[service_name]
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            
            # Start the service again
            services_config = self.config.get('services', {})
            if service_name in services_config:
                service_config = services_config[service_name]
                success = self._start_service(service_name, service_config)
                
                if success:
                    self.services[service_name].restart_count += 1
                    logger.info(f"Service {service_name} restarted successfully")
                else:
                    logger.error(f"Failed to restart service {service_name}")
                
                return success
            else:
                logger.error(f"Service {service_name} not found in configuration")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        check_interval = self.config.get('monitoring', {}).get('check_interval', 30)
        
        while self.running:
            try:
                self._check_service_health()
                self._check_system_resources()
                self._update_service_status()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _check_service_health(self) -> None:
        """Check health of all services."""
        for service_name, service_status in self.services.items():
            try:
                if service_name not in self.processes:
                    continue
                
                process = self.processes[service_name]
                
                # Check if process is still running
                if process.poll() is not None:
                    logger.warning(f"Service {service_name} has stopped")
                    service_status.status = "stopped"
                    
                    # Auto-restart if enabled
                    services_config = self.config.get('services', {})
                    if service_name in services_config:
                        service_config = services_config[service_name]
                        if service_config.get('auto_restart', False):
                            max_restarts = service_config.get('max_restarts', 5)
                            if service_status.restart_count < max_restarts:
                                logger.info(f"Auto-restarting service {service_name}")
                                self.restart_service(service_name)
                            else:
                                logger.error(f"Service {service_name} exceeded max restart attempts")
                                service_status.status = "error"
                                service_status.error_message = "Max restart attempts exceeded"
                
                # Update heartbeat
                service_status.last_heartbeat = datetime.now()
                
            except Exception as e:
                logger.error(f"Error checking health of service {service_name}: {e}")
    
    def _check_system_resources(self) -> None:
        """Check system resource usage."""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100
            
            max_memory = self.config.get('monitoring', {}).get('max_memory_usage', 0.8)
            max_cpu = self.config.get('monitoring', {}).get('max_cpu_usage', 0.9)
            
            if memory_usage > max_memory:
                logger.warning(f"High memory usage: {memory_usage:.1%}")
            
            if cpu_usage > max_cpu:
                logger.warning(f"High CPU usage: {cpu_usage:.1%}")
                
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
    
    def _update_service_status(self) -> None:
        """Update service status information."""
        for service_name, service_status in self.services.items():
            try:
                if service_name in self.processes:
                    process = self.processes[service_name]
                    if process.poll() is None:
                        service_status.status = "running"
                    else:
                        service_status.status = "stopped"
                        
            except Exception as e:
                logger.error(f"Error updating status for service {service_name}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'services': {name: status.to_dict() for name, status in self.services.items()},
                'system_resources': {
                    'memory_usage': psutil.virtual_memory().percent,
                    'cpu_usage': psutil.cpu_percent(),
                    'disk_usage': psutil.disk_usage('/').percent
                },
                'uptime': self._get_uptime()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'error': str(e)
            }
    
    def _get_uptime(self) -> str:
        """Get system uptime."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime = timedelta(seconds=uptime_seconds)
            return str(uptime)
        except Exception:
            return "Unknown"
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all_services()
        sys.exit(0)
    
    def wait_for_services(self, timeout: int = 60) -> bool:
        """Wait for all services to be ready."""
        logger.info(f"Waiting for services to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ready = True
            
            for service_name, service_status in self.services.items():
                if service_status.status != "running":
                    all_ready = False
                    break
            
            if all_ready:
                logger.info("All services are ready")
                return True
            
            time.sleep(2)
        
        logger.warning("Timeout waiting for services to be ready")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Institutional Trading System Launcher")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--command', choices=['start', 'stop', 'restart', 'status', 'health'], 
                       default='start', help='Command to execute')
    parser.add_argument('--service', help='Specific service name for restart command')
    parser.add_argument('--wait', type=int, default=60, help='Wait time for services to be ready')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = InstitutionalSystemLauncher(args.config)
    
    try:
        if args.command == 'start':
            logger.info("Starting institutional trading system...")
            if launcher.start_all_services():
                if launcher.wait_for_services(args.wait):
                    logger.info("System started successfully")
                    
                    # Keep running
                    try:
                        while launcher.running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Received interrupt, shutting down...")
                else:
                    logger.error("Services failed to start within timeout")
                    sys.exit(1)
            else:
                logger.error("Failed to start services")
                sys.exit(1)
                
        elif args.command == 'stop':
            logger.info("Stopping institutional trading system...")
            launcher.stop_all_services()
            logger.info("System stopped")
            
        elif args.command == 'restart':
            if args.service:
                logger.info(f"Restarting service {args.service}...")
                if launcher.restart_service(args.service):
                    logger.info(f"Service {args.service} restarted successfully")
                else:
                    logger.error(f"Failed to restart service {args.service}")
                    sys.exit(1)
            else:
                logger.info("Restarting all services...")
                launcher.stop_all_services()
                time.sleep(2)
                if launcher.start_all_services():
                    logger.info("All services restarted successfully")
                else:
                    logger.error("Failed to restart services")
                    sys.exit(1)
                    
        elif args.command == 'status':
            status = launcher.get_system_status()
            print(json.dumps(status, indent=2))
            
        elif args.command == 'health':
            status = launcher.get_system_status()
            
            # Check overall health
            all_healthy = True
            for service_name, service_status in status['services'].items():
                if service_status['status'] != 'running':
                    all_healthy = False
                    print(f"❌ {service_name}: {service_status['status']}")
                else:
                    print(f"✅ {service_name}: {service_status['status']}")
            
            if all_healthy:
                print("🎉 All services are healthy")
                sys.exit(0)
            else:
                print("⚠️  Some services are not healthy")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 