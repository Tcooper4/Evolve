"""
Health Service

This module implements health monitoring and service status functionality.

Note: This module was adapted from the legacy automation/services/automation_health.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
import psutil
import time
from datetime import datetime, timedelta
import requests
import socket
import ssl
import OpenSSL
from dataclasses import dataclass
from enum import Enum

class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    type: str
    target: str
    interval: int
    timeout: int
    retries: int
    threshold: float
    tags: Dict[str, str]

class HealthService:
    """Manages health checks and service status monitoring."""
    
    def __init__(self, config_path: str):
        """Initialize health service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self.running = False
    
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/health")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "health_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    self.config = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
            
            # Initialize health checks from config
            for check_config in self.config['health_checks']:
                self.health_checks[check_config['name']] = HealthCheck(
                    name=check_config['name'],
                    type=check_config['type'],
                    target=check_config['target'],
                    interval=check_config.get('interval', 60),
                    timeout=check_config.get('timeout', 5),
                    retries=check_config.get('retries', 3),
                    threshold=check_config.get('threshold', 0.8),
                    tags=check_config.get('tags', {})
                )
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
    
    async def check_http_health(self, check: HealthCheck) -> Dict[str, Any]:
        """Check HTTP endpoint health."""
        try:
            start_time = time.time()
            response = requests.get(
                check.target,
                timeout=check.timeout
            )
            response_time = time.time() - start_time
            
            return {
                'status': ServiceStatus.HEALTHY if response.status_code == 200 else ServiceStatus.UNHEALTHY,
                'response_time': response_time,
                'status_code': response.status_code,
                'error': None
            }
        except Exception as e:
            return {
                'status': ServiceStatus.UNHEALTHY,
                'response_time': None,
                'status_code': None,
                'error': str(e)
            }
    
    async def check_tcp_health(self, check: HealthCheck) -> Dict[str, Any]:
        """Check TCP endpoint health."""
        try:
            host, port = check.target.split(':')
            port = int(port)
            
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(check.timeout)
            sock.connect((host, port))
            response_time = time.time() - start_time
            sock.close()
            
            return {
                'status': ServiceStatus.HEALTHY,
                'response_time': response_time,
                'error': None
            }
        except Exception as e:
            return {
                'status': ServiceStatus.UNHEALTHY,
                'response_time': None,
                'error': str(e)
            }
    
    async def check_process_health(self, check: HealthCheck) -> Dict[str, Any]:
        """Check process health."""
        try:
            pid = int(check.target)
            process = psutil.Process(pid)
            
            if not process.is_running():
                return {
                    'status': ServiceStatus.UNHEALTHY,
                    'cpu_percent': None,
                    'memory_percent': None,
                    'error': 'Process not running'
                }
            
            return {
                'status': ServiceStatus.HEALTHY,
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'error': None
            }
        except Exception as e:
            return {
                'status': ServiceStatus.UNHEALTHY,
                'cpu_percent': None,
                'memory_percent': None,
                'error': str(e)
            }
    
    async def check_ssl_health(self, check: HealthCheck) -> Dict[str, Any]:
        """Check SSL certificate health."""
        try:
            host, port = check.target.split(':')
            port = int(port)
            
            cert = ssl.get_server_certificate((host, port))
            x509 = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
            
            expires_at = datetime.strptime(
                x509.get_notAfter().decode('ascii'),
                '%Y%m%d%H%M%SZ'
            )
            
            days_until_expiry = (expires_at - datetime.now()).days
            
            return {
                'status': ServiceStatus.HEALTHY if days_until_expiry > 30 else ServiceStatus.DEGRADED,
                'expires_at': expires_at.isoformat(),
                'days_until_expiry': days_until_expiry,
                'issuer': dict(x509.get_issuer().get_components()),
                'error': None
            }
        except Exception as e:
            return {
                'status': ServiceStatus.UNHEALTHY,
                'expires_at': None,
                'days_until_expiry': None,
                'issuer': None,
                'error': str(e)
            }
    
    async def run_health_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a health check."""
        try:
            if check.type == 'http':
                result = await self.check_http_health(check)
            elif check.type == 'tcp':
                result = await self.check_tcp_health(check)
            elif check.type == 'process':
                result = await self.check_process_health(check)
            elif check.type == 'ssl':
                result = await self.check_ssl_health(check)
            else:
                raise ValueError(f"Unsupported health check type: {check.type}")
            
            # Update service status
            self.service_status[check.name] = result['status']
            
            return {
                'check': check.name,
                'timestamp': datetime.now().isoformat(),
                'status': result['status'].value,
                'details': result
            }
        except Exception as e:
            self.logger.error(f"Error running health check {check.name}: {str(e)}")
            return {
                'check': check.name,
                'timestamp': datetime.now().isoformat(),
                'status': ServiceStatus.UNKNOWN.value,
                'error': str(e)
            }
    
    async def run_health_checks(self) -> List[Dict[str, Any]]:
        """Run all health checks."""
        try:
            tasks = []
            for check in self.health_checks.values():
                tasks.append(self.run_health_check(check))
            
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            self.logger.error(f"Error running health checks: {str(e)}")
            raise
    
    def get_service_status(self, service_name: str) -> Optional[ServiceStatus]:
        """Get status of a specific service."""
        return self.service_status.get(service_name)
    
    def get_all_service_statuses(self) -> Dict[str, ServiceStatus]:
        """Get status of all services."""
        return self.service_status.copy()
    
    async def start(self) -> None:
        """Start health monitoring."""
        try:
            self.running = True
            while self.running:
                results = await self.run_health_checks()
                self.logger.info(f"Health check results: {json.dumps(results, indent=2)}")
                await asyncio.sleep(self.config['check_interval'])
        except Exception as e:
            self.logger.error(f"Error in health monitoring: {str(e)}")
            raise
        finally:
            self.running = False
    
    def stop(self) -> None:
        """Stop health monitoring."""
        self.running = False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor service health')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    try:
        service = HealthService(args.config)
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logging.info("Health monitoring interrupted")
    except Exception as e:
        logging.error(f"Error in health monitoring: {str(e)}")
        raise

if __name__ == '__main__':
    main() 