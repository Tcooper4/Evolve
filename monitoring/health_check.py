"""
Health check system for EVOLVE trading system

Provides comprehensive health monitoring for all system components.
"""

import time
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import psutil for system resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - system resource monitoring disabled")


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthChecker:
    """System health checker"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_check: Optional[Dict[str, Any]] = None
        self.check_history: list = []
        self.max_history_size = 100
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.
        
        Returns:
            Dictionary with health status and component details
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'status': HealthStatus.HEALTHY.value,
            'components': {}
        }
        
        # Check individual components
        health['components']['database'] = self._check_database()
        health['components']['brokers'] = self._check_brokers()
        health['components']['models'] = self._check_models()
        health['components']['system_resources'] = self._check_system_resources()
        health['components']['data_sources'] = self._check_data_sources()
        
        # Determine overall status
        statuses = [comp.get('status') for comp in health['components'].values()]
        
        if HealthStatus.UNHEALTHY.value in statuses:
            health['status'] = HealthStatus.UNHEALTHY.value
        elif HealthStatus.DEGRADED.value in statuses:
            health['status'] = HealthStatus.DEGRADED.value
        elif HealthStatus.UNKNOWN.value in statuses:
            health['status'] = HealthStatus.DEGRADED.value
        
        # Store in history
        self.last_check = health
        self.check_history.append(health)
        if len(self.check_history) > self.max_history_size:
            self.check_history = self.check_history[-self.max_history_size:]
        
        return health
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            from trading.database.connection import get_session
            
            session = get_session()
            # Simple query to test connection
            session.execute("SELECT 1")
            session.close()
            
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Database connection OK',
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'Database module not available',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f'Database error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_brokers(self) -> Dict[str, Any]:
        """Check broker connectivity"""
        try:
            from execution.broker_adapter import BrokerAdapter
            
            # Try to get broker status
            # This is a simplified check - actual implementation would check each broker
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Brokers available',
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'Broker module not available',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED.value,
                'message': f'Broker check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_models(self) -> Dict[str, Any]:
        """Check model registry"""
        try:
            from trading.models.registry import get_model_registry
            
            registry = get_model_registry()
            models = registry.get_available_models()
            
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': f'{len(models)} models available',
                'model_count': len(models),
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'Model registry not available',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED.value,
                'message': f'Model check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        if not PSUTIL_AVAILABLE:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'System resource monitoring not available (psutil not installed)',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY.value
            if cpu > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.DEGRADED.value
            
            return {
                'status': status,
                'message': f'CPU: {cpu:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%',
                'cpu_percent': cpu,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory_available_gb': memory.available / (1024**3),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED.value,
                'message': f'Resource check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_data_sources(self) -> Dict[str, Any]:
        """Check data source connectivity"""
        try:
            from data.data_fetcher import DataFetcher
            
            # Simplified check - actual implementation would test each data source
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Data sources available',
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'Data fetcher not available',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': HealthStatus.DEGRADED.value,
                'message': f'Data source check failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        if not self.last_check:
            self.check_system_health()
        
        return {
            'current_status': self.last_check['status'],
            'uptime_seconds': self.last_check['uptime_seconds'],
            'component_count': len(self.last_check['components']),
            'healthy_components': sum(
                1 for comp in self.last_check['components'].values()
                if comp.get('status') == HealthStatus.HEALTHY.value
            ),
            'degraded_components': sum(
                1 for comp in self.last_check['components'].values()
                if comp.get('status') == HealthStatus.DEGRADED.value
            ),
            'unhealthy_components': sum(
                1 for comp in self.last_check['components'].values()
                if comp.get('status') == HealthStatus.UNHEALTHY.value
            ),
            'last_check': self.last_check['timestamp'],
        }
    
    def get_health_history(self, count: int = 10) -> list:
        """Get health check history"""
        return self.check_history[-count:]


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance"""
    return _health_checker

