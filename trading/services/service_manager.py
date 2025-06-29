"""
Service Manager

Manages all agent services, providing a centralized interface for starting,
stopping, and monitoring services via Redis pub/sub communication.
"""

import json
import logging
import time
import threading
from typing import Dict, Any, List, Optional
import redis
import subprocess
import os
import signal
from pathlib import Path

# Add report service import
from report.report_service import ReportService

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Manages all agent services with Redis pub/sub communication.
    
    Provides centralized control and monitoring of all services.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0):
        """
        Initialize the ServiceManager.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.pubsub = self.redis_client.pubsub()
        
        # Service configurations
        self.services = {
            'model_builder': {
                'script': 'launch_model_builder.py',
                'description': 'Model Builder Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'performance_critic': {
                'script': 'launch_performance_critic.py',
                'description': 'Performance Critic Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'updater': {
                'script': 'launch_updater.py',
                'description': 'Updater Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'research': {
                'script': 'launch_research.py',
                'description': 'Research Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'meta_tuner': {
                'script': 'launch_meta_tuner.py',
                'description': 'Meta Tuner Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'multimodal': {
                'script': 'launch_multimodal.py',
                'description': 'Multimodal Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'prompt_router': {
                'script': 'launch_prompt_router.py',
                'description': 'Prompt Router Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'quant_gpt': {
                'script': 'launch_quant_gpt.py',
                'description': 'QuantGPT Natural Language Interface',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'safe_executor': {
                'script': 'launch_safe_executor.py',
                'description': 'Safe Executor for User-Defined Models',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'report': {
                'script': 'launch_report_service.py',
                'description': 'Report Generation Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            },
            'reasoning': {
                'script': 'launch_reasoning_service.py',
                'description': 'Reasoning Logger Service',
                'status': 'stopped',
                'process': None,
                'pid': None
            }
        }
        
        # Manager state
        self.is_running = False
        self.monitor_thread = None
        
        # Subscribe to all service output channels
        self._subscribe_to_services()
        
        logger.info("ServiceManager initialized")
    
    def _subscribe_to_services(self):
        """Subscribe to all service output channels."""
        channels = [f"{service_name}_output" for service_name in self.services.keys()]
        self.pubsub.subscribe(*channels)
        
        # Start monitoring thread
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_services, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_services(self):
        """Monitor service messages and update status."""
        logger.info("Starting service monitoring...")
        
        while self.is_running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    self._handle_service_message(message)
            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                time.sleep(1)
    
    def _handle_service_message(self, message):
        """Handle messages from services."""
        try:
            data = json.loads(message['data'].decode('utf-8'))
            service_name = data.get('service')
            message_type = data.get('type')
            
            if service_name in self.services:
                if message_type == 'service_started':
                    self.services[service_name]['status'] = 'running'
                    logger.info(f"{service_name} service started")
                elif message_type == 'service_stopped':
                    self.services[service_name]['status'] = 'stopped'
                    logger.info(f"{service_name} service stopped")
                elif message_type == 'status':
                    self.services[service_name]['status'] = data.get('status', 'unknown')
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode service message: {e}")
        except Exception as e:
            logger.error(f"Error handling service message: {e}")
    
    def start_service(self, service_name: str) -> Dict[str, Any]:
        """
        Start a specific service.
        
        Args:
            service_name: Name of the service to start
            
        Returns:
            Result dictionary with status and details
        """
        if service_name not in self.services:
            return {
                'success': False,
                'error': f'Unknown service: {service_name}'
            }
        
        service_config = self.services[service_name]
        
        if service_config['status'] == 'running':
            return {
                'success': False,
                'error': f'{service_name} is already running'
            }
        
        try:
            # Get the script path
            script_path = Path(__file__).parent / service_config['script']
            
            if not script_path.exists():
                return {
                    'success': False,
                    'error': f'Service script not found: {script_path}'
                }
            
            # Start the service process
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent
            )
            
            # Update service configuration
            service_config['process'] = process
            service_config['pid'] = process.pid
            service_config['status'] = 'starting'
            
            logger.info(f"Started {service_name} service (PID: {process.pid})")
            
            return {
                'success': True,
                'service_name': service_name,
                'pid': process.pid,
                'status': 'starting'
            }
            
        except Exception as e:
            logger.error(f"Error starting {service_name} service: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def stop_service(self, service_name: str) -> Dict[str, Any]:
        """
        Stop a specific service.
        
        Args:
            service_name: Name of the service to stop
            
        Returns:
            Result dictionary with status and details
        """
        if service_name not in self.services:
            return {
                'success': False,
                'error': f'Unknown service: {service_name}'
            }
        
        service_config = self.services[service_name]
        
        if service_config['status'] == 'stopped':
            return {
                'success': False,
                'error': f'{service_name} is already stopped'
            }
        
        try:
            # Send stop command via Redis
            self.redis_client.publish(
                f"{service_name}_control",
                json.dumps({'type': 'stop'})
            )
            
            # Wait a bit for graceful shutdown
            time.sleep(2)
            
            # Force kill if still running
            if service_config['process'] and service_config['process'].poll() is None:
                service_config['process'].terminate()
                time.sleep(1)
                
                if service_config['process'].poll() is None:
                    service_config['process'].kill()
            
            # Update service configuration
            service_config['status'] = 'stopped'
            service_config['process'] = None
            service_config['pid'] = None
            
            logger.info(f"Stopped {service_name} service")
            
            return {
                'success': True,
                'service_name': service_name,
                'status': 'stopped'
            }
            
        except Exception as e:
            logger.error(f"Error stopping {service_name} service: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def start_all_services(self) -> Dict[str, Any]:
        """
        Start all services.
        
        Returns:
            Result dictionary with status for each service
        """
        results = {}
        
        for service_name in self.services.keys():
            results[service_name] = self.start_service(service_name)
            time.sleep(1)  # Small delay between starts
        
        return results
    
    def stop_all_services(self) -> Dict[str, Any]:
        """
        Stop all services.
        
        Returns:
            Result dictionary with status for each service
        """
        results = {}
        
        for service_name in self.services.keys():
            results[service_name] = self.stop_service(service_name)
            time.sleep(1)  # Small delay between stops
        
        return results
    
    def get_service_status(self, service_name: str = None) -> Dict[str, Any]:
        """
        Get status of services.
        
        Args:
            service_name: Specific service name or None for all
            
        Returns:
            Status dictionary
        """
        if service_name:
            if service_name not in self.services:
                return {'error': f'Unknown service: {service_name}'}
            
            service_config = self.services[service_name]
            return {
                'service_name': service_name,
                'status': service_config['status'],
                'pid': service_config['pid'],
                'description': service_config['description']
            }
        else:
            return {
                service_name: {
                    'status': config['status'],
                    'pid': config['pid'],
                    'description': config['description']
                }
                for service_name, config in self.services.items()
            }
    
    def send_message(self, service_name: str, message_type: str, data: Dict[str, Any]) -> bool:
        """
        Send a message to a specific service.
        
        Args:
            service_name: Name of the target service
            message_type: Type of message
            data: Message data
            
        Returns:
            True if message sent successfully
        """
        try:
            message = {
                'type': message_type,
                'data': data,
                'timestamp': time.time()
            }
            
            self.redis_client.publish(
                f"{service_name}_input",
                json.dumps(message)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to {service_name}: {e}")
            return False
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        running_services = sum(
            1 for config in self.services.values() 
            if config['status'] == 'running'
        )
        
        return {
            'total_services': len(self.services),
            'running_services': running_services,
            'stopped_services': len(self.services) - running_services,
            'services': self.get_service_status()
        }
    
    def shutdown(self):
        """Shutdown the service manager."""
        logger.info("Shutting down ServiceManager...")
        
        self.is_running = False
        
        # Stop all services
        self.stop_all_services()
        
        # Close Redis connection
        self.pubsub.close()
        
        logger.info("ServiceManager shutdown complete")


def main():
    """Main function for running the ServiceManager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Agent Service Manager')
    parser.add_argument('--action', choices=['start', 'stop', 'status', 'start-all', 'stop-all'], 
                       default='status', help='Action to perform')
    parser.add_argument('--service', help='Service name for start/stop actions')
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Initialize service manager
    manager = ServiceManager(
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    
    try:
        if args.action == 'start':
            if not args.service:
                print("Error: --service is required for start action")
                return {"status": "failed", "action": "start", "error": "Missing service parameter"}
            result = manager.start_service(args.service)
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "start", "result": result}
            
        elif args.action == 'stop':
            if not args.service:
                print("Error: --service is required for stop action")
                return {"status": "failed", "action": "stop", "error": "Missing service parameter"}
            result = manager.stop_service(args.service)
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "stop", "result": result}
            
        elif args.action == 'start-all':
            result = manager.start_all_services()
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "start-all", "result": result}
            
        elif args.action == 'stop-all':
            result = manager.stop_all_services()
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "stop-all", "result": result}
            
        elif args.action == 'status':
            result = manager.get_service_status()
            print(json.dumps(result, indent=2))
            return {"status": "completed", "action": "status", "result": result}
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        return {"status": "interrupted", "action": args.action}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "failed", "action": args.action, "error": str(e)}
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main() 