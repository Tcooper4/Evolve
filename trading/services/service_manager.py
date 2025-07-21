"""
Service Manager

Manages all agent services, providing a centralized interface for starting,
stopping, and monitoring services via Redis pub/sub communication.
"""

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict

import redis

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Manages all agent services with Redis pub/sub communication.

    Provides centralized control and monitoring of all services.
    """

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0
    ):
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
            "agent_api": {
                "script": "launch_agent_api.py",
                "description": "Agent API Service with WebSocket Support",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "model_builder": {
                "script": "launch_model_builder.py",
                "description": "Model Builder Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "performance_critic": {
                "script": "launch_performance_critic.py",
                "description": "Performance Critic Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "updater": {
                "script": "launch_updater.py",
                "description": "Updater Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "research": {
                "script": "launch_research.py",
                "description": "Research Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "meta_tuner": {
                "script": "launch_meta_tuner.py",
                "description": "Meta Tuner Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "multimodal": {
                "script": "launch_multimodal.py",
                "description": "Multimodal Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "prompt_router": {
                "script": "launch_prompt_router.py",
                "description": "Prompt Router Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "real_time_signal_center": {
                "script": "launch_real_time_signal_center.py",
                "description": "Real-time Signal Center Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
            "report_service": {
                "script": "launch_report_service.py",
                "description": "Report Service",
                "status": "stopped",
                "process": None,
                "pid": None,
            },
        }

        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None

        # Statistics
        self.stats = {
            "total_services": len(self.services),
            "running_services": 0,
            "stopped_services": len(self.services),
            "failed_services": 0,
            "start_time": time.time(),
        }

        logger.info(f"ServiceManager initialized with {len(self.services)} services")

    def _subscribe_to_services(self):
        """Subscribe to service status channels."""
        channels = [f"service:{name}" for name in self.services.keys()]
        self.pubsub.subscribe(*channels)
        logger.info(f"Subscribed to {len(channels)} service channels")

    def _monitor_services(self):
        """Monitor service status via Redis pub/sub."""
        logger.info("Starting service monitoring...")
        self._subscribe_to_services()

        for message in self.pubsub.listen():
            if message["type"] == "message":
                self._handle_service_message(message)

    def _handle_service_message(self, message):
        """Handle service status messages."""
        try:
            channel = message["channel"].decode("utf-8")
            data = json.loads(message["data"].decode("utf-8"))

            service_name = channel.split(":")[1]
            if service_name in self.services:
                self.services[service_name]["status"] = data.get("status", "unknown")
                logger.info(f"Service {service_name} status: {data.get('status')}")

        except Exception as e:
            logger.error(f"Error handling service message: {e}")

    def start_service(self, service_name: str) -> Dict[str, Any]:
        """
        Start a specific service.

        Args:
            service_name: Name of the service to start

        Returns:
            Dictionary with start result
        """
        if service_name not in self.services:
            return {"success": False, "error": f"Service '{service_name}' not found"}

        service = self.services[service_name]

        if service["status"] == "running":
            return {
                "success": False,
                "error": f"Service '{service_name}' is already running",
            }

        try:
            # Find the script file
            script_path = Path(__file__).parent / service["script"]
            if not script_path.exists():
                return {
                    "success": False,
                    "error": f"Script '{service['script']}' not found",
                }

            # Start the service process
            process = subprocess.Popen(
                ["python", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Update service status
            service["process"] = process
            service["pid"] = process.pid
            service["status"] = "starting"

            # Update statistics
            self.stats["running_services"] += 1
            self.stats["stopped_services"] -= 1

            logger.info(f"Started service '{service_name}' with PID {process.pid}")

            return {
                "success": True,
                "service_name": service_name,
                "pid": process.pid,
                "status": "starting",
            }

        except Exception as e:
            logger.error(f"Failed to start service '{service_name}': {e}")
            service["status"] = "failed"
            self.stats["failed_services"] += 1

            return {"success": False, "error": str(e)}

    def stop_service(self, service_name: str) -> Dict[str, Any]:
        """
        Stop a specific service.

        Args:
            service_name: Name of the service to stop

        Returns:
            Dictionary with stop result
        """
        if service_name not in self.services:
            return {"success": False, "error": f"Service '{service_name}' not found"}

        service = self.services[service_name]

        if service["status"] == "stopped":
            return {
                "success": False,
                "error": f"Service '{service_name}' is already stopped",
            }

        try:
            if service["process"]:
                # Send termination signal
                service["process"].terminate()

                # Wait for process to terminate
                try:
                    service["process"].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if timeout
                    service["process"].kill()
                    service["process"].wait()

            # Update service status
            service["process"] = None
            service["pid"] = None
            service["status"] = "stopped"

            # Update statistics
            if service["status"] == "running":
                self.stats["running_services"] -= 1
            self.stats["stopped_services"] += 1

            logger.info(f"Stopped service '{service_name}'")

            return {"success": True, "service_name": service_name, "status": "stopped"}

        except Exception as e:
            logger.error(f"Failed to stop service '{service_name}': {e}")
            return {"success": False, "error": str(e)}

    def start_all_services(self) -> Dict[str, Any]:
        """
        Start all services.

        Returns:
            Dictionary with start results
        """
        results = {}
        for service_name in self.services.keys():
            results[service_name] = self.start_service(service_name)

        return results

    def stop_all_services(self) -> Dict[str, Any]:
        """
        Stop all services.

        Returns:
            Dictionary with stop results
        """
        results = {}
        for service_name in self.services.keys():
            results[service_name] = self.stop_service(service_name)

        return results

    def get_service_status(self, service_name: str = None) -> Dict[str, Any]:
        """
        Get status of services.

        Args:
            service_name: Specific service name (optional)

        Returns:
            Dictionary with service status(es)
        """
        if service_name:
            if service_name not in self.services:
                return {"error": f"Service '{service_name}' not found"}

            service = self.services[service_name]
            return {
                "service_name": service_name,
                "status": service["status"],
                "pid": service["pid"],
                "description": service["description"],
            }

        # Return all services status
        statuses = {}
        for name, service in self.services.items():
            statuses[name] = {
                "status": service["status"],
                "pid": service["pid"],
                "description": service["description"],
            }

        return statuses

    def send_message(
        self, service_name: str, message_type: str, data: Dict[str, Any]
    ) -> bool:
        """
        Send message to a service via Redis.

        Args:
            service_name: Target service name
            message_type: Type of message
            data: Message data

        Returns:
            True if message sent successfully
        """
        try:
            message = {"type": message_type, "data": data, "timestamp": time.time()}

            channel = f"service:{service_name}:command"
            self.redis_client.publish(channel, json.dumps(message))

            logger.info(f"Sent {message_type} message to {service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message to {service_name}: {e}")
            return False

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "stats": self.stats,
            "uptime": time.time() - self.stats["start_time"],
            "monitoring_active": self.monitoring_active,
        }

    def shutdown(self) -> Dict[str, Any]:
        """
        Shutdown the service manager.

        Returns:
            Dictionary with shutdown result
        """
        try:
            # Stop monitoring
            if self.monitoring_active:
                self.monitoring_active = False
                if self.monitor_thread:
                    self.monitor_thread.join(timeout=5)

            # Stop all services
            stop_results = self.stop_all_services()

            # Close Redis connection
            self.pubsub.close()
            self.redis_client.close()

            logger.info("ServiceManager shutdown completed")

            return {
                "success": True,
                "stop_results": stop_results,
                "message": "ServiceManager shutdown completed",
            }

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Main function for running the service manager."""
    manager = ServiceManager()

    try:
        # Start monitoring in background
        manager.monitoring_active = True
        manager.monitor_thread = threading.Thread(target=manager._monitor_services)
        manager.monitor_thread.daemon = True
        manager.monitor_thread.start()

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        manager.shutdown()


if __name__ == "__main__":
    main()
