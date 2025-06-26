"""
Base Service Class

Provides Redis pub/sub communication infrastructure for all agent services.
"""

import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
import redis
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Base class for all agent services with Redis pub/sub communication.
    
    Each service runs independently and communicates via Redis channels.
    """
    
    def __init__(self, service_name: str, redis_host: str = 'localhost', 
                 redis_port: int = 6379, redis_db: int = 0):
        """
        Initialize the base service.
        
        Args:
            service_name: Name of the service (e.g., 'model_builder', 'critic')
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        """
        self.service_name = service_name
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.pubsub = self.redis_client.pubsub()
        
        # Service state
        self.is_running = False
        self.thread = None
        
        # Channel names
        self.input_channel = f"{service_name}_input"
        self.output_channel = f"{service_name}_output"
        self.control_channel = f"{service_name}_control"
        
        # Message handlers
        self.message_handlers = {}
        self._setup_message_handlers()
        
        logger.info(f"Initialized {service_name} service")
    
    def _setup_message_handlers(self):
        """Setup default message handlers."""
        self.message_handlers = {
            'start': self._handle_start,
            'stop': self._handle_stop,
            'status': self._handle_status,
            'ping': self._handle_ping
        }
    
    def start(self):
        """Start the service and begin listening for messages."""
        if self.is_running:
            logger.warning(f"{self.service_name} service is already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        
        # Subscribe to channels
        self.pubsub.subscribe(self.input_channel, self.control_channel)
        
        # Publish startup message
        self._publish_output({
            'type': 'service_started',
            'service': self.service_name,
            'timestamp': time.time()
        })
        
        logger.info(f"{self.service_name} service started")
    
    def stop(self):
        """Stop the service."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Publish shutdown message
        self._publish_output({
            'type': 'service_stopped',
            'service': self.service_name,
            'timestamp': time.time()
        })
        
        # Unsubscribe and close
        self.pubsub.unsubscribe()
        self.pubsub.close()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info(f"{self.service_name} service stopped")
    
    def _listen_loop(self):
        """Main listening loop for Redis messages."""
        logger.info(f"{self.service_name} listening on channels: {self.input_channel}, {self.control_channel}")
        
        while self.is_running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    self._process_message(message)
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                time.sleep(1)
    
    def _process_message(self, message):
        """Process incoming Redis message."""
        try:
            data = json.loads(message['data'].decode('utf-8'))
            message_type = data.get('type', 'unknown')
            
            # Handle control messages
            if message['channel'].decode('utf-8') == self.control_channel:
                handler = self.message_handlers.get(message_type)
                if handler:
                    handler(data)
                else:
                    logger.warning(f"Unknown control message type: {message_type}")
            
            # Handle input messages
            elif message['channel'].decode('utf-8') == self.input_channel:
                self._handle_input_message(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _handle_input_message(self, data: Dict[str, Any]):
        """Handle input message - to be implemented by subclasses."""
        try:
            result = self.process_message(data)
            if result:
                self._publish_output(result)
        except Exception as e:
            logger.error(f"Error processing input message: {e}")
            self._publish_output({
                'type': 'error',
                'service': self.service_name,
                'error': str(e),
                'original_message': data,
                'timestamp': time.time()
            })
    
    @abstractmethod
    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming message - must be implemented by subclasses.
        
        Args:
            data: Message data dictionary
            
        Returns:
            Response data dictionary or None
        """
        pass
    
    def _publish_output(self, data: Dict[str, Any]):
        """Publish message to output channel."""
        try:
            data['service'] = self.service_name
            data['timestamp'] = time.time()
            self.redis_client.publish(self.output_channel, json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to publish output: {e}")
    
    def send_message(self, target_service: str, message_type: str, data: Dict[str, Any]):
        """Send message to another service."""
        try:
            message = {
                'type': message_type,
                'from': self.service_name,
                'data': data,
                'timestamp': time.time()
            }
            self.redis_client.publish(f"{target_service}_input", json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message to {target_service}: {e}")
    
    def _handle_start(self, data: Dict[str, Any]):
        """Handle start control message."""
        logger.info(f"Received start command for {self.service_name}")
        self._publish_output({
            'type': 'status',
            'status': 'running',
            'service': self.service_name
        })
    
    def _handle_stop(self, data: Dict[str, Any]):
        """Handle stop control message."""
        logger.info(f"Received stop command for {self.service_name}")
        self.stop()
    
    def _handle_status(self, data: Dict[str, Any]):
        """Handle status request."""
        self._publish_output({
            'type': 'status',
            'status': 'running' if self.is_running else 'stopped',
            'service': self.service_name,
            'thread_alive': self.thread.is_alive() if self.thread else False
        })
    
    def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping message."""
        self._publish_output({
            'type': 'pong',
            'service': self.service_name,
            'timestamp': time.time()
        })
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            'service_name': self.service_name,
            'is_running': self.is_running,
            'input_channel': self.input_channel,
            'output_channel': self.output_channel,
            'control_channel': self.control_channel
        } 