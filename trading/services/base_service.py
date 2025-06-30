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
from datetime import datetime

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
        
        # Initialize Redis connection with error handling
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            self.pubsub = self.redis_client.pubsub()
            self.redis_available = True
            logger.info(f"Redis connection established for {service_name}")
        except Exception as e:
            logger.warning(f"Redis connection failed for {service_name}: {e}. Running in local mode.")
            self.redis_client = None
            self.pubsub = None
            self.redis_available = False
        
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
        
        return {
            'success': True,
            'message': f'{service_name} service initialized successfully',
            'timestamp': datetime.now().isoformat(),
            'redis_available': self.redis_available
        }

    def _setup_message_handlers(self) -> Dict[str, Callable]:
        """Setup default message handlers.
        
        Returns:
            Dictionary of message handlers
        """
        self.message_handlers = {
            'start': self._handle_start,
            'stop': self._handle_stop,
            'status': self._handle_status,
            'ping': self._handle_ping
        }
        
        return {
            'success': True,
            'result': self.message_handlers,
            'message': 'Message handlers setup completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def start(self) -> Dict[str, Any]:
        """Start the service and begin listening for messages.
        
        Returns:
            Dictionary with start status
        """
        if self.is_running:
            logger.warning(f"{self.service_name} service is already running")
            return {
                'success': False,
                'error': 'Service already running',
                'service': self.service_name,
                'timestamp': datetime.now().isoformat()
            }
        
        self.is_running = True
        
        if self.redis_available:
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
            
            logger.info(f"{self.service_name} service started with Redis")
            return {
                'success': True,
                'service': self.service_name,
                'mode': 'redis',
                'channels': [self.input_channel, self.control_channel],
                'timestamp': datetime.now().isoformat()
            }
        else:
            logger.info(f"{self.service_name} service started in local mode (no Redis)")
            return {
                'success': True,
                'service': self.service_name,
                'mode': 'local',
                'channels': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def stop(self) -> Dict[str, Any]:
        """Stop the service.
        
        Returns:
            Dictionary with stop status
        """
        if not self.is_running:
            return {
                'success': False,
                'error': 'Service not running',
                'service': self.service_name,
                'timestamp': datetime.now().isoformat()
            }
        
        self.is_running = False
        
        if self.redis_available:
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
        return {
            'success': True,
            'service': self.service_name,
            'thread_joined': self.thread and not self.thread.is_alive() if self.thread else True,
            'timestamp': datetime.now().isoformat()
        }
    
    def _listen_loop(self) -> Dict[str, Any]:
        """Main listening loop for Redis messages.
        
        Returns:
            Dictionary with loop status
        """
        logger.info(f"{self.service_name} listening on channels: {self.input_channel}, {self.control_channel}")
        
        message_count = 0
        while self.is_running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    self._process_message(message)
                    message_count += 1
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                time.sleep(1)
        
        return {
            'success': True,
            'service': self.service_name,
            'loop_ended': True,
            'messages_processed': message_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_message(self, message) -> Dict[str, Any]:
        """Process incoming Redis message.
        
        Returns:
            Dictionary with processing status
        """
        try:
            data = json.loads(message['data'].decode('utf-8'))
            message_type = data.get('type', 'unknown')
            
            # Handle control messages
            if message['channel'].decode('utf-8') == self.control_channel:
                handler = self.message_handlers.get(message_type)
                if handler:
                    handler(data)
                    return {
                        'success': True,
                        'message': f'Control message {message_type} processed',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Handle input messages
            elif message['channel'].decode('utf-8') == self.input_channel:
                return self._handle_input_message(data)
            
            return {
                'success': True,
                'message': f'Message {message_type} processed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_input_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle input messages by delegating to the abstract process_message method.
        
        Args:
            data: Message data
            
        Returns:
            Dictionary with processing result
        """
        try:
            result = self.process_message(data)
            if result:
                self._publish_output(result)
            
            return {
                'success': True,
                'message': 'Input message processed',
                'timestamp': datetime.now().isoformat(),
                'has_result': result is not None
            }
        except Exception as e:
            logger.error(f"Error handling input message: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @abstractmethod
    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a message (to be implemented by subclasses).
        
        Args:
            data: Message data
            
        Returns:
            Optional response data
        """
        pass
    
    def _publish_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Publish output message to Redis.
        
        Args:
            data: Data to publish
            
        Returns:
            Dictionary with publish status
        """
        if not self.redis_available:
            logger.warning(f"Cannot publish output: Redis not available for {self.service_name}")
            return {
                'success': False,
                'error': 'Redis not available',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Add service info to message
            data['service'] = self.service_name
            data['timestamp'] = time.time()
            
            # Publish to output channel
            self.redis_client.publish(self.output_channel, json.dumps(data))
            
            return {
                'success': True,
                'message': f'Output published to {self.output_channel}',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error publishing output: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def send_message(self, target_service: str, message_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to another service.
        
        Args:
            target_service: Name of the target service
            message_type: Type of message
            data: Message data
            
        Returns:
            Dictionary with send status
        """
        if not self.redis_available:
            return {
                'success': False,
                'error': 'Redis not available',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            message = {
                'type': message_type,
                'data': data,
                'from': self.service_name,
                'timestamp': time.time()
            }
            
            target_channel = f"{target_service}_input"
            self.redis_client.publish(target_channel, json.dumps(message))
            
            return {
                'success': True,
                'message': f'Message sent to {target_service}',
                'timestamp': datetime.now().isoformat(),
                'target_service': target_service,
                'message_type': message_type
            }
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_start(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle start control message.
        
        Args:
            data: Message data
            
        Returns:
            Dictionary with start status
        """
        logger.info(f"Received start command for {self.service_name}")
        return self.start()
    
    def _handle_stop(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop control message.
        
        Args:
            data: Message data
            
        Returns:
            Dictionary with stop status
        """
        logger.info(f"Received stop command for {self.service_name}")
        return self.stop()
    
    def _handle_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request.
        
        Args:
            data: Message data
            
        Returns:
            Dictionary with service status
        """
        status = self.get_service_info()
        self._publish_output(status)
        return status
    
    def _handle_ping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping message.
        
        Args:
            data: Message data
            
        Returns:
            Dictionary with ping response
        """
        response = {
            'type': 'pong',
            'service': self.service_name,
            'timestamp': time.time()
        }
        self._publish_output(response)
        
        return {
            'success': True,
            'message': 'Ping responded',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            'success': True,
            'result': {
                'service': self.service_name,
                'is_running': self.is_running,
                'redis_available': self.redis_available,
                'channels': {
                    'input': self.input_channel,
                    'output': self.output_channel,
                    'control': self.control_channel
                }
            },
            'message': 'Service information retrieved',
            'timestamp': datetime.now().isoformat()
        } 