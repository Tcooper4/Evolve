import json
import logging
import asyncio
from typing import Dict, Set, Optional, List
from fastapi import WebSocket, WebSocketDisconnect
from automation.notifications.notification_manager import NotificationManager
from system.infra.agents.auth.user_manager import UserManager
import os
import jwt
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from enum import Enum

# Setup logging
log_handler = RotatingFileHandler(
    os.getenv('LOG_FILE', 'trading.log'),
    maxBytes=int(os.getenv('LOG_MAX_SIZE', 10485760)),
    backupCount=int(os.getenv('LOG_BACKUP_COUNT', 5))
)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger = logging.getLogger('websocket')
logger.addHandler(log_handler)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

class ConnectionStatus(str, Enum):
    """WebSocket connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

class ConnectionState:
    """Connection state for a WebSocket client."""
    
    def __init__(self, username: str, websocket: WebSocket, token: str):
        self.username = username
        self.websocket = websocket
        self.token = token
        self.status = ConnectionStatus.CONNECTED
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # Start with 1 second
        self.max_reconnect_delay = 60.0  # Max 60 seconds
        self.last_reconnect_attempt = None
        self.is_admin = False
        self.metadata = {}
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def increment_reconnect_attempts(self):
        """Increment reconnect attempts and update delay."""
        self.reconnect_attempts += 1
        self.last_reconnect_attempt = datetime.utcnow()
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
    
    def can_reconnect(self) -> bool:
        """Check if reconnection is allowed."""
        return self.reconnect_attempts < self.max_reconnect_attempts
    
    def reset_reconnect_attempts(self):
        """Reset reconnect attempts after successful connection."""
        self.reconnect_attempts = 0
        self.reconnect_delay = 1.0
        self.last_reconnect_attempt = None

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[str, ConnectionState] = {}
        self.reconnect_queue: List[ConnectionState] = []
        self.reconnect_task = None
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # seconds
        
    async def start_reconnect_handler(self):
        """Start the reconnection handler task."""
        if self.reconnect_task is None or self.reconnect_task.done():
            self.reconnect_task = asyncio.create_task(self._reconnect_loop())
            logger.info("WebSocket reconnection handler started")
    
    async def stop_reconnect_handler(self):
        """Stop the reconnection handler task."""
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass
            logger.info("WebSocket reconnection handler stopped")
    
    async def _reconnect_loop(self):
        """Main reconnection loop."""
        while True:
            try:
                # Process reconnect queue
                await self._process_reconnect_queue()
                
                # Clean up stale connections
                await self._cleanup_stale_connections()
                
                # Wait before next iteration
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                logger.info("Reconnection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in reconnection loop: {e}")
                await asyncio.sleep(10)
    
    async def _process_reconnect_queue(self):
        """Process the reconnection queue."""
        current_time = datetime.utcnow()
        reconnect_candidates = []
        
        # Check which connections can be reconnected
        for state in self.reconnect_queue[:]:
            if not state.can_reconnect():
                # Remove from queue if max attempts reached
                self.reconnect_queue.remove(state)
                state.status = ConnectionStatus.FAILED
                logger.warning(f"Max reconnection attempts reached for user {state.username}")
                continue
            
            # Check if enough time has passed since last attempt
            if (state.last_reconnect_attempt is None or 
                (current_time - state.last_reconnect_attempt).total_seconds() >= state.reconnect_delay):
                reconnect_candidates.append(state)
        
        # Attempt reconnection for candidates
        for state in reconnect_candidates:
            await self._attempt_reconnection(state)
    
    async def _attempt_reconnection(self, state: ConnectionState):
        """Attempt to reconnect a client."""
        try:
            logger.info(f"Attempting reconnection for user {state.username} (attempt {state.reconnect_attempts + 1})")
            
            # Update state
            state.status = ConnectionStatus.RECONNECTING
            state.increment_reconnect_attempts()
            
            # Try to reconnect (this would typically involve creating a new WebSocket connection)
            # For now, we'll simulate the reconnection process
            success = await self._simulate_reconnection(state)
            
            if success:
                # Reconnection successful
                state.status = ConnectionStatus.CONNECTED
                state.reset_reconnect_attempts()
                self.reconnect_queue.remove(state)
                self.active_connections[state.username] = state.websocket
                self.connection_states[state.username] = state
                
                # Send reconnection success message
                await self._send_reconnection_success(state)
                
                logger.info(f"Successfully reconnected user {state.username}")
            else:
                # Reconnection failed, keep in queue for next attempt
                logger.warning(f"Reconnection failed for user {state.username}")
                
        except Exception as e:
            logger.error(f"Error during reconnection attempt for user {state.username}: {e}")
    
    async def _simulate_reconnection(self, state: ConnectionState) -> bool:
        """Simulate reconnection process (replace with actual implementation)."""
        # In a real implementation, this would:
        # 1. Create a new WebSocket connection
        # 2. Authenticate the user
        # 3. Restore the connection state
        
        # For now, we'll simulate success with some probability
        import random
        success_probability = max(0.1, 1.0 - (state.reconnect_attempts * 0.2))
        return random.random() < success_probability
    
    async def _send_reconnection_success(self, state: ConnectionState):
        """Send reconnection success message to client."""
        try:
            message = {
                'type': 'reconnection_success',
                'timestamp': datetime.utcnow().isoformat(),
                'attempts': state.reconnect_attempts,
                'message': 'Successfully reconnected to server'
            }
            await state.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending reconnection success message to {state.username}: {e}")
    
    async def _cleanup_stale_connections(self):
        """Clean up stale connections."""
        current_time = datetime.utcnow()
        stale_connections = []
        
        for username, state in self.connection_states.items():
            if state.status == ConnectionStatus.CONNECTED:
                # Check for inactivity
                if (current_time - state.last_activity).total_seconds() > self.connection_timeout:
                    stale_connections.append(username)
        
        for username in stale_connections:
            await self.disconnect(username, reason="inactivity_timeout")

    async def connect(self, websocket: WebSocket, token: str) -> bool:
        try:
            data = jwt.decode(
                token,
                os.getenv('JWT_SECRET'),
                algorithms=['HS256']
            )
            username = data['username']
            
            # Create connection state
            state = ConnectionState(username, websocket, token)
            state.is_admin = data.get('is_admin', False)
            
            # Store connection
            self.active_connections[username] = websocket
            self.connection_states[username] = state
            
            await websocket.accept()
            
            # Send connection success message
            await self._send_connection_success(state)
            
            # Start reconnection handler if not already running
            await self.start_reconnect_handler()
            
            logger.info(f"User {username} connected")
            return True
            
        except jwt.ExpiredSignatureError:
            logger.warning("Connection attempt with expired token")
            return False
        except jwt.InvalidTokenError:
            logger.warning("Connection attempt with invalid token")
            return False
        except Exception as e:
            logger.error(f"Error during WebSocket connection: {str(e)}")
            return False

    async def _send_connection_success(self, state: ConnectionState):
        """Send connection success message to client."""
        try:
            message = {
                'type': 'connection_success',
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': state.username,
                'is_admin': state.is_admin,
                'message': 'Successfully connected to server'
            }
            await state.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending connection success message to {state.username}: {e}")

    async def disconnect(self, username: str, reason: str = "user_disconnect") -> None:
        if username in self.active_connections:
            # Get connection state
            state = self.connection_states.get(username)
            
            if state:
                # Update state
                state.status = ConnectionStatus.DISCONNECTED
                
                # Send disconnect message
                await self._send_disconnect_message(state, reason)
                
                # Add to reconnect queue if appropriate
                if reason not in ["user_disconnect", "inactivity_timeout"] and state.can_reconnect():
                    self.reconnect_queue.append(state)
                    logger.info(f"Added user {username} to reconnection queue")
                else:
                    # Remove from states if not reconnecting
                    del self.connection_states[username]
            
            # Remove from active connections
            del self.active_connections[username]
            
            logger.info(f"User {username} disconnected (reason: {reason})")

    async def _send_disconnect_message(self, state: ConnectionState, reason: str):
        """Send disconnect message to client."""
        try:
            message = {
                'type': 'disconnect',
                'timestamp': datetime.utcnow().isoformat(),
                'reason': reason,
                'message': f'Disconnected from server: {reason}'
            }
            await state.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending disconnect message to {state.username}: {e}")

    async def broadcast(self, message: str) -> None:
        disconnected = []
        for username, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
                state = self.connection_states.get(username)
                if state:
                    state.update_activity()
            except WebSocketDisconnect:
                disconnected.append(username)
            except Exception as e:
                logger.error(f"Error broadcasting to {username}: {str(e)}")
                disconnected.append(username)
        
        # Handle disconnections
        for username in disconnected:
            await self.disconnect(username, reason="connection_lost")

    async def send_personal_message(self, username: str, message: str) -> bool:
        if username not in self.active_connections:
            logger.warning(f"Attempt to send message to disconnected user {username}")
            return False
        try:
            await self.active_connections[username].send_text(message)
            state = self.connection_states.get(username)
            if state:
                state.update_activity()
            return True
        except WebSocketDisconnect:
            await self.disconnect(username, reason="connection_lost")
            return False
        except Exception as e:
            logger.error(f"Error sending message to {username}: {str(e)}")
            return False

    def get_active_users(self) -> Dict[str, Dict]:
        return {
            username: {
                'connected_at': state.connected_at.isoformat(),
                'last_activity': state.last_activity.isoformat(),
                'is_admin': state.is_admin,
                'status': state.status.value,
                'reconnect_attempts': state.reconnect_attempts
            }
            for username, state in self.connection_states.items()
        }

    def get_connection_status(self, username: str) -> Optional[Dict]:
        """Get detailed connection status for a user."""
        state = self.connection_states.get(username)
        if not state:
            return None
        
        return {
            'username': state.username,
            'status': state.status.value,
            'connected_at': state.connected_at.isoformat(),
            'last_activity': state.last_activity.isoformat(),
            'reconnect_attempts': state.reconnect_attempts,
            'can_reconnect': state.can_reconnect(),
            'is_admin': state.is_admin,
            'metadata': state.metadata
        }

    def cleanup_inactive_sessions(self, max_inactive_time: int = 3600) -> None:
        now = datetime.utcnow()
        inactive = [
            username for username, state in self.connection_states.items()
            if (now - state.last_activity).total_seconds() > max_inactive_time
        ]
        for username in inactive:
            asyncio.create_task(self.disconnect(username, reason="inactivity"))

class WebSocketHandler:
    def __init__(self, manager: WebSocketManager):
        self.manager = manager

    async def handle_connection(self, websocket: WebSocket, token: str) -> None:
        if not await self.manager.connect(websocket, token):
            await websocket.close(code=1008)  # Policy Violation
            return
        
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get('type') == 'ping':
                        await websocket.send_json({'type': 'pong'})
                    elif message.get('type') == 'heartbeat':
                        await websocket.send_json({'type': 'heartbeat_ack'})
                    else:
                        await self.manager.broadcast(json.dumps(message))
                        
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {data}")
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {str(e)}")
        finally:
            try:
                data = jwt.decode(
                    token,
                    os.getenv('JWT_SECRET'),
                    algorithms=['HS256']
                )
                await self.manager.disconnect(data['username'], reason="connection_closed")
            except Exception as e:
                logger.error(f"Error disconnecting user: {e}")
                pass 