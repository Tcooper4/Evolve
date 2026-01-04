"""
WebSocket Client for Streamlit Pages

Connects to WebSocket server for real-time updates.
"""

import asyncio
import json
import logging
from typing import Callable, Optional, Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Try to import websockets library
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets library not available. Install with: pip install websockets")


class WebSocketClient:
    """WebSocket client for real-time updates."""
    
    def __init__(self, url: str = "ws://localhost:8001"):
        """Initialize WebSocket client.
        
        Args:
            url: WebSocket server URL
        """
        self.url = url
        self.connection: Optional[Any] = None
        self.callbacks: Dict[str, Callable] = {}
        self._listening = False
        self._listen_task: Optional[asyncio.Task] = None
    
    async def connect(self):
        """Connect to WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library not available. Install with: pip install websockets")
        
        try:
            self.connection = await websockets.connect(self.url)
            logger.info(f"Connected to WebSocket server at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        self._listening = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Disconnected from WebSocket server")
    
    def on(self, event: str, callback: Callable):
        """Register callback for event.
        
        Args:
            event: Event name to listen for
            callback: Function to call when event is received
        """
        self.callbacks[event] = callback
        logger.debug(f"Registered callback for event: {event}")
    
    async def listen(self):
        """Listen for messages from server."""
        if not self.connection:
            raise RuntimeError("Not connected to WebSocket server")
        
        self._listening = True
        
        try:
            async for message in self.connection:
                if not self._listening:
                    break
                
                try:
                    data = json.loads(message)
                    
                    event = data.get('event')
                    payload = data.get('payload', data.get('data', {}))
                    
                    # Call registered callback
                    if event in self.callbacks:
                        try:
                            self.callbacks[event](payload)
                        except Exception as e:
                            logger.error(f"Error in callback for event {event}: {e}")
                    else:
                        logger.debug(f"Unhandled event: {event}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._listening = False
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            self._listening = False
    
    def start_listening(self):
        """Start listening in background task."""
        if not self.connection:
            raise RuntimeError("Not connected to WebSocket server")
        
        if self._listen_task is None or self._listen_task.done():
            self._listen_task = asyncio.create_task(self.listen())
    
    async def send(self, event: str, data: Dict[str, Any]):
        """Send message to server.
        
        Args:
            event: Event name
            data: Event data payload
        """
        if not self.connection:
            raise RuntimeError("Not connected to WebSocket server")
        
        message = json.dumps({
            'event': event,
            'data': data
        })
        
        try:
            await self.connection.send(message)
            logger.debug(f"Sent event '{event}' to WebSocket server")
        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
            raise
    
    def is_connected(self) -> bool:
        """Check if connected to WebSocket server."""
        return self.connection is not None and not self.connection.closed
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status.
        
        Returns:
            Dictionary with connection status and info
        """
        return {
            'connected': self.is_connected(),
            'url': self.url,
            'listening': self._listening,
            'registered_events': list(self.callbacks.keys())
        }

