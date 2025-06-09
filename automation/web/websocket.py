import json
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from automation.notifications.notification_manager import NotificationManager

class WebSocketManager:
    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """Connect a new WebSocket client."""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        
        # Subscribe to notifications
        await self.notification_manager.subscribe(user_id, self._notification_callback)
        
        self.logger.info(f"WebSocket client connected for user {user_id}")
    
    async def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """Disconnect a WebSocket client."""
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                await self.notification_manager.unsubscribe(user_id)
        
        self.logger.info(f"WebSocket client disconnected for user {user_id}")
    
    async def _notification_callback(self, data: Dict) -> None:
        """Callback for notification updates."""
        user_id = data.get("user_id")
        if not user_id:
            return
        
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    self.logger.error(f"Error sending notification to WebSocket client: {str(e)}")
                    await self.disconnect(connection, user_id)
    
    async def broadcast(self, message: Dict, user_id: Optional[str] = None) -> None:
        """Broadcast a message to all connected clients or a specific user."""
        if user_id:
            if user_id in self.active_connections:
                for connection in self.active_connections[user_id]:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        self.logger.error(f"Error broadcasting message to WebSocket client: {str(e)}")
                        await self.disconnect(connection, user_id)
        else:
            for user_connections in self.active_connections.values():
                for connection in user_connections:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        self.logger.error(f"Error broadcasting message to WebSocket client: {str(e)}")
                        await self.disconnect(connection, user_id)

class WebSocketHandler:
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.logger = logging.getLogger(__name__)
    
    async def handle_websocket(self, websocket: WebSocket, user_id: str) -> None:
        """Handle WebSocket connection."""
        try:
            await self.websocket_manager.connect(websocket, user_id)
            
            while True:
                try:
                    # Wait for messages from the client
                    data = await websocket.receive_text()
                    
                    # Handle client messages if needed
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    self.logger.error(f"Error handling WebSocket message: {str(e)}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error in WebSocket handler: {str(e)}")
        finally:
            await self.websocket_manager.disconnect(websocket, user_id) 