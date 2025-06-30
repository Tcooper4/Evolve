import json
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from automation.notifications.notification_manager import NotificationManager
from system.infra.agents.auth.user_manager import UserManager
import os
import jwt
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

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

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, token: str) -> bool:
        try:
            data = jwt.decode(
                token,
                os.getenv('JWT_SECRET'),
                algorithms=['HS256']
            )
            username = data['username']
            self.active_connections[username] = websocket
            self.user_sessions[username] = {
                'connected_at': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'is_admin': data.get('is_admin', False)
            }
            await websocket.accept()
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

    def disconnect(self, username: str) -> None:
        if username in self.active_connections:
            del self.active_connections[username]
            del self.user_sessions[username]
            logger.info(f"User {username} disconnected")

    async def broadcast(self, message: str) -> None:
        disconnected = []
        for username, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
                self.user_sessions[username]['last_activity'] = datetime.utcnow()
            except WebSocketDisconnect:
                disconnected.append(username)
            except Exception as e:
                logger.error(f"Error broadcasting to {username}: {str(e)}")
                disconnected.append(username)
        for username in disconnected:
            self.disconnect(username)

    async def send_personal_message(self, username: str, message: str) -> bool:
        if username not in self.active_connections:
            logger.warning(f"Attempt to send message to disconnected user {username}")
            return False
        try:
            await self.active_connections[username].send_text(message)
            self.user_sessions[username]['last_activity'] = datetime.utcnow()
            return True
        except WebSocketDisconnect:
            self.disconnect(username)
            return False
        except Exception as e:
            logger.error(f"Error sending message to {username}: {str(e)}")
            return False

    def get_active_users(self) -> Dict[str, Dict]:
        return {
            username: {
                'connected_at': session['connected_at'].isoformat(),
                'last_activity': session['last_activity'].isoformat(),
                'is_admin': session['is_admin']
            }
            for username, session in self.user_sessions.items()
        }

    def cleanup_inactive_sessions(self, max_inactive_time: int = 3600) -> None:
        now = datetime.utcnow()
        inactive = [
            username for username, session in self.user_sessions.items()
            if (now - session['last_activity']).total_seconds() > max_inactive_time
        ]
        for username in inactive:
            self.disconnect(username)
            logger.info(f"Cleaned up inactive session for user {username}")

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
                    if message.get('type') == 'ping':
                        await websocket.send_json({'type': 'pong'})
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
                self.manager.disconnect(data['username'])
            except Exception as e:
                logger.error(f"Error disconnecting user: {e}")
                pass 