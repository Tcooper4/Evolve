"""
WebSocket Service for Real-time Agent Updates

Provides WebSocket endpoints for real-time agent status, metrics, and execution updates.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from trading.agents.agent_manager import AgentManager
from trading.utils.logging_utils import log_manager

logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: str

class WebSocketService:
    """Manages WebSocket connections for real-time agent updates."""
    
    def __init__(self, agent_manager: AgentManager):
        """Initialize WebSocket service."""
        self.agent_manager = agent_manager
        self.active_connections: Set[WebSocket] = set()
        self.connection_subscriptions: Dict[WebSocket, List[str]] = {}
        
        # Message types
        self.message_types = {
            'agent_status': self._handle_agent_status,
            'agent_metrics': self._handle_agent_metrics,
            'agent_execution': self._handle_agent_execution,
            'system_status': self._handle_system_status,
            'subscribe': self._handle_subscribe,
            'unsubscribe': self._handle_unsubscribe
        }
        
        logger.info("WebSocket Service initialized")
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_subscriptions[websocket] = []
        
        # Send welcome message
        await self.send_personal_message(websocket, {
            'type': 'connection_established',
            'data': {
                'message': 'WebSocket connection established',
                'available_types': list(self.message_types.keys())
            },
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection."""
        self.active_connections.remove(websocket)
        if websocket in self.connection_subscriptions:
            del self.connection_subscriptions[websocket]
        
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Send message to specific WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any], message_type: str = None) -> None:
        """Broadcast message to all connected clients."""
        if message_type:
            message['type'] = message_type
        
        message['timestamp'] = datetime.now().isoformat()
        
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def broadcast_to_subscribers(self, message: Dict[str, Any], message_type: str) -> None:
        """Broadcast message to subscribers of specific message type."""
        message['type'] = message_type
        message['timestamp'] = datetime.now().isoformat()
        
        disconnected = set()
        for connection, subscriptions in self.connection_subscriptions.items():
            if message_type in subscriptions:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to subscribers: {str(e)}")
                    disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def handle_message(self, websocket: WebSocket, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self.message_types:
                await self.message_types[message_type](websocket, data)
            else:
                await self.send_personal_message(websocket, {
                    'type': 'error',
                    'data': {
                        'message': f'Unknown message type: {message_type}',
                        'available_types': list(self.message_types.keys())
                    },
                    'timestamp': datetime.now().isoformat()
                })
        
        except json.JSONDecodeError:
            await self.send_personal_message(websocket, {
                'type': 'error',
                'data': {'message': 'Invalid JSON format'},
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
            await self.send_personal_message(websocket, {
                'type': 'error',
                'data': {'message': f'Internal server error: {str(e)}'},
                'timestamp': datetime.now().isoformat()
            })
    
    async def _handle_agent_status(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Handle agent status request."""
        agent_id = data.get('data', {}).get('agent_id')
        
        if agent_id:
            agent = self.agent_manager.get_agent(agent_id)
            if agent:
                await self.send_personal_message(websocket, {
                    'type': 'agent_status',
                    'data': {
                        'agent_id': agent_id,
                        'status': agent.status,
                        'capabilities': agent.capabilities,
                        'last_execution': agent.last_execution.isoformat() if hasattr(agent, 'last_execution') and agent.last_execution else None
                    },
                    'timestamp': datetime.now().isoformat()
                })
            else:
                await self.send_personal_message(websocket, {
                    'type': 'error',
                    'data': {'message': f'Agent not found: {agent_id}'},
                    'timestamp': datetime.now().isoformat()
                })
        else:
            # Send status of all agents
            agents = self.agent_manager.get_all_agents()
            agent_statuses = []
            for agent_id, agent in agents.items():
                agent_statuses.append({
                    'agent_id': agent_id,
                    'status': agent.status,
                    'capabilities': agent.capabilities,
                    'last_execution': agent.last_execution.isoformat() if hasattr(agent, 'last_execution') and agent.last_execution else None
                })
            
            await self.send_personal_message(websocket, {
                'type': 'agent_status',
                'data': {'agents': agent_statuses},
                'timestamp': datetime.now().isoformat()
            })
    
    async def _handle_agent_metrics(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Handle agent metrics request."""
        agent_id = data.get('data', {}).get('agent_id')
        
        if agent_id:
            agent = self.agent_manager.get_agent(agent_id)
            if agent and hasattr(agent, 'get_performance_metrics'):
                metrics = agent.get_performance_metrics()
                await self.send_personal_message(websocket, {
                    'type': 'agent_metrics',
                    'data': {
                        'agent_id': agent_id,
                        'metrics': metrics
                    },
                    'timestamp': datetime.now().isoformat()
                })
            else:
                await self.send_personal_message(websocket, {
                    'type': 'error',
                    'data': {'message': f'Agent not found or no metrics available: {agent_id}'},
                    'timestamp': datetime.now().isoformat()
                })
    
    async def _handle_agent_execution(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Handle agent execution request."""
        agent_id = data.get('data', {}).get('agent_id')
        task_data = data.get('data', {}).get('task_data', {})
        
        if agent_id:
            agent = self.agent_manager.get_agent(agent_id)
            if agent:
                try:
                    # Execute agent asynchronously
                    result = await agent.execute(task_data)
                    await self.send_personal_message(websocket, {
                        'type': 'agent_execution',
                        'data': {
                            'agent_id': agent_id,
                            'result': result,
                            'status': 'completed'
                        },
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    await self.send_personal_message(websocket, {
                        'type': 'agent_execution',
                        'data': {
                            'agent_id': agent_id,
                            'error': str(e),
                            'status': 'failed'
                        },
                        'timestamp': datetime.now().isoformat()
                    })
            else:
                await self.send_personal_message(websocket, {
                    'type': 'error',
                    'data': {'message': f'Agent not found: {agent_id}'},
                    'timestamp': datetime.now().isoformat()
                })
    
    async def _handle_system_status(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Handle system status request."""
        agents = self.agent_manager.get_all_agents()
        active_agents = [a for a in agents.values() if a.status == "active"]
        
        await self.send_personal_message(websocket, {
            'type': 'system_status',
            'data': {
                'total_agents': len(agents),
                'active_agents': len(active_agents),
                'system_status': 'healthy' if len(agents) < 100 else 'at_capacity',
                'connection_count': len(self.active_connections)
            },
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_subscribe(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Handle subscription request."""
        message_types = data.get('data', {}).get('message_types', [])
        
        if websocket in self.connection_subscriptions:
            for msg_type in message_types:
                if msg_type not in self.connection_subscriptions[websocket]:
                    self.connection_subscriptions[websocket].append(msg_type)
        
        await self.send_personal_message(websocket, {
            'type': 'subscription_confirmed',
            'data': {
                'message_types': message_types,
                'current_subscriptions': self.connection_subscriptions.get(websocket, [])
            },
            'timestamp': datetime.now().isoformat()
        })
    
    async def _handle_unsubscribe(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        """Handle unsubscription request."""
        message_types = data.get('data', {}).get('message_types', [])
        
        if websocket in self.connection_subscriptions:
            for msg_type in message_types:
                if msg_type in self.connection_subscriptions[websocket]:
                    self.connection_subscriptions[websocket].remove(msg_type)
        
        await self.send_personal_message(websocket, {
            'type': 'unsubscription_confirmed',
            'data': {
                'message_types': message_types,
                'current_subscriptions': self.connection_subscriptions.get(websocket, [])
            },
            'timestamp': datetime.now().isoformat()
        })
    
    async def broadcast_agent_update(self, agent_id: str, update_type: str, data: Dict[str, Any]) -> None:
        """Broadcast agent update to subscribers."""
        message = {
            'type': f'agent_{update_type}',
            'data': {
                'agent_id': agent_id,
                **data
            },
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_to_subscribers(message, f'agent_{update_type}')
    
    async def broadcast_system_update(self, update_type: str, data: Dict[str, Any]) -> None:
        """Broadcast system update to subscribers."""
        message = {
            'type': f'system_{update_type}',
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.broadcast_to_subscribers(message, f'system_{update_type}')
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            'total_connections': len(self.active_connections),
            'subscription_counts': {
                msg_type: sum(1 for subs in self.connection_subscriptions.values() if msg_type in subs)
                for msg_type in self.message_types.keys()
            },
            'timestamp': datetime.now().isoformat()
        } 