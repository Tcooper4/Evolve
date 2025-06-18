"""
Test suite for websocket functionality.

This module contains test cases for:
- WebSocket connection handling
- Message broadcasting
- Error handling
- Connection management
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
import websockets
from typing import List, Dict, Any

from web.websocket import WebSocketManager, WebSocketConnection
from logs.automation_logging import AutomationLogger

@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return Mock(spec=AutomationLogger)

@pytest.fixture
def websocket_manager(mock_logger):
    """Create a WebSocketManager instance with mock logger."""
    return WebSocketManager(logger=mock_logger)

@pytest.fixture
def mock_websocket():
    """Create a mock websocket connection."""
    return AsyncMock(spec=websockets.WebSocketServerProtocol)

@pytest.mark.asyncio
async def test_websocket_connection(websocket_manager, mock_websocket):
    """Test websocket connection handling."""
    # Mock connection
    connection = WebSocketConnection(mock_websocket, "test_client")
    
    # Add connection
    websocket_manager.add_connection(connection)
    
    # Verify connection was added
    assert connection in websocket_manager.connections
    assert connection.client_id == "test_client"
    
    # Remove connection
    websocket_manager.remove_connection(connection)
    
    # Verify connection was removed
    assert connection not in websocket_manager.connections

@pytest.mark.asyncio
async def test_message_broadcasting(websocket_manager, mock_websocket):
    """Test message broadcasting."""
    # Create test connections
    connections = [
        WebSocketConnection(mock_websocket, f"client_{i}")
        for i in range(3)
    ]
    
    # Add connections
    for conn in connections:
        websocket_manager.add_connection(conn)
    
    # Create test message
    message = {
        "type": "test",
        "data": "test message"
    }
    
    # Broadcast message
    await websocket_manager.broadcast_message(message)
    
    # Verify message was sent to all connections
    for conn in connections:
        conn.websocket.send.assert_called_with(json.dumps(message))

@pytest.mark.asyncio
async def test_message_broadcasting_with_filter(websocket_manager, mock_websocket):
    """Test message broadcasting with filter."""
    # Create test connections
    connections = [
        WebSocketConnection(mock_websocket, f"client_{i}")
        for i in range(3)
    ]
    
    # Add connections with different client types
    for i, conn in enumerate(connections):
        conn.client_type = "type_a" if i % 2 == 0 else "type_b"
        websocket_manager.add_connection(conn)
    
    # Create test message
    message = {
        "type": "test",
        "data": "test message"
    }
    
    # Broadcast message to type_a clients
    await websocket_manager.broadcast_message(
        message,
        filter_func=lambda conn: conn.client_type == "type_a"
    )
    
    # Verify message was sent only to type_a connections
    for i, conn in enumerate(connections):
        if i % 2 == 0:
            conn.websocket.send.assert_called_with(json.dumps(message))
        else:
            conn.websocket.send.assert_not_called()

@pytest.mark.asyncio
async def test_connection_error_handling(websocket_manager, mock_websocket):
    """Test connection error handling."""
    # Create connection
    connection = WebSocketConnection(mock_websocket, "test_client")
    websocket_manager.add_connection(connection)
    
    # Mock websocket error
    mock_websocket.send.side_effect = websockets.exceptions.ConnectionClosed(
        code=1000,
        reason="Test error"
    )
    
    # Try to send message
    message = {"type": "test", "data": "test message"}
    await websocket_manager.send_message(connection, message)
    
    # Verify connection was removed
    assert connection not in websocket_manager.connections
    
    # Verify error was logged
    websocket_manager.logger.error.assert_called_with(
        "WebSocket connection error",
        client_id=connection.client_id,
        error="Test error"
    )

@pytest.mark.asyncio
async def test_connection_heartbeat(websocket_manager, mock_websocket):
    """Test connection heartbeat."""
    # Create connection
    connection = WebSocketConnection(mock_websocket, "test_client")
    websocket_manager.add_connection(connection)
    
    # Start heartbeat
    heartbeat_task = asyncio.create_task(
        websocket_manager.start_heartbeat(connection)
    )
    
    # Wait for heartbeat
    await asyncio.sleep(0.1)
    
    # Verify heartbeat was sent
    mock_websocket.send.assert_called_with(
        json.dumps({"type": "heartbeat"})
    )
    
    # Cleanup
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

@pytest.mark.asyncio
async def test_connection_authentication(websocket_manager, mock_websocket):
    """Test connection authentication."""
    # Create connection
    connection = WebSocketConnection(mock_websocket, "test_client")
    
    # Mock authentication message
    auth_message = {
        "type": "auth",
        "token": "test_token"
    }
    mock_websocket.recv.return_value = json.dumps(auth_message)
    
    # Authenticate connection
    authenticated = await websocket_manager.authenticate_connection(connection)
    
    # Verify authentication
    assert authenticated
    assert connection.authenticated
    
    # Verify authentication was logged
    websocket_manager.logger.info.assert_called_with(
        "WebSocket connection authenticated",
        client_id=connection.client_id
    )

@pytest.mark.asyncio
async def test_connection_authentication_failure(websocket_manager, mock_websocket):
    """Test connection authentication failure."""
    # Create connection
    connection = WebSocketConnection(mock_websocket, "test_client")
    
    # Mock invalid authentication message
    auth_message = {
        "type": "auth",
        "token": "invalid_token"
    }
    mock_websocket.recv.return_value = json.dumps(auth_message)
    
    # Try to authenticate connection
    authenticated = await websocket_manager.authenticate_connection(connection)
    
    # Verify authentication failed
    assert not authenticated
    assert not connection.authenticated
    
    # Verify failure was logged
    websocket_manager.logger.warning.assert_called_with(
        "WebSocket authentication failed",
        client_id=connection.client_id
    )

@pytest.mark.asyncio
async def test_connection_message_handling(websocket_manager, mock_websocket):
    """Test connection message handling."""
    # Create connection
    connection = WebSocketConnection(mock_websocket, "test_client")
    websocket_manager.add_connection(connection)
    
    # Mock message handler
    message_handler = AsyncMock()
    websocket_manager.register_message_handler("test", message_handler)
    
    # Mock incoming message
    message = {
        "type": "test",
        "data": "test message"
    }
    mock_websocket.recv.return_value = json.dumps(message)
    
    # Handle message
    await websocket_manager.handle_message(connection)
    
    # Verify message was handled
    message_handler.assert_called_with(connection, message["data"])
    
    # Verify handling was logged
    websocket_manager.logger.info.assert_called_with(
        "WebSocket message handled",
        client_id=connection.client_id,
        message_type="test"
    )

@pytest.mark.asyncio
async def test_connection_cleanup(websocket_manager, mock_websocket):
    """Test connection cleanup."""
    # Create connection
    connection = WebSocketConnection(mock_websocket, "test_client")
    websocket_manager.add_connection(connection)
    
    # Cleanup connection
    await websocket_manager.cleanup_connection(connection)
    
    # Verify connection was removed
    assert connection not in websocket_manager.connections
    
    # Verify cleanup was logged
    websocket_manager.logger.info.assert_called_with(
        "WebSocket connection cleaned up",
        client_id=connection.client_id
    ) 