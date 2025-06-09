import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import WebSocket, WebSocketDisconnect
import json
from automation.web.websocket import WebSocketManager, WebSocketHandler
from automation.notifications.notification_manager import NotificationManager

@pytest.fixture
def mock_websocket():
    websocket = Mock(spec=WebSocket)
    websocket.accept = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.receive_text = AsyncMock()
    return websocket

@pytest.fixture
def mock_notification_manager():
    manager = Mock(spec=NotificationManager)
    manager.subscribe = AsyncMock()
    manager.unsubscribe = AsyncMock()
    return manager

@pytest.fixture
def websocket_manager(mock_notification_manager):
    return WebSocketManager(mock_notification_manager)

@pytest.fixture
def websocket_handler(websocket_manager):
    return WebSocketHandler(websocket_manager)

@pytest.mark.asyncio
async def test_websocket_connect(websocket_manager, mock_websocket):
    # Test data
    user_id = "user1"
    
    # Connect WebSocket
    await websocket_manager.connect(mock_websocket, user_id)
    
    # Verify connection
    assert user_id in websocket_manager.active_connections
    assert mock_websocket in websocket_manager.active_connections[user_id]
    mock_websocket.accept.assert_called_once()
    websocket_manager.notification_manager.subscribe.assert_called_once_with(
        user_id,
        websocket_manager._notification_callback
    )

@pytest.mark.asyncio
async def test_websocket_disconnect(websocket_manager, mock_websocket):
    # Test data
    user_id = "user1"
    
    # Connect and then disconnect
    await websocket_manager.connect(mock_websocket, user_id)
    await websocket_manager.disconnect(mock_websocket, user_id)
    
    # Verify disconnection
    assert user_id not in websocket_manager.active_connections
    websocket_manager.notification_manager.unsubscribe.assert_called_once_with(user_id)

@pytest.mark.asyncio
async def test_notification_callback(websocket_manager, mock_websocket):
    # Test data
    user_id = "user1"
    notification_data = {
        "user_id": user_id,
        "title": "Test Notification",
        "message": "This is a test"
    }
    
    # Connect WebSocket
    await websocket_manager.connect(mock_websocket, user_id)
    
    # Send notification
    await websocket_manager._notification_callback(notification_data)
    
    # Verify notification was sent
    mock_websocket.send_json.assert_called_once_with(notification_data)

@pytest.mark.asyncio
async def test_broadcast_to_user(websocket_manager, mock_websocket):
    # Test data
    user_id = "user1"
    message = {"type": "test", "data": "test message"}
    
    # Connect WebSocket
    await websocket_manager.connect(mock_websocket, user_id)
    
    # Broadcast message
    await websocket_manager.broadcast(message, user_id)
    
    # Verify message was sent
    mock_websocket.send_json.assert_called_once_with(message)

@pytest.mark.asyncio
async def test_broadcast_to_all(websocket_manager, mock_websocket):
    # Test data
    user_id = "user1"
    message = {"type": "test", "data": "test message"}
    
    # Connect WebSocket
    await websocket_manager.connect(mock_websocket, user_id)
    
    # Broadcast message to all
    await websocket_manager.broadcast(message)
    
    # Verify message was sent
    mock_websocket.send_json.assert_called_once_with(message)

@pytest.mark.asyncio
async def test_websocket_handler(websocket_handler, mock_websocket):
    # Test data
    user_id = "user1"
    ping_message = {"type": "ping"}
    
    # Mock WebSocket behavior
    mock_websocket.receive_text.return_value = json.dumps(ping_message)
    
    # Start handler
    handler_task = asyncio.create_task(websocket_handler.handle_websocket(mock_websocket, user_id))
    
    # Wait for a short time to process messages
    await asyncio.sleep(0.1)
    
    # Cancel handler
    handler_task.cancel()
    try:
        await handler_task
    except asyncio.CancelledError:
        pass
    
    # Verify WebSocket was connected and ping was handled
    mock_websocket.accept.assert_called_once()
    mock_websocket.send_json.assert_called_once_with({"type": "pong"})

@pytest.mark.asyncio
async def test_websocket_disconnect_handling(websocket_handler, mock_websocket):
    # Test data
    user_id = "user1"
    
    # Mock WebSocket disconnect
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()
    
    # Start handler
    await websocket_handler.handle_websocket(mock_websocket, user_id)
    
    # Verify WebSocket was connected and then disconnected
    mock_websocket.accept.assert_called_once()
    assert user_id not in websocket_handler.websocket_manager.active_connections

@pytest.mark.asyncio
async def test_multiple_connections(websocket_manager):
    # Test data
    user_id = "user1"
    mock_websocket1 = Mock(spec=WebSocket)
    mock_websocket1.accept = AsyncMock()
    mock_websocket1.send_json = AsyncMock()
    
    mock_websocket2 = Mock(spec=WebSocket)
    mock_websocket2.accept = AsyncMock()
    mock_websocket2.send_json = AsyncMock()
    
    # Connect multiple WebSockets
    await websocket_manager.connect(mock_websocket1, user_id)
    await websocket_manager.connect(mock_websocket2, user_id)
    
    # Verify both connections are active
    assert len(websocket_manager.active_connections[user_id]) == 2
    
    # Disconnect one WebSocket
    await websocket_manager.disconnect(mock_websocket1, user_id)
    
    # Verify one connection remains
    assert len(websocket_manager.active_connections[user_id]) == 1
    assert mock_websocket2 in websocket_manager.active_connections[user_id]

@pytest.mark.asyncio
async def test_error_handling(websocket_manager, mock_websocket):
    # Test data
    user_id = "user1"
    notification_data = {
        "user_id": user_id,
        "title": "Test Notification",
        "message": "This is a test"
    }
    
    # Connect WebSocket
    await websocket_manager.connect(mock_websocket, user_id)
    
    # Mock send_json to raise an exception
    mock_websocket.send_json.side_effect = Exception("Send error")
    
    # Send notification
    await websocket_manager._notification_callback(notification_data)
    
    # Verify WebSocket was disconnected
    assert user_id not in websocket_manager.active_connections 