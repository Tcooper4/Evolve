# Web Module

This module provides the web interface and WebSocket functionality for the agentic forecasting platform.

## Features

- WebSocket server
- Real-time updates
- Client connection management
- Message broadcasting
- Error handling
- Connection authentication

## Structure

- `websocket.py`: WebSocket server implementation
- `static/`: Static assets (JS, CSS, images)
- `templates/`: HTML templates
- `dashboard.js`: Dashboard functionality
- `websocket_client.js`: WebSocket client implementation

## Usage

```python
from web import WebSocketManager

# Initialize WebSocket manager
ws_manager = WebSocketManager()

# Start WebSocket server
await ws_manager.start_server("localhost", 8765)

# Broadcast message
await ws_manager.broadcast_message({
    "type": "update",
    "data": {...}
})
```

## Client Usage

```javascript
// Connect to WebSocket server
const ws = new WebSocketClient('ws://localhost:8765');

// Handle messages
ws.onMessage((message) => {
    console.log('Received:', message);
});

// Send message
ws.send({
    type: 'command',
    data: {...}
});
```

## Testing

Run web tests:
```bash
pytest tests/web/
```

## Security

- WebSocket authentication
- Message validation
- Connection rate limiting
- Error handling
- Secure WebSocket (WSS)

## Configuration

Web module can be configured via:
- Environment variables
- Configuration files
- Command-line arguments 