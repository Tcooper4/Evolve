"""
Launch WebSocket Service for Real-time Updates

Provides real-time updates for:
- Agent status changes
- Forecast completions
- Trade executions
- Risk alerts
- System health
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Launch WebSocket server."""
    
    logger.info("="*80)
    logger.info("Starting WebSocket Server...")
    logger.info("="*80)
    
    try:
        from system.infra.agents.web.websocket import WebSocketServer
        from system.infra.agents.web.middleware import WebSocketMiddleware
        
        # Initialize middleware
        middleware = WebSocketMiddleware()
        
        # Initialize WebSocket server
        ws_server = WebSocketServer(
            host='0.0.0.0',
            port=8001,
            middleware=middleware
        )
        
        # Start server
        logger.info("WebSocket Server initialized successfully")
        logger.info("Starting server on ws://0.0.0.0:8001")
        logger.info("Press Ctrl+C to stop the server")
        
        await ws_server.start()
        
        logger.info("WebSocket Server running on ws://localhost:8001")
        
        # Keep running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down WebSocket Server...")
            await ws_server.stop()
            logger.info("WebSocket Server stopped")
    
    except ImportError as e:
        logger.error(f"Failed to import WebSocket modules: {e}")
        logger.error("Make sure system.infra.agents.web.websocket and middleware are available")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting WebSocket server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutting down WebSocket Server...")
        sys.exit(0)

