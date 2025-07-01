#!/usr/bin/env python3
"""
Launch Script for Agent API Service

Starts the Agent API Service with WebSocket support for real-time agent updates.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the trading directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.services.agent_api_service import AgentAPIService
from trading.utils.logging_utils import log_manager

def setup_logging():
    """Set up logging for the launch script."""
    log_path = Path("logs/api")
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / "agent_api_launch.log"),
            logging.StreamHandler()
        ]
    )

async def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Agent API Service...")
        
        # Initialize and start the service
        service = AgentAPIService()
        
        logger.info("Agent API Service initialized successfully")
        logger.info("Available endpoints:")
        logger.info("  - REST API: http://localhost:8001")
        logger.info("  - API Docs: http://localhost:8001/docs")
        logger.info("  - WebSocket: ws://localhost:8001/ws")
        logger.info("  - WebSocket Test: http://localhost:8001/ws/test")
        
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("Agent API Service interrupted by user")
    except Exception as e:
        logger.error(f"Error starting Agent API Service: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 