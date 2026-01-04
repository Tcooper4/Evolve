"""
Launch Agent API Service

Provides REST API endpoints for:
- Agent status and health
- Triggering agent actions
- Retrieving agent results
- Managing agent configurations
"""

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

def main():
    """Launch the Agent API service."""
    
    logger.info("="*80)
    logger.info("Starting Agent API Service...")
    logger.info("="*80)
    
    try:
        from trading.services.agent_api_service import AgentAPIService
        
        # Initialize service
        api_service = AgentAPIService(
            host='0.0.0.0',
            port=8000,
            debug=False
        )
        
        # Start service
        logger.info("Agent API Service initialized successfully")
        logger.info("Starting server on http://0.0.0.0:8000")
        logger.info("API documentation available at http://localhost:8000/docs")
        logger.info("Press Ctrl+C to stop the server")
        
        api_service.start()
        
    except ImportError as e:
        logger.error(f"Failed to import AgentAPIService: {e}")
        logger.error("Make sure trading.services.agent_api_service is available")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting API service: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nShutting down API service...")
        sys.exit(0)

