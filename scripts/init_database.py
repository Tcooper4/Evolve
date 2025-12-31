#!/usr/bin/env python3
"""
Database Initialization Script

Initializes the database schema and creates all necessary tables.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database."""
    try:
        from trading.database import init_database
        
        logger.info("Initializing database...")
        init_database(create_tables=True)
        logger.info("Database initialized successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

