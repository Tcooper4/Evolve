"""
Main script to run all cleanup operations.
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_all.log'),
        logging.StreamHandler()
    ]
)

def run_cleanup():
    """Run all cleanup operations."""
    try:
        # Import cleanup scripts
        from consolidate_optimizers import consolidate_optimizers
        from cleanup_agents import cleanup_agents
        from cleanup_optimization import cleanup_optimization
        from repair import CodeRepair
        
        # Create backup directory
        backup_dir = Path(".") / "backup" / f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Run code repair
        logging.info("Running code repair...")
        repair = CodeRepair()
        repair.run()
        
        # Consolidate optimizers
        logging.info("Consolidating optimizers...")
        consolidate_optimizers()
        
        # Cleanup agents
        logging.info("Cleaning up agents...")
        cleanup_agents()
        
        # Cleanup optimization
        logging.info("Cleaning up optimization...")
        cleanup_optimization()
        
        logging.info("All cleanup operations completed successfully")
        
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_cleanup() 