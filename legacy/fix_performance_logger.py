#!/usr/bin/env python3
"""
Script to fix the PerformanceLogger __init__ method by removing the return None statement.
"""

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_performance_logger():
    """Fix the PerformanceLogger __init__ method."""
    file_path = "trading/optimization/performance_logger.py"
    
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found")
        return
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic line
    old_content = '        logger.info(f"Initialized PerformanceLogger with log directory: {log_dir}")\n        return None'
    new_content = '        logger.info(f"Initialized PerformanceLogger with log directory: {log_dir}")'
    
    if old_content in content:
        content = content.replace(old_content, new_content)
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.info("Fixed PerformanceLogger __init__ method")
    else:
        logger.warning("Could not find the problematic line to fix")

if __name__ == "__main__":
    fix_performance_logger() 