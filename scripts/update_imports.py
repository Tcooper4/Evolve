"""Script to update imports after restructuring."""

import os
import re
from pathlib import Path
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Files to update
FILES_TO_UPDATE: List[str] = [
    'tests/test_edge_cases.py',
    'tests/test_performance.py',
    'tests/test_real_world_scenario.py',
    'trading/utils/system_startup.py',
    'pages/settings.py',
    'pages/forecast.py',
    'core/router.py',
    'automate/daily_scheduler.py'
]

# Import replacements
IMPORT_REPLACEMENTS: List[Tuple[str, str]] = [
    (r'from trading\.agents\.router import AgentRouter',
     'from core.agents.router import AgentRouter'),
    (r'from trading\.agents\.self_improving_agent import SelfImprovingAgent',
     'from core.agents.self_improving_agent import SelfImprovingAgent')
]

def update_imports(file_path: str) -> None:
    """Update imports in a file.
    
    Args:
        file_path: Path to the file to update
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        
        for old_import, new_import in IMPORT_REPLACEMENTS:
            content = re.sub(old_import, new_import, content)
            
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Updated imports in {file_path}")
        else:
            logger.info(f"No changes needed in {file_path}")
            
    except Exception as e:
        logger.error(f"Error updating {file_path}: {str(e)}")

def main() -> None:
    """Main function to update imports in all files."""
    for file_path in FILES_TO_UPDATE:
        if os.path.exists(file_path):
            update_imports(file_path)
        else:
            logger.warning(f"File not found: {file_path}")

if __name__ == '__main__':
    main() 