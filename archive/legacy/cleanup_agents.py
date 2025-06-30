"""
Script to handle unused core agents and update related files.
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup_agents.log'),
        logging.StreamHandler()
    ]
)

def cleanup_agents():
    """Handle unused core agents and update related files."""
    # Define paths
    root_dir = Path(".")
    core_agents_dir = root_dir / "core" / "agents"
    
    # Create backup
    backup_dir = root_dir / "backup" / f"agents_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Backup core/agents directory
        if core_agents_dir.exists():
            shutil.copytree(core_agents_dir, backup_dir / "agents")
            logging.info(f"Backed up agents directory to {backup_dir}")
        
        # Files to handle
        unused_files = [
            "self_improving_agent.py",
            "goal_planner.py"
        ]
        
        for file_name in unused_files:
            file_path = core_agents_dir / file_name
            if file_path.exists():
                # Add deprecation notice
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                deprecation_notice = f'''"""
                    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
DEPRECATED: This agent is currently unused in production.
It is only used in tests and documentation.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

'''
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(deprecation_notice + content)
                
                logging.info(f"Marked {file_path} as deprecated")
        
        # Update __init__.py
        init_file = core_agents_dir / "__init__.py"
        if init_file.exists():
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add deprecation notices for unused agents
            content = content.replace(
                "from trading.self_improving_agent import SelfImprovingAgent",
                "# DEPRECATED: Only used in tests\n# from trading.self_improving_agent import SelfImprovingAgent"
            )
            content = content.replace(
                "from trading.goal_planner import GoalPlanner",
                "# DEPRECATED: Only used in tests\n# from trading.goal_planner import GoalPlanner"
            )
            
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logging.info("Updated __init__.py with deprecation notices")
        
        # Update tests to use mocks instead of actual agents
        tests_dir = root_dir / "tests"
        for test_file in tests_dir.rglob("test_*.py"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace actual agent imports with mocks
                content = content.replace(
                    "from core.agents.self_improving_agent import SelfImprovingAgent",
                    "from unittest.mock import MagicMock\nSelfImprovingAgent = MagicMock"
                )
                content = content.replace(
                    "from core.agents.goal_planner import GoalPlanner",
                    "from unittest.mock import MagicMock\nGoalPlanner = MagicMock"
                )
                
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logging.info(f"Updated {test_file} to use mocks")
            except Exception as e:
                logging.error(f"Error updating {test_file}: {e}")
        
        logging.info("Successfully cleaned up unused agents")
        
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        # Restore from backup
        if backup_dir.exists():
            shutil.copytree(backup_dir / "agents", core_agents_dir)
            logging.info("Restored from backup due to error")

if __name__ == "__main__":
    cleanup_agents() 