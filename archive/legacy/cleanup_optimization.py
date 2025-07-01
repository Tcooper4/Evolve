"""
Script to handle redundant optimization files and consolidate functionality.
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
        logging.FileHandler('cleanup_optimization.log'),
        logging.StreamHandler()
    ]
)

def cleanup_optimization():
    """Handle redundant optimization files and consolidate functionality."""
    # Define paths
    root_dir = Path(".")
    trading_optimization_dir = root_dir / "trading" / "optimization"
    
    # Create backup
    backup_dir = root_dir / "backup" / f"optimization_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Backup optimization directory
        if trading_optimization_dir.exists():
            shutil.copytree(trading_optimization_dir, backup_dir / "optimization")
            logging.info(f"Backed up optimization directory to {backup_dir}")
        
        # Files to handle
        redundant_files = [
            "optimizer.py",  # Redundant with strategy_optimizer.py
            "sandbox_optim_run.py"  # Development file
        ]
        
        for file_name in redundant_files:
            file_path = trading_optimization_dir / file_name
            if file_path.exists():
                # Add deprecation notice
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                deprecation_notice = f'''"""

DEPRECATED: This file is redundant or for development purposes only.
Please use strategy_optimizer.py for optimization functionality.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

'''
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(deprecation_notice + content)
                
                logging.info(f"Marked {file_path} as deprecated")
        
        # Consolidate optimizer.py functionality into strategy_optimizer.py
        optimizer_file = trading_optimization_dir / "optimizer.py"
        strategy_optimizer_file = trading_optimization_dir / "strategy_optimizer.py"
        
        if optimizer_file.exists() and strategy_optimizer_file.exists():
            with open(optimizer_file, 'r', encoding='utf-8') as f:
                optimizer_content = f.read()
            
            with open(strategy_optimizer_file, 'r', encoding='utf-8') as f:
                strategy_optimizer_content = f.read()
            
            # Add any unique functionality from optimizer.py to strategy_optimizer.py
            # This is a simplified example - you would need to do proper code analysis
            if "class Optimizer" in optimizer_content:
                # Extract the class and its methods
                optimizer_class = optimizer_content[optimizer_content.find("class Optimizer"):]
                optimizer_class = optimizer_class[:optimizer_class.find("\n\n")]
                
                # Add to strategy_optimizer.py
                strategy_optimizer_content += f"\n\n# Merged from optimizer.py\n{optimizer_class}"
                
                with open(strategy_optimizer_file, 'w', encoding='utf-8') as f:
                    f.write(strategy_optimizer_content)
                
                logging.info("Merged optimizer.py functionality into strategy_optimizer.py")
        
        # Update imports in all Python files
        for py_file in root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update imports
                content = content.replace(
                    "from trading.optimization.strategy_optimizer import",
                    "from trading.optimization.strategy_optimizer import"
                )
                content = content.replace(
                    "import trading.optimization.strategy_optimizer",
                    "import trading.optimization.strategy_optimizer"
                )
                
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logging.info(f"Updated imports in {py_file}")
            except Exception as e:
                logging.error(f"Error updating imports in {py_file}: {e}")
        
        logging.info("Successfully cleaned up optimization files")
        
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        # Restore from backup
        if backup_dir.exists():
            shutil.copytree(backup_dir / "optimization", trading_optimization_dir)
            logging.info("Restored from backup due to error")

if __name__ == "__main__":
    cleanup_optimization() 