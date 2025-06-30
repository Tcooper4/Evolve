"""
Script to consolidate duplicate optimizer files and update imports.
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
        logging.FileHandler('consolidate_optimizers.log'),
        logging.StreamHandler()
    ]
)

def consolidate_optimizers():
    """Consolidate duplicate optimizer files into trading/optimization."""
    # Define paths
    root_dir = Path(".")
    optimize_dir = root_dir / "optimize"
    trading_optimization_dir = root_dir / "trading" / "optimization"
    
    # Create backup
    backup_dir = root_dir / "backup" / f"optimizer_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Backup optimize directory
        if optimize_dir.exists():
            shutil.copytree(optimize_dir, backup_dir / "optimize")
            logging.info(f"Backed up optimize directory to {backup_dir}")
        
        # Move rsi_optimizer.py if it exists
        rsi_optimizer = optimize_dir / "rsi_optimizer.py"
        if rsi_optimizer.exists():
            # Compare with existing file
            trading_rsi_optimizer = trading_optimization_dir / "rsi_optimizer.py"
            if trading_rsi_optimizer.exists():
                # Add deprecation notice to old file
                with open(rsi_optimizer, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                deprecation_notice = f'''"""
                    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
DEPRECATED: This file has been consolidated into {trading_rsi_optimizer}
Please use the consolidated version instead.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

'''
                with open(rsi_optimizer, 'w', encoding='utf-8') as f:
                    f.write(deprecation_notice + content)
                
                logging.info(f"Marked {rsi_optimizer} as deprecated")
            else:
                # Move file to trading/optimization
                shutil.move(str(rsi_optimizer), str(trading_optimization_dir))
                logging.info(f"Moved {rsi_optimizer} to {trading_optimization_dir}")
        
        # Remove optimize directory if empty
        if optimize_dir.exists() and not any(optimize_dir.iterdir()):
            optimize_dir.rmdir()
            logging.info(f"Removed empty optimize directory")
        
        # Update imports in all Python files
        for py_file in root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update imports
                content = content.replace(
                    "from trading.optimization.rsi_optimizer import",
                    "from trading.optimization.rsi_optimizer import"
                )
                content = content.replace(
                    "import trading.optimization.rsi_optimizer",
                    "import trading.optimization.rsi_optimizer"
                )
                
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logging.info(f"Updated imports in {py_file}")
            except Exception as e:
                logging.error(f"Error updating imports in {py_file}: {e}")
        
        logging.info("Successfully consolidated optimizer files")
        
    except Exception as e:
        logging.error(f"Error during consolidation: {e}")
        # Restore from backup
        if backup_dir.exists():
            shutil.copytree(backup_dir / "optimize", optimize_dir)
            logging.info("Restored from backup due to error")

if __name__ == "__main__":
    consolidate_optimizers() 