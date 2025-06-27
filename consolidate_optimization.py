#!/usr/bin/env python3
"""
Optimization Module Consolidation Script

This script consolidates all optimization-related modules into the central
trading/optimization/ package and updates all imports across the codebase.
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main consolidation function."""
    root_dir = Path(".")
    
    # Source directories to consolidate
    source_dirs = {
        'optimizer': root_dir / "optimizer",
        'optimize': root_dir / "optimize", 
        'optimizers': root_dir / "optimizers"
    }
    
    # Target directory
    target_dir = root_dir / "trading" / "optimization"
    
    # Create target structure
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "core").mkdir(exist_ok=True)
    (target_dir / "strategies").mkdir(exist_ok=True)
    (target_dir / "visualization").mkdir(exist_ok=True)
    (target_dir / "utils").mkdir(exist_ok=True)
    
    logger.info("Starting optimization module consolidation...")
    
    # Consolidate optimizer/ directory
    if source_dirs['optimizer'].exists():
        logger.info("Consolidating optimizer/ directory...")
        
        # Move core optimizers
        core_dir = source_dirs['optimizer'] / "core"
        if core_dir.exists():
            for file_path in core_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    target_path = target_dir / "core" / file_path.name
                    if target_path.exists():
                        logger.warning(f"File already exists: {target_path}")
                    else:
                        shutil.copy2(file_path, target_path)
                        logger.info(f"Copied {file_path} to {target_path}")
        
        # Move strategy optimizers
        strategies_dir = source_dirs['optimizer'] / "strategies"
        if strategies_dir.exists():
            for file_path in strategies_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    target_path = target_dir / "strategies" / file_path.name
                    if target_path.exists():
                        logger.warning(f"File already exists: {target_path}")
                    else:
                        shutil.copy2(file_path, target_path)
                        logger.info(f"Copied {file_path} to {target_path}")
        
        # Move visualization
        viz_dir = source_dirs['optimizer'] / "visualization"
        if viz_dir.exists():
            for file_path in viz_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    target_path = target_dir / "visualization" / file_path.name
                    if target_path.exists():
                        logger.warning(f"File already exists: {target_path}")
                    else:
                        shutil.copy2(file_path, target_path)
                        logger.info(f"Copied {file_path} to {target_path}")
    
    # Consolidate optimize/ directory
    if source_dirs['optimize'].exists():
        logger.info("Consolidating optimize/ directory...")
        for file_path in source_dirs['optimize'].glob("*.py"):
            if file_path.name != "__init__.py":
                target_path = target_dir / file_path.name
                if target_path.exists():
                    logger.warning(f"File already exists: {target_path}")
                else:
                    shutil.copy2(file_path, target_path)
                    logger.info(f"Copied {file_path} to {target_path}")
    
    # Consolidate optimizers/ directory
    if source_dirs['optimizers'].exists():
        logger.info("Consolidating optimizers/ directory...")
        for file_path in source_dirs['optimizers'].glob("*.py"):
            if file_path.name == "consolidator.py":
                target_path = target_dir / "utils" / file_path.name
                shutil.copy2(file_path, target_path)
                logger.info(f"Copied {file_path} to {target_path}")
            elif file_path.name != "__init__.py":
                target_path = target_dir / file_path.name
                if target_path.exists():
                    logger.warning(f"File already exists: {target_path}")
                else:
                    shutil.copy2(file_path, target_path)
                    logger.info(f"Copied {file_path} to {target_path}")
    
    # Update imports in all Python files
    logger.info("Updating imports...")
    update_imports(root_dir)
    
    # Clean up source directories
    logger.info("Cleaning up source directories...")
    for name, source_dir in source_dirs.items():
        if source_dir.exists():
            try:
                shutil.rmtree(source_dir)
                logger.info(f"Removed {source_dir}")
            except Exception as e:
                logger.error(f"Error removing {source_dir}: {e}")
    
    logger.info("Consolidation completed!")

def update_imports(root_dir: Path):
    """Update imports in all Python files."""
    import_mappings = {
        "from optimizer.": "from trading.optimization.",
        "import optimizer.": "import trading.optimization.",
        "from optimize.": "from trading.optimization.",
        "import optimize.": "import trading.optimization.",
        "from optimizers.": "from trading.optimization.utils.",
        "import optimizers.": "import trading.optimization.utils.",
    }
    
    updated_count = 0
    for py_file in root_dir.rglob("*.py"):
        if any(part in ["backup", "__pycache__", ".git", ".venv"] for part in py_file.parts):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            for old_import, new_import in import_mappings.items():
                content = content.replace(old_import, new_import)
            
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                updated_count += 1
                logger.info(f"Updated imports in {py_file}")
                
        except Exception as e:
            logger.error(f"Error updating {py_file}: {e}")
    
    logger.info(f"Updated imports in {updated_count} files")

if __name__ == "__main__":
    main() 