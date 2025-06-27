"""
Comprehensive Optimization Module Consolidation Script

This script consolidates all optimization-related modules into a single coherent
trading/optimization/ package and updates all imports across the codebase.

Target structure:
trading/optimization/
├── __init__.py (main entry point)
├── core/ (base classes and core optimizers)
├── strategies/ (strategy-specific optimizers)
├── visualization/ (optimization visualization tools)
├── utils/ (utility functions)
└── configs/ (optimization configurations)
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationConsolidator:
    """Comprehensive optimizer consolidation and import management."""
    
    def __init__(self, root_dir: Optional[str] = None):
        """Initialize the consolidator."""
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        
        # Source directories to consolidate
        self.source_dirs = {
            'optimizer': self.root_dir / "optimizer",
            'optimize': self.root_dir / "optimize",
            'optimizers': self.root_dir / "optimizers"
        }
        
        # Target directory (the main optimization module)
        self.target_dir = self.root_dir / "trading" / "optimization"
        
        # Backup directory
        self.backup_dir = self.root_dir / "backup" / f"optimization_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Track all changes
        self.consolidation_results = {
            'files_moved': [],
            'files_merged': [],
            'imports_updated': [],
            'errors': [],
            'warnings': [],
            'backup_created': None,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_consolidation(self, create_backup: bool = True) -> Dict[str, Any]:
        """Run the complete consolidation process."""
        logger.info("Starting comprehensive optimization module consolidation")
        
        try:
            # Create backup
            if create_backup:
                self._create_backup()
            
            # Ensure target directory structure exists
            self._setup_target_structure()
            
            # Consolidate files from all source directories
            self._consolidate_all_modules()
            
            # Update all imports across the codebase
            self._update_all_imports()
            
            # Clean up source directories
            self._cleanup_source_directories()
            
            # Validate consolidation
            self._validate_consolidation()
            
            self.consolidation_results['success'] = True
            logger.info("Consolidation completed successfully")
            
        except Exception as e:
            error_msg = f"Error during consolidation: {str(e)}"
            logger.error(error_msg)
            self.consolidation_results['errors'].append(error_msg)
            
            # Attempt to restore from backup
            if self.consolidation_results['backup_created']:
                self._restore_from_backup()
        
        return self.consolidation_results
    
    def _create_backup(self) -> None:
        """Create backup of all optimization-related directories."""
        logger.info("Creating backup...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup each source directory
        for name, source_dir in self.source_dirs.items():
            if source_dir.exists():
                backup_path = self.backup_dir / name
                shutil.copytree(source_dir, backup_path)
                logger.info(f"Backed up {name} to {backup_path}")
        
        # Backup target directory if it exists
        if self.target_dir.exists():
            backup_path = self.backup_dir / "trading_optimization"
            shutil.copytree(self.target_dir, backup_path)
            logger.info(f"Backed up trading/optimization to {backup_path}")
        
        self.consolidation_results['backup_created'] = str(self.backup_dir)
    
    def _setup_target_structure(self) -> None:
        """Setup the target directory structure."""
        logger.info("Setting up target directory structure...")
        
        # Create main directories
        directories = [
            self.target_dir,
            self.target_dir / "core",
            self.target_dir / "strategies", 
            self.target_dir / "visualization",
            self.target_dir / "utils",
            self.target_dir / "configs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Create __init__.py if it doesn't exist
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    def _consolidate_all_modules(self) -> None:
        """Consolidate files from all source directories."""
        logger.info("Consolidating modules...")
        
        # Consolidate optimizer/ directory
        if self.source_dirs['optimizer'].exists():
            self._consolidate_optimizer_directory()
        
        # Consolidate optimize/ directory
        if self.source_dirs['optimize'].exists():
            self._consolidate_optimize_directory()
        
        # Consolidate optimizers/ directory (keep consolidator in utils)
        if self.source_dirs['optimizers'].exists():
            self._consolidate_optimizers_directory()
    
    def _consolidate_optimizer_directory(self) -> None:
        """Consolidate the optimizer/ directory."""
        logger.info("Consolidating optimizer/ directory...")
        
        optimizer_dir = self.source_dirs['optimizer']
        
        # Move core optimizers
        core_dir = optimizer_dir / "core"
        if core_dir.exists():
            for file_path in core_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    target_path = self.target_dir / "core" / file_path.name
                    self._move_file_safely(file_path, target_path, "core")
        
        # Move strategy optimizers
        strategies_dir = optimizer_dir / "strategies"
        if strategies_dir.exists():
            for file_path in strategies_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    target_path = self.target_dir / "strategies" / file_path.name
                    self._move_file_safely(file_path, target_path, "strategies")
        
        # Move visualization
        viz_dir = optimizer_dir / "visualization"
        if viz_dir.exists():
            for file_path in viz_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    target_path = self.target_dir / "visualization" / file_path.name
                    self._move_file_safely(file_path, target_path, "visualization")
        
        # Move main __init__.py and README
        for file_name in ["__init__.py", "README.md", "requirements.txt"]:
            file_path = optimizer_dir / file_name
            if file_path.exists():
                target_path = self.target_dir / file_name
                self._move_file_safely(file_path, target_path, "main")
    
    def _consolidate_optimize_directory(self) -> None:
        """Consolidate the optimize/ directory."""
        logger.info("Consolidating optimize/ directory...")
        
        optimize_dir = self.source_dirs['optimize']
        
        for file_path in optimize_dir.glob("*.py"):
            if file_path.name != "__init__.py":
                # Check if file already exists in target
                target_path = self.target_dir / file_path.name
                if target_path.exists():
                    # Merge files if they're different
                    self._merge_files(file_path, target_path)
                else:
                    # Move file directly
                    self._move_file_safely(file_path, target_path, "optimize")
    
    def _consolidate_optimizers_directory(self) -> None:
        """Consolidate the optimizers/ directory."""
        logger.info("Consolidating optimizers/ directory...")
        
        optimizers_dir = self.source_dirs['optimizers']
        
        for file_path in optimizers_dir.glob("*.py"):
            if file_path.name == "consolidator.py":
                # Move consolidator to utils
                target_path = self.target_dir / "utils" / file_path.name
                self._move_file_safely(file_path, target_path, "utils")
            else:
                # Move other files to main directory
                target_path = self.target_dir / file_path.name
                self._move_file_safely(file_path, target_path, "main")
    
    def _move_file_safely(self, source_path: Path, target_path: Path, category: str) -> None:
        """Safely move a file, handling conflicts."""
        try:
            if target_path.exists():
                # File already exists, create backup and merge
                backup_path = target_path.with_suffix(f"{target_path.suffix}.backup")
                shutil.copy2(target_path, backup_path)
                logger.info(f"Created backup of existing file: {backup_path}")
                
                # Merge files
                self._merge_files(source_path, target_path)
                self.consolidation_results['files_merged'].append(str(source_path))
            else:
                # Move file directly
                shutil.move(str(source_path), str(target_path))
                self.consolidation_results['files_moved'].append(str(source_path))
            
            logger.info(f"Moved {source_path} to {target_path}")
            
        except Exception as e:
            error_msg = f"Error moving {source_path}: {str(e)}"
            logger.error(error_msg)
            self.consolidation_results['errors'].append(error_msg)
    
    def _merge_files(self, source_path: Path, target_path: Path) -> None:
        """Merge two files, keeping the best parts of each."""
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                source_content = f.read()
            
            with open(target_path, 'r', encoding='utf-8') as f:
                target_content = f.read()
            
            # Simple merge strategy: append source content with a separator
            merged_content = f"""# Merged from {source_path}
# Original target: {target_path}
# Merge timestamp: {datetime.now().isoformat()}

{target_content}

# === MERGED CONTENT FROM {source_path} ===
{source_content}
"""
            
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            
            logger.info(f"Merged {source_path} into {target_path}")
            
        except Exception as e:
            error_msg = f"Error merging {source_path} and {target_path}: {str(e)}"
            logger.error(error_msg)
            self.consolidation_results['errors'].append(error_msg)
    
    def _update_all_imports(self) -> None:
        """Update all imports across the codebase."""
        logger.info("Updating imports across codebase...")
        
        # Define import mappings
        import_mappings = {
            # Old imports to new imports
            "from optimizer.": "from trading.optimization.",
            "import optimizer.": "import trading.optimization.",
            "from optimize.": "from trading.optimization.",
            "import optimize.": "import trading.optimization.",
            "from optimizers.": "from trading.optimization.utils.",
            "import optimizers.": "import trading.optimization.utils.",
            
            # Relative imports
            "from .optimizer.": "from trading.optimization.",
            "import .optimizer.": "import trading.optimization.",
            "from .optimize.": "from trading.optimization.",
            "import .optimize.": "import trading.optimization.",
            "from .optimizers.": "from trading.optimization.utils.",
            "import .optimizers.": "import trading.optimization.utils.",
        }
        
        # Process all Python files
        updated_files = 0
        for py_file in self.root_dir.rglob("*.py"):
            # Skip backup and cache directories
            if any(part in ["backup", "__pycache__", ".git", ".venv"] for part in py_file.parts):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply import mappings
                for old_import, new_import in import_mappings.items():
                    content = content.replace(old_import, new_import)
                
                # Write back if changes were made
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    updated_files += 1
                    self.consolidation_results['imports_updated'].append(str(py_file))
                    logger.info(f"Updated imports in {py_file}")
                    
            except Exception as e:
                error_msg = f"Error updating imports in {py_file}: {str(e)}"
                logger.error(error_msg)
                self.consolidation_results['errors'].append(error_msg)
        
        logger.info(f"Updated imports in {updated_files} files")
    
    def _cleanup_source_directories(self) -> None:
        """Clean up source directories after consolidation."""
        logger.info("Cleaning up source directories...")
        
        for name, source_dir in self.source_dirs.items():
            if source_dir.exists():
                try:
                    # Check if directory is empty
                    if not any(source_dir.rglob("*")):
                        source_dir.rmdir()
                        logger.info(f"Removed empty directory: {source_dir}")
                    else:
                        # Add deprecation notice to remaining files
                        self._add_deprecation_notices(source_dir)
                        logger.info(f"Added deprecation notices to {source_dir}")
                        
                except Exception as e:
                    error_msg = f"Error cleaning up {source_dir}: {str(e)}"
                    logger.error(error_msg)
                    self.consolidation_results['errors'].append(error_msg)
    
    def _add_deprecation_notices(self, directory: Path) -> None:
        """Add deprecation notices to files in directory."""
        for file_path in directory.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                deprecation_notice = f'''"""
DEPRECATED: This module has been consolidated into trading/optimization/
Please update your imports to use the consolidated version.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

'''
                
                if not content.startswith('"""DEPRECATED'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(deprecation_notice + content)
                        
            except Exception as e:
                logger.error(f"Error adding deprecation notice to {file_path}: {str(e)}")
    
    def _validate_consolidation(self) -> None:
        """Validate the consolidation results."""
        logger.info("Validating consolidation...")
        
        # Check that target directory has expected structure
        expected_files = [
            self.target_dir / "__init__.py",
            self.target_dir / "core" / "__init__.py",
            self.target_dir / "strategies" / "__init__.py",
            self.target_dir / "visualization" / "__init__.py",
            self.target_dir / "utils" / "__init__.py",
        ]
        
        for file_path in expected_files:
            if not file_path.exists():
                warning_msg = f"Expected file missing: {file_path}"
                logger.warning(warning_msg)
                self.consolidation_results['warnings'].append(warning_msg)
        
        # Check for import errors
        import_errors = self._check_import_errors()
        if import_errors:
            for error in import_errors:
                self.consolidation_results['warnings'].append(f"Import error: {error}")
    
    def _check_import_errors(self) -> List[str]:
        """Check for import errors in the consolidated module."""
        errors = []
        
        try:
            # Try to import the main optimization module
            import sys
            sys.path.insert(0, str(self.root_dir))
            
            import trading.optimization
            logger.info("Successfully imported trading.optimization")
            
        except ImportError as e:
            errors.append(f"Failed to import trading.optimization: {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error importing trading.optimization: {str(e)}")
        
        return errors
    
    def _restore_from_backup(self) -> None:
        """Restore from backup in case of error."""
        if not self.consolidation_results['backup_created']:
            return
        
        logger.info("Restoring from backup...")
        backup_dir = Path(self.consolidation_results['backup_created'])
        
        try:
            # Restore each directory
            for backup_subdir in backup_dir.iterdir():
                if backup_subdir.is_dir():
                    target_dir = self.root_dir / backup_subdir.name
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(backup_subdir, target_dir)
                    logger.info(f"Restored {backup_subdir.name}")
            
            logger.info("Restoration completed")
            
        except Exception as e:
            logger.error(f"Error during restoration: {str(e)}")

def main():
    """Main function to run consolidation."""
    consolidator = OptimizationConsolidator()
    results = consolidator.run_consolidation(create_backup=True)
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION CONSOLIDATION RESULTS")
    print("="*60)
    
    if results['success']:
        print("✅ Consolidation completed successfully!")
    else:
        print("❌ Consolidation failed!")
    
    print(f"\n📁 Files moved: {len(results['files_moved'])}")
    print(f"🔀 Files merged: {len(results['files_merged'])}")
    print(f"📝 Imports updated: {len(results['imports_updated'])}")
    print(f"⚠️  Warnings: {len(results['warnings'])}")
    print(f"❌ Errors: {len(results['errors'])}")
    
    if results['backup_created']:
        print(f"\n💾 Backup created at: {results['backup_created']}")
    
    if results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['errors']:
        print("\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 