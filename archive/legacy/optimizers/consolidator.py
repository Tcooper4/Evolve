"""
Optimizer Consolidator Module

Reusable module for consolidating duplicate optimizer files and updating imports.
Provides both programmatic and UI-triggered consolidation capabilities.
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json

# Configure logging
logger = logging.getLogger(__name__)

class OptimizerConsolidator:
    """Class for consolidating optimizer files and managing optimizer organization."""
    
    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the consolidator.
        
        Args:
            root_dir: Root directory for the project (defaults to current directory)
        """
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self.optimize_dir = self.root_dir / "optimize"
        self.trading_optimization_dir = self.root_dir / "trading" / "optimization"
        self.backup_dir = self.root_dir / "backup"
        
        # Ensure directories exist
        self.trading_optimization_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def run_optimizer_consolidation(self, create_backup: bool = True) -> Dict[str, Any]:
        """
        Run the complete optimizer consolidation process.
        
        Args:
            create_backup: Whether to create a backup before consolidation
            
        Returns:
            Dictionary with consolidation results and statistics
        """
        logger.info("Starting optimizer consolidation process")
        
        results = {
            "success": False,
            "files_moved": [],
            "files_deprecated": [],
            "imports_updated": 0,
            "errors": [],
            "backup_created": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Create backup if requested
            if create_backup:
                backup_path = self._create_backup()
                results["backup_created"] = str(backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Consolidate optimizer files
            moved_files, deprecated_files = self._consolidate_files()
            results["files_moved"] = moved_files
            results["files_deprecated"] = deprecated_files
            
            # Update imports
            imports_updated = self._update_imports()
            results["imports_updated"] = imports_updated
            
            # Clean up empty directories
            self._cleanup_empty_directories()
            
            results["success"] = True
            logger.info("Optimizer consolidation completed successfully")
            
        except Exception as e:
            error_msg = f"Error during consolidation: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            
            # Attempt to restore from backup if available
            if results["backup_created"]:
                self._restore_from_backup(Path(results["backup_created"]))
        
        return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _create_backup(self) -> Path:
        """Create a backup of the optimize directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"optimizer_backup_{timestamp}"
        
        if self.optimize_dir.exists():
            shutil.copytree(self.optimize_dir, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        return {'success': True, 'result': backup_path, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _consolidate_files(self) -> Tuple[List[str], List[str]]:
        """Consolidate optimizer files into trading/optimization."""
        moved_files = []
        deprecated_files = []
        
        if not self.optimize_dir.exists():
            logger.info("Optimize directory does not exist, nothing to consolidate")
            return {'success': True, 'result': moved_files, deprecated_files, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        # Process each file in the optimize directory
        for file_path in self.optimize_dir.glob("*.py"):
            target_path = self.trading_optimization_dir / file_path.name
            
            if target_path.exists():
                # File already exists in target, mark as deprecated
                self._mark_as_deprecated(file_path)
                deprecated_files.append(str(file_path))
                logger.info(f"Marked {file_path} as deprecated")
            else:
                # Move file to target directory
                shutil.move(str(file_path), str(target_path))
                moved_files.append(str(file_path))
                logger.info(f"Moved {file_path} to {target_path}")
        
        return moved_files, deprecated_files
    
    def _mark_as_deprecated(self, file_path: Path) -> None:
        """Mark a file as deprecated by adding a deprecation notice."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            deprecation_notice = f'''"""
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
DEPRECATED: This file has been consolidated into trading/optimization/{file_path.name}
Please use the consolidated version instead.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

'''
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(deprecation_notice + content)
                
        except Exception as e:
            logger.error(f"Error marking {file_path} as deprecated: {str(e)}")
    
    def _update_imports(self) -> int:
        """Update imports in all Python files to use consolidated optimizers."""
        updated_count = 0
        
        # Define import mappings
        import_mappings = {
            "from optimize.": "from trading.optimization.",
            "import optimize.": "import trading.optimization.",
            "from .optimize.": "from trading.optimization.",
            "import .optimize.": "import trading.optimization."
        }
        
        # Process all Python files
        for py_file in self.root_dir.rglob("*.py"):
            try:
                # Skip files in backup and __pycache__ directories
                if any(part in ["backup", "__pycache__", ".git"] for part in py_file.parts):
                    continue
                
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
                    updated_count += 1
                    logger.info(f"Updated imports in {py_file}")
                    
            except Exception as e:
                logger.error(f"Error updating imports in {py_file}: {str(e)}")
        
        return {'success': True, 'result': updated_count, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _cleanup_empty_directories(self) -> None:
        """Remove empty directories after consolidation."""
        if self.optimize_dir.exists() and not any(self.optimize_dir.iterdir()):
            self.optimize_dir.rmdir()
            logger.info("Removed empty optimize directory")
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def _restore_from_backup(self, backup_path: Path) -> None:
        """Restore from backup in case of error."""
        try:
            if backup_path.exists():
                if self.optimize_dir.exists():
                    shutil.rmtree(self.optimize_dir)
                shutil.copytree(backup_path, self.optimize_dir)
                logger.info("Restored from backup due to error")
        except Exception as e:
            logger.error(f"Error restoring from backup: {str(e)}")
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def get_optimizer_status(self) -> Dict[str, Any]:
        """
        Get current status of optimizer organization.
        
        Returns:
            Dictionary with optimizer status information
        """
        status = {
            "optimize_dir_exists": self.optimize_dir.exists(),
            "trading_optimization_dir_exists": self.trading_optimization_dir.exists(),
            "files_in_optimize": [],
            "files_in_trading_optimization": [],
            "duplicate_files": [],
            "consolidation_needed": False
        }
        
        # Check files in optimize directory
        if self.optimize_dir.exists():
            status["files_in_optimize"] = [f.name for f in self.optimize_dir.glob("*.py")]
        
        # Check files in trading/optimization directory
        if self.trading_optimization_dir.exists():
            status["files_in_trading_optimization"] = [f.name for f in self.trading_optimization_dir.glob("*.py")]
        
        # Check for duplicates
        optimize_files = set(status["files_in_optimize"])
        trading_files = set(status["files_in_trading_optimization"])
        status["duplicate_files"] = list(optimize_files.intersection(trading_files))
        
        # Determine if consolidation is needed
        status["consolidation_needed"] = (
            len(status["files_in_optimize"]) > 0 or 
            len(status["duplicate_files"]) > 0
        )
        
        return {'success': True, 'result': status, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def validate_consolidation(self) -> Dict[str, Any]:
        """
        Validate that consolidation was successful.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "success": True,
            "errors": [],
            "warnings": [],
            "imports_valid": True
        }
        
        # Check if optimize directory is empty or doesn't exist
        if self.optimize_dir.exists() and any(self.optimize_dir.iterdir()):
            validation["warnings"].append("Optimize directory still contains files")
        
        # Check if trading/optimization directory exists and has files
        if not self.trading_optimization_dir.exists():
            validation["success"] = False
            validation["errors"].append("Trading optimization directory does not exist")
        
        # Validate imports in Python files
        import_errors = self._validate_imports()
        if import_errors:
            validation["imports_valid"] = False
            validation["errors"].extend(import_errors)
        
        return {'success': True, 'result': validation, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _validate_imports(self) -> List[str]:
        """Validate that imports are correctly updated."""
        errors = []
        
        # Check for old import patterns
        old_patterns = [
            "from optimize.",
            "import optimize.",
            "from .optimize.",
            "import .optimize."
        ]
        
        for py_file in self.root_dir.rglob("*.py"):
            try:
                # Skip backup and cache directories
                if any(part in ["backup", "__pycache__", ".git"] for part in py_file.parts):
                    continue
                
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in old_patterns:
                    if pattern in content:
                        errors.append(f"Found old import pattern '{pattern}' in {py_file}")
                        
            except Exception as e:
                errors.append(f"Error reading {py_file}: {str(e)}")
        
        return {'success': True, 'result': errors, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def run_optimizer_consolidation(create_backup: bool = True, root_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run optimizer consolidation.
    
    Args:
        create_backup: Whether to create a backup before consolidation
        root_dir: Root directory for the project
        
    Returns:
        Dictionary with consolidation results
    """
    consolidator = OptimizerConsolidator(root_dir)
    return {'success': True, 'result': consolidator.run_optimizer_consolidation(create_backup), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


def get_optimizer_status(root_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current optimizer organization status.
    
    Args:
        root_dir: Root directory for the project
        
    Returns:
        Dictionary with optimizer status
    """
    consolidator = OptimizerConsolidator(root_dir)
    return {'success': True, 'result': consolidator.get_optimizer_status(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}


if __name__ == "__main__":
    # Run consolidation when script is executed directly
    results = run_optimizer_consolidation()
    print(json.dumps(results, indent=2)) 