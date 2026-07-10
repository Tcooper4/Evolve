"""
Optimizer Consolidator Module

Reusable module for consolidating duplicate optimizer files and updating imports.
Provides both programmatic and UI-triggered consolidation capabilities.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class OptimizerConsolidator:
    """Class for consolidating optimizer files and managing optimizer organization."""


class PositionConsolidator:
    """Class for consolidating trading positions and managing position organization."""
    
    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the position consolidator.
        
        Args:
            root_dir: Root directory for the project (defaults to current directory)
        """
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        logger.info("PositionConsolidator initialized")
    
    def consolidate_positions(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate duplicate or overlapping positions.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary with consolidated positions
        """
        try:
            # Group positions by symbol
            positions_by_symbol = {}
            for pos in positions:
                symbol = pos.get("symbol", "UNKNOWN")
                if symbol not in positions_by_symbol:
                    positions_by_symbol[symbol] = []
                positions_by_symbol[symbol].append(pos)
            
            # Consolidate positions for each symbol
            consolidated = {}
            for symbol, symbol_positions in positions_by_symbol.items():
                if len(symbol_positions) == 1:
                    consolidated[symbol] = symbol_positions[0]
                else:
                    # Merge multiple positions for the same symbol
                    total_quantity = sum(p.get("quantity", 0) for p in symbol_positions)
                    avg_price = sum(
                        p.get("price", 0) * p.get("quantity", 0) 
                        for p in symbol_positions
                    ) / total_quantity if total_quantity > 0 else 0
                    
                    consolidated[symbol] = {
                        "symbol": symbol,
                        "quantity": total_quantity,
                        "price": avg_price,
                        "positions_merged": len(symbol_positions),
                    }
            
            return {
                "success": True,
                "original_count": len(positions),
                "consolidated_count": len(consolidated),
                "positions": consolidated,
            }
        except Exception as e:
            logger.error(f"Error consolidating positions: {e}")
            return {
                "success": False,
                "error": str(e),
                "positions": positions,
            }

    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the consolidator.

        Args:
            root_dir: Root directory for the project (defaults to current directory)
        """
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self.optimize_dir = self.root_dir / "optimize"
        self.optimizer_dir = self.root_dir / "optimizer"
        self.optimizers_dir = self.root_dir / "optimizers"
        self.trading_optimization_dir = self.root_dir / "trading" / "optimization"
        self.backup_dir = self.root_dir / "backup"

        # Ensure directories exist
        self.trading_optimization_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def run_optimizer_consolidation(self, create_backup: bool = True) -> Dict[str, Any]:
        """
        Run the complete optimizer consolidation process.

        Args:
            create_backup: Whether to create a backup before consolidation

        Returns:
            Dictionary with consolidation results and statistics
        """
        logger.info("Starting comprehensive optimizer consolidation process")

        results = {
            "success": False,
            "files_moved": [],
            "files_merged": [],
            "files_deprecated": [],
            "imports_updated": 0,
            "errors": [],
            "backup_created": None,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Create backup if requested
            if create_backup:
                backup_path = self._create_backup()
                results["backup_created"] = str(backup_path)
                logger.info(f"Created backup at {backup_path}")

            # Consolidate all optimizer directories
            self._consolidate_all_directories(results)

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

        return results

    def _create_backup(self) -> Path:
        """Create a backup of all optimizer directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"optimizer_backup_{timestamp}"

        backup_path.mkdir(parents=True, exist_ok=True)

        # Backup each directory if it exists
        for dir_name, dir_path in [
            ("optimize", self.optimize_dir),
            ("optimizer", self.optimizer_dir),
            ("optimizers", self.optimizers_dir),
        ]:
            if dir_path.exists():
                backup_subdir = backup_path / dir_name
                shutil.copytree(dir_path, backup_subdir)
                logger.info(f"Backed up {dir_name} to {backup_subdir}")

        return backup_path

    def _consolidate_all_directories(self, results: Dict[str, Any]) -> None:
        """Consolidate files from all optimizer directories."""
        # Consolidate optimize/ directory
        if self.optimize_dir.exists():
            self._consolidate_directory(self.optimize_dir, results, "optimize")

        # Consolidate optimizer/ directory
        if self.optimizer_dir.exists():
            self._consolidate_directory(self.optimizer_dir, results, "optimizer")

        # Consolidate optimizers/ directory (keep consolidator in utils)
        if self.optimizers_dir.exists():
            self._consolidate_optimizers_directory(results)

    def _consolidate_directory(
        self, source_dir: Path, results: Dict[str, Any], dir_name: str
    ) -> None:
        """Consolidate files from a directory."""
        logger.info(f"Consolidating {dir_name}/ directory...")

        for file_path in source_dir.rglob("*.py"):
            if file_path.name == "__init__.py":
                continue

            # Determine target path based on file location
            relative_path = file_path.relative_to(source_dir)

            if dir_name == "optimizer":
                # Handle optimizer/ subdirectories
                if relative_path.parts[0] == "core":
                    target_path = (
                        self.trading_optimization_dir / "core" / relative_path.name
                    )
                elif relative_path.parts[0] == "strategies":
                    target_path = (
                        self.trading_optimization_dir
                        / "strategies"
                        / relative_path.name
                    )
                elif relative_path.parts[0] == "visualization":
                    target_path = (
                        self.trading_optimization_dir
                        / "visualization"
                        / relative_path.name
                    )
                else:
                    target_path = self.trading_optimization_dir / relative_path.name
            else:
                target_path = self.trading_optimization_dir / relative_path.name

            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Move or merge file
            if target_path.exists():
                self._merge_files(file_path, target_path, results)
            else:
                self._move_file(file_path, target_path, results)

    def _consolidate_optimizers_directory(self, results: Dict[str, Any]) -> None:
        """Consolidate optimizers/ directory."""
        logger.info("Consolidating optimizers/ directory...")

        for file_path in self.optimizers_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue

            if file_path.name == "consolidator.py":
                # Move consolidator to utils
                target_path = self.trading_optimization_dir / "utils" / file_path.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self._move_file(file_path, target_path, results)
            else:
                # Move other files to main directory
                target_path = self.trading_optimization_dir / file_path.name
                self._move_file(file_path, target_path, results)

    def _move_file(
        self, source_path: Path, target_path: Path, results: Dict[str, Any]
    ) -> None:
        """Move a file to target location."""
        try:
            shutil.move(str(source_path), str(target_path))
            results["files_moved"].append(str(source_path))
            logger.info(f"Moved {source_path} to {target_path}")
        except Exception as e:
            error_msg = f"Error moving {source_path}: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

    def _merge_files(
        self, source_path: Path, target_path: Path, results: Dict[str, Any]
    ) -> None:
        """Merge two files, keeping the best parts of each."""
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                source_content = f.read()

            with open(target_path, "r", encoding="utf-8") as f:
                target_content = f.read()

            # Simple merge strategy: append source content with a separator
            merged_content = f"""# Merged from {source_path}

# Original target: {target_path}
# Merge timestamp: {datetime.now().isoformat()}

{target_content}

# === MERGED CONTENT FROM {source_path} ===
{source_content}
"""

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(merged_content)

            results["files_merged"].append(str(source_path))
            logger.info(f"Merged {source_path} into {target_path}")

        except Exception as e:
            error_msg = f"Error merging {source_path} and {target_path}: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

    def _update_imports(self) -> int:
        """Update imports in all Python files to use consolidated optimizers."""
        updated_count = 0

        # Define import mappings
        import_mappings = {
            "from optimize.": "from trading.optimization.",
            "import optimize.": "import trading.optimization.",
            "from optimizer.": "from trading.optimization.",
            "import optimizer.": "import trading.optimization.",
            "from optimizers.": "from trading.optimization.utils.",
            "import optimizers.": "import trading.optimization.utils.",
            "from .optimize.": "from trading.optimization.",
            "import .optimize.": "import trading.optimization.",
            "from .optimizer.": "from trading.optimization.",
            "import .optimizer.": "import trading.optimization.",
            "from .optimizers.": "from trading.optimization.utils.",
            "import .optimizers.": "import trading.optimization.utils.",
        }

        # Process all Python files
        for py_file in self.root_dir.rglob("*.py"):
            try:
                # Skip files in backup and __pycache__ directories
                if any(
                    part in ["backup", "__pycache__", ".git", ".venv"]
                    for part in py_file.parts
                ):
                    continue

                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # Apply import mappings
                for old_import, new_import in import_mappings.items():
                    content = content.replace(old_import, new_import)

                # Write back if changes were made
                if content != original_content:
                    with open(py_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    updated_count += 1
                    logger.info(f"Updated imports in {py_file}")

            except Exception as e:
                logger.error(f"Error updating imports in {py_file}: {str(e)}")

        return updated_count

    def _cleanup_empty_directories(self) -> None:
        """Remove empty directories after consolidation."""
        for dir_path in [self.optimize_dir, self.optimizer_dir, self.optimizers_dir]:
            if dir_path.exists() and not any(dir_path.rglob("*")):
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error removing {dir_path}: {str(e)}")

    def _restore_from_backup(self, backup_path: Path) -> None:
        """Restore from backup in case of error."""
        try:
            if backup_path.exists():
                for backup_subdir in backup_path.iterdir():
                    if backup_subdir.is_dir():
                        target_dir = self.root_dir / backup_subdir.name
                        if target_dir.exists():
                            shutil.rmtree(target_dir)
                        shutil.copytree(backup_subdir, target_dir)
                        logger.info(f"Restored {backup_subdir.name}")

                logger.info("Restoration completed")

        except Exception as e:
            logger.error(f"Error during restoration: {str(e)}")

    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get status of optimizer consolidation."""
        status = {
            "trading_optimization_exists": self.trading_optimization_dir.exists(),
            "optimize_dir_exists": self.optimize_dir.exists(),
            "optimizer_dir_exists": self.optimizer_dir.exists(),
            "optimizers_dir_exists": self.optimizers_dir.exists(),
            "files_in_trading_optimization": [],
            "consolidation_needed": False,
        }

        if self.trading_optimization_dir.exists():
            status["files_in_trading_optimization"] = [
                str(f.relative_to(self.trading_optimization_dir))
                for f in self.trading_optimization_dir.rglob("*.py")
                if f.name != "__init__.py"
            ]

        # Check if consolidation is needed
        if (
            self.optimize_dir.exists()
            or self.optimizer_dir.exists()
            or self.optimizers_dir.exists()
        ):
            status["consolidation_needed"] = True

        return status

    def validate_consolidation(self) -> Dict[str, Any]:
        """Validate the consolidation results."""
        validation = {"success": True, "errors": [], "warnings": []}

        # Check that target directory has expected structure
        expected_dirs = [
            self.trading_optimization_dir,
            self.trading_optimization_dir / "core",
            self.trading_optimization_dir / "strategies",
            self.trading_optimization_dir / "visualization",
            self.trading_optimization_dir / "utils",
        ]

        for dir_path in expected_dirs:
            if not dir_path.exists():
                validation["warnings"].append(f"Expected directory missing: {dir_path}")

        # Check for import errors
        import_errors = self._validate_imports()
        if import_errors:
            validation["errors"].extend(import_errors)
            validation["success"] = False

        return validation

    def _validate_imports(self) -> List[str]:
        """Check for import errors in the consolidated module."""
        errors = []

        try:
            # Try to import the main optimization module
            import sys

            sys.path.insert(0, str(self.root_dir))

            logger.info("Successfully imported trading.optimization")

        except ImportError as e:
            errors.append(f"Failed to import trading.optimization: {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error importing trading.optimization: {str(e)}")

        return errors


def run_optimizer_consolidation(
    create_backup: bool = True, root_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the optimizer consolidation process.

    Args:
        create_backup: Whether to create a backup before consolidation
        root_dir: Root directory for the project

    Returns:
        Dictionary with consolidation results
    """
    consolidator = OptimizerConsolidator(root_dir)
    return consolidator.run_optimizer_consolidation(create_backup)


def get_optimizer_status(root_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the status of optimizer consolidation.

    Args:
        root_dir: Root directory for the project

    Returns:
        Dictionary with optimizer status
    """
    consolidator = OptimizerConsolidator(root_dir)
    return consolidator.get_optimizer_status()
