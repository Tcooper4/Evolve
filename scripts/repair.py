"""
Auto-repair script for the trading platform.
Scans and fixes import issues, encoding problems, and consolidates duplicate files.
"""

import ast
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repair.log'),
        logging.StreamHandler()
    ]
)

class CodeRepair:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.imports: Dict[str, Set[str]] = {}
        self.encoding_issues: List[Path] = []
        self.duplicate_files: List[Tuple[Path, Path]] = []
        self.broken_imports: Dict[Path, List[str]] = {}
        self.missing_packages: Set[str] = set()

    def scan_imports(self) -> None:
        """Scan all Python files for imports."""
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content)
                imports = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.add(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                
                self.imports[str(py_file)] = imports
            except Exception as e:
                logging.error(f"Error scanning {py_file}: {e}")

    def check_encoding(self) -> None:
        """Check for non-UTF-8 files."""
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                self.encoding_issues.append(py_file)

    def find_duplicates(self) -> None:
        """Find duplicate files based on content similarity."""
        file_contents: Dict[str, List[Path]] = {}
        
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Normalize content for comparison
                normalized = re.sub(r'\s+', ' ', content).strip()
                if normalized in file_contents:
                    file_contents[normalized].append(py_file)
                else:
                    file_contents[normalized] = [py_file]
            except Exception as e:
                logging.error(f"Error reading {py_file}: {e}")

        # Find duplicates
        for content, files in file_contents.items():
            if len(files) > 1:
                # Sort by path length to prefer shorter paths
                files.sort(key=lambda x: len(str(x)))
                for i in range(1, len(files)):
                    self.duplicate_files.append((files[0], files[i]))

    def fix_imports(self) -> None:
        """Fix broken imports and convert relative to absolute."""
        for py_file, imports in self.imports.items():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert relative imports to absolute
                content = re.sub(
                    r'from \.([a-zA-Z_][a-zA-Z0-9_]*) import',
                    r'from trading.\1 import',
                    content
                )
                
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                logging.error(f"Error fixing imports in {py_file}: {e}")

    def fix_encoding(self) -> None:
        """Convert files to UTF-8 encoding."""
        for py_file in self.encoding_issues:
            try:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(py_file, 'r', encoding=encoding) as f:
                            content = f.read()
                        # Write back as UTF-8
                        with open(py_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logging.info(f"Converted {py_file} from {encoding} to UTF-8")
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                logging.error(f"Error fixing encoding for {py_file}: {e}")

    def consolidate_duplicates(self) -> None:
        """Consolidate duplicate files."""
        for original, duplicate in self.duplicate_files:
            try:
                # Add deprecation notice to duplicate
                with open(duplicate, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                deprecation_notice = f'''"""
DEPRECATED: This file is a duplicate of {original}
Please use the original file instead.
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

'''
                with open(duplicate, 'w', encoding='utf-8') as f:
                    f.write(deprecation_notice + content)
                
                logging.info(f"Marked {duplicate} as deprecated, original: {original}")
            except Exception as e:
                logging.error(f"Error consolidating {duplicate}: {e}")

    def install_missing_packages(self) -> None:
        """Install missing pip packages."""
        for package in self.missing_packages:
            try:
                subprocess.run(['pip', 'install', package], check=True)
                logging.info(f"Installed missing package: {package}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error installing {package}: {e}")

    def cleanup(self) -> None:
        """Clean up cache and temporary files."""
        try:
            # Remove __pycache__ directories
            for cache_dir in self.root_dir.rglob("__pycache__"):
                shutil.rmtree(cache_dir)
            
            # Remove .egg-info directories
            for egg_dir in self.root_dir.rglob("*.egg-info"):
                shutil.rmtree(egg_dir)
            
            # Remove .coverage files
            for coverage_file in self.root_dir.rglob(".coverage*"):
                coverage_file.unlink()
            
            logging.info("Cleaned up cache and temporary files")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def execute_repair_operations(self) -> None:
        """Execute all repair operations.
        
        Performs a comprehensive scan and repair of the codebase including:
        - Import scanning and fixing
        - Encoding issue detection and resolution
        - Duplicate file identification and consolidation
        - Missing package installation
        - Cache cleanup
        """
        logging.info("Starting code repair...")
        
        self.scan_imports()
        self.check_encoding()
        self.find_duplicates()
        
        if self.encoding_issues:
            logging.info(f"Found {len(self.encoding_issues)} files with encoding issues")
            self.fix_encoding()
        
        if self.duplicate_files:
            logging.info(f"Found {len(self.duplicate_files)} duplicate files")
            self.consolidate_duplicates()
        
        self.fix_imports()
        self.install_missing_packages()
        self.cleanup()
        
        logging.info("Code repair completed")

if __name__ == "__main__":
    repair = CodeRepair()
    repair.execute_repair_operations() 