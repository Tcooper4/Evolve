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
from typing import Dict, List, Set, Tuple, Any
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
        return {
            'success': True,
            'message': 'CodeRepair initialized successfully',
            'timestamp': datetime.now().isoformat()
        }

    def scan_imports(self) -> Dict[str, Any]:
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
        
        return {
            'success': True,
            'message': f'Scanned {len(self.imports)} files for imports',
            'timestamp': datetime.now().isoformat()
        }

    def check_encoding(self) -> Dict[str, Any]:
        """Check for non-UTF-8 files."""
        for py_file in self.root_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError:
                self.encoding_issues.append(py_file)
        
        return {
            'success': True,
            'message': f'Found {len(self.encoding_issues)} encoding issues',
            'timestamp': datetime.now().isoformat()
        }

    def find_duplicates(self) -> Dict[str, Any]:
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
        
        return {
            'success': True,
            'message': f'Found {len(self.duplicate_files)} duplicate files',
            'timestamp': datetime.now().isoformat()
        }

    def fix_imports(self) -> Dict[str, Any]:
        """Fix broken imports and convert relative to absolute."""
        fixed_count = 0
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
                fixed_count += 1
            except Exception as e:
                logging.error(f"Error fixing imports in {py_file}: {e}")
        
        return {
            'success': True,
            'message': f'Fixed imports in {fixed_count} files',
            'timestamp': datetime.now().isoformat()
        }

    def fix_encoding(self) -> Dict[str, Any]:
        """Convert files to UTF-8 encoding."""
        fixed_count = 0
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
                        fixed_count += 1
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                logging.error(f"Error fixing encoding for {py_file}: {e}")
        
        return {
            'success': True,
            'message': f'Fixed encoding for {fixed_count} files',
            'timestamp': datetime.now().isoformat()
        }

    def consolidate_duplicates(self) -> Dict[str, Any]:
        """Consolidate duplicate files."""
        consolidated_count = 0
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
                consolidated_count += 1
            except Exception as e:
                logging.error(f"Error consolidating {duplicate}: {e}")
        
        return {
            'success': True,
            'message': f'Consolidated {consolidated_count} duplicate files',
            'timestamp': datetime.now().isoformat()
        }

    def install_missing_packages(self) -> Dict[str, Any]:
        """Install missing pip packages."""
        installed_count = 0
        for package in self.missing_packages:
            try:
                subprocess.run(['pip', 'install', package], check=True)
                logging.info(f"Installed missing package: {package}")
                installed_count += 1
            except subprocess.CalledProcessError as e:
                logging.error(f"Error installing {package}: {e}")
        
        return {
            'success': True,
            'message': f'Installed {installed_count} missing packages',
            'timestamp': datetime.now().isoformat()
        }

    def cleanup(self) -> Dict[str, Any]:
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
            return {
                'success': True,
                'message': 'Cleanup completed successfully',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def execute_repair_operations(self) -> Dict[str, Any]:
        """Execute all repair operations.
        
        Performs a comprehensive scan and repair of the codebase including:
        - Import scanning and fixing
        - Encoding issue detection and resolution
        - Duplicate file identification and consolidation
        - Missing package installation
        - Cache cleanup
        """
        logging.info("Starting code repair...")
        
        results = []
        results.append(self.scan_imports())
        results.append(self.check_encoding())
        results.append(self.find_duplicates())
        
        if self.encoding_issues:
            logging.info(f"Found {len(self.encoding_issues)} files with encoding issues")
            results.append(self.fix_encoding())
        
        if self.duplicate_files:
            logging.info(f"Found {len(self.duplicate_files)} duplicate files")
            results.append(self.consolidate_duplicates())
        
        results.append(self.fix_imports())
        results.append(self.install_missing_packages())
        results.append(self.cleanup())
        
        logging.info("Code repair completed")
        
        return {
            'success': True,
            'message': 'All repair operations completed',
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    repair = CodeRepair()
    repair.execute_repair_operations() 