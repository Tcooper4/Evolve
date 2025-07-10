#!/usr/bin/env python3
"""
Production Cleanup Script

This script performs comprehensive cleanup to ensure the Evolve system is production-ready:
1. Removes hardcoded values and API keys
2. Cleans up legacy files and directories
3. Removes .pyc, __pycache__, and temporary files
4. Standardizes logging and error messages
5. Validates all imports and dependencies
"""

import os
import sys
import shutil
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Any
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionCleanup:
    """Production cleanup manager."""
    
    def __init__(self, project_root: str):
        """Initialize cleanup manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.cleaned_files = []
        self.removed_files = []
        self.errors = []
        
        # Patterns to clean
        self.hardcoded_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'API_KEY\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
        ]
        
        # Files to remove
        self.remove_patterns = [
            '*.pyc',
            '__pycache__',
            '*.swp',
            '*.tmp',
            '*.log',
            '.DS_Store',
            'Thumbs.db'
        ]
        
        # Legacy directories
        self.legacy_dirs = [
            'trading/optimization/legacy',
            'trading/legacy',
            'legacy',
            'old',
            'backup_old',
            'temp',
            'tmp'
        ]
    
    def run_full_cleanup(self) -> Dict[str, Any]:
        """Run complete production cleanup.
        
        Returns:
            Cleanup results summary
        """
        logger.info("Starting production cleanup...")
        
        try:
            # Step 1: Remove hardcoded values
            self._remove_hardcoded_values()
            
            # Step 2: Clean up files and directories
            self._cleanup_files_and_dirs()
            
            # Step 3: Remove legacy directories
            self._remove_legacy_directories()
            
            # Step 4: Standardize logging
            self._standardize_logging()
            
            # Step 5: Validate imports
            self._validate_imports()
            
            # Step 6: Check for TODO items
            self._check_todo_items()
            
            # Step 7: Generate cleanup report
            report = self._generate_cleanup_report()
            
            logger.info("Production cleanup completed successfully!")
            return report
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.errors.append(str(e))
            return self._generate_cleanup_report()
    
    def _remove_hardcoded_values(self):
        """Remove hardcoded API keys and sensitive values."""
        logger.info("Removing hardcoded values...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                modified = False
                
                # Replace hardcoded patterns
                for pattern in self.hardcoded_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Replace with environment variable references
                        content = re.sub(
                            pattern,
                            lambda m: self._replace_with_env_var(m.group()),
                            content,
                            flags=re.IGNORECASE
                        )
                        modified = True
                
                # Replace hardcoded file paths
                content = self._replace_hardcoded_paths(content)
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.cleaned_files.append(str(file_path))
                    logger.info(f"Cleaned hardcoded values in: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error cleaning {file_path}: {e}")
                self.errors.append(f"Error cleaning {file_path}: {e}")
    
    def _replace_with_env_var(self, match: str) -> str:
        """Replace hardcoded value with environment variable reference."""
        # Extract the key name
        if '=' in match:
            key_part = match.split('=')[0].strip()
            key_name = key_part.split()[-1].upper()
            
            # Create environment variable reference
            return f"{key_part} = os.getenv('{key_name}', '')"
        
        return match
    
    def _replace_hardcoded_paths(self, content: str) -> str:
        """Replace hardcoded file paths with relative paths."""
        # Common hardcoded path patterns
        path_patterns = [
            (r'["\']/home/[^"\']+["\']', 'os.path.join(os.path.dirname(__file__), "data")'),
            (r'["\']C:\\[^"\']+["\']', 'os.path.join(os.path.dirname(__file__), "data")'),
            (r'["\']/Users/[^"\']+["\']', 'os.path.join(os.path.dirname(__file__), "data")'),
        ]
        
        for pattern, replacement in path_patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _cleanup_files_and_dirs(self):
        """Remove temporary and cache files."""
        logger.info("Cleaning up temporary files...")
        
        for pattern in self.remove_patterns:
            try:
                if '*' in pattern:
                    # Handle wildcard patterns
                    for file_path in self.project_root.rglob(pattern):
                        if file_path.is_file():
                            file_path.unlink()
                            self.removed_files.append(str(file_path))
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                            self.removed_files.append(str(file_path))
                else:
                    # Handle specific patterns
                    for file_path in self.project_root.rglob(pattern):
                        if file_path.is_dir():
                            shutil.rmtree(file_path)
                            self.removed_files.append(str(file_path))
                            
            except Exception as e:
                logger.error(f"Error removing {pattern}: {e}")
                self.errors.append(f"Error removing {pattern}: {e}")
    
    def _remove_legacy_directories(self):
        """Remove legacy directories."""
        logger.info("Removing legacy directories...")
        
        for legacy_dir in self.legacy_dirs:
            legacy_path = self.project_root / legacy_dir
            if legacy_path.exists():
                try:
                    shutil.rmtree(legacy_path)
                    self.removed_files.append(str(legacy_path))
                    logger.info(f"Removed legacy directory: {legacy_path}")
                except Exception as e:
                    logger.error(f"Error removing {legacy_path}: {e}")
                    self.errors.append(f"Error removing {legacy_path}: {e}")
    
    def _standardize_logging(self):
        """Standardize logging across all Python files."""
        logger.info("Standardizing logging...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file has logging
                if 'logging' in content or 'print(' in content:
                    # Standardize logging setup
                    if 'logging.getLogger(__name__)' not in content:
                        # Add proper logging setup
                        content = self._add_logging_setup(content)
                    
                    # Replace print statements with logging
                    content = self._replace_print_statements(content)
                    
                    # Ensure consistent error handling
                    content = self._standardize_error_handling(content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.cleaned_files.append(str(file_path))
                    
            except Exception as e:
                logger.error(f"Error standardizing logging in {file_path}: {e}")
                self.errors.append(f"Error standardizing logging in {file_path}: {e}")
    
    def _add_logging_setup(self, content: str) -> str:
        """Add proper logging setup to file."""
        import_pattern = r'import logging'
        logger_pattern = r'logger = logging\.getLogger\(__name__\)'
        
        # Add import if not present
        if 'import logging' not in content:
            # Find the last import statement
            lines = content.split('\n')
            import_index = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_index = i
            
            if import_index >= 0:
                lines.insert(import_index + 1, 'import logging')
                content = '\n'.join(lines)
        
        # Add logger if not present
        if 'logger = logging.getLogger(__name__)' not in content:
            lines = content.split('\n')
            
            # Find a good place to add logger (after imports)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_index = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            lines.insert(insert_index, 'logger = logging.getLogger(__name__)')
            content = '\n'.join(lines)
        
        return content
    
    def _replace_print_statements(self, content: str) -> str:
        """Replace print statements with logging."""
        # Replace print statements with appropriate logging levels
        print_pattern = r'print\s*\(\s*["\']([^"\']+)["\']\s*\)'
        
        def replace_print(match):
            message = match.group(1)
            if any(word in message.lower() for word in ['error', 'failed', 'exception']):
                return f'logger.error("{message}")'
            elif any(word in message.lower() for word in ['warning', 'warn']):
                return f'logger.warning("{message}")'
            elif any(word in message.lower() for word in ['debug', 'info']):
                return f'logger.info("{message}")'
            else:
                return f'logger.info("{message}")'
        
        content = re.sub(print_pattern, replace_print, content)
        return content
    
    def _standardize_error_handling(self, content: str) -> str:
        """Standardize error handling patterns."""
        # Replace bare except blocks
        bare_except_pattern = r'except:'
        content = re.sub(bare_except_pattern, 'except Exception as e:', content)
        
        # Add logging to exception handlers
        except_pattern = r'except Exception as e:\s*\n\s*pass'
        content = re.sub(except_pattern, 'except Exception as e:\n    logger.error(f"Error: {e}")', content)
        
        return content
    
    def _validate_imports(self):
        """Validate all Python imports."""
        logger.info("Validating imports...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                # Try to import the module
                module_name = str(file_path.relative_to(self.project_root)).replace('/', '.').replace('.py', '')
                
                # Skip __init__.py files
                if file_path.name == '__init__.py':
                    continue
                
                # Check for obvious import issues
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for missing imports
                missing_imports = self._check_missing_imports(content)
                if missing_imports:
                    logger.warning(f"Potential missing imports in {file_path}: {missing_imports}")
                
            except Exception as e:
                logger.error(f"Error validating imports in {file_path}: {e}")
                self.errors.append(f"Error validating imports in {file_path}: {e}")
    
    def _check_missing_imports(self, content: str) -> List[str]:
        """Check for potentially missing imports."""
        missing_imports = []
        
        # Common patterns that might need imports
        patterns = [
            (r'pd\.', 'pandas'),
            (r'np\.', 'numpy'),
            (r'plt\.', 'matplotlib.pyplot'),
            (r'st\.', 'streamlit'),
            (r'requests\.', 'requests'),
            (r'json\.', 'json'),
            (r'datetime\.', 'datetime'),
            (r'timedelta', 'datetime'),
        ]
        
        for pattern, module in patterns:
            if re.search(pattern, content) and f'import {module}' not in content and f'from {module}' not in content:
                missing_imports.append(module)
        
        return missing_imports
    
    def _check_todo_items(self):
        """Check for remaining TODO items."""
        logger.info("Checking for TODO items...")
        
        python_files = list(self.project_root.rglob("*.py"))
        todo_count = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count TODO items
                todos = re.findall(r'#\s*TODO', content, re.IGNORECASE)
                if todos:
                    todo_count += len(todos)
                    logger.warning(f"Found {len(todos)} TODO items in {file_path}")
                
            except Exception as e:
                logger.error(f"Error checking TODO items in {file_path}: {e}")
        
        if todo_count > 0:
            logger.warning(f"Total TODO items found: {todo_count}")
        else:
            logger.info("No TODO items found - code is clean!")
    
    def _generate_cleanup_report(self) -> Dict[str, Any]:
        """Generate cleanup report."""
        return {
            'cleaned_files': len(self.cleaned_files),
            'removed_files': len(self.removed_files),
            'errors': len(self.errors),
            'details': {
                'cleaned_files': self.cleaned_files,
                'removed_files': self.removed_files,
                'errors': self.errors
            }
        }

def main():
    """Main cleanup function."""
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run cleanup
    cleanup = ProductionCleanup(project_root)
    report = cleanup.run_full_cleanup()
    
    # Print summary
    print("\n" + "="*60)
    print("PRODUCTION CLEANUP SUMMARY")
    print("="*60)
    print(f"Files cleaned: {report['cleaned_files']}")
    print(f"Files removed: {report['removed_files']}")
    print(f"Errors encountered: {report['errors']}")
    
    if report['errors'] > 0:
        print("\nErrors:")
        for error in report['details']['errors']:
            print(f"  - {error}")
    
    print("\n" + "="*60)
    print("Cleanup completed!")
    print("="*60)

if __name__ == "__main__":
    main() 