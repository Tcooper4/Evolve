#!/usr/bin/env python
"""
Test health check script.

This script:
1. Checks for missing __init__.py files in test directories
2. Runs pytest --collect-only to check for import errors
3. Reports on test discovery and potential issues
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Set, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_test_directories(root_dir: str) -> List[Path]:
    """Find all test directories."""
    test_dirs = []
    for path in Path(root_dir).rglob('*'):
        if path.is_dir() and 'test' in path.name.lower():
            test_dirs.append(path)
    return test_dirs

def check_init_files(test_dirs: List[Path]) -> Dict[str, List[str]]:
    """Check for missing __init__.py files."""
    missing_init = []
    for test_dir in test_dirs:
        init_file = test_dir / '__init__.py'
        if not init_file.exists():
            missing_init.append(str(test_dir))
    return {'missing_init': missing_init}

def run_pytest_collect() -> Dict[str, List[str]]:
    """Run pytest --collect-only and capture output."""
    try:
        result = subprocess.run(
            ['pytest', '--collect-only', '-v'],
            capture_output=True,
            text=True,
            check=True
        )
        return {'success': True, 'output': result.stdout.splitlines()}
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'output': e.stdout.splitlines(),
            'error': e.stderr.splitlines()
        }

def check_test_files(test_dirs: List[Path]) -> Dict[str, List[str]]:
    """Check for test files and their imports."""
    test_files = []
    import_errors = []
    
    for test_dir in test_dirs:
        for path in test_dir.rglob('test_*.py'):
            test_files.append(str(path))
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'import' in content:
                        # Basic import check
                        if 'trading.' not in content and 'from trading' not in content:
                            import_errors.append(f"{path}: No trading package imports found")
            except Exception as e:
                import_errors.append(f"{path}: Error reading file - {str(e)}")
    
    return {
        'test_files': test_files,
        'import_errors': import_errors
    }

def main():
    """Main function to run all checks."""
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    logger.info("Starting test health check...")
    
    # Find test directories
    test_dirs = find_test_directories('tests')
    logger.info(f"Found {len(test_dirs)} test directories")
    
    # Check for missing __init__.py files
    init_check = check_init_files(test_dirs)
    if init_check['missing_init']:
        logger.warning("Missing __init__.py files in:")
        for dir_path in init_check['missing_init']:
            logger.warning(f"  - {dir_path}")
    else:
        logger.info("All test directories have __init__.py files")
    
    # Run pytest collect
    logger.info("Running pytest --collect-only...")
    collect_result = run_pytest_collect()
    if collect_result['success']:
        logger.info("Test collection successful")
        for line in collect_result['output']:
            logger.info(f"  {line}")
    else:
        logger.error("Test collection failed")
        for line in collect_result.get('error', []):
            logger.error(f"  {line}")
    
    # Check test files
    logger.info("Checking test files...")
    test_check = check_test_files(test_dirs)
    logger.info(f"Found {len(test_check['test_files'])} test files")
    if test_check['import_errors']:
        logger.warning("Potential import issues found:")
        for error in test_check['import_errors']:
            logger.warning(f"  - {error}")
    
    logger.info("Test health check complete")

if __name__ == '__main__':
    main() 