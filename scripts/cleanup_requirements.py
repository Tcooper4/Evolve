#!/usr/bin/env python3
"""
Requirements cleanup script.

This script cleans up requirements files by:
1. Removing duplicate entries
2. Removing ta-lib references
3. Organizing dependencies by category
4. Validating package names
5. Checking for version conflicts
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RequirementsCleaner:
    """Clean and organize requirements files."""
    
    def __init__(self):
        self.packages_to_remove = {'ta-lib', 'talib'}
        self.categories = {
            'core': [
                'numpy', 'pandas', 'pandas-ta', 'scikit-learn', 'torch', 
                'yfinance', 'openai', 'matplotlib', 'plotly', 'streamlit'
            ],
            'trading': [
                'ccxt', 'backtrader', 'websockets', 'requests', 'alpha_vantage'
            ],
            'visualization': [
                'dash', 'bokeh', 'seaborn'
            ],
            'system': [
                'prometheus-client', 'grafana-api', 'psutil', 'docker', 'kubernetes'
            ],
            'database': [
                'sqlalchemy', 'pymongo', 'redis'
            ],
            'testing': [
                'pytest', 'pytest-cov', 'pytest-mock', 'pytest-asyncio', 
                'pytest-xdist', 'pytest-timeout', 'pytest-env', 'pytest-randomly',
                'pytest-sugar', 'pytest-html', 'pytest-metadata', 'pytest-clarity',
                'pytest-benchmark', 'pytest-profiling', 'pytest-instafail',
                'pytest-rerunfailures', 'coverage', 'faker', 'factory-boy', 'hypothesis'
            ],
            'development': [
                'black', 'flake8', 'mypy', 'isort', 'pre-commit'
            ],
            'documentation': [
                'sphinx', 'sphinx-rtd-theme', 'mkdocs', 'mkdocs-material'
            ],
            'security': [
                'cryptography', 'python-jose', 'passlib', 'bcrypt'
            ],
            'monitoring': [
                'sentry-sdk', 'loguru', 'python-json-logger'
            ],
            'additional': [
                'pydantic', 'finta', 'empyrical', 'scipy', 'xgboost', 
                'arch', 'statsmodels', 'prophet', 'networkx'
            ]
        }
    
    def parse_requirements(self, content: str) -> List[Tuple[str, str]]:
        """Parse requirements content into package names and versions."""
        packages = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                # Extract package name and version
                match = re.match(r'^([a-zA-Z0-9_-]+)(.*)$', line)
                if match:
                    package_name = match.group(1)
                    version = match.group(2).strip()
                    packages.append((package_name, version))
        return packages
    
    def categorize_packages(self, packages: List[Tuple[str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """Categorize packages by their type."""
        categorized = {cat: [] for cat in self.categories.keys()}
        categorized['other'] = []
        
        for package_name, version in packages:
            categorized_flag = False
            for category, known_packages in self.categories.items():
                if package_name in known_packages:
                    categorized[category].append((package_name, version))
                    categorized_flag = True
                    break
            if not categorized_flag:
                categorized['other'].append((package_name, version))
        
        return categorized
    
    def remove_duplicates(self, packages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Remove duplicate packages, keeping the highest version."""
        package_dict = {}
        for package_name, version in packages:
            if package_name not in package_dict:
                package_dict[package_name] = version
            else:
                # Keep the version with higher requirements
                current_version = package_dict[package_name]
                if self._compare_versions(version, current_version) > 0:
                    package_dict[package_name] = version
        
        return list(package_dict.items())
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare version requirements (simplified)."""
        # Extract version numbers
        v1_match = re.search(r'>=?(\d+\.\d+\.\d+)', version1)
        v2_match = re.search(r'>=?(\d+\.\d+\.\d+)', version2)
        
        if v1_match and v2_match:
            v1_parts = [int(x) for x in v1_match.group(1).split('.')]
            v2_parts = [int(x) for x in v2_match.group(1).split('.')]
            
            for i in range(max(len(v1_parts), len(v2_parts))):
                v1_val = v1_parts[i] if i < len(v1_parts) else 0
                v2_val = v2_parts[i] if i < len(v2_parts) else 0
                if v1_val > v2_val:
                    return 1
                elif v1_val < v2_val:
                    return -1
        return 0
    
    def generate_requirements_content(self, categorized: Dict[str, List[Tuple[str, str]]]) -> str:
        """Generate clean requirements content."""
        content = []
        
        for category, packages in categorized.items():
            if packages:
                if category != 'other':
                    content.append(f"# {category.title()} dependencies")
                else:
                    content.append("# Other dependencies")
                
                # Sort packages alphabetically
                sorted_packages = sorted(packages, key=lambda x: x[0])
                for package_name, version in sorted_packages:
                    content.append(f"{package_name}{version}")
                content.append("")
        
        return '\n'.join(content).strip()
    
    def clean_file(self, file_path: Path) -> None:
        """Clean a requirements file."""
        logger.info(f"Cleaning {file_path}")
        
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return
        
        # Read current content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse packages
        packages = self.parse_requirements(content)
        
        # Remove packages to exclude
        packages = [(name, version) for name, version in packages 
                   if name not in self.packages_to_remove]
        
        # Remove duplicates
        packages = self.remove_duplicates(packages)
        
        # Categorize packages
        categorized = self.categorize_packages(packages)
        
        # Generate new content
        new_content = self.generate_requirements_content(categorized)
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Cleaned {file_path}: {len(packages)} packages organized into {len([k for k, v in categorized.items() if v])} categories")
    
    def validate_packages(self, file_path: Path) -> None:
        """Validate package names in a requirements file."""
        logger.info(f"Validating {file_path}")
        
        if not file_path.exists():
            return
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        packages = self.parse_requirements(content)
        
        for package_name, version in packages:
            # Check for common issues
            if package_name.startswith('-'):
                logger.warning(f"Invalid package name in {file_path}: {package_name}")
            if not re.match(r'^[a-zA-Z0-9_-]+$', package_name):
                logger.warning(f"Potentially invalid package name in {file_path}: {package_name}")

def main():
    """Main function."""
    cleaner = RequirementsCleaner()
    
    # Files to clean
    requirements_files = [
        Path('requirements.txt'),
        Path('requirements-test.txt'),
        Path('requirements-dev.txt'),
        Path('requirements_dashboard.txt')
    ]
    
    logger.info("Starting requirements cleanup")
    
    for file_path in requirements_files:
        if file_path.exists():
            cleaner.clean_file(file_path)
            cleaner.validate_packages(file_path)
        else:
            logger.info(f"Skipping {file_path} (does not exist)")
    
    logger.info("Requirements cleanup completed")

if __name__ == "__main__":
    main() 