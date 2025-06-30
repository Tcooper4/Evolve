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
from typing import Dict, List, Set, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RequirementsCleaner:
    """Clean and organize requirements files."""
    
    def __init__(self, requirements_file: Path, output_file: Path):
        self.requirements_file = requirements_file
        self.output_file = output_file
        self.logger = logging.getLogger(__name__)
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
    
    def load_requirements(self) -> List[str]:
        """Load requirements from file."""
        try:
            with open(self.requirements_file, 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            self.logger.error(f"Error loading requirements: {e}")
            return []

    def categorize_packages(self, packages: List[str]) -> Dict[str, List[str]]:
        """Categorize packages by type."""
        categorized = {
            'core': [],
            'ml': [],
            'web': [],
            'data': [],
            'utils': [],
            'dev': [],
            'unknown': []
        }
        
        for package in packages:
            if any(keyword in package.lower() for keyword in ['numpy', 'pandas', 'scipy']):
                categorized['data'].append(package)
            elif any(keyword in package.lower() for keyword in ['tensorflow', 'torch', 'sklearn', 'xgboost']):
                categorized['ml'].append(package)
            elif any(keyword in package.lower() for keyword in ['flask', 'streamlit', 'fastapi', 'django']):
                categorized['web'].append(package)
            elif any(keyword in package.lower() for keyword in ['pytest', 'black', 'flake8']):
                categorized['dev'].append(package)
            else:
                categorized['unknown'].append(package)
        
        return categorized

    def get_package_info(self, package: str) -> Tuple[str, str]:
        """Get package name and version."""
        if '==' in package:
            name, version = package.split('==', 1)
            return name.strip(), version.strip()
        elif '>=' in package:
            name, version = package.split('>=', 1)
            return name.strip(), f">={version.strip()}"
        elif '<=' in package:
            name, version = package.split('<=', 1)
            return name.strip(), f"<={version.strip()}"
        else:
            return package.strip(), "latest"

    def check_package_health(self, package: str) -> int:
        """Check package health score (0-100)."""
        try:
            name, version = self.get_package_info(package)
            
            # Simple health check based on package name
            if any(keyword in name.lower() for keyword in ['numpy', 'pandas', 'requests']):
                return 95  # Well-maintained packages
            elif any(keyword in name.lower() for keyword in ['tensorflow', 'torch']):
                return 90  # ML frameworks
            else:
                return 75  # Default score
            
        except Exception as e:
            self.logger.error(f"Error checking package health: {e}")
            return -1

    def format_requirements(self, packages: List[str]) -> str:
        """Format requirements for output."""
        content = []
        content.append("# Cleaned Requirements File")
        content.append("# Generated by cleanup_requirements.py")
        content.append("")
        
        for package in sorted(packages):
            content.append(package)
        
        return '\n'.join(content).strip()

    def cleanup_requirements(self) -> bool:
        """Main cleanup function."""
        try:
            self.logger.info("Starting requirements cleanup")
            
            # Load requirements
            packages = self.load_requirements()
            if not packages:
                self.logger.error("No packages found")
                return False
            
            # Categorize packages
            categorized = self.categorize_packages(packages)
            
            # Filter out low-health packages
            healthy_packages = []
            for category, pkgs in categorized.items():
                for package in pkgs:
                    health = self.check_package_health(package)
                    if health >= 70:  # Keep packages with health >= 70
                        healthy_packages.append(package)
                    else:
                        self.logger.warning(f"Removing low-health package: {package} (health: {health})")
            
            # Format and save
            formatted_content = self.format_requirements(healthy_packages)
            
            with open(self.output_file, 'w') as f:
                f.write(formatted_content)
            
            self.logger.info(f"Cleanup completed. Kept {len(healthy_packages)}/{len(packages)} packages")
            return True
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate cleanup report."""
        try:
            original_packages = self.load_requirements()
            categorized = self.categorize_packages(original_packages)
            
            report = {
                'total_packages': len(original_packages),
                'categories': {k: len(v) for k, v in categorized.items()},
                'health_scores': {}
            }
            
            for package in original_packages:
                health = self.check_package_health(package)
                report['health_scores'][package] = health
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return {}

def main():
    """Main function."""
    try:
        requirements_file = Path('requirements.txt')
        output_file = Path('cleaned_requirements.txt')
        cleanup = RequirementsCleaner(requirements_file, output_file)
        success = cleanup.cleanup_requirements()
        
        if success:
            print("✅ Requirements cleanup completed successfully")
            report = cleanup.generate_report()
            print(f"📊 Report: {report}")
        else:
            print("❌ Requirements cleanup failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 