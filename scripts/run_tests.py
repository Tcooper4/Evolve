"""Script to run tests with proper configuration."""

import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run tests with proper configuration."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Set up environment variables
    os.environ['PYTHONPATH'] = str(project_root)
    os.environ['PYTEST_ADDOPTS'] = '--tb=short -v'
    
    # Run pytest
    try:
        subprocess.check_call([
            sys.executable,
            '-m',
            'pytest',
            'tests/',
            '--cov=trading',
            '--cov-report=term-missing',
            '--cov-report=html',
            '--junitxml=test-results.xml',
            '--html=test-report.html'
        ])
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    """Main function to run tests."""
    print("Running tests...")
    run_tests()
    print("Tests completed!")

if __name__ == '__main__':
    main() 