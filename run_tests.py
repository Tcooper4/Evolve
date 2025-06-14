"""Test runner script with coverage and code formatting options."""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(fix_code=False):
    """Run tests with optional code formatting.
    
    Args:
        fix_code: Whether to run code formatters
    """
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Run code formatters if requested
    if fix_code:
        print("Running code formatters...")
        
        # Run black
        subprocess.run([
            sys.executable, "-m", "black",
            str(project_root / "trading"),
            str(project_root / "tests")
        ], check=True)
        
        # Run isort
        subprocess.run([
            sys.executable, "-m", "isort",
            str(project_root / "trading"),
            str(project_root / "tests")
        ], check=True)
        
        # Run flake8
        subprocess.run([
            sys.executable, "-m", "flake8",
            str(project_root / "trading"),
            str(project_root / "tests")
        ], check=True)
    
    # Run pytest with coverage
    print("\nRunning tests with coverage...")
    subprocess.run([
        sys.executable, "-m", "pytest",
        "--cov=trading",
        "--cov-report=term-missing",
        "--cov-report=html",
        str(project_root / "tests")
    ], check=True)

def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run tests with optional code formatting")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Run code formatters (black, isort, flake8)"
    )
    args = parser.parse_args()
    
    try:
        run_tests(fix_code=args.fix)
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 