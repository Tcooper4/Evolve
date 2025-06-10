import pytest
import sys
import os
from pathlib import Path

def main():
    """Run all tests in the test directory."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Add the project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Get the test directory
    test_dir = project_root / "tests"
    
    # Run pytest with verbosity and coverage
    args = [
        str(test_dir),
        "-v",  # Verbose output
        "--cov=trading",  # Measure code coverage
        "--cov-report=term-missing",  # Show missing lines in coverage report
        "--cov-report=html",  # Generate HTML coverage report
        "--cov-fail-under=80",  # Fail if coverage is below 80%
        "-W", "ignore::DeprecationWarning",  # Ignore deprecation warnings
        "-W", "ignore::UserWarning",  # Ignore user warnings
    ]
    
    # Add any command line arguments
    args.extend(sys.argv[1:])
    
    # Run the tests
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(main()) 