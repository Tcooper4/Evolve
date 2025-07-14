"""Script to set up the test environment."""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install test requirements."""
    requirements_file = Path(__file__).parent.parent / "requirements-test.txt"
    if requirements_file.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
    else:
        print("Warning: requirements-test.txt not found")


def create_test_directories():
    """Create necessary test directories."""
    test_dirs = [
        "test_model_save",
        "tests/fixtures/data",
        "tests/fixtures/models",
        "tests/fixtures/configs",
    ]

    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main function to set up test environment."""
    print("Setting up test environment...")

    # Install requirements
    install_requirements()

    # Create test directories
    create_test_directories()

    print("Test environment setup complete!")


if __name__ == "__main__":
    main()
