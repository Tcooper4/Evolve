"""Script to run tests with proper configuration and security."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_env_var_safe(key: str, default: Optional[str] = None) -> str:
    """
    Safely get environment variable with proper error handling.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value

    Raises:
        EnvironmentError: If required environment variable is missing
    """
    value = os.environ.get(key, default)
    if value is None:
        raise EnvironmentError(f"Required environment variable '{key}' is missing")
    return value


def validate_environment():
    """Validate required environment variables."""
    required_vars = ["PYTHONPATH", "PYTEST_ADDOPTS"]

    optional_vars = ["ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY", "POLYGON_API_KEY", "OPENAI_API_KEY"]

    # Check required variables
    for var in required_vars:
        try:
            get_env_var_safe(var)
            logger.info(f"✅ {var} is configured")
        except EnvironmentError as e:
            logger.warning(f"⚠️ {e}")

    # Check optional variables
    for var in optional_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"✅ {var} is configured: {value[:10]}...")
        else:
            logger.info(f"ℹ️ {var} is not configured (optional)")


def run_tests():
    """Run tests with proper configuration and security."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Validate environment
    validate_environment()

    # Set up environment variables safely
    os.environ["PYTHONPATH"] = str(project_root)
    os.environ["PYTEST_ADDOPTS"] = "--tb=short -v"

    # Run pytest with enhanced security
    try:
        logger.info("Starting test execution...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--cov=trading",
                "--cov-report=term-missing",
                "--cov-report=html",
                "--junitxml=test-results.xml",
                "--html=test-report.html",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )  # 5 minute timeout

        if result.returncode == 0:
            logger.info("✅ All tests passed successfully!")
            if result.stdout:
                logger.info("Test output:\n" + result.stdout[-1000:])  # Last 1000 chars
        else:
            logger.error(f"❌ Tests failed with exit code {result.returncode}")
            if result.stderr:
                logger.error("Error output:\n" + result.stderr)
            sys.exit(result.returncode)

    except subprocess.TimeoutExpired:
        logger.error("❌ Test execution timed out after 5 minutes")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"❌ Unexpected error during test execution: {e}")
        sys.exit(1)


def main():
    """Main function to run tests with enhanced security."""
    logger.info("🚀 Starting Evolve AI test suite...")

    try:
        run_tests()
        logger.info("✅ Test suite completed successfully!")
    except KeyboardInterrupt:
        logger.info("⚠️ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
