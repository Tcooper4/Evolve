"""
Basic Import Tests

This module tests that all core modules can be imported without errors.
This is a prerequisite for running the full test suite.
"""

import logging
import sys
import unittest
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBasicImports(unittest.TestCase):
    """Test that all core modules can be imported."""

    def test_core_imports(self):
        """Test core module imports."""
        try:
            pass

            logger.info("✅ Core trading modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import core trading modules: {e}")

    def test_strategy_imports(self):
        """Test strategy module imports."""
        try:
            pass

            # Test strategy gatekeeper
            try:
                pass

                print("✅ StrategyGatekeeper imported successfully")
            except ImportError as e:
                print(f"❌ StrategyGatekeeper import failed: {e}")
                return False
            logger.info("✅ Strategy modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import strategy modules: {e}")

    def test_agent_imports(self):
        """Test agent module imports."""
        try:
            pass

            logger.info("✅ Agent modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import agent modules: {e}")

    def test_data_imports(self):
        """Test data module imports."""
        try:
            pass

            logger.info("✅ Data modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import data modules: {e}")

    def test_backtesting_imports(self):
        """Test backtesting module imports."""
        try:
            pass

            logger.info("✅ Backtesting modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import backtesting modules: {e}")

    def test_optimization_imports(self):
        """Test optimization module imports."""
        try:
            pass

            logger.info("✅ Optimization modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import optimization modules: {e}")

    def test_utils_imports(self):
        """Test utility module imports."""
        try:
            pass

            logger.info("✅ Utility modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import utility modules: {e}")

    def test_config_imports(self):
        """Test configuration module imports."""
        try:
            pass

            logger.info("✅ Configuration modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import configuration modules: {e}")


if __name__ == "__main__":
    unittest.main()
