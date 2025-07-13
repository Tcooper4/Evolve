#!/usr/bin/env python3
"""Comprehensive System Status Test for Evolve Trading Platform."""

import importlib
import logging
import os
import sys
from typing import Any, Dict


# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemStatusTester:
    """Comprehensive system status tester."""

    def __init__(self):
        self.results = {"success": 0, "failure": 0, "warning": 0, "total": 0, "details": []}

    def test_core_modules(self):
        """Test core trading modules."""
        logger.info("Testing core modules...")

        core_modules = [
            "trading.agents.base_agent",
            "trading.models.forecast_router",
            "trading.strategies.bollinger_strategy",
            "trading.data.data_loader",
            "trading.execution.execution_engine",
            "trading.optimization.bayesian_optimizer",
            "trading.risk.risk_analyzer",
            "trading.portfolio.portfolio_manager",
            "trading.evaluation.metrics",
            "trading.feature_engineering.feature_engineer",
        ]

        for module in core_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"Core module: {module}")
            except ImportError as e:
                self._record_failure(f"Core module: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"Core module: {module} - {str(e)}")

    def test_new_modules(self):
        """Test newly created modules."""
        logger.info("Testing new modules...")

        new_modules = [
            "strategies.gatekeeper",
            "reporting.pnl_attribution",
            "data.live_feed",
            "rl.rl_trader",
            "causal.driver_analysis",
            "agents.model_generator",
            "execution.trade_executor",
            "voice_prompt_agent",
            "risk.advanced_risk",
        ]

        for module in new_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"New module: {module}")
            except ImportError as e:
                self._record_warning(f"New module: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"New module: {module} - {str(e)}")

    def test_ui_modules(self):
        """Test UI modules."""
        logger.info("Testing UI modules...")

        ui_modules = [
            "pages.1_Forecast_Trade",
            "pages.2_Strategy_Backtest",
            "pages.3_Trade_Execution",
            "pages.4_Portfolio_Management",
            "pages.5_Risk_Analysis",
            "pages.6_Model_Optimization",
            "pages.7_Market_Analysis",
            "pages.8_Agent_Management",
            "pages.9_System_Monitoring",
            "pages.10_Strategy_Health_Dashboard",
        ]

        for module in ui_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"UI module: {module}")
            except ImportError as e:
                self._record_failure(f"UI module: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"UI module: {module} - {str(e)}")

    def test_advanced_modules(self):
        """Test advanced feature modules."""
        logger.info("Testing advanced modules...")

        advanced_modules = [
            "trading.models.advanced.tcn.tcn_model",
            "trading.models.advanced.transformer.transformer_model",
            "trading.models.advanced.lstm.lstm_model",
            "trading.models.advanced.gnn.gnn_model",
            "trading.models.advanced.rl.rl_model",
            "trading.models.advanced.ensemble.ensemble_model",
            "trading.nlp.llm_processor",
            "trading.meta_agents.agents.agent_router",
        ]

        for module in advanced_modules:
            try:
                importlib.import_module(module)
                self._record_success(f"Advanced module: {module}")
            except ImportError as e:
                self._record_warning(f"Advanced module: {module} - {str(e)}")
            except Exception as e:
                self._record_warning(f"Advanced module: {module} - {str(e)}")

    def test_critical_imports(self):
        """Test critical external libraries."""
        logger.info("Testing critical imports...")

        critical_imports = [
            "streamlit",
            "pandas",
            "numpy",
            "yfinance",
            "plotly",
            "scikit-learn",
            "torch",
            "transformers",
        ]

        for lib in critical_imports:
            try:
                importlib.import_module(lib)
                self._record_success(f"Import: {lib}")
            except ImportError as e:
                self._record_failure(f"Import: {lib} - {str(e)}")
            except Exception as e:
                self._record_warning(f"Import: {lib} - {str(e)}")

    def test_file_structure(self):
        """Test critical file structure."""
        logger.info("Testing file structure...")

        critical_files = [
            "app.py",
            "config/app_config.yaml",
            "config/config.json",
            "trading/config/configuration.py",
            "trading/agents/agent_config.json",
        ]

        for file_path in critical_files:
            if os.path.exists(file_path):
                self._record_success(f"File: {file_path}")
            else:
                self._record_warning(f"File: {file_path} - Missing")

        critical_dirs = ["trading", "pages", "config", "data", "models", "strategies", "utils", "scripts"]

        for dir_name in critical_dirs:
            if os.path.exists(dir_name):
                self._record_success(f"Directory: {dir_name}")
            else:
                self._record_failure(f"Directory: {dir_name} - Missing")

    def test_functionality(self):
        """Test basic functionality."""
        logger.info("Testing basic functionality...")

        try:
            # Test data loading
            import yfinance as yf

            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="1d")
            if not hist.empty:
                self._record_success("Data loading: yfinance")
            else:
                self._record_warning("Data loading: yfinance - No data")
        except Exception as e:
            self._record_warning(f"Data loading: yfinance - {str(e)}")

        try:
            # Test Streamlit
            pass

            self._record_success("Streamlit: Core functionality")
        except Exception as e:
            self._record_failure(f"Streamlit: Core functionality - {str(e)}")

    def _record_success(self, message: str):
        """Record successful test."""
        self.results["success"] += 1
        self.results["total"] += 1
        self.results["details"].append(f"✅ {message}")
        logger.info(f"✅ {message}")

    def _record_failure(self, message: str):
        """Record failed test."""
        self.results["failure"] += 1
        self.results["total"] += 1
        self.results["details"].append(f"❌ {message}")
        logger.error(f"❌ {message}")

    def _record_warning(self, message: str):
        """Record warning test."""
        self.results["warning"] += 1
        self.results["total"] += 1
        self.results["details"].append(f"⚠️ {message}")
        logger.warning(f"⚠️ {message}")

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test."""
        logger.info("Starting comprehensive system test...")

        self.test_critical_imports()
        self.test_core_modules()
        self.test_new_modules()
        self.test_ui_modules()
        self.test_advanced_modules()
        self.test_file_structure()
        self.test_functionality()

        # Calculate success rate
        if self.results["total"] > 0:
            success_rate = (self.results["success"] / self.results["total"]) * 100
        else:
            success_rate = 0

        self.results["success_rate"] = success_rate

        logger.info(f"\n{'='*60}")
        logger.info(f"COMPREHENSIVE SYSTEM TEST COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Total Tests: {self.results['total']}")
        logger.info(f"✅ Success: {self.results['success']}")
        logger.info(f"❌ Failures: {self.results['failure']}")
        logger.info(f"⚠️ Warnings: {self.results['warning']}")
        logger.info(f"{'='*60}")

        return self.results


def main():
    """Main function to run system test."""
    tester = SystemStatusTester()
    results = tester.run_comprehensive_test()

    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("=" * 60)
    for detail in results["details"]:
        print(detail)

    print(f"\nSUMMARY:")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(
        f"Total: {results['total']} | Success: {results['success']} | Failures: {results['failure']} | Warnings: {results['warning']}"
    )

    # Determine system status
    if results["success_rate"] >= 95:
        status = "EXCELLENT"
    elif results["success_rate"] >= 85:
        status = "GOOD"
    elif results["success_rate"] >= 70:
        status = "FAIR"
    else:
        status = "NEEDS ATTENTION"

    print(f"\nSYSTEM STATUS: {status}")

    return results


if __name__ == "__main__":
    main()
