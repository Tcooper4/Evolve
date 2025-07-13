#!/usr/bin/env python3
"""Quick System Check for Evolve Trading Platform"""

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_imports() -> Dict[str, bool]:
    """Check critical imports."""
    results = {}

    critical_modules = ["streamlit", "pandas", "numpy", "yfinance", "plotly", "scikit-learn", "torch", "transformers"]

    for module in critical_modules:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError:
            results[module] = False

    return results


def check_core_modules() -> Dict[str, bool]:
    """Check core Evolve modules."""
    results = {}

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
            results[module] = True
        except ImportError:
            results[module] = False

    return results


def check_advanced_modules() -> Dict[str, bool]:
    """Check advanced modules."""
    results = {}

    advanced_modules = [
        "causal.causal_model",
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
            results[module] = True
        except ImportError:
            results[module] = False

    return results


def check_ui_modules() -> Dict[str, bool]:
    """Check UI modules."""
    results = {}

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
            results[module] = True
        except ImportError:
            results[module] = False

    return results


def check_config_loading() -> Dict[str, any]:
    """Check configuration loading from .env and fallback JSON."""
    results = {
        "env_loading": False,
        "json_fallback": False,
        "config_validation": False,
        "required_vars": {},
        "optional_vars": {},
        "config_paths": [],
        "errors": [],
    }

    try:
        # Check for .env file
        env_file = Path(".env")
        if env_file.exists():
            results["env_loading"] = True
            results["config_paths"].append(str(env_file.absolute()))
            logger.info(f"Found .env file: {env_file.absolute()}")
        else:
            results["errors"].append("No .env file found")

        # Check for JSON config files
        json_config_paths = [
            "config/app_config.json",
            "config/config.json",
            "config/settings.json",
            "config.json",
            "settings.json",
        ]

        for config_path in json_config_paths:
            config_file = Path(config_path)
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config_data = json.load(f)

                    results["json_fallback"] = True
                    results["config_paths"].append(str(config_file.absolute()))
                    logger.info(f"Found JSON config: {config_file.absolute()}")

                    # Validate config structure
                    if validate_config_structure(config_data):
                        results["config_validation"] = True
                    else:
                        results["errors"].append(f"Invalid config structure in {config_path}")

                except json.JSONDecodeError as e:
                    results["errors"].append(f"Invalid JSON in {config_path}: {e}")
                except Exception as e:
                    results["errors"].append(f"Error reading {config_path}: {e}")

        # Check required environment variables
        required_vars = ["OPENAI_API_KEY", "YAHOO_FINANCE_API_KEY", "REDIS_HOST", "REDIS_PORT", "DATABASE_URL"]

        for var in required_vars:
            value = os.getenv(var)
            if value:
                results["required_vars"][var] = True
            else:
                results["required_vars"][var] = False
                results["errors"].append(f"Missing required environment variable: {var}")

        # Check optional environment variables
        optional_vars = ["DEBUG", "LOG_LEVEL", "ENVIRONMENT", "APP_VERSION", "CORS_ORIGINS", "RATE_LIMIT"]

        for var in optional_vars:
            value = os.getenv(var)
            results["optional_vars"][var] = value is not None

        # Check config utility functions
        if check_config_utility_functions():
            results["config_validation"] = True
        else:
            results["errors"].append("Config utility functions not working properly")

    except Exception as e:
        results["errors"].append(f"Error during config check: {e}")

    return results


def validate_config_structure(config_data: dict) -> bool:
    """Validate configuration structure."""
    try:
        # Check for required top-level keys
        required_keys = ["app", "database", "redis", "api"]

        for key in required_keys:
            if key not in config_data:
                logger.warning(f"Missing required config key: {key}")
                return False

        # Check app configuration
        if "app" in config_data:
            app_config = config_data["app"]
            app_required = ["name", "version", "environment"]

            for key in app_required:
                if key not in app_config:
                    logger.warning(f"Missing required app config key: {key}")
                    return False

        # Check database configuration
        if "database" in config_data:
            db_config = config_data["database"]
            db_required = ["url", "type"]

            for key in db_required:
                if key not in db_config:
                    logger.warning(f"Missing required database config key: {key}")
                    return False

        # Check Redis configuration
        if "redis" in config_data:
            redis_config = config_data["redis"]
            redis_required = ["host", "port"]

            for key in redis_required:
                if key not in redis_config:
                    logger.warning(f"Missing required Redis config key: {key}")
                    return False

        return True

    except Exception as e:
        logger.error(f"Error validating config structure: {e}")
        return False


def check_config_utility_functions() -> bool:
    """Check if config utility functions are working."""
    try:
        # Try to import config utilities
        config_modules = ["utils.config_loader", "config.app_config", "trading.config.config_manager"]

        for module_name in config_modules:
            try:
                module = importlib.import_module(module_name)

                # Check if module has expected functions
                if hasattr(module, "load_config") or hasattr(module, "get_config"):
                    logger.info(f"Config utility found: {module_name}")
                    return True

            except ImportError:
                continue

        # If no config utilities found, check for basic config loading
        if os.getenv("OPENAI_API_KEY") or os.getenv("YAHOO_FINANCE_API_KEY"):
            logger.info("Basic environment variable loading working")
            return True

        return False

    except Exception as e:
        logger.error(f"Error checking config utilities: {e}")
        return False


def check_file_permissions() -> Dict[str, bool]:
    """Check file permissions for critical directories."""
    results = {}

    critical_dirs = ["logs", "data", "cache", "backups", "reports"]

    for dir_name in critical_dirs:
        dir_path = Path(dir_name)

        # Check if directory exists
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                results[f"{dir_name}_created"] = True
            except Exception as e:
                results[f"{dir_name}_created"] = False
                logger.error(f"Could not create directory {dir_name}: {e}")
        else:
            results[f"{dir_name}_exists"] = True

        # Check write permissions
        if dir_path.exists():
            try:
                test_file = dir_path / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                results[f"{dir_name}_writable"] = True
            except Exception as e:
                results[f"{dir_name}_writable"] = False
                logger.error(f"Directory {dir_name} not writable: {e}")

    return results


def main():
    """Run system check."""
    print("🔍 EVOLVE TRADING PLATFORM - SYSTEM CHECK")
    print("=" * 60)

    # Check imports
    print("\n📦 CRITICAL IMPORTS:")
    import_results = check_imports()
    import_success = sum(import_results.values())
    import_total = len(import_results)

    for module, success in import_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")

    # Check core modules
    print("\n🏗️ CORE MODULES:")
    core_results = check_core_modules()
    core_success = sum(core_results.values())
    core_total = len(core_results)

    for module, success in core_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")

    # Check advanced modules
    print("\n🚀 ADVANCED MODULES:")
    advanced_results = check_advanced_modules()
    advanced_success = sum(advanced_results.values())
    advanced_total = len(advanced_results)

    for module, success in advanced_results.items():
        status = "✅" if success else "⚠️"
        print(f"  {status} {module}")

    # Check UI modules
    print("\n🖥️ UI MODULES:")
    ui_results = check_ui_modules()
    ui_success = sum(ui_results.values())
    ui_total = len(ui_results)

    for module, success in ui_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {module}")

    # Check configuration loading
    print("\n⚙️ CONFIGURATION:")
    config_results = check_config_loading()

    # Environment variables
    print("  Environment Variables:")
    for var, success in config_results["required_vars"].items():
        status = "✅" if success else "❌"
        print(f"    {status} {var}")

    for var, success in config_results["optional_vars"].items():
        status = "✅" if success else "⚠️"
        print(f"    {status} {var}")

    # Config files
    print("  Configuration Files:")
    if config_results["env_loading"]:
        print("    ✅ .env file found")
    else:
        print("    ❌ .env file not found")

    if config_results["json_fallback"]:
        print("    ✅ JSON config files found")
    else:
        print("    ❌ No JSON config files found")

    if config_results["config_validation"]:
        print("    ✅ Config validation passed")
    else:
        print("    ❌ Config validation failed")

    # Check file permissions
    print("\n📁 FILE PERMISSIONS:")
    permission_results = check_file_permissions()

    for permission, success in permission_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {permission}")

    # Calculate overall success rate
    total_success = import_success + core_success + advanced_success + ui_success
    total_checks = import_total + core_total + advanced_total + ui_total

    # Add config and permission checks
    config_success = sum(
        [config_results["env_loading"], config_results["json_fallback"], config_results["config_validation"]]
    )
    config_total = 3

    permission_success = sum(permission_results.values())
    permission_total = len(permission_results)

    total_success += config_success + permission_success
    total_checks += config_total + permission_total

    success_rate = (total_success / total_checks) * 100 if total_checks > 0 else 0

    print("\n" + "=" * 60)
    print(f"📊 OVERALL SUCCESS RATE: {success_rate:.1f}%")
    print(f"✅ Success: {total_success}/{total_checks}")
    print(f"❌ Failures: {total_checks - total_success}")

    # Show configuration errors if any
    if config_results["errors"]:
        print(f"\n⚠️ CONFIGURATION ERRORS:")
        for error in config_results["errors"]:
            print(f"  - {error}")

    print("=" * 60)

    return {
        "success_rate": success_rate,
        "import_results": import_results,
        "core_results": core_results,
        "advanced_results": advanced_results,
        "ui_results": ui_results,
        "config_results": config_results,
        "permission_results": permission_results,
    }


if __name__ == "__main__":
    main()
