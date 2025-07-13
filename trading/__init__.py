"""
Evolve Trading System

An autonomous financial forecasting and trading strategy platform that leverages
multiple machine learning models to predict stock price movements, generate
technical trading signals, backtest strategies, and visualize performance.
"""

__version__ = "2.1.0"
__author__ = "Evolve Team"
__email__ = "support@evolve-trading.com"
__description__ = "Autonomous Financial Forecasting & Trading Platform"
__url__ = "https://github.com/Tcooper4/Evolve"
__license__ = "MIT"

# Core imports with error handling
try:
    pass

    MODELS_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Models import failed: {e}")
    MODELS_AVAILABLE = False
    raise ImportError("Model modules failed to load. Check logs.")

try:
    pass

    STRATEGIES_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Strategies import failed: {e}")
    STRATEGIES_AVAILABLE = False

try:
    pass

    DATA_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Data import failed: {e}")
    DATA_AVAILABLE = False

try:
    pass

    BACKTESTING_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Backtesting import failed: {e}")
    BACKTESTING_AVAILABLE = False

try:
    pass

    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Optimization import failed: {e}")
    OPTIMIZATION_AVAILABLE = False

try:
    pass

    RISK_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Risk import failed: {e}")
    RISK_AVAILABLE = False

try:
    pass

    PORTFOLIO_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Portfolio import failed: {e}")
    PORTFOLIO_AVAILABLE = False

try:
    pass

    AGENTS_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Agents import failed: {e}")
    AGENTS_AVAILABLE = False

try:
    from trading.utils import (
        ModelEvaluator,
    )

    UTILS_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Utils import failed: {e}")
    UTILS_AVAILABLE = False

# Version info


def get_version():
    """Get the current version of the Evolve trading system."""
    return __version__


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "url": __url__,
        "license": __license__,
    }


# Additional imports with error handling
try:
    pass

    MARKET_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Market import failed: {e}")
    MARKET_AVAILABLE = False

try:
    pass

    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Preprocessing import failed: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    pass

    UPDATER_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Updater import failed: {e}")
    UPDATER_AVAILABLE = False

try:
    pass

    NLP_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ NLP import failed: {e}")
    NLP_AVAILABLE = False

# Fix broken imports with proper error handling
try:
    from trading.evaluation import (
        ClassificationMetrics,
        ModelEvaluator,
        RegressionMetrics,
        TimeSeriesMetrics,
    )

    EVALUATION_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Evaluation metrics import failed: {e}")
    EVALUATION_AVAILABLE = False

    # Create fallback classes
    class RegressionMetrics:
        def __init__(self):
            logger.warning("⚠️ Using fallback RegressionMetrics")

    class ClassificationMetrics:
        def __init__(self):
            logger.warning("⚠️ Using fallback ClassificationMetrics")

    class TimeSeriesMetrics:
        def __init__(self):
            logger.warning("⚠️ Using fallback TimeSeriesMetrics")

    class ModelEvaluator:
        def __init__(self):
            logger.warning("⚠️ Using fallback ModelEvaluator")


# Automatic module discovery for all subcomponents


def discover_available_modules():
    """Automatically discover and import all available subcomponents."""
    import logging
    import os

    logger = logging.getLogger(__name__)
    discovered_modules = {}

    # Define subcomponent directories to scan
    subcomponent_dirs = [
        "agents",
        "strategies",
        "models",
        "data",
        "backtesting",
        "optimization",
        "risk",
        "portfolio",
        "utils",
        "market",
        "nlp",
        "evaluation",
        "services",
        "execution",
        "analysis",
        "feature_engineering",
        "memory",
        "llm",
        "commentary",
        "signals",
        "visualization",
        "report",
        "config",
        "core",
    ]

    # Scan each directory for available modules
    for subdir in subcomponent_dirs:
        subdir_path = os.path.join(os.path.dirname(__file__), subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            discovered_modules[subdir] = []

            # Look for __init__.py files and Python modules
            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)

                # Check if it's a Python file or directory with __init__.py
                if (item.endswith(".py") and not item.startswith("__")) or (
                    os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "__init__.py"))
                ):
                    module_name = item.replace(".py", "")
                    if module_name not in ["__init__", "__pycache__"]:
                        discovered_modules[subdir].append(module_name)

    return discovered_modules


def auto_import_subcomponents():
    """Automatically import discovered subcomponents."""
    import logging

    logger = logging.getLogger(__name__)

    discovered = discover_available_modules()
    imported_modules = {}

    for subdir, modules in discovered.items():
        imported_modules[subdir] = {}

        for module_name in modules:
            try:
                # Try to import the module
                module_path = f"trading.{subdir}.{module_name}"
                module = importlib.import_module(module_path)
                imported_modules[subdir][module_name] = module
                logger.debug(f"✅ Successfully imported {module_path}")

            except ImportError as e:
                logger.debug(f"⚠️ Failed to import {module_path}: {e}")
                continue
            except Exception as e:
                logger.debug(f"⚠️ Error importing {module_path}: {e}")
                continue

    return imported_modules


def get_available_subcomponents():
    """Get a list of all available subcomponents."""
    discovered = discover_available_modules()
    available = {}

    for subdir, modules in discovered.items():
        if modules:  # Only include non-empty directories
            available[subdir] = modules

    return available


def import_subcomponent(subdir: str, module_name: str):
    """Import a specific subcomponent module."""
    import importlib
    import logging

    logger = logging.getLogger(__name__)

    try:
        module_path = f"trading.{subdir}.{module_name}"
        module = importlib.import_module(module_path)
        logger.debug(f"✅ Successfully imported {module_path}")
        return module
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import {module_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Error importing {module_path}: {e}")
        return None


# Auto-discover and import available modules on package import
try:
    AVAILABLE_SUBCOMPONENTS = get_available_subcomponents()
    IMPORTED_MODULES = auto_import_subcomponents()
    AUTO_DISCOVERY_AVAILABLE = True
except Exception as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Auto-discovery failed: {e}")
    AVAILABLE_SUBCOMPONENTS = {}
    IMPORTED_MODULES = {}
    AUTO_DISCOVERY_AVAILABLE = False

# Convenience functions for accessing discovered modules


def get_agent_modules():
    """Get all available agent modules."""
    return AVAILABLE_SUBCOMPONENTS.get("agents", [])


def get_strategy_modules():
    """Get all available strategy modules."""
    return AVAILABLE_SUBCOMPONENTS.get("strategies", [])


def get_model_modules():
    """Get all available model modules."""
    return AVAILABLE_SUBCOMPONENTS.get("models", [])


def get_data_modules():
    """Get all available data modules."""
    return AVAILABLE_SUBCOMPONENTS.get("data", [])


def get_service_modules():
    """Get all available service modules."""
    return AVAILABLE_SUBCOMPONENTS.get("services", [])


def get_utility_modules():
    """Get all available utility modules."""
    return AVAILABLE_SUBCOMPONENTS.get("utils", [])


# Module availability status


def get_module_status():
    """Get status of all module imports."""
    return {
        "models": MODELS_AVAILABLE,
        "strategies": STRATEGIES_AVAILABLE,
        "data": DATA_AVAILABLE,
        "backtesting": BACKTESTING_AVAILABLE,
        "optimization": OPTIMIZATION_AVAILABLE,
        "risk": RISK_AVAILABLE,
        "portfolio": PORTFOLIO_AVAILABLE,
        "agents": AGENTS_AVAILABLE,
        "utils": UTILS_AVAILABLE,
        "market": MARKET_AVAILABLE,
        "preprocessing": PREPROCESSING_AVAILABLE,
        "updater": UPDATER_AVAILABLE,
        "nlp": NLP_AVAILABLE,
        "evaluation": EVALUATION_AVAILABLE,
        "auto_discovery": AUTO_DISCOVERY_AVAILABLE,
    }


# Export all discovered modules for easy access
__all__ = [
    # Core modules
    "get_version",
    "get_version_info",
    "get_module_status",
    # Auto-discovery functions
    "discover_available_modules",
    "auto_import_subcomponents",
    "get_available_subcomponents",
    "import_subcomponent",
    # Convenience functions
    "get_agent_modules",
    "get_strategy_modules",
    "get_model_modules",
    "get_data_modules",
    "get_service_modules",
    "get_utility_modules",
    # Module availability flags
    "MODELS_AVAILABLE",
    "STRATEGIES_AVAILABLE",
    "DATA_AVAILABLE",
    "BACKTESTING_AVAILABLE",
    "OPTIMIZATION_AVAILABLE",
    "RISK_AVAILABLE",
    "PORTFOLIO_AVAILABLE",
    "AGENTS_AVAILABLE",
    "UTILS_AVAILABLE",
    "MARKET_AVAILABLE",
    "PREPROCESSING_AVAILABLE",
    "UPDATER_AVAILABLE",
    "NLP_AVAILABLE",
    "EVALUATION_AVAILABLE",
    "AUTO_DISCOVERY_AVAILABLE",
    # Discovered components
    "AVAILABLE_SUBCOMPONENTS",
    "IMPORTED_MODULES",
]
