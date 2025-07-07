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
    from .models import (
        LSTMModel, TCNModel, ARIMAModel, XGBoostModel,
        BaseModel, TransformerForecaster, GNNForecaster, DQNStrategyOptimizer
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Models import failed: {e}")
    MODELS_AVAILABLE = False

try:
    from .strategies import (
        StrategyManager, Strategy, StrategyMetrics,
        BollingerStrategy, BollingerConfig,
        MACDStrategy, MACDConfig,
        SMAStrategy, SMAConfig,
        generate_signals, get_signals
    )
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Strategies import failed: {e}")
    STRATEGIES_AVAILABLE = False

try:
    from .data import (
        DataLoader, DataProvider, DataPreprocessor, 
        AlphaVantageProvider, YFinanceProvider
    )
    DATA_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Data import failed: {e}")
    DATA_AVAILABLE = False

try:
    from .backtesting import (
        BacktestEngine, PerformanceAnalyzer, RiskMetricsEngine, 
        PositionSizingEngine, Trade, TradeType
    )
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Backtesting import failed: {e}")
    BACKTESTING_AVAILABLE = False

try:
    from .optimization import (
        StrategyOptimizer, BaseOptimizer, OptimizationVisualizer
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Optimization import failed: {e}")
    OPTIMIZATION_AVAILABLE = False

try:
    from .risk import (
        RiskManager
    )
    RISK_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Risk import failed: {e}")
    RISK_AVAILABLE = False

try:
    from .portfolio import (
        PortfolioManager
    )
    PORTFOLIO_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Portfolio import failed: {e}")
    PORTFOLIO_AVAILABLE = False

try:
    from .agents import (
        PromptRouterAgent, ModelBuilderAgent
    )
    AGENTS_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Agents import failed: {e}")
    AGENTS_AVAILABLE = False

try:
    from .utils import (
        LogManager, ModelLogger, DataLogger, PerformanceLogger
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
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'url': __url__,
        'license': __license__
    }

# Additional imports with error handling
try:
    from trading.market import MarketAnalyzer, MarketData, MarketIndicators
    MARKET_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Market import failed: {e}")
    MARKET_AVAILABLE = False

try:
    from .data.preprocessing import FeatureEngineering, DataValidator, DataScaler
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Preprocessing import failed: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from .agents.updater import UpdaterAgent
    UPDATER_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Updater import failed: {e}")
    UPDATER_AVAILABLE = False

try:
    from trading.nlp import NLInterface, PromptProcessor, ResponseFormatter, LLMProcessor
    NLP_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ NLP import failed: {e}")
    NLP_AVAILABLE = False

# Fix broken imports with proper error handling
try:
    from trading.evaluation import ModelEvaluator, RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics
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

try:
    from trading.visualization import TimeSeriesPlotter, PerformancePlotter, FeatureImportancePlotter, PredictionPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Visualization import failed: {e}")
    VISUALIZATION_AVAILABLE = False

try:
    from trading.strategies import StrategyManager
    STRATEGY_MANAGER_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Strategy manager import failed: {e}")
    STRATEGY_MANAGER_AVAILABLE = False

try:
    from trading.execution import ExecutionEngine
    EXECUTION_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Execution import failed: {e}")
    EXECUTION_AVAILABLE = False

try:
    from trading.config import ConfigManager
    CONFIG_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Config import failed: {e}")
    CONFIG_AVAILABLE = False

try:
    from trading.knowledge_base import TradingRules
    KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Knowledge base import failed: {e}")
    KNOWLEDGE_AVAILABLE = False

# Build __all__ list dynamically based on available modules
__all__ = []

if MODELS_AVAILABLE:
    __all__.extend(['LSTMModel', 'TCNModel', 'ARIMAModel', 'XGBoostModel', 'BaseModel', 'TransformerForecaster', 'GNNForecaster', 'DQNStrategyOptimizer'])

if STRATEGIES_AVAILABLE:
    __all__.extend(['StrategyManager', 'Strategy', 'StrategyMetrics', 'BollingerStrategy', 'BollingerConfig', 'MACDStrategy', 'MACDConfig', 'SMAStrategy', 'SMAConfig', 'generate_signals', 'get_signals'])

if DATA_AVAILABLE:
    __all__.extend(['DataLoader', 'DataProvider', 'DataPreprocessor', 'AlphaVantageProvider', 'YFinanceProvider'])

if BACKTESTING_AVAILABLE:
    __all__.extend(['BacktestEngine', 'PerformanceAnalyzer', 'RiskMetricsEngine', 'PositionSizingEngine', 'Trade', 'TradeType'])

if OPTIMIZATION_AVAILABLE:
    __all__.extend(['StrategyOptimizer', 'BaseOptimizer', 'OptimizationVisualizer'])

if RISK_AVAILABLE:
    __all__.extend(['RiskManager'])

if PORTFOLIO_AVAILABLE:
    __all__.extend(['PortfolioManager'])

if AGENTS_AVAILABLE:
    __all__.extend(['PromptRouterAgent', 'ExecutionAgent', 'ModelBuilderAgent', 'StrategySelectorAgent', 'MarketRegimeAgent', 'AgentRegistry'])

if UTILS_AVAILABLE:
    __all__.extend(['LogManager', 'ModelLogger', 'DataLogger', 'PerformanceLogger'])

if MARKET_AVAILABLE:
    __all__.extend(['MarketAnalyzer', 'MarketData', 'MarketIndicators'])

if PREPROCESSING_AVAILABLE:
    __all__.extend(['FeatureEngineering', 'DataValidator', 'DataScaler'])

if UPDATER_AVAILABLE:
    __all__.append('UpdaterAgent')

if NLP_AVAILABLE:
    __all__.extend(['NLInterface', 'PromptProcessor', 'ResponseFormatter', 'LLMProcessor'])

if EVALUATION_AVAILABLE:
    __all__.extend(['ModelEvaluator', 'RegressionMetrics', 'ClassificationMetrics', 'TimeSeriesMetrics'])

if VISUALIZATION_AVAILABLE:
    __all__.extend(['TimeSeriesPlotter', 'PerformancePlotter', 'FeatureImportancePlotter', 'PredictionPlotter'])

if STRATEGY_MANAGER_AVAILABLE:
    __all__.append('StrategyManager')

if EXECUTION_AVAILABLE:
    __all__.append('ExecutionEngine')

if CONFIG_AVAILABLE:
    __all__.append('ConfigManager')

if KNOWLEDGE_AVAILABLE:
    __all__.append('TradingRules')
