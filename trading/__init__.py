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

# Core imports
from .models import (
    LSTMModel, XGBoostModel, ProphetModel, ARIMAModel, 
    EnsembleModel, BaseModel, ModelRegistry
)
from .strategies import (
    RSIStrategy, MACDStrategy, BollingerStrategy, SMAStrategy,
    HybridEngine, StrategyRegistry, CustomStrategyHandler
)
from .data import (
    DataLoader, DataProvider, DataPreprocessor, 
    AlphaVantageProvider, YFinanceProvider
)
from .backtesting import (
    Backtester, PerformanceAnalyzer, RiskMetrics, 
    PositionSizer, TradeModels
)
from .optimization import (
    StrategyOptimizer, PortfolioOptimizer, OptunaOptimizer,
    BaseOptimizer, OptimizationVisualizer
)
from .risk import (
    RiskManager, PositionSizingEngine, RiskAnalyzer,
    RiskAdjustedStrategy, RiskMetrics
)
from .portfolio import (
    PortfolioManager, PortfolioSimulator, PositionSizer,
    LLMUtils
)
from .agents import (
    PromptRouterAgent, ExecutionAgent, ModelBuilderAgent,
    StrategySelectorAgent, MarketRegimeAgent, AgentRegistry
)
from .utils import (
    SafeExecutor, ReasoningLogger, PerformanceLogger,
    ErrorHandler, ConfigUtils, DataUtils
)

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

from trading.market import MarketAnalyzer, MarketData, MarketIndicators
from .data.preprocessing import FeatureEngineering, DataValidator, DataScaler
from trading.models.base_model import BaseModel
from trading.models import (
    LSTMModel,
    TCNModel,
    TransformerForecaster,
    GNNForecaster,
    DQNStrategyOptimizer
)
# Optimization modules are available through individual imports
OPTIMIZATION_AVAILABLE = True
from trading.portfolio import PortfolioManager
from trading.risk import RiskManager
from trading.utils import LogManager, ModelLogger, DataLogger
# from trading.memory import PerformanceMemory
from .agents.updater import UpdaterAgent
from trading.nlp import NLInterface, PromptProcessor, ResponseFormatter, LLMProcessor

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

from trading.visualization import TimeSeriesPlotter, PerformancePlotter, FeatureImportancePlotter, PredictionPlotter
from trading.strategies import StrategyManager
from trading.execution import ExecutionEngine
from trading.config import ConfigManager
from trading.knowledge_base import TradingRules

__all__ = [
    'MarketAnalyzer',
    'MarketData',
    'MarketIndicators',
    'DataPreprocessor',
    'FeatureEngineering',
    'DataValidator',
    'DataScaler',
    'AlphaVantageProvider',
    'YFinanceProvider',
    'BaseModel',
    'LSTMModel',
    'TCNModel',
    'TransformerForecaster',
    'GNNForecaster',

    'PortfolioManager',
    'RiskManager',
    'LogManager',
    'ModelLogger',
    'DataLogger',
    'ModelEvaluator',
    'RegressionMetrics',
    'ClassificationMetrics',
    'TimeSeriesMetrics',
    'TimeSeriesPlotter',
    'PerformancePlotter',
    'FeatureImportancePlotter',
    'PredictionPlotter',
    'StrategyManager',
    'ExecutionEngine',
    'ConfigManager',
    'TradingRules'
]

if OPTIMIZATION_AVAILABLE:
    __all__.append('DQNStrategyOptimizer')

# from trading.memory import PerformanceMemory
# __all__.append('PerformanceMemory')

__all__.append('UpdaterAgent')
