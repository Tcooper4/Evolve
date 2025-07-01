"""Trading system package."""

__version__ = '0.1.0'

from trading.market import MarketAnalyzer, MarketData, MarketIndicators
from .data.preprocessing import DataPreprocessor, FeatureEngineering, DataValidator, DataScaler
from .data.providers import AlphaVantageProvider, YFinanceProvider
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
from trading.utils import LogManager, ModelLogger, DataLogger, PerformanceLogger
# from trading.memory import PerformanceMemory
from .agents.updater import UpdaterAgent
from trading.nlp import NLInterface, PromptProcessor, ResponseFormatter, LLMProcessor

# Fix broken imports with proper error handling
try:
    from trading.evaluation import ModelEvaluator, RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, RiskMetrics
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

    class RiskMetrics:
        def __init__(self):
            logger.warning("⚠️ Using fallback RiskMetrics")

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
    'PerformanceLogger',
    'NLInterface',
    'PromptProcessor',
    'ResponseFormatter',
    'LLMProcessor',
    'ModelEvaluator',
    'RegressionMetrics',
    'ClassificationMetrics',
    'TimeSeriesMetrics',
    'RiskMetrics',
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
