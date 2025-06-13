"""Trading system package."""

__version__ = '0.1.0'

from .analysis import MarketAnalyzer
from .data import DataPreprocessor, FeatureEngineering, DataValidator, DataScaler
from .data.providers import AlphaVantageProvider, YFinanceProvider
from .models import (
    BaseModel,
    LSTMModel,
    TCNModel,
    TransformerForecaster,
    GNNForecaster,
    DQNStrategyOptimizer
)
try:
    from .optimization import Optimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from .portfolio import PortfolioManager
from .risk import RiskManager
from .utils import LogManager, ModelLogger, DataLogger, PerformanceLogger
from .memory import PerformanceMemory
from .nlp import NLInterface, PromptProcessor, ResponseFormatter, LLMProcessor
from .market import MarketData, MarketIndicators
from .evaluation import ModelEvaluator, RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, RiskMetrics
from .visualization import TimeSeriesPlotter, PerformancePlotter, FeatureImportancePlotter, PredictionPlotter
from .strategies import StrategyManager
from .execution import ExecutionEngine
from .config import ConfigManager
from .knowledge_base import TradingRules

__all__ = [
    'MarketAnalyzer',
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
    'Optimizer',
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
    'MarketData',
    'MarketIndicators',
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

__all__.append('PerformanceMemory')
