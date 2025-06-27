"""Trading system package."""

__version__ = '0.1.0'

from trading.market import MarketAnalyzer, MarketData, MarketIndicators
from .data.preprocessing import DataPreprocessor, FeatureEngineering, DataValidator, DataScaler
from .data.providers import AlphaVantageProvider, YFinanceProvider
from trading.base_model import BaseModel
from trading.models import (
    LSTMModel,
    TCNModel,
    TransformerForecaster,
    GNNForecaster,
    DQNStrategyOptimizer
)
try:
    from trading.optimization import Optimizer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from trading.portfolio import PortfolioManager
from trading.risk import RiskManager
from trading.utils import LogManager, ModelLogger, DataLogger, PerformanceLogger
# from trading.memory import PerformanceMemory
from .agents.updater import ModelUpdater
from trading.nlp import NLInterface, PromptProcessor, ResponseFormatter, LLMProcessor
from trading.evaluation import ModelEvaluator, RegressionMetrics, ClassificationMetrics, TimeSeriesMetrics, RiskMetrics
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

__all__.append('ModelUpdater')
