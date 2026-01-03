from .advanced.rl.strategy_optimizer import DQNStrategyOptimizer
from .advanced.transformer.time_series_transformer import TransformerForecaster
from .arima_model import ARIMAModel
from .base_model import BaseModel
from .forecast_router import ForecastRouter
from .garch_model import GARCHModel, create_garch_model
from .lstm_model import LSTMModel
from .ridge_model import RidgeModel, create_ridge_model
from .tcn_model import TCNModel
from .xgboost_model import XGBoostModel

# GNN model (optional - requires PyTorch)
try:
    from .advanced.gnn.gnn_model import GNNForecaster
    GNN_AVAILABLE = True
except ImportError:
    GNNForecaster = None
    GNN_AVAILABLE = False

__all__ = [
    "BaseModel",
    "LSTMModel",
    "TCNModel",
    "ARIMAModel",
    "XGBoostModel",
    "GARCHModel",
    "RidgeModel",
    "create_garch_model",
    "create_ridge_model",
    "TransformerForecaster",
    "DQNStrategyOptimizer",
    "ForecastRouter",
]

if GNN_AVAILABLE:
    __all__.append("GNNForecaster")
