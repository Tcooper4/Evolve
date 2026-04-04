"""Graph Neural Network models for market forecasting."""

try:
    from .gnn_model import GNNForecaster, GNNModel, GraphNeuralNetwork
    __all__ = ['GNNForecaster', 'GNNModel', 'GraphNeuralNetwork']
except ImportError as e:
    # PyTorch not available
    __all__ = []
