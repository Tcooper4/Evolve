import torch
import json
from trading.agents.updater import ModelUpdater
from trading.models.advanced.ensemble.ensemble_model import EnsembleForecaster
from trading.memory import PerformanceMemory


class DummyModel:
    """Simple model returning zeros."""

    def __init__(self, config=None):
        self.config = config or {}

    def __call__(self, x):
        return torch.zeros(x.shape[0], 1)


def create_ensemble(tmp_path):
    memory_path = tmp_path / "perf.json"
    perf_mem = PerformanceMemory(str(memory_path))
    config = {
        'models': [
            {'class': DummyModel},
            {'class': DummyModel}
        ],
        'memory_path': str(memory_path)
    }
    model = EnsembleForecaster(config=config)
    return model, perf_mem


def test_weights_update_with_new_metrics(tmp_path):
    model, perf_mem = create_ensemble(tmp_path)
    updater = ModelUpdater(model, perf_mem)

    initial = model.model_weights.clone()

    perf_mem.update('AAPL', 'model_0', {'mse': 0.5, 'sharpe': 1.0})
    perf_mem.update('AAPL', 'model_1', {'mse': 0.1, 'sharpe': 2.0})

    updater.update_model_weights('AAPL', 'mse')

    updated = model.model_weights
    assert not torch.allclose(initial, updated)

