import pytest
import numpy as np
import torch
from trading.models.advanced.ensemble.ensemble_model import EnsembleForecaster
from trading.models.lstm_model import LSTMForecaster

def make_sample_data():
    # 100 samples, 2 features each
    X = np.random.randn(100, 2).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    return X, y

def test_ensemble_forecaster_fit_predict():
    X, y = make_sample_data()
    model1 = LSTMForecaster()
    model2 = LSTMForecaster()
    ensemble = EnsembleForecaster(models=[model1, model2])
    ensemble.fit(X, epochs=3, batch_size=4)
    preds = ensemble.predict(X)
    assert preds['predictions'].shape == y.shape
    assert not np.isnan(preds['predictions']).any()
    assert not np.isinf(preds['predictions']).any()

def test_ensemble_forecaster_save_load():
    X, y = make_sample_data()
    model1 = LSTMForecaster()
    model2 = LSTMForecaster()
    ensemble = EnsembleForecaster(models=[model1, model2])
    ensemble.fit(X, epochs=2, batch_size=4)
    import tempfile, os
    save_path = tempfile.mktemp()
    ensemble.save(save_path)
    loaded_ensemble = EnsembleForecaster(models=[LSTMForecaster(), LSTMForecaster()])
    loaded_ensemble.load(save_path)
    preds1 = ensemble.predict(X)
    preds2 = loaded_ensemble.predict(X)
    np.testing.assert_allclose(preds1['predictions'], preds2['predictions'], rtol=1e-2, atol=1e-2)
    os.remove(save_path) 