import pytest
import numpy as np
import torch
from trading.models.advanced.gnn.gnn_model import GNNForecaster

def make_sample_graph():
    # 5 nodes, 2 features each
    X = np.random.randn(5, 2).astype(np.float32)
    adj = np.eye(5, dtype=np.float32)  # Identity for simplicity
    y = np.random.randn(5, 1).astype(np.float32)
    return X, adj, y

def test_gnn_forecaster_fit_predict():
    X, adj, y = make_sample_graph()
    model = GNNForecaster()
    model.fit(X, adj, y, epochs=3)
    preds = model.predict(X, adj)
    assert preds.shape == y.shape
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()

def test_gnn_forecaster_save_load():
    X, adj, y = make_sample_graph()
    model = GNNForecaster()
    model.fit(X, adj, y, epochs=2)
    import tempfile, os
    save_path = tempfile.mktemp()
    model.save(save_path)
    loaded_model = GNNForecaster()
    loaded_model.load(save_path)
    preds1 = model.predict(X, adj)
    preds2 = loaded_model.predict(X, adj)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-2, atol=1e-2)
    os.remove(save_path) 