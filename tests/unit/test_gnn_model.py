import pytest
import numpy as np
import torch
import pandas as pd
from trading.models.advanced.gnn.gnn_model import GNNForecaster
from tests.unit.base_test import BaseModelTest

class TestGNNModel(BaseModelTest):
    """Test suite for GNN model."""
    
    @pytest.fixture
    def model_class(self):
        return GNNForecaster
        
    @pytest.fixture
    def model_config(self):
        return {
            'input_size': 2,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'use_lr_scheduler': True
        }
    
    def test_gnn_specific_features(self, model_class, model_config, sample_data):
        """Test GNN-specific features."""
        model = model_class(config=model_config)
        # Convert sample data to graph format
        X = sample_data[['close', 'volume']].values
        adj = np.eye(len(X))  # Identity matrix for simple graph
        model.fit(X, adj, epochs=2, batch_size=4)
        assert model.config['hidden_size'] == model_config['hidden_size']
        
    def test_gnn_graph_handling(self, model_class, model_config, sample_data):
        """Test GNN graph handling."""
        model = model_class(config=model_config)
        X = sample_data[['close', 'volume']].values
        adj = np.eye(len(X))
        model.fit(X, adj, epochs=2, batch_size=4)
        predictions = model.predict(X, adj)
        assert predictions.shape[0] == len(X)
        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any() 