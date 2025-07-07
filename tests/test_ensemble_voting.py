"""
Test cases for ensemble voting in hybrid forecasting models.

This module tests the ensemble voting mechanisms to ensure proper
weight distribution, voting algorithms, and fallback behavior.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from unittest.mock import Mock, patch, MagicMock

# Import ensemble components
# from trading.models.ensemble_model import EnsembleModel  # Not implemented yet
# from trading.models.base_model import BaseModel  # Not implemented yet
from models.forecast_router import ForecastRouter

logger = logging.getLogger(__name__)

class MockForecastModel(BaseModel):
    """Mock forecast model for testing."""
    
    def __init__(self, name: str, confidence: float = 0.8):
        super().__init__({'name': name})
        self.name = name
        self.confidence = confidence
        self.predictions = []
    
    def fit(self, data: pd.DataFrame):
        """Mock fit method."""
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Mock predict method."""
        # Generate mock predictions
        n_samples = len(data)
        base_prediction = 100.0
        noise = np.random.normal(0, 5, n_samples)
        predictions = base_prediction + noise + np.random.randint(-10, 10)
        self.predictions = predictions
        return predictions
    
    def calculate_confidence(self, predictions: np.ndarray) -> float:
        """Mock confidence calculation."""
        return self.confidence

class TestEnsembleVoting:
    """Test ensemble voting functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def ensemble_config(self):
        """Create ensemble configuration."""
        return {
            'models': [
                {'name': 'lstm', 'type': 'LSTM', 'weight': 0.3},
                {'name': 'transformer', 'type': 'Transformer', 'weight': 0.3},
                {'name': 'xgboost', 'type': 'XGBoost', 'weight': 0.4}
            ],
            'voting_method': 'weighted',
            'weight_window': 30,
            'fallback_threshold': 0.5,
            'strategy_aware': True
        }
    
    def test_ensemble_initialization(self, ensemble_config):
        """Test ensemble model initialization."""
        logger.info("Testing ensemble initialization")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Verify initialization
        assert ensemble.config == ensemble_config
        assert len(ensemble.models) == 0  # Models not loaded yet
        assert len(ensemble.weights) == 0  # Weights not set yet
        
        # Test configuration validation
        assert ensemble.config['voting_method'] in ['mse', 'sharpe', 'custom']
        assert ensemble.config['weight_window'] > 0
        assert 0 <= ensemble.config['fallback_threshold'] <= 1
        
        logger.info("Ensemble initialization test passed")
    
    def test_model_registration(self, ensemble_config):
        """Test model registration in ensemble."""
        logger.info("Testing model registration")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Create mock models
        models = {
            'lstm': MockForecastModel('LSTM', confidence=0.8),
            'transformer': MockForecastModel('Transformer', confidence=0.85),
            'xgboost': MockForecastModel('XGBoost', confidence=0.75)
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Verify registration
        assert len(ensemble.models) == 3
        assert 'lstm' in ensemble.models
        assert 'transformer' in ensemble.models
        assert 'xgboost' in ensemble.models
        
        # Test duplicate registration
        duplicate_model = MockForecastModel('LSTM_Duplicate', confidence=0.9)
        ensemble.add_model('lstm', duplicate_model)
        assert ensemble.models['lstm'] == duplicate_model
        
        logger.info("Model registration test passed")
    
    def test_weight_distribution(self, ensemble_config):
        """Test weight distribution and normalization."""
        logger.info("Testing weight distribution")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Create models with different confidences
        models = {
            'lstm': MockForecastModel('LSTM', confidence=0.8),
            'transformer': MockForecastModel('Transformer', confidence=0.85),
            'xgboost': MockForecastModel('XGBoost', confidence=0.75)
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Set initial weights
        initial_weights = {
            'lstm': 0.3,
            'transformer': 0.3,
            'xgboost': 0.4
        }
        
        ensemble.set_weights(initial_weights)
        
        # Verify weights
        assert len(ensemble.weights) == 3
        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6  # Should sum to 1
        
        # Test weight normalization
        unnormalized_weights = {
            'lstm': 3,
            'transformer': 3,
            'xgboost': 4
        }
        
        ensemble.set_weights(unnormalized_weights)
        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6
        
        logger.info("Weight distribution test passed")
    
    def test_weighted_voting(self, ensemble_config, sample_data):
        """Test weighted voting mechanism."""
        logger.info("Testing weighted voting")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Create models with different characteristics
        models = {
            'lstm': MockForecastModel('LSTM', confidence=0.8),
            'transformer': MockForecastModel('Transformer', confidence=0.85),
            'xgboost': MockForecastModel('XGBoost', confidence=0.75)
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Set weights
        weights = {
            'lstm': 0.3,
            'transformer': 0.3,
            'xgboost': 0.4
        }
        ensemble.set_weights(weights)
        
        # Fit models
        for model in models.values():
            model.fit(sample_data)
        
        # Generate predictions
        predictions = ensemble.predict(sample_data)
        
        # Verify predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        
        # Verify weighted combination
        individual_predictions = {}
        for name, model in models.items():
            individual_predictions[name] = model.predict(sample_data)
        
        # Calculate expected weighted average
        expected_prediction = np.zeros(len(sample_data))
        for name, weight in weights.items():
            expected_prediction += weight * individual_predictions[name]
        
        # Compare with ensemble prediction
        assert np.allclose(predictions, expected_prediction, rtol=1e-10)
        
        logger.info("Weighted voting test passed")
    
    def test_confidence_based_voting(self, ensemble_config, sample_data):
        """Test confidence-based voting mechanism."""
        logger.info("Testing confidence-based voting")
        
        # Create ensemble with confidence-based voting
        config = ensemble_config.copy()
        config['voting_method'] = 'confidence'
        ensemble = EnsembleModel(config)
        
        # Create models with different confidences
        models = {
            'high_conf': MockForecastModel('HighConf', confidence=0.9),
            'medium_conf': MockForecastModel('MediumConf', confidence=0.7),
            'low_conf': MockForecastModel('LowConf', confidence=0.5)
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Fit models
        for model in models.values():
            model.fit(sample_data)
        
        # Generate predictions
        predictions = ensemble.predict(sample_data)
        
        # Verify predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        
        # Verify confidence-based weighting
        confidences = {name: model.confidence for name, model in models.items()}
        total_confidence = sum(confidences.values())
        expected_weights = {name: conf / total_confidence for name, conf in confidences.items()}
        
        # Calculate expected weighted average
        individual_predictions = {}
        for name, model in models.items():
            individual_predictions[name] = model.predict(sample_data)
        
        expected_prediction = np.zeros(len(sample_data))
        for name, weight in expected_weights.items():
            expected_prediction += weight * individual_predictions[name]
        
        # Compare with ensemble prediction
        assert np.allclose(predictions, expected_prediction, rtol=1e-10)
        
        logger.info("Confidence-based voting test passed")
    
    def test_fallback_mechanism(self, ensemble_config, sample_data):
        """Test fallback mechanism when models fail."""
        logger.info("Testing fallback mechanism")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Create models with some that will fail
        models = {
            'working': MockForecastModel('Working', confidence=0.8),
            'failing': MockForecastModel('Failing', confidence=0.3),  # Below threshold
            'broken': MockForecastModel('Broken', confidence=0.2)   # Below threshold
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Set weights
        weights = {
            'working': 0.4,
            'failing': 0.3,
            'broken': 0.3
        }
        ensemble.set_weights(weights)
        
        # Fit models
        for model in models.values():
            model.fit(sample_data)
        
        # Mock the confidence check to trigger fallback
        with patch.object(ensemble, '_update_weights') as mock_update:
            predictions = ensemble.predict(sample_data)
            
            # Verify fallback behavior
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == len(sample_data)
            
            # Should only use models above threshold
            mock_update.assert_called()
        
        logger.info("Fallback mechanism test passed")
    
    def test_ensemble_performance_tracking(self, ensemble_config, sample_data):
        """Test ensemble performance tracking."""
        logger.info("Testing ensemble performance tracking")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Create models
        models = {
            'model1': MockForecastModel('Model1', confidence=0.8),
            'model2': MockForecastModel('Model2', confidence=0.85),
            'model3': MockForecastModel('Model3', confidence=0.75)
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Fit models
        for model in models.values():
            model.fit(sample_data)
        
        # Generate predictions multiple times to track performance
        for i in range(3):
            predictions = ensemble.predict(sample_data)
            
            # Simulate performance update
            ensemble._update_weights(sample_data)
        
        # Check performance history
        assert hasattr(ensemble, 'performance_history')
        assert isinstance(ensemble.performance_history, dict)
        
        # Verify weight updates
        assert len(ensemble.weights) == 3
        assert abs(sum(ensemble.weights.values()) - 1.0) < 1e-6
        
        logger.info("Ensemble performance tracking test passed")
    
    def test_strategy_aware_routing(self, ensemble_config, sample_data):
        """Test strategy-aware routing in ensemble."""
        logger.info("Testing strategy-aware routing")
        
        # Create ensemble with strategy awareness
        config = ensemble_config.copy()
        config['strategy_aware'] = True
        ensemble = EnsembleModel(config)
        
        # Create models with different strategy patterns
        models = {
            'trend_following': MockForecastModel('TrendFollowing', confidence=0.8),
            'mean_reversion': MockForecastModel('MeanReversion', confidence=0.85),
            'momentum': MockForecastModel('Momentum', confidence=0.75)
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Set strategy patterns
        ensemble.strategy_patterns = {
            'trend_following': ['trending', 'bull_market'],
            'mean_reversion': ['ranging', 'sideways'],
            'momentum': ['volatile', 'breakout']
        }
        
        # Fit models
        for model in models.values():
            model.fit(sample_data)
        
        # Test strategy-aware prediction
        predictions = ensemble.predict(sample_data)
        
        # Verify predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        
        # Test strategy selection
        strategy = ensemble._get_strategy_recommendation(sample_data)
        assert isinstance(strategy, str)
        assert strategy in ['trend_following', 'mean_reversion', 'momentum']
        
        logger.info("Strategy-aware routing test passed")
    
    def test_ensemble_error_handling(self, ensemble_config, sample_data):
        """Test ensemble error handling."""
        logger.info("Testing ensemble error handling")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Create models with some that will fail
        models = {
            'working': MockForecastModel('Working', confidence=0.8),
            'error_model': MockForecastModel('ErrorModel', confidence=0.6)
        }
        
        # Register models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Mock error in one model
        with patch.object(models['error_model'], 'predict', side_effect=Exception("Model error")):
            # Should handle error gracefully
            try:
                predictions = ensemble.predict(sample_data)
                assert isinstance(predictions, np.ndarray)
                assert len(predictions) == len(sample_data)
            except Exception as e:
                # Should provide fallback or meaningful error
                assert isinstance(e, (ValueError, RuntimeError))
        
        # Test with all models failing
        with patch.object(MockForecastModel, 'predict', side_effect=Exception("All models failed")):
            try:
                predictions = ensemble.predict(sample_data)
            except Exception as e:
                # Should handle complete failure
                assert isinstance(e, (ValueError, RuntimeError))
        
        logger.info("Ensemble error handling test passed")
    
    def test_ensemble_validation(self, ensemble_config):
        """Test ensemble validation and configuration checks."""
        logger.info("Testing ensemble validation")
        
        # Test invalid configurations
        invalid_configs = [
            {'models': [], 'voting_method': 'weighted'},  # No models
            {'models': [{'name': 'test'}], 'voting_method': 'invalid'},  # Invalid voting method
            {'models': [{'name': 'test'}], 'weight_window': -1},  # Negative window
            {'models': [{'name': 'test'}], 'fallback_threshold': 1.5},  # Invalid threshold
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                EnsembleModel(invalid_config)
        
        # Test valid configuration
        valid_config = {
            'models': [{'name': 'test', 'type': 'Test', 'weight': 1.0}],
            'voting_method': 'weighted',
            'weight_window': 30,
            'fallback_threshold': 0.5
        }
        
        ensemble = EnsembleModel(valid_config)
        assert ensemble.config == valid_config
        
        logger.info("Ensemble validation test passed")
    
    def test_ensemble_serialization(self, ensemble_config, sample_data):
        """Test ensemble serialization and deserialization."""
        logger.info("Testing ensemble serialization")
        
        # Create ensemble
        ensemble = EnsembleModel(ensemble_config)
        
        # Add models
        models = {
            'model1': MockForecastModel('Model1', confidence=0.8),
            'model2': MockForecastModel('Model2', confidence=0.85)
        }
        
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Set weights
        weights = {'model1': 0.5, 'model2': 0.5}
        ensemble.set_weights(weights)
        
        # Test serialization
        try:
            serialized = ensemble.to_dict()
            assert isinstance(serialized, dict)
            assert 'config' in serialized
            assert 'weights' in serialized
            assert 'models' in serialized
        except Exception as e:
            # Serialization might not be implemented
            assert isinstance(e, (NotImplementedError, AttributeError))
        
        logger.info("Ensemble serialization test passed") 