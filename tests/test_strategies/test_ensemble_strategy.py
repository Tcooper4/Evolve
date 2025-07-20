"""
Tests for Weighted Ensemble Strategy

This module tests the WeightedEnsembleStrategy functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from trading.strategies.ensemble import (
    WeightedEnsembleStrategy,
    EnsembleConfig,
    create_ensemble_strategy,
    create_rsi_macd_bollinger_ensemble,
)


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 50)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.normal(0, 0.005, 50)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 50))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 50))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 50),
    })
    
    data.set_index('Date', inplace=True)
    return data


@pytest.fixture
def sample_strategy_signals():
    """Create sample strategy signals for testing."""
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    
    # Create sample signals for different strategies
    signals = {}
    
    # RSI signals
    rsi_signals = pd.DataFrame(index=dates)
    rsi_signals['signal'] = np.random.choice([-1, 0, 1], 50, p=[0.3, 0.4, 0.3])
    rsi_signals['confidence'] = np.random.uniform(0.5, 0.9, 50)
    signals['rsi'] = rsi_signals
    
    # MACD signals
    macd_signals = pd.DataFrame(index=dates)
    macd_signals['signal'] = np.random.choice([-1, 0, 1], 50, p=[0.25, 0.5, 0.25])
    macd_signals['confidence'] = np.random.uniform(0.6, 0.95, 50)
    signals['macd'] = macd_signals
    
    # Bollinger signals
    bollinger_signals = pd.DataFrame(index=dates)
    bollinger_signals['signal'] = np.random.choice([-1, 0, 1], 50, p=[0.2, 0.6, 0.2])
    bollinger_signals['confidence'] = np.random.uniform(0.4, 0.8, 50)
    signals['bollinger'] = bollinger_signals
    
    return signals


class TestWeightedEnsembleStrategy:
    """Test cases for WeightedEnsembleStrategy."""

    def test_initialization(self):
        """Test ensemble strategy initialization."""
        # Test with default config
        ensemble = WeightedEnsembleStrategy()
        assert ensemble.config.strategy_weights == {"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}
        assert ensemble.config.combination_method == "weighted_average"
        assert ensemble.config.confidence_threshold == 0.6
        
        # Test with custom config
        custom_weights = {"rsi": 0.5, "macd": 0.3, "bollinger": 0.2}
        config = EnsembleConfig(strategy_weights=custom_weights)
        ensemble = WeightedEnsembleStrategy(config)
        assert ensemble.config.strategy_weights == custom_weights

    def test_weight_validation(self):
        """Test weight validation and normalization."""
        ensemble = WeightedEnsembleStrategy()
        
        # Test valid weights
        valid_weights = {"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}
        assert ensemble.validate_weights(valid_weights) is True
        
        # Test invalid weights (don't sum to 1.0)
        invalid_weights = {"rsi": 0.5, "macd": 0.5, "bollinger": 0.5}
        assert ensemble.validate_weights(invalid_weights) is False
        
        # Test normalization
        normalized = ensemble.normalize_weights(invalid_weights)
        assert abs(sum(normalized.values()) - 1.0) < 0.01

    def test_combine_signals_weighted_average(self, sample_strategy_signals):
        """Test signal combination using weighted average method."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        
        # Combine signals
        combined = ensemble.combine_signals(sample_strategy_signals)
        
        # Check output structure
        assert isinstance(combined, pd.DataFrame)
        assert "signal" in combined.columns
        assert "confidence" in combined.columns
        assert "consensus" in combined.columns
        assert len(combined) > 0
        
        # Check signal values are reasonable
        assert combined["signal"].min() >= -1
        assert combined["signal"].max() <= 1
        assert combined["confidence"].min() >= 0
        assert combined["confidence"].max() <= 1

    def test_combine_signals_voting(self, sample_strategy_signals):
        """Test signal combination using voting method."""
        config = EnsembleConfig(
            strategy_weights={"rsi": 0.33, "macd": 0.33, "bollinger": 0.34},
            combination_method="voting"
        )
        ensemble = WeightedEnsembleStrategy(config)
        
        # Combine signals
        combined = ensemble.combine_signals(sample_strategy_signals, method="voting")
        
        # Check output structure
        assert isinstance(combined, pd.DataFrame)
        assert "signal" in combined.columns
        assert "confidence" in combined.columns
        assert "consensus" in combined.columns

    def test_confidence_threshold_filtering(self, sample_strategy_signals):
        """Test that confidence threshold filters out low-confidence signals."""
        # Create ensemble with high confidence threshold
        config = EnsembleConfig(
            strategy_weights={"rsi": 0.5, "macd": 0.5},
            confidence_threshold=0.8
        )
        ensemble = WeightedEnsembleStrategy(config)
        
        # Combine signals
        combined = ensemble.combine_signals(sample_strategy_signals)
        
        # Check that signals below threshold are filtered out
        low_confidence_mask = combined["confidence"] < 0.8
        assert combined.loc[low_confidence_mask, "signal"].sum() == 0

    def test_update_weights(self):
        """Test dynamic weight updates."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        initial_weights = ensemble.config.strategy_weights.copy()
        
        # Update weights
        new_weights = {"rsi": 0.6, "macd": 0.3, "bollinger": 0.1}
        result = ensemble.update_weights(new_weights)
        
        # Check result
        assert result["success"] is True
        assert result["result"]["status"] == "weights_updated"
        assert ensemble.config.strategy_weights == new_weights
        
        # Check that signals are reset
        assert ensemble.signals is None
        assert ensemble.positions is None

    def test_calculate_positions(self, sample_strategy_signals):
        """Test position calculation."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        
        # Generate signals first
        ensemble.combine_signals(sample_strategy_signals)
        
        # Calculate positions
        positions = ensemble.calculate_positions(pd.DataFrame())
        
        # Check output structure
        assert isinstance(positions, pd.DataFrame)
        assert "position" in positions.columns
        assert "confidence" in positions.columns
        assert "consensus" in positions.columns
        
        # Check position bounds
        assert positions["position"].min() >= -1
        assert positions["position"].max() <= 1

    def test_get_performance_metrics(self, sample_strategy_signals):
        """Test performance metrics calculation."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        
        # Generate signals first
        ensemble.combine_signals(sample_strategy_signals)
        
        # Get metrics
        metrics = ensemble.get_performance_metrics()
        
        # Check result structure
        assert metrics["success"] is True
        assert "total_signals" in metrics["result"]
        assert "buy_signals" in metrics["result"]
        assert "sell_signals" in metrics["result"]
        assert "avg_confidence" in metrics["result"]
        assert "strategy_weights" in metrics["result"]

    def test_get_parameters(self):
        """Test parameter retrieval."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        params = ensemble.get_parameters()
        
        assert "strategy_weights" in params
        assert "combination_method" in params
        assert "confidence_threshold" in params
        assert "consensus_threshold" in params

    def test_set_parameters(self):
        """Test parameter setting."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        
        new_params = {
            "confidence_threshold": 0.8,
            "consensus_threshold": 0.7,
            "position_size_multiplier": 1.5
        }
        
        result = ensemble.set_parameters(new_params)
        
        assert result["success"] is True
        assert ensemble.config.confidence_threshold == 0.8
        assert ensemble.config.consensus_threshold == 0.7
        assert ensemble.config.position_size_multiplier == 1.5

    def test_factory_functions(self):
        """Test factory functions for creating ensemble strategies."""
        # Test create_ensemble_strategy
        weights = {"rsi": 0.5, "macd": 0.5}
        ensemble = create_ensemble_strategy(weights, "weighted_average")
        assert isinstance(ensemble, WeightedEnsembleStrategy)
        assert ensemble.config.strategy_weights == weights
        
        # Test create_rsi_macd_bollinger_ensemble
        ensemble = create_rsi_macd_bollinger_ensemble()
        assert isinstance(ensemble, WeightedEnsembleStrategy)
        expected_weights = {"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}
        assert ensemble.config.strategy_weights == expected_weights

    def test_missing_strategy_handling(self, sample_strategy_signals):
        """Test handling of missing strategy signals."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        
        # Remove one strategy from signals
        partial_signals = {k: v for k, v in sample_strategy_signals.items() if k != "bollinger"}
        
        # Should still work and normalize weights
        combined = ensemble.combine_signals(partial_signals)
        assert isinstance(combined, pd.DataFrame)
        assert len(combined) > 0

    def test_empty_signals_handling(self):
        """Test handling of empty strategy signals."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        
        with pytest.raises(ValueError, match="No strategy signals provided"):
            ensemble.combine_signals({})

    def test_invalid_combination_method(self, sample_strategy_signals):
        """Test handling of invalid combination method."""
        ensemble = create_rsi_macd_bollinger_ensemble()
        
        with pytest.raises(ValueError, match="Unknown combination method"):
            ensemble.combine_signals(sample_strategy_signals, method="invalid_method")


if __name__ == "__main__":
    pytest.main([__file__])
