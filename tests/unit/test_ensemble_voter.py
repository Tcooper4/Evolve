import pytest
from trading.strategies.ensemble import WeightedEnsembleStrategy

def test_normalize_weights_output_shape():
    """Test that normalize_weights returns correct output shape."""
    strategy = WeightedEnsembleStrategy()
    scores = {'model_a': 0.2, 'model_b': 0.3}
    result = strategy.normalize_weights(scores)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'model_a', 'model_b'}
    assert abs(sum(result.values()) - 1.0) < 0.01  # Should sum to 1.0

def test_normalize_weights_with_zero_weights():
    """Test normalize_weights with all zero weights."""
    strategy = WeightedEnsembleStrategy()
    scores = {'model_a': 0.0, 'model_b': 0.0, 'model_c': 0.0}
    result = strategy.normalize_weights(scores)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'model_a', 'model_b', 'model_c'}
    # Should be equal weights when all inputs are zero
    expected_weight = 1.0 / 3.0
    for weight in result.values():
        assert abs(weight - expected_weight) < 0.01

def test_ensemble_strategy_initialization():
    """Test WeightedEnsembleStrategy initialization."""
    strategy = WeightedEnsembleStrategy()
    assert strategy.config is not None
    assert isinstance(strategy.config.strategy_weights, dict) 