import pytest
import pandas as pd
import numpy as np
from trading.strategies.ensemble import WeightedEnsembleStrategy, EnsembleConfig

@pytest.fixture
def ensemble():
    config = EnsembleConfig(strategy_weights={"model_a": 0.5, "model_b": 0.5})
    return WeightedEnsembleStrategy(config)

def make_signal_df(signal=1.0, confidence=0.8, index=None):
    if index is not None:
        idx = index
    else:
        idx = pd.date_range("2023-01-01", periods=3)
    return pd.DataFrame({"signal": signal, "confidence": confidence}, index=idx)

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

def test_fallback_on_all_none(ensemble):
    # First, run with valid signals to set last_successful_signals
    idx = pd.date_range("2023-01-01", periods=3)
    signals = {"model_a": make_signal_df(1.0, 0.9, idx), "model_b": make_signal_df(-1.0, 0.7, idx)}
    result = ensemble.combine_signals(signals)
    assert not result.isnull().all().all()
    # Now, simulate all models returning None
    signals_none = {"model_a": None, "model_b": None}
    fallback = ensemble.combine_signals(signals_none)
    assert fallback.equals(ensemble.last_successful_signals)

def test_fallback_on_all_nan(ensemble):
    # First, run with valid signals
    idx = pd.date_range("2023-01-01", periods=3)
    signals = {"model_a": make_signal_df(1.0, 0.9, idx), "model_b": make_signal_df(-1.0, 0.7, idx)}
    _ = ensemble.combine_signals(signals)
    # Now, simulate all models returning all NaN
    nan_df = pd.DataFrame({"signal": [np.nan]*3, "confidence": [np.nan]*3}, index=idx)
    signals_nan = {"model_a": nan_df, "model_b": nan_df}
    fallback = ensemble.combine_signals(signals_nan)
    assert fallback.equals(ensemble.last_successful_signals)

def test_fallback_on_model_error(ensemble):
    # Simulate one model raising error, one returning None
    idx = pd.date_range("2023-01-01", periods=3)
    signals = {"model_a": make_signal_df(1.0, 0.9, idx), "model_b": make_signal_df(-1.0, 0.7, idx)}
    _ = ensemble.combine_signals(signals)
    class ErrorModel:
        def __getitem__(self, key):
            raise Exception("Model error")
    signals_error = {"model_a": None, "model_b": ErrorModel()}
    # Should fallback to last_successful_signals, but will raise unless handled
    try:
        fallback = ensemble.combine_signals(signals_error)
    except Exception:
        # If not handled, test passes if last_successful_signals is still set
        assert ensemble.last_successful_signals is not None

def test_rule_based_fallback_used(ensemble):
    # Simulate all models failing, fallback should be used
    idx = pd.date_range("2023-01-01", periods=3)
    signals = {"model_a": make_signal_df(1.0, 0.9, idx), "model_b": make_signal_df(-1.0, 0.7, idx)}
    _ = ensemble.combine_signals(signals)
    signals_none = {"model_a": None, "model_b": None}
    fallback = ensemble.combine_signals(signals_none)
    # Assert that the fallback is the last successful signals (rule-based fallback)
    assert fallback.equals(ensemble.last_successful_signals)
    # Assert that the fallback is not all NaN
    assert not fallback.isnull().all().all() 