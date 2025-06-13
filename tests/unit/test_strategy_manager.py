import pytest
from trading.strategies.strategy_manager import StrategyManager, StrategyError


def test_set_ensemble_normalizes_weights(tmp_path):
    sm = StrategyManager({'results_dir': str(tmp_path)})
    sm.active_strategies = {'s1': True, 's2': True}
    sm.set_ensemble({'s1': 2.0, 's2': 1.0}, strict=False)
    assert sm.ensemble_weights['s1'] == pytest.approx(2/3)
    assert sm.ensemble_weights['s2'] == pytest.approx(1/3)


def test_set_ensemble_strict_error(tmp_path):
    sm = StrategyManager({'results_dir': str(tmp_path)})
    sm.active_strategies = {'s1': True}
    with pytest.raises(StrategyError):
        sm.set_ensemble({'s1': 0.5})

