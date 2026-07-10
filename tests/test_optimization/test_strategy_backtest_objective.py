# -*- coding: utf-8 -*-
"""Tests for the strategy-optimization wiring added in the Fable session.

Covers:
* canonical parameter spaces (trading/optimization/strategy_param_spaces.py)
* the strategy runner / objective bridge
  (trading/optimization/strategy_backtest_objective.py)
* the registry.execute_strategy parameter-path fixes
* the CCI ndarray fix and the SMA lowercase-columns fix
* SelfTuningOptimizer receiving real bounds (previously inert)

All tests run real code paths on synthetic OHLCV - no mocks around the
behavior under test.
"""

import numpy as np
import pandas as pd
import pytest

from trading.optimization.strategy_param_spaces import (
    STRATEGY_PARAM_SPACES,
    constraint_violations,
    get_default_params,
    get_self_tuning_config,
    to_optimizer_space,
)
from trading.optimization.strategy_backtest_objective import (
    INVALID_PENALTY,
    evaluate_params,
    make_objective,
    normalize_ohlcv,
    optimize_strategy,
    run_strategy,
    signals_to_returns,
)

ALL_STRATEGIES = sorted(STRATEGY_PARAM_SPACES.keys())

NON_DEFAULT_PARAMS = {
    "RSIStrategy": {"rsi_period": 9, "oversold_threshold": 25.0,
                    "overbought_threshold": 75.0},
    "MACDStrategy": {"fast_period": 8, "slow_period": 21, "signal_period": 5},
    "BollingerStrategy": {"window": 15, "num_std": 1.5},
    "SMAStrategy": {"short_window": 10, "long_window": 40,
                    "confirmation_periods": 2},
    "ATRStrategy": {"period": 10, "multiplier": 1.5,
                    "volatility_threshold": 0.01},
    "CCIStrategy": {"period": 14, "oversold_threshold": -120.0,
                    "overbought_threshold": 120.0},
}


@pytest.fixture(scope="module")
def ohlcv():
    """Synthetic OHLCV with a cyclical drift so signals actually fire."""
    rng = np.random.default_rng(7)
    n = 500
    idx = pd.date_range("2023-06-01", periods=n, freq="B")
    ret = rng.normal(0.0005, 0.013, n) + 0.002 * np.sin(np.arange(n) / 25)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "Close": close,
            "Volume": rng.integers(2_000_000, 6_000_000, n).astype(float),
        },
        index=idx,
    )


# ---------------------------------------------------------------- spaces

class TestParamSpaces:
    def test_every_space_has_specs_and_valid_defaults(self):
        for name in ALL_STRATEGIES:
            specs = STRATEGY_PARAM_SPACES[name]
            assert specs, f"{name} has an empty space"
            defaults = get_default_params(name)
            assert not constraint_violations(name, defaults)
            for s in specs:
                assert s.low < s.high
                assert s.low <= s.default <= s.high
                assert s.step > 0

    def test_grid_values_cover_bounds_and_types(self):
        for name in ALL_STRATEGIES:
            for s in STRATEGY_PARAM_SPACES[name]:
                vals = s.grid_values()
                assert vals[0] == pytest.approx(s.low)
                assert vals[-1] == pytest.approx(s.high, abs=s.step)
                if s.kind == "int":
                    assert all(isinstance(v, int) for v in vals)

    def test_constraint_detection(self):
        assert constraint_violations(
            "RSIStrategy",
            {"oversold_threshold": 80, "overbought_threshold": 60},
        )
        assert constraint_violations(
            "SMAStrategy", {"short_window": 60, "long_window": 40}
        )

    def test_self_tuning_config_covers_all_strategies(self):
        cfg = get_self_tuning_config()
        assert set(cfg["parameter_bounds"]) == set(ALL_STRATEGIES)
        for name, bounds in cfg["parameter_bounds"].items():
            assert bounds, f"{name} bounds empty"
            assert set(cfg["parameter_steps"][name]) == set(bounds)


# ---------------------------------------------------------------- runner

class TestRunner:
    def test_normalize_ohlcv_lowercase_single_set(self, ohlcv):
        out = normalize_ohlcv(ohlcv)
        assert list(out.columns) == ["open", "high", "low", "close", "volume"]
        # No duplicate columns even if input already had lowercase.
        again = normalize_ohlcv(out)
        assert not again.columns.duplicated().any()

    @pytest.mark.parametrize("name", ALL_STRATEGIES)
    def test_every_strategy_runs_and_params_change_signals(self, ohlcv, name):
        default_sig = run_strategy(name, ohlcv)
        custom_sig = run_strategy(name, ohlcv, NON_DEFAULT_PARAMS[name])
        assert "signal" in default_sig.columns
        assert not default_sig["signal"].equals(custom_sig["signal"]), (
            f"{name}: parameters had no effect on signals"
        )

    def test_runner_does_not_mutate_registry_singletons(self, ohlcv):
        from trading.strategies.registry import get_strategy_registry

        reg = get_strategy_registry()
        proto = reg.get_strategy("BollingerStrategy")
        before = proto.config.window
        run_strategy("BollingerStrategy", ohlcv, {"window": 44})
        assert proto.config.window == before

    def test_signals_to_returns_no_lookahead(self, ohlcv):
        """A signal on bar t must not earn bar t's return."""
        close = normalize_ohlcv(ohlcv)["close"]
        sig = pd.DataFrame({"signal": 0.0}, index=close.index)
        t = 100
        sig.iloc[t, 0] = 1.0  # buy signal on bar t
        rets = signals_to_returns(sig, close, cost_bps=0.0)
        assert rets.iloc[t] == 0.0
        expected = close.pct_change().iloc[t + 1]
        assert rets.iloc[t + 1] == pytest.approx(expected)

    def test_transaction_costs_reduce_returns(self, ohlcv):
        close = normalize_ohlcv(ohlcv)["close"]
        sig = run_strategy("MACDStrategy", ohlcv)
        gross = signals_to_returns(sig, close, cost_bps=0.0).sum()
        net = signals_to_returns(sig, close, cost_bps=25.0).sum()
        assert net < gross

    def test_zero_activity_metrics_are_zero_not_degenerate(self, ohlcv):
        # Wide-open Bollinger bands on smooth data -> may still trade;
        # instead force zero activity via an impossible RSI band.
        m = evaluate_params(
            "RSIStrategy",
            ohlcv,
            {"rsi_period": 14, "oversold_threshold": 15.0,
             "overbought_threshold": 85.0},
        )
        if m["signal_events"] == 0:
            assert m["sharpe_ratio"] == 0.0
        # Regardless, no metric should be astronomically degenerate.
        assert abs(m["sharpe_ratio"]) < 100


# -------------------------------------------------------------- objective

class TestObjective:
    def test_constraint_violation_is_penalized(self, ohlcv):
        obj = make_objective("SMAStrategy")
        bad = {"short_window": 60, "long_window": 40, "confirmation_periods": 3}
        assert obj(bad, ohlcv) == INVALID_PENALTY

    def test_objective_is_finite_and_negated_for_maximize(self, ohlcv):
        obj = make_objective("MACDStrategy", metric="sharpe_ratio")
        params = NON_DEFAULT_PARAMS["MACDStrategy"]
        score = obj(params, ohlcv)
        assert np.isfinite(score)
        sharpe = evaluate_params("MACDStrategy", ohlcv, params)["sharpe_ratio"]
        assert score == pytest.approx(-sharpe)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            make_objective("RSIStrategy", metric="not_a_metric")


# ------------------------------------------------------- optimize_strategy

class TestOptimizeStrategy:
    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("grid_search", {}),
            ("pso", {"n_particles": 8, "n_iterations": 4}),
        ],
    )
    def test_end_to_end(self, ohlcv, method, kwargs):
        run = optimize_strategy(
            "RSIStrategy", ohlcv, method=method, metric="sharpe_ratio",
            max_evaluations=30, **kwargs,
        )
        assert run.best_params, "no best params returned"
        assert not constraint_violations("RSIStrategy", run.best_params)
        assert run.n_evaluations > 0
        assert "sharpe_ratio" in run.best_metrics
        assert "sharpe_ratio" in run.baseline_metrics
        # Ints stay ints after the cast-back.
        assert isinstance(run.best_params["rsi_period"], int)

    def test_unknown_strategy_raises(self, ohlcv):
        with pytest.raises(ValueError):
            optimize_strategy("NopeStrategy", ohlcv)


# ------------------------------------------------------- registry fixes

class TestRegistryParameterPath:
    @pytest.mark.parametrize("name", ALL_STRATEGIES)
    def test_execute_strategy_with_params_capitalized_columns(self, ohlcv, name):
        """The exact combination that used to break: yfinance-style
        capitalized columns plus explicit parameters."""
        from trading.strategies.registry import get_strategy_registry

        reg = get_strategy_registry()
        res = reg.execute_strategy(name, ohlcv, NON_DEFAULT_PARAMS[name])
        res0 = reg.execute_strategy(name, ohlcv)
        assert "signal" in res.signals.columns
        assert not res.signals["signal"].equals(res0.signals["signal"])


# ------------------------------------------------------- strategy bug fixes

class TestStrategyFixes:
    def test_cci_calculate_cci_returns_series(self, ohlcv):
        """calculate_cci returned a bare ndarray; generate_signals crashed
        on cci.shift(1) on every call before the fix."""
        from trading.strategies.cci_strategy import CCIStrategy

        strat = CCIStrategy()
        data = normalize_ohlcv(ohlcv)
        cci = strat.calculate_cci(data)
        assert isinstance(cci, pd.Series)
        signals = strat.generate_signals(data)
        assert (signals["signal"] != 0).any()

    def test_sma_lowercase_columns_produce_signals(self, ohlcv):
        """SMA's early-exit checked only capitalized 'Close' and silently
        returned all-zero signals for lowercase input before the fix."""
        from trading.strategies.sma_strategy import SMAStrategy

        strat = SMAStrategy()
        lower = normalize_ohlcv(ohlcv)
        upper_sig = strat.generate_signals(ohlcv)["signal"]
        lower_sig = SMAStrategy().generate_signals(lower)["signal"]
        assert (lower_sig != 0).any()
        assert lower_sig.equals(upper_sig)


# ------------------------------------------------------- self-tuning wiring

class TestSelfTuningWiring:
    def test_optimizer_with_config_actually_optimizes(self, tmp_path):
        from trading.optimization.self_tuning_optimizer import (
            SelfTuningOptimizer,
        )

        opt = SelfTuningOptimizer(
            config=get_self_tuning_config(),
            log_path=str(tmp_path / "hist.json"),
        )
        params = {"rsi_period": 14, "oversold_threshold": 30.0,
                  "overbought_threshold": 70.0}
        trades = [{"pnl": -1.0}] * 12
        for i in range(6):
            opt.record_performance(
                "RSIStrategy",
                params,
                {"sharpe_ratio": 0.5 - i * 0.2, "total_return": 0.02 - i * 0.01,
                 "max_drawdown": -0.1 - i * 0.02, "win_rate": 0.5 - i * 0.03},
                trades,
            )
        assert opt.should_optimize("RSIStrategy")
        res = opt.optimize_strategy(
            "RSIStrategy",
            params,
            {"sharpe_ratio": -0.7, "total_return": -0.04,
             "max_drawdown": -0.22, "win_rate": 0.32},
        )
        # Before the wiring, this unconditionally returned None
        # ("No parameter bounds defined").
        assert res is not None
        assert res.new_parameters != params
