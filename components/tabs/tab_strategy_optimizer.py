# -*- coding: utf-8 -*-
"""Strategy Optimizer tab (pages/5_Backtest.py).

First live wiring of the audited optimization cluster (grid search /
genetic / PSO / Bayesian). The cluster was verified working end-to-end by
the audit but had zero call sites in the app; this tab exposes it as a real
feature: pick a strategy, an objective, and a search method; the optimizer
searches the canonical parameter space against real OHLCV history and
reports optimized-vs-default performance with full transparency
(convergence, evaluation count, exact parameters).

Optimized parameters can be applied with one click: they are stored in
``st.session_state["evolve_optimized_params"][strategy]`` and the Strategy
backtest tab picks them up automatically.
"""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

_METHODS = {
    "Grid search": ("grid_search", "Exhaustive/sampled sweep. Deterministic, best for small spaces."),
    "Bayesian": ("bayesian", "Gaussian-process guided search. Best score per evaluation; slower per step."),
    "Genetic": ("genetic", "Evolutionary search. Good for rugged, interacting parameters."),
    "Particle swarm": ("pso", "Swarm search. Fast, good middle ground."),
}

_ACCENT = "#00d4ff"
_GREEN = "#00e08a"
_RED = "#ff4d6d"


def _fmt_metric(name: str, value: float) -> str:
    if name in ("total_return", "annualized_return", "max_drawdown",
                "volatility_annual", "win_rate"):
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def render() -> None:
    """Render the Strategy Optimizer tab."""
    try:
        from trading.optimization.strategy_backtest_objective import (
            METRIC_CHOICES,
            optimize_strategy,
        )
        from trading.optimization.strategy_param_spaces import (
            STRATEGY_PARAM_SPACES,
            get_default_params,
            get_param_specs,
        )
    except Exception as e:
        st.warning(f"Optimizer backend unavailable: {e}")
        return

    st.markdown("#### Strategy optimizer")
    st.caption(
        "Search a strategy's parameter space against real history and compare "
        "against its defaults. Transaction costs are charged inside the "
        "objective, so hyperactive parameter sets don't win by accident. "
        "Results are in-sample — treat them as candidates to validate "
        "walk-forward, not as guarantees."
    )

    strategies = sorted(STRATEGY_PARAM_SPACES.keys())

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        strat = st.selectbox("Strategy", strategies, key="opt_strategy")
    with c2:
        symbol = st.text_input("Symbol", "SPY", key="opt_symbol").strip().upper()
    with c3:
        method_label = st.selectbox("Method", list(_METHODS.keys()), key="opt_method")
    method, method_help = _METHODS[method_label]
    st.caption(method_help)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        metric = st.selectbox(
            "Objective",
            list(METRIC_CHOICES.keys()),
            format_func=lambda k: METRIC_CHOICES[k],
            key="opt_metric",
        )
    with c5:
        budget = st.slider("Evaluation budget", 20, 300, 120, 20, key="opt_budget")
    with c6:
        cost_bps = st.number_input(
            "Cost per side (bps)", 0.0, 50.0, 5.0, 0.5, key="opt_cost_bps"
        )
    with c7:
        lookback_years = st.selectbox(
            "History", [1, 2, 3, 5], index=1,
            format_func=lambda y: f"{y}y", key="opt_lookback",
        )

    specs = get_param_specs(strat)
    with st.expander("Parameter space", expanded=False):
        space_df = pd.DataFrame(
            [
                {
                    "Parameter": s.label or s.name,
                    "Type": s.kind,
                    "Min": s.low,
                    "Max": s.high,
                    "Step": s.step,
                    "Default": s.default,
                }
                for s in specs
            ]
        )
        st.dataframe(space_df, hide_index=True, width="stretch")
        selected = st.multiselect(
            "Tune only these parameters (empty = all)",
            [s.name for s in specs],
            key="opt_selected_params",
        )

    applied = st.session_state.get("evolve_optimized_params", {})
    if strat in applied:
        st.info(
            f"Optimized parameters are currently applied to {strat} in the "
            f"Strategy backtest tab: `{applied[strat]}`"
        )
        if st.button("Clear applied parameters", key="opt_clear_applied"):
            applied.pop(strat, None)
            st.session_state["evolve_optimized_params"] = applied
            st.rerun()

    if st.button("Run optimization", type="primary", key="opt_run"):
        if not symbol:
            st.warning("Enter a symbol.")
            return
        try:
            import yfinance as yf

            with st.spinner(f"Loading {symbol} history…"):
                start = (date.today() - timedelta(days=365 * lookback_years)).isoformat()
                raw = yf.Ticker(symbol).history(start=start, auto_adjust=True)
            if raw is None or raw.empty:
                st.warning(f"No price data for {symbol}.")
                return
            with st.spinner(
                f"Optimizing {strat} via {method_label.lower()} "
                f"({budget} evaluation budget)…"
            ):
                run = optimize_strategy(
                    strat,
                    raw,
                    method=method,
                    metric=metric,
                    selected_params=selected or None,
                    max_evaluations=int(budget),
                    cost_bps=float(cost_bps),
                )
            st.session_state["opt_last_run"] = run
            st.session_state["opt_last_symbol"] = symbol
        except Exception:
            import traceback

            st.warning("Optimization failed.")
            with st.expander("Error details", expanded=False):
                st.code(traceback.format_exc())
            return

    run = st.session_state.get("opt_last_run")
    if run is None:
        return
    sym = st.session_state.get("opt_last_symbol", "")

    st.markdown("---")
    st.markdown(f"#### Results — {run.strategy_name} on {sym}")
    st.caption(
        f"{run.n_evaluations} evaluations · {run.optimization_time:.1f}s · "
        f"objective: {METRIC_CHOICES.get(run.metric, run.metric)}"
    )

    base_v = run.baseline_metrics.get(run.metric, 0.0)
    best_v = run.best_metrics.get(run.metric, 0.0)
    improved = (best_v > base_v) if run.metric != "max_drawdown" else (
        abs(best_v) < abs(base_v)
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            METRIC_CHOICES.get(run.metric, run.metric),
            _fmt_metric(run.metric, best_v),
            delta=f"{best_v - base_v:+.2f} vs defaults"
            if run.metric not in ("total_return", "max_drawdown")
            else f"{(best_v - base_v) * 100:+.2f}pp vs defaults",
        )
    with m2:
        st.metric(
            "Sharpe",
            _fmt_metric("sharpe_ratio", run.best_metrics["sharpe_ratio"]),
            delta=f"{run.best_metrics['sharpe_ratio'] - run.baseline_metrics['sharpe_ratio']:+.2f}",
        )
    with m3:
        st.metric(
            "Total return",
            _fmt_metric("total_return", run.best_metrics["total_return"]),
            delta=f"{(run.best_metrics['total_return'] - run.baseline_metrics['total_return']) * 100:+.1f}pp",
        )
    with m4:
        st.metric(
            "Max drawdown",
            _fmt_metric("max_drawdown", run.best_metrics["max_drawdown"]),
            delta=f"{(run.best_metrics['max_drawdown'] - run.baseline_metrics['max_drawdown']) * 100:+.1f}pp",
            delta_color="inverse",
        )

    if not improved:
        st.info(
            "The search did not beat the default parameters on this objective "
            "— that's a legitimate result, not a failure. Defaults may already "
            "sit near this space's optimum for this symbol and window, or the "
            "budget may be too small."
        )

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("##### Parameters")
        rows = []
        defaults = run.baseline_params or get_default_params(run.strategy_name)
        for k in sorted(set(defaults) | set(run.best_params)):
            rows.append(
                {
                    "Parameter": k,
                    "Default": defaults.get(k, "—"),
                    "Optimized": run.best_params.get(k, "—"),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if run.best_params:
            if st.button(
                "Apply to strategy backtest", key="opt_apply", type="primary"
            ):
                applied = st.session_state.get("evolve_optimized_params", {})
                applied[run.strategy_name] = dict(run.best_params)
                st.session_state["evolve_optimized_params"] = applied
                st.success(
                    f"Applied. The Strategy backtest tab will now run "
                    f"{run.strategy_name} with these parameters."
                )

    with right:
        st.markdown("##### Convergence")
        history = [
            s for s in (run.convergence_history or []) if abs(s) < 1e8
        ]
        if len(history) >= 2:
            import plotly.graph_objects as go

            best_so_far = np.minimum.accumulate(np.asarray(history, dtype=float))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=history,
                    mode="markers",
                    name="Evaluations",
                    marker=dict(color="#4a6080", size=5, opacity=0.6),
                )
            )
            fig.add_trace(
                go.Scatter(
                    y=best_so_far,
                    mode="lines",
                    name="Best so far",
                    line=dict(color=_ACCENT, width=2),
                )
            )
            fig.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Evaluation",
                yaxis_title="Objective (lower is better)",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig, width="stretch", key="opt_convergence")
        else:
            st.caption(
                "Not enough finite evaluations to chart convergence "
                "(most candidates may have violated constraints)."
            )
