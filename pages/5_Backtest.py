# -*- coding: utf-8 -*-
"""
Backtest page -- Strategy testing, walk-forward, RL trainer, reports.
"""
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

import numpy as np
import pandas as pd

from utils.dataframe_utils import normalize_for_display
from utils.risk_metrics import compute_performance_metrics
from trading.backtesting.backtester import Backtester
from trading.backtesting.trade_models import TradeType
from trading.strategies.registry import get_strategy_registry
from trading.data.ticker_resolver import resolve_ticker

logger = logging.getLogger(__name__)


def _get_forecasting_backend():
    try:
        from trading.data.data_loader import DataLoader, DataLoadRequest
        from trading.data.providers.yfinance_provider import YFinanceProvider
        from trading.models.lstm_model import LSTMForecaster
        from trading.models.xgboost_model import XGBoostModel
        from trading.models.prophet_model import ProphetModel
        from trading.models.arima_model import ARIMAModel
        from trading.data.preprocessing import FeatureEngineering, DataPreprocessor
        from trading.agents.model_selector_agent import ModelSelectorAgent
        from trading.market.market_analyzer import MarketAnalyzer

        return {
            "DataLoader": DataLoader,
            "DataLoadRequest": DataLoadRequest,
            "YFinanceProvider": YFinanceProvider,
            "LSTMForecaster": LSTMForecaster,
            "XGBoostModel": XGBoostModel,
            "ProphetModel": ProphetModel,
            "ARIMAModel": ARIMAModel,
            "FeatureEngineering": FeatureEngineering,
            "DataPreprocessor": DataPreprocessor,
            "ModelSelectorAgent": ModelSelectorAgent,
            "MarketAnalyzer": MarketAnalyzer,
        }
    except Exception as e:
        logger.warning("Backtest: forecasting backend not available: %s", e)
        return None


if "forecasting_backend" not in st.session_state:
    st.session_state.forecasting_backend = _get_forecasting_backend()


def _backtest_signal_series(strat_signals, price_index: pd.Index):
    """Align strategy output to price index as a float signal per bar."""
    if strat_signals is None or strat_signals.empty:
        return pd.Series(0.0, index=price_index)
    if "signal" in strat_signals.columns:
        col = strat_signals["signal"]
    else:
        num = strat_signals.select_dtypes(include=[np.number])
        col = num.iloc[:, 0] if not num.empty else pd.Series(
            0.0, index=strat_signals.index
        )
    aligned = col.reindex(price_index).ffill().fillna(0.0)
    return aligned.astype(float)


st.markdown("### Backtest")
st.caption(
    "Walk-forward validation, multi-model comparison, and historical strategy tests"
)

tab_wf, tab_compare, tab_backtest, tab_optimizer = st.tabs([
    "Walk-forward validation",
    "Strategy comparison",
    "Backtest",
    "Optimizer",
])

with tab_wf:
    st.caption(
        "Walk-forward results are cached and used by the forecast router to qualify "
        "confidence on AI recommendations. Run for any ticker you actively trade."
    )
    _wf_preview = str(st.session_state.get("wf_symbol", "AAPL")).strip().upper()
    _wf_key = f"wf_{_wf_preview}"
    if _wf_key in st.session_state:
        cached = st.session_state[_wf_key]
        st.success(
            f"Cached result for {_wf_preview}: Directional accuracy "
            f"{cached.get('mean_directional_accuracy', 0):.1%}"
        )

    st.markdown("#### Walk-forward validation")
    st.caption(
        "Tests how well a model would have performed using rolling out-of-sample "
        "windows."
    )

    wf_col1, wf_col2, wf_col3 = st.columns(3)
    with wf_col1:
        wf_symbol = st.text_input(
            "Symbol",
            "AAPL",
            key="wf_symbol",
        )
        wf_symbol = resolve_ticker(
            wf_symbol.strip().upper(),
            validate=False,
        )
    with wf_col2:
        wf_model = st.selectbox(
            "Model",
            ["xgboost", "ridge", "arima", "catboost", "prophet"],
            key="wf_model",
        )
    with wf_col3:
        wf_window_type = st.selectbox(
            "Window Type",
            ["expanding", "rolling"],
            key="wf_window_type",
        )

    wf_col4, wf_col5, wf_col6 = st.columns(3)
    with wf_col4:
        wf_train = st.number_input(
            "Train Window (days)",
            min_value=60,
            max_value=504,
            value=252,
            step=21,
            key="wf_train",
        )
    with wf_col5:
        wf_test = st.number_input(
            "Test Window (days)",
            min_value=21,
            max_value=126,
            value=63,
            step=21,
            key="wf_test",
        )
    with wf_col6:
        wf_step = st.number_input(
            "Step Size (days)",
            min_value=5,
            max_value=63,
            value=21,
            step=5,
            key="wf_step",
        )

    if st.button(
        "Run walk-forward validation",
        key="wf_run_btn",
        type="primary",
    ):
        try:
            from trading.data.price_cache import get_history
            from trading.validation.walk_forward_utils import WalkForwardValidator

            with st.spinner(
                f"Running walk-forward validation on "
                f"{wf_symbol} with {wf_model}..."
            ):
                hist = get_history(
                    wf_symbol.strip().upper(),
                    period="3y",
                )
                if hist.empty or len(hist) < wf_train + wf_test:
                    st.warning(
                        f"Insufficient data for {wf_symbol}. "
                        f"Need at least {wf_train + wf_test} trading days."
                    )
                else:
                    validator = WalkForwardValidator(
                        model_name=wf_model,
                        symbol=wf_symbol,
                        window_type=wf_window_type,
                    )
                    wf_result = validator.run(
                        data=hist,
                        train_window=int(wf_train),
                        test_window=int(wf_test),
                        step_size=int(wf_step),
                    )
                    summary = wf_result.model_performance

                    if summary.get("error"):
                        st.warning(
                            f"Validation error: {summary['error']}"
                        )
                    else:
                        st.markdown("#### Results summary")
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Windows", summary.get("n_windows", 0))
                        m2.metric(
                            "Mean MAPE",
                            f"{summary.get('mean_mape', 0):.1f}%"
                            if summary.get("mean_mape") is not None
                            else "N/A",
                        )
                        m3.metric(
                            "Directional Accuracy",
                            f"{summary.get('mean_directional_accuracy', 0)*100:.1f}%"
                            if summary.get("mean_directional_accuracy")
                            is not None
                            else "N/A",
                        )
                        m4.metric(
                            "Hit rate (MAPE < 5%)",
                            f"{summary.get('hit_rate_5pct', 0)*100:.1f}%"
                            if summary.get("hit_rate_5pct") is not None
                            else "N/A",
                        )
                        m5.metric(
                            "Hit rate (MAPE < 10%)",
                            f"{summary.get('hit_rate_10pct', 0)*100:.1f}%"
                            if summary.get("hit_rate_10pct") is not None
                            else "N/A",
                        )
                        m6.metric(
                            "Consistency Score",
                            f"{summary.get('consistency_score', 0):.1f}/10"
                            if summary.get("consistency_score") is not None
                            else "N/A",
                        )

                        df = wf_result.to_dataframe()
                        if not df.empty:
                            st.markdown("#### Per-window results")
                            st.dataframe(
                                normalize_for_display(df),
                                width='stretch',
                                key="wf_results_df",
                            )

                        st.session_state["wf_last_results"] = summary
                        st.session_state[
                            f"wf_{wf_symbol.strip().upper()}"
                        ] = summary

        except Exception as e:
            st.caption(f"Walk-forward analysis unavailable: {e}")

with tab_compare:
    from components.tabs.tab_backtester import render as render_backtester
    from components.tabs.tab_model_comparison import render as render_model_comparison
    from trading.data.price_cache import get_history

    _be = st.session_state.get("forecasting_backend")
    st.markdown("#### Strategy comparison")
    st.caption(
        "Advanced forecasting configuration and side-by-side model comparison. "
        "Load price history on the Analyze page or enter a ticker below."
    )
    _cmp_sym = st.text_input(
        "Ticker",
        value=st.session_state.get("analyze_ticker", "AAPL"),
        key="backtest_compare_ticker",
    ).strip().upper()
    _cmp_sym = resolve_ticker(_cmp_sym, validate=False)
    _cmp_hist = get_history(_cmp_sym, period="1y") if _cmp_sym else pd.DataFrame()
    _kw = dict(
        ticker=_cmp_sym or "AAPL",
        hist=_cmp_hist,
        period="1y",
        period_label="1Y",
        trader_mode="Short-term",
        _interval="1d",
        _tf_label="1d",
        backend=_be,
    )
    if not _be:
        st.warning(
            "Forecasting backend could not be loaded; strategy comparison tabs "
            "need Analyze dependencies."
        )
    else:
        render_backtester(**_kw)
        st.markdown("---")
        render_model_comparison(**_kw)

with tab_backtest:
    st.markdown("#### Strategy backtest")
    st.caption(
        "Run a discovered strategy on historical OHLCV; equity curve uses the "
        "platform backtester with simple signal-follow rules."
    )

    try:
        _reg = get_strategy_registry()
        _names = sorted(_reg.get_strategy_names())
    except Exception as e:
        _reg = None
        _names = []
        st.caption(f"Strategy registry unavailable: {e}")

    if not _names:
        st.info(
            "No strategies were discovered. Check trading/strategies for "
            "*_strategy.py modules."
        )
    else:
        b1, b2, b3 = st.columns(3)
        with b1:
            bt_strategy = st.selectbox("Strategy", _names, key="bt_strategy_name")
        with b2:
            bt_symbol = st.text_input(
                "Symbol",
                "AAPL",
                key="bt_symbol",
            ).strip().upper()
            bt_symbol = resolve_ticker(bt_symbol, validate=False)
        with b3:
            bt_capital = st.number_input(
                "Initial capital ($)",
                min_value=1000.0,
                value=100000.0,
                step=1000.0,
                key="bt_initial_cash",
            )
        d1, d2 = st.columns(2)
        with d1:
            bt_start = st.date_input(
                "Start",
                value=date.today() - timedelta(days=365 * 2),
                key="bt_start",
            )
        with d2:
            bt_end = st.date_input(
                "End",
                value=date.today(),
                key="bt_end",
            )

        if st.button("Run backtest", key="bt_run_btn", type="primary"):
            if not bt_symbol:
                st.warning("Enter a symbol.")
            elif bt_start >= bt_end:
                st.warning("End date must be after start date.")
            elif _reg is None:
                st.warning("Strategy registry not available.")
            else:
                try:
                    import yfinance as yf

                    with st.spinner(
                        f"Loading {bt_symbol} and running {bt_strategy}…"
                    ):
                        raw = yf.Ticker(bt_symbol).history(
                            start=bt_start.isoformat(),
                            end=bt_end.isoformat(),
                            auto_adjust=True,
                        )
                        if raw.empty:
                            st.warning(
                                f"No price data for {bt_symbol} in range."
                            )
                        else:
                            _cm = {c.lower(): c for c in raw.columns}
                            _cc = _cm.get("close", raw.columns[0])
                            closes = raw[_cc].astype(float)
                            price_df = pd.DataFrame({bt_symbol: closes})
                            price_df = price_df.dropna()

                            _opt_params = st.session_state.get(
                                "evolve_optimized_params", {}
                            ).get(bt_strategy)
                            if _opt_params:
                                st.caption(
                                    f"Using optimized parameters from the "
                                    f"Optimizer tab: `{_opt_params}`"
                                )
                            strat_res = _reg.execute_strategy(
                                bt_strategy, raw, _opt_params
                            )
                            sig_series = _backtest_signal_series(
                                strat_res.signals, price_df.index
                            )

                            bt_engine = Backtester(
                                data=price_df,
                                initial_cash=float(bt_capital),
                            )
                            buy_thr = 0.01
                            sell_thr = -0.01
                            for ts in price_df.index:
                                price = float(price_df.loc[ts, bt_symbol])
                                sig = float(sig_series.loc[ts])
                                pos = float(
                                    bt_engine.positions.get(bt_symbol, 0) or 0
                                )
                                if sig > buy_thr and pos < 1e-9:
                                    spend = bt_engine.cash_account * 0.95
                                    qty = int(spend / price) if price > 0 else 0
                                    if qty > 0:
                                        bt_engine.execute_trade(
                                            ts,
                                            bt_symbol,
                                            float(qty),
                                            price,
                                            TradeType.BUY,
                                            bt_strategy,
                                            sig,
                                        )
                                elif sig < sell_thr and pos > 1e-9:
                                    bt_engine.execute_trade(
                                        ts,
                                        bt_symbol,
                                        pos,
                                        price,
                                        TradeType.SELL,
                                        bt_strategy,
                                        sig,
                                    )

                            results = bt_engine.run()
                            ec = results.get("equity_curve")
                            mraw = results.get("metrics") or {}

                            st.markdown("#### Results")
                            r1, r2, r3, r4 = st.columns(4)
                            sharpe_v = None
                            mdd_v = None
                            wr_v = None
                            if (
                                ec is not None
                                and not ec.empty
                                and "equity_curve" in ec.columns
                            ):
                                rets = ec["equity_curve"].pct_change().dropna()
                                if len(rets) > 5:
                                    pm = compute_performance_metrics(rets)
                                    sharpe_v = pm.sharpe_ratio
                                    mdd_v = pm.max_drawdown
                                    wr_v = pm.win_rate
                            tr = mraw.get("total_return")
                            if tr is None and ec is not None and not ec.empty:
                                eq = ec["equity_curve"].dropna()
                                if len(eq) > 1:
                                    tr = float(eq.iloc[-1] / eq.iloc[0] - 1)
                            with r1:
                                st.metric(
                                    "Total return",
                                    f"{float(tr or 0)*100:.2f}%"
                                    if tr is not None
                                    else "N/A",
                                )
                            with r2:
                                st.metric(
                                    "Sharpe (approx)",
                                    f"{sharpe_v:.2f}"
                                    if sharpe_v is not None
                                    else "N/A",
                                )
                            with r3:
                                st.metric(
                                    "Max drawdown",
                                    f"{float(mdd_v)*100:.2f}%"
                                    if mdd_v is not None
                                    else "N/A",
                                )
                            with r4:
                                st.metric(
                                    "Win rate (daily)",
                                    f"{float(wr_v)*100:.1f}%"
                                    if wr_v is not None
                                    else "N/A",
                                )

                            trades = results.get("trades") or []
                            if trades:
                                st.markdown("#### Trades")
                                tdf = pd.DataFrame(trades)
                                st.dataframe(
                                    normalize_for_display(tdf),
                                    width='stretch',
                                    key="bt_trades_df",
                                )

                            if ec is not None and not ec.empty:
                                st.markdown("#### Equity curve")
                                import plotly.graph_objects as go

                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=ec.index,
                                        y=ec["equity_curve"],
                                        mode="lines",
                                        name="Equity",
                                    )
                                )
                                fig.update_layout(
                                    height=400,
                                    margin=dict(l=20, r=20, t=40, b=20),
                                    title=f"{bt_symbol} — {bt_strategy}",
                                )
                                st.plotly_chart(fig, width="stretch")

                except Exception as ex:
                    import traceback

                    st.warning("Backtest failed.")
                    with st.expander("Error details", expanded=False):
                        st.code(traceback.format_exc())
                    st.caption(f"{type(ex).__name__}: {ex}")

    st.markdown("---")
    st.markdown("#### Enhanced multi-model backtest")
    try:
        from trading.backtesting.enhanced_backtester import run_multi_model_backtest
        from trading.models.arima_model import ARIMAModel
        from trading.models.ridge_model import RidgeModel

        _e_sym = st.text_input(
            "Symbol (enhanced)",
            value="AAPL",
            key="enhanced_bt_symbol",
        ).strip().upper()
        _e_sym = resolve_ticker(_e_sym, validate=False)
        if st.button("Run enhanced backtest", key="enhanced_bt_btn", type="primary"):
            with st.spinner("Running multi-model comparison..."):
                import yfinance as yf

                _raw = yf.Ticker(_e_sym).history(period="2y", auto_adjust=True)
                if _raw.empty:
                    st.warning("No data.")
                else:
                    _cm = {c.lower(): c for c in _raw.columns}
                    _cc = _cm.get("close", _raw.columns[0])
                    _ohlc = _raw.rename(
                        columns={
                            _cc: "close",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Volume": "volume",
                        }
                    )
                    _reg = get_strategy_registry()
                    _sn = st.session_state.get("bt_strategy_name") or (
                        _names[0] if _names else ""
                    )
                    _st = _reg.get_strategy(_sn) if _reg and _sn else None
                    if _st is None:
                        st.caption("Pick a valid strategy above for enhanced run.")
                    else:
                        _models = [
                            ARIMAModel(
                                {
                                    "order": (2, 1, 0),
                                    "use_auto_arima": True,
                                    "target_column": "close",
                                }
                            ),
                            RidgeModel(
                                {
                                    "target_column": "close",
                                    "alpha": 1.0,
                                    "max_iter": 1000,
                                }
                            ),
                        ]
                        em_results = run_multi_model_backtest(
                            _ohlc,
                            _models,
                            _st,
                            _e_sym,
                            forecast_period=7,
                        )
                        if em_results and not em_results.get("error"):
                            inner = em_results.get("results") or {}
                            pm = inner.get("performance_metrics") or {}
                            ec2 = inner.get("equity_curve")
                            sharpe_e = None
                            if (
                                ec2 is not None
                                and hasattr(ec2, "empty")
                                and not ec2.empty
                            ):
                                _col_eq = (
                                    "equity_curve"
                                    if "equity_curve" in ec2.columns
                                    else ec2.columns[0]
                                )
                                _rets = ec2[_col_eq].pct_change().dropna()
                                if len(_rets) > 5:
                                    sharpe_e = compute_performance_metrics(
                                        _rets
                                    ).sharpe_ratio
                            mdl = em_results.get("models") or []
                            best_label = (
                                mdl[0]
                                if len(mdl) == 1
                                else "Ensemble"
                            )
                            if len(mdl) > 1:
                                best_label = f"{len(mdl)} models"
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Models run", len(mdl))
                            c2.metric("Best / mode", str(best_label)[:28])
                            c3.metric(
                                "Ensemble return",
                                f"{float(pm.get('total_return', 0)):.1%}",
                            )
                            c4.metric(
                                "Sharpe (approx)",
                                f"{sharpe_e:.2f}"
                                if sharpe_e is not None
                                else "N/A",
                            )
                            st.caption(
                                f"Strategy: {em_results.get('strategy', '')} · "
                                f"Symbol: {em_results.get('symbol', '')}"
                            )
                        else:
                            st.caption(
                                f"Enhanced backtest: "
                                f"{em_results.get('error', 'no result')}"
                            )
    except Exception as e:
        st.caption(f"Enhanced backtest unavailable: {e}")

with tab_optimizer:
    try:
        from components.tabs.tab_strategy_optimizer import (
            render as render_strategy_optimizer,
        )

        render_strategy_optimizer()
    except Exception as e:
        st.warning(f"Strategy optimizer unavailable: {e}")

try:
    from ui.page_assistant import render_page_assistant

    render_page_assistant("Backtest")
except Exception:
    pass
