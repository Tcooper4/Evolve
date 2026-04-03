# -*- coding: utf-8 -*-
"""
Backtest page -- Strategy testing, walk-forward, RL trainer, reports.
"""
import sys
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


st.markdown("---")
st.subheader("Strategy backtest")
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
        bt_symbol = st.text_input("Symbol", "AAPL", key="bt_symbol").strip().upper()
    with b3:
        bt_capital = st.number_input(
            "Initial capital ($)",
            min_value=1000.0,
            value=100000.0,
            step=1000.0,
            key="bt_initial_cash",
        )
    d1, d2 = st.columns(2)
    from datetime import date, timedelta

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

    if st.button("▶ Run backtest", key="bt_run_btn", type="primary"):
        if not bt_symbol:
            st.warning("Enter a symbol.")
        elif bt_start >= bt_end:
            st.warning("End date must be after start date.")
        elif _reg is None:
            st.warning("Strategy registry not available.")
        else:
            try:
                import yfinance as yf

                with st.spinner(f"Loading {bt_symbol} and running {bt_strategy}…"):
                    raw = yf.Ticker(bt_symbol).history(
                        start=bt_start.isoformat(),
                        end=bt_end.isoformat(),
                        auto_adjust=True,
                    )
                    if raw.empty:
                        st.warning(f"No price data for {bt_symbol} in range.")
                    else:
                        _cm = {c.lower(): c for c in raw.columns}
                        _cc = _cm.get("close", raw.columns[0])
                        closes = raw[_cc].astype(float)
                        price_df = pd.DataFrame({bt_symbol: closes})
                        price_df = price_df.dropna()

                        strat_res = _reg.execute_strategy(
                            bt_strategy, raw
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
                            pos = float(bt_engine.positions.get(bt_symbol, 0) or 0)
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
                        if ec is not None and not ec.empty and "equity_curve" in ec.columns:
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
                                use_container_width=True,
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
                            st.plotly_chart(fig, use_container_width=True)

            except Exception as ex:
                import traceback

                st.warning("Backtest failed.")
                with st.expander("Error details", expanded=False):
                    st.code(traceback.format_exc())
                st.caption(f"{type(ex).__name__}: {ex}")

try:
    st.markdown("---")
    st.subheader("📊 Walk-Forward Validation")
    st.caption(
        "Tests how well a model would have performed "
        "using rolling out-of-sample windows."
    )

    wf_col1, wf_col2, wf_col3 = st.columns(3)
    with wf_col1:
        wf_symbol = st.text_input(
            "Symbol", "AAPL", key="wf_symbol"
        )
    with wf_col2:
        wf_model = st.selectbox(
            "Model",
            ["xgboost", "ridge", "arima",
             "catboost", "prophet"],
            key="wf_model"
        )
    with wf_col3:
        wf_window_type = st.selectbox(
            "Window Type",
            ["expanding", "rolling"],
            key="wf_window_type"
        )

    wf_col4, wf_col5, wf_col6 = st.columns(3)
    with wf_col4:
        wf_train = st.number_input(
            "Train Window (days)",
            min_value=60, max_value=504,
            value=252, step=21,
            key="wf_train"
        )
    with wf_col5:
        wf_test = st.number_input(
            "Test Window (days)",
            min_value=21, max_value=126,
            value=63, step=21,
            key="wf_test"
        )
    with wf_col6:
        wf_step = st.number_input(
            "Step Size (days)",
            min_value=5, max_value=63,
            value=21, step=5,
            key="wf_step"
        )

    if st.button(
        "▶ Run Walk-Forward Validation",
        key="wf_run_btn",
        type="primary"
    ):
        try:
            import yfinance as yf
            from trading.validation.walk_forward_utils import (
                WalkForwardValidator
            )
            from utils.dataframe_utils import normalize_for_display

            with st.spinner(
                f"Running walk-forward validation on "
                f"{wf_symbol} with {wf_model}..."
            ):
                hist = yf.Ticker(wf_symbol).history(
                    period="3y"
                )
                if hist.empty or len(hist) < wf_train + wf_test:
                    st.warning(
                        f"Insufficient data for {wf_symbol}. "
                        f"Need at least "
                        f"{wf_train + wf_test} trading days."
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
                            f"Validation error: "
                            f"{summary['error']}"
                        )
                    else:
                        # Summary metrics
                        st.markdown("#### Results Summary")
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric(
                            "Windows",
                            summary.get("n_windows", 0)
                        )
                        m2.metric(
                            "Mean MAPE",
                            f"{summary.get('mean_mape', 0):.1f}%"
                            if summary.get('mean_mape') is not None
                            else "N/A"
                        )
                        m3.metric(
                            "Directional Accuracy",
                            f"{summary.get('mean_directional_accuracy', 0)*100:.1f}%"
                            if summary.get('mean_directional_accuracy')
                            is not None
                            else "N/A"
                        )
                        m4.metric(
                            "Mean Sharpe",
                            f"{summary.get('mean_sharpe_ratio', 0):.2f}"
                            if summary.get('mean_sharpe_ratio') is not None
                            else "N/A"
                        )
                        m5.metric(
                            "Consistency Score",
                            f"{summary.get('consistency_score', 0):.1f}/10"
                            if summary.get('consistency_score') is not None
                            else "N/A"
                        )

                        # Per-window table
                        df = wf_result.to_dataframe()
                        if not df.empty:
                            st.markdown(
                                "#### Per-Window Results"
                            )
                            st.dataframe(
                                normalize_for_display(df),
                                use_container_width=True,
                                key="wf_results_df"
                            )

                        # Store in session state
                        st.session_state[
                            "wf_last_results"
                        ] = summary

        except Exception as e:
            st.caption(
                f"Walk-forward validation unavailable: {e}"
            )
except Exception as e:
    st.caption(
        f"Walk-forward section unavailable: {e}"
    )

# Page Assistant
try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Backtest")
except Exception:
    pass
