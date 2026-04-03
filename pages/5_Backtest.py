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

import runpy

# TODO S44: inline old_3_Strategy_Testing.py
# Keeping runpy — file is too large to
# safely inline in one session
try:
    runpy.run_path(
        str(project_root / "scripts" / "old_3_Strategy_Testing.py"),
        run_name="__main__",
    )
except Exception as e:
    import traceback

    st.warning(
        "⚠️ Strategy testing module failed "
        "to load."
    )
    with st.expander("Error details", expanded=False):
        st.code(traceback.format_exc())
    if st.button("🔄 Retry", key="backtest_retry"):
        st.rerun()

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
