# -*- coding: utf-8 -*-
"""
Analyze page: tab definitions stay here; bodies live in components/tabs/.
"""
import streamlit as st

from components.tabs.tab_ai_model_selection import render as render_ai_model_selection
from components.tabs.tab_causal import render as render_causal
from components.tabs.tab_diagnostics import render as render_diagnostics
from components.tabs.tab_earnings import render as render_earnings
from components.tabs.tab_market_analysis import render as render_market_analysis
from components.tabs.tab_monte_carlo import render as render_monte_carlo
from components.tabs.tab_multi_asset_gnn import render as render_multi_asset_gnn
from components.tabs.tab_options_chain import render as render_options_chain
from components.tabs.tab_quick_forecast import render as render_quick_forecast
from components.tabs.tab_scanner_signals import render as render_scanner_signals


def render_tabbed_analyze_sections(
    ticker: str,
    hist,
    period: str,
    period_label: str,
    trader_mode: str,
    _interval: str,
    _tf_label: str,
) -> None:
    _be = st.session_state.get("forecasting_backend")
    if not _be:
        st.error("Forecasting backend could not be loaded.")
        return
    backend = {
        "DataLoader": _be["DataLoader"],
        "DataLoadRequest": _be["DataLoadRequest"],
        "YFinanceProvider": _be["YFinanceProvider"],
        "LSTMForecaster": _be["LSTMForecaster"],
        "XGBoostModel": _be["XGBoostModel"],
        "ProphetModel": _be["ProphetModel"],
        "ARIMAModel": _be["ARIMAModel"],
        "FeatureEngineering": _be["FeatureEngineering"],
        "DataPreprocessor": _be["DataPreprocessor"],
        "ModelSelectorAgent": _be["ModelSelectorAgent"],
        "MarketAnalyzer": _be["MarketAnalyzer"],
    }
    kw = dict(
        ticker=ticker,
        hist=hist,
        period=period,
        period_label=period_label,
        trader_mode=trader_mode,
        _interval=_interval,
        _tf_label=_tf_label,
        backend=backend,
    )

    st.markdown("---")
    tab_forecast, tab_risk, tab_market, tab_technical, tab_research = st.tabs(
        [
            "Forecast",
            "Risk",
            "Market",
            "Technical",
            "Research",
        ]
    )

    with tab_forecast:
        render_quick_forecast(**kw)
        st.markdown("---")
        with st.expander("Model selection details"):
            render_ai_model_selection(**kw)

    with tab_risk:
        c_mc, c_opt = st.columns([3, 2])
        with c_mc:
            render_monte_carlo(**kw)
        with c_opt:
            render_options_chain(**kw)
        st.markdown("---")
        render_scanner_signals(**kw)

    with tab_market:
        render_market_analysis(**kw)
        st.markdown("---")
        render_earnings(**kw)

    with tab_technical:
        render_diagnostics(**kw)
        st.markdown("---")
        st.caption(
            "Chart patterns feed into AI Score automatically. View detected patterns "
            "in the deep dive for this ticker."
        )
        if st.button(
            f"Open deep dive for {ticker}",
            key="analyze_open_deep_dive",
            type="primary",
        ):
            st.session_state["deep_dive_ticker"] = ticker
            st.switch_page("pages/1_Dashboard.py")

    with tab_research:
        render_multi_asset_gnn(**kw)
        st.markdown("---")
        render_causal(**kw)
