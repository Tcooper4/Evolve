# -*- coding: utf-8 -*-
"""
Analyze page: tab definitions stay here; bodies live in components/tabs/.
"""
import streamlit as st

from components.tabs.tab_ai_model_selection import render as render_ai_model_selection
from components.tabs.tab_backtester import render as render_backtester
from components.tabs.tab_causal import render as render_causal
from components.tabs.tab_diagnostics import render as render_diagnostics
from components.tabs.tab_earnings import render as render_earnings
from components.tabs.tab_market_analysis import render as render_market_analysis
from components.tabs.tab_model_comparison import render as render_model_comparison
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab_options, tab_insider, tab_earnings, tab_diag = st.tabs(
        [
            "🚀 Quick Forecast",
            "⚙️ Advanced Forecasting",
            "🤖 AI Model Selection",
            "📊 Model Comparison",
            "📈 Market Analysis",
            "🔗 Multi-Asset (GNN)",
            "🎲 Monte Carlo",
            "📊 Options & Short",
            "🕵️ Insider Flow",
            "📅 Earnings",
            "📐 Diagnostics",
        ]
    )

    with tab1:
        render_quick_forecast(**kw)
    with tab2:
        render_backtester(**kw)
    with tab3:
        render_ai_model_selection(**kw)
    with tab4:
        render_model_comparison(**kw)
    with tab5:
        render_market_analysis(**kw)
    with tab6:
        render_multi_asset_gnn(**kw)
    with tab7:
        render_monte_carlo(**kw)
    with tab_options:
        render_options_chain(**kw)
    with tab_insider:
        render_scanner_signals(**kw)
    with tab_earnings:
        render_earnings(**kw)
    with tab_diag:
        render_diagnostics(**kw)
        render_causal(**kw)
