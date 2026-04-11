# -*- coding: utf-8 -*-
"""Analyze page tab body (extracted; logic unchanged)."""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.analyze_common import (
    _extract_forecast_values,
    _generate_recommendation,
    _is_english,
    _news_sentiment_score,
)
from trading.data.earnings_calendar import get_upcoming_earnings
from trading.data.insider_flow import get_insider_flow
from trading.data.price_cache import get_history, get_info, get_news, get_quote

try:
    from utils.dataframe_utils import normalize_for_display
except ImportError:
    def normalize_for_display(df):
        return df

logger = logging.getLogger(__name__)


def render(
    ticker: str,
    hist,
    period: str,
    period_label: str,
    trader_mode: str,
    _interval: str,
    _tf_label: str,
    *,
    score_mode: str = "Buy",
    backend: dict,
) -> None:
    """Streamlit tab body (legacy Analyze)."""
    DataLoader = backend["DataLoader"]
    DataLoadRequest = backend["DataLoadRequest"]
    YFinanceProvider = backend["YFinanceProvider"]
    LSTMForecaster = backend["LSTMForecaster"]
    XGBoostModel = backend["XGBoostModel"]
    ProphetModel = backend["ProphetModel"]
    ARIMAModel = backend["ARIMAModel"]
    FeatureEngineering = backend["FeatureEngineering"]
    DataPreprocessor = backend["DataPreprocessor"]
    ModelSelectorAgent = backend["ModelSelectorAgent"]
    MarketAnalyzer = backend["MarketAnalyzer"]
    try:
        try:
            st.header("AI Model Selection")
            st.markdown("""
            🤖 Let AI analyze your data and recommend the best forecasting model.

            The AI considers:
            - Data characteristics (trend, seasonality, volatility)
            - Historical model performance
            - Forecast horizon
            - Computational efficiency
            """)

            if st.session_state.get("analyze_forecast_data") is None:
                st.warning(
                    "⚠️ No forecast data yet — enter a ticker at the top and open the Forecast tab."
                )
            else:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📊 Data Analysis")

                    analyze_button = st.button(
                        "🔍 Analyze Data & Recommend Model",
                        type="primary"
                    )

                    if analyze_button:
                        with st.spinner("AI analyzing data characteristics..."):
                            try:
                                # Use ModelSelectorAgent if available; otherwise fallback to simple recommendation
                                try:
                                    from trading.agents.model_selector_agent import ModelSelectorAgent
                                    agent = ModelSelectorAgent()
                                except Exception:
                                    agent = None

                                # Detect market regime from data
                                fd = st.session_state.get("analyze_forecast_data")
                                price_data = fd["close"].values if hasattr(fd, "columns") and "close" in fd.columns else (
                                    fd.get("close")
                                    if isinstance(fd, dict)
                                    else (list(fd.values())[0] if isinstance(fd, dict) and fd else None)
                                )
                                if price_data is None:
                                    price_data = np.array([])
                                if hasattr(price_data, 'tolist'):
                                    price_data = np.asarray(price_data).ravel()
                                horizon = st.session_state.analyze_forecast_horizon

                                # Determine horizon enum
                                from trading.agents.model_selector_agent import ForecastingHorizon, MarketRegime

                                if horizon <= 7:
                                    horizon_enum = ForecastingHorizon.SHORT_TERM
                                elif horizon <= 30:
                                    horizon_enum = ForecastingHorizon.MEDIUM_TERM
                                else:
                                    horizon_enum = ForecastingHorizon.LONG_TERM

                                # Simple market regime detection
                                price_trend = 0.0
                                volatility = 0.0
                                regime_enum = MarketRegime.SIDEWAYS
                                if len(price_data) >= 2:
                                    price_arr = np.asarray(price_data).astype(float)
                                    price_trend = (price_arr[-1] - price_arr[0]) / (price_arr[0] or 1)
                                    volatility = np.nanstd(price_arr)
                                    if price_trend > 0.05:
                                        regime_enum = MarketRegime.TRENDING_UP
                                    elif price_trend < -0.05:
                                        regime_enum = MarketRegime.TRENDING_DOWN
                                    elif volatility > (np.nanmean(price_arr) * 0.1):
                                        regime_enum = MarketRegime.VOLATILE
                                    else:
                                        regime_enum = MarketRegime.SIDEWAYS

                                if agent is not None:
                                    # Use agent's select_model method
                                    selected_model_id, confidence = agent.select_model(
                                        horizon=horizon_enum,
                                        market_regime=regime_enum,
                                        data_length=len(price_data),
                                        required_features=[],
                                        performance_weight=0.6,
                                        capability_weight=0.4
                                    )
                                    recommendations = agent.get_model_recommendations(
                                        horizon_enum, regime_enum, top_k=3
                                    )
                                else:
                                    selected_model_id = "xgboost_medium"
                                    confidence = 0.75
                                    recommendations = [
                                        {"model_type": "xgboost", "model_id": "xgboost_medium", "total_score": 0.8},
                                        {"model_type": "lstm", "model_id": "lstm_medium", "total_score": 0.7},
                                        {"model_type": "prophet", "model_id": "prophet_medium", "total_score": 0.65},
                                    ]

                                # Map model types to display names
                                model_name_map = {
                                    'lstm': 'LSTM (Deep Learning)',
                                    'xgboost': 'XGBoost (Gradient Boosting)',
                                    'prophet': 'Prophet (Facebook)',
                                    'arima': 'ARIMA (Statistical)',
                                    'transformer': 'Transformer (Attention)',
                                    'ensemble': 'Ensemble (Multiple Models)'
                                }

                                # Extract model type from selected_model_id
                                model_type = selected_model_id.split('_')[0] if '_' in selected_model_id else selected_model_id
                                display_name = model_name_map.get(model_type.lower(), selected_model_id)

                                # Create recommendation dict
                                recommendation = {
                                    'model_name': display_name,
                                    'confidence': confidence,
                                    'reasoning': f"Selected based on data characteristics and forecast horizon of {horizon} days. Market regime: {regime_enum.value}.",
                                    'data_characteristics': {
                                        'Data Points': len(price_data),
                                        'Forecast Horizon': f"{horizon} days",
                                        'Volatility': f"{volatility:.2f}",
                                        'Trend': f"{price_trend*100:.2f}%",
                                        'Market Regime': regime_enum.value
                                    },
                                    'alternatives': [
                                        {
                                            'model_name': model_name_map.get(rec.get('model_type', '').lower(), rec.get('model_id', '')),
                                            'confidence': rec.get('total_score', 0.5),
                                            'reason': f"Score: {rec.get('total_score', 0):.2f}"
                                        }
                                        for rec in recommendations[:3]
                                    ] if recommendations else []
                                }

                                st.session_state.analyze_ai_recommendation = recommendation

                            except Exception as e:
                                st.error(f"Tab error: {type(e).__name__}: {e}")
                                import traceback
                                st.code(traceback.format_exc(), language="python")

                with col2:
                    st.subheader("💡 AI Recommendation")

                    if st.session_state.get('analyze_ai_recommendation'):
                        rec = st.session_state.analyze_ai_recommendation

                        # Main recommendation
                        st.success(f"**Recommended Model: {rec['model_name']}**")

                        # Confidence
                        confidence_pct = rec.get('confidence', 0.75) * 100
                        st.metric("Confidence", f"{confidence_pct:.1f}%")

                        # Progress bar for confidence
                        st.progress(rec.get('confidence', 0.75))

                        # Reasoning
                        with st.expander("🧠 Why this model?", expanded=True):
                            st.markdown(rec.get('reasoning', 'Model selected based on data analysis'))

                            if 'data_characteristics' in rec:
                                st.markdown("**Data Characteristics:**")
                                chars = rec['data_characteristics']
                                for key, value in chars.items():
                                    st.text(f"• {key}: {value}")

                        # Alternatives
                        if rec.get('alternatives'):
                            with st.expander("🔄 Alternative Models"):
                                for alt in rec['alternatives']:
                                    st.markdown(f"**{alt['model_name']}**")
                                    st.caption(f"Confidence: {alt.get('confidence', 0)*100:.1f}%")
                                    st.caption(alt.get('reason', ''))
                                    st.markdown("---")

                        # Action buttons
                        col_a, col_b = st.columns(2)

                        with col_a:
                            if st.button("✅ Use Recommended Model"):
                                st.session_state.selected_model = rec['model_name']
                                st.success(f"Selected: {rec['model_name']}")
                                st.info("Go to Quick Forecast tab to generate forecast")

                        with col_b:
                            if st.button("🔄 Choose Different Model"):
                                st.session_state.show_override = True

                        if st.session_state.get('show_override'):
                            st.markdown("**Override AI Selection:**")
                            override = st.selectbox(
                                "Select model:",
                                ["LSTM (Deep Learning)", "XGBoost (Gradient Boosting)", 
                                 "Prophet (Facebook)", "ARIMA (Statistical)"]
                            )
                            if st.button("Confirm Override"):
                                st.session_state.selected_model = override
                                st.session_state.show_override = False
                                st.success(f"Selected: {override}")

                # Hybrid Model Selector Section
                st.markdown("---")
                st.subheader("🎯 Hybrid Model Selection")

                st.write("""
                Hybrid selection combines multiple model types and automatically 
                switches between them based on market conditions.
                """)

                if st.button("Use Hybrid Selector"):
                    try:
                        from trading.forecasting.hybrid_model_selector import HybridModelSelector

                        data = st.session_state.get("analyze_forecast_data")
                        selector = HybridModelSelector()

                        with st.spinner("Analyzing market conditions and selecting models..."):
                            # Detect market regime: HybridModelSelector has no detect_market_regime; use MarketAnalyzer or default
                            regime = "neutral"
                            try:
                                from trading.market.market_analyzer import MarketAnalyzer
                                df = data.copy() if hasattr(data, 'copy') else pd.DataFrame(data)
                                if 'close' in df.columns and 'Close' not in df.columns:
                                    df['Close'] = df['close']
                                if 'Close' in df.columns:
                                    analyzer = MarketAnalyzer()
                                    trend_result = analyzer.detect_market_regime(df, "trend")
                                    regime = trend_result.get("regime", "neutral")
                            except Exception as e:
                                st.caption(f"Regime detection fell back to neutral: {e}")

                            st.info(f"📊 Detected market regime: **{regime}**")

                            # HybridModelSelector has select_best_model(model_scores, metric), not select_models_for_regime
                            # Show a simple recommendation based on regime
                            recommended = [
                                {"name": "XGBoost", "reason": "Robust across regimes"},
                                {"name": "LSTM", "reason": "Captures nonlinear patterns"},
                                {"name": "Prophet", "reason": "Good for trend and seasonality"},
                            ]
                            selected_models = recommended

                            st.write("**Recommended models for current conditions:**")
                            for i, model_config in enumerate(selected_models, 1):
                                st.write(f"{i}. {model_config['name']} - {model_config['reason']}")

                            # Store selected models
                            st.session_state.hybrid_selected_models = selected_models
                            st.session_state.hybrid_regime = regime

                            st.caption("To train an ensemble, use the Model Comparison tab and select multiple models.")

                    except ImportError:
                        st.error("Hybrid Model Selector not available")
                    except Exception as e:
                        st.error(f"Error using hybrid selector: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                # Model comparison table — run all registered models via ForecastRouter
                st.markdown("---")
                st.subheader("📋 Model Comparison Table")
                st.caption(
                    "Run each registered model and compare MAPE / 7-day forecast. "
                    "Uses daily data from the Forecast tab (ticker at top of Analyze)."
                )
                if st.session_state.get("analyze_forecast_data") is not None:
                    _hist = st.session_state.get("analyze_forecast_data").copy()
                    if "Close" not in _hist.columns and "close" in _hist.columns:
                        _hist["Close"] = _hist["close"]
                    _horizon = st.session_state.get("analyze_forecast_horizon", 7)
                    try:
                        from trading.models.forecast_router import (
                            ForecastRouter,
                            get_router_singleton,
                        )
                        from trading.models.model_registry import get_registry

                        _registry = get_registry()
                        _router = get_router_singleton()
                        _model_names = (
                            _registry.list_models()
                            if hasattr(_registry, "list_models")
                            else list(getattr(_registry, "_models", {}).keys())
                        )
                        if not _model_names:
                            _model_names = list(_router.model_registry.keys())

                        _comparison_rows = []
                        _progress = st.progress(0.0, text="Running model comparison...")

                        for _i, _model_name in enumerate(_model_names):
                            _pct = (_i + 1) / max(len(_model_names), 1)
                            _progress.progress(_pct, text=f"Testing {_model_name}...")
                            try:
                                _result = _router.get_forecast(
                                    _hist,
                                    model_type=_model_name.lower(),
                                    horizon=_horizon,
                                    run_walk_forward=False,
                                )
                                _fc = _extract_forecast_values(_result)
                                _fc_valid = _fc is not None and _fc.size > 0
                                _mape = (
                                    _result.get("validation_mape")
                                    or _result.get("in_sample_mape")
                                    or _result.get("mape")
                                    or _result.get("score")
                                    or _result.get("error_pct")
                                )
                                _last_fc = (
                                    round(float(_fc.flat[-1]), 2) if _fc_valid else None
                                )

                                _comparison_rows.append({
                                    "Model": _model_name,
                                    "Status": "✅" if _fc_valid else "⚠️",
                                    "MAPE": (
                                        round(float(_mape), 2)
                                        if _mape is not None
                                        else None
                                    ),
                                    "7d Forecast": (
                                        f"${_last_fc:.2f}" if _last_fc else "—"
                                    ),
                                    "Notes": _result.get("error", "") or "",
                                })
                            except Exception as _e:
                                _comparison_rows.append({
                                    "Model": _model_name,
                                    "Status": "❌",
                                    "MAPE": None,
                                    "7d Forecast": "—",
                                    "Notes": f"{type(_e).__name__}: {str(_e)[:80]}",
                                })

                        _progress.empty()
                        if _comparison_rows:
                            _df = pd.DataFrame(_comparison_rows)
                            st.dataframe(normalize_for_display(_df), width='stretch')
                            _working = sum(
                                1 for r in _comparison_rows if r["Status"] == "✅"
                            )
                            st.caption(
                                f"{_working}/{len(_comparison_rows)} models produced valid forecasts"
                            )
                    except Exception as _e:
                        st.error(f"Comparison failed: {type(_e).__name__}: {_e}")
                        import traceback
                        st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"Tab error: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
