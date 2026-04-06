# -*- coding: utf-8 -*-
"""Analyze page tab body (extracted; logic unchanged)."""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components
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
        st.header("Advanced Forecasting")
        st.markdown("Full model configuration with hyperparameter tuning and feature engineering")

        if st.session_state.get("analyze_forecast_data") is None:
            st.warning("⚠️ Please load data first in the Quick Forecast tab")
        else:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("⚙️ Configuration")

                # Model selection using registry
                try:
                    from trading.models.model_registry import get_registry

                    registry = get_registry()

                    # Get all single-asset models (includes advanced ones, excludes GNN)
                    available_models = registry.get_advanced_models()

                    if not available_models:
                        st.warning("⚠️ No models available. Using default models.")
                        available_models = ["LSTM", "XGBoost", "Prophet", "ARIMA"]

                    # Create friendly names
                    model_display_names = {
                        'LSTM': 'LSTM (Deep Learning)',
                        'XGBoost': 'XGBoost (Gradient Boosting)',
                        'Prophet': 'Prophet (Seasonality)',
                        'ARIMA': 'ARIMA (Statistical)',
                        'Ensemble': 'Ensemble (Multi-Model)',
                        'TCN': 'TCN (Temporal Conv)',
                        'GARCH': 'GARCH (Volatility)',
                        'Autoformer': 'Autoformer (Advanced Transformer)',
                        'CatBoost': 'CatBoost (Categorical Boosting)',
                        'Ridge': 'Ridge (Linear Baseline)'
                    }

                    display_names = [model_display_names.get(m, m) for m in available_models if m in model_display_names or m not in model_display_names]

                    selected_display = st.selectbox(
                        "Select Model Type",
                        display_names if display_names else available_models,
                        help="Advanced forecasting with all available models"
                    )

                    # Convert back to model name
                    if display_names:
                        model_type = [k for k, v in model_display_names.items() if v == selected_display]
                        model_type = model_type[0] if model_type else selected_display
                    else:
                        model_type = selected_display

                    # Show model info
                    model_info = registry.get_model_info(model_type)
                    if model_info:
                        st.info(f"ℹ️ {model_info.get('description', 'No description')}")

                        # Show requirements
                        reqs = []
                        if model_info.get('requires_gpu'):
                            reqs.append("🖥️ GPU recommended")
                        if model_info.get('min_data_points'):
                            reqs.append(f"📊 Min {model_info['min_data_points']} data points")

                        if reqs:
                            st.caption(" • ".join(reqs))

                    # Auto-select best model (Advanced only; runs all models, slow)
                    if st.button("🔬 Auto-select best model", key="auto_select_best_advanced"):
                        data_for_best = st.session_state.get("analyze_forecast_data")
                        if data_for_best is not None and len(data_for_best) >= 60:
                            try:
                                from trading.models.forecast_router import (
                                    ForecastRouter,
                                    get_router_singleton,
                                )
                                _router = get_router_singleton()
                                _horizon = st.session_state.get("analyze_forecast_horizon", 7)
                                _symbol = st.session_state.get("analyze_symbol", "")
                                _df = data_for_best.copy()
                                if "Close" in _df.columns and "close" not in _df.columns:
                                    _df = _df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
                                best_key, best_result = _router.select_best_model(_df, _horizon, symbol=_symbol or None)
                                display_map = {"lstm": "LSTM", "xgboost": "XGBoost", "prophet": "Prophet", "arima": "ARIMA", "ridge": "Ridge", "garch": "GARCH", "tcn": "TCN", "catboost": "CatBoost", "ensemble": "Ensemble", "transformer": "Transformer", "autoformer": "Autoformer", "hybrid": "Hybrid"}
                                best_display = display_map.get(best_key, best_key.capitalize())
                                st.session_state["advanced_forecast_best_model"] = best_display
                                st.session_state["advanced_best_model_scores"] = best_result
                                st.success(f"Best model: **{best_display}** (validation MAPE: {best_result.get('validation_mape') or 'N/A'}%)")
                            except Exception as e:
                                st.warning(f"Auto-select failed: {e}")
                        else:
                            st.warning("Load at least 60 rows of data first in Quick Forecast tab.")
                    if st.session_state.get("advanced_best_model_scores"):
                        with st.expander("📊 Model validation scores (walk-forward)", expanded=False):
                            scores = st.session_state["advanced_best_model_scores"].get("scores", [])
                            if scores is not None and len(scores) > 0:
                                tab_df = pd.DataFrame(scores)
                                st.dataframe(normalize_for_display(tab_df), width='stretch')
                except Exception as e:
                    st.warning(f"⚠️ Could not load model registry: {e}. Using default models.")
                    model_type = st.selectbox(
                        "Model Type",
                        ["LSTM", "XGBoost", "Prophet", "ARIMA", "Ridge", "TCN", "GARCH", "CatBoost", "Ensemble"]
                    )
                    model_info = {}

                st.markdown("---")
                st.markdown(f"**{model_type} Parameters**")

                # Model-specific hyperparameters
                params = {}

                if model_type == "LSTM":
                    params['hidden_dim'] = st.slider("Hidden Dimension", 16, 256, 64, 16)
                    params['num_layers'] = st.slider("Number of Layers", 1, 5, 2)
                    params['dropout'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
                    params['learning_rate'] = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
                    params['sequence_length'] = st.slider("Sequence Length", 30, 120, 60, 10)

                elif model_type == "XGBoost":
                    params['n_estimators'] = st.slider("Number of Trees", 50, 500, 100, 50)
                    params['max_depth'] = st.slider("Max Depth", 3, 15, 5, 1)
                    params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                    params['subsample'] = st.slider("Subsample Ratio", 0.5, 1.0, 0.8, 0.1)
                    params['colsample_bytree'] = st.slider("Feature Sampling", 0.5, 1.0, 0.8, 0.1)

                elif model_type == "Prophet":
                    params['changepoint_prior_scale'] = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001)
                    params['seasonality_prior_scale'] = st.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0, 0.1)
                    params['holidays_prior_scale'] = st.slider("Holidays Prior Scale", 0.01, 10.0, 10.0, 0.1)
                    params['seasonality_mode'] = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'])

                elif model_type == "ARIMA":
                    p = st.slider("AR Order (p)", 0, 10, 5, 1)
                    d = st.slider("Differencing (d)", 0, 2, 1, 1)
                    q = st.slider("MA Order (q)", 0, 10, 0, 1)
                    params['order'] = (p, d, q)
                    params['use_auto_arima'] = st.checkbox("Use Auto ARIMA", value=True)

                elif model_type == "GARCH":
                    st.markdown("**GARCH Parameters:**")
                    params['p'] = st.slider("p (AR order)", 1, 5, 1, 1)
                    params['q'] = st.slider("q (MA order)", 1, 5, 1, 1)

                elif model_type == "Autoformer":
                    st.markdown("**Autoformer Parameters:**")
                    params['seq_len'] = st.slider("Sequence Length", 30, 120, 60, 10)
                    params['pred_len'] = st.session_state.analyze_forecast_horizon
                    params['d_model'] = st.slider("Model Dimension", 64, 512, 128, 32)

                elif model_type == "CatBoost":
                    st.markdown("**CatBoost Parameters:**")
                    params['iterations'] = st.slider("Iterations", 100, 1000, 500, 50)
                    params['depth'] = st.slider("Tree Depth", 3, 10, 6, 1)
                    params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.1, 0.03, 0.01)

                elif model_type == "TCN":
                    st.markdown("**TCN Parameters:**")
                    params['num_channels'] = [64, 128, 256]  # Can be made configurable
                    params['kernel_size'] = st.slider("Kernel Size", 2, 5, 3, 1)
                    params['dropout'] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)

                elif model_type == "Ridge":
                    st.markdown("**Ridge Parameters:**")
                    params['alpha'] = st.slider("Regularization (alpha)", 0.01, 10.0, 1.0, 0.1)

                st.markdown("---")
                st.subheader("🔧 Feature Engineering")

                use_technical = st.checkbox("Add Technical Indicators", value=False)
                indicators = []
                if use_technical:
                    indicators = st.multiselect(
                        "Select Indicators",
                        ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "ATR"],
                        default=["SMA", "RSI"]
                    )

                use_lags = st.checkbox("Add Lag Features", value=False)
                lag_periods = []
                if use_lags:
                    lag_periods = st.multiselect(
                        "Lag Periods",
                        [1, 2, 3, 5, 7, 14, 21, 30],
                        default=[1, 7, 14]
                    )

                use_macro_features = st.checkbox(
                    "Include macro-economic features",
                    value=False,
                    help="Add features like interest rates, inflation, GDP growth"
                )

                macro_features_list = []
                if use_macro_features:
                    macro_features_list = st.multiselect(
                        "Select Macro Features",
                        ["interest_rate", "inflation", "gdp_growth", "unemployment"],
                        default=["interest_rate", "inflation"],
                        help="Choose which macroeconomic indicators to include"
                    )

                normalize = st.checkbox("Normalize Data", value=True)

                train_button = st.button("🚀 Train Model", type="primary")

            with col2:
                st.subheader("📊 Results")

                if train_button:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Prepare data
                        data = st.session_state.get("analyze_forecast_data")
                        if data is None:
                            raise RuntimeError("No analyze_forecast_data in session_state; please load data first.")
                        data = data.copy()
                        # Normalize column case (yfinance uses Close, etc.)
                        if "close" in data.columns and "Close" not in data.columns:
                            data["Close"] = data["close"]
                        if "Close" not in data.columns:
                            data["Close"] = data.iloc[:, 0]
                        # Ensure proper column names (capitalize for FeatureEngineering)
                        if "close" in data.columns:
                            data["Close"] = data["close"]
                        if "open" in data.columns:
                            data['Open'] = data['open']
                        if 'high' in data.columns:
                            data['High'] = data['high']
                        if 'low' in data.columns:
                            data['Low'] = data['low']
                        if 'volume' in data.columns:
                            data['Volume'] = data['volume']

                        # Ensure we have required columns
                        if 'Open' not in data.columns:
                            data['Open'] = data['Close']
                        if 'High' not in data.columns:
                            data['High'] = data['Close']
                        if 'Low' not in data.columns:
                            data['Low'] = data['Close']
                        if 'Volume' not in data.columns:
                            data['Volume'] = 1000000

                        # Feature engineering
                        if use_technical:
                            status_text.text("Adding technical indicators...")
                            progress_bar.progress(0.2)
                            fe = FeatureEngineering()

                            if "SMA" in indicators or "EMA" in indicators:
                                ma_features = fe.calculate_moving_averages(data)
                                data = pd.concat([data, ma_features], axis=1)

                            if "RSI" in indicators:
                                rsi_features = fe.calculate_rsi(data)
                                data = pd.concat([data, rsi_features], axis=1)

                            if "MACD" in indicators:
                                macd_features = fe.calculate_macd(data)
                                data = pd.concat([data, macd_features], axis=1)

                            if "Bollinger Bands" in indicators:
                                bb_features = fe.calculate_bollinger_bands(data)
                                data = pd.concat([data, bb_features], axis=1)

                        if use_lags:
                            status_text.text("Adding lag features...")
                            progress_bar.progress(0.4)
                            for lag in lag_periods:
                                data[f'lag_{lag}'] = data['Close'].shift(lag)

                        # Macro feature engineering
                        if use_macro_features and macro_features_list:
                            status_text.text("Adding macro-economic features...")
                            progress_bar.progress(0.5)
                            try:
                                from trading.feature_engineering.macro_feature_engineering import MacroFeatureEngineer

                                engineer = MacroFeatureEngineer()

                                # Add macro features to data (enrich_trading_data adds FRED/World Bank features)
                                data_with_macro = engineer.enrich_trading_data(
                                    data,
                                    include_sentiment=('sentiment' in (macro_features_list or []))
                                )

                                # Show what was added
                                new_cols = set(data_with_macro.columns) - set(data.columns)
                                if new_cols:
                                    st.write("**Macro features added:**")
                                    for col in new_cols:
                                        st.write(f"• {col}")

                                # Use data_with_macro for training
                                data = data_with_macro
                            except ImportError:
                                st.warning("Macro Feature Engineer not available. Skipping macro features.")
                            except Exception as e:
                                st.warning(f"Could not add macro features: {e}. Continuing without them.")

                        # Remove NaN
                        data = data.dropna()

                        if len(data) < 30:
                            st.error("Not enough data after feature engineering. Try fewer features or more historical data.")
                            progress_bar.empty()
                            status_text.empty()
                        else:
                            # Normalize
                            if normalize:
                                status_text.text("Normalizing data...")
                                progress_bar.progress(0.6)
                                preprocessor = DataPreprocessor()
                                # Normalize only numeric columns
                                numeric_cols = data.select_dtypes(include=[np.number]).columns
                                for col in numeric_cols:
                                    if col != 'Close':  # Don't normalize target yet
                                        data[col] = (data[col] - data[col].mean()) / (data[col].std() + 1e-8)

                            # Train model
                            status_text.text(f"Training {model_type}...")
                            progress_bar.progress(0.8)

                            # Create model using registry
                            try:
                                from trading.models.model_registry import get_registry
                                registry = get_registry()
                                ModelClass = registry.get(model_type)

                                if ModelClass is None:
                                    st.error(f"Model {model_type} not available")
                                    model = None
                                else:
                                    # Create model config
                                    model_config = {
                                        "target_column": "close" if "close" in data.columns else "Close",
                                        "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"]
                                    }

                                    # Add model-specific parameters
                                    if model_type == "LSTM":
                                        model_config.update({
                                            "sequence_length": params.get('sequence_length', 60),
                                            "hidden_size": params.get('hidden_dim', 64),
                                            "num_layers": params.get('num_layers', 2),
                                            "dropout": params.get('dropout', 0.2),
                                            "learning_rate": params.get('learning_rate', 0.001),
                                            "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                        })
                                    elif model_type == "XGBoost":
                                        model_config.update({
                                            "n_estimators": params.get('n_estimators', 100),
                                            "max_depth": params.get('max_depth', 5),
                                            "learning_rate": params.get('learning_rate', 0.1),
                                            "subsample": params.get('subsample', 0.8),
                                            "colsample_bytree": params.get('colsample_bytree', 0.8)
                                        })
                                    elif model_type == "Prophet":
                                        model_config.update({
                                            "date_column": "ds",
                                            "prophet_params": {
                                                "changepoint_prior_scale": params.get('changepoint_prior_scale', 0.05),
                                                "seasonality_prior_scale": params.get('seasonality_prior_scale', 10.0),
                                                "holidays_prior_scale": params.get('holidays_prior_scale', 10.0),
                                                "seasonality_mode": params.get('seasonality_mode', 'additive')
                                            }
                                        })
                                    elif model_type == "ARIMA":
                                        model_config.update({
                                            "order": params.get('order', (5, 1, 0)),
                                            "use_auto_arima": params.get('use_auto_arima', True)
                                        })
                                    elif model_type == "GARCH":
                                        model_config.update({
                                            "p": params.get('p', 1),
                                            "q": params.get('q', 1)
                                        })
                                    elif model_type == "Autoformer":
                                        model_config.update({
                                            "seq_len": params.get('seq_len', 60),
                                            "pred_len": params.get('pred_len', st.session_state.analyze_forecast_horizon),
                                            "d_model": params.get('d_model', 128)
                                        })
                                    elif model_type == "CatBoost":
                                        model_config.update({
                                            "iterations": params.get('iterations', 500),
                                            "depth": params.get('depth', 6),
                                            "learning_rate": params.get('learning_rate', 0.03)
                                        })
                                    elif model_type == "TCN":
                                        model_config.update({
                                            "num_channels": params.get('num_channels', [64, 128, 256]),
                                            "kernel_size": params.get('kernel_size', 3),
                                            "dropout": params.get('dropout', 0.2)
                                        })
                                    elif model_type == "Ridge":
                                        model_config.update({
                                            "alpha": params.get('alpha', 1.0)
                                        })

                                    model = ModelClass(model_config)
                            except Exception as e:
                                st.warning(f"Could not use model registry: {e}. Using direct imports.")
                                # Fallback to direct imports
                                model_config = {
                                    "target_column": "close" if "close" in data.columns else "Close"
                                }

                                if model_type == "LSTM":
                                    model_config.update({
                                        "sequence_length": params.get('sequence_length', 60),
                                        "hidden_size": params.get('hidden_dim', 64),
                                        "num_layers": params.get('num_layers', 2),
                                        "dropout": params.get('dropout', 0.2),
                                        "learning_rate": params.get('learning_rate', 0.001),
                                        "feature_columns": list(data.columns) if len(data.columns) > 0 else ["close"],
                                        "input_size": len(data.columns) if len(data.columns) > 0 else 1
                                    })
                                    model = LSTMForecaster(model_config)
                                elif model_type == "XGBoost":
                                    model_config.update({
                                        "n_estimators": params.get('n_estimators', 100),
                                        "max_depth": params.get('max_depth', 5),
                                        "learning_rate": params.get('learning_rate', 0.1),
                                        "subsample": params.get('subsample', 0.8),
                                        "colsample_bytree": params.get('colsample_bytree', 0.8)
                                    })
                                    model = XGBoostModel(model_config)
                                elif model_type == "Prophet":
                                    model_config.update({
                                        "date_column": "ds",
                                        "prophet_params": {
                                            "changepoint_prior_scale": params.get('changepoint_prior_scale', 0.05),
                                            "seasonality_prior_scale": params.get('seasonality_prior_scale', 10.0),
                                            "holidays_prior_scale": params.get('holidays_prior_scale', 10.0),
                                            "seasonality_mode": params.get('seasonality_mode', 'additive')
                                        }
                                    })
                                    model = ProphetModel(model_config)
                                elif model_type == "ARIMA":
                                    model_config.update({
                                        "order": params.get('order', (5, 1, 0)),
                                        "use_auto_arima": params.get('use_auto_arima', True)
                                    })
                                    model = ARIMAModel(model_config)
                                else:
                                    st.error(f"Model {model_type} not available with direct imports. Please use model registry.")
                                    model = None

                            # Train model
                            if model_type == "Prophet":
                                train_df = pd.DataFrame({
                                    'ds': data.index,
                                    'y': data[model_config["target_column"]]
                                })
                                fit_result = model.fit(train_df)
                            elif model_type == "ARIMA":
                                fit_result = model.fit(data[model_config["target_column"]])
                            else:
                                # Most regression models accept (data, target); some ignore target
                                fit_result = model.fit(data, data[model_config["target_column"]])

                            # Validate that model is actually fitted before forecasting
                            fit_success = True
                            if isinstance(fit_result, dict) and "success" in fit_result:
                                fit_success = bool(fit_result.get("success", True))
                            is_fitted_flag = True
                            if hasattr(model, "is_fitted"):
                                is_fitted_flag = bool(getattr(model, "is_fitted"))
                            elif hasattr(model, "model"):
                                is_fitted_flag = getattr(model, "model") is not None

                            if not (fit_success and is_fitted_flag):
                                detail = ""
                                if isinstance(fit_result, dict):
                                    detail = fit_result.get("error") or str(fit_result)
                                st.error(f"{model_type} training did not complete successfully. Details: {detail}")
                                return

                            # Generate forecast - try with uncertainty if available; support both forecast() and predict() (e.g. HybridModel)
                            progress_bar.progress(0.9)
                            horizon = st.session_state.analyze_forecast_horizon
                            if hasattr(model, 'forecast_with_uncertainty'):
                                try:
                                    forecast_result = model.forecast_with_uncertainty(data, horizon=horizon, num_samples=100)
                                except Exception as e:
                                    st.warning(f"Could not generate forecast with uncertainty: {e}. Using standard forecast.")
                                    if hasattr(model, 'forecast'):
                                        forecast_result = model.forecast(data, horizon=horizon)
                                    else:
                                        preds = model.predict(data) if hasattr(model, 'predict') else np.array([])
                                        forecast_result = {"forecast": preds[-horizon:].tolist() if len(preds) >= horizon else preds.tolist(), "horizon": horizon}
                            elif hasattr(model, 'forecast'):
                                forecast_result = model.forecast(data, horizon=horizon)
                            elif hasattr(model, 'predict'):
                                preds = np.asarray(model.predict(data))
                                preds = preds.astype("float64").ravel()
                                if preds.size == 0:
                                    forecast_result = {"forecast": [], "horizon": horizon}
                                else:
                                    tail = preds[-horizon:] if preds.size >= horizon else preds
                                    forecast_result = {"forecast": tail.tolist(), "horizon": horizon}
                            else:
                                forecast_result = {"forecast": [], "horizon": horizon}

                            # Store full forecast result for confidence intervals
                            st.session_state.current_forecast_result = forecast_result
                            st.session_state.current_model_instance = model  # trained instance for explainability

                            # Postprocess forecast
                            try:
                                from trading.forecasting.forecast_postprocessor import ForecastPostprocessor

                                postprocessor = ForecastPostprocessor()

                                # Extract forecast values (handles all result formats)
                                forecast_vals = _extract_forecast_values(forecast_result)
                                if forecast_vals is not None:
                                    forecast_vals = forecast_vals.tolist() if hasattr(forecast_vals, 'tolist') else list(forecast_vals)
                                else:
                                    forecast_vals = []

                                # Postprocess forecast
                                processed_forecast = postprocessor.process(
                                    forecast=forecast_vals,
                                    historical_data=data,
                                    apply_smoothing=True,
                                    remove_outliers=True,
                                    ensure_realistic_bounds=True
                                )

                                # Update forecast_result with processed version
                                if isinstance(forecast_result, dict):
                                    forecast_result['forecast'] = processed_forecast['values']
                                    forecast_result['postprocessing_notes'] = processed_forecast.get('notes', [])
                                else:
                                    forecast_result = processed_forecast['values']

                                # Update session state
                                st.session_state.current_forecast_result = forecast_result
                            except ImportError:
                                pass  # Silently fail if postprocessor not available
                            except Exception as e:
                                logger.warning(f"Forecast postprocessing failed: {e}")

                            progress_bar.progress(1.0)
                            status_text.text("Complete!")

                            st.success("✅ Model trained successfully!")

                            # Extract and validate forecast values (robust for all result formats)
                            forecast_values = _extract_forecast_values(forecast_result)
                            if forecast_values is not None:
                                forecast_values = np.asarray(forecast_values).ravel()
                            if isinstance(forecast_result, dict):
                                forecast_dates = forecast_result.get('dates', pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=st.session_state.analyze_forecast_horizon,
                                    freq='D'
                                ))
                            else:
                                forecast_dates = pd.date_range(
                                    start=data.index[-1] + timedelta(days=1),
                                    periods=st.session_state.analyze_forecast_horizon,
                                    freq='D'
                                )

                            if isinstance(forecast_values, (list, np.ndarray)):
                                forecast_array = np.array(
                                    [v for v in forecast_values if v is not None],
                                    dtype="float64"
                                ).flatten()
                            elif forecast_values is None:
                                forecast_array = np.array([], dtype="float64")
                            else:
                                forecast_array = np.array([float(forecast_values)], dtype="float64")

                            if forecast_array.size == 0:
                                st.error(f"{model_type} returned an empty forecast. Please check the model configuration.")
                                return

                            # Sanity-check price range relative to last known price
                            # Allow forecasts in the range [0.4 * last_price, 2.0 * last_price]
                            last_price_series = None
                            if "close" in data.columns:
                                last_price_series = data["close"]
                            elif "Close" in data.columns:
                                last_price_series = data["Close"]

                            if last_price_series is not None and not last_price_series.empty:
                                last_price = float(last_price_series.iloc[-1])
                                lower_bound = 0.4 * last_price
                                upper_bound = 2.0 * last_price
                                if not np.all((forecast_array >= lower_bound) & (forecast_array <= upper_bound)):
                                    st.error(
                                        f"{model_type} produced forecasts outside the expected range "
                                        f"[{lower_bound:.2f}, {upper_bound:.2f}] based on last close "
                                        f"({last_price:.2f}). Please review the model configuration and data scaling."
                                    )
                                    return

                            # Store forecast
                            forecast_df = pd.DataFrame(
                                {'forecast': forecast_array},
                                index=forecast_dates[: len(forecast_array)],
                            )

                            st.session_state.advanced_forecast = forecast_df
                            st.session_state.advanced_model = model_type

                            # Display forecast chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data[model_config["target_column"]],
                                mode='lines',
                                name='Historical',
                                line=dict(color='blue', width=2)
                            ))
                            fig.add_trace(go.Scatter(
                                x=forecast_df.index,
                                y=forecast_df['forecast'],
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=8)
                            ))
                            fig.update_layout(
                                title=f"{st.session_state.analyze_symbol} - Advanced Forecast ({model_type})",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                hovermode='x unified',
                                height=500
                            )
                            st.plotly_chart(fig)

                            # Display forecast table
                            st.markdown("**Forecast Values:**")
                            display_df = forecast_df.copy()
                            try:
                                _idx = pd.to_datetime(display_df.index)
                                if hasattr(_idx, "tz") and _idx.tz is not None:
                                    _idx = _idx.tz_localize(None)
                                display_df.index = _idx.strftime("%Y-%m-%d")
                            except Exception:
                                pass
                            display_df["forecast"] = display_df["forecast"].apply(
                                lambda x: f"${x:.2f}"
                                if isinstance(x, (int, float)) and not pd.isna(x)
                                else "—"
                            )
                            display_df.rename(columns={"forecast": "Forecast"}, inplace=True)

                            # Add bounds if available
                            _lb = None
                            _ub = None
                            if isinstance(forecast_result, dict):
                                _lb = forecast_result.get("lower_bound")
                                _ub = forecast_result.get("upper_bound")
                            _n = len(display_df)
                            display_df["Lower Bound"] = (
                                [f"${v:.2f}" for v in _lb[:_n]]
                                if _lb and len(_lb) >= _n
                                else ["—"] * _n
                            )
                            display_df["Upper Bound"] = (
                                [f"${v:.2f}" for v in _ub[:_n]]
                                if _ub and len(_ub) >= _n
                                else ["—"] * _n
                            )

                            st.dataframe(normalize_for_display(display_df), width="stretch")

                            # Add explainability section
                            st.markdown("---")
                            st.subheader("🔍 Model Explainability")

                            with st.expander("📊 View Feature Importance & Explanations", expanded=False):
                                st.write("Understand what drives the model's predictions")

                                # Display cached explanation (persist across reruns)
                                if st.session_state.get('forecast_explanation_tab2') is not None:
                                    cached = st.session_state.forecast_explanation_tab2
                                    if cached.get('success') and cached.get('explanation'):
                                        expl_obj = cached['explanation']
                                        if hasattr(expl_obj, 'feature_importance') and expl_obj.feature_importance:
                                            st.write("**📊 Feature Importance:**")
                                            importance_data = expl_obj.feature_importance
                                            if isinstance(importance_data, dict):
                                                importance_df = pd.DataFrame(
                                                    importance_data.items(),
                                                    columns=['Feature', 'Importance']
                                                ).sort_values('Importance', ascending=False)
                                                fig_imp = px.bar(
                                                    importance_df, x='Importance', y='Feature',
                                                    orientation='h', title='What Drives the Predictions?',
                                                    labels={'Importance': 'Impact on Prediction'}
                                                )
                                                st.plotly_chart(fig_imp)
                                        if hasattr(expl_obj, 'explanation_text') and expl_obj.explanation_text:
                                            st.write("**💬 Plain English Explanation:**")
                                            st.info(expl_obj.explanation_text)
                                    if st.button("Clear explanation", key="clear_explanation_tab2"):
                                        st.session_state.forecast_explanation_tab2 = None

                                if st.button("Generate Explanation", key="explain_btn_tab2"):
                                    try:
                                        with st.spinner("Analyzing model predictions..."):
                                            from trading.models.forecast_explainability import ForecastExplainability
                                            explainer = ForecastExplainability()
                                            fv = _extract_forecast_values(forecast_result)
                                            forecast_value = float(np.asarray(fv).flat[0]) if fv is not None and np.asarray(fv).size > 0 else float(data[model_config["target_column"]].iloc[-1])
                                            features = data.copy()
                                            target_history = features[model_config["target_column"]] if model_config["target_column"] in features.columns else features.iloc[:, 0]
                                            try:
                                                explanation = explainer.explain_forecast(
                                                    forecast_id=f"forecast_{st.session_state.analyze_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                                    symbol=st.session_state.analyze_symbol,
                                                    forecast_value=forecast_value,
                                                    model=model,
                                                    features=features,
                                                    target_history=target_history,
                                                    horizon=st.session_state.analyze_forecast_horizon
                                                )
                                                # Persist explanation in session state so it survives reruns
                                                st.session_state.forecast_explanation_tab2 = {
                                                    "success": True,
                                                    "explanation": explanation,
                                                }
                                                st.success("✅ Explanation generated successfully!")
                                            except Exception:
                                                st.caption("Model explainability unavailable for this model type.")
                                    except ImportError:
                                        st.warning("⚠️ Forecast explainability requires SHAP library")
                                        st.code("pip install shap")
                                    except Exception as e:
                                        st.error(f"Error generating explanation: {e}")

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        progress_bar.empty()
                        status_text.empty()

                # Display previous forecast if exists
                if st.session_state.get('advanced_forecast') is not None:
                    st.markdown("---")
                    st.markdown(f"**Previous Forecast ({st.session_state.get('advanced_model', 'Unknown')})**")
                    prev_forecast = st.session_state.advanced_forecast
                    st.dataframe(normalize_for_display(prev_forecast.tail(10)))
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
