# -*- coding: utf-8 -*-
"""Multi-Asset (GNN) tab (legacy `with tab6:`)."""
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


def _bound_col(arr, n: int):
    """Format lower/upper bound column; safe for numpy arrays (no truthiness on ndarray)."""
    try:
        if arr is None:
            return ["—"] * n
        _a = np.asarray(arr, dtype=float)
        if _a.size >= n:
            return [f"${float(v):.2f}" for v in _a[:n]]
    except Exception:
        pass
    return ["—"] * n


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
            st.header("🔗 Multi-Asset Forecasting with Graph Neural Networks")
            st.markdown("""
            GNN models relationships between correlated assets. Perfect for:
            - **Portfolio-level forecasting** - Predict entire portfolio together
            - **Sector analysis** - Model how sector stocks move together
            - **Market contagion** - Understand how shocks spread
            - **Relationship-based predictions** - Use asset correlations for better forecasts
            """)

            st.info("💡 GNN requires 3-20 correlated assets. It won't work with single tickers.")

            # Multi-ticker input
            st.subheader("📊 Step 1: Select Multiple Assets")

            col1, col2 = st.columns([2, 1])

            with col1:
                tickers_input = st.text_area(
                    "Enter ticker symbols (one per line)",
                    value="AAPL\nMSFT\nGOOGL\nAMZN\nMETA",
                    height=150,
                    help="Enter 3-20 correlated stocks. Tech stocks, bank stocks, etc."
                )

                tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.strip()]

                if len(tickers) < 3:
                    st.error("❌ GNN requires at least 3 assets")
                elif len(tickers) > 20:
                    st.warning("⚠️ Too many assets (max 20). Performance may be slow.")
                else:
                    st.success(f"✅ {len(tickers)} assets selected")

            with col2:
                st.write("**Settings:**")

                correlation_threshold = st.slider(
                    "Correlation Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Assets with correlation > this will be connected in graph"
                )

                gnn_horizon = st.slider(
                    "Forecast Days",
                    min_value=1,
                    max_value=30,
                    value=7
                )

                gnn_epochs = st.slider(
                    "Training Epochs",
                    min_value=20,
                    max_value=100,
                    value=50,
                    help="More epochs = better accuracy but slower"
                )

            # Load multi-asset data
            if st.button("📥 Load Multi-Asset Data", type="primary", width='stretch'):
                if len(tickers) < 3:
                    st.error("Please select at least 3 assets")
                elif len(tickers) > 20:
                    st.error("Please limit to 20 assets for performance")
                else:
                    try:
                        with st.spinner(f"Loading data for {len(tickers)} assets..."):
                            from trading.data.data_loader import DataLoader, DataLoadRequest
                            from datetime import datetime, timedelta

                            loader = DataLoader()

                            # Load data for each ticker
                            multi_asset_data = {}
                            failed_tickers = []

                            progress_bar = st.progress(0)

                            for i, ticker in enumerate(tickers):
                                try:
                                    request = DataLoadRequest(
                                        ticker=ticker,
                                        start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                                        end_date=datetime.now().strftime("%Y-%m-%d"),
                                        interval="1d"
                                    )
                                    response = loader.load_market_data(request)

                                    if response.success and response.data is not None:
                                        # Get close prices
                                        data = response.data
                                        if 'close' in data.columns:
                                            multi_asset_data[ticker] = data['close']
                                        elif 'Close' in data.columns:
                                            multi_asset_data[ticker] = data['Close']
                                    else:
                                        failed_tickers.append(ticker)
                                except Exception as e:
                                    logger.error(f"Error loading {ticker}: {e}")
                                    failed_tickers.append(ticker)

                                progress_bar.progress((i + 1) / len(tickers))

                            progress_bar.empty()

                            if len(multi_asset_data) < 3:
                                st.error(f"Could not load enough assets. Failed: {', '.join(failed_tickers)}")
                            else:
                                # Combine into DataFrame
                                multi_df = pd.DataFrame(multi_asset_data)

                                # Align dates (drop rows with any NaN)
                                multi_df = multi_df.dropna()

                                if len(multi_df) < 100:
                                    st.error("Insufficient overlapping data. Try different tickers or longer date range.")
                                else:
                                    st.session_state.gnn_data = multi_df
                                    st.session_state.gnn_tickers = list(multi_df.columns)

                                    st.success(f"✅ Loaded {len(multi_df)} days of data for {len(multi_df.columns)} assets")

                                    if failed_tickers:
                                        st.warning(f"⚠️ Could not load: {', '.join(failed_tickers)}")

                    except Exception as e:
                        st.error(f"Error loading data: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Display correlation matrix and train GNN
            if 'gnn_data' in st.session_state and st.session_state.gnn_data is not None:
                st.markdown("---")
                st.subheader("📊 Step 2: View Asset Correlations")

                multi_df = st.session_state.gnn_data

                # Calculate correlation matrix
                corr_matrix = multi_df.corr()

                # Plot heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title=f"Asset Correlation Matrix ({len(multi_df.columns)} assets)"
                )

                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr)

                # Show graph connections
                num_connections = (corr_matrix.abs() > correlation_threshold).sum().sum() - len(multi_df.columns)
                st.info(f"🔗 Graph will have {num_connections // 2} edges (connections) based on {correlation_threshold:.0%} threshold")

                # Generate forecast
                st.markdown("---")
                st.subheader("🚀 Step 3: Generate GNN Forecast")

                target_asset = st.selectbox(
                    "Select primary asset to forecast",
                    options=st.session_state.gnn_tickers,
                    help="GNN will forecast this asset using all connected assets"
                )

                if st.button("🔮 Train GNN & Generate Forecast", type="primary", width='stretch'):
                    try:
                        # Type guards: ensure scalars (sliders can sometimes be dict in edge cases)
                        _gnn_epochs = gnn_epochs
                        _gnn_horizon = gnn_horizon
                        _correlation_threshold = correlation_threshold
                        if isinstance(_gnn_epochs, dict):
                            _gnn_epochs = int(_gnn_epochs.get("value", _gnn_epochs.get("epochs", 50)))
                        else:
                            _gnn_epochs = int(_gnn_epochs)
                        if isinstance(_gnn_horizon, dict):
                            _gnn_horizon = int(_gnn_horizon.get("value", _gnn_horizon.get("horizon", 7)))
                        else:
                            _gnn_horizon = int(_gnn_horizon)
                        if isinstance(_correlation_threshold, dict):
                            _correlation_threshold = float(_correlation_threshold.get("value", _correlation_threshold.get("threshold", 0.5)))
                        else:
                            _correlation_threshold = float(_correlation_threshold)
                        with st.spinner(f"Training Graph Neural Network on {len(multi_df.columns)} assets..."):
                            from trading.models.advanced.gnn.gnn_model import GNNForecaster

                            # Progress indicator
                            progress_text = st.empty()

                            # Initialize GNN
                            progress_text.text("Initializing GNN model...")
                            gnn = GNNForecaster(
                                num_assets=len(multi_df.columns),
                                hidden_size=64,
                                num_layers=2,
                                seq_length=30,
                                correlation_threshold=_correlation_threshold
                            )

                            # Train
                            progress_text.text(f"Training for {_gnn_epochs} epochs...")
                            gnn.fit(multi_df, epochs=_gnn_epochs, batch_size=32)

                            # Generate forecast
                            progress_text.text("Generating forecast...")
                            forecast_result = gnn.forecast(
                                multi_df,
                                horizon=_gnn_horizon,
                                target_asset=target_asset
                            )

                            progress_text.empty()
                            st.success("✅ GNN forecast generated!")

                            # Display forecast chart
                            st.subheader(f"📈 {target_asset} Forecast")

                            fig = go.Figure()

                            # Historical data
                            fig.add_trace(go.Scatter(
                                x=multi_df.index,
                                y=multi_df[target_asset],
                                name='Historical',
                                line=dict(color='blue', width=2)
                            ))

                            # Forecast
                            fig.add_trace(go.Scatter(
                                x=forecast_result['dates'],
                                y=forecast_result['forecast'],
                                name='GNN Forecast',
                                line=dict(color='red', width=2, dash='dash')
                            ))

                            fig.update_layout(
                                title=f"GNN Multi-Asset Forecast for {target_asset}",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                hovermode='x unified',
                                height=500
                            )

                            st.plotly_chart(fig)

                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Forecast Horizon", f"{_gnn_horizon} days")
                            with col2:
                                st.metric("Assets Used", len(multi_df.columns))
                            with col3:
                                avg_conf = forecast_result['confidence'].mean()
                                st.metric("Avg Confidence", f"{avg_conf:.1%}")
                            with col4:
                                last_price = multi_df[target_asset].iloc[-1]
                                forecast_return = ((forecast_result['forecast'][-1] / last_price) - 1) * 100
                                st.metric("Forecast Return", f"{forecast_return:+.2f}%")

                            # Show relationship matrix
                            st.markdown("---")
                            st.subheader("🔗 Learned Asset Relationships")

                            relationship_matrix = gnn.get_relationship_matrix()

                            fig_rel = px.imshow(
                                relationship_matrix,
                                labels=dict(color="Connection Strength"),
                                x=multi_df.columns,
                                y=multi_df.columns,
                                color_continuous_scale='Viridis',
                                title="GNN Asset Relationship Graph (1 = connected, 0 = not connected)"
                            )

                            fig_rel.update_layout(height=500)
                            st.plotly_chart(fig_rel)

                            # Forecast table
                            with st.expander("📋 View Forecast Data"):
                                forecast_df = pd.DataFrame({
                                    'Date': forecast_result['dates'],
                                    'Forecast': forecast_result['forecast'],
                                    'Confidence': forecast_result['confidence']
                                })
                                _lb = forecast_result.get("lower_bound") if isinstance(forecast_result, dict) else None
                                _ub = forecast_result.get("upper_bound") if isinstance(forecast_result, dict) else None
                                _n = len(forecast_df)
                                forecast_df["Lower Bound"] = _bound_col(_lb, _n)
                                forecast_df["Upper Bound"] = _bound_col(_ub, _n)
                                st.dataframe(normalize_for_display(forecast_df))

                    except ImportError:
                        st.error("❌ GNN model not available. Make sure it has been recreated using the prompts.")
                    except Exception as e:
                        import traceback
                        st.error(f"GNN error: {e}")
                        with st.expander("Traceback"):
                            st.code(traceback.format_exc(), language="python")
        except Exception as e:
            st.caption(f"Tab unavailable: {e}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
