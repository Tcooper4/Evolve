"""
Model Explainability Page

This page provides tools for understanding model predictions and feature importance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trading.data.preprocessing import DataPreprocessor
from trading.feature_engineering.feature_engineer import FeatureEngineering

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    st.set_page_config(page_title="Model Explainability", page_icon="🔍", layout="wide")

    st.title("🔍 Model Explainability & Interpretability")
    st.markdown(
        "Understand how your models make predictions and which features are most important."
    )

    # Sidebar for model selection
    st.sidebar.header("Model Configuration")

    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["LSTM", "XGBoost", "Ensemble", "Prophet", "ARIMA"],
        help="Choose the type of model to analyze",
    )

    # Add explainer selection dropdown
    explainer_method = st.sidebar.selectbox(
        "Explainability Method",
        ["SHAP", "LIME", "Integrated"],
        help="Choose the explainability engine",
    )

    # Data upload
    st.sidebar.subheader("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your data (CSV)",
        type=["csv"],
        help="Upload the data used for model training/prediction",
    )

    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        [
            "Feature Importance",
            "SHAP Values",
            "Model Components",
            "Prediction Breakdown",
        ],
        help="Choose the type of explainability analysis",
    )

    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.success(
                f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
            )

            # Display data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(data.head())
                st.write(f"**Data Info:**")
                st.write(f"- Shape: {data.shape}")
                st.write(f"- Columns: {list(data.columns)}")
                st.write(f"- Missing values: {data.isnull().sum().sum()}")

            # Main analysis area
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"📈 {analysis_type} Analysis")

                if analysis_type == "Feature Importance":
                    show_feature_importance(data, model_type)
                elif analysis_type == "SHAP Values":
                    show_shap_analysis(data, model_type, explainer_method)
                elif analysis_type == "Model Components":
                    show_model_components(data, model_type)
                elif analysis_type == "Prediction Breakdown":
                    show_prediction_breakdown(data, model_type)

            with col2:
                st.subheader("📋 Analysis Summary")
                show_analysis_summary(data, model_type, analysis_type)

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    else:
        # Show placeholder content
        st.info("👆 Please upload a CSV file to begin analysis")

        # Show example of what the page can do
        st.subheader("🔍 What This Page Can Do")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
            **🎯 Feature Importance**
            - Identify which features drive predictions
            - Rank features by importance
            - Visualize feature contributions
            """
            )

        with col2:
            st.markdown(
                """
            **📊 SHAP Values**
            - Understand individual predictions
            - See feature interactions
            - Analyze prediction explanations
            """
            )

        with col3:
            st.markdown(
                """
            **🔧 Model Components**
            - Break down model predictions
            - Analyze seasonal patterns
            - Understand trend components
            """
            )

        with col4:
            st.markdown(
                """
            **📈 Prediction Breakdown**
            - Decompose forecasts
            - Analyze uncertainty
            - Track prediction evolution
            """
            )


def show_feature_importance(data, model_type):
    """Display feature importance analysis."""

    # Preprocess data
    DataPreprocessor()
    FeatureEngineering()

    # Prepare features
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        # Use correlation as simple feature importance
        target_col = numeric_data.columns[-1]
        features = numeric_data.drop(columns=[target_col])

        # Calculate correlation-based importance
        correlations = features.corrwith(numeric_data[target_col]).abs()
        correlations = correlations.sort_values(ascending=False)

        # Create visualization
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=correlations.values,
                y=correlations.index,
                orientation="h",
                marker_color="lightblue",
            )
        )

        fig.update_layout(
            title=f"Feature Importance ({model_type})",
            xaxis_title="Absolute Correlation",
            yaxis_title="Features",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show feature importance table
        st.subheader("📋 Feature Importance Ranking")
        importance_df = pd.DataFrame(
            {
                "Feature": correlations.index,
                "Importance": correlations.values,
                "Rank": range(1, len(correlations) + 1),
            }
        )
        st.dataframe(importance_df)

    else:
        st.warning("⚠️ Not enough numeric features for importance analysis")


def show_shap_analysis(data, model_type, explainer_method="SHAP"):
    """Display SHAP/LIME/Integrated analysis."""
    st.info(
        f"🔍 {explainer_method} analysis requires a trained model. This is a demonstration."
    )
    # Placeholder for backend integration
    fig = go.Figure()
    fig.add_annotation(
        text=f"{explainer_method} analysis would show here with a trained model",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="blue"),
    )
    fig.update_layout(
        title=f"{explainer_method} Values Analysis", height=400, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
    **What {explainer_method} Analysis Shows:**
    - **Feature Contributions**: How each feature affects predictions
    - **Individual Explanations**: Why specific predictions were made
    - **Feature Interactions**: How features work together
    - **Global Patterns**: Overall feature importance across the dataset
    """
    )


def show_model_components(data, model_type):
    """Display model components analysis."""

    # Create time series visualization
    if len(data) > 1:
        # Use first numeric column as time series
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            time_series = data[numeric_cols[0]]

            # Create decomposition-like visualization
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    "Original Data",
                    "Trend Component",
                    "Seasonal Component",
                ),
                vertical_spacing=0.1,
            )

            # Original data
            fig.add_trace(
                go.Scatter(y=time_series, name="Original", line=dict(color="blue")),
                row=1,
                col=1,
            )

            # Mock trend (simple moving average)
            trend = time_series.rolling(
                window=min(7, len(time_series) // 4), center=True
            ).mean()
            fig.add_trace(
                go.Scatter(y=trend, name="Trend", line=dict(color="red")), row=2, col=1
            )

            # Mock seasonal (residuals)
            seasonal = time_series - trend
            fig.add_trace(
                go.Scatter(y=seasonal, name="Seasonal", line=dict(color="green")),
                row=3,
                col=1,
            )

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Component statistics
            st.subheader("📊 Component Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Trend Strength", f"{abs(trend.corr(time_series)):.3f}")
            with col2:
                st.metric("Seasonal Strength", f"{abs(seasonal.corr(time_series)):.3f}")
            with col3:
                st.metric("Noise Level", f"{1 - abs(trend.corr(time_series)):.3f}")


def show_prediction_breakdown(data, model_type):
    """Display prediction breakdown analysis."""

    st.info(
        "🔍 Prediction breakdown requires model predictions. This is a demonstration."
    )

    # Create mock prediction breakdown
    if len(data) > 10:
        # Use last 10 points as "predictions"
        recent_data = data.tail(10)
        numeric_cols = recent_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            values = recent_data[numeric_cols[0]]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode="lines+markers",
                    name="Predictions",
                    line=dict(color="blue", width=2),
                )
            )

            # Add confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values * 1.1,
                    mode="lines",
                    name="Upper Bound",
                    line=dict(color="lightblue", dash="dash"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values * 0.9,
                    mode="lines",
                    name="Lower Bound",
                    line=dict(color="lightblue", dash="dash"),
                    fill="tonexty",
                )
            )

            fig.update_layout(
                title="Prediction Breakdown with Confidence Intervals",
                xaxis_title="Time Steps",
                yaxis_title="Predicted Values",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)


def show_analysis_summary(data, model_type, analysis_type):
    """Display analysis summary."""

    st.markdown(
        f"""
    **📋 Analysis Summary**

    **Model Type:** {model_type}
    **Analysis:** {analysis_type}
    **Data Points:** {len(data):,}
    **Features:** {len(data.columns)}
    """
    )

    # Data quality metrics
    st.subheader("📊 Data Quality")

    missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    st.metric("Missing Data", f"{missing_pct:.1f}%")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    st.metric("Numeric Features", len(numeric_cols))

    categorical_cols = data.select_dtypes(include=["object"]).columns
    st.metric("Categorical Features", len(categorical_cols))

    # Recommendations
    st.subheader("💡 Recommendations")

    if missing_pct > 5:
        st.warning("⚠️ Consider handling missing values")

    if len(numeric_cols) < 3:
        st.info("ℹ️ More features may improve model performance")

    if len(data) < 100:
        st.warning("⚠️ Consider collecting more data for better analysis")


if __name__ == "__main__":
    main()
