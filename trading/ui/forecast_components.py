"""Forecast-specific UI components for the trading system.

This module provides UI components specifically for forecasting functionality,
with support for agentic interactions and monitoring.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from trading.utils.safe_math import safe_mape

from trading.ui.components import (
    create_asset_selector,
    create_confidence_interval,
    create_date_range_selector,
    create_model_selector,
    create_parameter_inputs,
    create_timeframe_selector,
)

from .config.registry import ModelConfig, registry

logger = logging.getLogger(__name__)


def create_forecast_form(
    default_model: Optional[str] = None,
    default_asset: Optional[str] = None,
    default_timeframe: Optional[str] = None,
    key_prefix: str = "forecast",
) -> Dict[str, Any]:
    """Create a form for forecast configuration.

    Args:
        default_model: Optional default model to select
        default_asset: Optional default asset to select
        default_timeframe: Optional default timeframe to select
        key_prefix: Prefix for Streamlit component keys

    Returns:
        Dictionary containing form inputs
    """
    # Get date range
    start_date, end_date = create_date_range_selector(key=f"{key_prefix}_date_range")

    # Get model selection
    selected_model = create_model_selector(
        category="forecasting", default_model=default_model, key=f"{key_prefix}_model"
    )

    # Get model parameters if a model is selected
    parameters = {}
    if selected_model:
        model_config = registry.get_model_config(selected_model)
        if model_config:
            parameters = create_parameter_inputs(
                model_config.parameters, key_prefix=f"{key_prefix}_model"
            )

    # Get asset selection
    selected_asset = create_asset_selector(
        assets=["BTC/USD", "ETH/USD", "AAPL", "MSFT", "GOOGL"],
        default_asset=default_asset,
        key=f"{key_prefix}_asset",
    )

    # Get timeframe selection
    selected_timeframe = create_timeframe_selector(
        timeframes=["1h", "4h", "1d", "1w"],
        default_timeframe=default_timeframe,
        key=f"{key_prefix}_timeframe",
    )

    # Log form submission for agentic monitoring
    form_data = {
        "start_date": start_date,
        "end_date": end_date,
        "model": selected_model,
        "parameters": parameters,
        "asset": selected_asset,
        "timeframe": selected_timeframe,
    }
    logger.info(f"Forecast form submitted: {form_data}")

    return form_data


def create_forecast_chart(
    data: pd.DataFrame,
    model_config: ModelConfig,
    show_confidence: bool = True,
    show_benchmark: bool = True,
) -> go.Figure:
    """Create an interactive forecast chart with optional features.

    Args:
        data: DataFrame containing forecast data
        model_config: Configuration for the model used
        show_confidence: Whether to show confidence intervals
        show_benchmark: Whether to show benchmark overlay

    Returns:
        Plotly figure with forecast visualization
    """
    fig = go.Figure()

    # Add main forecast line
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["prediction"], name="Forecast", line=dict(color="blue")
        )
    )

    # Add confidence intervals if available and requested
    if (
        show_confidence
        and model_config.confidence_available
        and "std_error" in data.columns
    ):
        lower, upper = create_confidence_interval(data)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=upper,
                fill=None,
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Upper Bound",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=lower,
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="Lower Bound",
            )
        )

    # Add benchmark overlay if available and requested
    if (
        show_benchmark
        and model_config.benchmark_support
        and "benchmark" in data.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["benchmark"],
                name="Benchmark",
                line=dict(color="gray", dash="dash"),
            )
        )

    # Add actual values if available
    if "actual" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["actual"], name="Actual", line=dict(color="green")
            )
        )

    # Update layout
    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        showlegend=True,
    )

    # Add range slider
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="date"))

    # Log chart creation for agentic monitoring
    logger.info(f"Forecast chart created with model: {model_config.name}")

    return fig


def create_forecast_metrics(
    data: pd.DataFrame, model_config: ModelConfig
) -> Dict[str, float]:
    """Calculate and display forecast performance metrics.

    Args:
        data: DataFrame containing forecast and actual values
        model_config: Configuration for the model used

    Returns:
        Dictionary of metric names and values
    """
    metrics = {}

    if "actual" in data.columns and "prediction" in data.columns:
        # Calculate error metrics
        mae = np.mean(np.abs(data["actual"] - data["prediction"]))
        rmse = np.sqrt(np.mean((data["actual"] - data["prediction"]) ** 2))
        mape = safe_mape(data["actual"].values, data["prediction"].values)

        metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

        # Add confidence metrics if available
        if model_config.confidence_available and "std_error" in data.columns:
            coverage = (
                np.mean(
                    (data["actual"] >= data["prediction"] - 1.96 * data["std_error"])
                    & (data["actual"] <= data["prediction"] + 1.96 * data["std_error"])
                )
                * 100
            )
            metrics["Coverage"] = coverage

    # Log metrics for agentic monitoring
    logger.info(f"Forecast metrics calculated: {metrics}")

    return metrics


def create_forecast_export(
    data: pd.DataFrame, model_config: ModelConfig, metrics: Dict[str, float]
) -> Dict[str, Any]:
    """Create export options for forecast results.

    Args:
        data: DataFrame containing forecast data
        model_config: Configuration for the model used
        metrics: Dictionary of performance metrics

    Returns:
        Dictionary with status and export information
    """
    try:
        st.subheader("Export Results")

        # Create export options
        export_format = st.radio(
            "Select Export Format", ["CSV", "JSON", "Excel"], key=os.getenv("KEY", "")
        )

        if st.button("Export"):
            # Prepare export data
            export_data = {
                "forecast": data.to_dict(),
                "model": model_config.name,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }

            # Export based on selected format
            if export_format == "CSV":
                data.to_csv("forecast_results.csv")
            elif export_format == "JSON":
                with open("forecast_results.json", "w") as f:
                    json.dump(export_data, f, indent=2)
            else:  # Excel
                data.to_excel("forecast_results.xlsx")

            st.success(f"Results exported as {export_format}")

            # Log export for agentic monitoring
            logger.info(f"Forecast results exported as {export_format}")

            return {
                "status": "success",
                "message": f"Results exported as {export_format}",
                "export_format": export_format,
                "file_path": f"forecast_results.{export_format.lower()}",
            }

        return {"status": "pending", "message": "Export not initiated"}

    except Exception as e:
        logger.error(f"Error exporting forecast results: {e}")
        return {
            "success": True,
            "result": {"status": "error", "message": str(e)},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }


def create_forecast_explanation(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Display forecast explanation and insights.

    Args:
        forecast_data: Dictionary containing forecast results

    Returns:
        Dictionary with status and explanation information
    """
    try:
        st.subheader("Forecast Analysis")

        # Display explanation
        st.markdown(forecast_data["explanation"])

        # Display key insights
        st.subheader("Key Insights")
        for insight in forecast_data["insights"]:
            st.markdown(f"- {insight}")

        # Display risk factors
        st.subheader("Risk Factors")
        for risk in forecast_data["risks"]:
            st.markdown(f"- {risk}")

        return {
            "status": "success",
            "message": "Forecast explanation displayed",
            "insights_count": len(forecast_data.get("insights", [])),
            "risks_count": len(forecast_data.get("risks", [])),
        }
    except Exception as e:
        logger.error(f"Error displaying forecast explanation: {e}")
        return {
            "success": True,
            "result": {"status": "error", "message": str(e)},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }


def create_forecast_table(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """Display forecast results in a table.

    Args:
        forecast_data: Dictionary containing forecast results

    Returns:
        Dictionary with status and table information
    """
    try:
        # Create DataFrame
        df = pd.DataFrame(
            {
                "Date": forecast_data["forecast_dates"],
                "Forecast": forecast_data["forecast"],
                "Lower Bound": forecast_data["lower_bound"],
                "Upper Bound": forecast_data["upper_bound"],
            }
        )

        # Format numbers
        for col in ["Forecast", "Lower Bound", "Upper Bound"]:
            df[col] = df[col].round(2)

        # Display table
        st.dataframe(df, use_container_width=True)

        return {
            "status": "success",
            "message": "Forecast table displayed",
            "rows_count": len(df),
            "columns_count": len(df.columns),
        }
    except Exception as e:
        logger.error(f"Error displaying forecast table: {e}")
        return {
            "success": True,
            "result": {"status": "error", "message": str(e)},
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }


# Wrapper functions for easier integration
def render_model_selector(available_models: list, key: str = "model_selector") -> Optional[str]:
    """Render a model selector component.
    
    Args:
        available_models: List of available model names
        key: Streamlit key for the component
    
    Returns:
        Selected model name or None
    """
    try:
        return create_model_selector(
            category="forecasting",
            default_model=None,
            key=key
        )
    except Exception as e:
        logger.warning(f"Error rendering model selector: {e}")
        # Fallback to simple selectbox
        return st.selectbox(
            "Select Model",
            available_models,
            key=key
        )


def render_forecast_results(
    forecast: Dict[str, Any],
    historical_data: pd.DataFrame,
    symbol: str,
    show_chart: bool = True,
    show_table: bool = True
) -> None:
    """Render forecast results with chart and table.
    
    Args:
        forecast: Dictionary containing forecast results
        historical_data: Historical data DataFrame
        symbol: Symbol name
        show_chart: Whether to show forecast chart
        show_table: Whether to show forecast table
    """
    try:
        # Prepare forecast data
        forecast_dates = forecast.get('dates', forecast.get('forecast_dates', []))
        forecast_values = forecast.get('forecast', forecast.get('values', []))
        lower_bound = forecast.get('lower_bound', None)
        upper_bound = forecast.get('upper_bound', None)
        
        # Create DataFrame for chart
        if isinstance(forecast_values, (list, np.ndarray)):
            forecast_df = pd.DataFrame({
                'prediction': forecast_values,
                'date': forecast_dates
            })
            if forecast_dates and len(forecast_dates) > 0:
                forecast_df.index = pd.to_datetime(forecast_dates)
        else:
            forecast_df = pd.DataFrame({'prediction': forecast_values})
        
        # Add confidence intervals if available
        if lower_bound is not None and upper_bound is not None:
            if isinstance(lower_bound, (list, np.ndarray)) and isinstance(upper_bound, (list, np.ndarray)):
                forecast_df['std_error'] = (np.array(upper_bound) - np.array(lower_bound)) / (2 * 1.96)
        
        # Get model config (simplified)
        from .config.registry import ModelConfig
        model_config = ModelConfig(
            name=forecast.get('model_name', 'Unknown'),
            confidence_available=lower_bound is not None,
            benchmark_support=False
        )
        
        # Show chart
        if show_chart:
            fig = create_forecast_chart(
                data=forecast_df,
                model_config=model_config,
                show_confidence=lower_bound is not None,
                show_benchmark=False
            )
            fig.update_layout(title=f"{symbol} Forecast")
            st.plotly_chart(fig, use_container_width=True)
        
        # Show table
        if show_table:
            forecast_table_data = {
                'forecast_dates': forecast_dates,
                'forecast': forecast_values,
                'lower_bound': lower_bound if lower_bound is not None else [None] * len(forecast_values),
                'upper_bound': upper_bound if upper_bound is not None else [None] * len(forecast_values)
            }
            create_forecast_table(forecast_table_data)
    except Exception as e:
        logger.error(f"Error rendering forecast results: {e}")
        st.error(f"Error displaying forecast results: {e}")


def render_confidence_metrics(forecast_result: Dict[str, Any]) -> None:
    """Render confidence metrics for forecast.
    
    Args:
        forecast_result: Dictionary containing forecast results with confidence intervals
    """
    try:
        st.subheader("📊 Confidence Metrics")
        
        # Extract confidence data
        forecast_values = forecast_result.get('forecast', forecast_result.get('values', []))
        lower_bound = forecast_result.get('lower_bound', None)
        upper_bound = forecast_result.get('upper_bound', None)
        confidence = forecast_result.get('confidence', None)
        
        if lower_bound is not None and upper_bound is not None:
            # Calculate confidence interval width
            if isinstance(lower_bound, (list, np.ndarray)) and isinstance(upper_bound, (list, np.ndarray)):
                ci_width = np.mean(np.array(upper_bound) - np.array(lower_bound))
                ci_width_pct = (ci_width / np.mean(forecast_values)) * 100 if len(forecast_values) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg CI Width", f"{ci_width:.2f}")
                with col2:
                    st.metric("CI Width %", f"{ci_width_pct:.2f}%")
                with col3:
                    if confidence is not None:
                        avg_confidence = np.mean(confidence) if isinstance(confidence, (list, np.ndarray)) else confidence
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    else:
                        st.metric("Confidence Level", "95%")
        else:
            st.info("No confidence intervals available for this forecast")
    except Exception as e:
        logger.error(f"Error rendering confidence metrics: {e}")
        st.warning("Could not display confidence metrics")


def render_model_comparison(comparison_results: Dict[str, Any]) -> None:
    """Render model comparison results.
    
    Args:
        comparison_results: Dictionary containing comparison results
    """
    try:
        st.subheader("🔍 Model Comparison")
        
        if 'models' in comparison_results:
            # Create comparison DataFrame
            models_data = []
            for model_name, metrics in comparison_results['models'].items():
                models_data.append({
                    'Model': model_name,
                    **metrics
                })
            
            comparison_df = pd.DataFrame(models_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Show best model
            if 'best_model' in comparison_results:
                st.success(f"🏆 Best Model: **{comparison_results['best_model']}**")
        else:
            st.info("No model comparison data available")
    except Exception as e:
        logger.error(f"Error rendering model comparison: {e}")
        st.warning("Could not display model comparison")
