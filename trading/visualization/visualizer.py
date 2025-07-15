"""
Enhanced Visualizer Module

Enhanced with Batch 11 features: comprehensive input validation for all DataFrame overlays
and clear exception handling with detailed error messages.

This module provides visualization utilities for trading data with robust input validation
and error handling.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Custom exception for visualization errors."""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class EnhancedVisualizer:
    """Enhanced visualizer with comprehensive input validation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.validation_config = self.config.get("validation", {
            "strict_mode": True,
            "allow_missing_data": False,
            "min_data_points": 10,
            "max_data_points": 10000,
            "required_columns": ["timestamp", "close"],
            "numeric_columns": ["open", "high", "low", "close", "volume"]
        })
        
        self.logger = logging.getLogger(__name__)
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        allow_missing: Optional[bool] = None
    ) -> bool:
        """Validate DataFrame for visualization.
        
        Args:
            df: DataFrame to validate
            required_columns: Required columns (default from config)
            numeric_columns: Numeric columns (default from config)
            min_rows: Minimum number of rows (default from config)
            max_rows: Maximum number of rows (default from config)
            allow_missing: Whether to allow missing data (default from config)
            
        Returns:
            True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Use config defaults if not provided
            required_columns = required_columns or self.validation_config["required_columns"]
            numeric_columns = numeric_columns or self.validation_config["numeric_columns"]
            min_rows = min_rows or self.validation_config["min_data_points"]
            max_rows = max_rows or self.validation_config["max_data_points"]
            allow_missing = allow_missing if allow_missing is not None else self.validation_config["allow_missing_data"]
            
            # Check if DataFrame is None or empty
            if df is None:
                raise DataValidationError("DataFrame is None")
            
            if df.empty:
                raise DataValidationError("DataFrame is empty")
            
            # Check DataFrame type
            if not isinstance(df, pd.DataFrame):
                raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")
            
            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
            
            # Check numeric columns
            non_numeric_columns = []
            for col in numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        non_numeric_columns.append(col)
            
            if non_numeric_columns:
                raise DataValidationError(f"Non-numeric columns found: {non_numeric_columns}")
            
            # Check row count
            if len(df) < min_rows:
                raise DataValidationError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
            
            if len(df) > max_rows:
                raise DataValidationError(f"DataFrame has {len(df)} rows, maximum allowed: {max_rows}")
            
            # Check for missing data
            if not allow_missing:
                missing_data = df[required_columns].isnull().sum()
                columns_with_missing = missing_data[missing_data > 0]
                if not columns_with_missing.empty:
                    raise DataValidationError(f"Missing data found in columns: {columns_with_missing.to_dict()}")
            
            # Check for infinite values
            infinite_columns = []
            for col in numeric_columns:
                if col in df.columns:
                    if np.isinf(df[col]).any():
                        infinite_columns.append(col)
            
            if infinite_columns:
                raise DataValidationError(f"Infinite values found in columns: {infinite_columns}")
            
            # Check for duplicate timestamps if timestamp column exists
            if "timestamp" in df.columns:
                duplicates = df["timestamp"].duplicated().sum()
                if duplicates > 0:
                    self.logger.warning(f"Found {duplicates} duplicate timestamps")
            
            self.logger.debug(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
            return True
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            else:
                raise DataValidationError(f"Unexpected error during validation: {str(e)}")
    
    def validate_price_data(self, df: pd.DataFrame) -> bool:
        """Validate price data specifically.
        
        Args:
            df: Price DataFrame
            
        Returns:
            True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Basic DataFrame validation
            self.validate_dataframe(
                df, 
                required_columns=["timestamp", "close"],
                numeric_columns=["open", "high", "low", "close", "volume"]
            )
            
            # Price-specific validations
            price_columns = ["open", "high", "low", "close"]
            available_price_cols = [col for col in price_columns if col in df.columns]
            
            if len(available_price_cols) < 2:
                raise DataValidationError("At least 2 price columns required (close + one other)")
            
            # Check price relationships
            for idx, row in df.iterrows():
                if "high" in df.columns and "low" in df.columns:
                    if row["high"] < row["low"]:
                        raise DataValidationError(f"High price ({row['high']}) < Low price ({row['low']}) at index {idx}")
                
                if "open" in df.columns and "close" in df.columns:
                    if row["open"] <= 0 or row["close"] <= 0:
                        raise DataValidationError(f"Non-positive price found at index {idx}: open={row['open']}, close={row['close']}")
            
            # Check for reasonable price ranges (prices should be > 0 and < 1M)
            for col in available_price_cols:
                if (df[col] <= 0).any():
                    raise DataValidationError(f"Non-positive values found in {col}")
                if (df[col] > 1000000).any():
                    self.logger.warning(f"Unusually high prices found in {col}")
            
            return True
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            else:
                raise DataValidationError(f"Price data validation error: {str(e)}")
    
    def validate_volume_data(self, df: pd.DataFrame) -> bool:
        """Validate volume data specifically.
        
        Args:
            df: Volume DataFrame
            
        Returns:
            True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Basic DataFrame validation
            self.validate_dataframe(
                df, 
                required_columns=["timestamp", "volume"],
                numeric_columns=["volume"]
            )
            
            # Volume-specific validations
            if "volume" in df.columns:
                if (df["volume"] < 0).any():
                    raise DataValidationError("Negative volume values found")
                
                # Check for reasonable volume ranges
                if (df["volume"] > 1e12).any():
                    self.logger.warning("Unusually high volume values found")
            
            return True
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            else:
                raise DataValidationError(f"Volume data validation error: {str(e)}")
    
    def validate_indicator_data(self, df: pd.DataFrame, indicator_name: str) -> bool:
        """Validate technical indicator data.
        
        Args:
            df: Indicator DataFrame
            indicator_name: Name of the indicator
            
        Returns:
            True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Basic DataFrame validation
            self.validate_dataframe(df, required_columns=["timestamp"])
            
            # Indicator-specific validations
            if indicator_name.lower() in ["rsi", "stoch", "cci"]:
                # Oscillators should be between -100 and 100
                for col in df.columns:
                    if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
                        if (df[col] < -100).any() or (df[col] > 100).any():
                            self.logger.warning(f"{indicator_name} values outside expected range [-100, 100]")
            
            elif indicator_name.lower() in ["macd", "ema", "sma"]:
                # Moving averages should be positive
                for col in df.columns:
                    if col != "timestamp" and pd.api.types.is_numeric_dtype(df[col]):
                        if (df[col] <= 0).any():
                            self.logger.warning(f"{indicator_name} contains non-positive values")
            
            return True
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise
            else:
                raise DataValidationError(f"Indicator data validation error: {str(e)}")
    
    def plot_candlestick_chart(
        self, 
        df: pd.DataFrame,
        title: str = "Price Chart",
        overlay_indicators: Optional[Dict[str, pd.DataFrame]] = None,
        volume: bool = True
    ) -> go.Figure:
        """Plot candlestick chart with validation.
        
        Args:
            df: Price DataFrame
            title: Chart title
            overlay_indicators: Dictionary of indicator DataFrames to overlay
            volume: Whether to include volume subplot
            
        Returns:
            Plotly figure
            
        Raises:
            VisualizationError: If plotting fails
        """
        try:
            # Validate input data
            self.validate_price_data(df)
            
            # Create subplots
            if volume and "volume" in df.columns:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=(title, 'Volume'),
                    row_width=[0.7, 0.3]
                )
            else:
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=(title,)
                )
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Add volume if requested
            if volume and "volume" in df.columns:
                colors = ['red' if close < open else 'green' 
                         for close, open in zip(df['close'], df['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
            
            # Add overlay indicators
            if overlay_indicators:
                for indicator_name, indicator_df in overlay_indicators.items():
                    try:
                        self.validate_indicator_data(indicator_df, indicator_name)
                        
                        for col in indicator_df.columns:
                            if col != "timestamp" and pd.api.types.is_numeric_dtype(indicator_df[col]):
                                fig.add_trace(
                                    go.Scatter(
                                        x=indicator_df['timestamp'],
                                        y=indicator_df[col],
                                        name=f"{indicator_name}_{col}",
                                        mode='lines'
                                    ),
                                    row=1, col=1
                                )
                    except DataValidationError as e:
                        self.logger.warning(f"Skipping indicator {indicator_name}: {e}")
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Price",
                height=600
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create candlestick chart: {str(e)}")
    
    def plot_line_chart(
        self, 
        df: pd.DataFrame,
        y_column: str,
        title: str = "Line Chart",
        overlay_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> go.Figure:
        """Plot line chart with validation.
        
        Args:
            df: DataFrame with data
            y_column: Column to plot on y-axis
            title: Chart title
            overlay_data: Dictionary of additional DataFrames to overlay
            
        Returns:
            Plotly figure
            
        Raises:
            VisualizationError: If plotting fails
        """
        try:
            # Validate input data
            self.validate_dataframe(
                df, 
                required_columns=["timestamp", y_column],
                numeric_columns=[y_column]
            )
            
            # Create figure
            fig = go.Figure()
            
            # Add main line
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[y_column],
                    mode='lines',
                    name=y_column
                )
            )
            
            # Add overlay data
            if overlay_data:
                for overlay_name, overlay_df in overlay_data.items():
                    try:
                        self.validate_dataframe(
                            overlay_df,
                            required_columns=["timestamp"],
                            numeric_columns=[col for col in overlay_df.columns if col != "timestamp"]
                        )
                        
                        for col in overlay_df.columns:
                            if col != "timestamp" and pd.api.types.is_numeric_dtype(overlay_df[col]):
                                fig.add_trace(
                                    go.Scatter(
                                        x=overlay_df['timestamp'],
                                        y=overlay_df[col],
                                        mode='lines',
                                        name=f"{overlay_name}_{col}"
                                    )
                                )
                    except DataValidationError as e:
                        self.logger.warning(f"Skipping overlay {overlay_name}: {e}")
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title=y_column,
                height=400
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create line chart: {str(e)}")
    
    def plot_histogram(
        self, 
        df: pd.DataFrame,
        column: str,
        title: str = "Histogram",
        bins: int = 50
    ) -> go.Figure:
        """Plot histogram with validation.
        
        Args:
            df: DataFrame with data
            column: Column to plot
            title: Chart title
            bins: Number of bins
            
        Returns:
            Plotly figure
            
        Raises:
            VisualizationError: If plotting fails
        """
        try:
            # Validate input data
            self.validate_dataframe(
                df,
                required_columns=[column],
                numeric_columns=[column]
            )
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=df[column],
                    nbinsx=bins,
                    name=column
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=column,
                yaxis_title="Frequency",
                height=400
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create histogram: {str(e)}")
    
    def plot_scatter(
        self, 
        df: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Scatter Plot",
        color_column: Optional[str] = None
    ) -> go.Figure:
        """Plot scatter plot with validation.
        
        Args:
            df: DataFrame with data
            x_column: Column for x-axis
            y_column: Column for y-axis
            title: Chart title
            color_column: Optional column for color coding
            
        Returns:
            Plotly figure
            
        Raises:
            VisualizationError: If plotting fails
        """
        try:
            # Validate input data
            required_columns = [x_column, y_column]
            numeric_columns = [x_column, y_column]
            
            if color_column:
                required_columns.append(color_column)
                if pd.api.types.is_numeric_dtype(df[color_column]):
                    numeric_columns.append(color_column)
            
            self.validate_dataframe(
                df,
                required_columns=required_columns,
                numeric_columns=numeric_columns
            )
            
            # Create scatter plot
            if color_column:
                fig = px.scatter(
                    df, 
                    x=x_column, 
                    y=y_column, 
                    color=color_column,
                    title=title
                )
            else:
                fig = px.scatter(
                    df, 
                    x=x_column, 
                    y=y_column,
                    title=title
                )
            
            fig.update_layout(height=400)
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create scatter plot: {str(e)}")
    
    def plot_heatmap(
        self, 
        correlation_matrix: pd.DataFrame,
        title: str = "Correlation Heatmap"
    ) -> go.Figure:
        """Plot correlation heatmap with validation.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Chart title
            
        Returns:
            Plotly figure
            
        Raises:
            VisualizationError: If plotting fails
        """
        try:
            # Validate correlation matrix
            if correlation_matrix is None or correlation_matrix.empty:
                raise DataValidationError("Correlation matrix is empty")
            
            if not isinstance(correlation_matrix, pd.DataFrame):
                raise DataValidationError("Correlation matrix must be a pandas DataFrame")
            
            # Check if matrix is square
            if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
                raise DataValidationError("Correlation matrix must be square")
            
            # Check if values are between -1 and 1
            if (correlation_matrix < -1).any().any() or (correlation_matrix > 1).any().any():
                raise DataValidationError("Correlation values must be between -1 and 1")
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title=title,
                height=500
            )
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create heatmap: {str(e)}")
    
    def export_chart(
        self, 
        fig: go.Figure, 
        filename: str,
        format: str = "html"
    ) -> bool:
        """Export chart to file with validation.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format (html, png, jpg, svg, pdf)
            
        Returns:
            True if export successful
            
        Raises:
            VisualizationError: If export fails
        """
        try:
            if fig is None:
                raise DataValidationError("Figure is None")
            
            if not isinstance(fig, go.Figure):
                raise DataValidationError("Expected Plotly Figure object")
            
            valid_formats = ["html", "png", "jpg", "svg", "pdf"]
            if format not in valid_formats:
                raise DataValidationError(f"Invalid format. Must be one of: {valid_formats}")
            
            # Export based on format
            if format == "html":
                fig.write_html(filename)
            elif format == "png":
                fig.write_image(filename)
            elif format == "jpg":
                fig.write_image(filename)
            elif format == "svg":
                fig.write_image(filename)
            elif format == "pdf":
                fig.write_image(filename)
            
            self.logger.info(f"Chart exported successfully to {filename}")
            return True
            
        except Exception as e:
            raise VisualizationError(f"Failed to export chart: {str(e)}")


# Convenience functions
def create_visualizer(config: Optional[Dict[str, Any]] = None) -> EnhancedVisualizer:
    """Create an enhanced visualizer instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        EnhancedVisualizer instance
    """
    return EnhancedVisualizer(config)


def validate_and_plot(
    df: pd.DataFrame,
    plot_type: str,
    **kwargs
) -> go.Figure:
    """Validate data and create plot in one step.
    
    Args:
        df: DataFrame to plot
        plot_type: Type of plot ('candlestick', 'line', 'histogram', 'scatter', 'heatmap')
        **kwargs: Additional arguments for the plot function
        
    Returns:
        Plotly figure
        
    Raises:
        VisualizationError: If plotting fails
    """
    visualizer = EnhancedVisualizer()
    
    if plot_type == "candlestick":
        return visualizer.plot_candlestick_chart(df, **kwargs)
    elif plot_type == "line":
        return visualizer.plot_line_chart(df, **kwargs)
    elif plot_type == "histogram":
        return visualizer.plot_histogram(df, **kwargs)
    elif plot_type == "scatter":
        return visualizer.plot_scatter(df, **kwargs)
    elif plot_type == "heatmap":
        return visualizer.plot_heatmap(df, **kwargs)
    else:
        raise VisualizationError(f"Unknown plot type: {plot_type}") 