from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class ResponseData:
    """Data class to hold response content and metadata."""
    content: Dict[str, Any]
    type: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class ResponseFormatter:
    """Class to format and visualize different types of responses."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the response formatter.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "config"
        
        # Load templates and visualization settings
        self.templates = self._load_templates()
        self.viz_settings = self._load_viz_settings()def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load response templates from JSON file."""
        try:
            with open(self.config_dir / "response_templates.json", "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading templates: {e}")
            return {}
            
    def _load_viz_settings(self) -> Dict[str, Dict[str, Any]]:
        """Load visualization settings from JSON file."""
        try:
            with open(self.config_dir / "viz_settings.json", "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading visualization settings: {e}")
            return {}
            
    def format_response(self, response_data: ResponseData) -> str:
        """Format response based on its type.
        
        Args:
            response_data: ResponseData object containing content and metadata
            
        Returns:
            Formatted response string
        """
        try:
            template = self.templates.get(response_data.type, {}).get("template", "")
            if not template:
                self.logger.warning(f"No template found for response type: {response_data.type}")
                return str(response_data.content)
                
            return template.format(**response_data.content)
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return str(response_data.content)
            
    def create_visualization(self, response_data: ResponseData) -> Optional[go.Figure]:
        """Create visualization based on response type.
        
        Args:
            response_data: ResponseData object containing content and metadata
            
        Returns:
            Plotly figure object or None if visualization cannot be created
        """
        try:
            viz_type = self.templates.get(response_data.type, {}).get("visualization", {}).get("type")
            if not viz_type:
                self.logger.warning(f"No visualization type found for response type: {response_data.type}")
                return None
                
            viz_method = getattr(self, f"_create_{viz_type}_viz", None)
            if not viz_method:
                self.logger.warning(f"No visualization method found for type: {viz_type}")
                return None
                
            return viz_method(response_data)
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")

    def _create_line_viz(self, response_data: ResponseData) -> go.Figure:
        """Create line plot visualization."""
        settings = self.viz_settings.get("line", {}).get("default", {})
        
        fig = go.Figure()
        
        if response_data.type == "forecast":
            # Add historical data
            fig.add_trace(go.Scatter(
                x=response_data.content["historical_dates"],
                y=response_data.content["historical_values"],
                name="Historical",
                line=dict(color=settings["line_color"], width=settings["line_width"])
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=response_data.content["forecast_dates"],
                y=response_data.content["forecast_values"],
                name="Forecast",
                line=dict(color="#ff7f0e", width=settings["line_width"])
            ))
            
            # Add confidence intervals if available
            if "confidence_intervals" in response_data.content:
                lower, upper = response_data.content["confidence_intervals"]
                fig.add_trace(go.Scatter(
                    x=response_data.content["forecast_dates"],
                    y=upper,
                    fill=None,
                    mode="lines",
                    line_color="rgba(255, 127, 14, 0.2)",
                    name="Upper Bound"
                ))
                fig.add_trace(go.Scatter(
                    x=response_data.content["forecast_dates"],
                    y=lower,
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(255, 127, 14, 0.2)",
                    name="Lower Bound"
                ))
                
        elif response_data.type == "monitor":
            # Add performance line
            fig.add_trace(go.Scatter(
                x=response_data.content["dates"],
                y=response_data.content["performance"],
                name="Performance",
                line=dict(color="#2ca02c", width=settings["line_width"])
            ))
            
            # Add threshold lines if available
            if "thresholds" in response_data.content:
                for threshold in response_data.content["thresholds"]:
                    fig.add_trace(go.Scatter(
                        x=response_data.content["dates"],
                        y=[threshold["value"]] * len(response_data.content["dates"]),
                        name=threshold["name"],
                        line=dict(color="#d62728", width=1, dash="dash")
                    ))
                    
        # Update layout
        fig.update_layout(
            title=response_data.content.get("title", ""),
            xaxis_title=response_data.content.get("xaxis_title", ""),
            yaxis_title=response_data.content.get("yaxis_title", ""),
            showlegend=settings["show_legend"],
            template=self.viz_settings["layout"]["default"]["template"]
        )
        
        return fig
        
    def _create_candlestick_viz(self, response_data: ResponseData) -> go.Figure:
        """Create candlestick chart visualization."""
        settings = self.viz_settings.get("candlestick", {}).get("default", {})
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=response_data.content["dates"],
            open=response_data.content["open"],
            high=response_data.content["high"],
            low=response_data.content["low"],
            close=response_data.content["close"],
            name="Price",
            increasing_line_color=settings["increasing_color"],
            decreasing_line_color=settings["decreasing_color"]
        ), row=1, col=1)
        
        # Add volume if available
        if "volume" in response_data.content:
            fig.add_trace(go.Bar(
                x=response_data.content["dates"],
                y=response_data.content["volume"],
                name="Volume",
                marker_color=settings["volume_color"]
            ), row=2, col=1)
            
        # Add technical indicators if available
        if "indicators" in response_data.content:
            for indicator in response_data.content["indicators"]:
                fig.add_trace(go.Scatter(
                    x=response_data.content["dates"],
                    y=indicator["values"],
                    name=indicator["name"],
                    line=dict(color=indicator.get("color", "#1f77b4"))
                ), row=1, col=1)
                
        # Update layout
        fig.update_layout(
            title=response_data.content.get("title", ""),
            xaxis_title=response_data.content.get("xaxis_title", ""),
            yaxis_title=response_data.content.get("yaxis_title", ""),
            showlegend=True,
            template=self.viz_settings["layout"]["default"]["template"]
        )
        
        return fig
        
    def _create_scatter_viz(self, response_data: ResponseData) -> go.Figure:
        """Create scatter plot visualization."""
        settings = self.viz_settings.get("scatter", {}).get("default", {})
        
        fig = go.Figure()
        
        if response_data.type == "recommendation":
            # Add price data
            fig.add_trace(go.Scatter(
                x=response_data.content["dates"],
                y=response_data.content["prices"],
                name="Price",
                line=dict(color="#1f77b4")
            ))
            
            # Add entry point
            fig.add_trace(go.Scatter(
                x=[response_data.content["entry_date"]],
                y=[response_data.content["entry_price"]],
                name="Entry",
                mode="markers",
                marker=dict(
                    color="#2ca02c",
                    size=settings["marker_size"],
                    symbol="triangle-up"
                )
            ))
            
            # Add stop loss
            fig.add_trace(go.Scatter(
                x=[response_data.content["entry_date"]],
                y=[response_data.content["stop_loss"]],
                name="Stop Loss",
                mode="markers",
                marker=dict(
                    color="#d62728",
                    size=settings["marker_size"],
                    symbol="triangle-down"
                )
            ))
            
            # Add take profit
            fig.add_trace(go.Scatter(
                x=[response_data.content["entry_date"]],
                y=[response_data.content["take_profit"]],
                name="Take Profit",
                mode="markers",
                marker=dict(
                    color="#1f77b4",
                    size=settings["marker_size"],
                    symbol="star"
                )
            ))
            
        elif response_data.type == "optimize":
            # Add optimization points
            fig.add_trace(go.Scatter(
                x=response_data.content["x_values"],
                y=response_data.content["y_values"],
                name="Points",
                mode="markers",
                marker=dict(
                    color=settings["other_points_color"],
                    size=settings["marker_size"]
                )
            ))
            
            # Add best point
            fig.add_trace(go.Scatter(
                x=[response_data.content["best_x"]],
                y=[response_data.content["best_y"]],
                name="Best Point",
                mode="markers",
                marker=dict(
                    color=settings["best_point_color"],
                    size=settings["marker_size"] * 1.5,
                    symbol="star"
                )
            ))
            
            # Add optimization path if available
            if "path" in response_data.content:
                fig.add_trace(go.Scatter(
                    x=response_data.content["path_x"],
                    y=response_data.content["path_y"],
                    name="Optimization Path",
                    line=dict(color=settings["path_color"], dash="dot")
                ))
                
        # Update layout
        fig.update_layout(
            title=response_data.content.get("title", ""),
            xaxis_title=response_data.content.get("xaxis_title", ""),
            yaxis_title=response_data.content.get("yaxis_title", ""),
            showlegend=settings["show_legend"],
            template=self.viz_settings["layout"]["default"]["template"]
        )
        
        return fig
        
    def _create_bar_viz(self, response_data: ResponseData) -> go.Figure:
        """Create bar chart visualization."""
        settings = self.viz_settings.get("bar", {}).get("default", {})
        
        fig = go.Figure()
        
        if response_data.type == "explanation":
            # Add bars for each point
            for point in response_data.content["points"]:
                color = settings["positive_color"] if point["sentiment"] > 0 else \
                        settings["negative_color"] if point["sentiment"] < 0 else \
                        settings["neutral_color"]
                        
                fig.add_trace(go.Bar(
                    x=[point["name"]],
                    y=[point["value"]],
                    name=point["name"],
                    marker_color=color
                ))
                
        elif response_data.type == "validate":
            # Add accuracy bars
            fig.add_trace(go.Bar(
                x=response_data.content["metrics"],
                y=response_data.content["values"],
                name="Metrics",
                marker_color=settings["accuracy_color"]
            ))
            
        # Update layout
        fig.update_layout(
            title=response_data.content.get("title", ""),
            xaxis_title=response_data.content.get("xaxis_title", ""),
            yaxis_title=response_data.content.get("yaxis_title", ""),
            showlegend=settings["show_legend"],
            template=self.viz_settings["layout"]["default"]["template"]
        )
        
        return fig
        
    def _create_heatmap_viz(self, response_data: ResponseData) -> go.Figure:
        """Create heatmap visualization."""
        settings = self.viz_settings.get("heatmap", {}).get("default", {})
        
        fig = go.Figure()
        
        if response_data.type == "compare":
            # Add correlation heatmap
            fig.add_trace(go.Heatmap(
                z=response_data.content["correlation_matrix"],
                x=response_data.content["assets"],
                y=response_data.content["assets"],
                colorscale=settings["colorscale"],
                showscale=settings["show_scale"]
            ))
            
        # Update layout
        fig.update_layout(
            title=response_data.content.get("title", ""),
            xaxis_title=response_data.content.get("xaxis_title", ""),
            yaxis_title=response_data.content.get("yaxis_title", ""),
            template=self.viz_settings["layout"]["default"]["template"]
        )
        
        return fig