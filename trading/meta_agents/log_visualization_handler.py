"""
Log Visualization Handler

This module implements handlers for log visualization and analysis.
It provides functionality for creating and managing log visualizations,
including time series, level distributions, and error trend analysis.

Note: This module was adapted from the legacy automation/core/log_visualization_handler.py file.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

@dataclass
class LogEntry:
    """Represents a single log entry with metadata."""
    timestamp: datetime
    level: str
    message: str
    source: str
    metadata: Dict[str, Any]

@dataclass
class LogVisualization:
    """Represents a log visualization with its data and layout."""
    id: str
    type: str
    data: Dict[str, Any]
    layout: Dict[str, Any]
    metadata: Dict[str, Any]

class LogVisualizationHandler:
    """Handler for log visualization and analysis."""
    
    def __init__(self, config: Dict):
        """Initialize the log visualization handler."""
        self.config = config
        self.setup_logging()
        self.log_entries: List[LogEntry] = []
        self.visualizations: Dict[str, LogVisualization] = {}
        self.visualization_templates = self._load_visualization_templates()
    
    def setup_logging(self):
        """Configure logging for log visualization."""
        log_path = Path("logs/visualization")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "visualization.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_visualization_templates(self) -> Dict:
        """Load visualization templates."""
        try:
            templates_path = Path("config/visualization_templates.yaml")
            if templates_path.exists():
                with open(templates_path, 'r') as f:
                    return yaml.safe_load(f)
            return self._get_default_templates()
        except Exception as e:
            self.logger.error(f"Error loading visualization templates: {str(e)}")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict:
        """Get default visualization templates."""
        return {
            'time_series': {
                'type': 'line',
                'layout': {
                    'title': 'Log Entries Over Time',
                    'xaxis_title': 'Time',
                    'yaxis_title': 'Count',
                    'showlegend': True
                }
            },
            'level_distribution': {
                'type': 'pie',
                'layout': {
                    'title': 'Log Level Distribution',
                    'showlegend': True
                }
            },
            'source_distribution': {
                'type': 'bar',
                'layout': {
                    'title': 'Log Source Distribution',
                    'xaxis_title': 'Source',
                    'yaxis_title': 'Count',
                    'showlegend': False
                }
            },
            'error_trend': {
                'type': 'line',
                'layout': {
                    'title': 'Error Trend Analysis',
                    'xaxis_title': 'Time',
                    'yaxis_title': 'Error Count',
                    'showlegend': True
                }
            }
        }
    
    async def add_log_entry(self, entry: LogEntry):
        """Add a log entry to the visualization system."""
        try:
            self.log_entries.append(entry)
            self.logger.info(f"Added log entry from {entry.source}")
        except Exception as e:
            self.logger.error(f"Error adding log entry: {str(e)}")
            raise
    
    async def create_visualization(self, viz_id: str, viz_type: str, time_range: Optional[timedelta] = None) -> LogVisualization:
        """Create a new log visualization."""
        try:
            if viz_id in self.visualizations:
                raise ValueError(f"Visualization {viz_id} already exists")
            
            template = self.visualization_templates.get(viz_type)
            if not template:
                raise ValueError(f"Unknown visualization type: {viz_type}")
            
            # Filter log entries by time range if specified
            entries = self.log_entries
            if time_range:
                cutoff = datetime.now() - time_range
                entries = [e for e in entries if e.timestamp >= cutoff]
            
            # Create visualization based on type
            if viz_type == 'time_series':
                data = self._create_time_series_data(entries)
            elif viz_type == 'level_distribution':
                data = self._create_level_distribution_data(entries)
            elif viz_type == 'source_distribution':
                data = self._create_source_distribution_data(entries)
            elif viz_type == 'error_trend':
                data = self._create_error_trend_data(entries)
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            visualization = LogVisualization(
                id=viz_id,
                type=viz_type,
                data=data,
                layout=template['layout'],
                metadata={'created_at': datetime.now().isoformat()}
            )
            
            self.visualizations[viz_id] = visualization
            self.logger.info(f"Created visualization: {viz_id}")
            return visualization
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            raise
    
    def _create_time_series_data(self, entries: List[LogEntry]) -> Dict:
        """Create time series visualization data."""
        df = pd.DataFrame([
            {'timestamp': e.timestamp, 'level': e.level}
            for e in entries
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Group by time intervals and count entries
        time_series = df.groupby([pd.Grouper(freq='1H'), 'level']).size().unstack(fill_value=0)
        
        return {
            'x': time_series.index.tolist(),
            'y': {level: time_series[level].tolist() for level in time_series.columns},
            'type': 'line'
        }
    
    def _create_level_distribution_data(self, entries: List[LogEntry]) -> Dict:
        """Create level distribution visualization data."""
        df = pd.DataFrame([
            {'level': e.level}
            for e in entries
        ])
        
        level_counts = df['level'].value_counts()
        
        return {
            'labels': level_counts.index.tolist(),
            'values': level_counts.values.tolist(),
            'type': 'pie'
        }
    
    def _create_source_distribution_data(self, entries: List[LogEntry]) -> Dict:
        """Create source distribution visualization data."""
        df = pd.DataFrame([
            {'source': e.source}
            for e in entries
        ])
        
        source_counts = df['source'].value_counts()
        
        return {
            'x': source_counts.index.tolist(),
            'y': source_counts.values.tolist(),
            'type': 'bar'
        }
    
    def _create_error_trend_data(self, entries: List[LogEntry]) -> Dict:
        """Create error trend visualization data."""
        df = pd.DataFrame([
            {'timestamp': e.timestamp, 'is_error': e.level in ['ERROR', 'CRITICAL']}
            for e in entries
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Group by time intervals and count errors
        error_trend = df.groupby(pd.Grouper(freq='1H'))['is_error'].sum()
        
        return {
            'x': error_trend.index.tolist(),
            'y': error_trend.values.tolist(),
            'type': 'line'
        }
    
    async def update_visualization(self, viz_id: str, updates: Dict[str, Any]) -> LogVisualization:
        """Update a visualization's properties."""
        try:
            if viz_id not in self.visualizations:
                raise ValueError(f"Visualization {viz_id} not found")
            
            visualization = self.visualizations[viz_id]
            for key, value in updates.items():
                if hasattr(visualization, key):
                    setattr(visualization, key, value)
            
            self.logger.info(f"Updated visualization: {viz_id}")
            return visualization
            
        except Exception as e:
            self.logger.error(f"Error updating visualization: {str(e)}")
            raise
    
    async def render_visualization(self, viz_id: str) -> str:
        """Render a visualization to HTML."""
        try:
            if viz_id not in self.visualizations:
                raise ValueError(f"Visualization {viz_id} not found")
            
            visualization = self.visualizations[viz_id]
            
            if visualization.type == 'time_series':
                fig = self._render_time_series(visualization)
            elif visualization.type == 'level_distribution':
                fig = self._render_level_distribution(visualization)
            elif visualization.type == 'source_distribution':
                fig = self._render_source_distribution(visualization)
            elif visualization.type == 'error_trend':
                fig = self._render_error_trend(visualization)
            else:
                raise ValueError(f"Unsupported visualization type: {visualization.type}")
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Error rendering visualization: {str(e)}")
            raise
    
    def _render_time_series(self, visualization: LogVisualization) -> go.Figure:
        """Render a time series visualization."""
        fig = go.Figure()
        
        for level, values in visualization.data['y'].items():
            fig.add_trace(go.Scatter(
                x=visualization.data['x'],
                y=values,
                name=level,
                mode='lines'
            ))
        
        fig.update_layout(**visualization.layout)
        return fig
    
    def _render_level_distribution(self, visualization: LogVisualization) -> go.Figure:
        """Render a level distribution visualization."""
        fig = go.Figure(data=[go.Pie(
            labels=visualization.data['labels'],
            values=visualization.data['values']
        )])
        
        fig.update_layout(**visualization.layout)
        return fig
    
    def _render_source_distribution(self, visualization: LogVisualization) -> go.Figure:
        """Render a source distribution visualization."""
        fig = go.Figure(data=[go.Bar(
            x=visualization.data['x'],
            y=visualization.data['y']
        )])
        
        fig.update_layout(**visualization.layout)
        return fig
    
    def _render_error_trend(self, visualization: LogVisualization) -> go.Figure:
        """Render an error trend visualization."""
        fig = go.Figure(data=[go.Scatter(
            x=visualization.data['x'],
            y=visualization.data['y'],
            mode='lines',
            name='Errors'
        )])
        
        fig.update_layout(**visualization.layout)
        return fig
    
    def get_visualization(self, viz_id: str) -> Optional[LogVisualization]:
        """Get a visualization by ID."""
        return self.visualizations.get(viz_id)
    
    def get_all_visualizations(self) -> List[LogVisualization]:
        """Get all visualizations."""
        return list(self.visualizations.values())
    
    def get_visualization_templates(self) -> Dict:
        """Get the current visualization templates."""
        return self.visualization_templates 