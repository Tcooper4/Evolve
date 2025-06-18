"""Enhanced report generator with modular chart generation and agentic insights."""

import os
import json
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import pdfkit
import weasyprint
from jinja2 import Environment, FileSystemLoader
import openai
from dataclasses import dataclass
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/report/logs/report_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    theme: str = 'light'  # 'light' or 'dark'
    include_sections: List[str] = None  # List of sections to include
    export_formats: List[str] = None  # List of export formats
    strategy_params: Dict[str, Any] = None  # Strategy parameters
    model_config: Dict[str, Any] = None  # Model configuration
    run_metadata: Dict[str, Any] = None  # Run metadata
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.include_sections is None:
            self.include_sections = [
                'equity_curve',
                'drawdown',
                'returns_distribution',
                'rolling_metrics',
                'strategy_metrics',
                'trade_analysis'
            ]
        if self.export_formats is None:
            self.export_formats = ['html', 'pdf']
        if self.strategy_params is None:
            self.strategy_params = {}
        if self.model_config is None:
            self.model_config = {}
        if self.run_metadata is None:
            self.run_metadata = {
                'user': os.getenv('USER', 'unknown'),
                'run_time': datetime.utcnow().isoformat(),
                'model_version': '1.0.0',
                'strategy_name': 'unknown'
            }

class ReportGenerator:
    """Enhanced report generator with modular chart generation and agentic insights."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize report generator.
        
        Args:
            config: Optional report configuration
        """
        self.config = config or ReportConfig()
        
        # Setup Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
        # Create necessary directories
        os.makedirs('trading/report/logs', exist_ok=True)
        os.makedirs('trading/report/data', exist_ok=True)
        os.makedirs('trading/report/output', exist_ok=True)
        
        # Load theme configuration
        self.theme_config = self._load_theme_config()
        
        logger.info(f"Initialized ReportGenerator with config: {self.config}")
    
    def _load_theme_config(self) -> Dict[str, Any]:
        """Load theme configuration.
        
        Returns:
            Theme configuration dictionary
        """
        theme_file = os.path.join(os.path.dirname(__file__), 'configs', 'themes.json')
        try:
            with open(theme_file, 'r') as f:
                themes = json.load(f)
            return themes[self.config.theme]
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Failed to load theme config: {e}. Using default theme.")
            return {
                'background_color': '#ffffff',
                'text_color': '#000000',
                'grid_color': '#e0e0e0',
                'plot_colors': px.colors.qualitative.Set1
            }
    
    def _validate_performance_data(self, data: Dict[str, Any]) -> bool:
        """Validate performance data has all required fields.
        
        Args:
            data: Performance data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'equity_curve',
            'returns',
            'trades',
            'metrics'
        ]
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        return True
    
    def _create_equity_curve_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create equity curve chart.
        
        Args:
            data: Performance data dictionary
            
        Returns:
            Plotly figure
        """
        equity_curve = pd.Series(data['equity_curve'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color=self.theme_config['plot_colors'][0])
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity',
            template=self.config.theme,
            plot_bgcolor=self.theme_config['background_color'],
            paper_bgcolor=self.theme_config['background_color'],
            font=dict(color=self.theme_config['text_color']),
            showlegend=True
        )
        
        return fig
    
    def _create_drawdown_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create drawdown chart.
        
        Args:
            data: Performance data dictionary
            
        Returns:
            Plotly figure
        """
        equity_curve = pd.Series(data['equity_curve'])
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color=self.theme_config['plot_colors'][1])
        ))
        
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template=self.config.theme,
            plot_bgcolor=self.theme_config['background_color'],
            paper_bgcolor=self.theme_config['background_color'],
            font=dict(color=self.theme_config['text_color']),
            showlegend=True
        )
        
        return fig
    
    def _create_returns_distribution_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create returns distribution chart.
        
        Args:
            data: Performance data dictionary
            
        Returns:
            Plotly figure
        """
        returns = pd.Series(data['returns'])
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns,
            name='Returns',
            marker_color=self.theme_config['plot_colors'][2],
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        x = np.linspace(returns.min(), returns.max(), 100)
        y = np.exp(-(x - returns.mean())**2 / (2 * returns.std()**2)) / (returns.std() * np.sqrt(2 * np.pi))
        y = y * len(returns) * (returns.max() - returns.min()) / 50  # Scale to match histogram
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color=self.theme_config['plot_colors'][3])
        ))
        
        fig.update_layout(
            title='Returns Distribution',
            xaxis_title='Return',
            yaxis_title='Frequency',
            template=self.config.theme,
            plot_bgcolor=self.theme_config['background_color'],
            paper_bgcolor=self.theme_config['background_color'],
            font=dict(color=self.theme_config['text_color']),
            showlegend=True
        )
        
        return fig
    
    def _create_rolling_metrics_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create rolling metrics chart.
        
        Args:
            data: Performance data dictionary
            
        Returns:
            Plotly figure
        """
        returns = pd.Series(data['returns'])
        window = 252  # Annual window
        
        # Calculate rolling metrics
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name='Rolling Sharpe',
                line=dict(color=self.theme_config['plot_colors'][0])
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Rolling Volatility',
                line=dict(color=self.theme_config['plot_colors'][1])
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Rolling Metrics',
            template=self.config.theme,
            plot_bgcolor=self.theme_config['background_color'],
            paper_bgcolor=self.theme_config['background_color'],
            font=dict(color=self.theme_config['text_color']),
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=False)
        fig.update_yaxes(title_text="Volatility", secondary_y=True)
        
        return fig
    
    def _generate_performance_insights(self, data: Dict[str, Any]) -> str:
        """Generate performance insights using GPT.
        
        Args:
            data: Performance data dictionary
            
        Returns:
            Generated insights text
        """
        try:
            # Prepare metrics for GPT
            metrics = data['metrics']
            prompt = f"""
            Analyze the following trading strategy performance metrics and provide insights:
            
            Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
            Win Rate: {metrics.get('win_rate', 0):.2%}
            Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
            Total Return: {metrics.get('total_return', 0):.2%}
            Average Trade: {metrics.get('avg_trade', 0):.2%}
            Profit Factor: {metrics.get('profit_factor', 0):.2f}
            
            Provide:
            1. Overall performance assessment
            2. Key strengths and weaknesses
            3. Specific improvement suggestions
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a trading strategy analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return "Failed to generate performance insights."
    
    def _export_data(self, data: Dict[str, Any], output_dir: str) -> None:
        """Export performance data to files.
        
        Args:
            data: Performance data dictionary
            output_dir: Output directory
        """
        # Export metrics
        metrics_df = pd.DataFrame([data['metrics']])
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Export equity curve
        equity_df = pd.Series(data['equity_curve']).to_frame('equity')
        equity_df.to_csv(os.path.join(output_dir, 'equity_curve.csv'))
        
        # Export returns
        returns_df = pd.Series(data['returns']).to_frame('returns')
        returns_df.to_csv(os.path.join(output_dir, 'returns.csv'))
        
        # Export trades
        trades_df = pd.DataFrame(data['trades'])
        trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
        
        # Export full data as JSON
        with open(os.path.join(output_dir, 'full_data.json'), 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_report(self, performance_data: Dict[str, Any], output_path: str) -> str:
        """Generate performance report.
        
        Args:
            performance_data: Performance data dictionary
            output_path: Output path for report
            
        Returns:
            Path to generated report
        """
        # Validate data
        if not self._validate_performance_data(performance_data):
            raise ValueError("Invalid performance data")
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate charts
        charts = {}
        if 'equity_curve' in self.config.include_sections:
            charts['equity_curve'] = self._create_equity_curve_chart(performance_data)
        if 'drawdown' in self.config.include_sections:
            charts['drawdown'] = self._create_drawdown_chart(performance_data)
        if 'returns_distribution' in self.config.include_sections:
            charts['returns_distribution'] = self._create_returns_distribution_chart(performance_data)
        if 'rolling_metrics' in self.config.include_sections:
            charts['rolling_metrics'] = self._create_rolling_metrics_chart(performance_data)
        
        # Generate insights
        insights = self._generate_performance_insights(performance_data)
        
        # Export data if requested
        if 'csv' in self.config.export_formats or 'json' in self.config.export_formats:
            self._export_data(performance_data, output_dir)
        
        # Load template
        template = self.env.get_template('report_template.html')
        
        # Render template
        html_content = template.render(
            charts=charts,
            metrics=performance_data['metrics'],
            insights=insights,
            strategy_params=self.config.strategy_params,
            model_config=self.config.model_config,
            run_metadata=self.config.run_metadata,
            theme=self.theme_config
        )
        
        # Save HTML report
        html_path = output_path.replace('.pdf', '.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Convert to PDF if requested
        if 'pdf' in self.config.export_formats:
            try:
                pdfkit.from_file(html_path, output_path)
            except Exception as e:
                logger.warning(f"Failed to generate PDF with pdfkit: {e}")
                try:
                    weasyprint.HTML(string=html_content).write_pdf(output_path)
                except Exception as e:
                    logger.error(f"Failed to generate PDF with weasyprint: {e}")
        
        return output_path

__all__ = ["ReportGenerator", "ReportConfig"] 