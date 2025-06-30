"""Optimization visualization tools."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from .base_optimizer import OptimizationResult
import optuna

class OptimizationVisualizer:
    """Visualizer for optimization results."""
    
    def __init__(self, results: List[OptimizationResult]):
        """Initialize visualizer.
        
        Args:
            results: List of optimization results
        """
        self.results = results
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def plot_convergence(
        self,
        metric: str = 'sharpe_ratio',
        title: str = "Optimization Convergence"
    ) -> go.Figure:
        """Plot optimization convergence.
        
        Args:
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Sort results by timestamp
        sorted_results = sorted(
            self.results,
            key=lambda x: x.timestamp
        )
        
        # Extract metric values
        values = [r.metrics[metric] for r in sorted_results]
        timestamps = [r.timestamp for r in sorted_results]
        
        # Plot metric values
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name=metric
        ))
        
        # Add best value line
        best_value = max(values)
        fig.add_hline(
            y=best_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best: {best_value:.2f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=metric,
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_metric_tradeoffs(
        self,
        x_metric: str = 'sharpe_ratio',
        y_metric: str = 'max_drawdown',
        title: str = "Metric Tradeoffs"
    ) -> go.Figure:
        """Plot metric tradeoffs.
        
        Args:
            x_metric: X-axis metric
            y_metric: Y-axis metric
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Extract metric values
        x_values = [r.metrics[x_metric] for r in self.results]
        y_values = [r.metrics[y_metric] for r in self.results]
        
        # Plot scatter
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                size=10,
                color=[r.metrics['sharpe_ratio'] for r in self.results],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            text=[f"Trial {r.trial_id}" for r in self.results],
            name="Trials"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_parameter_importance(
        self,
        study: optuna.Study,
        title: str = "Parameter Importance"
    ) -> go.Figure:
        """Plot parameter importance.
        
        Args:
            study: Optuna study
            title: Plot title
            
        Returns:
            Plotly figure
        """
        importance = optuna.importance.get_param_importances(study)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(importance.keys()),
            y=list(importance.values()),
            name="Importance"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Parameter",
            yaxis_title="Importance Score",
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_parameter_distributions(
        self,
        title: str = "Parameter Distributions"
    ) -> go.Figure:
        """Plot parameter distributions.
        
        Args:
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Get all parameters
        param_names = set()
        for result in self.results:
            param_names.update(result.parameters.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=len(param_names),
            cols=1,
            subplot_titles=list(param_names)
        )
        
        # Plot each parameter
        for i, param in enumerate(param_names, 1):
            values = [r.parameters[param] for r in self.results]
            
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=param
                ),
                row=i,
                col=1
            )
        
        fig.update_layout(
            title=title,
            height=300 * len(param_names),
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_equity_curves(
        self,
        n_curves: int = 5,
        title: str = "Top Equity Curves"
    ) -> go.Figure:
        """Plot top equity curves.
        
        Args:
            n_curves: Number of curves to plot
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Sort results by Sharpe ratio
        sorted_results = sorted(
            self.results,
            key=lambda x: x.metrics['sharpe_ratio'],
            reverse=True
        )[:n_curves]
        
        fig = go.Figure()
        
        # Plot each curve
        for i, result in enumerate(sorted_results):
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve,
                name=f"Trial {result.trial_id} (Sharpe: {result.metrics['sharpe_ratio']:.2f})"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Equity",
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def plot_drawdowns(
        self,
        n_curves: int = 5,
        title: str = "Top Drawdowns"
    ) -> go.Figure:
        """Plot top drawdowns.
        
        Args:
            n_curves: Number of curves to plot
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Sort results by Sharpe ratio
        sorted_results = sorted(
            self.results,
            key=lambda x: x.metrics['sharpe_ratio'],
            reverse=True
        )[:n_curves]
        
        fig = go.Figure()
        
        # Plot each drawdown
        for i, result in enumerate(sorted_results):
            fig.add_trace(go.Scatter(
                x=result.drawdown.index,
                y=result.drawdown,
                name=f"Trial {result.trial_id} (Sharpe: {result.metrics['sharpe_ratio']:.2f})",
                fill='tozeroy'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            template="plotly_white"
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def create_dashboard(
        self,
        study: Optional[optuna.Study] = None
    ) -> List[go.Figure]:
        """Create optimization dashboard.
        
        Args:
            study: Optuna study (optional)
            
        Returns:
            List of plotly figures
        """
        plots = []
        
        # Add convergence plot
        plots.append(self.plot_convergence())
        
        # Add metric tradeoffs
        plots.append(self.plot_metric_tradeoffs())
        
        # Add parameter importance if study is provided
        if study:
            plots.append(self.plot_parameter_importance(study))
        
        # Add parameter distributions
        plots.append(self.plot_parameter_distributions())
        
        # Add equity curves
        plots.append(self.plot_equity_curves())
        
        # Add drawdowns
        plots.append(self.plot_drawdowns())
        
        return {'success': True, 'result': plots, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    @staticmethod
    def display_optimization_summary(results: Dict[str, Any]) -> None:
        """Display optimization summary.
        
        Args:
            results: Optimization results dictionary
        """
        import streamlit as st
        
        st.subheader("Optimization Results")
        
        # Display best parameters
        if "best_params" in results:
            st.write("**Best Parameters:**")
            for param, value in results["best_params"].items():
                st.write(f"- {param}: {value}")
        
        # Display best score
        if "best_score" in results:
            st.write(f"**Best Score:** {results['best_score']:.4f}")
        
        # Display optimization time
        if "optimization_time" in results:
            st.write(f"**Optimization Time:** {results['optimization_time']:.2f} seconds")
        
        # Display number of iterations
        if "n_iterations" in results:
            st.write(f"**Number of Iterations:** {results['n_iterations']}") 
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}