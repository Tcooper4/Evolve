"""
Optimization Visualizer.

This module provides visualization tools for optimization results, designed to
be used with Streamlit for interactive display.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

class OptimizationVisualizer:
    """Visualization tools for optimization results."""
    
    @staticmethod
    def plot_optimization_progress(results: Dict) -> go.Figure:
        """Plot optimization progress over iterations.
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Plotly figure object
        """
        # Extract scores
        scores = [r['score'] for r in results['all_results']]
        iterations = range(1, len(scores) + 1)
        
        # Create figure
        fig = go.Figure()
        
        # Add score line
        fig.add_trace(go.Scatter(
            x=iterations,
            y=scores,
            mode='lines+markers',
            name='Score',
            line=dict(color='blue')
        ))
        
        # Add best score line
        best_scores = [max(scores[:i+1]) for i in range(len(scores))]
        fig.add_trace(go.Scatter(
            x=iterations,
            y=best_scores,
            mode='lines',
            name='Best Score',
            line=dict(color='green', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Optimization Progress',
            xaxis_title='Iteration',
            yaxis_title='Score',
            showlegend=True
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    @staticmethod
    def plot_parameter_importance(results: Dict) -> go.Figure:
        """Plot parameter importance based on optimization results.
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Plotly figure object
        """
        # Extract parameters and scores
        params = []
        scores = []
        for result in results['all_results']:
            params.append(result['params'])
            scores.append(result['score'])
        
        # Convert to DataFrame
        df = pd.DataFrame(params)
        df['score'] = scores
        
        # Calculate parameter importance
        importance = {}
        for param in df.columns[:-1]:  # Exclude score column
            correlation = df[param].corr(df['score'])
            importance[param] = abs(correlation)
        
        # Create figure
        fig = go.Figure()
        
        # Add importance bars
        fig.add_trace(go.Bar(
            x=list(importance.keys()),
            y=list(importance.values()),
            name='Parameter Importance'
        ))
        
        # Update layout
        fig.update_layout(
            title='Parameter Importance',
            xaxis_title='Parameter',
            yaxis_title='Importance (|Correlation|)',
            showlegend=False
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    @staticmethod
    def plot_parameter_distributions(results: Dict) -> go.Figure:
        """Plot parameter value distributions for best results.
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Plotly figure object
        """
        # Extract parameters and scores
        params = []
        scores = []
        for result in results['all_results']:
            params.append(result['params'])
            scores.append(result['score'])
        
        # Convert to DataFrame
        df = pd.DataFrame(params)
        df['score'] = scores
        
        # Get top 20% of results
        top_n = int(len(df) * 0.2)
        top_df = df.nlargest(top_n, 'score')
        
        # Create subplots
        param_names = [col for col in df.columns if col != 'score']
        n_params = len(param_names)
        fig = make_subplots(rows=n_params, cols=1,
                          subplot_titles=param_names)
        
        # Add histograms for each parameter
        for i, param in enumerate(param_names, 1):
            fig.add_trace(
                go.Histogram(x=top_df[param], name=param),
                row=i, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Parameter Distributions (Top 20% Results)',
            height=300 * n_params,
            showlegend=False
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    @staticmethod
    def plot_parameter_correlations(results: Dict) -> go.Figure:
        """Plot parameter correlation matrix.
        
        Args:
            results: Dictionary containing optimization results
            
        Returns:
            Plotly figure object
        """
        # Extract parameters and scores
        params = []
        scores = []
        for result in results['all_results']:
            params.append(result['params'])
            scores.append(result['score'])
        
        # Convert to DataFrame
        df = pd.DataFrame(params)
        df['score'] = scores
        
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        # Update layout
        fig.update_layout(
            title='Parameter Correlations',
            xaxis_title='Parameter',
            yaxis_title='Parameter'
        )
        
        return {'success': True, 'result': fig, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    @staticmethod
    def display_optimization_summary(results: Dict) -> None:
        """Display optimization summary in Streamlit.
        
        Args:
            results: Dictionary containing optimization results
        """
        st.subheader('Optimization Summary')
        
        # Display best score
        st.metric('Best Score', f"{results['best_score']:.4f}")
        
        # Display best parameters
        st.subheader('Best Parameters')
        for param, value in results['best_params'].items():
            st.text(f"{param}: {value}")
        
        # Display progress plot
        st.subheader('Optimization Progress')
        st.plotly_chart(OptimizationVisualizer.plot_optimization_progress(results))
        
        # Display parameter importance
        st.subheader('Parameter Importance')
        st.plotly_chart(OptimizationVisualizer.plot_parameter_importance(results))
        
        # Display parameter distributions
        st.subheader('Parameter Distributions')
        st.plotly_chart(OptimizationVisualizer.plot_parameter_distributions(results))
        
        # Display parameter correlations
        st.subheader('Parameter Correlations')
        st.plotly_chart(OptimizationVisualizer.plot_parameter_correlations(results)) 
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}