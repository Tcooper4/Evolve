"""
LLM utilities for generating strategy summaries and commentary.
"""

import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def generate_strategy_commentary(df: pd.DataFrame) -> str:
    """
    Generate AI commentary on strategy performance.
    
    Args:
        df: DataFrame with strategy performance metrics
        
    Returns:
        String with AI-generated commentary
    """
    try:
        if df.empty:
            return "No performance data available for analysis."
            
        # Calculate summary statistics
        avg_mse = df['MSE'].mean()
        avg_sharpe = df['Sharpe Ratio'].mean()
        avg_win_rate = df['Win Rate'].mean()
        
        # Identify best and worst performers
        best_model = df.loc[df['Sharpe Ratio'].idxmax(), 'Model']
        worst_model = df.loc[df['Sharpe Ratio'].idxmin(), 'Model']
        
        # Generate commentary
        commentary = f"""
STRATEGY PERFORMANCE ANALYSIS
============================

OVERVIEW:
- Average MSE: {avg_mse:.4f}
- Average Sharpe Ratio: {avg_sharpe:.4f}
- Average Win Rate: {avg_win_rate:.2%}

TOP PERFORMER:
- Model: {best_model}
- Sharpe Ratio: {df.loc[df['Sharpe Ratio'].idxmax(), 'Sharpe Ratio']:.4f}

BOTTOM PERFORMER:
- Model: {worst_model}
- Sharpe Ratio: {df.loc[df['Sharpe Ratio'].idxmin(), 'Sharpe Ratio']:.4f}

RECOMMENDATIONS:
"""
        
        # Add specific recommendations
        if avg_sharpe < 1.0:
            commentary += "- Consider reducing risk or improving model selection\n"
        if avg_win_rate < 0.6:
            commentary += "- Strategy accuracy needs improvement\n"
        if avg_mse > 0.1:
            commentary += "- Model predictions have high error rates\n"
            
        if avg_sharpe > 1.5 and avg_win_rate > 0.7:
            commentary += "- Overall performance is excellent\n"
            
        return commentary.strip()
        
    except Exception as e:
        logger.error(f"Error generating strategy commentary: {str(e)}")
        return f"Error generating commentary: {str(e)}" 