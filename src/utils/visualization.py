"""
Visualization utilities for market analysis results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketVisualizer:
    """Class for visualizing market analysis results."""
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize the market visualizer.
        
        Args:
            style: Plot style to use ('seaborn', 'ggplot', 'bmh', etc.)
        """
        plt.style.use(style)
        self.figsize = (12, 6)
        self.dpi = 100
        
    def plot_price_data(
        self,
        data: pd.DataFrame,
        title: str = "Price Data",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot price data with volume.
        
        Args:
            data: DataFrame with price data
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            ax1.plot(data.index, data['close'], label='Close Price', color='blue')
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Plot volume
            ax2.bar(data.index, data['volume'], label='Volume', color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting price data: {str(e)}")
    
    def plot_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        title: str = "Technical Indicators",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot technical indicators.
        
        Args:
            data: DataFrame with price and indicator data
            indicators: List of indicator columns to plot
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            ax1.plot(data.index, data['close'], label='Close Price', color='blue')
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Plot indicators
            for indicator in indicators:
                if indicator in data.columns:
                    ax2.plot(data.index, data[indicator], label=indicator)
            
            ax2.set_ylabel('Indicator Value')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting technical indicators: {str(e)}")
    
    def plot_market_regime(
        self,
        data: pd.DataFrame,
        regime: Dict,
        title: str = "Market Regime Analysis",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot market regime analysis.
        
        Args:
            data: DataFrame with price data
            regime: Dictionary containing regime information
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            ax1.plot(data.index, data['close'], label='Close Price', color='blue')
            ax1.set_title(f"{title} - {regime.get('name', 'Unknown Regime')}")
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Plot regime metrics
            metrics = regime.get('metrics', {})
            if metrics:
                ax2.bar(metrics.keys(), metrics.values())
                ax2.set_ylabel('Metric Value')
                ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting market regime: {str(e)}")
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot correlation matrix of selected columns.
        
        Args:
            data: DataFrame with data
            columns: List of columns to include in correlation matrix
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            if columns:
                corr_data = data[columns]
            else:
                corr_data = data.select_dtypes(include=[np.number])
            
            corr_matrix = corr_data.corr()
            
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(title)
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
    
    def plot_distribution(
        self,
        data: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot distribution of a column.
        
        Args:
            data: DataFrame with data
            column: Column to plot distribution for
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            sns.histplot(data[column], kde=True)
            plt.title(title or f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting distribution: {str(e)}")
    
    def plot_rolling_statistics(
        self,
        data: pd.DataFrame,
        column: str,
        window: int = 20,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot rolling statistics for a column.
        
        Args:
            data: DataFrame with data
            column: Column to calculate rolling statistics for
            window: Rolling window size
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # Calculate rolling statistics
            rolling_mean = data[column].rolling(window=window).mean()
            rolling_std = data[column].rolling(window=window).std()
            
            # Plot
            plt.plot(data.index, data[column], label=f'{column}', alpha=0.5)
            plt.plot(data.index, rolling_mean, label=f'{window}-day Moving Average')
            plt.fill_between(
                data.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.2,
                label=f'{window}-day Standard Deviation'
            )
            
            plt.title(title or f"Rolling Statistics for {column}")
            plt.xlabel("Date")
            plt.ylabel(column)
            plt.legend()
            plt.grid(True)
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting rolling statistics: {str(e)}") 