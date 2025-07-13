"""
Visualization utilities for market analysis results.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataVisualizer:
    """Class for visualizing data analysis results."""

    def __init__(self, style: str = "seaborn"):
        """
        Initialize the data visualizer.

        Args:
            style: Plot style to use ('seaborn', 'ggplot', 'bmh', etc.)
        """
        plt.style.use(style)
        self.figsize = (12, 6)
        self.dpi = 100

    def plot_data_overview(
        self, data: pd.DataFrame, title: str = "Data Overview", save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot data overview with basic statistics.

        Args:
            data: DataFrame with data
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

            # Plot 1: Data distribution
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data[numeric_cols].hist(ax=axes[0, 0], bins=20)
                axes[0, 0].set_title("Data Distribution")

            # Plot 2: Missing values
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                missing_data.plot(kind="bar", ax=axes[0, 1])
                axes[0, 1].set_title("Missing Values")
                axes[0, 1].tick_params(axis="x", rotation=45)

            # Plot 3: Correlation matrix
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[1, 0])
                axes[1, 0].set_title("Correlation Matrix")

            # Plot 4: Data types
            data_types = data.dtypes.value_counts()
            data_types.plot(kind="bar", ax=axes[1, 1])
            axes[1, 1].set_title("Data Types")
            axes[1, 1].tick_params(axis="x", rotation=45)

            plt.suptitle(title)
            plt.tight_layout()

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting data overview: {str(e)}")

    def plot_time_series(
        self,
        data: pd.DataFrame,
        columns: List[str],
        title: str = "Time Series Data",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot time series data.

        Args:
            data: DataFrame with time series data
            columns: List of columns to plot
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)

            for column in columns:
                if column in data.columns:
                    plt.plot(data.index, data[column], label=column)

            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)

            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting time series: {str(e)}")

    def plot_box_plots(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Box Plots",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot box plots for numeric columns.

        Args:
            data: DataFrame with data
            columns: List of columns to plot (if None, uses all numeric columns)
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if len(columns) > 0:
                plt.figure(figsize=self.figsize, dpi=self.dpi)
                data[columns].boxplot()
                plt.title(title)
                plt.xticks(rotation=45)
                plt.tight_layout()

                if save_path:
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path)
                    logger.info(f"Plot saved to {save_path}")

                plt.show()
            else:
                logger.warning("No numeric columns found for box plots")

        except Exception as e:
            logger.error(f"Error plotting box plots: {str(e)}")

    def plot_scatter_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Scatter Matrix",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot scatter matrix for numeric columns.

        Args:
            data: DataFrame with data
            columns: List of columns to plot (if None, uses all numeric columns)
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if len(columns) >= 2:
                plt.figure(figsize=(12, 12), dpi=self.dpi)
                pd.plotting.scatter_matrix(data[columns], alpha=0.5, figsize=(12, 12))
                plt.suptitle(title)
                plt.tight_layout()

                if save_path:
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path)
                    logger.info(f"Plot saved to {save_path}")

                plt.show()
            else:
                logger.warning("Need at least 2 numeric columns for scatter matrix")

        except Exception as e:
            logger.error(f"Error plotting scatter matrix: {str(e)}")

    def plot_summary_statistics(
        self, data: pd.DataFrame, title: str = "Summary Statistics", save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot summary statistics.

        Args:
            data: DataFrame with data
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])

            if len(numeric_data.columns) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

                # Mean and std
                means = numeric_data.mean()
                stds = numeric_data.std()

                axes[0, 0].bar(range(len(means)), means)
                axes[0, 0].set_title("Mean Values")
                axes[0, 0].set_xticks(range(len(means)))
                axes[0, 0].set_xticklabels(means.index, rotation=45)

                axes[0, 1].bar(range(len(stds)), stds)
                axes[0, 1].set_title("Standard Deviations")
                axes[0, 1].set_xticks(range(len(stds)))
                axes[0, 1].set_xticklabels(stds.index, rotation=45)

                # Min and max
                mins = numeric_data.min()
                maxs = numeric_data.max()

                axes[1, 0].bar(range(len(mins)), mins)
                axes[1, 0].set_title("Minimum Values")
                axes[1, 0].set_xticks(range(len(mins)))
                axes[1, 0].set_xticklabels(mins.index, rotation=45)

                axes[1, 1].bar(range(len(maxs)), maxs)
                axes[1, 1].set_title("Maximum Values")
                axes[1, 1].set_xticks(range(len(maxs)))
                axes[1, 1].set_xticklabels(maxs.index, rotation=45)

                plt.suptitle(title)
                plt.tight_layout()

                if save_path:
                    save_path = Path(save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path)
                    logger.info(f"Plot saved to {save_path}")

                plt.show()
            else:
                logger.warning("No numeric columns found for summary statistics")

        except Exception as e:
            logger.error(f"Error plotting summary statistics: {str(e)}")


class MarketVisualizer:
    """Class for visualizing market analysis results."""

    def __init__(self, style: str = "seaborn"):
        """
        Initialize the market visualizer.

        Args:
            style: Plot style to use ('seaborn', 'ggplot', 'bmh', etc.)
        """
        plt.style.use(style)
        self.figsize = (12, 6)
        self.dpi = 100

    def plot_price_data(
        self, data: pd.DataFrame, title: str = "Price Data", save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot price data with volume.

        Args:
            data: DataFrame with price data
            title: Plot title
            save_path: Optional path to save the plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=self.figsize, dpi=self.dpi, gridspec_kw={"height_ratios": [3, 1]}
            )

            # Plot price data
            ax1.plot(data.index, data["close"], label="Close Price", color="blue")
            ax1.set_title(title)
            ax1.set_ylabel("Price")
            ax1.grid(True)
            ax1.legend()

            # Plot volume
            ax2.bar(data.index, data["volume"], label="Volume", color="gray", alpha=0.5)
            ax2.set_ylabel("Volume")
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
        save_path: Optional[Union[str, Path]] = None,
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
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=self.figsize, dpi=self.dpi, gridspec_kw={"height_ratios": [3, 1]}
            )

            # Plot price data
            ax1.plot(data.index, data["close"], label="Close Price", color="blue")
            ax1.set_title(title)
            ax1.set_ylabel("Price")
            ax1.grid(True)
            ax1.legend()

            # Plot indicators
            for indicator in indicators:
                if indicator in data.columns:
                    ax2.plot(data.index, data[indicator], label=indicator)

            ax2.set_ylabel("Indicator Value")
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
        save_path: Optional[Union[str, Path]] = None,
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
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=self.figsize, dpi=self.dpi, gridspec_kw={"height_ratios": [3, 1]}
            )

            # Plot price data
            ax1.plot(data.index, data["close"], label="Close Price", color="blue")
            ax1.set_title(f"{title} - {regime.get('name', 'Unknown Regime')}")
            ax1.set_ylabel("Price")
            ax1.grid(True)
            ax1.legend()

            # Plot regime metrics
            metrics = regime.get("metrics", {})
            if metrics:
                ax2.bar(metrics.keys(), metrics.values())
                ax2.set_ylabel("Metric Value")
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
        save_path: Optional[Union[str, Path]] = None,
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
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
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
        self, data: pd.DataFrame, column: str, title: Optional[str] = None, save_path: Optional[Union[str, Path]] = None
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
        save_path: Optional[Union[str, Path]] = None,
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
            plt.plot(data.index, data[column], label=f"{column}", alpha=0.5)
            plt.plot(data.index, rolling_mean, label=f"{window}-day Moving Average")
            plt.fill_between(
                data.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.2,
                label=f"{window}-day Standard Deviation",
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
