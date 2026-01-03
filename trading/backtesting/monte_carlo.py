"""
Monte Carlo Simulation for Backtesting

This module provides comprehensive Monte Carlo simulation capabilities for backtesting,
including bootstrapped historical returns, percentile calculations, and visualization.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    n_simulations: int = 1000
    confidence_levels: List[float] = None
    bootstrap_method: str = "historical"  # "historical", "parametric", "block"
    block_size: int = 20  # For block bootstrap
    random_seed: int = 42
    initial_capital: float = 10000.0
    include_dividends: bool = True
    transaction_costs: bool = True
    cost_rate: float = 0.001  # 0.1% per trade

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.05, 0.50, 0.95]  # 5th, 50th, 95th percentiles


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio backtesting.

    This class provides methods to simulate portfolio performance using
    bootstrapped historical returns and calculate confidence intervals.
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize the Monte Carlo simulator.

        Args:
            config: Configuration for the simulation
        """
        self.config = config or MonteCarloConfig()
        self.results = {}
        self.simulated_paths = None
        self.percentiles = None

        # Set random seed
        np.random.seed(self.config.random_seed)

    def simulate_portfolio_paths(
        self,
        returns: pd.Series,
        initial_capital: Optional[float] = None,
        n_simulations: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Simulate portfolio paths using bootstrapped historical returns.

        Args:
            returns: Historical returns series (daily, weekly, etc.)
            initial_capital: Starting portfolio value
            n_simulations: Number of simulation paths

        Returns:
            DataFrame with simulated portfolio paths
        """
        if returns.empty:
            raise ValueError("Returns series cannot be empty")

        # Use config values if not provided
        initial_capital = initial_capital or self.config.initial_capital
        n_simulations = n_simulations or self.config.n_simulations

        # Clean returns data
        returns = returns.dropna()
        n_periods = len(returns)

        if n_periods < 30:
            warnings.warn(f"Limited historical data: {n_periods} periods")

        logger.info(f"Simulating {n_simulations} paths over {n_periods} periods")

        # Generate bootstrapped return paths
        if self.config.bootstrap_method == "historical":
            paths = self._bootstrap_historical_returns(
                returns, n_simulations, n_periods
            )
        elif self.config.bootstrap_method == "block":
            paths = self._block_bootstrap_returns(returns, n_simulations, n_periods)
        elif self.config.bootstrap_method == "parametric":
            paths = self._parametric_bootstrap_returns(
                returns, n_simulations, n_periods
            )
        else:
            raise ValueError(
                f"Unknown bootstrap method: {self.config.bootstrap_method}"
            )

        # Convert to portfolio values
        portfolio_paths = self._calculate_portfolio_values(paths, initial_capital)

        # Store results
        self.simulated_paths = portfolio_paths
        self.results["n_simulations"] = n_simulations
        self.results["n_periods"] = n_periods
        self.results["initial_capital"] = initial_capital

        return portfolio_paths

    def _bootstrap_historical_returns(
        self, returns: pd.Series, n_simulations: int, n_periods: int
    ) -> np.ndarray:
        """
        Bootstrap historical returns by sampling with replacement.

        Args:
            returns: Historical returns
            n_simulations: Number of simulation paths
            n_periods: Number of periods to simulate

        Returns:
            Array of bootstrapped return paths
        """
        returns_array = returns.values
        paths = np.zeros((n_simulations, n_periods))

        for i in range(n_simulations):
            # Sample returns with replacement
            sampled_returns = np.random.choice(
                returns_array, size=n_periods, replace=True
            )
            paths[i] = sampled_returns

        return paths

    def _block_bootstrap_returns(
        self, returns: pd.Series, n_simulations: int, n_periods: int
    ) -> np.ndarray:
        """
        Block bootstrap to preserve time series structure.

        Args:
            returns: Historical returns
            n_simulations: Number of simulation paths
            n_periods: Number of periods to simulate

        Returns:
            Array of block-bootstrapped return paths
        """
        returns_array = returns.values
        n_historical = len(returns_array)
        block_size = self.config.block_size

        # Calculate number of blocks needed
        n_blocks_needed = int(np.ceil(n_periods / block_size))

        paths = np.zeros((n_simulations, n_periods))

        for i in range(n_simulations):
            path = []
            for _ in range(n_blocks_needed):
                # Sample a random starting point
                start_idx = np.random.randint(0, n_historical - block_size + 1)
                block = returns_array[start_idx : start_idx + block_size]
                path.extend(block)

            # Trim to exact length if needed
            paths[i] = path[:n_periods]

        return paths

    def _parametric_bootstrap_returns(
        self, returns: pd.Series, n_simulations: int, n_periods: int
    ) -> np.ndarray:
        """
        Parametric bootstrap assuming normal distribution.

        Args:
            returns: Historical returns
            n_simulations: Number of simulation paths
            n_periods: Number of periods to simulate

        Returns:
            Array of parametrically bootstrapped return paths
        """
        mean_return = returns.mean()
        std_return = returns.std()

        paths = np.random.normal(
            mean_return, std_return, size=(n_simulations, n_periods)
        )

        return paths

    def _calculate_portfolio_values(
        self, return_paths: np.ndarray, initial_capital: float
    ) -> pd.DataFrame:
        """
        Calculate portfolio values from return paths.

        Args:
            return_paths: Array of return paths
            initial_capital: Starting portfolio value

        Returns:
            DataFrame with portfolio value paths
        """
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + return_paths, axis=1)

        # Calculate portfolio values
        portfolio_values = cumulative_returns * initial_capital

        # Create DataFrame with proper indexing
        n_periods = portfolio_values.shape[1]
        dates = pd.date_range(
            start=pd.Timestamp.now().date(), periods=n_periods, freq="D"
        )

        df = pd.DataFrame(
            portfolio_values.T,
            index=dates,
            columns=[f"Path_{i}" for i in range(portfolio_values.shape[0])],
        )

        return df

    def calculate_percentiles(
        self, confidence_levels: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Calculate percentile bands for the simulated paths.

        Args:
            confidence_levels: List of percentiles to calculate

        Returns:
            DataFrame with percentile values over time
        """
        if self.simulated_paths is None:
            raise ValueError(
                "Must run simulation first. Call simulate_portfolio_paths()"
            )

        confidence_levels = confidence_levels or self.config.confidence_levels

        # Calculate percentiles across all paths for each time period
        percentiles = {}
        for level in confidence_levels:
            percentile_values = np.percentile(
                self.simulated_paths.values, level * 100, axis=1
            )
            percentiles[f"P{int(level * 100)}"] = percentile_values

        # Add mean path
        percentiles["Mean"] = self.simulated_paths.mean(axis=1)

        # Create DataFrame
        percentile_df = pd.DataFrame(percentiles, index=self.simulated_paths.index)

        self.percentiles = percentile_df
        return percentile_df

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Calculate summary statistics from the simulation.

        Returns:
            Dictionary with summary statistics
        """
        if self.simulated_paths is None:
            raise ValueError(
                "Must run simulation first. Call simulate_portfolio_paths()"
            )

        # Final values for all paths
        final_values = self.simulated_paths.iloc[-1]
        initial_capital = self.results["initial_capital"]

        # Calculate statistics - Safely calculate total returns with division-by-zero protection
        if initial_capital > 1e-10:
            total_returns = (final_values - initial_capital) / initial_capital
        else:
            total_returns = pd.Series(0.0, index=final_values.index)

        stats = {
            "mean_final_value": final_values.mean(),
            "std_final_value": final_values.std(),
            "mean_total_return": total_returns.mean(),
            "std_total_return": total_returns.std(),
            "min_final_value": final_values.min(),
            "max_final_value": final_values.max(),
            "min_total_return": total_returns.min(),
            "max_total_return": total_returns.max(),
            "var_95": np.percentile(total_returns, 5),  # 95% VaR
            "var_99": np.percentile(total_returns, 1),  # 99% VaR
            "cvar_95": total_returns[
                total_returns <= np.percentile(total_returns, 5)
            ].mean(),
            "cvar_99": total_returns[
                total_returns <= np.percentile(total_returns, 1)
            ].mean(),
            "probability_of_loss": (total_returns < 0).mean(),
            "probability_of_20_percent_loss": (total_returns < -0.20).mean(),
            "probability_of_50_percent_loss": (total_returns < -0.50).mean(),
        }

        # Add percentile statistics
        if self.percentiles is not None:
            for col in self.percentiles.columns:
                if col.startswith("P"):
                    final_percentile = self.percentiles[col].iloc[-1]
                    stats[f"final_{col.lower()}"] = final_percentile
                    stats[f"return_{col.lower()}"] = (
                        final_percentile - initial_capital
                    ) / initial_capital

        return stats

    def plot_simulation_results(
        self,
        figsize: Tuple[int, int] = (12, 8),
        show_paths: bool = True,
        n_paths_to_show: int = 50,
        alpha_paths: float = 0.1,
        confidence_bands: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive visualization of the Monte Carlo simulation results.

        Args:
            figsize: Figure size
            show_paths: Whether to show individual paths
            n_paths_to_show: Number of individual paths to display
            alpha_paths: Transparency for individual paths
            confidence_bands: Whether to show confidence bands
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        if self.simulated_paths is None:
            raise ValueError(
                "Must run simulation first. Call simulate_portfolio_paths()"
            )

        # Calculate percentiles if not already done
        if self.percentiles is None:
            self.calculate_percentiles()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

        # Plot 1: Portfolio Value Paths with Confidence Bands
        if show_paths:
            # Show subset of individual paths
            paths_to_show = min(n_paths_to_show, self.simulated_paths.shape[1])
            for i in range(paths_to_show):
                ax1.plot(
                    self.simulated_paths.index,
                    self.simulated_paths.iloc[:, i],
                    color="lightgray",
                    alpha=alpha_paths,
                    linewidth=0.5,
                )

        # Plot confidence bands
        if confidence_bands:
            # 5th percentile (lower bound)
            ax1.fill_between(
                self.percentiles.index,
                self.percentiles["P5"],
                self.percentiles["P95"],
                alpha=0.3,
                color="blue",
                label="90% Confidence Interval",
            )

            # 50th percentile (median)
            ax1.plot(
                self.percentiles.index,
                self.percentiles["P50"],
                color="red",
                linewidth=2,
                label="Median (50th percentile)",
            )

            # Mean path
            ax1.plot(
                self.percentiles.index,
                self.percentiles["Mean"],
                color="green",
                linewidth=2,
                linestyle="--",
                label="Mean Path",
            )

        ax1.set_title(
            "Monte Carlo Simulation: Portfolio Value Paths",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Return Distribution at Final Period
        final_returns = (
            self.simulated_paths.iloc[-1] - self.results["initial_capital"]
        ) / self.results["initial_capital"]

        ax2.hist(final_returns, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        ax2.axvline(
            final_returns.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {final_returns.mean():.2%}",
        )
        ax2.axvline(
            np.percentile(final_returns, 5),
            color="orange",
            linestyle="--",
            label=f"5th percentile: {np.percentile(final_returns, 5):.2%}",
        )
        ax2.axvline(
            np.percentile(final_returns, 95),
            color="orange",
            linestyle="--",
            label=f"95th percentile: {np.percentile(final_returns, 95):.2%}",
        )

        ax2.set_title("Distribution of Final Returns", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Total Return", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")

        return fig

    def create_detailed_report(self) -> Dict[str, any]:
        """
        Create a detailed report of the Monte Carlo simulation.

        Returns:
            Dictionary containing comprehensive simulation results
        """
        if self.simulated_paths is None:
            raise ValueError(
                "Must run simulation first. Call simulate_portfolio_paths()"
            )

        # Calculate percentiles if not already done
        if self.percentiles is None:
            self.calculate_percentiles()

        # Get summary statistics
        stats = self.get_summary_statistics()

        # Create detailed report
        report = {
            "simulation_config": {
                "n_simulations": self.results["n_simulations"],
                "n_periods": self.results["n_periods"],
                "initial_capital": self.results["initial_capital"],
                "bootstrap_method": self.config.bootstrap_method,
                "confidence_levels": self.config.confidence_levels,
            },
            "summary_statistics": stats,
            "percentile_analysis": {
                "final_values": self.percentiles.iloc[-1].to_dict(),
                "max_drawdowns": self._calculate_max_drawdowns(),
                "volatility_analysis": self._calculate_volatility_analysis(),
            },
            "risk_metrics": {
                "var_95": stats["var_95"],
                "var_99": stats["var_99"],
                "cvar_95": stats["cvar_95"],
                "cvar_99": stats["cvar_99"],
                "probability_of_loss": stats["probability_of_loss"],
            },
        }

        return report

    def _calculate_max_drawdowns(self) -> Dict[str, float]:
        """Calculate maximum drawdowns for each percentile path."""
        drawdowns = {}

        for col in self.percentiles.columns:
            if col.startswith("P") or col == "Mean":
                equity_curve = self.percentiles[col]
                # Use safe drawdown utility
                from trading.utils.safe_math import safe_drawdown
                drawdown = safe_drawdown(equity_curve)
                drawdowns[f"max_drawdown_{col}"] = float(drawdown.min())

        return drawdowns

    def _calculate_volatility_analysis(self) -> Dict[str, float]:
        """Calculate volatility analysis for the simulation."""
        # Calculate daily returns for all paths
        daily_returns = self.simulated_paths.pct_change().dropna()

        # Calculate volatility for each path
        path_volatilities = daily_returns.std(axis=0)

        return {
            "mean_volatility": path_volatilities.mean(),
            "volatility_std": path_volatilities.std(),
            "min_volatility": path_volatilities.min(),
            "max_volatility": path_volatilities.max(),
            "volatility_5th_percentile": np.percentile(path_volatilities, 5),
            "volatility_95th_percentile": np.percentile(path_volatilities, 95),
        }


def run_monte_carlo_analysis(
    returns: pd.Series,
    initial_capital: float = 10000.0,
    n_simulations: int = 1000,
    bootstrap_method: str = "historical",
    confidence_levels: List[float] = None,
    plot_results: bool = True,
    save_plot: Optional[str] = None,
    return_simulator: bool = False,
) -> Union[Dict[str, any], Tuple[Dict[str, any], MonteCarloSimulator]]:
    """
    Convenience function to run a complete Monte Carlo analysis.

    Args:
        returns: Historical returns series
        initial_capital: Starting portfolio value
        n_simulations: Number of simulation paths
        bootstrap_method: Bootstrap method to use
        confidence_levels: Percentiles to calculate
        plot_results: Whether to create visualization
        save_plot: Optional path to save plot
        return_simulator: Whether to return the simulator object

    Returns:
        Dictionary with analysis results (and optionally the simulator)
    """
    # Create configuration
    config = MonteCarloConfig(
        n_simulations=n_simulations,
        bootstrap_method=bootstrap_method,
        confidence_levels=confidence_levels,
        initial_capital=initial_capital,
    )

    # Create and run simulator
    simulator = MonteCarloSimulator(config)

    # Run simulation
    simulator.simulate_portfolio_paths(returns, initial_capital, n_simulations)

    # Calculate percentiles
    simulator.calculate_percentiles()

    # Create report
    report = simulator.create_detailed_report()

    # Create plot if requested
    if plot_results:
        simulator.plot_simulation_results(save_path=save_plot)

    if return_simulator:
        return report, simulator
    else:
        return report


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    sample_returns = pd.Series(
        np.random.normal(0.0005, 0.02, 252),  # Daily returns with 0.05% mean, 2% std
        index=dates,
    )

    # Run Monte Carlo analysis
    results = run_monte_carlo_analysis(
        returns=sample_returns,
        initial_capital=10000.0,
        n_simulations=1000,
        bootstrap_method="historical",
        plot_results=True,
    )

    print("Monte Carlo Analysis Results:")
    print("=" * 50)
    print(
        f"Mean Final Value: ${results['summary_statistics']['mean_final_value']:,.2f}"
    )
    print(
        f"5th Percentile Final Value: ${results['summary_statistics']['final_p5']:,.2f}"
    )
    print(
        f"95th Percentile Final Value: ${results['summary_statistics']['final_p95']:,.2f}"
    )
    print(
        f"Probability of Loss: {results['summary_statistics']['probability_of_loss']:.2%}"
    )
    print(f"95% VaR: {results['summary_statistics']['var_95']:.2%}")
    print(f"95% CVaR: {results['summary_statistics']['cvar_95']:.2%}")
