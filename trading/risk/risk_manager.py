"""Enhanced risk management module with comprehensive metrics and safeguards.
Enhanced with Batch 11 features: dynamic volatility model integration (rolling std and GARCH)
to adjust position size.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError as e:
    print("⚠️ plotly not available. Disabling interactive plotting capabilities.")
    print(f"   Missing: {e}")
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

# Try to import scipy
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError as e:
    print("⚠️ scipy not available. Disabling optimization-based risk management.")
    print(f"   Missing: {e}")
    minimize = None
    SCIPY_AVAILABLE = False

# Try to import GARCH models
try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False


class VolatilityModel(Enum):
    """Volatility model types."""
    ROLLING_STD = "rolling_std"
    EWMA = "ewma"
    GARCH = "garch"
    HYBRID = "hybrid"


@dataclass
class VolatilityForecast:
    """Volatility forecast data class."""
    
    timestamp: str
    current_volatility: float
    forecasted_volatility: float
    confidence_interval: Tuple[float, float]
    model_type: str
    model_parameters: Dict[str, Any]
    forecast_horizon: int


@dataclass
class PositionSizeRecommendation:
    """Position size recommendation data class."""
    
    timestamp: str
    symbol: str
    base_position_size: float
    volatility_adjusted_size: float
    risk_adjusted_size: float
    final_recommendation: float
    volatility_model: str
    confidence_score: float
    risk_factors: Dict[str, float]


@dataclass
class RiskMetrics:
    """Risk metrics data class."""

    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation: float
    kelly_fraction: float
    expected_shortfall: float
    tail_risk: float
    skewness: float
    kurtosis: float
    timestamp: str


@dataclass
class StressTestResult:
    """Stress test result data class."""

    scenario_name: str
    portfolio_value_change: float
    var_change: float
    cvar_change: float
    max_drawdown_change: float
    sharpe_ratio_change: float
    timestamp: str


class DynamicVolatilityModel:
    """Dynamic volatility model for position sizing."""
    
    def __init__(
        self, 
        model_type: VolatilityModel = VolatilityModel.ROLLING_STD,
        window: int = 252,
        alpha: float = 0.94,
        garch_order: Tuple[int, int] = (1, 1)
    ):
        """Initialize dynamic volatility model.
        
        Args:
            model_type: Type of volatility model
            window: Rolling window size
            alpha: EWMA decay factor
            garch_order: GARCH model order (p, q)
        """
        self.model_type = model_type
        self.window = window
        self.alpha = alpha
        self.garch_order = garch_order
        self.volatility_history = []
        self.forecasts = []
        
        self.logger = logging.getLogger(__name__)
        
        if model_type == VolatilityModel.GARCH and not GARCH_AVAILABLE:
            self.logger.warning("GARCH not available, falling back to rolling std")
            self.model_type = VolatilityModel.ROLLING_STD
    
    def calculate_rolling_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling volatility using standard deviation.
        
        Args:
            returns: Returns series
            
        Returns:
            Rolling volatility series
        """
        return returns.rolling(window=self.window).std() * np.sqrt(252)
    
    def calculate_ewma_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate exponentially weighted moving average volatility.
        
        Args:
            returns: Returns series
            
        Returns:
            EWMA volatility series
        """
        return returns.ewm(alpha=self.alpha).std() * np.sqrt(252)
    
    def calculate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate GARCH volatility.
        
        Args:
            returns: Returns series
            
        Returns:
            GARCH volatility series
        """
        if not GARCH_AVAILABLE:
            self.logger.warning("GARCH not available, using rolling std")
            return self.calculate_rolling_volatility(returns)
        
        try:
            # Fit GARCH model
            model = arch_model(
                returns * 100,  # Convert to percentage
                vol='Garch', 
                p=self.garch_order[0], 
                q=self.garch_order[1]
            )
            results = model.fit(disp='off')
            
            # Get conditional volatility
            conditional_vol = results.conditional_volatility / 100  # Convert back
            return pd.Series(conditional_vol, index=returns.index) * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"GARCH model fitting failed: {e}")
            return self.calculate_rolling_volatility(returns)
    
    def calculate_hybrid_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate hybrid volatility combining multiple models.
        
        Args:
            returns: Returns series
            
        Returns:
            Hybrid volatility series
        """
        rolling_vol = self.calculate_rolling_volatility(returns)
        ewma_vol = self.calculate_ewma_volatility(returns)
        
        # Combine with equal weights
        hybrid_vol = 0.5 * rolling_vol + 0.5 * ewma_vol
        
        # Add GARCH if available
        if GARCH_AVAILABLE:
            try:
                garch_vol = self.calculate_garch_volatility(returns)
                # Use GARCH for recent periods, hybrid for older
                recent_mask = returns.index >= returns.index[-min(100, len(returns))]
                hybrid_vol[recent_mask] = 0.4 * rolling_vol[recent_mask] + 0.3 * ewma_vol[recent_mask] + 0.3 * garch_vol[recent_mask]
            except Exception as e:
                self.logger.warning(f"GARCH calculation failed in hybrid volatility: {e}")
                # Continue with rolling and EWMA only
        
        return hybrid_vol
    
    def forecast_volatility(
        self, 
        returns: pd.Series, 
        horizon: int = 5
    ) -> VolatilityForecast:
        """Forecast volatility for the next period.
        
        Args:
            returns: Returns series
            horizon: Forecast horizon in days
            
        Returns:
            Volatility forecast
        """
        try:
            if self.model_type == VolatilityModel.ROLLING_STD:
                current_vol = self.calculate_rolling_volatility(returns).iloc[-1]
                forecast_vol = current_vol  # Simple persistence
            
            elif self.model_type == VolatilityModel.EWMA:
                current_vol = self.calculate_ewma_volatility(returns).iloc[-1]
                forecast_vol = current_vol  # EWMA persistence
            
            elif self.model_type == VolatilityModel.GARCH:
                if GARCH_AVAILABLE:
                    model = arch_model(
                        returns * 100, 
                        vol='Garch', 
                        p=self.garch_order[0], 
                        q=self.garch_order[1]
                    )
                    results = model.fit(disp='off')
                    forecast = results.forecast(horizon=horizon)
                    current_vol = results.conditional_volatility.iloc[-1] / 100 * np.sqrt(252)
                    forecast_vol = forecast.variance.iloc[-1] ** 0.5 / 100 * np.sqrt(252)
                else:
                    current_vol = self.calculate_rolling_volatility(returns).iloc[-1]
                    forecast_vol = current_vol
            
            elif self.model_type == VolatilityModel.HYBRID:
                current_vol = self.calculate_hybrid_volatility(returns).iloc[-1]
                # Simple trend-based forecast
                recent_vol = self.calculate_hybrid_volatility(returns).tail(20)
                trend = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
                forecast_vol = current_vol + trend * horizon
            
            else:
                current_vol = self.calculate_rolling_volatility(returns).iloc[-1]
                forecast_vol = current_vol
            
            # Calculate confidence interval
            vol_series = self.calculate_rolling_volatility(returns).dropna()
            vol_std = vol_series.std()
            confidence_interval = (
                max(0, forecast_vol - 1.96 * vol_std),
                forecast_vol + 1.96 * vol_std
            )
            
            forecast = VolatilityForecast(
                timestamp=datetime.now().isoformat(),
                current_volatility=current_vol,
                forecasted_volatility=forecast_vol,
                confidence_interval=confidence_interval,
                model_type=self.model_type.value,
                model_parameters={
                    "window": self.window,
                    "alpha": self.alpha,
                    "garch_order": self.garch_order
                },
                forecast_horizon=horizon
            )
            
            self.forecasts.append(forecast)
            return forecast
            
        except Exception as e:
            self.logger.error(f"Volatility forecasting failed: {e}")
            # Return simple forecast
            current_vol = returns.std() * np.sqrt(252)
            return VolatilityForecast(
                timestamp=datetime.now().isoformat(),
                current_volatility=current_vol,
                forecasted_volatility=current_vol,
                confidence_interval=(current_vol * 0.8, current_vol * 1.2),
                model_type="fallback",
                model_parameters={},
                forecast_horizon=horizon
            )


class RiskManager:
    """Enhanced risk management module with comprehensive metrics and safeguards."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk manager.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize metrics
        self.returns = None
        self.benchmark_returns = None
        self.metrics_history = []
        self.current_metrics = None

        # Initialize dynamic volatility model
        volatility_config = self.config.get("volatility_model", {})
        self.volatility_model = DynamicVolatilityModel(
            model_type=VolatilityModel(volatility_config.get("type", "rolling_std")),
            window=volatility_config.get("window", 252),
            alpha=volatility_config.get("alpha", 0.94),
            garch_order=volatility_config.get("garch_order", (1, 1))
        )
        
        # Position sizing parameters
        self.position_config = self.config.get("position_sizing", {
            "max_position_size": 0.1,  # 10% of portfolio
            "volatility_scaling": True,
            "risk_adjustment": True,
            "confidence_threshold": 0.7
        })

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add file handler for debug logs if no handlers exist
        if not self.logger.handlers:
            try:
                os.makedirs("trading/risk/logs", exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create trading/risk/logs: {e}")
            try:
                os.makedirs("trading/risk/results", exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create trading/risk/results: {e}")

            debug_handler = logging.FileHandler("trading/risk/logs/risk_debug.log")
            debug_handler.setLevel(logging.DEBUG)
            debug_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            debug_handler.setFormatter(debug_formatter)
            self.logger.addHandler(debug_handler)

        self.logger.info(f"Initialized RiskManager with config: {self.config}")

    def update_returns(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
    ) -> None:
        """Update returns data.

        Args:
            returns: Returns series
            benchmark_returns: Optional benchmark returns series
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns

        # Calculate metrics
        self._calculate_metrics()

        self.logger.info("Updated returns and recalculated metrics")

    def _calculate_metrics(self) -> None:
        """Calculate risk metrics."""
        if self.returns is None or self.returns.empty:
            self.logger.warning("No returns data available for metric calculation")

        # Calculate basic metrics
        volatility = self.returns.std() * np.sqrt(252)
        excess_returns = self.returns - (self.config.get("risk_free_rate", 0.02) / 252)

        # Calculate Sharpe ratio
        sharpe_ratio = (
            np.sqrt(252) * excess_returns.mean() / volatility if volatility != 0 else 0
        )

        # Calculate Sortino ratio
        downside_returns = self.returns[self.returns < 0]
        downside_vol = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        )
        sortino_ratio = (
            np.sqrt(252) * excess_returns.mean() / downside_vol
            if downside_vol != 0
            else 0
        )

        # Calculate VaR and CVaR at different confidence levels
        var_95 = np.percentile(self.returns, 5)
        cvar_95 = self.returns[self.returns <= var_95].mean()
        var_99 = np.percentile(self.returns, 1)
        cvar_99 = self.returns[self.returns <= var_99].mean()

        # Calculate Expected Shortfall (same as CVaR)
        expected_shortfall = cvar_95

        # Calculate Tail Risk (probability of extreme losses)
        tail_threshold = np.percentile(self.returns, 1)
        tail_risk = len(self.returns[self.returns <= tail_threshold]) / len(
            self.returns
        )

        # Calculate higher moments
        skewness = self.returns.skew()
        kurtosis = self.returns.kurtosis()

        # Calculate maximum drawdown
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calculate beta and correlation
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            beta = (
                self.returns.cov(self.benchmark_returns) / self.benchmark_returns.var()
            )
            correlation = self.returns.corr(self.benchmark_returns)
        else:
            beta = 1.0
            correlation = 0.0

        # Calculate Kelly fraction
        win_rate = len(self.returns[self.returns > 0]) / len(self.returns)
        avg_win = (
            self.returns[self.returns > 0].mean()
            if len(self.returns[self.returns > 0]) > 0
            else 0
        )
        avg_loss = (
            abs(self.returns[self.returns < 0].mean())
            if len(self.returns[self.returns < 0]) > 0
            else 0
        )
        kelly_fraction = (
            (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            if avg_win != 0
            else 0
        )

        # Create metrics object
        self.current_metrics = RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            correlation=correlation,
            kelly_fraction=kelly_fraction,
            expected_shortfall=expected_shortfall,
            tail_risk=tail_risk,
            skewness=skewness,
            kurtosis=kurtosis,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Add to history
        self.metrics_history.append(self.current_metrics)

        self.logger.info(f"Calculated metrics: {self.current_metrics}")

    def calculate_historical_var(
        self, confidence_level: float = 0.95, window: int = 252
    ) -> float:
        """Calculate historical VaR.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            window: Rolling window size

        Returns:
            Historical VaR value
        """
        if self.returns is None or self.returns.empty:
            return 0.0

        percentile = (1 - confidence_level) * 100
        return np.percentile(self.returns, percentile)

    def calculate_parametric_var(
        self, confidence_level: float = 0.95, window: int = 252
    ) -> float:
        """Calculate parametric VaR assuming normal distribution.

        Args:
            confidence_level: Confidence level
            window: Rolling window size

        Returns:
            Parametric VaR value
        """
        if self.returns is None or self.returns.empty:
            return 0.0

        # Use rolling window for volatility
        rolling_vol = self.returns.rolling(window=window).std().iloc[-1]
        mean_return = self.returns.rolling(window=window).mean().iloc[-1]

        # Z-score for confidence level
        from scipy.stats import norm

        z_score = norm.ppf(1 - confidence_level)

        return mean_return + z_score * rolling_vol

    def calculate_monte_carlo_var(
        self,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        time_horizon: int = 1,
    ) -> float:
        """Calculate VaR using Monte Carlo simulation.

        Args:
            confidence_level: Confidence level
            n_simulations: Number of simulations
            time_horizon: Time horizon in days

        Returns:
            Monte Carlo VaR value
        """
        if self.returns is None or self.returns.empty:
            return 0.0

        # Estimate parameters
        mean_return = self.returns.mean()
        volatility = self.returns.std()

        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            volatility * np.sqrt(time_horizon),
            n_simulations,
        )

        # Calculate VaR
        percentile = (1 - confidence_level) * 100
        return np.percentile(simulated_returns, percentile)

    def run_stress_tests(
        self, portfolio_value: float = 1000000.0
    ) -> List[StressTestResult]:
        """Run comprehensive stress tests.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            List of stress test results
        """
        if self.returns is None or self.returns.empty:
            self.logger.warning("No returns data available for stress testing")
            return []

        stress_scenarios = {
            "Market Crash": {"return_shock": -0.20, "volatility_multiplier": 2.0},
            "Interest Rate Hike": {"return_shock": -0.10, "volatility_multiplier": 1.5},
            "Liquidity Crisis": {"return_shock": -0.15, "volatility_multiplier": 2.5},
            "Currency Crisis": {"return_shock": -0.25, "volatility_multiplier": 3.0},
            "Geopolitical Risk": {"return_shock": -0.12, "volatility_multiplier": 1.8},
            "Economic Recession": {"return_shock": -0.18, "volatility_multiplier": 2.2},
            "Technology Bubble Burst": {
                "return_shock": -0.30,
                "volatility_multiplier": 2.8,
            },
            "Oil Price Shock": {"return_shock": -0.08, "volatility_multiplier": 1.6},
        }

        results = []
        base_metrics = self.current_metrics

        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply stress scenario
            stressed_returns = (
                self.returns * scenario_params["volatility_multiplier"]
                + scenario_params["return_shock"]
            )

            # Calculate stressed metrics
            stressed_volatility = stressed_returns.std() * np.sqrt(252)
            stressed_var_95 = np.percentile(stressed_returns, 5)
            stressed_cvar_95 = stressed_returns[
                stressed_returns <= stressed_var_95
            ].mean()

            # Calculate portfolio value change
            portfolio_return = stressed_returns.mean() * 252
            portfolio_value_change = portfolio_value * portfolio_return

            # Calculate drawdown change
            cum_returns = (1 + stressed_returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            stressed_max_drawdown = drawdowns.min()

            # Calculate Sharpe ratio change
            risk_free_rate = self.config.get("risk_free_rate", 0.02)
            excess_returns = stressed_returns - (risk_free_rate / 252)
            stressed_sharpe = (
                np.sqrt(252) * excess_returns.mean() / stressed_volatility
                if stressed_volatility != 0
                else 0
            )

            result = StressTestResult(
                scenario_name=scenario_name,
                portfolio_value_change=portfolio_value_change,
                var_change=stressed_var_95 - base_metrics.var_95,
                cvar_change=stressed_cvar_95 - base_metrics.cvar_95,
                max_drawdown_change=stressed_max_drawdown - base_metrics.max_drawdown,
                sharpe_ratio_change=stressed_sharpe - base_metrics.sharpe_ratio,
                timestamp=datetime.utcnow().isoformat(),
            )
            results.append(result)

        self.logger.info(f"Completed {len(results)} stress test scenarios")
        return results

    def calculate_conditional_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR).

        Args:
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        if self.returns is None or self.returns.empty:
            return 0.0

        var_threshold = np.percentile(self.returns, (1 - confidence_level) * 100)
        tail_returns = self.returns[self.returns <= var_threshold]

        return tail_returns.mean() if len(tail_returns) > 0 else 0.0

    def calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (ES).

        Args:
            confidence_level: Confidence level

        Returns:
            Expected Shortfall value
        """
        return self.calculate_conditional_var(confidence_level)

    def get_position_limits(self) -> Dict[str, float]:
        """Get position size limits based on risk metrics.

        Returns:
            Dictionary of position limits
        """
        if self.returns is None or self.returns.empty:
            self.logger.warning("No returns data available for position limits")
            return {}

        if self.current_metrics is None:
            self.logger.warning("No metrics available for position limits")
            return {}

        # Get base limits from config
        max_position_size = self.config.get("max_position_size", 0.2)
        max_leverage = self.config.get("max_leverage", 1.0)

        # Adjust based on risk metrics
        volatility_factor = 1.0 / (1.0 + self.current_metrics.volatility)
        drawdown_factor = 1.0 + self.current_metrics.max_drawdown
        kelly_factor = min(self.current_metrics.kelly_fraction, 0.5)  # Cap at 0.5

        # Calculate final limits
        position_limit = (
            max_position_size * volatility_factor * drawdown_factor * kelly_factor
        )
        leverage_limit = max_leverage * volatility_factor * drawdown_factor

        return {
            "position_limit": position_limit,
            "leverage_limit": leverage_limit,
            "kelly_fraction": kelly_factor,
        }

    def optimize_position_sizes(
        self, expected_returns: pd.Series, covariance: pd.DataFrame
    ) -> pd.Series:
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available. Using equal position sizes.")
            return pd.Series(1.0 / len(expected_returns), index=expected_returns.index)
        """Optimize position sizes using SLSQP.

        Args:
            expected_returns: Expected returns series
            covariance: Covariance matrix

        Returns:
            Optimized position sizes
        """
        n_assets = len(expected_returns)

        # Define objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            return {
                "success": True,
                "result": {
                    "success": True,
                    "result": -portfolio_return / portfolio_vol
                    if portfolio_vol != 0
                    else 0,
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                },
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }

        # Define constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}  # weights sum to 1
        ]

        # Define bounds
        bounds = [(0, 1) for _ in range(n_assets)]

        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        return pd.Series(result.x, index=expected_returns.index)

    def plot_risk_metrics(self) -> go.Figure:
        if not PLOTLY_AVAILABLE:
            self.logger.warning("plotly not available. Cannot create interactive plots.")
            return None
        """Plot time series of risk metrics.

        Returns:
            Plotly figure
        """
        if not self.metrics_history:
            self.logger.warning("No metrics history available for plotting")

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Sharpe Ratio", "Volatility", "Drawdown"),
            vertical_spacing=0.1,
        )

        # Extract metrics
        timestamps = [m.timestamp for m in self.metrics_history]
        sharpe_ratios = [m.sharpe_ratio for m in self.metrics_history]
        volatilities = [m.volatility for m in self.metrics_history]
        drawdowns = [m.max_drawdown for m in self.metrics_history]

        # Add traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=sharpe_ratios, name="Sharpe Ratio"), row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=timestamps, y=volatilities, name="Volatility"), row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=timestamps, y=drawdowns, name="Drawdown"), row=3, col=1
        )

        # Update layout
        fig.update_layout(
            height=900, showlegend=True, title_text="Risk Metrics History"
        )

        return fig

    def cleanup_old_results(self, max_files: int = 5) -> Dict[str, Any]:
        """Clean up old result files.

        Args:
            max_files: Maximum number of files to retain

        Returns:
            Dictionary with cleanup status and details
        """
        try:
            results_dir = "trading/risk/results"

            # Get all result files
            result_files = []
            for file in os.listdir(results_dir):
                if file.endswith((".json", ".csv")):
                    path = os.path.join(results_dir, file)
                    result_files.append((path, os.path.getmtime(path)))

            # Sort by modification time
            result_files.sort(key=lambda x: x[1], reverse=True)

            # Remove old files
            removed_files = []
            for path, _ in result_files[max_files:]:
                try:
                    os.remove(path)
                    removed_files.append(path)
                    self.logger.info(f"Removed old result file: {path}")
                except Exception as e:
                    self.logger.error(f"Failed to remove file {path}: {e}")

            return {
                "success": True,
                "message": f"Cleanup completed. Removed {len(removed_files)} files",
                "removed_files": removed_files,
                "files_retained": len(result_files) - len(removed_files),
                "max_files": max_files,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                "success": False,
                "message": f"Error during cleanup: {str(e)}",
                "max_files": max_files,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def export_risk_report(self, filepath: str) -> Dict[str, Any]:
        """Export risk report to file.

        Args:
            filepath: Output file path

        Returns:
            Dictionary with export status and details
        """
        try:
            if not self.metrics_history:
                return {
                    "success": False,
                    "message": "No metrics history available for export",
                    "filepath": filepath,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Calculate additional risk measures
            historical_var_95 = self.calculate_historical_var(0.95)
            historical_var_99 = self.calculate_historical_var(0.99)
            parametric_var_95 = self.calculate_parametric_var(0.95)
            monte_carlo_var_95 = self.calculate_monte_carlo_var(0.95)

            # Run stress tests
            stress_test_results = self.run_stress_tests()

            # Generate risk summary
            risk_summary = self.generate_risk_summary()

            # Prepare comprehensive data
            report_data = {
                "config": self.config,
                "metrics_history": [asdict(m) for m in self.metrics_history],
                "current_metrics": asdict(self.current_metrics)
                if self.current_metrics
                else None,
                "additional_risk_measures": {
                    "historical_var_95": historical_var_95,
                    "historical_var_99": historical_var_99,
                    "parametric_var_95": parametric_var_95,
                    "monte_carlo_var_95": monte_carlo_var_95,
                    "conditional_var_95": self.calculate_conditional_var(0.95),
                    "expected_shortfall_95": self.calculate_expected_shortfall(0.95),
                },
                "stress_test_results": [
                    asdict(result) for result in stress_test_results
                ],
                "risk_summary": risk_summary,
                "returns_summary": {
                    "mean": float(self.returns.mean()),
                    "std": float(self.returns.std()),
                    "min": float(self.returns.min()),
                    "max": float(self.returns.max()),
                    "skewness": float(self.returns.skew()),
                    "kurtosis": float(self.returns.kurtosis()),
                }
                if self.returns is not None
                else None,
                "data_summary": {
                    "total_observations": len(self.returns)
                    if self.returns is not None
                    else 0,
                    "date_range": {
                        "start": self.returns.index[0].isoformat()
                        if self.returns is not None and len(self.returns) > 0
                        else None,
                        "end": self.returns.index[-1].isoformat()
                        if self.returns is not None and len(self.returns) > 0
                        else None,
                    },
                },
            }

            # Export based on file extension
            if filepath.endswith(".json"):
                with open(filepath, "w") as f:
                    json.dump(report_data, f, indent=2)
            elif filepath.endswith(".csv"):
                # Convert metrics history to DataFrame
                df = pd.DataFrame([asdict(m) for m in self.metrics_history])
                df.to_csv(filepath, index=False)
            else:
                return {
                    "success": False,
                    "message": "Unsupported file format. Use .json or .csv",
                    "filepath": filepath,
                    "timestamp": datetime.utcnow().isoformat(),
                }

            self.logger.info(f"Exported risk report to {filepath}")

            return {
                "success": True,
                "message": f"Risk report exported successfully to {filepath}",
                "filepath": filepath,
                "file_size": os.path.getsize(filepath),
                "metrics_count": len(self.metrics_history),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error exporting risk report: {e}")
            return {
                "success": False,
                "message": f"Error exporting risk report: {str(e)}",
                "filepath": filepath,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def calculate_dynamic_position_size(
        self, 
        symbol: str,
        base_position_size: float,
        expected_return: float = 0.0,
        current_volatility: Optional[float] = None,
        confidence_score: float = 0.5
    ) -> PositionSizeRecommendation:
        """Calculate dynamic position size based on volatility model.
        
        Args:
            symbol: Trading symbol
            base_position_size: Base position size
            expected_return: Expected return
            current_volatility: Current volatility (if None, will be calculated)
            confidence_score: Confidence score for the signal
            
        Returns:
            Position size recommendation
        """
        try:
            if self.returns is None or self.returns.empty:
                self.logger.warning("No returns data available for position sizing")
                return PositionSizeRecommendation(
                    timestamp=datetime.now().isoformat(),
                    symbol=symbol,
                    base_position_size=base_position_size,
                    volatility_adjusted_size=base_position_size,
                    risk_adjusted_size=base_position_size,
                    final_recommendation=base_position_size,
                    volatility_model="none",
                    confidence_score=confidence_score,
                    risk_factors={}
                )
            
            # Get volatility forecast
            volatility_forecast = self.volatility_model.forecast_volatility(self.returns)
            
            # Calculate volatility-adjusted position size
            if self.position_config.get("volatility_scaling", True):
                # Inverse volatility scaling
                target_volatility = self.config.get("target_volatility", 0.15)
                volatility_ratio = target_volatility / volatility_forecast.forecasted_volatility
                volatility_adjusted_size = base_position_size * volatility_ratio
            else:
                volatility_adjusted_size = base_position_size
            
            # Calculate risk-adjusted position size
            if self.position_config.get("risk_adjustment", True):
                # Kelly criterion adjustment
                kelly_fraction = self.current_metrics.kelly_fraction if self.current_metrics else 0.0
                risk_adjusted_size = volatility_adjusted_size * max(0, kelly_fraction)
                
                # Confidence adjustment
                confidence_threshold = self.position_config.get("confidence_threshold", 0.7)
                if confidence_score < confidence_threshold:
                    risk_adjusted_size *= confidence_score / confidence_threshold
            else:
                risk_adjusted_size = volatility_adjusted_size
            
            # Apply position limits
            max_position = self.position_config.get("max_position_size", 0.1)
            final_recommendation = min(risk_adjusted_size, max_position)
            
            # Calculate risk factors
            risk_factors = {
                "volatility_ratio": volatility_ratio if self.position_config.get("volatility_scaling", True) else 1.0,
                "kelly_fraction": self.current_metrics.kelly_fraction if self.current_metrics else 0.0,
                "confidence_adjustment": confidence_score / confidence_threshold if confidence_score < confidence_threshold else 1.0,
                "current_volatility": volatility_forecast.current_volatility,
                "forecasted_volatility": volatility_forecast.forecasted_volatility
            }
            
            recommendation = PositionSizeRecommendation(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                base_position_size=base_position_size,
                volatility_adjusted_size=volatility_adjusted_size,
                risk_adjusted_size=risk_adjusted_size,
                final_recommendation=final_recommendation,
                volatility_model=volatility_forecast.model_type,
                confidence_score=confidence_score,
                risk_factors=risk_factors
            )
            
            self.logger.info(f"Position size calculated for {symbol}: {final_recommendation:.4f}")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return PositionSizeRecommendation(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                base_position_size=base_position_size,
                volatility_adjusted_size=base_position_size,
                risk_adjusted_size=base_position_size,
                final_recommendation=base_position_size,
                volatility_model="error",
                confidence_score=confidence_score,
                risk_factors={"error": str(e)}
            )

    def get_volatility_forecast(self) -> Optional[VolatilityForecast]:
        """Get the latest volatility forecast.
        
        Returns:
            Latest volatility forecast or None
        """
        if self.returns is None or self.returns.empty:
            return None
        
        return self.volatility_model.forecast_volatility(self.returns)

    def update_volatility_model(
        self, 
        model_type: VolatilityModel,
        **kwargs
    ) -> bool:
        """Update volatility model configuration.
        
        Args:
            model_type: New volatility model type
            **kwargs: Additional model parameters
            
        Returns:
            True if update successful
        """
        try:
            self.volatility_model.model_type = model_type
            
            if "window" in kwargs:
                self.volatility_model.window = kwargs["window"]
            if "alpha" in kwargs:
                self.volatility_model.alpha = kwargs["alpha"]
            if "garch_order" in kwargs:
                self.volatility_model.garch_order = kwargs["garch_order"]
            
            self.logger.info(f"Updated volatility model to {model_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating volatility model: {e}")
            return False


__all__ = ["RiskManager", "RiskMetrics"]
