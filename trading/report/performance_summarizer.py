"""
Performance Summarizer

Enhanced with Batch 10 features: configurable risk_free_rate parameter instead of hardcoded value.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class PerformanceSummarizer:
    """Enhanced performance summarizer with configurable risk-free rate."""

    def __init__(self, risk_free_rate: float = 0.01, trading_days_per_year: int = 252):
        """Initialize the performance summarizer.
        
        Args:
            risk_free_rate: Risk-free rate for calculations (default: 0.01 = 1%)
            trading_days_per_year: Number of trading days per year (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.summary_history = []
        
        logger.info(f"PerformanceSummarizer initialized with risk_free_rate={risk_free_rate:.4f}")

    def calculate_performance_metrics(
        self, 
        returns: Union[pd.Series, List[float]], 
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics.
        
        Args:
            returns: Series or list of returns
            risk_free_rate: Optional override for risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Use provided risk_free_rate or default
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            
            # Convert to pandas Series if needed
            if isinstance(returns, list):
                returns = pd.Series(returns)
            
            if returns.empty:
                logger.warning("Empty returns data provided")
                return self._get_empty_metrics()
            
            # Remove NaN values
            returns = returns.dropna()
            
            if len(returns) == 0:
                logger.warning("No valid returns data after removing NaN values")
                return self._get_empty_metrics()
            
            # Calculate basic metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)
            
            # Risk-adjusted metrics
            excess_returns = returns - rf_rate / self.trading_days_per_year
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year) if returns.std() > 0 else 0
            
            # Sortino ratio (using downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Value at Risk and Conditional VaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Beta and Alpha (assuming market returns are available)
            beta = 1.0  # Default value, would be calculated with market data
            alpha = annualized_return - (rf_rate + beta * (0.08 - rf_rate))  # Assuming 8% market return
            
            # Information ratio
            information_ratio = excess_returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Treynor ratio
            treynor_ratio = excess_returns.mean() / beta if beta != 0 else 0
            
            # Win rate and profit factor
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
            profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else float('inf')
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Create metrics dictionary
            metrics = {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "beta": beta,
                "alpha": alpha,
                "information_ratio": information_ratio,
                "treynor_ratio": treynor_ratio,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "risk_free_rate_used": rf_rate,
                "trading_days_per_year": self.trading_days_per_year,
                "data_points": len(returns)
            }
            
            # Store summary
            self._store_summary(metrics, returns)
            
            logger.info(f"Performance metrics calculated successfully for {len(returns)} data points")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._get_empty_metrics()

    def calculate_rolling_metrics(
        self, 
        returns: pd.Series, 
        window: int = 252, 
        risk_free_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            risk_free_rate: Optional override for risk-free rate
            
        Returns:
            DataFrame with rolling metrics
        """
        try:
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            
            if returns.empty:
                logger.warning("Empty returns data provided for rolling metrics")
                return pd.DataFrame()
            
            # Calculate rolling metrics
            rolling_metrics = pd.DataFrame()
            
            # Rolling returns
            rolling_metrics['rolling_return'] = returns.rolling(window).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Rolling volatility
            rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(self.trading_days_per_year)
            
            # Rolling Sharpe ratio
            rolling_metrics['rolling_sharpe'] = returns.rolling(window).apply(
                lambda x: self._calculate_rolling_sharpe(x, rf_rate)
            )
            
            # Rolling maximum drawdown
            rolling_metrics['rolling_max_dd'] = returns.rolling(window).apply(
                self._calculate_rolling_drawdown
            )
            
            # Rolling win rate
            rolling_metrics['rolling_win_rate'] = returns.rolling(window).apply(
                lambda x: (x > 0).mean()
            )
            
            logger.info(f"Rolling metrics calculated with window={window}")
            return rolling_metrics
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()

    def _calculate_rolling_sharpe(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate rolling Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / self.trading_days_per_year
        return excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year) if returns.std() > 0 else 0

    def _calculate_rolling_drawdown(self, returns: pd.Series) -> float:
        """Calculate rolling maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def compare_strategies(
        self, 
        strategy_returns: Dict[str, Union[pd.Series, List[float]]], 
        risk_free_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """Compare multiple strategies.
        
        Args:
            strategy_returns: Dictionary of strategy names to returns
            risk_free_rate: Optional override for risk-free rate
            
        Returns:
            DataFrame with comparison metrics
        """
        try:
            comparison_data = []
            
            for strategy_name, returns in strategy_returns.items():
                metrics = self.calculate_performance_metrics(returns, risk_free_rate)
                metrics['strategy'] = strategy_name
                comparison_data.append(metrics)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.set_index('strategy')
            
            # Sort by Sharpe ratio
            comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
            
            logger.info(f"Strategy comparison completed for {len(strategy_returns)} strategies")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return pd.DataFrame()

    def generate_performance_report(
        self, 
        returns: Union[pd.Series, List[float]], 
        strategy_name: str = "Strategy",
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            returns: Returns data
            strategy_name: Name of the strategy
            risk_free_rate: Optional override for risk-free rate
            
        Returns:
            Dictionary with comprehensive report
        """
        try:
            # Calculate metrics
            metrics = self.calculate_performance_metrics(returns, risk_free_rate)
            
            # Calculate rolling metrics
            if isinstance(returns, list):
                returns = pd.Series(returns)
            rolling_metrics = self.calculate_rolling_metrics(returns, risk_free_rate=risk_free_rate)
            
            # Generate report
            report = {
                "strategy_name": strategy_name,
                "timestamp": datetime.now().isoformat(),
                "risk_free_rate_used": metrics["risk_free_rate_used"],
                "trading_days_per_year": self.trading_days_per_year,
                "data_points": metrics["data_points"],
                "metrics": metrics,
                "rolling_metrics": rolling_metrics.to_dict() if not rolling_metrics.empty else {},
                "summary": self._generate_summary_text(metrics),
                "risk_assessment": self._assess_risk(metrics),
                "recommendations": self._generate_recommendations(metrics)
            }
            
            logger.info(f"Performance report generated for {strategy_name}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {
                "error": str(e),
                "strategy_name": strategy_name,
                "timestamp": datetime.now().isoformat()
            }

    def _generate_summary_text(self, metrics: Dict[str, float]) -> str:
        """Generate summary text from metrics."""
        summary_parts = []
        
        # Performance summary
        total_return_pct = metrics["total_return"] * 100
        annualized_return_pct = metrics["annualized_return"] * 100
        volatility_pct = metrics["volatility"] * 100
        sharpe = metrics["sharpe_ratio"]
        max_dd_pct = metrics["max_drawdown"] * 100
        
        summary_parts.append(f"Total Return: {total_return_pct:.2f}%")
        summary_parts.append(f"Annualized Return: {annualized_return_pct:.2f}%")
        summary_parts.append(f"Volatility: {volatility_pct:.2f}%")
        summary_parts.append(f"Sharpe Ratio: {sharpe:.2f}")
        summary_parts.append(f"Maximum Drawdown: {max_dd_pct:.2f}%")
        
        return "; ".join(summary_parts)

    def _assess_risk(self, metrics: Dict[str, float]) -> str:
        """Assess risk level based on metrics."""
        sharpe = metrics["sharpe_ratio"]
        max_dd = abs(metrics["max_drawdown"])
        volatility = metrics["volatility"]
        
        risk_score = 0
        
        # Sharpe ratio assessment
        if sharpe > 1.5:
            risk_score += 1
        elif sharpe < 0.5:
            risk_score -= 1
        
        # Drawdown assessment
        if max_dd < 0.1:
            risk_score += 1
        elif max_dd > 0.3:
            risk_score -= 1
        
        # Volatility assessment
        if volatility < 0.15:
            risk_score += 1
        elif volatility > 0.3:
            risk_score -= 1
        
        if risk_score >= 2:
            return "Low Risk"
        elif risk_score >= 0:
            return "Medium Risk"
        else:
            return "High Risk"

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        sharpe = metrics["sharpe_ratio"]
        max_dd = abs(metrics["max_drawdown"])
        win_rate = metrics["win_rate"]
        profit_factor = metrics["profit_factor"]
        
        if sharpe < 1.0:
            recommendations.append("Consider improving risk-adjusted returns")
        
        if max_dd > 0.2:
            recommendations.append("Implement better risk management to reduce drawdowns")
        
        if win_rate < 0.4:
            recommendations.append("Review entry/exit criteria to improve win rate")
        
        if profit_factor < 1.5:
            recommendations.append("Focus on improving profit factor through better position sizing")
        
        if not recommendations:
            recommendations.append("Strategy performance is good, consider scaling up")
        
        return recommendations

    def _store_summary(self, metrics: Dict[str, float], returns: pd.Series):
        """Store summary for historical tracking."""
        summary_record = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "data_points": len(returns),
            "risk_free_rate": metrics["risk_free_rate_used"]
        }
        self.summary_history.append(summary_record)

    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure."""
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "beta": 1.0,
            "alpha": 0.0,
            "information_ratio": 0.0,
            "treynor_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "risk_free_rate_used": self.risk_free_rate,
            "trading_days_per_year": self.trading_days_per_year,
            "data_points": 0
        }

    def get_summary_history(self) -> List[Dict[str, Any]]:
        """Get history of all summaries."""
        return self.summary_history

    def export_summary_history(self, filename: str) -> bool:
        """Export summary history to file."""
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.summary_history, f, indent=2, default=str)
            logger.info(f"Summary history exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting summary history: {e}")
            return False
