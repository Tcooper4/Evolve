"""Enhanced risk management module with comprehensive metrics and safeguards."""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, asdict

@dataclass
class RiskMetrics:
    """Risk metrics data class."""
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation: float
    kelly_fraction: float
    timestamp: str

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
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Add file handler for debug logs if no handlers exist
        if not self.logger.handlers:
            try:
                os.makedirs('trading/risk/logs', exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create trading/risk/logs: {e}")
            try:
                os.makedirs('trading/risk/results', exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create trading/risk/results: {e}")
                
            debug_handler = logging.FileHandler('trading/risk/logs/risk_debug.log')
            debug_handler.setLevel(logging.DEBUG)
            debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            debug_handler.setFormatter(debug_formatter)
            self.logger.addHandler(debug_handler)
        
        self.logger.info(f"Initialized RiskManager with config: {self.config}")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def update_returns(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> None:
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
        excess_returns = self.returns - (self.config.get('risk_free_rate', 0.02) / 252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / volatility if volatility != 0 else 0
        
        # Calculate Sortino ratio
        downside_returns = self.returns[self.returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_vol if downside_vol != 0 else 0
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(self.returns, 5)
        cvar_95 = self.returns[self.returns <= var_95].mean()
        
        # Calculate maximum drawdown
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate beta and correlation
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            beta = self.returns.cov(self.benchmark_returns) / self.benchmark_returns.var()
            correlation = self.returns.corr(self.benchmark_returns)
        else:
            beta = 1.0
            correlation = 0.0
        
        # Calculate Kelly fraction
        win_rate = len(self.returns[self.returns > 0]) / len(self.returns)
        avg_win = self.returns[self.returns > 0].mean() if len(self.returns[self.returns > 0]) > 0 else 0
        avg_loss = abs(self.returns[self.returns < 0].mean()) if len(self.returns[self.returns < 0]) > 0 else 0
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win != 0 else 0
        
        # Create metrics object
        self.current_metrics = RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            volatility=volatility,
            beta=beta,
            correlation=correlation,
            kelly_fraction=kelly_fraction,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Add to history
        self.metrics_history.append(self.current_metrics)
        
        self.logger.info(f"Calculated metrics: {self.current_metrics}")
    
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
        max_position_size = self.config.get('max_position_size', 0.2)
        max_leverage = self.config.get('max_leverage', 1.0)
        
        # Adjust based on risk metrics
        volatility_factor = 1.0 / (1.0 + self.current_metrics.volatility)
        drawdown_factor = 1.0 + self.current_metrics.max_drawdown
        kelly_factor = min(self.current_metrics.kelly_fraction, 0.5)  # Cap at 0.5
        
        # Calculate final limits
        position_limit = max_position_size * volatility_factor * drawdown_factor * kelly_factor
        leverage_limit = max_leverage * volatility_factor * drawdown_factor
        
        return {
            'position_limit': position_limit,
            'leverage_limit': leverage_limit,
            'kelly_fraction': kelly_factor
        }
    
    def optimize_position_sizes(self, expected_returns: pd.Series,
                              covariance: pd.DataFrame) -> pd.Series:
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
            return {'success': True, 'result': {'success': True, 'result': -portfolio_return / portfolio_vol if portfolio_vol != 0 else 0, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # weights sum to 1
        ]
        
        # Define bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return pd.Series(result.x, index=expected_returns.index)
    
    def plot_risk_metrics(self) -> go.Figure:
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
            subplot_titles=('Sharpe Ratio', 'Volatility', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        # Extract metrics
        timestamps = [m.timestamp for m in self.metrics_history]
        sharpe_ratios = [m.sharpe_ratio for m in self.metrics_history]
        volatilities = [m.volatility for m in self.metrics_history]
        drawdowns = [m.max_drawdown for m in self.metrics_history]
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=sharpe_ratios, name='Sharpe Ratio'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=volatilities, name='Volatility'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=drawdowns, name='Drawdown'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text='Risk Metrics History'
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
            results_dir = 'trading/risk/results'
            
            # Get all result files
            result_files = []
            for file in os.listdir(results_dir):
                if file.endswith(('.json', '.csv')):
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
                'success': True,
                'message': f'Cleanup completed. Removed {len(removed_files)} files',
                'removed_files': removed_files,
                'files_retained': len(result_files) - len(removed_files),
                'max_files': max_files,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                'success': False,
                'message': f'Error during cleanup: {str(e)}',
                'max_files': max_files,
                'timestamp': datetime.utcnow().isoformat()
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
                    'success': False,
                    'message': 'No metrics history available for export',
                    'filepath': filepath,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Prepare data
            report_data = {
                'config': self.config,
                'metrics_history': [asdict(m) for m in self.metrics_history],
                'current_metrics': asdict(self.current_metrics) if self.current_metrics else None,
                'returns_summary': {
                    'mean': float(self.returns.mean()),
                    'std': float(self.returns.std()),
                    'min': float(self.returns.min()),
                    'max': float(self.returns.max())
                } if self.returns is not None else None
            }
            
            # Export based on file extension
            if filepath.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2)
            elif filepath.endswith('.csv'):
                # Convert metrics history to DataFrame
                df = pd.DataFrame([asdict(m) for m in self.metrics_history])
                df.to_csv(filepath, index=False)
            else:
                return {
                    'success': False,
                    'message': 'Unsupported file format. Use .json or .csv',
                    'filepath': filepath,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            self.logger.info(f"Exported risk report to {filepath}")
            
            return {
                'success': True,
                'message': f'Risk report exported successfully to {filepath}',
                'filepath': filepath,
                'file_size': os.path.getsize(filepath),
                'metrics_count': len(self.metrics_history),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting risk report: {e}")
            return {
                'success': False,
                'message': f'Error exporting risk report: {str(e)}',
                'filepath': filepath,
                'timestamp': datetime.utcnow().isoformat()
            }

__all__ = ["RiskManager", "RiskMetrics"] 