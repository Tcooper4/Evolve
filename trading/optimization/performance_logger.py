"""Performance logger for strategy optimization."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
try:
    debug_handler = logging.FileHandler('trading/optimization/logs/optimization_debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)
except Exception as e:
    # Fallback to basic logging if file handler fails
    pass

class PerformanceMetrics(BaseModel):
    """Performance metrics for a strategy run."""
    
    timestamp: datetime = Field(..., description="Timestamp of the run")
    strategy: str = Field(..., description="Strategy name")
    config: Dict[str, Any] = Field(..., description="Strategy configuration")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate")
    max_drawdown: float = Field(..., ge=0, le=1, description="Maximum drawdown")
    mse: float = Field(..., ge=0, description="Mean squared error")
    alpha: float = Field(..., description="Strategy alpha")
    regime: str = Field(..., description="Market regime")
    reason: str = Field("", description="Reason for the run")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "config": self.config,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "mse": self.mse,
            "alpha": self.alpha,
            "regime": self.regime,
            "reason": self.reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create from dictionary.
        
        Args:
            data: Dictionary with performance data
            
        Returns:
            PerformanceMetrics instance
        """
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            strategy=data["strategy"],
            config=data["config"],
            sharpe_ratio=data["sharpe_ratio"],
            win_rate=data["win_rate"],
            max_drawdown=data["max_drawdown"],
            mse=data["mse"],
            alpha=data["alpha"],
            regime=data["regime"],
            reason=data.get("reason", "")
        )

class PerformanceLogger:
    """Logger for strategy performance metrics."""
    
    def __init__(self, log_dir: str = "trading/optimization/logs"):
        """Initialize the logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "optimization_metrics.jsonl")
        
        try:
            _ = os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create log_dir: {e}")
        
        logger.info(f"Initialized PerformanceLogger with log directory: {log_dir}")
    
    def log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics.
        
        Args:
            metrics: PerformanceMetrics instance
        """
        try:
            # Convert to dictionary
            log_entry = metrics.to_dict()
            
            # Append to JSONL file
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            logger.info(f"Logged metrics for {metrics.strategy}")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    def load_metrics(self, strategy: Optional[str] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> List[PerformanceMetrics]:
        """Load performance metrics.
        
        Args:
            strategy: Optional strategy name to filter by
            start_date: Optional start date to filter by
            end_date: Optional end date to filter by
            
        Returns:
            List of PerformanceMetrics instances
        """
        try:
            if not os.path.exists(self.metrics_file):
                return []
                
            metrics = []
            with open(self.metrics_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    
                    # Apply filters
                    if strategy and data["strategy"] != strategy:
                        continue
                        
                    timestamp = datetime.fromisoformat(data["timestamp"])
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                        
                    metrics.append(PerformanceMetrics.from_dict(data))
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return []
    
    def get_strategy_performance(self, strategy: str,
                               window_days: int = 30) -> pd.DataFrame:
        """Get performance metrics for a strategy.
        
        Args:
            strategy: Strategy name
            window_days: Number of days to look back
            
        Returns:
            DataFrame with performance metrics
        """
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - pd.Timedelta(days=window_days)
            
            # Load metrics
            metrics = self.load_metrics(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date
            )
            
            if not metrics:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame([m.to_dict() for m in metrics])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return pd.DataFrame()
    
    def get_best_config(self, strategy: str,
                       metric: str = "sharpe_ratio",
                       window_days: int = 30) -> Optional[Dict[str, Any]]:
        """Get best configuration for a strategy.
        
        Args:
            strategy: Strategy name
            metric: Metric to optimize
            window_days: Number of days to look back
            
        Returns:
            Best configuration dictionary or None
        """
        try:
            # Get performance data
            df = self.get_strategy_performance(strategy, window_days)
            
            if df.empty:
                return None
                
            # Find best configuration
            best_idx = df[metric].idxmax()
            best_config = df.loc[best_idx, "config"]
            
            return best_config
            
        except Exception as e:
            logger.error(f"Error getting best config: {e}")
            return None

    def analyze_performance(self, strategy: str,
                          window_days: int = 30) -> Dict[str, Any]:
        """Analyze strategy performance.
        
        Args:
            strategy: Strategy name
            window_days: Number of days to look back
            
        Returns:
            Dictionary with performance analysis
        """
        try:
            # Get performance data
            df = self.get_strategy_performance(strategy, window_days)
            
            if df.empty:
                return {}
                
            # Calculate statistics
            analysis = {
                "mean_sharpe": df["sharpe_ratio"].mean(),
                "std_sharpe": df["sharpe_ratio"].std(),
                "mean_win_rate": df["win_rate"].mean(),
                "mean_drawdown": df["max_drawdown"].mean(),
                "mean_alpha": df["alpha"].mean(),
                "best_sharpe": df["sharpe_ratio"].max(),
                "worst_sharpe": df["sharpe_ratio"].min(),
                "best_win_rate": df["win_rate"].max(),
                "worst_drawdown": df["max_drawdown"].max(),
                "regime_performance": df.groupby("regime")["sharpe_ratio"].mean().to_dict()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {}