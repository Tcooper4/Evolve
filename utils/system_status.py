"""System status and metrics calculation utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_system_scorecard() -> Dict[str, Any]:
    """Calculate system performance metrics from logs and goals.
    
    Returns:
        Dictionary containing:
        - sharpe_7d: Average Sharpe ratio over last 7 days
        - sharpe_30d: Average Sharpe ratio over last 30 days
        - win_rate: Percentage of profitable trades
        - mse_avg: Average Mean Squared Error
        - goal_status: Dictionary of goal achievement status
        - last_10_entries: DataFrame of last 10 performance entries
        - trades_per_day: Series of trades per day
    """
    try:
        # Load performance log
        log_file = Path("memory/logs/performance_log.csv")
        if not log_file.exists():
            return {
                "sharpe_7d": 0.0,
                "sharpe_30d": 0.0,
                "win_rate": 0.0,
                "mse_avg": 0.0,
                "goal_status": {},
                "last_10_entries": pd.DataFrame(),
                "trades_per_day": pd.Series()
            }
            
        df = pd.read_csv(log_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate date ranges
        now = datetime.now()
        date_7d = now - timedelta(days=7)
        date_30d = now - timedelta(days=30)
        
        # Calculate metrics
        sharpe_7d = df[df['timestamp'] >= date_7d]['sharpe'].mean()
        sharpe_30d = df[df['timestamp'] >= date_30d]['sharpe'].mean()
        
        # Calculate win rate
        profitable_trades = df[df['sharpe'] > 0].shape[0]
        total_trades = df.shape[0]
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate MSE average
        mse_avg = df['mse'].mean()
        
        # Get last 10 entries
        last_10_entries = df.sort_values('timestamp', ascending=False).head(10)
        
        # Calculate trades per day
        trades_per_day = df.groupby(df['timestamp'].dt.date).size()
        
        # Load goal status
        goal_file = Path("memory/goals/status.json")
        goal_status = {}
        if goal_file.exists():
            with open(goal_file, 'r') as f:
                goal_status = json.load(f)
        
        return {
            "sharpe_7d": round(sharpe_7d, 2),
            "sharpe_30d": round(sharpe_30d, 2),
            "win_rate": round(win_rate * 100, 1),  # Convert to percentage
            "mse_avg": round(mse_avg, 4),
            "goal_status": goal_status,
            "last_10_entries": last_10_entries,
            "trades_per_day": trades_per_day
        }
        
    except Exception as e:
        error_msg = f"Error calculating system metrics: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
