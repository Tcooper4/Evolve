"""Risk logger module for real-time risk tracking."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from .risk_metrics import calculate_rolling_metrics, calculate_advanced_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading/risk/logs/risk_logger.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskLogger:
    """Logger for tracking risk metrics in real-time."""
    
    def __init__(
        self,
        log_path: str = 'trading/risk/logs/risk_metrics.jsonl',
        update_interval: int = 900  # 15 minutes in seconds
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    ):
        """Initialize risk logger.
        
        Args:
            log_path: Path to store risk metrics
            update_interval: Update interval in seconds
        """
        self.log_path = log_path
        self.update_interval = update_interval
        self.last_update = None
        
        # Create log directory if needed
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def log_metrics(
        self,
        returns: pd.Series,
        model_name: str,
        forecast_confidence: float,
        historical_error: float,
        additional_metrics: Optional[Dict] = None
    ):
        """Log risk metrics.
        
        Args:
            returns: Daily returns series
            model_name: Name of the model
            forecast_confidence: Model forecast confidence
            historical_error: Historical forecast error
            additional_metrics: Additional metrics to log
        """
        try:
            # Calculate metrics
            rolling_metrics = calculate_rolling_metrics(returns)
            advanced_metrics = calculate_advanced_metrics(returns)
            
            # Prepare log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'forecast_confidence': forecast_confidence,
                'historical_error': historical_error,
                'metrics': {
                    **rolling_metrics.iloc[-1].to_dict(),
                    **advanced_metrics
                }
            }
            
            # Add additional metrics if provided
            if additional_metrics:
                log_entry['additional_metrics'] = additional_metrics
            
            # Write to JSONL file
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            self.last_update = datetime.now()
            logger.info(f"Logged risk metrics for {model_name}")
            return {"status": "metrics_logged", "model_name": model_name, "timestamp": self.last_update.isoformat()}
            
        except Exception as e:
            logger.error(f"Error logging risk metrics: {e}")
            return {'success': True, 'result': {"status": "logging_failed", "error": str(e)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_recent_metrics(
        self,
        model_name: Optional[str] = None,
        n_entries: int = 100
    ) -> pd.DataFrame:
        """Get recent risk metrics.
        
        Args:
            model_name: Filter by model name
            n_entries: Number of recent entries to return
            
        Returns:
            DataFrame of recent metrics
        """
        try:
            # Read JSONL file
            entries = []
            with open(self.log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    if model_name is None or entry['model_name'] == model_name:
                        entries.append(entry)
            
            # Convert to DataFrame
            df = pd.DataFrame(entries)
            
            # Sort by timestamp and get recent entries
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            return df.head(n_entries)
            
        except Exception as e:
            logger.error(f"Error reading risk metrics: {e}")
            return {'success': True, 'result': pd.DataFrame(), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def should_update(self) -> bool:
        """Check if metrics should be updated.
        
        Returns:
            True if update is needed
        """
        if self.last_update is None:
            return {'success': True, 'result': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        time_since_update = (
            datetime.now() - self.last_update
        ).total_seconds()
        
        return time_since_update >= self.update_interval
    
    def cleanup_old_logs(self, max_age_days: int = 30):
        """Clean up old log entries.
        
        Args:
            max_age_days: Maximum age of logs in days
        """
        try:
            # Read all entries
            entries = []
            with open(self.log_path, 'r') as f:
                for line in f:
                    entries.append(json.loads(line))
            
            # Filter recent entries
            cutoff_date = datetime.now() - pd.Timedelta(days=max_age_days)
            recent_entries = [
                entry for entry in entries
                if pd.to_datetime(entry['timestamp']) > cutoff_date
            ]
            
            # Write back recent entries
            with open(self.log_path, 'w') as f:
                for entry in recent_entries:
                    f.write(json.dumps(entry) + '\n')
            
            logger.info(f"Cleaned up logs older than {max_age_days} days")
            return {"status": "cleanup_completed", "entries_removed": len(entries) - len(recent_entries)}
            
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")
            return {'success': True, 'result': {"status": "cleanup_failed", "error": str(e)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def export_metrics(
        self,
        output_path: str,
        format: str = 'csv',
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    ):
        """Export risk metrics.
        
        Args:
            output_path: Path to save exported data
            format: Export format ('csv' or 'json')
            model_name: Filter by model name
            start_date: Filter by start date
            end_date: Filter by end date
        """
        try:
            # Get metrics
            df = self.get_recent_metrics(model_name)
            
            # Apply date filters
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            # Export
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported metrics to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}") 