"""Persistent storage for model performance metrics with robust file handling and enhanced features."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from filelock import FileLock
import shutil

logger = logging.getLogger(__name__)

class PerformanceMemory:
    """Persistent storage for model performance metrics with thread-safe operations.
    
    This class provides a robust way to store and retrieve model performance metrics
    with support for file locking, backups, and enhanced metric structures.
    """

    def __init__(self, path: str = "model_performance.json", backup_dir: str = "backups"):
        """Initialize the performance memory storage.
        
        Args:
            path: Path to the main performance JSON file
            backup_dir: Directory to store backup files
        """
        self.path = Path(path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.lock_path = Path(f"{path}.lock")
        self.lock = FileLock(str(self.lock_path))
        
        # Initialize empty file if it doesn't exist
        if not self.path.exists():
            self.path.write_text("{}")
            
        # Create backup on initialization
        self._create_backup()

    def _create_backup(self) -> None:
        """Create a backup of the current performance file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"model_performance_{timestamp}.json"
            if self.path.exists():
                shutil.copy2(self.path, backup_path)
                
                # Clean up old backups (keep last 7 days)
                cutoff = datetime.now() - timedelta(days=7)
                for backup in self.backup_dir.glob("model_performance_*.json"):
                    try:
                        timestamp = datetime.strptime(backup.stem.split("_")[-1], "%Y%m%d_%H%M%S")
                        if timestamp < cutoff:
                            backup.unlink()
                    except ValueError:
                        continue
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    def _load_with_retry(self, max_retries: int = 3) -> Dict[str, Any]:
        """Load data with retry mechanism and backup fallback.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            Loaded data dictionary
        """
        for attempt in range(max_retries):
            try:
                with self.lock:
                    with open(self.path, "r") as f:
                        return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    # Try loading from backup
                    backup_files = sorted(self.backup_dir.glob("model_performance_*.json"))
                    if backup_files:
                        try:
                            with open(backup_files[-1], "r") as f:
                                return json.load(f)
                        except Exception as e:
                            logger.error(f"Failed to load from backup: {e}")
                    return {}
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                if attempt == max_retries - 1:
                    return {}
            time.sleep(0.1 * (attempt + 1))
        return {}

    def load(self) -> Dict[str, Any]:
        """Load performance data from disk with retry mechanism.
        
        Returns:
            Dictionary containing performance data
        """
        return self._load_with_retry()

    def save(self, data: Dict[str, Any]) -> None:
        """Save performance data to disk with file locking.
        
        Args:
            data: Dictionary containing performance data to save
        """
        try:
            with self.lock:
                # Write to temporary file first
                temp_path = self.path.with_suffix('.tmp')
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=4)
                
                # Atomic rename
                temp_path.replace(self.path)
                
                # Create backup every 24 hours
                if not hasattr(self, '_last_backup') or \
                   datetime.now() - self._last_backup > timedelta(hours=24):
                    self._create_backup()
                    self._last_backup = datetime.now()
                    
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise

    def update(self, ticker: str, model: str, metrics: Dict[str, Any]) -> None:
        """Update stored metrics for a ticker and model with enhanced metadata.
        
        Args:
            ticker: Stock ticker symbol
            model: Model identifier
            metrics: Dictionary of metrics with optional metadata
        """
        # Add metadata
        enhanced_metrics = {
            **metrics,
            'timestamp': datetime.now().isoformat(),
            'dataset_size': metrics.get('dataset_size', 0),
            'confidence_intervals': metrics.get('confidence_intervals', {})
        }
        
        data = self.load()
        ticker_data = data.get(ticker, {})
        ticker_data[model] = enhanced_metrics
        data[ticker] = ticker_data
        self.save(data)

    def get_metrics(self, ticker: str) -> Dict[str, Dict[str, Any]]:
        """Get metrics for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary of model metrics for the ticker
        """
        return self.load().get(ticker, {})

    def get_best_model(self, ticker: str, metric: str = "mse") -> Optional[str]:
        """Get the best performing model for a ticker based on specified metric.
        
        Args:
            ticker: Stock ticker symbol
            metric: Metric to use for comparison (default: "mse")
            
        Returns:
            Name of the best performing model or None if no models found
        """
        metrics = self.get_metrics(ticker)
        if not metrics:
            return None
            
        best_model = None
        best_value = float('inf')
        
        for model, model_metrics in metrics.items():
            if metric in model_metrics:
                value = model_metrics[metric]
                if value < best_value:
                    best_value = value
                    best_model = model
                    
        return best_model

    def clear(self) -> None:
        """Clear all stored performance data."""
        self.save({})

    def remove_model(self, ticker: str, model: str) -> None:
        """Remove a specific model's metrics for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            model: Model identifier to remove
        """
        data = self.load()
        if ticker in data and model in data[ticker]:
            del data[ticker][model]
            if not data[ticker]:
                del data[ticker]
            self.save(data)

    def get_all_tickers(self) -> List[str]:
        """Get list of all tickers with stored metrics.
        
        Returns:
            List of ticker symbols
        """
        return list(self.load().keys())
