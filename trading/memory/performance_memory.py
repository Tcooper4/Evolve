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
        
        return {
            'success': True,
            'message': 'PerformanceMemory initialized successfully',
            'timestamp': datetime.now().isoformat()
        }

    def _create_backup(self) -> Dict[str, Any]:
        """Create a backup of the current performance file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"model_performance_{timestamp}.json"
            if self.path.exists():
                shutil.copy2(self.path, backup_path)
                
                # Clean up old backups (keep last 7 days)
                cutoff = datetime.now() - timedelta(days=7)
                cleaned_count = 0
                for backup in self.backup_dir.glob("model_performance_*.json"):
                    try:
                        backup_timestamp = datetime.strptime(backup.stem.split("_")[-1], "%Y%m%d_%H%M%S")
                        if backup_timestamp < cutoff:
                            backup.unlink()
                            cleaned_count += 1
                    except ValueError:
                        continue
                
                return {
                    'success': True,
                    'message': f'Backup created and {cleaned_count} old backups cleaned',
                    'timestamp': datetime.now().isoformat(),
                    'backup_path': str(backup_path),
                    'cleaned_count': cleaned_count
                }
            else:
                return {
                    'success': True,
                    'message': 'No file to backup',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

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
                        data = json.load(f)
                        return {
                            'success': True,
                            'result': data,
                            'message': f'Data loaded successfully on attempt {attempt + 1}',
                            'timestamp': datetime.now().isoformat()
                        }
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    # Try loading from backup
                    backup_files = sorted(self.backup_dir.glob("model_performance_*.json"))
                    if backup_files:
                        try:
                            with open(backup_files[-1], "r") as f:
                                data = json.load(f)
                                return {
                                    'success': True,
                                    'result': data,
                                    'message': 'Data loaded from backup',
                                    'timestamp': datetime.now().isoformat()
                                }
                        except Exception as e:
                            logger.error(f"Failed to load from backup: {e}")
                    return {
                        'success': False,
                        'error': 'Failed to load data after all retries',
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            time.sleep(0.1 * (attempt + 1))
        
        return {
            'success': False,
            'error': 'Failed to load data after all retries',
            'timestamp': datetime.now().isoformat()
        }

    def load(self) -> Dict[str, Any]:
        """Load performance data from disk with retry mechanism.
        
        Returns:
            Dictionary containing performance data
        """
        result = self._load_with_retry()
        if result['success']:
            return {
                'success': True,
                'result': result['result'],
                'message': 'Performance data loaded successfully',
                'timestamp': datetime.now().isoformat()
            }
        else:
            return result

    def save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save performance data to disk with file locking.
        
        Args:
            data: Dictionary containing performance data to save
            
        Returns:
            Dictionary containing save operation status
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
                    backup_result = self._create_backup()
                    self._last_backup = datetime.now()
                
                return {
                    'success': True,
                    'message': 'Performance data saved successfully',
                    'timestamp': datetime.now().isoformat(),
                    'data_size': len(data)
                }
                    
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def update(self, ticker: str, model: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update stored metrics for a ticker and model with enhanced metadata.
        
        Args:
            ticker: Stock ticker symbol
            model: Model identifier
            metrics: Dictionary of metrics with optional metadata
            
        Returns:
            Dictionary containing update operation status
        """
        try:
            # Add metadata
            enhanced_metrics = {
                **metrics,
                'timestamp': datetime.now().isoformat(),
                'dataset_size': metrics.get('dataset_size', 0),
                'confidence_intervals': metrics.get('confidence_intervals', {})
            }
            
            load_result = self.load()
            if not load_result['success']:
                return load_result
            
            data = load_result['result']
            ticker_data = data.get(ticker, {})
            ticker_data[model] = enhanced_metrics
            data[ticker] = ticker_data
            
            save_result = self.save(data)
            if save_result['success']:
                return {
                    'success': True,
                    'message': f'Metrics updated for {ticker}/{model}',
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'model': model
                }
            else:
                return save_result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get metrics for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing metrics for the ticker
        """
        load_result = self.load()
        if load_result['success']:
            data = load_result['result']
            ticker_metrics = data.get(ticker, {})
            return {
                'success': True,
                'result': ticker_metrics,
                'message': f'Retrieved metrics for {ticker}',
                'timestamp': datetime.now().isoformat(),
                'models_count': len(ticker_metrics)
            }
        else:
            return load_result

    def get_best_model(self, ticker: str, metric: str = "mse") -> Dict[str, Any]:
        """Get the best performing model for a ticker based on specified metric.
        
        Args:
            ticker: Stock ticker symbol
            metric: Metric to use for comparison (default: "mse")
            
        Returns:
            Dictionary containing best model information
        """
        metrics_result = self.get_metrics(ticker)
        if not metrics_result['success']:
            return metrics_result
        
        metrics = metrics_result['result']
        if not metrics:
            return {
                'success': True,
                'result': None,
                'message': f'No models found for {ticker}',
                'timestamp': datetime.now().isoformat()
            }
            
        best_model = None
        best_value = float('inf')
        
        for model, model_metrics in metrics.items():
            if metric in model_metrics:
                value = model_metrics[metric]
                if value < best_value:
                    best_value = value
                    best_model = model
        
        return {
            'success': True,
            'result': best_model,
            'message': f'Best model for {ticker} using {metric}',
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model,
            'best_value': best_value if best_model else None
        }

    def clear(self) -> Dict[str, Any]:
        """Clear all stored performance data.
        
        Returns:
            Dictionary containing clear operation status
        """
        try:
            save_result = self.save({})
            if save_result['success']:
                return {
                    'success': True,
                    'message': 'All performance data cleared',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return save_result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def remove_model(self, ticker: str, model: str) -> Dict[str, Any]:
        """Remove a specific model's metrics for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            model: Model identifier to remove
            
        Returns:
            Dictionary containing remove operation status
        """
        try:
            load_result = self.load()
            if not load_result['success']:
                return load_result
            
            data = load_result['result']
            if ticker in data and model in data[ticker]:
                del data[ticker][model]
                if not data[ticker]:  # Remove ticker if no models left
                    del data[ticker]
                
                save_result = self.save(data)
                if save_result['success']:
                    return {
                        'success': True,
                        'message': f'Model {model} removed for {ticker}',
                        'timestamp': datetime.now().isoformat(),
                        'ticker': ticker,
                        'model': model
                    }
                else:
                    return save_result
            else:
                return {
                    'success': True,
                    'message': f'Model {model} not found for {ticker}',
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'model': model
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_all_tickers(self) -> List[str]:
        """Get list of all tickers with stored metrics.
        
        Returns:
            List of ticker symbols
        """
        return list(self.load().keys())
