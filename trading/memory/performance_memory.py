"""Persistent storage for model performance metrics with robust file handling and enhanced features."""

import json
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from filelock import FileLock

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
                    "success": True,
                    "message": f"Backup created and {cleaned_count} old backups cleaned",
                    "timestamp": datetime.now().isoformat(),
                    "backup_path": str(backup_path),
                    "cleaned_count": cleaned_count,
                }
            else:
                return {"success": True, "message": "No file to backup", "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

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
                            "success": True,
                            "result": data,
                            "message": f"Data loaded successfully on attempt {attempt + 1}",
                            "timestamp": datetime.now().isoformat(),
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
                                    "success": True,
                                    "result": data,
                                    "message": "Data loaded from backup",
                                    "timestamp": datetime.now().isoformat(),
                                }
                        except Exception as e:
                            logger.error(f"Failed to load from backup: {e}")
                    return {
                        "success": False,
                        "error": "Failed to load data after all retries",
                        "timestamp": datetime.now().isoformat(),
                    }
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
            time.sleep(0.1 * (attempt + 1))

        return {
            "success": False,
            "error": "Failed to load data after all retries",
            "timestamp": datetime.now().isoformat(),
        }

    def load(self) -> Dict[str, Any]:
        """Load performance data from disk with retry mechanism.

        Returns:
            Dictionary containing performance data
        """
        result = self._load_with_retry()
        if result["success"]:
            return {
                "success": True,
                "result": result["result"],
                "message": "Performance data loaded successfully",
                "timestamp": datetime.now().isoformat(),
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
                temp_path = self.path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=4)

                # Atomic rename
                temp_path.replace(self.path)

                # Create backup every 24 hours
                if not hasattr(self, "_last_backup") or datetime.now() - self._last_backup > timedelta(hours=24):
                    backup_result = self._create_backup()
                    self._last_backup = datetime.now()

                return {
                    "success": True,
                    "message": "Performance data saved successfully",
                    "timestamp": datetime.now().isoformat(),
                    "data_size": len(data),
                }

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

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
                "timestamp": datetime.now().isoformat(),
                "dataset_size": metrics.get("dataset_size", 0),
                "confidence_intervals": metrics.get("confidence_intervals", {}),
            }

            load_result = self.load()
            if not load_result["success"]:
                return load_result

            data = load_result["result"]
            ticker_data = data.get(ticker, {})
            ticker_data[model] = enhanced_metrics
            data[ticker] = ticker_data

            save_result = self.save(data)
            if save_result["success"]:
                return {
                    "success": True,
                    "message": f"Metrics updated for {ticker}/{model}",
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "model": model,
                }
            else:
                return save_result
        except Exception as e:
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def get_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get metrics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing metrics for the ticker
        """
        load_result = self.load()
        if load_result["success"]:
            data = load_result["result"]
            ticker_metrics = data.get(ticker, {})
            return {
                "success": True,
                "result": ticker_metrics,
                "message": f"Retrieved metrics for {ticker}",
                "timestamp": datetime.now().isoformat(),
                "models_count": len(ticker_metrics),
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
        if not metrics_result["success"]:
            return metrics_result

        metrics = metrics_result["result"]
        if not metrics:
            return {
                "success": True,
                "result": None,
                "message": f"No models found for {ticker}",
                "timestamp": datetime.now().isoformat(),
            }

        best_model = None
        best_value = float("inf")

        for model, model_metrics in metrics.items():
            if metric in model_metrics:
                value = model_metrics[metric]
                if value < best_value:
                    best_value = value
                    best_model = model

        return {
            "success": True,
            "result": best_model,
            "message": f"Best model for {ticker} using {metric}",
            "timestamp": datetime.now().isoformat(),
            "best_model": best_model,
            "best_value": best_value if best_model else None,
        }

    def clear(self) -> Dict[str, Any]:
        """Clear all stored performance data.

        Returns:
            Dictionary containing clear operation status
        """
        try:
            save_result = self.save({})
            if save_result["success"]:
                return {
                    "success": True,
                    "message": "All performance data cleared",
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return save_result
        except Exception as e:
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

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
            if not load_result["success"]:
                return load_result

            data = load_result["result"]
            if ticker in data and model in data[ticker]:
                del data[ticker][model]
                if not data[ticker]:  # Remove ticker if no models left
                    del data[ticker]

                save_result = self.save(data)
                if save_result["success"]:
                    return {
                        "success": True,
                        "message": f"Model {model} removed for {ticker}",
                        "timestamp": datetime.now().isoformat(),
                        "ticker": ticker,
                        "model": model,
                    }
                else:
                    return save_result
            else:
                return {
                    "success": True,
                    "message": f"Model {model} not found for {ticker}",
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "model": model,
                }
        except Exception as e:
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def get_all_tickers(self) -> List[str]:
        """Get list of all tickers in the performance memory.

        Returns:
            List of ticker symbols
        """
        try:
            data = self.load()
            if data["success"]:
                return list(data["result"].keys())
            else:
                logger.error(f"Failed to load data: {data['error']}")
                return []
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            return []

    def detect_anomalies(
        self, ticker: str, model: str, current_metrics: Dict[str, float], threshold_std: float = 2.0
    ) -> Dict[str, Any]:
        """Detect anomalies by comparing current metrics to historical performance.

        Args:
            ticker: Stock ticker symbol
            model: Model name
            current_metrics: Current performance metrics
            threshold_std: Number of standard deviations for anomaly detection

        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            data = self.load()
            if not data["success"]:
                return {
                    "success": False,
                    "error": "Failed to load performance data",
                    "timestamp": datetime.now().isoformat(),
                }

            ticker_data = data["result"].get(ticker, {})
            model_data = ticker_data.get(model, {})

            if not model_data:
                return {
                    "success": False,
                    "error": f"No historical data found for {ticker}/{model}",
                    "timestamp": datetime.now().isoformat(),
                }

            anomalies = {}
            historical_stats = {}

            for metric_name, current_value in current_metrics.items():
                if metric_name in ["timestamp", "last_updated"]:
                    continue

                # Get historical values for this metric
                historical_values = []
                for record in model_data.get("history", []):
                    if metric_name in record.get("metrics", {}):
                        historical_values.append(float(record["metrics"][metric_name]))

                if len(historical_values) < 5:
                    # Need at least 5 data points for statistical analysis
                    continue

                # Calculate statistics
                mean_val = sum(historical_values) / len(historical_values)
                variance = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
                std_dev = variance**0.5

                historical_stats[metric_name] = {
                    "mean": mean_val,
                    "std_dev": std_dev,
                    "min": min(historical_values),
                    "max": max(historical_values),
                    "count": len(historical_values),
                }

                # Check for anomalies
                if std_dev > 0:
                    z_score = abs(current_value - mean_val) / std_dev

                    if z_score > threshold_std:
                        anomaly_type = "high" if current_value > mean_val else "low"

                        # Determine if anomaly is good or bad based on metric type
                        is_positive_metric = metric_name in [
                            "r2",
                            "accuracy",
                            "sharpe_ratio",
                            "total_return",
                            "win_rate",
                        ]

                        if is_positive_metric:
                            is_good_anomaly = current_value > mean_val
                        else:
                            is_good_anomaly = current_value < mean_val

                        anomalies[metric_name] = {
                            "current_value": current_value,
                            "historical_mean": mean_val,
                            "z_score": z_score,
                            "anomaly_type": anomaly_type,
                            "is_good_anomaly": is_good_anomaly,
                            "percentile": self._calculate_percentile(current_value, historical_values),
                            "severity": "high" if z_score > 3.0 else "medium" if z_score > 2.5 else "low",
                        }

            return {
                "success": True,
                "ticker": ticker,
                "model": model,
                "anomalies": anomalies,
                "historical_stats": historical_stats,
                "threshold_std": threshold_std,
                "anomaly_count": len(anomalies),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _calculate_percentile(self, value: float, historical_values: List[float]) -> float:
        """Calculate percentile of current value in historical distribution.

        Args:
            value: Current value
            historical_values: List of historical values

        Returns:
            Percentile (0-100)
        """
        try:
            if not historical_values:
                return 50.0

            sorted_values = sorted(historical_values)
            position = 0

            for i, hist_val in enumerate(sorted_values):
                if value <= hist_val:
                    position = i
                    break
                position = i + 1

            percentile = (position / len(sorted_values)) * 100
            return min(100.0, max(0.0, percentile))

        except Exception as e:
            logger.error(f"Error calculating percentile: {e}")
            return 50.0

    def get_performance_trends(self, ticker: str, model: str, metric: str, window_size: int = 10) -> Dict[str, Any]:
        """Get performance trends for a specific metric.

        Args:
            ticker: Stock ticker symbol
            model: Model name
            metric: Metric name to analyze
            window_size: Number of recent records to analyze

        Returns:
            Dictionary containing trend analysis
        """
        try:
            data = self.load()
            if not data["success"]:
                return {
                    "success": False,
                    "error": "Failed to load performance data",
                    "timestamp": datetime.now().isoformat(),
                }

            ticker_data = data["result"].get(ticker, {})
            model_data = ticker_data.get(model, {})

            if not model_data:
                return {
                    "success": False,
                    "error": f"No data found for {ticker}/{model}",
                    "timestamp": datetime.now().isoformat(),
                }

            # Get recent history
            history = model_data.get("history", [])
            if len(history) < window_size:
                window_size = len(history)

            recent_history = history[-window_size:]

            # Extract metric values
            metric_values = []
            timestamps = []

            for record in recent_history:
                if metric in record.get("metrics", {}):
                    metric_values.append(float(record["metrics"][metric]))
                    timestamps.append(record.get("timestamp", ""))

            if len(metric_values) < 2:
                return {
                    "success": False,
                    "error": f"Insufficient data for trend analysis of {metric}",
                    "timestamp": datetime.now().isoformat(),
                }

            # Calculate trend
            x = list(range(len(metric_values)))
            slope, intercept = self._linear_regression(x, metric_values)

            # Calculate trend strength (R-squared)
            y_pred = [slope * xi + intercept for xi in x]
            ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(metric_values))
            ss_tot = sum((y - sum(metric_values) / len(metric_values)) ** 2 for y in metric_values)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Determine trend direction
            if slope > 0.001:
                trend_direction = "improving"
            elif slope < -0.001:
                trend_direction = "declining"
            else:
                trend_direction = "stable"

            return {
                "success": True,
                "ticker": ticker,
                "model": model,
                "metric": metric,
                "trend_direction": trend_direction,
                "slope": slope,
                "trend_strength": r_squared,
                "current_value": metric_values[-1],
                "average_value": sum(metric_values) / len(metric_values),
                "volatility": self._calculate_volatility(metric_values),
                "data_points": len(metric_values),
                "window_size": window_size,
                "timestamps": timestamps,
                "values": metric_values,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _linear_regression(self, x: List[float], y: List[float]) -> tuple[float, float]:
        """Perform simple linear regression.

        Args:
            x: X values
            y: Y values

        Returns:
            Tuple of (slope, intercept)
        """
        try:
            n = len(x)
            if n != len(y) or n < 2:
                return 0.0, 0.0

            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            intercept = (sum_y - slope * sum_x) / n

            return slope, intercept

        except Exception as e:
            logger.error(f"Error in linear regression: {e}")
            return 0.0, 0.0

    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation) of values.

        Args:
            values: List of values

        Returns:
            Volatility measure
        """
        try:
            if len(values) < 2:
                return 0.0

            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance**0.5

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def get_performance_summary(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive performance summary for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing performance summary
        """
        try:
            data = self.load()
            if not data["success"]:
                return {
                    "success": False,
                    "error": "Failed to load performance data",
                    "timestamp": datetime.now().isoformat(),
                }

            ticker_data = data["result"].get(ticker, {})

            if not ticker_data:
                return {
                    "success": False,
                    "error": f"No data found for {ticker}",
                    "timestamp": datetime.now().isoformat(),
                }

            summary = {
                "ticker": ticker,
                "models": list(ticker_data.keys()),
                "total_models": len(ticker_data),
                "best_models": {},
                "recent_anomalies": [],
                "performance_trends": {},
            }

            # Find best model for each metric
            metrics = ["mse", "mae", "r2", "accuracy", "sharpe_ratio", "total_return"]

            for metric in metrics:
                best_model = self.get_best_model(ticker, metric)
                if best_model["success"]:
                    summary["best_models"][metric] = best_model["result"]

            # Get recent anomalies (last 7 days)
            cutoff_date = datetime.now() - timedelta(days=7)

            for model_name, model_data in ticker_data.items():
                # Check for recent anomalies
                if "history" in model_data:
                    recent_records = [
                        record
                        for record in model_data["history"]
                        if datetime.fromisoformat(record.get("timestamp", "")) > cutoff_date
                    ]

                    if recent_records:
                        latest_metrics = recent_records[-1].get("metrics", {})
                        anomaly_result = self.detect_anomalies(ticker, model_name, latest_metrics)

                        if anomaly_result["success"] and anomaly_result["anomalies"]:
                            summary["recent_anomalies"].append(
                                {"model": model_name, "anomalies": anomaly_result["anomalies"]}
                            )

                # Get performance trends
                for metric in metrics:
                    trend_result = self.get_performance_trends(ticker, model_name, metric)
                    if trend_result["success"]:
                        if model_name not in summary["performance_trends"]:
                            summary["performance_trends"][model_name] = {}
                        summary["performance_trends"][model_name][metric] = trend_result

            summary["timestamp"] = datetime.now().isoformat()
            summary["success"] = True

            return summary

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}
