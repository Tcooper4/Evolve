from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import json

from trading.models.advanced.ensemble.ensemble_model import EnsembleForecaster
from trading.memory import PerformanceMemory
from trading.utils.exceptions import UpdaterError


@dataclass
class UpdateConfig:
    """Configuration for model updates."""
    ticker: str
    metric: str = "mse"
    interval: int = 60
    min_history: int = 10
    weight_threshold: float = 0.01
    custom_metric_fn: Optional[Callable[[Dict[str, float]], float]] = None


class ModelUpdater:
    """Periodically update ensemble model weights based on stored metrics."""

    def __init__(
        self,
        model: EnsembleForecaster,
        memory: Optional[PerformanceMemory] = None,
        default_interval: int = 60,
        max_workers: int = 4,
        on_update: Optional[Callable[[str, Dict[str, float]], None]] = None
    ):
        """Initialize the model updater.
        
        Args:
            model: Ensemble model to update
            memory: Performance memory instance
            default_interval: Default update interval in seconds
            max_workers: Maximum number of worker threads
            on_update: Optional callback function called after each update
        """
        self.model = model
        self.memory = memory or PerformanceMemory()
        self.default_interval = default_interval
        self.max_workers = max_workers
        self.on_update = on_update
        
        # Threading and state management
        self._stop_event = threading.Event()
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Active update configurations
        self._configs: Dict[str, UpdateConfig] = {}

    def update_model_weights(
        self,
        ticker: str,
        metric: str = "mse",
        min_history: int = 10,
        weight_threshold: float = 0.01,
        custom_metric_fn: Optional[Callable[[Dict[str, float]], float]] = None
    ) -> Dict[str, float]:
        """Update model weights using the specified metric.
        
        Args:
            ticker: Ticker symbol
            metric: Metric to use for weight updates
            min_history: Minimum number of historical points required
            weight_threshold: Minimum change in weights to trigger update
            custom_metric_fn: Optional custom metric function
            
        Returns:
            Dictionary of model weights before and after update
            
        Raises:
            UpdaterError: If update fails
        """
        try:
            with self._lock:
                # Get current weights
                old_weights = self.model.get_weights(ticker).copy()
                
                # Get metrics from memory
                metrics = self.memory.get_metrics(ticker)
                if not metrics:
                    raise UpdaterError(f"No metrics found for {ticker}")
                
                # Validate metrics structure
                if not self._validate_metrics(metrics):
                    raise UpdaterError(f"Invalid metrics structure for {ticker}")
                
                # Check history length
                if len(metrics) < min_history:
                    self.logger.warning(
                        f"Insufficient history for {ticker}: {len(metrics)} < {min_history}"
                    )
                    return {"old": old_weights, "new": old_weights}
                
                # Calculate new weights
                if custom_metric_fn:
                    new_weights = self._calculate_weights_custom(metrics, custom_metric_fn)
                else:
                    new_weights = self._calculate_weights(metrics, metric)
                
                # Check weight change threshold
                if not self._should_update(old_weights, new_weights, weight_threshold):
                    self.logger.info(f"Weight change below threshold for {ticker}")
                    return {"old": old_weights, "new": old_weights}
                
                # Update weights
                self.model.update_weights(ticker, new_weights)
                
                # Log weight changes
                self._log_weight_changes(ticker, old_weights, new_weights)
                
                # Call update callback if provided
                if self.on_update:
                    self.on_update(ticker, new_weights)
                
                return {"old": old_weights, "new": new_weights}
                
        except Exception as e:
            self.logger.error(f"Failed to update weights for {ticker}: {str(e)}")
            raise UpdaterError(f"Update failed: {str(e)}")

    def start_periodic_updates(
        self,
        tickers: Union[str, List[str]],
        metric: str = "mse",
        interval: Optional[int] = None,
        min_history: int = 10,
        weight_threshold: float = 0.01,
        custom_metric_fn: Optional[Callable[[Dict[str, float]], float]] = None
    ) -> None:
        """Start background updates of model weights.
        
        Args:
            tickers: Single ticker or list of tickers to update
            metric: Metric to use for updates
            interval: Update interval in seconds (optional)
            min_history: Minimum history required
            weight_threshold: Minimum weight change threshold
            custom_metric_fn: Optional custom metric function
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        for ticker in tickers:
            if ticker in self._threads and self._threads[ticker].is_alive():
                self.logger.warning(f"Update thread already running for {ticker}")
                continue
                
            config = UpdateConfig(
                ticker=ticker,
                metric=metric,
                interval=interval or self.default_interval,
                min_history=min_history,
                weight_threshold=weight_threshold,
                custom_metric_fn=custom_metric_fn
            )
            
            self._configs[ticker] = config
            self._stop_event.clear()
            
            thread = threading.Thread(
                target=self._run,
                args=(config,),
                daemon=True
            )
            self._threads[ticker] = thread
            thread.start()
            
            self.logger.info(f"Started periodic updates for {ticker}")

    def stop(self, ticker: Optional[str] = None) -> None:
        """Stop periodic updates.
        
        Args:
            ticker: Optional ticker to stop updates for. If None, stops all updates.
        """
        if ticker:
            if ticker in self._threads:
                self._stop_event.set()
                self._threads[ticker].join()
                del self._threads[ticker]
                del self._configs[ticker]
                self.logger.info(f"Stopped updates for {ticker}")
        else:
            self._stop_event.set()
            for t in self._threads.values():
                t.join()
            self._threads.clear()
            self._configs.clear()
            self.logger.info("Stopped all updates")

    def status(self) -> Dict[str, Any]:
        """Get current updater status.
        
        Returns:
            Dictionary containing status information
        """
        with self._lock:
            return {
                "active_tickers": list(self._threads.keys()),
                "configs": {
                    ticker: {
                        "metric": config.metric,
                        "interval": config.interval,
                        "min_history": config.min_history,
                        "weight_threshold": config.weight_threshold
                    }
                    for ticker, config in self._configs.items()
                },
                "is_running": any(t.is_alive() for t in self._threads.values())
            }

    def _run(self, config: UpdateConfig) -> None:
        """Background update loop.
        
        Args:
            config: Update configuration
        """
        while not self._stop_event.is_set():
            try:
                self.update_model_weights(
                    config.ticker,
                    config.metric,
                    config.min_history,
                    config.weight_threshold,
                    config.custom_metric_fn
                )
                self.logger.debug(
                    f"Updated weights for {config.ticker} using {config.metric}"
                )
            except Exception as e:
                self.logger.error(
                    f"Error updating {config.ticker}: {str(e)}"
                )
            time.sleep(config.interval)

    def _validate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Validate metrics structure.
        
        Args:
            metrics: Metrics dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(metrics, dict):
                return False
            for model_id, model_metrics in metrics.items():
                if not isinstance(model_metrics, dict):
                    return False
                if not all(isinstance(v, (int, float)) for v in model_metrics.values()):
                    return False
            return True
        except Exception:
            return False

    def _calculate_weights(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric: str
    ) -> Dict[str, float]:
        """Calculate model weights from metrics.
        
        Args:
            metrics: Dictionary of model metrics
            metric: Metric to use for weight calculation
            
        Returns:
            Dictionary of model weights
        """
        try:
            # Get metric values
            metric_values = {
                model_id: model_metrics.get(metric, float('inf'))
                for model_id, model_metrics in metrics.items()
            }
            
            # Handle infinite values
            if all(v == float('inf') for v in metric_values.values()):
                return {model_id: 1.0/len(metric_values) for model_id in metric_values}
            
            # Convert to weights (lower is better for MSE, higher is better for Sharpe)
            if metric.lower() == "mse":
                weights = {
                    model_id: 1.0 / (value + 1e-10)
                    for model_id, value in metric_values.items()
                }
            else:  # sharpe or other metrics where higher is better
                weights = {
                    model_id: max(0, value)
                    for model_id, value in metric_values.items()
                }
            
            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                return {model_id: w/total for model_id, w in weights.items()}
            else:
                return {model_id: 1.0/len(weights) for model_id in weights}
                
        except Exception as e:
            self.logger.error(f"Error calculating weights: {str(e)}")
            return {model_id: 1.0/len(metrics) for model_id in metrics}

    def _calculate_weights_custom(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_fn: Callable[[Dict[str, float]], float]
    ) -> Dict[str, float]:
        """Calculate weights using custom metric function.
        
        Args:
            metrics: Dictionary of model metrics
            metric_fn: Custom metric function
            
        Returns:
            Dictionary of model weights
        """
        try:
            weights = {
                model_id: metric_fn(model_metrics)
                for model_id, model_metrics in metrics.items()
            }
            
            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                return {model_id: w/total for model_id, w in weights.items()}
            else:
                return {model_id: 1.0/len(weights) for model_id in weights}
                
        except Exception as e:
            self.logger.error(f"Error calculating custom weights: {str(e)}")
            return {model_id: 1.0/len(metrics) for model_id in metrics}

    def _should_update(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        threshold: float
    ) -> bool:
        """Check if weight changes exceed threshold.
        
        Args:
            old_weights: Current weights
            new_weights: Proposed new weights
            threshold: Minimum change threshold
            
        Returns:
            True if update should proceed
        """
        try:
            max_change = max(
                abs(new_weights.get(model_id, 0) - old_weights.get(model_id, 0))
                for model_id in set(old_weights) | set(new_weights)
            )
            return max_change >= threshold
        except Exception:
            return True

    def _log_weight_changes(
        self,
        ticker: str,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> None:
        """Log weight changes.
        
        Args:
            ticker: Ticker symbol
            old_weights: Previous weights
            new_weights: New weights
        """
        changes = {
            model_id: {
                "old": old_weights.get(model_id, 0),
                "new": new_weights.get(model_id, 0),
                "change": new_weights.get(model_id, 0) - old_weights.get(model_id, 0)
            }
            for model_id in set(old_weights) | set(new_weights)
        }
        
        self.logger.info(
            f"Weight changes for {ticker}:\n{json.dumps(changes, indent=2)}"
        )

