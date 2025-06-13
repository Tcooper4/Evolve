from __future__ import annotations

import threading
import time
from typing import Optional

from trading.models.advanced.ensemble.ensemble_model import EnsembleForecaster
from trading.memory import PerformanceMemory


class ModelUpdater:
    """Periodically update ensemble model weights based on stored metrics."""

    def __init__(self, model: EnsembleForecaster, memory: Optional[PerformanceMemory] = None, interval: int = 60):
        self.model = model
        self.memory = memory or PerformanceMemory()
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def update_model_weights(self, ticker: str, metric: str = "mse") -> None:
        """Update model weights using the specified metric."""
        self.model.update_weights_from_memory(ticker, metric)

    def start_periodic_updates(self, ticker: str, metric: str = "mse") -> None:
        """Start background updates of model weights."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(ticker, metric), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop periodic updates."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _run(self, ticker: str, metric: str) -> None:
        while not self._stop_event.is_set():
            self.update_model_weights(ticker, metric)
            time.sleep(self.interval)

