import json
from pathlib import Path
from typing import Dict, Any

class PerformanceMemory:
    """Persistent storage for model performance metrics."""

    def __init__(self, path: str = "model_performance.json"):
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("{}")

    def load(self) -> Dict[str, Any]:
        """Load performance data from disk."""
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        """Save performance data to disk."""
        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)

    def update(self, ticker: str, model: str, metrics: Dict[str, float]) -> None:
        """Update stored metrics for a ticker and model."""
        data = self.load()
        ticker_data = data.get(ticker, {})
        ticker_data[model] = metrics
        data[ticker] = ticker_data
        self.save(data)

    def get_metrics(self, ticker: str) -> Dict[str, Dict[str, float]]:
        """Get metrics for a ticker."""
        return self.load().get(ticker, {})
