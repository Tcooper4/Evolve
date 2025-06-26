"""
AgentMemory: Persistent memory for agent decisions, outcomes, and history.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from filelock import FileLock

logger = logging.getLogger(__name__)

class AgentMemory:
    """
    Persistent memory for all agents using agent_memory.json.
    Each agent has its own section for decisions, model scores, trade outcomes, and tuning history.
    Thread-safe and robust.
    """
    def __init__(self, path: str = "agent_memory.json"):
        self.path = Path(path)
        self.lock_path = Path(f"{path}.lock")
        self.lock = FileLock(str(self.lock_path))
        if not self.path.exists():
            self.path.write_text(json.dumps({}))

    def _load(self) -> Dict[str, Any]:
        with self.lock:
            with open(self.path, "r") as f:
                return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        with self.lock:
            temp_path = self.path.with_suffix('.tmp')
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.path)

    def log_outcome(self, agent: str, run_type: str, outcome: Dict[str, Any]) -> None:
        """
        Log an outcome for an agent (decision, score, trade, tuning, etc.).
        Args:
            agent: Name of the agent (e.g., 'ModelBuilderAgent')
            run_type: Type of run (e.g., 'build', 'evaluate', 'tune', 'trade')
            outcome: Dict with details (must include 'model_id' or similar)
        """
        data = self._load()
        now = datetime.now().isoformat()
        agent_section = data.setdefault(agent, {})
        run_section = agent_section.setdefault(run_type, [])
        entry = {"timestamp": now, **outcome}
        run_section.append(entry)
        # Keep only last 1000 entries per run_type
        if len(run_section) > 1000:
            run_section[:] = run_section[-1000:]
        self._save(data)

    def get_history(self, agent: str, run_type: Optional[str] = None, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve past outcomes for an agent, optionally filtered by run_type and/or model_id.
        Args:
            agent: Name of the agent
            run_type: Type of run (optional)
            model_id: Filter by model_id (optional)
        Returns:
            List of outcome dicts
        """
        data = self._load().get(agent, {})
        if run_type:
            runs = data.get(run_type, [])
        else:
            # All run types
            runs = []
            for v in data.values():
                if isinstance(v, list):
                    runs.extend(v)
        if model_id:
            runs = [r for r in runs if r.get("model_id") == model_id]
        return runs

    def get_recent_performance(self, agent: str, run_type: str, metric: str, window: int = 10) -> List[float]:
        """
        Get recent values of a performance metric for trend analysis.
        Args:
            agent: Name of the agent
            run_type: Type of run
            metric: Metric key (e.g., 'sharpe_ratio')
            window: Number of most recent entries to consider
        Returns:
            List of metric values (most recent last)
        """
        history = self.get_history(agent, run_type)
        values = [r.get(metric) for r in history if metric in r]
        return values[-window:]

    def is_improving(self, agent: str, run_type: str, metric: str, window: int = 10) -> Optional[bool]:
        """
        Detect if a metric is improving (increasing or decreasing, depending on metric).
        Args:
            agent: Name of the agent
            run_type: Type of run
            metric: Metric key
            window: Number of recent entries to consider
        Returns:
            True if improving, False if degrading, None if not enough data
        """
        values = self.get_recent_performance(agent, run_type, metric, window)
        if len(values) < 2:
            return None
        # Simple trend: compare last value to mean of previous
        prev_mean = sum(values[:-1]) / (len(values) - 1)
        last = values[-1]
        # For metrics where higher is better
        if metric in {"sharpe_ratio", "win_rate", "total_return", "calmar_ratio"}:
            return last > prev_mean
        # For metrics where lower is better
        elif metric in {"drawdown", "max_drawdown", "mse", "rmse"}:
            return last < prev_mean
        else:
            return None

    def clear(self) -> None:
        """Clear all agent memory."""
        self._save({}) 