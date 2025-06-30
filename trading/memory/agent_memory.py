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

    def log_outcome(self, agent: str, run_type: str, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log an outcome for an agent (decision, score, trade, tuning, etc.).
        Args:
            agent: Name of the agent (e.g., 'ModelBuilderAgent')
            run_type: Type of run (e.g., 'build', 'evaluate', 'tune', 'trade')
            outcome: Dict with details (must include 'model_id' or similar)
        Returns:
            Dictionary with logging status and details
        """
        try:
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
            
            return {
                'success': True,
                'message': f'Outcome logged successfully for {agent}',
                'agent': agent,
                'run_type': run_type,
                'timestamp': now,
                'entry_count': len(run_section)
            }
            
        except Exception as e:
            logger.error(f"Error logging outcome: {e}")
            return {
                'success': False,
                'message': f'Error logging outcome: {str(e)}',
                'agent': agent,
                'run_type': run_type,
                'timestamp': datetime.now().isoformat()
            }

    def get_history(self, agent: str, run_type: Optional[str] = None, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve past outcomes for an agent, optionally filtered by run_type and/or model_id.
        Args:
            agent: Name of the agent
            run_type: Type of run (optional)
            model_id: Filter by model_id (optional)
        Returns:
            Dictionary with history data and status
        """
        try:
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
            
            return {
                'success': True,
                'message': f'History retrieved for {agent}',
                'agent': agent,
                'run_type': run_type,
                'model_id': model_id,
                'history': runs,
                'count': len(runs),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return {
                'success': False,
                'message': f'Error getting history: {str(e)}',
                'agent': agent,
                'run_type': run_type,
                'model_id': model_id,
                'history': [],
                'count': 0,
                'timestamp': datetime.now().isoformat()
            }

    def get_recent_performance(self, agent: str, run_type: str, metric: str, window: int = 10) -> dict:
        """
        Get recent values of a performance metric for trend analysis.
        Args:
            agent: Name of the agent
            run_type: Type of run
            metric: Metric key (e.g., 'sharpe_ratio')
            window: Number of most recent entries to consider
        Returns:
            Dictionary with performance data and status
        """
        try:
            history_result = self.get_history(agent, run_type)
            if not history_result.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get history',
                    'agent': agent,
                    'run_type': run_type,
                    'metric': metric,
                    'timestamp': datetime.now().isoformat()
                }
            
            history = history_result.get('history', [])
            values = [r.get(metric) for r in history if metric in r]
            recent_values = values[-window:]
            
            return {
                'success': True,
                'message': f'Recent performance retrieved for {agent}',
                'agent': agent,
                'run_type': run_type,
                'metric': metric,
                'values': recent_values,
                'count': len(recent_values),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'agent': agent,
                'run_type': run_type,
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }

    def is_improving(self, agent: str, run_type: str, metric: str, window: int = 10) -> dict:
        """
        Detect if a metric is improving (increasing or decreasing, depending on metric).
        Args:
            agent: Name of the agent
            run_type: Type of run
            metric: Metric key
            window: Number of recent entries to consider
        Returns:
            Dictionary with improvement analysis and status
        """
        try:
            performance_result = self.get_recent_performance(agent, run_type, metric, window)
            if not performance_result.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to get performance data',
                    'agent': agent,
                    'run_type': run_type,
                    'metric': metric,
                    'timestamp': datetime.now().isoformat()
                }
            
            values = performance_result.get('values', [])
            if len(values) < 2:
                return {
                    'success': True,
                    'result': None,
                    'message': 'Not enough data for trend analysis',
                    'agent': agent,
                    'run_type': run_type,
                    'metric': metric,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Simple trend: compare last value to mean of previous
            prev_mean = sum(values[:-1]) / (len(values) - 1)
            last = values[-1]
            
            # For metrics where higher is better
            if metric in {"sharpe_ratio", "win_rate", "total_return", "calmar_ratio"}:
                is_improving = last > prev_mean
            # For metrics where lower is better
            elif metric in {"drawdown", "max_drawdown", "mse", "rmse"}:
                is_improving = last < prev_mean
            else:
                is_improving = None
            
            return {
                'success': True,
                'result': is_improving,
                'message': f'Trend analysis completed for {agent}',
                'agent': agent,
                'run_type': run_type,
                'metric': metric,
                'current_value': last,
                'previous_mean': prev_mean,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'agent': agent,
                'run_type': run_type,
                'metric': metric,
                'timestamp': datetime.now().isoformat()
            }

    def clear(self) -> Dict[str, Any]:
        """Clear all agent memory.
        
        Returns:
            Dictionary with clear status and details
        """
        try:
            self._save({})
            return {
                'success': True,
                'message': 'All agent memory cleared successfully',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error clearing agent memory: {e}")
            return {
                'success': False,
                'message': f'Error clearing agent memory: {str(e)}',
                'timestamp': datetime.now().isoformat()
            } 