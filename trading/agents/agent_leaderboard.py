"""
Agent Leaderboard

Tracks agent/model performance (Sharpe, drawdown, win rate, etc.), automatically
deprecates underperformers, and provides leaderboard data for dashboards/reports.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np

@dataclass
class AgentPerformance:
    agent_name: str
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    last_updated: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"  # active, deprecated
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['last_updated'] = self.last_updated.isoformat()
        return d

class AgentLeaderboard:
    """
    Tracks agent/model performance, deprecates underperformers, and provides leaderboard data.
    """
    def __init__(self, deprecation_thresholds: Optional[Dict[str, float]] = None):
        self.logger = logging.getLogger(__name__)
        self.leaderboard: Dict[str, AgentPerformance] = {}
        self.deprecation_thresholds = deprecation_thresholds or {
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.25,  # 25%
            'win_rate': 0.45
        }
        self.history: List[Dict[str, Any]] = []
        self.init_status = {
            'success': True,
            'message': 'AgentLeaderboard initialized successfully',
            'timestamp': datetime.now().isoformat()
        }

    def update_performance(self, agent_name: str, sharpe_ratio: float, max_drawdown: float, win_rate: float, total_return: float, extra_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update or add agent performance."""
        perf = AgentPerformance(
            agent_name=agent_name,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_return=total_return,
            status="active",
            extra_metrics=extra_metrics or {}
        )
        self.leaderboard[agent_name] = perf
        self.history.append(perf.to_dict())
        self._check_deprecation(agent_name)
        self.logger.info(f"Updated performance for {agent_name}: Sharpe={sharpe_ratio:.2f}, Drawdown={max_drawdown:.2%}, WinRate={win_rate:.2%}")
        return {
            'success': True,
            'message': f'Performance updated for {agent_name}',
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name
        }

    def _check_deprecation(self, agent_name: str) -> Dict[str, Any]:
        """Deprecate agent if it falls below thresholds."""
        perf = self.leaderboard[agent_name]
        deprecated = False
        if perf.sharpe_ratio < self.deprecation_thresholds['sharpe_ratio']:
            deprecated = True
        if perf.max_drawdown > self.deprecation_thresholds['max_drawdown']:
            deprecated = True
        if perf.win_rate < self.deprecation_thresholds['win_rate']:
            deprecated = True
        if deprecated:
            perf.status = "deprecated"
            self.logger.warning(f"Agent {agent_name} deprecated due to underperformance.")
        else:
            perf.status = "active"
        return {
            'success': True,
            'message': f'Deprecation check completed for {agent_name}',
            'timestamp': datetime.now().isoformat(),
            'deprecated': deprecated
        }

    def get_leaderboard(self, top_n: int = 10, sort_by: str = "sharpe_ratio") -> Dict[str, Any]:
        """Return leaderboard sorted by metric (default: Sharpe)."""
        sorted_agents = sorted(
            self.leaderboard.values(),
            key=lambda x: getattr(x, sort_by),
            reverse=True
        )
        return {
            'success': True,
            'result': [a.to_dict() for a in sorted_agents[:top_n]],
            'message': 'Leaderboard retrieved',
            'timestamp': datetime.now().isoformat()
        }

    def get_deprecated_agents(self) -> Dict[str, Any]:
        """Return list of deprecated agent names."""
        return {
            'success': True,
            'result': [a.agent_name for a in self.leaderboard.values() if a.status == "deprecated"],
            'message': 'Deprecated agents retrieved',
            'timestamp': datetime.now().isoformat()
        }

    def get_active_agents(self) -> Dict[str, Any]:
        """Return list of active agent names."""
        return {
            'success': True,
            'result': [a.agent_name for a in self.leaderboard.values() if a.status == "active"],
            'message': 'Active agents retrieved',
            'timestamp': datetime.now().isoformat()
        }

    def get_history(self, limit: int = 100) -> Dict[str, Any]:
        """Return recent performance history."""
        return {
            'success': True,
            'result': self.history[-limit:],
            'message': 'Performance history retrieved',
            'timestamp': datetime.now().isoformat()
        }

    def as_dataframe(self) -> Dict[str, Any]:
        """Return leaderboard as a pandas DataFrame (for dashboard/report)."""
        import pandas as pd
        data = [a.to_dict()['result'] for a in self.leaderboard.values()]
        return {
            'success': True,
            'result': pd.DataFrame(data),
            'message': 'Leaderboard DataFrame created',
            'timestamp': datetime.now().isoformat()
        }