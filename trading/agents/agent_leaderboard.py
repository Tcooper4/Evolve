"""
Agent Leaderboard

Tracks agent/model performance (Sharpe, drawdown, win rate, etc.), automatically
deprecates underperformers, and provides leaderboard data for dashboards/reports.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"
    TESTING = "testing"

class SortMetric(Enum):
    """Available sorting metrics."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    TOTAL_RETURN = "total_return"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    PROFIT_FACTOR = "profit_factor"

@dataclass
class LeaderboardRequest:
    """Leaderboard request."""
    action: str  # 'get_leaderboard', 'update_performance', 'get_agent_performance', 'get_deprecated', 'get_active', 'get_history', 'get_stats'
    agent_name: Optional[str] = None
    top_n: Optional[int] = None
    sort_by: Optional[str] = None
    status_filter: Optional[str] = None
    limit: Optional[int] = None
    performance_data: Optional[Dict[str, Any]] = None

@dataclass
class LeaderboardResult:
    """Leaderboard result."""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class AgentPerformance:
    """Represents agent performance metrics."""
    agent_name: str
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    status: AgentStatus = AgentStatus.ACTIVE
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate additional metrics if not provided."""
        if self.calmar_ratio == 0.0 and self.max_drawdown != 0:
            self.calmar_ratio = self.total_return / abs(self.max_drawdown)
        
        if self.sortino_ratio == 0.0 and self.volatility != 0:
            # Simplified sortino calculation
            self.sortino_ratio = self.total_return / self.volatility if self.volatility > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['last_updated'] = self.last_updated.isoformat()
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentPerformance':
        """Create from dictionary."""
        if isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        if isinstance(data['status'], str):
            data['status'] = AgentStatus(data['status'])
        return cls(**data)
    
    def get_score(self, metric: SortMetric = SortMetric.SHARPE_RATIO) -> float:
        """Get score for a specific metric."""
        metric_map = {
            SortMetric.SHARPE_RATIO: self.sharpe_ratio,
            SortMetric.MAX_DRAWDOWN: -self.max_drawdown,  # Negative for sorting
            SortMetric.WIN_RATE: self.win_rate,
            SortMetric.TOTAL_RETURN: self.total_return,
            SortMetric.CALMAR_RATIO: self.calmar_ratio,
            SortMetric.SORTINO_RATIO: self.sortino_ratio,
            SortMetric.PROFIT_FACTOR: self.profit_factor
        }
        return metric_map.get(metric, self.sharpe_ratio)

class AgentLeaderboard:
    """
    Tracks agent/model performance, deprecates underperformers, and provides leaderboard data.
    """
    
    def __init__(self, deprecation_thresholds: Optional[Dict[str, float]] = None,
                 data_dir: str = "data/leaderboard"):
        """
        Initialize the agent leaderboard.
        
        Args:
            deprecation_thresholds: Thresholds for deprecating agents
            data_dir: Directory for storing leaderboard data
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.leaderboard: Dict[str, AgentPerformance] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Deprecation thresholds
        self.deprecation_thresholds = deprecation_thresholds or {
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.25,  # 25%
            'win_rate': 0.45,
            'calmar_ratio': 0.3,
            'sortino_ratio': 0.5
        }
        
        # Performance tracking
        self.stats = {
            'total_agents': 0,
            'active_agents': 0,
            'deprecated_agents': 0,
            'last_update': None
        }
        
        # Load existing data
        self._load_data()
        
        self.logger.info(f"AgentLeaderboard initialized with {len(self.leaderboard)} agents")
    
    def _load_data(self) -> None:
        """Load leaderboard data from files."""
        try:
            # Load leaderboard
            leaderboard_file = self.data_dir / "leaderboard.json"
            if leaderboard_file.exists():
                with open(leaderboard_file, 'r') as f:
                    data = json.load(f)
                    self.leaderboard = {
                        name: AgentPerformance.from_dict(perf_data)
                        for name, perf_data in data.items()
                    }
            
            # Load history
            history_file = self.data_dir / "history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.history = json.load(f)
            
            self._update_stats()
            self.logger.info(f"Loaded {len(self.leaderboard)} agents from storage")
            
        except Exception as e:
            self.logger.error(f"Error loading leaderboard data: {e}")
    
    def _save_data(self) -> None:
        """Save leaderboard data to files."""
        try:
            # Save leaderboard
            leaderboard_file = self.data_dir / "leaderboard.json"
            with open(leaderboard_file, 'w') as f:
                json.dump(
                    {name: perf.to_dict() for name, perf in self.leaderboard.items()},
                    f, indent=2
                )
            
            # Save history (keep last 1000 entries)
            history_file = self.data_dir / "history.json"
            with open(history_file, 'w') as f:
                json.dump(self.history[-1000:], f, indent=2)
            
            self.logger.debug("Leaderboard data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving leaderboard data: {e}")
    
    def _update_stats(self) -> None:
        """Update internal statistics."""
        self.stats['total_agents'] = len(self.leaderboard)
        self.stats['active_agents'] = sum(1 for a in self.leaderboard.values() if a.status == AgentStatus.ACTIVE)
        self.stats['deprecated_agents'] = sum(1 for a in self.leaderboard.values() if a.status == AgentStatus.DEPRECATED)
        self.stats['last_update'] = datetime.now().isoformat()
    
    def update_performance(self, agent_name: str, sharpe_ratio: float, max_drawdown: float, 
                          win_rate: float, total_return: float, extra_metrics: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update or add agent performance.
        
        Args:
            agent_name: Name of the agent
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate
            total_return: Total return
            extra_metrics: Additional performance metrics
            metadata: Additional metadata
            
        Returns:
            Dictionary with update result
        """
        try:
            # Calculate additional metrics
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            volatility = extra_metrics.get('volatility', 0.0) if extra_metrics else 0.0
            sortino_ratio = total_return / volatility if volatility > 0 else 0.0
            profit_factor = extra_metrics.get('profit_factor', 0.0) if extra_metrics else 0.0
            
            perf = AgentPerformance(
                agent_name=agent_name,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_return=total_return,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                profit_factor=profit_factor,
                volatility=volatility,
                beta=extra_metrics.get('beta', 0.0) if extra_metrics else 0.0,
                alpha=extra_metrics.get('alpha', 0.0) if extra_metrics else 0.0,
                information_ratio=extra_metrics.get('information_ratio', 0.0) if extra_metrics else 0.0,
                treynor_ratio=extra_metrics.get('treynor_ratio', 0.0) if extra_metrics else 0.0,
                status=AgentStatus.ACTIVE,
                extra_metrics=extra_metrics or {},
                metadata=metadata or {}
            )
            
            self.leaderboard[agent_name] = perf
            self.history.append(perf.to_dict())
            
            # Check deprecation
            deprecation_result = self._check_deprecation(agent_name)
            
            # Update stats and save
            self._update_stats()
            self._save_data()
            
            self.logger.info(f"Updated performance for {agent_name}: Sharpe={sharpe_ratio:.2f}, "
                           f"Drawdown={max_drawdown:.2%}, WinRate={win_rate:.2%}")
            
            return {
                'success': True,
                'message': f'Performance updated for {agent_name}',
                'timestamp': datetime.now().isoformat(),
                'agent': agent_name,
                'deprecation_check': deprecation_result
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance for {agent_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'agent': agent_name
            }
    
    def _check_deprecation(self, agent_name: str) -> Dict[str, Any]:
        """
        Check if agent should be deprecated based on performance thresholds.
        
        Args:
            agent_name: Name of the agent to check
            
        Returns:
            Dictionary with deprecation result
        """
        try:
            perf = self.leaderboard[agent_name]
            deprecated = False
            reasons = []
            
            # Check each threshold
            if perf.sharpe_ratio < self.deprecation_thresholds['sharpe_ratio']:
                deprecated = True
                reasons.append(f"Sharpe ratio {perf.sharpe_ratio:.2f} below threshold {self.deprecation_thresholds['sharpe_ratio']}")
            
            if perf.max_drawdown > self.deprecation_thresholds['max_drawdown']:
                deprecated = True
                reasons.append(f"Max drawdown {perf.max_drawdown:.2%} above threshold {self.deprecation_thresholds['max_drawdown']:.2%}")
            
            if perf.win_rate < self.deprecation_thresholds['win_rate']:
                deprecated = True
                reasons.append(f"Win rate {perf.win_rate:.2%} below threshold {self.deprecation_thresholds['win_rate']:.2%}")
            
            if perf.calmar_ratio < self.deprecation_thresholds.get('calmar_ratio', 0.0):
                deprecated = True
                reasons.append(f"Calmar ratio {perf.calmar_ratio:.2f} below threshold {self.deprecation_thresholds.get('calmar_ratio', 0.0)}")
            
            if perf.sortino_ratio < self.deprecation_thresholds.get('sortino_ratio', 0.0):
                deprecated = True
                reasons.append(f"Sortino ratio {perf.sortino_ratio:.2f} below threshold {self.deprecation_thresholds.get('sortino_ratio', 0.0)}")
            
            # Update status
            if deprecated:
                perf.status = AgentStatus.DEPRECATED
                self.logger.warning(f"Agent {agent_name} deprecated: {', '.join(reasons)}")
            else:
                perf.status = AgentStatus.ACTIVE
            
            return {
                'deprecated': deprecated,
                'reasons': reasons,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error checking deprecation for {agent_name}: {e}")
            return {
                'deprecated': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_leaderboard(self, top_n: int = 10, sort_by: Union[str, SortMetric] = SortMetric.SHARPE_RATIO,
                       status_filter: Optional[AgentStatus] = None) -> Dict[str, Any]:
        """
        Get leaderboard sorted by specified metric.
        
        Args:
            top_n: Number of top agents to return
            sort_by: Metric to sort by
            status_filter: Filter by agent status
            
        Returns:
            Dictionary with leaderboard data
        """
        try:
            # Convert string to enum if needed
            if isinstance(sort_by, str):
                sort_by = SortMetric(sort_by)
            
            # Filter agents
            agents = list(self.leaderboard.values())
            if status_filter:
                agents = [a for a in agents if a.status == status_filter]
            
            # Sort agents
            sorted_agents = sorted(
                agents,
                key=lambda x: x.get_score(sort_by),
                reverse=True
            )
            
            return {
                'success': True,
                'result': [a.to_dict() for a in sorted_agents[:top_n]],
                'total_agents': len(agents),
                'sort_metric': sort_by.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting leaderboard: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """
        Get performance data for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary with agent performance data
        """
        try:
            if agent_name not in self.leaderboard:
                return {
                    'success': False,
                    'error': f'Agent {agent_name} not found',
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'success': True,
                'result': self.leaderboard[agent_name].to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance for {agent_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_deprecated_agents(self) -> Dict[str, Any]:
        """Get list of deprecated agents."""
        try:
            deprecated = [a.to_dict() for a in self.leaderboard.values() 
                         if a.status == AgentStatus.DEPRECATED]
            
            return {
                'success': True,
                'result': deprecated,
                'count': len(deprecated),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting deprecated agents: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_active_agents(self) -> Dict[str, Any]:
        """Get list of active agents."""
        try:
            active = [a.to_dict() for a in self.leaderboard.values() 
                     if a.status == AgentStatus.ACTIVE]
            
            return {
                'success': True,
                'result': active,
                'count': len(active),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting active agents: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_history(self, agent_name: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """
        Get performance history.
        
        Args:
            agent_name: Filter by agent name (optional)
            limit: Maximum number of entries to return
            
        Returns:
            Dictionary with history data
        """
        try:
            history = self.history.copy()
            
            if agent_name:
                history = [h for h in history if h.get('agent_name') == agent_name]
            
            return {
                'success': True,
                'result': history[-limit:],
                'total_entries': len(history),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting history: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get leaderboard statistics."""
        try:
            return {
                'success': True,
                'result': {
                    **self.stats,
                    'deprecation_thresholds': self.deprecation_thresholds,
                    'data_dir': str(self.data_dir)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def as_dataframe(self, status_filter: Optional[AgentStatus] = None) -> Dict[str, Any]:
        """
        Get leaderboard as pandas DataFrame.
        
        Args:
            status_filter: Filter by agent status
            
        Returns:
            Dictionary with DataFrame result
        """
        try:
            agents = list(self.leaderboard.values())
            if status_filter:
                agents = [a for a in agents if a.status == status_filter]
            
            data = [a.to_dict() for a in agents]
            df = pd.DataFrame(data)
            
            return {
                'success': True,
                'result': df,
                'shape': df.shape,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating DataFrame: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def export_data(self, filepath: str = "leaderboard_export.json") -> Dict[str, Any]:
        """
        Export leaderboard data to file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            Dictionary with export result
        """
        try:
            export_data = {
                'leaderboard': {name: perf.to_dict() for name, perf in self.leaderboard.items()},
                'history': self.history[-1000:],  # Last 1000 entries
                'stats': self.stats,
                'deprecation_thresholds': self.deprecation_thresholds,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Leaderboard data exported to {filepath}")
            
            return {
                'success': True,
                'filepath': filepath,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global leaderboard instance
_leaderboard: Optional[AgentLeaderboard] = None

def get_leaderboard() -> AgentLeaderboard:
    """Get the global leaderboard instance."""
    global _leaderboard
    if _leaderboard is None:
        _leaderboard = AgentLeaderboard()
    return _leaderboard

# Convenience functions
def update_agent_performance(agent_name: str, **kwargs) -> Dict[str, Any]:
    """Update agent performance using the global leaderboard."""
    return get_leaderboard().update_performance(agent_name, **kwargs)

def get_top_agents(top_n: int = 10, sort_by: str = "sharpe_ratio") -> Dict[str, Any]:
    """Get top agents using the global leaderboard."""
    return get_leaderboard().get_leaderboard(top_n=top_n, sort_by=sort_by)

def get_leaderboard_stats() -> Dict[str, Any]:
    """Get leaderboard statistics using the global leaderboard."""
    return get_leaderboard().get_stats()