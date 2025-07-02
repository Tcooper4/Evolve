"""
Tests for AgentLeaderboard

Comprehensive test suite for the AgentLeaderboard class covering performance tracking,
deprecation logic, leaderboard retrieval, and edge cases.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from trading.agents.agent_leaderboard import AgentLeaderboard, AgentPerformance


class TestAgentPerformance:
    """Test the AgentPerformance dataclass."""
    
    def test_agent_performance_creation(self):
        """Test creating an AgentPerformance instance."""
        perf = AgentPerformance(
            agent_name="test_agent",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.65,
            total_return=0.25
        )
        
        assert perf.agent_name == "test_agent"
        assert perf.sharpe_ratio == 1.5
        assert perf.max_drawdown == 0.15
        assert perf.win_rate == 0.65
        assert perf.total_return == 0.25
        assert perf.status == "active"
        assert isinstance(perf.last_updated, datetime)
        assert perf.extra_metrics == {}
    
    def test_agent_performance_with_extra_metrics(self):
        """Test creating AgentPerformance with extra metrics."""
        extra_metrics = {
            "volatility": 0.20,
            "calmar_ratio": 2.0,
            "profit_factor": 1.8
        }
        
        perf = AgentPerformance(
            agent_name="test_agent",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.65,
            total_return=0.25,
            extra_metrics=extra_metrics
        )
        
        assert perf.extra_metrics == extra_metrics
    
    def test_to_dict_method(self):
        """Test the to_dict method."""
        perf = AgentPerformance(
            agent_name="test_agent",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.65,
            total_return=0.25
        )
        
        result = perf.to_dict()
        
        assert isinstance(result, dict)
        assert result["agent_name"] == "test_agent"
        assert result["sharpe_ratio"] == 1.5
        assert result["status"] == "active"
        assert "last_updated" in result
        assert isinstance(result["last_updated"], str)  # ISO format


class TestAgentLeaderboard:
    """Test the AgentLeaderboard class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.leaderboard = AgentLeaderboard()
    
    def test_initialization(self):
        """Test leaderboard initialization."""
        assert self.leaderboard.leaderboard == {}
        assert self.leaderboard.history == []
        assert self.leaderboard.deprecation_thresholds == {
            'sharpe_ratio': 0.5,
            'max_drawdown': 0.25,
            'win_rate': 0.45
        }
    
    def test_custom_deprecation_thresholds(self):
        """Test initialization with custom deprecation thresholds."""
        custom_thresholds = {
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.20,
            'win_rate': 0.50
        }
        
        leaderboard = AgentLeaderboard(deprecation_thresholds=custom_thresholds)
        assert leaderboard.deprecation_thresholds == custom_thresholds
    
    def test_update_performance_new_agent(self):
        """Test updating performance for a new agent."""
        self.leaderboard.update_performance(
            agent_name="test_agent",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.65,
            total_return=0.25
        )
        
        assert "test_agent" in self.leaderboard.leaderboard
        perf = self.leaderboard.leaderboard["test_agent"]
        assert perf.sharpe_ratio == 1.5
        assert perf.max_drawdown == 0.15
        assert perf.win_rate == 0.65
        assert perf.total_return == 0.25
        assert perf.status == "active"
        assert len(self.leaderboard.history) == 1
    
    def test_update_performance_existing_agent(self):
        """Test updating performance for an existing agent."""
        # Add initial performance
        self.leaderboard.update_performance(
            agent_name="test_agent",
            sharpe_ratio=1.0,
            max_drawdown=0.20,
            win_rate=0.60,
            total_return=0.20
        )
        
        # Update performance
        self.leaderboard.update_performance(
            agent_name="test_agent",
            sharpe_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.70,
            total_return=0.30
        )
        
        perf = self.leaderboard.leaderboard["test_agent"]
        assert perf.sharpe_ratio == 2.0
        assert perf.max_drawdown == 0.10
        assert perf.win_rate == 0.70
        assert perf.total_return == 0.30
        assert len(self.leaderboard.history) == 2
    
    def test_update_performance_with_extra_metrics(self):
        """Test updating performance with extra metrics."""
        extra_metrics = {
            "volatility": 0.20,
            "calmar_ratio": 2.0,
            "profit_factor": 1.8
        }
        
        self.leaderboard.update_performance(
            agent_name="test_agent",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.65,
            total_return=0.25,
            extra_metrics=extra_metrics
        )
        
        perf = self.leaderboard.leaderboard["test_agent"]
        assert perf.extra_metrics == extra_metrics
    
    def test_deprecation_low_sharpe(self):
        """Test deprecation due to low Sharpe ratio."""
        self.leaderboard.update_performance(
            agent_name="weak_agent",
            sharpe_ratio=0.3,  # Below threshold of 0.5
            max_drawdown=0.15,
            win_rate=0.65,
            total_return=0.25
        )
        
        perf = self.leaderboard.leaderboard["weak_agent"]
        assert perf.status == "deprecated"
    
    def test_deprecation_high_drawdown(self):
        """Test deprecation due to high drawdown."""
        self.leaderboard.update_performance(
            agent_name="risky_agent",
            sharpe_ratio=1.5,
            max_drawdown=0.30,  # Above threshold of 0.25
            win_rate=0.65,
            total_return=0.25
        )
        
        perf = self.leaderboard.leaderboard["risky_agent"]
        assert perf.status == "deprecated"
    
    def test_deprecation_low_win_rate(self):
        """Test deprecation due to low win rate."""
        self.leaderboard.update_performance(
            agent_name="unlucky_agent",
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            win_rate=0.40,  # Below threshold of 0.45
            total_return=0.25
        )
        
        perf = self.leaderboard.leaderboard["unlucky_agent"]
        assert perf.status == "deprecated"
    
    def test_no_deprecation_good_performance(self):
        """Test that good performance doesn't trigger deprecation."""
        self.leaderboard.update_performance(
            agent_name="good_agent",
            sharpe_ratio=1.5,  # Above threshold
            max_drawdown=0.20,  # Below threshold
            win_rate=0.60,  # Above threshold
            total_return=0.25
        )
        
        perf = self.leaderboard.leaderboard["good_agent"]
        assert perf.status == "active"
    
    def test_get_leaderboard_default(self):
        """Test getting leaderboard with default parameters."""
        # Add multiple agents
        agents_data = [
            ("agent1", 1.5, 0.15, 0.65, 0.25),
            ("agent2", 2.0, 0.10, 0.70, 0.30),
            ("agent3", 1.0, 0.20, 0.60, 0.20),
        ]
        
        for agent_name, sharpe, drawdown, win_rate, total_return in agents_data:
            self.leaderboard.update_performance(
                agent_name, sharpe, drawdown, win_rate, total_return
            )
        
        result = self.leaderboard.get_leaderboard()
        
        assert len(result) == 3
        # Should be sorted by Sharpe ratio (default)
        assert result[0]["agent_name"] == "agent2"  # Highest Sharpe
        assert result[1]["agent_name"] == "agent1"
        assert result[2]["agent_name"] == "agent3"  # Lowest Sharpe
    
    def test_get_leaderboard_custom_sort(self):
        """Test getting leaderboard with custom sort parameter."""
        # Add multiple agents
        agents_data = [
            ("agent1", 1.5, 0.15, 0.65, 0.25),
            ("agent2", 2.0, 0.10, 0.70, 0.30),
            ("agent3", 1.0, 0.20, 0.60, 0.20),
        ]
        
        for agent_name, sharpe, drawdown, win_rate, total_return in agents_data:
            self.leaderboard.update_performance(
                agent_name, sharpe, drawdown, win_rate, total_return
            )
        
        result = self.leaderboard.get_leaderboard(sort_by="total_return")
        
        assert len(result) == 3
        # Should be sorted by total return
        assert result[0]["agent_name"] == "agent2"  # Highest return
        assert result[1]["agent_name"] == "agent1"
        assert result[2]["agent_name"] == "agent3"  # Lowest return
    
    def test_get_leaderboard_top_n(self):
        """Test getting leaderboard with top_n limit."""
        # Add multiple agents
        agents_data = [
            ("agent1", 1.5, 0.15, 0.65, 0.25),
            ("agent2", 2.0, 0.10, 0.70, 0.30),
            ("agent3", 1.0, 0.20, 0.60, 0.20),
            ("agent4", 1.8, 0.12, 0.68, 0.28),
        ]
        
        for agent_name, sharpe, drawdown, win_rate, total_return in agents_data:
            self.leaderboard.update_performance(
                agent_name, sharpe, drawdown, win_rate, total_return
            )
        
        result = self.leaderboard.get_leaderboard(top_n=2)
        
        assert len(result) == 2
        assert result[0]["agent_name"] == "agent2"  # Highest Sharpe
        assert result[1]["agent_name"] == "agent4"  # Second highest Sharpe
    
    def test_get_deprecated_agents(self):
        """Test getting list of deprecated agents."""
        # Add agents with different statuses
        self.leaderboard.update_performance("active_agent", 1.5, 0.15, 0.65, 0.25)
        self.leaderboard.update_performance("deprecated_agent1", 0.3, 0.15, 0.65, 0.25)  # Low Sharpe
        self.leaderboard.update_performance("deprecated_agent2", 1.5, 0.30, 0.65, 0.25)  # High drawdown
        
        deprecated = self.leaderboard.get_deprecated_agents()
        
        assert len(deprecated) == 2
        assert "deprecated_agent1" in deprecated
        assert "deprecated_agent2" in deprecated
        assert "active_agent" not in deprecated
    
    def test_get_active_agents(self):
        """Test getting list of active agents."""
        # Add agents with different statuses
        self.leaderboard.update_performance("active_agent1", 1.5, 0.15, 0.65, 0.25)
        self.leaderboard.update_performance("active_agent2", 2.0, 0.10, 0.70, 0.30)
        self.leaderboard.update_performance("deprecated_agent", 0.3, 0.15, 0.65, 0.25)
        
        active = self.leaderboard.get_active_agents()
        
        assert len(active) == 2
        assert "active_agent1" in active
        assert "active_agent2" in active
        assert "deprecated_agent" not in active
    
    def test_get_history(self):
        """Test getting performance history."""
        # Add multiple updates
        self.leaderboard.update_performance("agent1", 1.0, 0.15, 0.65, 0.25)
        self.leaderboard.update_performance("agent2", 1.5, 0.10, 0.70, 0.30)
        self.leaderboard.update_performance("agent1", 1.2, 0.12, 0.68, 0.28)  # Update existing
        
        history = self.leaderboard.get_history()
        
        assert len(history) == 3
        assert all(isinstance(entry, dict) for entry in history)
        assert all("agent_name" in entry for entry in history)
    
    def test_get_history_limit(self):
        """Test getting performance history with limit."""
        # Add multiple updates
        for i in range(10):
            self.leaderboard.update_performance(f"agent{i}", 1.0 + i*0.1, 0.15, 0.65, 0.25)
        
        history = self.leaderboard.get_history(limit=5)
        
        assert len(history) == 5
        assert history[-1]["agent_name"] == "agent4"  # Last entry in limited history
    
    def test_as_dataframe(self):
        """Test converting leaderboard to pandas DataFrame."""
        # Add some agents
        agents_data = [
            ("agent1", 1.5, 0.15, 0.65, 0.25),
            ("agent2", 2.0, 0.10, 0.70, 0.30),
        ]
        
        for agent_name, sharpe, drawdown, win_rate, total_return in agents_data:
            self.leaderboard.update_performance(
                agent_name, sharpe, drawdown, win_rate, total_return
            )
        
        df = self.leaderboard.as_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == [
            'agent_name', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
            'total_return', 'last_updated', 'status', 'extra_metrics'
        ]
        assert df.iloc[0]['agent_name'] == 'agent1'
        assert df.iloc[1]['agent_name'] == 'agent2'
    
    def test_empty_leaderboard(self):
        """Test behavior with empty leaderboard."""
        # Test get_leaderboard with empty leaderboard
        result = self.leaderboard.get_leaderboard()
        assert result == []
        
        # Test get_deprecated_agents with empty leaderboard
        deprecated = self.leaderboard.get_deprecated_agents()
        assert deprecated == []
        
        # Test get_active_agents with empty leaderboard
        active = self.leaderboard.get_active_agents()
        assert active == []
        
        # Test get_history with empty leaderboard
        history = self.leaderboard.get_history()
        assert history == []
        
        # Test as_dataframe with empty leaderboard
        df = self.leaderboard.as_dataframe()
        assert df.empty
    
    def test_invalid_sort_parameter(self):
        """Test behavior with invalid sort parameter."""
        # Add an agent
        self.leaderboard.update_performance("agent1", 1.5, 0.15, 0.65, 0.25)
        
        # Test with invalid sort parameter
        with pytest.raises(AttributeError):
            self.leaderboard.get_leaderboard(sort_by="invalid_metric")
    
    def test_negative_values(self):
        """Test handling of negative performance values."""
        # Test with negative Sharpe ratio
        self.leaderboard.update_performance("negative_agent", -0.5, 0.15, 0.65, 0.25)
        
        perf = self.leaderboard.leaderboard["negative_agent"]
        assert perf.sharpe_ratio == -0.5
        assert perf.status == "deprecated"  # Should be deprecated due to low Sharpe
    
    def test_zero_values(self):
        """Test handling of zero performance values."""
        # Test with zero values
        self.leaderboard.update_performance("zero_agent", 0.0, 0.0, 0.0, 0.0)
        
        perf = self.leaderboard.leaderboard["zero_agent"]
        assert perf.sharpe_ratio == 0.0
        assert perf.max_drawdown == 0.0
        assert perf.win_rate == 0.0
        assert perf.total_return == 0.0
        assert perf.status == "deprecated"  # Should be deprecated due to low values
    
    def test_very_high_values(self):
        """Test handling of very high performance values."""
        # Test with very high values
        self.leaderboard.update_performance("super_agent", 10.0, 0.01, 0.95, 2.0)
        
        perf = self.leaderboard.leaderboard["super_agent"]
        assert perf.sharpe_ratio == 10.0
        assert perf.max_drawdown == 0.01
        assert perf.win_rate == 0.95
        assert perf.total_return == 2.0
        assert perf.status == "active"  # Should remain active
    
    def test_unicode_agent_names(self):
        """Test handling of unicode agent names."""
        # Test with unicode agent name
        self.leaderboard.update_performance("测试代理", 1.5, 0.15, 0.65, 0.25)
        
        assert "测试代理" in self.leaderboard.leaderboard
        perf = self.leaderboard.leaderboard["测试代理"]
        assert perf.agent_name == "测试代理"
        assert perf.status == "active"
    
    def test_large_number_of_agents(self):
        """Test performance with large number of agents."""
        # Add many agents
        for i in range(100):
            self.leaderboard.update_performance(
                f"agent_{i}", 
                1.0 + (i % 10) * 0.1, 
                0.10 + (i % 5) * 0.02, 
                0.50 + (i % 10) * 0.03, 
                0.20 + (i % 10) * 0.05
            )
        
        assert len(self.leaderboard.leaderboard) == 100
        assert len(self.leaderboard.history) == 100
        
        # Test leaderboard retrieval
        result = self.leaderboard.get_leaderboard(top_n=10)
        assert len(result) == 10
        
        # Test deprecated agents
        deprecated = self.leaderboard.get_deprecated_agents()
        assert len(deprecated) > 0  # Some should be deprecated
        
        # Test active agents
        active = self.leaderboard.get_active_agents()
        assert len(active) > 0  # Some should be active
        assert len(active) + len(deprecated) == 100


class TestAgentLeaderboardIntegration:
    """Integration tests for AgentLeaderboard."""
    
    def test_full_workflow(self):
        """Test complete workflow from performance update to deprecation."""
        leaderboard = AgentLeaderboard()
        
        # 1. Add initial agents
        leaderboard.update_performance("agent1", 1.5, 0.15, 0.65, 0.25)
        leaderboard.update_performance("agent2", 0.3, 0.15, 0.65, 0.25)  # Will be deprecated
        
        # 2. Check initial state
        assert len(leaderboard.get_active_agents()) == 1
        assert len(leaderboard.get_deprecated_agents()) == 1
        
        # 3. Update agent performance
        leaderboard.update_performance("agent1", 2.0, 0.10, 0.70, 0.30)
        
        # 4. Check updated state
        perf = leaderboard.leaderboard["agent1"]
        assert perf.sharpe_ratio == 2.0
        assert perf.status == "active"
        
        # 5. Get leaderboard
        top_agents = leaderboard.get_leaderboard(top_n=5)
        assert len(top_agents) == 2
        assert top_agents[0]["agent_name"] == "agent1"  # Higher Sharpe
        
        # 6. Export to DataFrame
        df = leaderboard.as_dataframe()
        assert len(df) == 2
        assert "agent1" in df["agent_name"].values
        assert "agent2" in df["agent_name"].values
    
    def test_concurrent_updates(self):
        """Test handling of concurrent performance updates."""
        import threading
        import time
        
        leaderboard = AgentLeaderboard()
        results = []
        
        def update_agent(agent_id):
            """Update agent performance in a thread."""
            for i in range(10):
                leaderboard.update_performance(
                    f"agent_{agent_id}",
                    1.0 + i * 0.1,
                    0.15,
                    0.65,
                    0.25
                )
                time.sleep(0.01)  # Small delay
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_agent, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(leaderboard.leaderboard) == 5
        assert len(leaderboard.history) == 50  # 5 agents * 10 updates each
        
        # All agents should be active (good performance)
        active_agents = leaderboard.get_active_agents()
        assert len(active_agents) == 5 