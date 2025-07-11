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

    def test_leaderboard_re_ranking_after_evaluation(self):
        """Test that leaderboard re-ranks after each evaluation."""
        print("\n🏆 Testing Leaderboard Re-ranking After Each Evaluation")
        
        # Add multiple agents with different performance levels
        agents_data = [
            {"name": "agent_a", "sharpe": 1.0, "drawdown": 0.15, "win_rate": 0.60, "return": 0.20},
            {"name": "agent_b", "sharpe": 1.5, "drawdown": 0.10, "win_rate": 0.65, "return": 0.25},
            {"name": "agent_c", "sharpe": 0.8, "drawdown": 0.20, "win_rate": 0.55, "return": 0.15},
            {"name": "agent_d", "sharpe": 2.0, "drawdown": 0.08, "win_rate": 0.70, "return": 0.30},
        ]
        
        # Track rankings after each update
        rankings_history = []
        
        for i, agent_data in enumerate(agents_data):
            print(f"\n📊 Adding agent {agent_data['name']} (iteration {i+1})")
            
            # Update performance
            self.leaderboard.update_performance(
                agent_name=agent_data["name"],
                sharpe_ratio=agent_data["sharpe"],
                max_drawdown=agent_data["drawdown"],
                win_rate=agent_data["win_rate"],
                total_return=agent_data["return"]
            )
            
            # Get current ranking
            current_ranking = self.leaderboard.get_leaderboard(sort_by="sharpe_ratio")
            current_ranks = [perf.agent_name for perf in current_ranking]
            
            print(f"  Current ranking: {current_ranks}")
            rankings_history.append(current_ranks.copy())
            
            # Verify ranking changes after each addition
            if i > 0:
                previous_ranks = rankings_history[i-1]
                if current_ranks != previous_ranks:
                    print(f"  ✅ Ranking changed: {previous_ranks} -> {current_ranks}")
                else:
                    print(f"  ⚠️ Ranking unchanged: {current_ranks}")
                
                # Assert that rankings are properly ordered by Sharpe ratio
                current_sharpes = [self.leaderboard.leaderboard[name].sharpe_ratio for name in current_ranks]
                expected_sharpes = sorted(current_sharpes, reverse=True)
                self.assertEqual(current_sharpes, expected_sharpes, 
                               f"Rankings not properly ordered by Sharpe ratio: {current_sharpes}")
        
        # Test ranking by different metrics
        print(f"\n📈 Testing different ranking metrics:")
        
        # Test ranking by total return
        return_ranking = self.leaderboard.get_leaderboard(sort_by="total_return")
        return_ranks = [perf.agent_name for perf in return_ranking]
        return_values = [self.leaderboard.leaderboard[name].total_return for name in return_ranks]
        expected_returns = sorted(return_values, reverse=True)
        self.assertEqual(return_values, expected_returns, 
                        f"Rankings not properly ordered by total return: {return_values}")
        print(f"  Return ranking: {return_ranks}")
        
        # Test ranking by win rate
        winrate_ranking = self.leaderboard.get_leaderboard(sort_by="win_rate")
        winrate_ranks = [perf.agent_name for perf in winrate_ranking]
        winrate_values = [self.leaderboard.leaderboard[name].win_rate for name in winrate_ranks]
        expected_winrates = sorted(winrate_values, reverse=True)
        self.assertEqual(winrate_values, expected_winrates, 
                        f"Rankings not properly ordered by win rate: {winrate_values}")
        print(f"  Win rate ranking: {winrate_ranks}")
        
        # Test ranking by max drawdown (ascending - lower is better)
        drawdown_ranking = self.leaderboard.get_leaderboard(sort_by="max_drawdown")
        drawdown_ranks = [perf.agent_name for perf in drawdown_ranking]
        drawdown_values = [self.leaderboard.leaderboard[name].max_drawdown for name in drawdown_ranks]
        expected_drawdowns = sorted(drawdown_values)  # Ascending order
        self.assertEqual(drawdown_values, expected_drawdowns, 
                        f"Rankings not properly ordered by max drawdown: {drawdown_values}")
        print(f"  Drawdown ranking: {drawdown_ranks}")
        
        # Test that top agent is consistent across different metrics
        top_sharpe = rankings_history[-1][0]  # Top by Sharpe
        top_return = return_ranks[0]  # Top by return
        top_winrate = winrate_ranks[0]  # Top by win rate
        
        print(f"  Top by Sharpe: {top_sharpe}")
        print(f"  Top by Return: {top_return}")
        print(f"  Top by Win Rate: {top_winrate}")
        
        # Verify that agent_d should be top by Sharpe (2.0)
        self.assertEqual(top_sharpe, "agent_d", f"Expected agent_d to be top by Sharpe, got {top_sharpe}")
        
        # Test ranking stability after multiple updates
        print(f"\n🔄 Testing ranking stability after multiple updates:")
        
        # Update the top agent multiple times with same performance
        for i in range(3):
            self.leaderboard.update_performance(
                agent_name="agent_d",
                sharpe_ratio=2.0,
                max_drawdown=0.08,
                win_rate=0.70,
                total_return=0.30
            )
            
            stable_ranking = self.leaderboard.get_leaderboard(sort_by="sharpe_ratio")
            stable_ranks = [perf.agent_name for perf in stable_ranking]
            
            print(f"  Update {i+1}: {stable_ranks}")
            
            # Ranking should remain stable
            self.assertEqual(stable_ranks[0], "agent_d", 
                           f"Top agent should remain agent_d after update {i+1}")
        
        print("✅ Leaderboard re-ranking test completed")


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