"""Tests for the self-improving agent."""

import pytest
from unittest.mock import Mock, patch
from trading.agents.self_improving_agent import SelfImprovingAgent
from trading.agents.performance_tracker import PerformanceTracker
from trading.agents.goal_planner import GoalPlanner

class TestSelfImprovingAgent:
    @pytest.fixture
    def agent(self):
        """Create a self-improving agent instance for testing."""
        return SelfImprovingAgent()

    @pytest.fixture
    def performance_tracker(self):
        """Create a performance tracker for testing."""
        return PerformanceTracker()

    @pytest.fixture
    def goal_planner(self):
        """Create a goal planner for testing."""
        return GoalPlanner()

    def test_agent_initialization(self, agent):
        """Test that the agent initializes correctly."""
        assert agent is not None
        assert hasattr(agent, 'performance_tracker')
        assert hasattr(agent, 'goal_planner')
        assert hasattr(agent, 'strategy_selector')

    def test_performance_tracking(self, agent, performance_tracker):
        """Test that performance is tracked correctly."""
        # Mock performance data
        performance_data = {
            'strategy': 'RSI',
            'returns': 0.05,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.1
        }
        
        # Track performance
        agent.track_performance(performance_data)
        
        # Verify performance was tracked
        assert agent.performance_tracker.get_performance('RSI') is not None
        assert agent.performance_tracker.get_performance('RSI')['returns'] == 0.05

    def test_goal_planning(self, agent, goal_planner):
        """Test that goals are planned correctly."""
        # Create a test goal
        goal = {
            'type': 'maximize_returns',
            'constraints': {
                'max_drawdown': -0.2,
                'min_sharpe': 1.0
            }
        }
        
        # Plan goal
        plan = agent.plan_goal(goal)
        
        # Verify plan
        assert plan is not None
        assert 'actions' in plan
        assert 'metrics' in plan
        assert len(plan['actions']) > 0

    def test_strategy_selection(self, agent):
        """Test that strategies are selected correctly."""
        # Mock market conditions
        market_conditions = {
            'volatility': 'high',
            'trend': 'bullish',
            'volume': 'normal'
        }
        
        # Select strategy
        strategy = agent.select_strategy(market_conditions)
        
        # Verify strategy selection
        assert strategy is not None
        assert isinstance(strategy, str)
        assert strategy in ['RSI', 'MACD', 'Bollinger', 'SMA']

    def test_learning_process(self, agent):
        """Test that the agent learns from experience."""
        # Mock historical performance
        historical_performance = [
            {'strategy': 'RSI', 'returns': 0.05, 'sharpe_ratio': 1.5},
            {'strategy': 'MACD', 'returns': 0.03, 'sharpe_ratio': 1.2}
        ]
        
        # Learn from performance
        agent.learn_from_performance(historical_performance)
        
        # Verify learning
        assert agent.performance_tracker.get_best_strategy() is not None
        assert agent.performance_tracker.get_best_strategy() == 'RSI'

    def test_goal_adaptation(self, agent):
        """Test that goals are adapted based on performance."""
        # Mock initial goal
        initial_goal = {
            'type': 'maximize_returns',
            'constraints': {
                'max_drawdown': -0.2,
                'min_sharpe': 1.0
            }
        }
        
        # Mock performance
        performance = {
            'returns': 0.03,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.15
        }
        
        # Adapt goal
        adapted_goal = agent.adapt_goal(initial_goal, performance)
        
        # Verify goal adaptation
        assert adapted_goal is not None
        assert adapted_goal['constraints']['min_sharpe'] < initial_goal['constraints']['min_sharpe']

    def test_error_handling(self, agent):
        """Test that the agent handles errors gracefully."""
        # Test with invalid performance data
        with pytest.raises(ValueError):
            agent.track_performance({})
        
        # Test with invalid goal
        with pytest.raises(ValueError):
            agent.plan_goal({})
        
        # Test with invalid market conditions
        with pytest.raises(ValueError):
            agent.select_strategy({})

    @patch('trading.agents.self_improving_agent.SelfImprovingAgent.learn_from_performance')
    def test_end_to_end_improvement(self, mock_learn, agent):
        """Test end-to-end self-improvement process."""
        # Mock performance data
        performance_data = {
            'strategy': 'RSI',
            'returns': 0.05,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.1
        }
        
        # Process improvement
        agent.improve(performance_data)
        
        # Verify improvement process
        assert mock_learn.called
        assert agent.performance_tracker.get_performance('RSI') is not None 