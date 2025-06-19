"""Tests for the Goal Planner agent."""

import pytest
import pandas as pd
import numpy as np
from trading.agents.goal_planner import GoalPlanner

class TestGoalPlanner:
    @pytest.fixture
    def agent(self):
        """Create a Goal Planner agent instance for testing."""
        return GoalPlanner(
            initial_balance=10000,
            risk_tolerance=0.5,
            time_horizon=365
        )

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_agent_initialization(self, agent):
        """Test that agent initializes with correct parameters."""
        assert agent.initial_balance == 10000
        assert agent.risk_tolerance == 0.5
        assert agent.time_horizon == 365
        assert agent.name == 'GoalPlanner'

    def test_goal_setting(self, agent):
        """Test that goals are set correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        
        assert isinstance(goals, dict)
        assert 'target_return' in goals
        assert 'max_drawdown' in goals
        assert 'min_sharpe' in goals
        assert goals['target_return'] == 0.15
        assert goals['max_drawdown'] == 0.1
        assert goals['min_sharpe'] == 1.5

    def test_goal_validation(self, agent):
        """Test that goals are validated correctly."""
        with pytest.raises(ValueError):
            agent.set_goals(target_return=-0.1)  # Invalid target return
        with pytest.raises(ValueError):
            agent.set_goals(max_drawdown=1.5)  # Invalid max drawdown
        with pytest.raises(ValueError):
            agent.set_goals(min_sharpe=-0.5)  # Invalid minimum Sharpe ratio

    def test_goal_planning(self, agent, sample_data):
        """Test that goal planning works correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        plan = agent.create_plan(sample_data, goals)
        
        assert isinstance(plan, dict)
        assert 'strategies' in plan
        assert 'allocations' in plan
        assert 'timeline' in plan
        assert len(plan['strategies']) > 0
        assert len(plan['allocations']) == len(plan['strategies'])
        assert len(plan['timeline']) > 0

    def test_goal_tracking(self, agent, sample_data):
        """Test that goal tracking works correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        agent.create_plan(sample_data, goals)
        
        # Simulate some performance data
        performance = pd.Series(
            np.random.normal(0.001, 0.02, len(sample_data)),
            index=sample_data.index
        )
        
        tracking = agent.track_goals(performance)
        
        assert isinstance(tracking, dict)
        assert 'return' in tracking
        assert 'drawdown' in tracking
        assert 'sharpe' in tracking
        assert all(isinstance(v, float) for v in tracking.values())

    def test_goal_adjustment(self, agent, sample_data):
        """Test that goals are adjusted correctly based on performance."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        agent.create_plan(sample_data, goals)
        
        # Simulate underperformance
        performance = pd.Series(
            np.random.normal(-0.001, 0.02, len(sample_data)),
            index=sample_data.index
        )
        
        adjusted_goals = agent.adjust_goals(performance)
        
        assert isinstance(adjusted_goals, dict)
        assert 'target_return' in adjusted_goals
        assert 'max_drawdown' in adjusted_goals
        assert 'min_sharpe' in adjusted_goals
        assert adjusted_goals['target_return'] < goals['target_return']

    def test_risk_assessment(self, agent, sample_data):
        """Test that risk assessment works correctly."""
        risk_metrics = agent.assess_risk(sample_data)
        
        assert isinstance(risk_metrics, dict)
        assert 'volatility' in risk_metrics
        assert 'var' in risk_metrics
        assert 'cvar' in risk_metrics
        assert all(isinstance(v, float) for v in risk_metrics.values())

    def test_strategy_selection(self, agent, sample_data):
        """Test that strategy selection works correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        strategies = agent.select_strategies(sample_data, goals)
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert all(isinstance(s, str) for s in strategies)

    def test_allocation_optimization(self, agent, sample_data):
        """Test that allocation optimization works correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        strategies = agent.select_strategies(sample_data, goals)
        allocations = agent.optimize_allocations(strategies, sample_data, goals)
        
        assert isinstance(allocations, dict)
        assert len(allocations) == len(strategies)
        assert sum(allocations.values()) == pytest.approx(1.0)
        assert all(0 <= v <= 1 for v in allocations.values())

    def test_plan_execution(self, agent, sample_data):
        """Test that plan execution works correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        plan = agent.create_plan(sample_data, goals)
        execution = agent.execute_plan(plan, sample_data)
        
        assert isinstance(execution, dict)
        assert 'trades' in execution
        assert 'performance' in execution
        assert 'risk_metrics' in execution

    def test_plan_monitoring(self, agent, sample_data):
        """Test that plan monitoring works correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        plan = agent.create_plan(sample_data, goals)
        execution = agent.execute_plan(plan, sample_data)
        monitoring = agent.monitor_plan(execution, goals)
        
        assert isinstance(monitoring, dict)
        assert 'status' in monitoring
        assert 'deviations' in monitoring
        assert 'recommendations' in monitoring

    def test_plan_adjustment(self, agent, sample_data):
        """Test that plan adjustment works correctly."""
        goals = agent.set_goals(
            target_return=0.15,
            max_drawdown=0.1,
            min_sharpe=1.5
        )
        plan = agent.create_plan(sample_data, goals)
        execution = agent.execute_plan(plan, sample_data)
        monitoring = agent.monitor_plan(execution, goals)
        adjusted_plan = agent.adjust_plan(plan, monitoring)
        
        assert isinstance(adjusted_plan, dict)
        assert 'strategies' in adjusted_plan
        assert 'allocations' in adjusted_plan
        assert 'timeline' in adjusted_plan 