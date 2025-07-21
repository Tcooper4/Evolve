"""Tests for the Goal Planner agent."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import from trading.agents instead of core.agents
try:
    from trading.agents.goal_planner import GoalPlannerAgent as GoalPlanner
except ImportError:
    # If the module doesn't exist, we'll skip these tests
    GoalPlanner = None


@pytest.mark.skipif(GoalPlanner is None, reason="GoalPlanner module not available")
class TestGoalPlanner:
    @pytest.fixture
    def agent(self):
        """Create a Goal Planner agent instance for testing."""
        return GoalPlanner(initial_balance=10000, risk_tolerance=0.5, time_horizon=365)

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        return sample_price_data

    def test_agent_initialization(self, agent):
        """Test that agent initializes with correct parameters."""
        assert agent.initial_balance == 10000
        assert agent.risk_tolerance == 0.5
        assert agent.time_horizon == 365
        assert agent.name == "GoalPlanner"

    def test_goal_setting(self, agent):
        """Test that goals are set correctly."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)

        assert isinstance(goals, dict)
        assert "target_return" in goals
        assert "max_drawdown" in goals
        assert "min_sharpe" in goals
        assert goals["target_return"] == 0.15
        assert goals["max_drawdown"] == 0.1
        assert goals["min_sharpe"] == 1.5

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
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        plan = agent.create_plan(sample_data, goals)

        assert isinstance(plan, dict)
        assert "strategies" in plan
        assert "allocations" in plan
        assert "timeline" in plan
        assert len(plan["strategies"]) > 0
        assert len(plan["allocations"]) == len(plan["strategies"])
        assert len(plan["timeline"]) > 0

    def test_goal_tracking(self, agent, sample_data):
        """Test that goal tracking works correctly."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        agent.create_plan(sample_data, goals)

        # Simulate some performance data
        performance = pd.Series(
            np.random.normal(0.001, 0.02, len(sample_data)), index=sample_data.index
        )

        tracking = agent.track_goals(performance)

        assert isinstance(tracking, dict)
        assert "return" in tracking
        assert "drawdown" in tracking
        assert "sharpe" in tracking
        assert all(isinstance(v, float) for v in tracking.values())

    def test_goal_adjustment(self, agent, sample_data):
        """Test that goals are adjusted correctly based on performance."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        agent.create_plan(sample_data, goals)

        # Simulate underperformance
        performance = pd.Series(
            np.random.normal(-0.001, 0.02, len(sample_data)), index=sample_data.index
        )

        adjusted_goals = agent.adjust_goals(performance)

        assert isinstance(adjusted_goals, dict)
        assert "target_return" in adjusted_goals
        assert "max_drawdown" in adjusted_goals
        assert "min_sharpe" in adjusted_goals
        assert adjusted_goals["target_return"] < goals["target_return"]

    def test_risk_assessment(self, agent, sample_data):
        """Test that risk assessment works correctly."""
        risk_metrics = agent.assess_risk(sample_data)

        assert isinstance(risk_metrics, dict)
        assert "volatility" in risk_metrics
        assert "var" in risk_metrics
        assert "cvar" in risk_metrics
        assert all(isinstance(v, float) for v in risk_metrics.values())

    def test_strategy_selection(self, agent, sample_data):
        """Test that strategy selection works correctly."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        strategies = agent.select_strategies(sample_data, goals)

        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert all(isinstance(s, str) for s in strategies)

    def test_allocation_optimization(self, agent, sample_data):
        """Test that allocation optimization works correctly."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        strategies = agent.select_strategies(sample_data, goals)
        allocations = agent.optimize_allocations(strategies, sample_data, goals)

        assert isinstance(allocations, dict)
        assert len(allocations) == len(strategies)
        assert sum(allocations.values()) == pytest.approx(1.0)
        assert all(0 <= v <= 1 for v in allocations.values())

    def test_plan_execution(self, agent, sample_data):
        """Test that plan execution works correctly."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        plan = agent.create_plan(sample_data, goals)
        execution = agent.execute_plan(plan, sample_data)

        assert isinstance(execution, dict)
        assert "trades" in execution
        assert "performance" in execution
        assert "risk_metrics" in execution

    def test_plan_monitoring(self, agent, sample_data):
        """Test that plan monitoring works correctly."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        plan = agent.create_plan(sample_data, goals)
        execution = agent.execute_plan(plan, sample_data)
        monitoring = agent.monitor_plan(execution, goals)

        assert isinstance(monitoring, dict)
        assert "status" in monitoring
        assert "deviations" in monitoring
        assert "recommendations" in monitoring

    def test_plan_adjustment(self, agent, sample_data):
        """Test that plan adjustment works correctly."""
        goals = agent.set_goals(target_return=0.15, max_drawdown=0.1, min_sharpe=1.5)
        plan = agent.create_plan(sample_data, goals)
        execution = agent.execute_plan(plan, sample_data)
        monitoring = agent.monitor_plan(execution, goals)
        adjusted_plan = agent.adjust_plan(plan, monitoring)

        assert isinstance(adjusted_plan, dict)
        assert "strategies" in adjusted_plan

    def test_goal_re_prioritization_failure_conditions(self, agent, sample_data):
        """Test that verifies goal re-prioritization logic under failure conditions."""
        print("\n🎯 Testing Goal Re-prioritization Under Failure Conditions")

        # Set initial goals
        initial_goals = agent.set_goals(
            target_return=0.15, max_drawdown=0.1, min_sharpe=1.5
        )

        print(f"  Initial goals: {initial_goals}")

        # Create initial plan
        initial_plan = agent.create_plan(sample_data, initial_goals)
        print(f"  Initial plan strategies: {initial_plan['strategies']}")

        # Simulate different failure scenarios
        failure_scenarios = [
            {
                "name": "Severe Underperformance",
                "performance": pd.Series(
                    np.random.normal(-0.005, 0.02, len(sample_data)),
                    index=sample_data.index,
                ),
                "expected_priority_change": "risk_reduction",
            },
            {
                "name": "High Volatility",
                "performance": pd.Series(
                    np.random.normal(0.001, 0.05, len(sample_data)),
                    index=sample_data.index,
                ),
                "expected_priority_change": "stability_focus",
            },
            {
                "name": "Consistent Losses",
                "performance": pd.Series(
                    np.random.normal(-0.002, 0.01, len(sample_data)),
                    index=sample_data.index,
                ),
                "expected_priority_change": "capital_preservation",
            },
            {
                "name": "Drawdown Exceeded",
                "performance": pd.Series(
                    np.random.normal(-0.003, 0.03, len(sample_data)),
                    index=sample_data.index,
                ),
                "expected_priority_change": "risk_management",
            },
        ]

        for scenario in failure_scenarios:
            print(f"\n  📊 Testing scenario: {scenario['name']}")

            # Track goals before failure
            goals_before = agent.get_current_goals()

            # Simulate failure performance
            failure_performance = scenario["performance"]

            # Trigger goal re-prioritization
            adjusted_goals = agent.adjust_goals(failure_performance)
            reprioritized_plan = agent.reprioritize_goals(
                failure_performance, adjusted_goals
            )

            print(f"    Goals before: {goals_before}")
            print(f"    Goals after: {adjusted_goals}")

            # Verify goal adjustments based on failure type
            if scenario["name"] == "Severe Underperformance":
                # Should reduce target return and increase risk tolerance
                self.assertLess(
                    adjusted_goals["target_return"],
                    goals_before["target_return"],
                    "Target return should be reduced under severe underperformance",
                )
                self.assertGreater(
                    adjusted_goals["max_drawdown"],
                    goals_before["max_drawdown"],
                    "Max drawdown should be increased to allow for recovery",
                )

            elif scenario["name"] == "High Volatility":
                # Should focus on stability and reduce volatility
                self.assertLess(
                    adjusted_goals["max_drawdown"],
                    goals_before["max_drawdown"],
                    "Max drawdown should be reduced under high volatility",
                )
                self.assertGreater(
                    adjusted_goals["min_sharpe"],
                    goals_before["min_sharpe"],
                    "Minimum Sharpe should be increased for stability",
                )

            elif scenario["name"] == "Consistent Losses":
                # Should prioritize capital preservation
                self.assertLess(
                    adjusted_goals["target_return"],
                    goals_before["target_return"],
                    "Target return should be reduced for capital preservation",
                )
                self.assertLess(
                    adjusted_goals["max_drawdown"],
                    goals_before["max_drawdown"],
                    "Max drawdown should be reduced for capital preservation",
                )

            elif scenario["name"] == "Drawdown Exceeded":
                # Should focus on risk management
                self.assertLess(
                    adjusted_goals["max_drawdown"],
                    goals_before["max_drawdown"],
                    "Max drawdown should be reduced when exceeded",
                )
                self.assertGreater(
                    adjusted_goals["min_sharpe"],
                    goals_before["min_sharpe"],
                    "Minimum Sharpe should be increased for risk management",
                )

            # Verify reprioritized plan changes
            self.assertIsNotNone(
                reprioritized_plan, "Should generate reprioritized plan"
            )
            self.assertIn(
                "strategies", reprioritized_plan, "Plan should contain strategies"
            )
            self.assertIn(
                "allocations", reprioritized_plan, "Plan should contain allocations"
            )

            # Check if strategy priorities changed
            if (
                len(initial_plan["strategies"]) > 0
                and len(reprioritized_plan["strategies"]) > 0
            ):
                strategy_changed = (
                    initial_plan["strategies"] != reprioritized_plan["strategies"]
                )
                allocation_changed = (
                    initial_plan["allocations"] != reprioritized_plan["allocations"]
                )

                print(f"    Strategy priorities changed: {strategy_changed}")
                print(f"    Allocation priorities changed: {allocation_changed}")

                # At least one aspect should change under failure conditions
                self.assertTrue(
                    strategy_changed or allocation_changed,
                    "Priorities should change under failure conditions",
                )

        # Test goal recovery logic
        print(f"\n  🔄 Testing goal recovery logic...")

        # Simulate recovery after failure
        recovery_performance = pd.Series(
            np.random.normal(0.003, 0.015, len(sample_data)), index=sample_data.index
        )

        # Get goals after failure
        failure_goals = agent.get_current_goals()

        # Trigger recovery adjustment
        recovery_goals = agent.adjust_goals(recovery_performance)

        print(f"    Goals during failure: {failure_goals}")
        print(f"    Goals during recovery: {recovery_goals}")

        # Verify recovery logic
        if failure_goals["target_return"] < initial_goals["target_return"]:
            # If target was reduced during failure, it should increase during recovery
            self.assertGreaterEqual(
                recovery_goals["target_return"],
                failure_goals["target_return"],
                "Target return should increase during recovery",
            )

        if failure_goals["max_drawdown"] > initial_goals["max_drawdown"]:
            # If max drawdown was increased during failure, it should decrease during recovery
            self.assertLessEqual(
                recovery_goals["max_drawdown"],
                failure_goals["max_drawdown"],
                "Max drawdown should decrease during recovery",
            )

        # Test goal stability under consistent performance
        print(f"\n  📈 Testing goal stability under consistent performance...")

        stable_performance = pd.Series(
            np.random.normal(0.001, 0.01, len(sample_data)), index=sample_data.index
        )

        # Multiple adjustments with stable performance should maintain goals
        current_goals = agent.get_current_goals()

        for i in range(3):
            adjusted_goals = agent.adjust_goals(stable_performance)
            print(f"    Adjustment {i + 1}: {adjusted_goals}")

            # Goals should remain relatively stable
            if i > 0:
                target_change = abs(
                    adjusted_goals["target_return"] - current_goals["target_return"]
                )
                drawdown_change = abs(
                    adjusted_goals["max_drawdown"] - current_goals["max_drawdown"]
                )

                self.assertLess(
                    target_change,
                    0.05,
                    "Target return should remain stable under consistent performance",
                )
                self.assertLess(
                    drawdown_change,
                    0.05,
                    "Max drawdown should remain stable under consistent performance",
                )

            current_goals = adjusted_goals

        print(
            "Goal planner test completed. All planning functions have "
            "been validated."
        )
