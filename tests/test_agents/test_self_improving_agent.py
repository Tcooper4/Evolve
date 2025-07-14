"""Tests for the self-improving agent."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestSelfImprovingAgent:
    """Test self-improving agent functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock self-improving agent for testing."""
        from core.agents.self_improving_agent import SelfImprovingAgent

        return SelfImprovingAgent()

    @pytest.fixture
    def sample_performance_data(self):
        """Create sample performance data for testing."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "accuracy": np.random.uniform(0.6, 0.9, 30),
                "sharpe_ratio": np.random.uniform(0.5, 2.0, 30),
                "max_drawdown": np.random.uniform(-0.2, -0.05, 30),
                "total_return": np.random.uniform(0.1, 0.5, 30),
                "win_rate": np.random.uniform(0.5, 0.8, 30),
            }
        )

    def test_agent_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert hasattr(mock_agent, "performance_history")
        assert hasattr(mock_agent, "learning_rate")
        assert hasattr(mock_agent, "improvement_threshold")
        assert hasattr(mock_agent, "strategy_weights")
        assert hasattr(mock_agent, "goal_planner")

    def test_performance_tracking(self, mock_agent, sample_performance_data):
        """Test that the agent tracks performance correctly."""
        # Add performance data
        for _, row in sample_performance_data.iterrows():
            mock_agent.track_performance(
                {
                    "date": row["date"],
                    "accuracy": row["accuracy"],
                    "sharpe_ratio": row["sharpe_ratio"],
                    "max_drawdown": row["max_drawdown"],
                    "total_return": row["total_return"],
                    "win_rate": row["win_rate"],
                }
            )

        # Check that performance history is updated
        assert len(mock_agent.performance_history) == len(sample_performance_data)
        assert isinstance(mock_agent.performance_history, pd.DataFrame)

    def test_goal_planning(self, mock_agent):
        """Test that the agent plans goals correctly."""
        current_performance = {
            "accuracy": 0.7,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15,
            "total_return": 0.25,
            "win_rate": 0.65,
        }

        with patch.object(mock_agent, "plan_goals") as mock_plan:
            mock_plan.return_value = {
                "target_accuracy": 0.8,
                "target_sharpe": 1.5,
                "target_drawdown": -0.1,
                "target_return": 0.35,
                "target_win_rate": 0.7,
                "improvement_actions": [
                    "Optimize RSI parameters",
                    "Add risk management rules",
                    "Improve entry/exit timing",
                ],
            }

            goals = mock_agent.plan_goals(current_performance)

            assert goals["target_accuracy"] > current_performance["accuracy"]
            assert goals["target_sharpe"] > current_performance["sharpe_ratio"]
            assert "improvement_actions" in goals

    def test_strategy_selection(self, mock_agent, sample_performance_data):
        """Test that the agent selects strategies based on performance."""
        # Add performance data
        mock_agent.performance_history = sample_performance_data

        market_conditions = {
            "volatility": "high",
            "trend": "upward",
            "volume": "increasing",
        }

        with patch.object(mock_agent, "select_strategy") as mock_select:
            mock_select.return_value = {
                "strategy": "momentum",
                "confidence": 0.85,
                "reasoning": "High volatility and upward trend favor momentum strategies",
                "parameters": {"lookback_period": 20, "threshold": 0.02},
            }

            strategy = mock_agent.select_strategy(market_conditions)

            assert strategy["strategy"] == "momentum"
            assert strategy["confidence"] > 0.5
            assert "parameters" in strategy

    def test_learning_process(self, mock_agent, sample_performance_data):
        """Test that the agent learns from historical performance."""
        # Add performance data
        mock_agent.performance_history = sample_performance_data

        with patch.object(mock_agent, "learn_from_history") as mock_learn:
            mock_learn.return_value = {
                "insights": [
                    "RSI strategy performs better in trending markets",
                    "MACD strategy shows higher win rate in volatile conditions",
                    "Bollinger Bands strategy has lower drawdown",
                ],
                "parameter_adjustments": {
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "bollinger_std": 2.0,
                },
                "strategy_weights": {"rsi": 0.4, "macd": 0.35, "bollinger": 0.25},
            }

            learning_result = mock_agent.learn_from_history()

            assert "insights" in learning_result
            assert "parameter_adjustments" in learning_result
            assert "strategy_weights" in learning_result

    def test_goal_adaptation(self, mock_agent):
        """Test that the agent adapts goals based on performance."""
        current_goals = {
            "target_accuracy": 0.8,
            "target_sharpe": 1.5,
            "target_drawdown": -0.1,
        }

        recent_performance = {
            "accuracy": 0.75,
            "sharpe_ratio": 1.3,
            "max_drawdown": -0.12,
        }

        with patch.object(mock_agent, "adapt_goals") as mock_adapt:
            mock_adapt.return_value = {
                "target_accuracy": 0.82,  # Slightly increased
                "target_sharpe": 1.6,  # Slightly increased
                "target_drawdown": -0.08,  # Slightly improved
                "adaptation_reason": "Performance is close to targets, increasing goals",
            }

            adapted_goals = mock_agent.adapt_goals(current_goals, recent_performance)

            assert adapted_goals["target_accuracy"] >= current_goals["target_accuracy"]
            assert adapted_goals["target_sharpe"] >= current_goals["target_sharpe"]
            assert adapted_goals["target_drawdown"] >= current_goals["target_drawdown"]

    def test_error_handling(self, mock_agent):
        """Test that the agent handles errors gracefully."""
        # Test with invalid performance data
        invalid_performance = {
            "accuracy": "invalid",
            "sharpe_ratio": None,
            "max_drawdown": "not_a_number",
        }

        with pytest.raises(ValueError):
            mock_agent.track_performance(invalid_performance)

        # Test with empty performance data
        empty_performance = {}

        with pytest.raises(ValueError):
            mock_agent.track_performance(empty_performance)

    def test_performance_analysis(self, mock_agent, sample_performance_data):
        """Test that the agent analyzes performance trends."""
        # Add performance data
        mock_agent.performance_history = sample_performance_data

        with patch.object(mock_agent, "analyze_performance") as mock_analyze:
            mock_analyze.return_value = {
                "trend": "improving",
                "volatility": "decreasing",
                "consistency": "high",
                "recommendations": [
                    "Continue current strategy mix",
                    "Consider reducing position sizes",
                    "Monitor drawdown closely",
                ],
            }

            analysis = mock_agent.analyze_performance()

            assert analysis["trend"] in ["improving", "declining", "stable"]
            assert "recommendations" in analysis

    def test_strategy_optimization(self, mock_agent):
        """Test that the agent optimizes strategy parameters."""
        current_parameters = {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "bollinger_window": 20,
            "bollinger_std": 2.0,
        }

        with patch.object(mock_agent, "optimize_parameters") as mock_optimize:
            mock_optimize.return_value = {
                "rsi_period": 16,
                "macd_fast": 10,
                "macd_slow": 24,
                "bollinger_window": 18,
                "bollinger_std": 2.2,
                "expected_improvement": 0.05,
                "confidence": 0.8,
            }

            optimized_params = mock_agent.optimize_parameters(current_parameters)

            assert optimized_params["expected_improvement"] > 0
            assert optimized_params["confidence"] > 0.5

    def test_continuous_improvement(self, mock_agent, sample_performance_data):
        """Test that the agent continuously improves over time."""
        # Simulate multiple improvement cycles
        initial_performance = {
            "accuracy": 0.6,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.25,
            "total_return": 0.15,
            "win_rate": 0.55,
        }

        mock_agent.track_performance(initial_performance)

        # Simulate improvement cycles
        for i in range(5):
            with patch.object(mock_agent, "improve") as mock_improve:
                mock_improve.return_value = {
                    "improvement_made": True,
                    "new_performance": {
                        "accuracy": 0.6 + (i + 1) * 0.02,
                        "sharpe_ratio": 0.8 + (i + 1) * 0.1,
                        "max_drawdown": -0.25 + (i + 1) * 0.02,
                        "total_return": 0.15 + (i + 1) * 0.03,
                        "win_rate": 0.55 + (i + 1) * 0.02,
                    },
                }

                improvement = mock_agent.improve()

                assert improvement["improvement_made"] is True
                assert (
                    improvement["new_performance"]["accuracy"]
                    > initial_performance["accuracy"]
                )

    def test_risk_management(self, mock_agent):
        """Test that the agent implements risk management."""
        current_risk_metrics = {
            "var_95": -0.02,
            "max_drawdown": -0.15,
            "volatility": 0.18,
            "beta": 1.1,
        }

        with patch.object(mock_agent, "assess_risk") as mock_assess:
            mock_assess.return_value = {
                "risk_level": "moderate",
                "risk_score": 0.6,
                "recommendations": [
                    "Reduce position sizes by 20%",
                    "Add stop-loss orders",
                    "Diversify across more strategies",
                ],
                "max_position_size": 0.15,
            }

            risk_assessment = mock_agent.assess_risk(current_risk_metrics)

            assert risk_assessment["risk_level"] in ["low", "moderate", "high"]
            assert (
                risk_assessment["risk_score"] >= 0
                and risk_assessment["risk_score"] <= 1
            )
            assert "recommendations" in risk_assessment

    def test_performance_benchmarking(self, mock_agent, sample_performance_data):
        """Test that the agent benchmarks performance against standards."""
        mock_agent.performance_history = sample_performance_data

        benchmarks = {
            "sp500_return": 0.12,
            "risk_free_rate": 0.02,
            "market_volatility": 0.15,
            "peer_performance": 0.18,
        }

        with patch.object(mock_agent, "benchmark_performance") as mock_benchmark:
            mock_benchmark.return_value = {
                "vs_sp500": 0.08,  # 8% better than S&P 500
                "vs_risk_free": 0.23,  # 23% better than risk-free rate
                "information_ratio": 1.2,
                "alpha": 0.06,
                "ranking": "top_quartile",
            }

            benchmark_result = mock_agent.benchmark_performance(benchmarks)

            assert benchmark_result["vs_sp500"] > 0
            assert benchmark_result["information_ratio"] > 0
            assert benchmark_result["ranking"] in [
                "top_quartile",
                "second_quartile",
                "third_quartile",
                "bottom_quartile",
            ]

    def test_learning_rate_adaptation(self, mock_agent):
        """Test that the agent adapts its learning rate based on performance."""
        initial_learning_rate = mock_agent.learning_rate

        # Simulate poor performance
        poor_performance = {"accuracy": 0.4, "sharpe_ratio": 0.3, "max_drawdown": -0.3}

        with patch.object(mock_agent, "adapt_learning_rate") as mock_adapt:
            mock_adapt.return_value = {
                "new_learning_rate": initial_learning_rate * 1.5,
                "reason": "Poor performance detected, increasing learning rate",
            }

            adaptation = mock_agent.adapt_learning_rate(poor_performance)

            assert adaptation["new_learning_rate"] > initial_learning_rate

        # Simulate good performance
        good_performance = {"accuracy": 0.9, "sharpe_ratio": 2.0, "max_drawdown": -0.05}

        with patch.object(mock_agent, "adapt_learning_rate") as mock_adapt:
            mock_adapt.return_value = {
                "new_learning_rate": initial_learning_rate * 0.8,
                "reason": "Good performance detected, reducing learning rate",
            }

            adaptation = mock_agent.adapt_learning_rate(good_performance)

            assert adaptation["new_learning_rate"] < initial_learning_rate

    def test_goal_achievement_tracking(self, mock_agent):
        """Test that the agent tracks goal achievement progress."""
        goals = {
            "target_accuracy": 0.8,
            "target_sharpe": 1.5,
            "target_drawdown": -0.1,
            "target_return": 0.25,
        }

        current_performance = {
            "accuracy": 0.75,
            "sharpe_ratio": 1.3,
            "max_drawdown": -0.12,
            "total_return": 0.20,
        }

        with patch.object(mock_agent, "track_goal_progress") as mock_track:
            mock_track.return_value = {
                "accuracy_progress": 0.75,
                "sharpe_progress": 0.87,
                "drawdown_progress": 0.80,
                "return_progress": 0.80,
                "overall_progress": 0.81,
                "estimated_completion": "2024-03-15",
                "next_milestone": "Achieve 80% accuracy",
            }

            progress = mock_agent.track_goal_progress(goals, current_performance)

            assert (
                progress["overall_progress"] >= 0 and progress["overall_progress"] <= 1
            )
            assert "estimated_completion" in progress
            assert "next_milestone" in progress

    def test_learns_from_failed_predictions_and_adjusts_weights(
        self, mock_agent, sample_performance_data
    ):
        """Test that the agent learns from failed predictions and adjusts strategy weights."""
        print("\n🎯 Testing Learning from Failed Predictions and Weight Adjustment")

        # Set up initial strategy weights
        initial_weights = {
            "rsi_strategy": 0.4,
            "macd_strategy": 0.35,
            "bollinger_strategy": 0.25,
        }
        mock_agent.strategy_weights = initial_weights.copy()

        # Create failed prediction scenarios
        failed_predictions = [
            {
                "prediction_id": "pred_001",
                "strategy_used": "rsi_strategy",
                "predicted_action": "buy",
                "actual_action": "sell",
                "confidence": 0.85,
                "market_conditions": {"volatility": "high", "trend": "sideways"},
                "loss_amount": 0.05,
                "timestamp": "2024-01-01T10:00:00",
            },
            {
                "prediction_id": "pred_002",
                "strategy_used": "macd_strategy",
                "predicted_action": "sell",
                "actual_action": "buy",
                "confidence": 0.78,
                "market_conditions": {"volatility": "low", "trend": "upward"},
                "loss_amount": 0.03,
                "timestamp": "2024-01-01T11:00:00",
            },
            {
                "prediction_id": "pred_003",
                "strategy_used": "rsi_strategy",
                "predicted_action": "hold",
                "actual_action": "buy",
                "confidence": 0.92,
                "market_conditions": {"volatility": "medium", "trend": "upward"},
                "loss_amount": 0.08,
                "timestamp": "2024-01-01T12:00:00",
            },
            {
                "prediction_id": "pred_004",
                "strategy_used": "bollinger_strategy",
                "predicted_action": "buy",
                "actual_action": "buy",
                "confidence": 0.88,
                "market_conditions": {"volatility": "high", "trend": "upward"},
                "loss_amount": 0.0,  # Successful prediction
                "timestamp": "2024-01-01T13:00:00",
            },
        ]

        # Track failed predictions
        for prediction in failed_predictions:
            mock_agent.track_failed_prediction(prediction)
            print(f"  📉 Tracked failed prediction: {prediction['prediction_id']}")
            print(f"    Strategy: {prediction['strategy_used']}")
            print(f"    Loss: {prediction['loss_amount']:.3f}")

        # Verify failed predictions are tracked
        failed_predictions_history = mock_agent.get_failed_predictions_history()
        self.assertEqual(
            len(failed_predictions_history),
            len(failed_predictions),
            "All failed predictions should be tracked",
        )

        # Analyze failure patterns
        print(f"\n  🔍 Analyzing failure patterns...")

        failure_analysis = mock_agent.analyze_failure_patterns()

        # Verify failure analysis
        self.assertIsNotNone(failure_analysis, "Failure analysis should not be None")
        self.assertIn(
            "strategy_failure_rates",
            failure_analysis,
            "Should have strategy failure rates",
        )
        self.assertIn(
            "market_condition_failures",
            failure_analysis,
            "Should have market condition failures",
        )
        self.assertIn(
            "confidence_failure_correlation",
            failure_analysis,
            "Should have confidence correlation",
        )
        self.assertIn(
            "recommended_adjustments",
            failure_analysis,
            "Should have recommended adjustments",
        )

        print(f"  Strategy failure rates: {failure_analysis['strategy_failure_rates']}")
        print(
            f"  Market condition failures: {failure_analysis['market_condition_failures']}"
        )
        print(
            f"  Confidence correlation: {failure_analysis['confidence_failure_correlation']:.3f}"
        )

        # Test weight adjustment based on failures
        print(f"\n  ⚖️ Testing weight adjustment...")

        # Get initial weights
        initial_rsi_weight = mock_agent.strategy_weights["rsi_strategy"]
        initial_macd_weight = mock_agent.strategy_weights["macd_strategy"]
        initial_bollinger_weight = mock_agent.strategy_weights["bollinger_strategy"]

        print(
            f"  Initial weights - RSI: {initial_rsi_weight:.3f}, MACD: {initial_macd_weight:.3f}, Bollinger: {initial_bollinger_weight:.3f}"
        )

        # Adjust weights based on failures
        adjustment_result = mock_agent.adjust_strategy_weights_based_on_failures()

        # Verify adjustment result
        self.assertIsNotNone(adjustment_result, "Adjustment result should not be None")
        self.assertIn("old_weights", adjustment_result, "Should have old weights")
        self.assertIn("new_weights", adjustment_result, "Should have new weights")
        self.assertIn(
            "adjustment_reasons", adjustment_result, "Should have adjustment reasons"
        )
        self.assertIn(
            "expected_improvement",
            adjustment_result,
            "Should have expected improvement",
        )

        # Get new weights
        new_rsi_weight = mock_agent.strategy_weights["rsi_strategy"]
        new_macd_weight = mock_agent.strategy_weights["macd_strategy"]
        new_bollinger_weight = mock_agent.strategy_weights["bollinger_strategy"]

        print(
            f"  New weights - RSI: {new_rsi_weight:.3f}, MACD: {new_macd_weight:.3f}, Bollinger: {new_bollinger_weight:.3f}"
        )
        print(f"  Adjustment reasons: {adjustment_result['adjustment_reasons']}")

        # Verify weight adjustments make sense
        # RSI strategy had 2 failures, should be reduced
        self.assertLess(
            new_rsi_weight,
            initial_rsi_weight,
            "RSI weight should be reduced due to failures",
        )

        # MACD strategy had 1 failure, should be slightly reduced
        self.assertLessEqual(
            new_macd_weight,
            initial_macd_weight,
            "MACD weight should be reduced or unchanged",
        )

        # Bollinger strategy had 0 failures, should be increased
        self.assertGreater(
            new_bollinger_weight,
            initial_bollinger_weight,
            "Bollinger weight should be increased due to success",
        )

        # Verify weights still sum to approximately 1.0
        total_weight = new_rsi_weight + new_macd_weight + new_bollinger_weight
        self.assertAlmostEqual(
            total_weight, 1.0, places=2, msg="Total weights should sum to 1.0"
        )

        # Test learning rate adaptation based on failure frequency
        print(f"\n  📚 Testing learning rate adaptation...")

        # Get initial learning rate
        initial_learning_rate = mock_agent.learning_rate

        # Adapt learning rate based on failures
        adaptation_result = mock_agent.adapt_learning_rate_from_failures()

        # Verify adaptation result
        self.assertIsNotNone(adaptation_result, "Adaptation result should not be None")
        self.assertIn(
            "old_learning_rate", adaptation_result, "Should have old learning rate"
        )
        self.assertIn(
            "new_learning_rate", adaptation_result, "Should have new learning rate"
        )
        self.assertIn(
            "adaptation_factor", adaptation_result, "Should have adaptation factor"
        )

        new_learning_rate = mock_agent.learning_rate

        print(f"  Learning rate: {initial_learning_rate:.3f} → {new_learning_rate:.3f}")
        print(f"  Adaptation factor: {adaptation_result['adaptation_factor']:.3f}")

        # Verify learning rate increases with more failures
        self.assertGreater(
            new_learning_rate,
            initial_learning_rate,
            "Learning rate should increase with failures",
        )

        # Test strategy performance tracking after adjustments
        print(f"\n  📊 Testing performance tracking after adjustments...")

        # Simulate new predictions with adjusted weights
        new_predictions = [
            {
                "strategy_used": "bollinger_strategy",  # Higher weight now
                "predicted_action": "buy",
                "actual_action": "buy",
                "confidence": 0.90,
                "profit_amount": 0.04,
                "timestamp": "2024-01-02T10:00:00",
            },
            {
                "strategy_used": "rsi_strategy",  # Lower weight now
                "predicted_action": "sell",
                "actual_action": "sell",
                "confidence": 0.75,
                "profit_amount": 0.02,
                "timestamp": "2024-01-02T11:00:00",
            },
        ]

        for prediction in new_predictions:
            mock_agent.track_prediction_performance(prediction)

        # Get updated performance metrics
        updated_performance = mock_agent.get_strategy_performance_metrics()

        # Verify updated performance
        self.assertIsNotNone(
            updated_performance, "Updated performance should not be None"
        )
        self.assertIn(
            "bollinger_strategy",
            updated_performance,
            "Should have Bollinger performance",
        )
        self.assertIn(
            "rsi_strategy", updated_performance, "Should have RSI performance"
        )
        self.assertIn(
            "macd_strategy", updated_performance, "Should have MACD performance"
        )

        print(f"  Updated performance metrics:")
        for strategy, metrics in updated_performance.items():
            print(f"    {strategy}: {metrics}")

        # Test failure threshold monitoring
        print(f"\n  🚨 Testing failure threshold monitoring...")

        # Set failure thresholds
        failure_thresholds = {
            "rsi_strategy": 0.3,  # 30% failure rate threshold
            "macd_strategy": 0.25,  # 25% failure rate threshold
            "bollinger_strategy": 0.2,  # 20% failure rate threshold
        }

        # Check if any strategies exceed thresholds
        threshold_alerts = mock_agent.check_failure_thresholds(failure_thresholds)

        # Verify threshold alerts
        self.assertIsNotNone(threshold_alerts, "Threshold alerts should not be None")
        self.assertIn("alerts", threshold_alerts, "Should have alerts")
        self.assertIn(
            "recommended_actions", threshold_alerts, "Should have recommended actions"
        )

        print(f"  Threshold alerts: {threshold_alerts['alerts']}")
        print(f"  Recommended actions: {threshold_alerts['recommended_actions']}")

        # Test continuous learning loop
        print(f"\n  🔄 Testing continuous learning loop...")

        # Simulate continuous learning cycle
        learning_cycle_result = mock_agent.run_continuous_learning_cycle()

        # Verify learning cycle
        self.assertIsNotNone(
            learning_cycle_result, "Learning cycle result should not be None"
        )
        self.assertIn("cycle_id", learning_cycle_result, "Should have cycle ID")
        self.assertIn(
            "improvements_made", learning_cycle_result, "Should have improvements"
        )
        self.assertIn(
            "performance_change",
            learning_cycle_result,
            "Should have performance change",
        )
        self.assertIn(
            "next_cycle_recommendations",
            learning_cycle_result,
            "Should have recommendations",
        )

        print(f"  Learning cycle ID: {learning_cycle_result['cycle_id']}")
        print(f"  Improvements made: {learning_cycle_result['improvements_made']}")
        print(
            f"  Performance change: {learning_cycle_result['performance_change']:.3f}"
        )

        # Verify that learning leads to improvement
        self.assertGreater(
            learning_cycle_result["performance_change"],
            0,
            "Learning should lead to performance improvement",
        )

        print("✅ Learning from failed predictions and weight adjustment test completed")
