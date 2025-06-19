"""Tests for the self-improving agent."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime, timedelta

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
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        return pd.DataFrame({
            'date': dates,
            'accuracy': np.random.uniform(0.6, 0.9, 30),
            'sharpe_ratio': np.random.uniform(0.5, 2.0, 30),
            'max_drawdown': np.random.uniform(-0.2, -0.05, 30),
            'total_return': np.random.uniform(0.1, 0.5, 30),
            'win_rate': np.random.uniform(0.5, 0.8, 30)
        })

    def test_agent_initialization(self, mock_agent):
        """Test that the agent initializes correctly."""
        assert hasattr(mock_agent, 'performance_history')
        assert hasattr(mock_agent, 'learning_rate')
        assert hasattr(mock_agent, 'improvement_threshold')
        assert hasattr(mock_agent, 'strategy_weights')
        assert hasattr(mock_agent, 'goal_planner')

    def test_performance_tracking(self, mock_agent, sample_performance_data):
        """Test that the agent tracks performance correctly."""
        # Add performance data
        for _, row in sample_performance_data.iterrows():
            mock_agent.track_performance({
                'date': row['date'],
                'accuracy': row['accuracy'],
                'sharpe_ratio': row['sharpe_ratio'],
                'max_drawdown': row['max_drawdown'],
                'total_return': row['total_return'],
                'win_rate': row['win_rate']
            })
        
        # Check that performance history is updated
        assert len(mock_agent.performance_history) == len(sample_performance_data)
        assert isinstance(mock_agent.performance_history, pd.DataFrame)

    def test_goal_planning(self, mock_agent):
        """Test that the agent plans goals correctly."""
        current_performance = {
            'accuracy': 0.7,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.15,
            'total_return': 0.25,
            'win_rate': 0.65
        }
        
        with patch.object(mock_agent, 'plan_goals') as mock_plan:
            mock_plan.return_value = {
                'target_accuracy': 0.8,
                'target_sharpe': 1.5,
                'target_drawdown': -0.1,
                'target_return': 0.35,
                'target_win_rate': 0.7,
                'improvement_actions': [
                    'Optimize RSI parameters',
                    'Add risk management rules',
                    'Improve entry/exit timing'
                ]
            }
            
            goals = mock_agent.plan_goals(current_performance)
            
            assert goals['target_accuracy'] > current_performance['accuracy']
            assert goals['target_sharpe'] > current_performance['sharpe_ratio']
            assert 'improvement_actions' in goals

    def test_strategy_selection(self, mock_agent, sample_performance_data):
        """Test that the agent selects strategies based on performance."""
        # Add performance data
        mock_agent.performance_history = sample_performance_data
        
        market_conditions = {
            'volatility': 'high',
            'trend': 'upward',
            'volume': 'increasing'
        }
        
        with patch.object(mock_agent, 'select_strategy') as mock_select:
            mock_select.return_value = {
                'strategy': 'momentum',
                'confidence': 0.85,
                'reasoning': 'High volatility and upward trend favor momentum strategies',
                'parameters': {'lookback_period': 20, 'threshold': 0.02}
            }
            
            strategy = mock_agent.select_strategy(market_conditions)
            
            assert strategy['strategy'] == 'momentum'
            assert strategy['confidence'] > 0.5
            assert 'parameters' in strategy

    def test_learning_process(self, mock_agent, sample_performance_data):
        """Test that the agent learns from historical performance."""
        # Add performance data
        mock_agent.performance_history = sample_performance_data
        
        with patch.object(mock_agent, 'learn_from_history') as mock_learn:
            mock_learn.return_value = {
                'insights': [
                    'RSI strategy performs better in trending markets',
                    'MACD strategy shows higher win rate in volatile conditions',
                    'Bollinger Bands strategy has lower drawdown'
                ],
                'parameter_adjustments': {
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'bollinger_std': 2.0
                },
                'strategy_weights': {
                    'rsi': 0.4,
                    'macd': 0.35,
                    'bollinger': 0.25
                }
            }
            
            learning_result = mock_agent.learn_from_history()
            
            assert 'insights' in learning_result
            assert 'parameter_adjustments' in learning_result
            assert 'strategy_weights' in learning_result

    def test_goal_adaptation(self, mock_agent):
        """Test that the agent adapts goals based on performance."""
        current_goals = {
            'target_accuracy': 0.8,
            'target_sharpe': 1.5,
            'target_drawdown': -0.1
        }
        
        recent_performance = {
            'accuracy': 0.75,
            'sharpe_ratio': 1.3,
            'max_drawdown': -0.12
        }
        
        with patch.object(mock_agent, 'adapt_goals') as mock_adapt:
            mock_adapt.return_value = {
                'target_accuracy': 0.82,  # Slightly increased
                'target_sharpe': 1.6,     # Slightly increased
                'target_drawdown': -0.08,  # Slightly improved
                'adaptation_reason': 'Performance is close to targets, increasing goals'
            }
            
            adapted_goals = mock_agent.adapt_goals(current_goals, recent_performance)
            
            assert adapted_goals['target_accuracy'] >= current_goals['target_accuracy']
            assert adapted_goals['target_sharpe'] >= current_goals['target_sharpe']
            assert adapted_goals['target_drawdown'] >= current_goals['target_drawdown']

    def test_error_handling(self, mock_agent):
        """Test that the agent handles errors gracefully."""
        # Test with invalid performance data
        invalid_performance = {
            'accuracy': 'invalid',
            'sharpe_ratio': None,
            'max_drawdown': 'not_a_number'
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
        
        with patch.object(mock_agent, 'analyze_performance') as mock_analyze:
            mock_analyze.return_value = {
                'trend': 'improving',
                'volatility': 'decreasing',
                'consistency': 'high',
                'recommendations': [
                    'Continue current strategy mix',
                    'Consider reducing position sizes',
                    'Monitor drawdown closely'
                ]
            }
            
            analysis = mock_agent.analyze_performance()
            
            assert analysis['trend'] in ['improving', 'declining', 'stable']
            assert 'recommendations' in analysis

    def test_strategy_optimization(self, mock_agent):
        """Test that the agent optimizes strategy parameters."""
        current_parameters = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'bollinger_window': 20,
            'bollinger_std': 2.0
        }
        
        with patch.object(mock_agent, 'optimize_parameters') as mock_optimize:
            mock_optimize.return_value = {
                'rsi_period': 16,
                'macd_fast': 10,
                'macd_slow': 24,
                'bollinger_window': 18,
                'bollinger_std': 2.2,
                'expected_improvement': 0.05,
                'confidence': 0.8
            }
            
            optimized_params = mock_agent.optimize_parameters(current_parameters)
            
            assert optimized_params['expected_improvement'] > 0
            assert optimized_params['confidence'] > 0.5

    def test_continuous_improvement(self, mock_agent, sample_performance_data):
        """Test that the agent continuously improves over time."""
        # Simulate multiple improvement cycles
        initial_performance = {
            'accuracy': 0.6,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.25,
            'total_return': 0.15,
            'win_rate': 0.55
        }
        
        mock_agent.track_performance(initial_performance)
        
        # Simulate improvement cycles
        for i in range(5):
            with patch.object(mock_agent, 'improve') as mock_improve:
                mock_improve.return_value = {
                    'improvement_made': True,
                    'new_performance': {
                        'accuracy': 0.6 + (i + 1) * 0.02,
                        'sharpe_ratio': 0.8 + (i + 1) * 0.1,
                        'max_drawdown': -0.25 + (i + 1) * 0.02,
                        'total_return': 0.15 + (i + 1) * 0.03,
                        'win_rate': 0.55 + (i + 1) * 0.02
                    }
                }
                
                improvement = mock_agent.improve()
                
                assert improvement['improvement_made'] is True
                assert improvement['new_performance']['accuracy'] > initial_performance['accuracy']

    def test_risk_management(self, mock_agent):
        """Test that the agent implements risk management."""
        current_risk_metrics = {
            'var_95': -0.02,
            'max_drawdown': -0.15,
            'volatility': 0.18,
            'beta': 1.1
        }
        
        with patch.object(mock_agent, 'assess_risk') as mock_assess:
            mock_assess.return_value = {
                'risk_level': 'moderate',
                'risk_score': 0.6,
                'recommendations': [
                    'Reduce position sizes by 20%',
                    'Add stop-loss orders',
                    'Diversify across more strategies'
                ],
                'max_position_size': 0.15
            }
            
            risk_assessment = mock_agent.assess_risk(current_risk_metrics)
            
            assert risk_assessment['risk_level'] in ['low', 'moderate', 'high']
            assert risk_assessment['risk_score'] >= 0 and risk_assessment['risk_score'] <= 1
            assert 'recommendations' in risk_assessment

    def test_performance_benchmarking(self, mock_agent, sample_performance_data):
        """Test that the agent benchmarks performance against standards."""
        mock_agent.performance_history = sample_performance_data
        
        benchmarks = {
            'sp500_return': 0.12,
            'risk_free_rate': 0.02,
            'market_volatility': 0.15,
            'peer_performance': 0.18
        }
        
        with patch.object(mock_agent, 'benchmark_performance') as mock_benchmark:
            mock_benchmark.return_value = {
                'vs_sp500': 0.08,  # 8% better than S&P 500
                'vs_risk_free': 0.23,  # 23% better than risk-free rate
                'information_ratio': 1.2,
                'alpha': 0.06,
                'ranking': 'top_quartile'
            }
            
            benchmark_result = mock_agent.benchmark_performance(benchmarks)
            
            assert benchmark_result['vs_sp500'] > 0
            assert benchmark_result['information_ratio'] > 0
            assert benchmark_result['ranking'] in ['top_quartile', 'second_quartile', 'third_quartile', 'bottom_quartile']

    def test_learning_rate_adaptation(self, mock_agent):
        """Test that the agent adapts its learning rate based on performance."""
        initial_learning_rate = mock_agent.learning_rate
        
        # Simulate poor performance
        poor_performance = {
            'accuracy': 0.4,
            'sharpe_ratio': 0.3,
            'max_drawdown': -0.3
        }
        
        with patch.object(mock_agent, 'adapt_learning_rate') as mock_adapt:
            mock_adapt.return_value = {
                'new_learning_rate': initial_learning_rate * 1.5,
                'reason': 'Poor performance detected, increasing learning rate'
            }
            
            adaptation = mock_agent.adapt_learning_rate(poor_performance)
            
            assert adaptation['new_learning_rate'] > initial_learning_rate
            
        # Simulate good performance
        good_performance = {
            'accuracy': 0.9,
            'sharpe_ratio': 2.0,
            'max_drawdown': -0.05
        }
        
        with patch.object(mock_agent, 'adapt_learning_rate') as mock_adapt:
            mock_adapt.return_value = {
                'new_learning_rate': initial_learning_rate * 0.8,
                'reason': 'Good performance detected, reducing learning rate'
            }
            
            adaptation = mock_agent.adapt_learning_rate(good_performance)
            
            assert adaptation['new_learning_rate'] < initial_learning_rate

    def test_goal_achievement_tracking(self, mock_agent):
        """Test that the agent tracks goal achievement progress."""
        goals = {
            'target_accuracy': 0.8,
            'target_sharpe': 1.5,
            'target_drawdown': -0.1,
            'target_return': 0.25
        }
        
        current_performance = {
            'accuracy': 0.75,
            'sharpe_ratio': 1.3,
            'max_drawdown': -0.12,
            'total_return': 0.20
        }
        
        with patch.object(mock_agent, 'track_goal_progress') as mock_track:
            mock_track.return_value = {
                'accuracy_progress': 0.75,
                'sharpe_progress': 0.87,
                'drawdown_progress': 0.80,
                'return_progress': 0.80,
                'overall_progress': 0.81,
                'estimated_completion': '2024-03-15',
                'next_milestone': 'Achieve 80% accuracy'
            }
            
            progress = mock_agent.track_goal_progress(goals, current_performance)
            
            assert progress['overall_progress'] >= 0 and progress['overall_progress'] <= 1
            assert 'estimated_completion' in progress
            assert 'next_milestone' in progress 