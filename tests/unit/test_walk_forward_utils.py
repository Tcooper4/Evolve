Unit tests for walk-forward validation utilities.

Tests the walk_forward_validate utility function and related helpers.

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from trading.validation.walk_forward_utils import (
    walk_forward_validate,
    _calculate_performance_summary,
    _calculate_performance_trend,
    get_walk_forward_agent
)
from trading.agents.walk_forward_agent import WalkForwardResult


class TestWalkForwardUtils:
   Test cases for walk-forward validation utilities."   @pytest.fixture
    def sample_data(self):
      Create sample data for testing."        dates = pd.date_range(start=20201 end=202312 freq='D')
        np.random.seed(42
        data = pd.DataFrame({
         datedates,
      target: np.random.randn(len(dates)).cumsum() + 100      feature1: np.random.randn(len(dates)),
        feature2: np.random.randn(len(dates)),
        feature3: np.random.randn(len(dates))
        })
        return data

    @pytest.fixture
    def mock_model_factory(self):
      Create a mock model factory for testing."        def factory():
            model = MagicMock()
            model.fit.return_value = None
            model.predict.return_value = np.random.randn(50)
            return model
        return factory

    @patch('trading.validation.walk_forward_utils.WalkForwardAgent')
    def test_walk_forward_validate_success(self, mock_agent_class, sample_data, mock_model_factory):
      Test successful walk-forward validation.        # Mock the agent
        mock_agent = MagicMock()
        mock_results =        WalkForwardResult(
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 30),
                test_start=datetime(2020, 7, 1),
                test_end=datetime(2020, 9, 30),
                model_performance=[object Object]sharpe_ratio': 08 max_drawdown:00.1},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test'}
            )
        ]
        mock_agent.run_walk_forward_validation.return_value = mock_results
        mock_agent_class.return_value = mock_agent
        
        result = walk_forward_validate(
            data=sample_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3'],
            model_factory=mock_model_factory,
            train_window_days=180,
            test_window_days=90,
            step_size_days=30
        )
        
        assert result[success] is True
        assert 'results' in result
        assert 'performance_summary' in result
        assert result['total_windows'] == 1
        assert result[train_window_days'] == 180     assert result['test_window_days'] ==90     assert result['step_size_days'] ==30   @patch('trading.validation.walk_forward_utils.WalkForwardAgent')
    def test_walk_forward_validate_failure(self, mock_agent_class, sample_data, mock_model_factory):
      Test walk-forward validation with failure.        # Mock the agent to raise an exception
        mock_agent = MagicMock()
        mock_agent.run_walk_forward_validation.side_effect = Exception("Test error")
        mock_agent_class.return_value = mock_agent
        
        result = walk_forward_validate(
            data=sample_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3'],
            model_factory=mock_model_factory
        )
        
        assert result['success'] is False
        assert error in result
        assert 'Test error in result['error]def test_calculate_performance_summary_empty_results(self):
      Test performance summary calculation with empty results."
        summary = _calculate_performance_summary(    assert summary == [object Object]def test_calculate_performance_summary_with_results(self):
      Test performance summary calculation with results."""
        mock_results =        WalkForwardResult(
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 30),
                test_start=datetime(2020, 7, 1),
                test_end=datetime(2020, 9, 30),
                model_performance=[object Object]sharpe_ratio': 08 max_drawdown: 00.1mse: 0.02win_rate},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test}    ),
            WalkForwardResult(
                train_start=datetime(2020, 4, 1),
                train_end=datetime(2020, 9, 30),
                test_start=datetime(2020, 10, 1),
                test_end=datetime(2020, 12, 31),
                model_performance=[object Object]sharpe_ratio': 12 max_drawdown: -015mse: 0.03win_rate},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test'}
            )
        ]
        
        summary = _calculate_performance_summary(mock_results)
        
        assert 'mean_sharpe' in summary
        assert 'std_sharpe' in summary
        assert mean_drawdown' in summary
        assertmax_drawdown' in summary
        assert 'mean_mse' in summary
        assert mean_win_rate' in summary
        assert total_windows' in summary
        assert performance_trend' in summary
        
        assert summary['total_windows'] == 2
        assert summarymean_sharpe'] == 10  # (00.81.2 /2def test_calculate_performance_trend_insufficient_data(self):
      Test performance trend calculation with insufficient data."""
        mock_results =        WalkForwardResult(
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 30),
                test_start=datetime(2020, 7, 1),
                test_end=datetime(2020, 9, 30),
                model_performance=[object Object]sharpe_ratio': 0.8},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test'}
            )
        ]
        
        trend = _calculate_performance_trend(mock_results)
        assert trend['trend'] == insufficient_data'

    def test_calculate_performance_trend_improving(self):
      Test performance trend calculation with improving performance."""
        mock_results =        WalkForwardResult(
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 30),
                test_start=datetime(2020, 7, 1),
                test_end=datetime(2020, 9, 30),
                model_performance=[object Object]sharpe_ratio': 0.5},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test}    ),
            WalkForwardResult(
                train_start=datetime(2020, 4, 1),
                train_end=datetime(2020, 9, 30),
                test_start=datetime(2020, 10, 1),
                test_end=datetime(2020, 12, 31),
                model_performance=[object Object]sharpe_ratio': 1.5},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test}    ),
            WalkForwardResult(
                train_start=datetime(2020, 7, 1),
                train_end=datetime(2020, 12, 31),
                test_start=datetime(2021, 1, 1),
                test_end=datetime(2021, 3, 31),
                model_performance=[object Object]sharpe_ratio': 2.0},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test'}
            )
        ]
        
        trend = _calculate_performance_trend(mock_results)
        
        assert trend[early_sharpe] == 0.5rst result
        assert trendlate_sharpe] == 2.0ast result
        assert trend['trend'] ==1.5         #20.5      assert trend[trend_direction'] == 'improving'

    def test_calculate_performance_trend_declining(self):
      Test performance trend calculation with declining performance."""
        mock_results =        WalkForwardResult(
                train_start=datetime(2020, 1, 1),
                train_end=datetime(2020, 6, 30),
                test_start=datetime(2020, 7, 1),
                test_end=datetime(2020, 9, 30),
                model_performance=[object Object]sharpe_ratio': 2.0},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test}    ),
            WalkForwardResult(
                train_start=datetime(2020, 4, 1),
                train_end=datetime(2020, 9, 30),
                test_start=datetime(2020, 10, 1),
                test_end=datetime(2020, 12, 31),
                model_performance=[object Object]sharpe_ratio': 1.5},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test}    ),
            WalkForwardResult(
                train_start=datetime(2020, 7, 1),
                train_end=datetime(2020, 12, 31),
                test_start=datetime(2021, 1, 1),
                test_end=datetime(2021, 3, 31),
                model_performance=[object Object]sharpe_ratio': 0.5},
                predictions=pd.Series(np.random.randn(90)),
                actual_values=pd.Series(np.random.randn(90)),
                model_metadata={'model_type': 'test'}
            )
        ]
        
        trend = _calculate_performance_trend(mock_results)
        
        assert trend[early_sharpe] == 2.0rst result
        assert trendlate_sharpe] == 0.5ast result
        assert trend['trend'] == -1.5        #0.52      assert trend[trend_direction'] == 'declining'

    def test_get_walk_forward_agent(self):
      Test getting a configured walk-forward agent instance."       agent = get_walk_forward_agent()
        
        assert agent is not None
        assert hasattr(agent,config')
        assert agent.config.name == 'WalkForwardValidator'
        assert agent.config.enabled is True
        assert agent.config.priority ==1
    def test_walk_forward_validate_default_parameters(self, sample_data, mock_model_factory):
      Test walk-forward validation with default parameters."""
        with patch('trading.validation.walk_forward_utils.WalkForwardAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run_walk_forward_validation.return_value = []
            mock_agent_class.return_value = mock_agent
            
            result = walk_forward_validate(
                data=sample_data,
                target_column='target,           feature_columns=['feature1'],
                model_factory=mock_model_factory
            )
            
            # Check that default parameters are used
            assert result[train_window_days'] == 252
            assert result['test_window_days'] == 63
            assert result['step_size_days'] == 21
    def test_walk_forward_validate_custom_parameters(self, sample_data, mock_model_factory):
      Test walk-forward validation with custom parameters."""
        with patch('trading.validation.walk_forward_utils.WalkForwardAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run_walk_forward_validation.return_value = []
            mock_agent_class.return_value = mock_agent
            
            result = walk_forward_validate(
                data=sample_data,
                target_column='target,           feature_columns=['feature1'],
                model_factory=mock_model_factory,
                train_window_days=100              test_window_days=20              step_size_days=10            custom_param='test_value'
            )
            
            # Check that custom parameters are used
            assert result[train_window_days'] == 100
            assert result['test_window_days'] == 20
            assert result['step_size_days'] == 10 