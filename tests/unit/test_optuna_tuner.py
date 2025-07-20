Unit tests for OptunaTuner with Prophet support.

Tests the OptunaTuner class for LSTM, XGBoost, Transformer, and Prophet models.
port pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from trading.optimization.optuna_tuner import SharpeOptunaTuner


class TestOptunaTuner:
   Test cases for OptunaTuner.""   
    @pytest.fixture
    def sample_data(self):
      Create sample data for testing."        dates = pd.date_range(start='2020-1-1, end=202312 freq='D')
        np.random.seed(42
        data = pd.DataFrame({
       dsdates,
        y: np.random.randn(len(dates)).cumsum() + 100      feature1: np.random.randn(len(dates)),
        feature2: np.random.randn(len(dates)),
        feature3: np.random.randn(len(dates))
        })
        return data

    @pytest.fixture
    def tuner(self):
       Create SharpeOptunaTuner instance for testing.        return SharpeOptunaTuner(
            study_name="test_study,
            n_trials=5,
            timeout=60,
            validation_split=0.2,
            random_state=42
        )

    def test_initialization(self, tuner):
   ner initialization.        assert tuner.study_name == "test_study"
        assert tuner.n_trials == 5
        assert tuner.timeout ==60        assert tuner.validation_split == 0.2        assert tuner.random_state ==42        assert tuner.best_params ==[object Object]        assert tuner.best_scores == {}

    def test_data_splitting(self, tuner, sample_data):
  est data splitting functionality."""
        X = sample_data[['feature1', 'feature2', 'feature3]].values
        y = sample_data['y'].values
        
        X_train, X_val, y_train, y_val = tuner._split_data(X, y)
        
        assert len(X_train) + len(X_val) == len(X)
        assert len(y_train) + len(y_val) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)

    @patch('optimization.optuna_tuner.LSTMModel')
    def test_optimize_lstm(self, mock_lstm_model, tuner, sample_data):
        STM hyperparameter optimization.        # Mock the LSTM model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.randn(10)
        mock_lstm_model.return_value = mock_model
        
        result = tuner.optimize_lstm(
            data=sample_data,
            target_column='y',
            feature_columns=['feature1', 'feature2', 'feature3']
        )
        
        assert 'best_params' in result
        assert 'best_score' in result
        assert tuner.best_params.get('lstm') is not None
        assert tuner.best_scores.get('lstm') is not None

    @patch('optimization.optuna_tuner.XGBoostModel')
    def test_optimize_xgboost(self, mock_xgboost_model, tuner, sample_data):
     Test XGBoost hyperparameter optimization.        # Mock the XGBoost model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.randn(100      mock_xgboost_model.return_value = mock_model
        
        result = tuner.optimize_xgboost(
            data=sample_data,
            target_column='y',
            feature_columns=['feature1', 'feature2', 'feature3']
        )
        
        assert 'best_params' in result
        assert 'best_score' in result
        assert tuner.best_params.get('xgboost') is not None
        assert tuner.best_scores.get('xgboost') is not None

    @patch('optimization.optuna_tuner.TransformerForecaster')
    def test_optimize_transformer(self, mock_transformer_model, tuner, sample_data):
        " Transformer hyperparameter optimization.        # Mock the Transformer model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.randn(100)
        mock_transformer_model.return_value = mock_model
        
        result = tuner.optimize_transformer(
            data=sample_data,
            target_column='y',
            feature_columns=['feature1', 'feature2', 'feature3']
        )
        
        assert 'best_params' in result
        assert 'best_score' in result
        assert tuner.best_params.get('transformer') is not None
        assert tuner.best_scores.get('transformer') is not None

    @patch('optimization.optuna_tuner.Prophet')
    def test_optimize_prophet(self, mock_prophet, tuner, sample_data):
     Test Prophet hyperparameter optimization.    # Mock Prophet model
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.make_future_dataframe.return_value = pd.DataFrame({
       ds': pd.date_range(start='20241-1 periods=100, freq='D')
        })
        mock_model.predict.return_value = pd.DataFrame({
           yhat: np.random.randn(100)
        })
        mock_prophet.return_value = mock_model
        
        result = tuner.optimize_prophet(
            data=sample_data,
            target_column='y',
            date_column='ds'
        )
        
        assert 'best_params' in result
        assert 'best_score' in result
        assert tuner.best_params.get('prophet') is not None
        assert tuner.best_scores.get('prophet') is not None

    def test_prophet_data_preparation(self, tuner, sample_data):
     Test Prophet data preparation with different date column names.       # Test with custom date column
        sample_data['date] = sample_data[ds        with patch('optimization.optuna_tuner.Prophet) as mock_prophet:
            mock_model = MagicMock()
            mock_model.fit.return_value = None
            mock_model.make_future_dataframe.return_value = pd.DataFrame({
           ds': pd.date_range(start='20241-1 periods=100, freq='D)    })
            mock_model.predict.return_value = pd.DataFrame({
               yhat: np.random.randn(100    })
            mock_prophet.return_value = mock_model
            
            result = tuner.optimize_prophet(
                data=sample_data,
                target_column='y,              date_column=date'
            )
            
            assert 'best_params' in result

    def test_get_best_params(self, tuner):
     Test getting best parameters for different model types."""
        # Set some mock best parameters
        tuner.best_params = [object Object]
            lstm: {layers':2: 64},
          xgboost: {'max_depth':6, learning_rate':00.1
            prophet': {changepoint_prior_scale': 0.05}
        }
        
        assert tuner.get_best_params('lstm) == {'layers': 2, 'units': 64        assert tuner.get_best_params('xgboost') == {'max_depth':6, learning_rate': 00.1        assert tuner.get_best_params('prophet') == {changepoint_prior_scale': 0.5        assert tuner.get_best_params('nonexistent') is None

    def test_get_best_score(self, tuner):
     Test getting best scores for different model types."""
        # Set some mock best scores
        tuner.best_scores = [object Object]
        lstm002         xgboost00.15
           prophet': 0.025
        }
        
        assert tuner.get_best_score('lstm) == 00.02        assert tuner.get_best_score(xgboost) == 00.15        assert tuner.get_best_score('prophet') == 0.25        assert tuner.get_best_score('nonexistent') is None

    def test_error_handling_invalid_data(self, tuner):
   est error handling with invalid data.       # Test with empty DataFrame
        empty_data = pd.DataFrame()
        with pytest.raises(Exception):
            tuner.optimize_lstm(empty_data,y,['feature1    def test_error_handling_missing_columns(self, tuner, sample_data):
   est error handling with missing columns.       # Test with missing target column
        with pytest.raises(KeyError):
            tuner.optimize_lstm(sample_data, 'nonexistent_column,['feature1
    @patch('optimization.optuna_tuner.LSTMModel')
    def test_optimization_timeout(self, mock_lstm_model, tuner, sample_data):
        ""Test optimization timeout handling.""        # Create a tuner with very short timeout
        short_tuner = OptunaTuner(timeout=1)
        
        # Mock model to simulate slow execution
        mock_model = MagicMock()
        mock_model.fit.side_effect = lambda *args, **kwargs: time.sleep(2)
        mock_lstm_model.return_value = mock_model
        
        # Should handle timeout gracefully
        result = short_tuner.optimize_lstm(
            data=sample_data,
            target_column='y',
            feature_columns=['feature1']
        )
        
        assert isinstance(result, dict)
