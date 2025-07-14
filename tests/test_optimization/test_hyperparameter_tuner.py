"""
Tests for Hyperparameter Optimizer with Multiple Backends

Tests the enhanced hyperparameter optimizer supporting Optuna, scikit-optimize,
Random Search, and Grid Search backends.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from trading.optimization.optuna_optimizer import HyperparameterOptimizer, get_optimizer


class TestHyperparameterOptimizer:
    """Test the enhanced hyperparameter optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.randn(100))
        return X, y

    @pytest.fixture
    def sample_lstm_data(self):
        """Create sample LSTM data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10, 3)  # (samples, timesteps, features)
        y = np.random.randn(100)
        return X, y

    def test_optimizer_initialization(self):
        """Test optimizer initialization with different backends."""
        # Test Optuna backend
        optimizer = HyperparameterOptimizer(backend="optuna")
        assert optimizer.backend == "optuna"
        assert optimizer.study_name == "evolve_optimization"

        # Test Random backend
        optimizer = HyperparameterOptimizer(backend="random")
        assert optimizer.backend == "random"

        # Test Grid backend
        optimizer = HyperparameterOptimizer(backend="grid")
        assert optimizer.backend == "grid"

    @patch("trading.optimization.optuna_optimizer.OPTUNA_AVAILABLE", True)
    @patch("trading.optimization.optuna_optimizer.optuna")
    def test_xgboost_optuna_optimization(self, mock_optuna, sample_data):
        """Test XGBoost optimization with Optuna backend."""
        X, y = sample_data

        # Mock Optuna study
        mock_study = Mock()
        mock_study.best_params = {"n_estimators": 100, "max_depth": 5}
        mock_study.best_value = 0.5
        mock_study.trials_dataframe.return_value = pd.DataFrame()
        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(backend="optuna")
        optimizer.study = mock_study

        # Mock the evaluation function
        with patch.object(optimizer, "_evaluate_xgboost_params", return_value=0.5):
            result = optimizer.optimize_xgboost(X, y, n_trials=10)

        assert result["backend"] == "optuna"
        assert result["best_score"] == 0.5
        assert result["n_trials"] == 10
        assert "best_params" in result

    @patch("trading.optimization.optuna_optimizer.SKOPT_AVAILABLE", True)
    @patch("trading.optimization.optuna_optimizer.gp_minimize")
    def test_xgboost_skopt_optimization(self, mock_gp_minimize, sample_data):
        """Test XGBoost optimization with scikit-optimize backend."""
        X, y = sample_data

        # Mock scikit-optimize result
        mock_result = Mock()
        mock_result.x = [100, 5, 0.1, 0.8, 0.8, 0.01, 0.01]
        mock_result.fun = 0.6
        mock_gp_minimize.return_value = mock_result

        optimizer = HyperparameterOptimizer(backend="skopt")

        # Mock the evaluation function
        with patch.object(optimizer, "_evaluate_xgboost_params", return_value=0.6):
            result = optimizer.optimize_xgboost(X, y, n_trials=10)

        assert result["backend"] == "skopt"
        assert result["best_score"] == 0.6
        assert result["n_trials"] == 10
        assert "best_params" in result

    @patch("trading.optimization.optuna_optimizer.RandomizedSearchCV")
    def test_xgboost_random_optimization(self, mock_random_search, sample_data):
        """Test XGBoost optimization with Random Search backend."""
        X, y = sample_data

        # Mock RandomizedSearchCV
        mock_search = Mock()
        mock_search.best_params_ = {"n_estimators": 200, "max_depth": 7}
        mock_search.best_score_ = -0.25  # Negative because sklearn uses negative MSE
        mock_search.cv_results_ = {"params": []}
        mock_random_search.return_value = mock_search

        optimizer = HyperparameterOptimizer(backend="random")

        result = optimizer.optimize_xgboost(X, y, n_trials=10)

        assert result["backend"] == "random"
        assert result["best_score"] == 0.5  # sqrt(-(-0.25))
        assert result["n_trials"] == 10
        assert "best_params" in result

    @patch("trading.optimization.optuna_optimizer.GridSearchCV")
    def test_xgboost_grid_optimization(self, mock_grid_search, sample_data):
        """Test XGBoost optimization with Grid Search backend."""
        X, y = sample_data

        # Mock GridSearchCV
        mock_search = Mock()
        mock_search.best_params_ = {"n_estimators": 300, "max_depth": 3}
        mock_search.best_score_ = -0.16  # Negative because sklearn uses negative MSE
        mock_search.cv_results_ = {"params": [{}] * 8}  # 8 combinations
        mock_grid_search.return_value = mock_search

        optimizer = HyperparameterOptimizer(backend="grid")

        result = optimizer.optimize_xgboost(X, y)

        assert result["backend"] == "grid"
        assert result["best_score"] == 0.4  # sqrt(-(-0.16))
        assert result["n_trials"] == 8
        assert "best_params" in result

    @patch("trading.optimization.optuna_optimizer.OPTUNA_AVAILABLE", True)
    @patch("trading.optimization.optuna_optimizer.optuna")
    def test_lstm_optuna_optimization(self, mock_optuna, sample_lstm_data):
        """Test LSTM optimization with Optuna backend."""
        X, y = sample_lstm_data

        # Mock Optuna study
        mock_study = Mock()
        mock_study.best_params = {"lstm_units": 128, "dropout_rate": 0.2}
        mock_study.best_value = 0.8
        mock_study.trials_dataframe.return_value = pd.DataFrame()
        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(backend="optuna")
        optimizer.study = mock_study

        # Mock the evaluation function
        with patch.object(optimizer, "_evaluate_lstm_params", return_value=0.8):
            result = optimizer.optimize_lstm(X, y, n_trials=5)

        assert result["backend"] == "optuna"
        assert result["best_score"] == 0.8
        assert result["n_trials"] == 5
        assert "best_params" in result

    @patch("trading.optimization.optuna_optimizer.SKOPT_AVAILABLE", True)
    @patch("trading.optimization.optuna_optimizer.gp_minimize")
    def test_lstm_skopt_optimization(self, mock_gp_minimize, sample_lstm_data):
        """Test LSTM optimization with scikit-optimize backend."""
        X, y = sample_lstm_data

        # Mock scikit-optimize result
        mock_result = Mock()
        mock_result.x = [128, 0.2, 0.001, 32, 20, 30]
        mock_result.fun = 0.9
        mock_gp_minimize.return_value = mock_result

        optimizer = HyperparameterOptimizer(backend="skopt")

        # Mock the evaluation function
        with patch.object(optimizer, "_evaluate_lstm_params", return_value=0.9):
            result = optimizer.optimize_lstm(X, y, n_trials=5)

        assert result["backend"] == "skopt"
        assert result["best_score"] == 0.9
        assert result["n_trials"] == 5
        assert "best_params" in result

    @patch("numpy.random.choice")
    def test_lstm_random_optimization(self, mock_choice, sample_lstm_data):
        """Test LSTM optimization with Random Search backend."""
        X, y = sample_lstm_data

        # Mock random choice to return predictable values
        mock_choice.side_effect = [128, 0.2, 0.001, 32, 20, 30] * 3  # 3 trials

        optimizer = HyperparameterOptimizer(backend="random")

        # Mock the evaluation function
        with patch.object(optimizer, "_evaluate_lstm_params", return_value=0.7):
            result = optimizer.optimize_lstm(X, y, n_trials=3)

        assert result["backend"] == "random"
        assert result["best_score"] == 0.7
        assert result["n_trials"] == 3
        assert "best_params" in result
        assert len(result["optimization_history"]) == 3

    def test_lstm_grid_optimization(self, sample_lstm_data):
        """Test LSTM optimization with Grid Search backend."""
        X, y = sample_lstm_data

        optimizer = HyperparameterOptimizer(backend="grid")

        # Mock the evaluation function
        with patch.object(optimizer, "_evaluate_lstm_params", return_value=0.6):
            result = optimizer.optimize_lstm(X, y)

        assert result["backend"] == "grid"
        assert result["best_score"] == 0.6
        assert result["n_trials"] == 64  # 2^6 combinations
        assert "best_params" in result
        assert len(result["optimization_history"]) == 64

    def test_evaluate_xgboost_params(self, sample_data):
        """Test XGBoost parameter evaluation."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer()

        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "random_state": 42,
        }

        # Mock XGBRegressor
        with patch(
            "trading.optimization.optuna_optimizer.xgb.XGBRegressor"
        ) as mock_xgb:
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0.1, 0.2, 0.3])
            mock_xgb.return_value = mock_model

            score = optimizer._evaluate_xgboost_params(X, y, params)

        assert isinstance(score, float)
        assert score >= 0

    def test_evaluate_lstm_params(self, sample_lstm_data):
        """Test LSTM parameter evaluation."""
        X, y = sample_lstm_data
        optimizer = HyperparameterOptimizer()

        params = {
            "lstm_units": 64,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "sequence_length": 20,
        }

        # Mock LSTMModel
        with patch("trading.optimization.optuna_optimizer.LSTMModel") as mock_lstm:
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0.1, 0.2, 0.3])
            mock_lstm.return_value = mock_model

            score = optimizer._evaluate_lstm_params(X, y, params)

        assert isinstance(score, float)
        assert score >= 0

    def test_prepare_sequences(self, sample_lstm_data):
        """Test sequence preparation for LSTM."""
        X, y = sample_lstm_data
        optimizer = HyperparameterOptimizer()

        sequence_length = 5
        X_seq, y_seq = optimizer._prepare_sequences(X, y, sequence_length)

        assert X_seq.shape[0] == y_seq.shape[0]
        assert X_seq.shape[1] == sequence_length
        assert X_seq.shape[2] == X.shape[2]
        assert len(X_seq) == len(X) - sequence_length

    def test_save_and_load_best_params(self, tmp_path):
        """Test saving and loading best parameters."""
        optimizer = HyperparameterOptimizer()
        optimizer.best_params_dir = tmp_path

        params = {"n_estimators": 100, "max_depth": 5}
        score = 0.5

        # Test saving
        optimizer._save_best_params("test_model", params, score)

        # Test loading
        loaded_params = optimizer.get_best_params("test_model")
        assert loaded_params == params

    def test_unsupported_backend(self, sample_data):
        """Test error handling for unsupported backends."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(backend="unsupported")

        with pytest.raises(ValueError, match="Backend unsupported not available"):
            optimizer.optimize_xgboost(X, y)

    def test_optimization_failure_handling(self, sample_data):
        """Test handling of optimization failures."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(backend="optuna")

        # Mock study to raise exception
        optimizer.study = Mock()
        optimizer.study.optimize.side_effect = Exception("Optimization failed")

        result = optimizer.optimize_xgboost(X, y, n_trials=10)

        assert "error" in result
        assert "Optimization failed" in result["error"]

    def test_get_optimizer_function(self):
        """Test the get_optimizer function."""
        # Test default backend
        optimizer = get_optimizer()
        assert isinstance(optimizer, HyperparameterOptimizer)
        assert optimizer.backend == "optuna"

        # Test custom backend
        optimizer = get_optimizer(backend="random")
        assert optimizer.backend == "random"

    @patch("trading.optimization.optuna_optimizer.OPTUNA_AVAILABLE", False)
    def test_optuna_unavailable(self, sample_data):
        """Test behavior when Optuna is not available."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(backend="optuna")

        with pytest.raises(ValueError, match="Backend optuna not available"):
            optimizer.optimize_xgboost(X, y)

    @patch("trading.optimization.optuna_optimizer.SKOPT_AVAILABLE", False)
    def test_skopt_unavailable(self, sample_data):
        """Test behavior when scikit-optimize is not available."""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(backend="skopt")

        with pytest.raises(ValueError, match="Backend skopt not available"):
            optimizer.optimize_xgboost(X, y)

    def test_plot_optimization_history(self):
        """Test plotting optimization history."""
        optimizer = HyperparameterOptimizer(backend="optuna")

        # Mock matplotlib and optuna visualization
        with patch("matplotlib.pyplot") as mock_plt, patch(
            "trading.optimization.optuna_optimizer.optuna"
        ) as mock_optuna:
            mock_optuna.visualization.matplotlib.plot_optimization_history = Mock()
            mock_optuna.visualization.matplotlib.plot_param_importances = Mock()

            optimizer.study = Mock()
            optimizer.plot_optimization_history()

            # Verify plots were called
            mock_plt.subplots.assert_called_once()
            mock_plt.show.assert_called_once()
