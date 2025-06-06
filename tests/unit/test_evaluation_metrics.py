import pytest
import numpy as np
import pandas as pd
from trading.evaluation.metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    TimeSeriesMetrics,
    RiskMetrics
)

class TestEvaluationMetrics:
    """Test suite for evaluation metrics."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
        actual = pd.Series(np.random.randn(n_samples) * 10 + 100, index=dates)
        predicted = actual + np.random.randn(n_samples) * 2  # Add some noise
        
        return actual, predicted
    
    @pytest.fixture
    def regression_metrics(self):
        return RegressionMetrics()
    
    @pytest.fixture
    def classification_metrics(self):
        return ClassificationMetrics()
    
    @pytest.fixture
    def time_series_metrics(self):
        return TimeSeriesMetrics()
    
    @pytest.fixture
    def risk_metrics(self):
        return RiskMetrics()
    
    def test_regression_metrics(self, regression_metrics, sample_data):
        """Test regression metrics."""
        actual, predicted = sample_data
        
        # Calculate metrics
        mse = regression_metrics.mean_squared_error(actual, predicted)
        rmse = regression_metrics.root_mean_squared_error(actual, predicted)
        mae = regression_metrics.mean_absolute_error(actual, predicted)
        r2 = regression_metrics.r2_score(actual, predicted)
        
        # Check metric properties
        assert isinstance(mse, float)
        assert isinstance(rmse, float)
        assert isinstance(mae, float)
        assert isinstance(r2, float)
        
        assert mse >= 0
        assert rmse >= 0
        assert mae >= 0
        assert r2 <= 1
        
        # Check relationship between MSE and RMSE
        assert rmse == np.sqrt(mse)
    
    def test_classification_metrics(self, classification_metrics):
        """Test classification metrics."""
        # Generate sample data
        actual = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        predicted = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        
        # Calculate metrics
        accuracy = classification_metrics.accuracy(actual, predicted)
        precision = classification_metrics.precision(actual, predicted)
        recall = classification_metrics.recall(actual, predicted)
        f1 = classification_metrics.f1_score(actual, predicted)
        
        # Check metric properties
        assert isinstance(accuracy, float)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)
        
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
    
    def test_time_series_metrics(self, time_series_metrics, sample_data):
        """Test time series metrics."""
        actual, predicted = sample_data
        
        # Calculate metrics
        mape = time_series_metrics.mean_absolute_percentage_error(actual, predicted)
        smape = time_series_metrics.symmetric_mean_absolute_percentage_error(actual, predicted)
        mase = time_series_metrics.mean_absolute_scaled_error(actual, predicted)
        
        # Check metric properties
        assert isinstance(mape, float)
        assert isinstance(smape, float)
        assert isinstance(mase, float)
        
        assert mape >= 0
        assert smape >= 0
        assert mase >= 0
    
    def test_risk_metrics(self, risk_metrics, sample_data):
        """Test risk metrics."""
        actual, predicted = sample_data
        
        # Calculate metrics
        returns = actual.pct_change().dropna()
        predicted_returns = predicted.pct_change().dropna()
        
        sharpe = risk_metrics.sharpe_ratio(returns)
        sortino = risk_metrics.sortino_ratio(returns)
        max_drawdown = risk_metrics.maximum_drawdown(returns)
        var = risk_metrics.value_at_risk(returns)
        
        # Check metric properties
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(max_drawdown, float)
        assert isinstance(var, float)
        
        assert max_drawdown <= 0
        assert var < 0
    
    def test_metric_consistency(self, regression_metrics, sample_data):
        """Test metric consistency."""
        actual, predicted = sample_data
        
        # Test with perfect predictions
        perfect_predicted = actual.copy()
        mse_perfect = regression_metrics.mean_squared_error(actual, perfect_predicted)
        r2_perfect = regression_metrics.r2_score(actual, perfect_predicted)
        
        assert mse_perfect == 0
        assert r2_perfect == 1
        
        # Test with constant predictions
        constant_predicted = pd.Series(actual.mean(), index=actual.index)
        r2_constant = regression_metrics.r2_score(actual, constant_predicted)
        
        assert r2_constant == 0
    
    def test_metric_robustness(self, regression_metrics, sample_data):
        """Test metric robustness."""
        actual, predicted = sample_data
        
        # Test with outliers
        outlier_actual = actual.copy()
        outlier_actual.iloc[0] = 1e6
        outlier_predicted = predicted.copy()
        outlier_predicted.iloc[0] = 1e6 + 1
        
        mse_outlier = regression_metrics.mean_squared_error(outlier_actual, outlier_predicted)
        mae_outlier = regression_metrics.mean_absolute_error(outlier_actual, outlier_predicted)
        
        assert mse_outlier > 0
        assert mae_outlier > 0
    
    def test_metric_scale_invariance(self, regression_metrics, sample_data):
        """Test metric scale invariance."""
        actual, predicted = sample_data
        
        # Test with scaled data
        scaled_actual = actual * 100
        scaled_predicted = predicted * 100
        
        mse_original = regression_metrics.mean_squared_error(actual, predicted)
        mse_scaled = regression_metrics.mean_squared_error(scaled_actual, scaled_predicted)
        
        # MSE should scale with the square of the scaling factor
        assert abs(mse_scaled - mse_original * 10000) < 1e-10
    
    def test_metric_symmetry(self, regression_metrics, sample_data):
        """Test metric symmetry."""
        actual, predicted = sample_data
        
        # Test metric symmetry
        mse_forward = regression_metrics.mean_squared_error(actual, predicted)
        mse_backward = regression_metrics.mean_squared_error(predicted, actual)
        
        assert mse_forward == mse_backward 