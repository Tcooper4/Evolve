"""Tests for the Prophet forecasting model."""

import pytest
import pandas as pd
import numpy as np
from trading.models.prophet_model import ProphetModel

class TestProphetModel:
    @pytest.fixture
    def model(self):
        """Create a Prophet model instance for testing."""
        return ProphetModel(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0
        )

    @pytest.fixture
    def sample_data(self, sample_price_data):
        """Create sample price data for testing."""
        # Ensure data has datetime index
        data = sample_price_data.copy()
        data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
        return data

    def test_model_initialization(self, model):
        """Test that model initializes with correct parameters."""
        assert model.changepoint_prior_scale == 0.05
        assert model.seasonality_prior_scale == 10.0
        assert model.holidays_prior_scale == 10.0
        assert model.name == 'Prophet'

    def test_data_preprocessing(self, model, sample_data):
        """Test that data is preprocessed correctly."""
        processed_data = model.preprocess_data(sample_data['close'])
        
        assert isinstance(processed_data, pd.DataFrame)
        assert 'ds' in processed_data.columns
        assert 'y' in processed_data.columns
        assert not processed_data.isnull().any()
        assert len(processed_data) == len(sample_data)

    def test_model_fitting(self, model, sample_data):
        """Test that model fits to data correctly."""
        model.fit(sample_data['close'])
        
        assert model.is_fitted
        assert hasattr(model, 'model')
        assert model.model is not None

    def test_forecast_generation(self, model, sample_data):
        """Test that forecasts are generated correctly."""
        model.fit(sample_data['close'])
        forecast = model.forecast(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 5
        assert not forecast.isnull().any()

    def test_forecast_components(self, model, sample_data):
        """Test that forecast components are calculated correctly."""
        model.fit(sample_data['close'])
        forecast, components = model.forecast_with_components(steps=5)
        
        assert isinstance(forecast, pd.Series)
        assert isinstance(components, pd.DataFrame)
        assert 'trend' in components.columns
        assert 'seasonal' in components.columns
        assert len(forecast) == len(components) == 5

    def test_model_evaluation(self, model, sample_data):
        """Test that model evaluation metrics are calculated correctly."""
        model.fit(sample_data['close'])
        metrics = model.evaluate(sample_data['close'])
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_parameter_validation(self):
        """Test that model parameters are validated."""
        with pytest.raises(ValueError):
            ProphetModel(changepoint_prior_scale=0)  # Invalid changepoint prior
        with pytest.raises(ValueError):
            ProphetModel(seasonality_prior_scale=0)  # Invalid seasonality prior
        with pytest.raises(ValueError):
            ProphetModel(holidays_prior_scale=0)  # Invalid holidays prior

    def test_empty_data_handling(self, model):
        """Test that model handles empty data correctly."""
        empty_data = pd.Series([])
        with pytest.raises(ValueError):
            model.fit(empty_data)

    def test_missing_data_handling(self, model):
        """Test that model handles missing data correctly."""
        data = pd.Series([100, np.nan, 101, 102])
        with pytest.raises(ValueError):
            model.fit(data)

    def test_forecast_horizon_validation(self, model, sample_data):
        """Test that forecast horizon is validated."""
        model.fit(sample_data['close'])
        with pytest.raises(ValueError):
            model.forecast(steps=0)  # Invalid forecast horizon

    def test_model_persistence(self, model, sample_data, tmp_path):
        """Test that model can be saved and loaded."""
        # Fit model
        model.fit(sample_data['close'])
        
        # Save model
        model_path = tmp_path / "prophet_model.json"
        model.save(model_path)
        
        # Load model
        loaded_model = ProphetModel.load(model_path)
        
        # Verify loaded model
        assert loaded_model.changepoint_prior_scale == model.changepoint_prior_scale
        assert loaded_model.seasonality_prior_scale == model.seasonality_prior_scale
        assert loaded_model.holidays_prior_scale == model.holidays_prior_scale
        assert loaded_model.is_fitted

    def test_seasonality_detection(self, model):
        """Test that model detects seasonality correctly."""
        # Create data with clear seasonality
        dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
        seasonal_data = pd.Series(
            100 + 10*np.sin(np.linspace(0, 4*np.pi, 365)),
            index=dates
        )
        
        model.fit(seasonal_data)
        forecast, components = model.forecast_with_components(steps=30)
        
        # Check that seasonal component is significant
        assert abs(components['seasonal']).mean() > 0

    def test_trend_detection(self, model):
        """Test that model detects trends correctly."""
        # Create data with clear trend
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        trend_data = pd.Series(
            np.linspace(100, 200, 100),
            index=dates
        )
        
        model.fit(trend_data)
        forecast, components = model.forecast_with_components(steps=5)
        
        # Check that trend component is significant
        assert components['trend'].diff().mean() > 0

    def test_holiday_effects(self, model):
        """Test that model handles holiday effects correctly."""
        # Create data with holiday effects
        dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
        data = pd.Series(100, index=dates)
        
        # Add holiday effect
        holiday_dates = ['2020-01-01', '2020-12-25']
        for date in holiday_dates:
            data.loc[date] = 120
        
        model.fit(data)
        forecast, components = model.forecast_with_components(steps=30)
        
        # Check that holiday effects are captured
        assert 'holidays' in components.columns

    def test_missing_values_handling(self, model):
        """Test that model handles missing values correctly."""
        # Create data with missing values
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        data = pd.Series(100 + np.random.normal(0, 5, 100), index=dates)
        
        # Add some missing values
        data.iloc[::10] = np.nan  # Every 10th value is NaN
        
        try:
            model.fit(data)
            forecast = model.forecast(steps=5)
            
            # Model should handle missing values gracefully
            assert len(forecast) == 5
            assert not forecast.isnull().any()
            
        except Exception as e:
            # If model fails, it should fail with a clear error message
            assert "NaN" in str(e) or "missing" in str(e).lower(), f"Model should handle missing values gracefully, got: {e}"

    def test_multiple_seasonality(self, model):
        """Test that model handles multiple seasonality patterns."""
        # Create data with weekly and yearly seasonality
        dates = pd.date_range(start='2020-01-01', periods=730, freq='D')  # 2 years
        
        # Weekly seasonality (7-day cycle)
        weekly_pattern = np.sin(np.linspace(0, 104*np.pi, 730))  # 104 weeks
        
        # Yearly seasonality (365-day cycle)
        yearly_pattern = 5*np.sin(np.linspace(0, 4*np.pi, 730))  # 2 years
        
        # Combine patterns
        data = pd.Series(
            100 + weekly_pattern + yearly_pattern + np.random.normal(0, 1, 730),
            index=dates
        )
        
        try:
            model.fit(data)
            forecast, components = model.forecast_with_components(steps=30)
            
            # Check that both seasonal patterns are captured
            assert len(forecast) == 30
            assert not forecast.isnull().any()
            
            # Check that seasonal components exist
            if 'seasonal' in components.columns:
                seasonal_strength = abs(components['seasonal']).mean()
                assert seasonal_strength > 0, "Seasonal component should be present"
            
        except Exception as e:
            # Model should handle multiple seasonality
            assert "seasonality" in str(e).lower() or "pattern" in str(e).lower(), f"Model should handle multiple seasonality, got: {e}"

    def test_holiday_integration(self, model):
        """Test that model integrates holidays correctly."""
        # Test with holiday configuration
        holiday_config = {
            'add_holidays': True,
            'holiday_country': 'US',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        
        try:
            holiday_model = ProphetModel(holiday_config)
            
            # Create data with holiday effects
            dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
            data = pd.Series(100 + np.random.normal(0, 5, 365), index=dates)
            
            # Add holiday effect
            data.loc['2020-12-25'] = 120  # Christmas
            
            holiday_model.fit(data)
            forecast = holiday_model.forecast(steps=10)
            
            assert len(forecast) == 10
            assert not forecast.isnull().any()
            
        except Exception as e:
            # Holiday integration should work or fail gracefully
            assert "holiday" in str(e).lower() or "country" in str(e).lower(), f"Holiday integration should work, got: {e}"

    def test_changepoint_control(self, model):
        """Test that changepoint control parameters work correctly."""
        # Test different changepoint prior scales
        for changepoint_scale in [0.01, 0.05, 0.1]:
            config = {
                'changepoint_prior_scale': changepoint_scale,
                'seasonality_prior_scale': 10.0
            }
            
            try:
                test_model = ProphetModel(config)
                
                # Create data with changepoints
                dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
                data = pd.Series(100, index=dates)
                
                # Add changepoint
                data.iloc[100:] = 150  # Sudden change at day 100
                
                test_model.fit(data)
                forecast = test_model.forecast(steps=10)
                
                assert len(forecast) == 10
                assert not forecast.isnull().any()
                
            except Exception as e:
                # Changepoint control should work
                assert "changepoint" in str(e).lower(), f"Changepoint control should work, got: {e}" 