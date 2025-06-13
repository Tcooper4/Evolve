import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from trading.data.providers.yfinance_provider import YFinanceProvider
from trading.data.providers.alpha_vantage_provider import AlphaVantageProvider

class TestDataProviders:
    """Test suite for data providers."""
    
    @pytest.fixture
    def yfinance_provider(self):
        return YFinanceProvider(delay=1)  # 1 second delay between requests
    
    @pytest.fixture
    def alpha_vantage_provider(self):
        # Use environment variable with fallback for testing
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'test_key')
        return AlphaVantageProvider(api_key=api_key)
    
    def test_yfinance_provider_initialization(self, yfinance_provider):
        """Test YFinance provider initialization."""
        assert yfinance_provider.delay == 1
        assert yfinance_provider.session is not None
    
    def test_yfinance_provider_get_data(self, yfinance_provider):
        """Test YFinance provider data retrieval."""
        # Test with a known stock
        data = yfinance_provider.get_data(
            symbol='AAPL',
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert isinstance(data.index, pd.DatetimeIndex)
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def test_yfinance_provider_rate_limiting(self, yfinance_provider):
        """Test YFinance provider rate limiting."""
        start_time = datetime.now()
        
        # Make multiple requests
        for _ in range(3):
            yfinance_provider.get_data(
                symbol='AAPL',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Check if rate limiting is working
        assert duration >= 2  # At least 2 seconds for 3 requests with 1s delay
    
    def test_yfinance_provider_error_handling(self, yfinance_provider):
        """Test YFinance provider error handling."""
        # Test with invalid symbol
        with pytest.raises(ValueError):
            yfinance_provider.get_data(
                symbol='INVALID_SYMBOL',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
        
        # Test with invalid date range
        with pytest.raises(ValueError):
            yfinance_provider.get_data(
                symbol='AAPL',
                start_date=datetime.now(),
                end_date=datetime.now() - timedelta(days=1)
            )
    
    def test_alpha_vantage_provider_initialization(self, alpha_vantage_provider):
        """Test Alpha Vantage provider initialization."""
        assert alpha_vantage_provider.api_key == 'test_key'
        assert alpha_vantage_provider.base_url == 'https://www.alphavantage.co/query'
    
    def test_alpha_vantage_provider_get_data(self, alpha_vantage_provider, monkeypatch):
        """Test Alpha Vantage provider data retrieval."""
        # Mock the API response
        def mock_get(*args, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self.json_data = {
                        'Meta Data': {
                            '1. Information': 'Daily Prices',
                            '2. Symbol': 'AAPL',
                            '3. Last Refreshed': '2024-01-01'
                        },
                        'Time Series (Daily)': {
                            '2024-01-01': {
                                '1. open': '100.0',
                                '2. high': '101.0',
                                '3. low': '99.0',
                                '4. close': '100.5',
                                '5. volume': '1000000'
                            }
                        }
                    }
                
                def json(self):
                    return self.json_data
            
            return MockResponse()
        
        monkeypatch.setattr('requests.get', mock_get)
        
        data = alpha_vantage_provider.get_data(
            symbol='AAPL',
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert isinstance(data.index, pd.DatetimeIndex)
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def test_alpha_vantage_provider_error_handling(self, alpha_vantage_provider, monkeypatch):
        """Test Alpha Vantage provider error handling."""
        # Mock API error response
        def mock_get(*args, **kwargs):
            class MockResponse:
                def __init__(self):
                    self.status_code = 400
                    self.json_data = {'Error Message': 'Invalid API call'}
                
                def json(self):
                    return self.json_data
            
            return MockResponse()
        
        monkeypatch.setattr('requests.get', mock_get)
        
        with pytest.raises(ValueError):
            alpha_vantage_provider.get_data(
                symbol='AAPL',
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now()
            )
    
    def test_data_provider_common_features(self, yfinance_provider, alpha_vantage_provider):
        """Test common features across data providers."""
        for provider in [yfinance_provider, alpha_vantage_provider]:
            # Test data validation
            data = pd.DataFrame({
                'Open': [100.0],
                'High': [101.0],
                'Low': [99.0],
                'Close': [100.5],
                'Volume': [1000000]
            }, index=[datetime.now()])
            
            validated_data = provider._validate_data(data)
            assert isinstance(validated_data, pd.DataFrame)
            assert not validated_data.isna().any().any()
            
            # Test data cleaning
            cleaned_data = provider._clean_data(data)
            assert isinstance(cleaned_data, pd.DataFrame)
            assert not cleaned_data.isna().any().any()
            
            # Test data normalization
            normalized_data = provider._normalize_data(data)
            assert isinstance(normalized_data, pd.DataFrame)
            assert not normalized_data.isna().any().any() 