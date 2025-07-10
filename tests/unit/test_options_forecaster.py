"""
Tests for Options Forecaster

This module provides comprehensive tests for the options forecaster including:
- Implied volatility calculation
- Greeks calculation
- Sensitivity analysis
- Options pricing
- Volatility surface modeling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'trading'))

from trading.options.options_forecaster import (
    OptionsForecaster, 
    OptionContract, 
    VolatilitySurface
)

class TestOptionsForecaster:
    """Test class for OptionsForecaster."""
    
    @pytest.fixture
    def forecaster(self):
        """Create a test forecaster instance."""
        config = {
            'risk_free_rate': 0.02,
            'dividend_yield': 0.01,
            'api_keys': {
                'polygon': 'test_key'
            }
        }
        return OptionsForecaster(config)
    
    @pytest.fixture
    def sample_options_data(self):
        """Create sample options data for testing."""
        return pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'strike': [100.0, 110.0, 90.0, 105.0],
            'expiration': [
                datetime.now() + timedelta(days=30),
                datetime.now() + timedelta(days=30),
                datetime.now() + timedelta(days=30),
                datetime.now() + timedelta(days=30)
            ],
            'option_type': ['call', 'call', 'put', 'put'],
            'underlying_price': [100.0, 100.0, 100.0, 100.0],
            'bid': [5.0, 2.0, 3.0, 1.5],
            'ask': [5.5, 2.5, 3.5, 2.0],
            'volume': [100, 50, 75, 25],
            'open_interest': [1000, 500, 750, 250],
            'implied_volatility': [0.25, 0.20, 0.30, 0.22]
        })
    
    def test_initialization(self, forecaster):
        """Test forecaster initialization."""
        assert forecaster.risk_free_rate == 0.02
        assert forecaster.dividend_yield == 0.01
        assert forecaster.polygon_key == 'test_key'
        assert forecaster.yahoo_enabled is True
    
    def test_black_scholes_price_call(self, forecaster):
        """Test Black-Scholes price calculation for call options."""
        # Test case: ATM call option
        S, K, T, sigma = 100.0, 100.0, 0.5, 0.3
        price = forecaster._black_scholes_price(S, K, T, sigma, 'call')
        
        assert price > 0
        assert isinstance(price, float)
        
        # Test case: ITM call option
        price_itm = forecaster._black_scholes_price(110.0, 100.0, 0.5, 0.3, 'call')
        assert price_itm > price  # ITM should be more expensive than ATM
        
        # Test case: OTM call option
        price_otm = forecaster._black_scholes_price(90.0, 100.0, 0.5, 0.3, 'call')
        assert price_otm < price  # OTM should be less expensive than ATM
    
    def test_black_scholes_price_put(self, forecaster):
        """Test Black-Scholes price calculation for put options."""
        # Test case: ATM put option
        S, K, T, sigma = 100.0, 100.0, 0.5, 0.3
        price = forecaster._black_scholes_price(S, K, T, sigma, 'put')
        
        assert price > 0
        assert isinstance(price, float)
        
        # Test case: ITM put option
        price_itm = forecaster._black_scholes_price(90.0, 100.0, 0.5, 0.3, 'put')
        assert price_itm > price  # ITM should be more expensive than ATM
        
        # Test case: OTM put option
        price_otm = forecaster._black_scholes_price(110.0, 100.0, 0.5, 0.3, 'put')
        assert price_otm < price  # OTM should be less expensive than ATM
    
    def test_black_scholes_price_edge_cases(self, forecaster):
        """Test Black-Scholes price calculation edge cases."""
        # Test zero time to expiration
        price = forecaster._black_scholes_price(100.0, 100.0, 0.0, 0.3, 'call')
        assert price == 0.0  # Should be 0 for ATM option at expiration
        
        # Test very high volatility
        price = forecaster._black_scholes_price(100.0, 100.0, 0.5, 2.0, 'call')
        assert price > 0
        assert isinstance(price, float)
        
        # Test very low volatility
        price = forecaster._black_scholes_price(100.0, 100.0, 0.5, 0.01, 'call')
        assert price > 0
        assert isinstance(price, float)
    
    def test_calculate_implied_volatility(self, forecaster):
        """Test implied volatility calculation."""
        # Test case: Known option price
        S, K, T = 100.0, 100.0, 0.5
        known_price = 10.0  # Known option price
        
        iv = forecaster.calculate_implied_volatility(known_price, S, K, T, 'call')
        
        assert iv > 0
        assert isinstance(iv, float)
        assert iv <= 5.0  # Should be within reasonable bounds
        
        # Verify the calculated IV reproduces the known price
        calculated_price = forecaster._black_scholes_price(S, K, T, iv, 'call')
        assert abs(calculated_price - known_price) < 0.01
    
    def test_calculate_implied_volatility_edge_cases(self, forecaster):
        """Test implied volatility calculation edge cases."""
        # Test zero option price
        iv = forecaster.calculate_implied_volatility(0.0, 100.0, 100.0, 0.5, 'call')
        assert iv > 0  # Should return a reasonable default
        
        # Test very high option price
        iv = forecaster.calculate_implied_volatility(50.0, 100.0, 100.0, 0.5, 'call')
        assert iv > 0
        assert isinstance(iv, float)
    
    def test_calculate_greeks_call(self, forecaster):
        """Test Greeks calculation for call options."""
        S, K, T, sigma = 100.0, 100.0, 0.5, 0.3
        greeks = forecaster.calculate_greeks(S, K, T, sigma, 'call')
        
        # Check all Greeks are present
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        
        # Check delta bounds for call
        assert 0 <= greeks['delta'] <= 1
        
        # Check gamma is positive
        assert greeks['gamma'] > 0
        
        # Check vega is positive
        assert greeks['vega'] > 0
    
    def test_calculate_greeks_put(self, forecaster):
        """Test Greeks calculation for put options."""
        S, K, T, sigma = 100.0, 100.0, 0.5, 0.3
        greeks = forecaster.calculate_greeks(S, K, T, sigma, 'put')
        
        # Check all Greeks are present
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        
        # Check delta bounds for put
        assert -1 <= greeks['delta'] <= 0
        
        # Check gamma is positive
        assert greeks['gamma'] > 0
        
        # Check vega is positive
        assert greeks['vega'] > 0
    
    def test_calculate_greeks_sensitivity_analysis(self, forecaster):
        """Test Greeks sensitivity to parameter changes."""
        # Base case
        S, K, T, sigma = 100.0, 100.0, 0.5, 0.3
        base_greeks = forecaster.calculate_greeks(S, K, T, sigma, 'call')
        
        # Test delta sensitivity to underlying price
        greeks_up = forecaster.calculate_greeks(S * 1.01, K, T, sigma, 'call')
        greeks_down = forecaster.calculate_greeks(S * 0.99, K, T, sigma, 'call')
        
        # Delta should increase with underlying price for calls
        assert greeks_up['delta'] > base_greeks['delta']
        assert greeks_down['delta'] < base_greeks['delta']
        
        # Test gamma sensitivity to underlying price
        # Gamma should be highest near ATM
        greeks_atm = forecaster.calculate_greeks(100.0, 100.0, T, sigma, 'call')
        greeks_itm = forecaster.calculate_greeks(110.0, 100.0, T, sigma, 'call')
        greeks_otm = forecaster.calculate_greeks(90.0, 100.0, T, sigma, 'call')
        
        # ATM gamma should be higher than ITM/OTM
        assert greeks_atm['gamma'] > greeks_itm['gamma']
        assert greeks_atm['gamma'] > greeks_otm['gamma']
        
        # Test vega sensitivity to volatility
        greeks_high_vol = forecaster.calculate_greeks(S, K, T, sigma * 1.5, 'call')
        greeks_low_vol = forecaster.calculate_greeks(S, K, T, sigma * 0.5, 'call')
        
        # Vega should be higher for higher volatility
        assert greeks_high_vol['vega'] > greeks_low_vol['vega']
    
    def test_calculate_greeks_edge_cases(self, forecaster):
        """Test Greeks calculation edge cases."""
        # Test zero time to expiration
        greeks = forecaster.calculate_greeks(100.0, 100.0, 0.0, 0.3, 'call')
        assert greeks['delta'] in [0.0, 1.0]  # Should be binary at expiration
        assert greeks['gamma'] == 0.0  # No gamma at expiration
        assert greeks['theta'] == 0.0  # No theta at expiration
        assert greeks['vega'] == 0.0  # No vega at expiration
        
        # Test very high volatility
        greeks = forecaster.calculate_greeks(100.0, 100.0, 0.5, 2.0, 'call')
        assert all(isinstance(v, float) for v in greeks.values())
        assert all(not np.isnan(v) for v in greeks.values())
    
    @patch('trading.options.options_forecaster.requests.get')
    def test_get_polygon_options_chain(self, mock_get, forecaster):
        """Test getting options chain from Polygon API."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [
                {
                    'ticker': 'AAPL240119C00100000',
                    'strike_price': '100.0',
                    'expiration_date': '2024-01-19',
                    'contract_type': 'call',
                    'underlying_price': '100.0',
                    'bid': '5.0',
                    'ask': '5.5',
                    'volume': '100',
                    'open_interest': '1000'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test API call
        result = forecaster._get_polygon_options_chain('AAPL')
        
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]['symbol'] == 'AAPL240119C00100000'
        assert result.iloc[0]['strike'] == 100.0
        assert result.iloc[0]['option_type'] == 'call'
    
    @patch('trading.options.options_forecaster.requests.get')
    def test_get_polygon_options_chain_error(self, mock_get, forecaster):
        """Test Polygon API error handling."""
        # Mock failed response
        mock_get.side_effect = Exception("API Error")
        
        result = forecaster._get_polygon_options_chain('AAPL')
        assert result is None
    
    def test_generate_synthetic_options_chain(self, forecaster):
        """Test synthetic options chain generation."""
        symbol = 'AAPL'
        expiration_date = datetime.now() + timedelta(days=30)
        
        result = forecaster._generate_synthetic_options_chain(symbol, expiration_date)
        
        assert not result.empty
        assert len(result) > 0
        assert 'symbol' in result.columns
        assert 'strike' in result.columns
        assert 'expiration' in result.columns
        assert 'option_type' in result.columns
        assert 'underlying_price' in result.columns
        assert 'bid' in result.columns
        assert 'ask' in result.columns
        
        # Check option types
        option_types = result['option_type'].unique()
        assert 'call' in option_types
        assert 'put' in option_types
        
        # Check strikes are reasonable
        assert result['strike'].min() > 0
        assert result['strike'].max() > result['strike'].min()
    
    def test_build_volatility_surface(self, forecaster, sample_options_data):
        """Test volatility surface building."""
        with patch.object(forecaster, 'get_options_chain', return_value=sample_options_data):
            surface = forecaster.build_volatility_surface('AAPL')
            
            assert isinstance(surface, VolatilitySurface)
            assert len(surface.strikes) > 0
            assert len(surface.expirations) > 0
            assert len(surface.implied_volatilities) > 0
            assert surface.surface_type == 'call'
            assert isinstance(surface.calculation_date, datetime)
    
    def test_build_volatility_surface_empty_data(self, forecaster):
        """Test volatility surface building with empty data."""
        with patch.object(forecaster, 'get_options_chain', return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No options data available"):
                forecaster.build_volatility_surface('AAPL')
    
    def test_forecast_implied_volatility(self, forecaster):
        """Test implied volatility forecasting."""
        with patch.object(forecaster, 'build_volatility_surface'):
            forecast = forecaster.forecast_implied_volatility('AAPL', 30)
            
            assert 'symbol' in forecast
            assert 'current_iv' in forecast
            assert 'forecast_iv' in forecast
            assert 'forecast_uncertainty' in forecast
            assert 'confidence_interval' in forecast
            assert 'forecast_date' in forecast
            
            assert forecast['symbol'] == 'AAPL'
            assert forecast['current_iv'] > 0
            assert forecast['forecast_iv'] > 0
            assert forecast['forecast_uncertainty'] > 0
            
            # Check confidence interval
            ci = forecast['confidence_interval']
            assert 'lower' in ci
            assert 'upper' in ci
            assert ci['lower'] < ci['upper']
    
    def test_analyze_options_flow(self, forecaster, sample_options_data):
        """Test options flow analysis."""
        with patch.object(forecaster, 'get_options_chain', return_value=sample_options_data):
            analysis = forecaster.analyze_options_flow('AAPL', 7)
            
            assert 'symbol' in analysis
            assert 'put_call_ratio' in analysis
            assert 'total_call_volume' in analysis
            assert 'total_put_volume' in analysis
            assert 'iv_skew' in analysis
            assert 'active_strikes' in analysis
            assert 'analysis_date' in analysis
            
            assert analysis['symbol'] == 'AAPL'
            assert analysis['put_call_ratio'] >= 0
            assert analysis['total_call_volume'] >= 0
            assert analysis['total_put_volume'] >= 0
            assert isinstance(analysis['iv_skew'], float)
            assert isinstance(analysis['active_strikes'], dict)
    
    def test_analyze_options_flow_empty_data(self, forecaster):
        """Test options flow analysis with empty data."""
        with patch.object(forecaster, 'get_options_chain', return_value=pd.DataFrame()):
            analysis = forecaster.analyze_options_flow('AAPL', 7)
            
            assert 'symbol' in analysis
            assert 'error' in analysis
            assert analysis['error'] == 'No options data available'
    
    def test_greeks_monotonicity(self, forecaster):
        """Test that Greeks follow expected monotonicity properties."""
        # Test delta monotonicity for calls
        S, K, T, sigma = 100.0, 100.0, 0.5, 0.3
        
        # Delta should increase with underlying price for calls
        delta_90 = forecaster.calculate_greeks(90.0, K, T, sigma, 'call')['delta']
        delta_100 = forecaster.calculate_greeks(100.0, K, T, sigma, 'call')['delta']
        delta_110 = forecaster.calculate_greeks(110.0, K, T, sigma, 'call')['delta']
        
        assert delta_90 < delta_100 < delta_110
        
        # Delta should decrease with underlying price for puts
        delta_90_put = forecaster.calculate_greeks(90.0, K, T, sigma, 'put')['delta']
        delta_100_put = forecaster.calculate_greeks(100.0, K, T, sigma, 'put')['delta']
        delta_110_put = forecaster.calculate_greeks(110.0, K, T, sigma, 'put')['delta']
        
        assert delta_90_put > delta_100_put > delta_110_put
    
    def test_put_call_parity(self, forecaster):
        """Test put-call parity relationship."""
        S, K, T, sigma = 100.0, 100.0, 0.5, 0.3
        
        # Calculate call and put prices
        call_price = forecaster._black_scholes_price(S, K, T, sigma, 'call')
        put_price = forecaster._black_scholes_price(S, K, T, sigma, 'put')
        
        # Put-call parity: C - P = S - K*exp(-r*T)
        expected_diff = S - K * np.exp(-forecaster.risk_free_rate * T)
        actual_diff = call_price - put_price
        
        # Should be approximately equal (within numerical precision)
        assert abs(actual_diff - expected_diff) < 0.01
    
    def test_volatility_surface_interpolation(self, forecaster):
        """Test volatility surface interpolation capabilities."""
        # Create a simple volatility surface
        strikes = np.array([90, 100, 110])
        expirations = np.array([0.25, 0.5, 1.0])
        implied_vols = np.array([
            [0.25, 0.30, 0.35],
            [0.20, 0.25, 0.30],
            [0.15, 0.20, 0.25]
        ])
        
        surface = VolatilitySurface(
            strikes=strikes,
            expirations=expirations,
            implied_volatilities=implied_vols.flatten(),
            surface_type='call',
            calculation_date=datetime.now()
        )
        
        # Test surface properties
        assert len(surface.strikes) == 3
        assert len(surface.expirations) == 3
        assert len(surface.implied_volatilities) == 9
        assert surface.surface_type == 'call'
    
    def test_error_handling(self, forecaster):
        """Test error handling in various scenarios."""
        # Test with invalid parameters
        with pytest.raises(Exception):
            forecaster._black_scholes_price(-100, 100, 0.5, 0.3, 'call')
        
        # Test with invalid option type
        with pytest.raises(Exception):
            forecaster._black_scholes_price(100, 100, 0.5, 0.3, 'invalid')
        
        # Test with negative time
        greeks = forecaster.calculate_greeks(100, 100, -0.5, 0.3, 'call')
        assert all(v == 0.0 for v in greeks.values())

if __name__ == "__main__":
    pytest.main([__file__]) 