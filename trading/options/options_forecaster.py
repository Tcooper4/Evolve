"""
Options Forecaster

This module provides comprehensive options forecasting capabilities including:
- Implied volatility modeling
- Options pricing using Black-Scholes and other models
- Greeks calculation
- Options flow analysis
- Volatility surface modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Try to import external libraries
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Option contract data structure."""
    symbol: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    underlying_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

@dataclass
class VolatilitySurface:
    """Volatility surface data structure."""
    strikes: np.ndarray
    expirations: np.ndarray
    implied_volatilities: np.ndarray
    surface_type: str  # 'call', 'put', or 'mid'
    calculation_date: datetime

class OptionsForecaster:
    """Options forecasting and analysis engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize options forecaster.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.dividend_yield = self.config.get('dividend_yield', 0.0)
        self.api_keys = self.config.get('api_keys', {})
        
        # Initialize data providers
        self.polygon_key = self.api_keys.get('polygon')
        self.yahoo_enabled = self.config.get('yahoo_enabled', True)
        
        # Volatility surface cache
        self.volatility_surfaces = {}
        self.options_data_cache = {}
        
        logger.info("Options forecaster initialized")
    
    def get_options_chain(self, symbol: str, expiration_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get options chain for a symbol.
        
        Args:
            symbol: Stock symbol
            expiration_date: Specific expiration date (optional)
            
        Returns:
            DataFrame with options chain data
        """
        try:
            # Try Polygon API first
            if self.polygon_key and REQUESTS_AVAILABLE:
                options_data = self._get_polygon_options_chain(symbol, expiration_date)
                if options_data is not None:
                    return options_data
            
            # Fallback to Yahoo Finance
            if self.yahoo_enabled and YFINANCE_AVAILABLE:
                options_data = self._get_yahoo_options_chain(symbol, expiration_date)
                if options_data is not None:
                    return options_data
            
            # Generate synthetic data for testing
            logger.warning(f"Using synthetic options data for {symbol}")
            return self._generate_synthetic_options_chain(symbol, expiration_date)
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_polygon_options_chain(self, symbol: str, expiration_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get options chain from Polygon API."""
        try:
            if not self.polygon_key:
                return None
            
            # Build API URL
            base_url = "https://api.polygon.io/v3/reference/options/contracts"
            params = {
                'underlying_ticker': symbol,
                'apiKey': self.polygon_key
            }
            
            if expiration_date:
                params['expiration_date'] = expiration_date.strftime('%Y-%m-%d')
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data:
                logger.warning(f"No options data found for {symbol}")
                return None
            
            # Process results
            options_list = []
            for contract in data['results']:
                option_data = {
                    'symbol': contract.get('ticker'),
                    'strike': float(contract.get('strike_price', 0)),
                    'expiration': datetime.strptime(contract.get('expiration_date'), '%Y-%m-%d'),
                    'option_type': contract.get('contract_type', '').lower(),
                    'underlying_price': float(contract.get('underlying_price', 0)),
                    'bid': float(contract.get('bid', 0)),
                    'ask': float(contract.get('ask', 0)),
                    'volume': int(contract.get('volume', 0)),
                    'open_interest': int(contract.get('open_interest', 0))
                }
                options_list.append(option_data)
            
            return pd.DataFrame(options_list)
            
        except Exception as e:
            logger.error(f"Error getting Polygon options data: {e}")
            return None
    
    def _get_yahoo_options_chain(self, symbol: str, expiration_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Get options chain from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            
            if expiration_date:
                # Get specific expiration
                options = ticker.option_chain(expiration_date.strftime('%Y-%m-%d'))
            else:
                # Get all available expirations
                expirations = ticker.options
                if not expirations:
                    return None
                
                # Use nearest expiration
                nearest_exp = expirations[0]
                options = ticker.option_chain(nearest_exp)
            
            # Combine calls and puts
            calls_df = options.calls.copy()
            calls_df['option_type'] = 'call'
            
            puts_df = options.puts.copy()
            puts_df['option_type'] = 'put'
            
            combined_df = pd.concat([calls_df, puts_df], ignore_index=True)
            
            # Standardize column names
            column_mapping = {
                'strike': 'strike',
                'lastPrice': 'last_price',
                'bid': 'bid',
                'ask': 'ask',
                'volume': 'volume',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_volatility'
            }
            
            combined_df = combined_df.rename(columns=column_mapping)
            
            # Add missing columns
            if 'expiration' not in combined_df.columns:
                combined_df['expiration'] = expiration_date or datetime.strptime(nearest_exp, '%Y-%m-%d')
            
            if 'symbol' not in combined_df.columns:
                combined_df['symbol'] = symbol
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error getting Yahoo options data: {e}")
            return None
    
    def _generate_synthetic_options_chain(self, symbol: str, expiration_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate synthetic options data for testing."""
        if expiration_date is None:
            expiration_date = datetime.now() + timedelta(days=30)
        
        # Get underlying price (synthetic)
        underlying_price = 100.0  # Synthetic price
        
        # Generate strikes around current price
        strikes = np.arange(underlying_price * 0.7, underlying_price * 1.3, 5)
        
        options_list = []
        
        for strike in strikes:
            # Calculate synthetic implied volatility
            moneyness = np.log(strike / underlying_price)
            time_to_expiry = (expiration_date - datetime.now()).days / 365
            
            # Synthetic IV surface
            atm_iv = 0.25  # 25% ATM volatility
            iv = atm_iv * (1 + 0.1 * moneyness + 0.05 * moneyness**2)
            iv = max(0.05, min(0.8, iv))  # Clamp between 5% and 80%
            
            # Calculate option prices using Black-Scholes
            call_price = self._black_scholes_price(underlying_price, strike, time_to_expiry, iv, 'call')
            put_price = self._black_scholes_price(underlying_price, strike, time_to_expiry, iv, 'put')
            
            # Call option
            options_list.append({
                'symbol': f"{symbol}",
                'strike': strike,
                'expiration': expiration_date,
                'option_type': 'call',
                'underlying_price': underlying_price,
                'bid': call_price * 0.95,
                'ask': call_price * 1.05,
                'volume': np.random.randint(10, 1000),
                'open_interest': np.random.randint(100, 10000),
                'implied_volatility': iv
            })
            
            # Put option
            options_list.append({
                'symbol': f"{symbol}",
                'strike': strike,
                'expiration': expiration_date,
                'option_type': 'put',
                'underlying_price': underlying_price,
                'bid': put_price * 0.95,
                'ask': put_price * 1.05,
                'volume': np.random.randint(10, 1000),
                'open_interest': np.random.randint(100, 10000),
                'implied_volatility': iv
            })
        
        return pd.DataFrame(options_list)
    
    def calculate_implied_volatility(self, option_price: float, underlying_price: float, 
                                   strike: float, time_to_expiry: float, 
                                   option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Market price of the option
            underlying_price: Current price of underlying asset
            strike: Strike price
            time_to_expiry: Time to expiration in years
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility
        """
        try:
            def objective(sigma):
                """Objective function for volatility calculation."""
                bs_price = self._black_scholes_price(underlying_price, strike, time_to_expiry, sigma, option_type)
                return bs_price - option_price
            
            # Initial guess for volatility
            initial_guess = 0.3
            
            # Use scipy's minimize_scalar for robust optimization
            result = minimize_scalar(
                lambda x: abs(objective(x)),
                bounds=(0.001, 5.0),  # Reasonable volatility bounds
                method='bounded'
            )
            
            if result.success:
                return result.x
            else:
                logger.warning("Failed to converge on implied volatility")
                return 0.3  # Default fallback
                
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return 0.3
    
    def _black_scholes_price(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        try:
            if T <= 0:
                return max(0, S - K) if option_type == 'call' else max(0, K - S)
            
            d1 = (np.log(S / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                price = S * np.exp(-self.dividend_yield * T) * norm.cdf(d1) - K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - S * np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
            
            return price
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return 0.0
    
    def calculate_greeks(self, underlying_price: float, strike: float, time_to_expiry: float, 
                        implied_volatility: float, option_type: str = 'call') -> Dict[str, float]:
        """Calculate option Greeks.
        
        Args:
            underlying_price: Current price of underlying asset
            strike: Strike price
            time_to_expiry: Time to expiration in years
            implied_volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with Greeks values
        """
        try:
            if time_to_expiry <= 0:
                return {
                    'delta': 1.0 if option_type == 'call' and underlying_price > strike else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            
            d1 = (np.log(underlying_price / strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * implied_volatility**2) * time_to_expiry) / (implied_volatility * np.sqrt(time_to_expiry))
            d2 = d1 - implied_volatility * np.sqrt(time_to_expiry)
            
            # Delta
            if option_type == 'call':
                delta = np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1)
            else:
                delta = np.exp(-self.dividend_yield * time_to_expiry) * (norm.cdf(d1) - 1)
            
            # Gamma
            gamma = np.exp(-self.dividend_yield * time_to_expiry) * norm.pdf(d1) / (underlying_price * implied_volatility * np.sqrt(time_to_expiry))
            
            # Theta
            theta_term1 = -(underlying_price * implied_volatility * np.exp(-self.dividend_yield * time_to_expiry) * norm.pdf(d1)) / (2 * np.sqrt(time_to_expiry))
            theta_term2 = self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            theta_term3 = self.dividend_yield * underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1)
            
            if option_type == 'call':
                theta = theta_term1 - theta_term2 + theta_term3
            else:
                theta = theta_term1 - theta_term2 + theta_term3
            
            # Vega
            vega = underlying_price * np.exp(-self.dividend_yield * time_to_expiry) * np.sqrt(time_to_expiry) * norm.pdf(d1)
            
            # Rho
            if option_type == 'call':
                rho = strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                rho = -strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    def build_volatility_surface(self, symbol: str, expiration_dates: Optional[List[datetime]] = None) -> VolatilitySurface:
        """Build volatility surface for a symbol.
        
        Args:
            symbol: Stock symbol
            expiration_dates: List of expiration dates (optional)
            
        Returns:
            VolatilitySurface object
        """
        try:
            # Get options data
            options_df = self.get_options_chain(symbol)
            
            if options_df.empty:
                raise ValueError(f"No options data available for {symbol}")
            
            # Filter by expiration dates if provided
            if expiration_dates:
                options_df = options_df[options_df['expiration'].isin(expiration_dates)]
            
            # Group by strike and expiration
            grouped = options_df.groupby(['strike', 'expiration'])
            
            strikes = []
            expirations = []
            implied_vols = []
            
            for (strike, expiration), group in grouped:
                # Use mid-price for IV calculation
                mid_price = (group['bid'].mean() + group['ask'].mean()) / 2
                
                if mid_price > 0:
                    # Calculate time to expiration
                    time_to_expiry = (expiration - datetime.now()).days / 365
                    
                    if time_to_expiry > 0:
                        # Calculate implied volatility
                        iv = self.calculate_implied_volatility(
                            mid_price, 
                            group['underlying_price'].iloc[0], 
                            strike, 
                            time_to_expiry, 
                            'call'  # Use calls for surface
                        )
                        
                        strikes.append(strike)
                        expirations.append(time_to_expiry)
                        implied_vols.append(iv)
            
            if not strikes:
                raise ValueError("No valid options data for volatility surface")
            
            # Create volatility surface
            surface = VolatilitySurface(
                strikes=np.array(strikes),
                expirations=np.array(expirations),
                implied_volatilities=np.array(implied_vols),
                surface_type='call',
                calculation_date=datetime.now()
            )
            
            # Cache the surface
            self.volatility_surfaces[symbol] = surface
            
            return surface
            
        except Exception as e:
            logger.error(f"Error building volatility surface for {symbol}: {e}")
            raise
    
    def forecast_implied_volatility(self, symbol: str, forecast_horizon: int = 30) -> Dict[str, Any]:
        """Forecast implied volatility for a symbol.
        
        Args:
            symbol: Stock symbol
            forecast_horizon: Forecast horizon in days
            
        Returns:
            Dictionary with volatility forecast
        """
        try:
            # Build current volatility surface
            surface = self.build_volatility_surface(symbol)
            
            # Simple volatility forecasting using mean reversion
            current_iv = np.mean(surface.implied_volatilities)
            iv_volatility = np.std(surface.implied_volatilities)
            
            # Forecast using mean reversion model
            long_term_iv = 0.25  # Long-term average IV
            mean_reversion_speed = 0.1  # Speed of mean reversion
            
            forecast_iv = current_iv + mean_reversion_speed * (long_term_iv - current_iv) * (forecast_horizon / 365)
            
            # Add uncertainty
            forecast_uncertainty = iv_volatility * np.sqrt(forecast_horizon / 365)
            
            return {
                'symbol': symbol,
                'current_iv': current_iv,
                'forecast_iv': forecast_iv,
                'forecast_uncertainty': forecast_uncertainty,
                'confidence_interval': {
                    'lower': max(0.05, forecast_iv - 2 * forecast_uncertainty),
                    'upper': min(1.0, forecast_iv + 2 * forecast_uncertainty)
                },
                'forecast_date': datetime.now() + timedelta(days=forecast_horizon)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting implied volatility for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def analyze_options_flow(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Analyze options flow for a symbol.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with options flow analysis
        """
        try:
            # Get options data for multiple expirations
            options_df = self.get_options_chain(symbol)
            
            if options_df.empty:
                return {'symbol': symbol, 'error': 'No options data available'}
            
            # Calculate put-call ratio
            calls = options_df[options_df['option_type'] == 'call']
            puts = options_df[options_df['option_type'] == 'put']
            
            put_call_ratio = len(puts) / len(calls) if len(calls) > 0 else 0
            
            # Calculate volume-weighted metrics
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            
            # Find most active strikes
            active_strikes = options_df.groupby('strike')['volume'].sum().sort_values(ascending=False).head(5)
            
            # Calculate IV skew (difference between OTM put and OTM call IV)
            atm_strike = options_df['underlying_price'].iloc[0]
            
            otm_calls = calls[calls['strike'] > atm_strike]
            otm_puts = puts[puts['strike'] < atm_strike]
            
            if not otm_calls.empty and not otm_puts.empty:
                call_iv = otm_calls['implied_volatility'].mean()
                put_iv = otm_puts['implied_volatility'].mean()
                iv_skew = put_iv - call_iv
            else:
                iv_skew = 0.0
            
            return {
                'symbol': symbol,
                'put_call_ratio': put_call_ratio,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'iv_skew': iv_skew,
                'active_strikes': active_strikes.to_dict(),
                'analysis_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing options flow for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

# Example usage
if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize forecaster
    config = {
        'risk_free_rate': 0.02,
        'dividend_yield': 0.01,
        'api_keys': {
            'polygon': 'your_polygon_key_here'
        }
    }
    
    forecaster = OptionsForecaster(config)
    
    # Get options chain
    options_chain = forecaster.get_options_chain('AAPL')
    logger.info(f"Retrieved {len(options_chain)} options contracts")
    
    # Calculate implied volatility
    iv = forecaster.calculate_implied_volatility(5.0, 100.0, 50.0, 0.5, 'call')
    logger.info(f"Implied volatility: {iv:.2%}")
    
    # Calculate Greeks
    greeks = forecaster.calculate_greeks(100.0, 50.0, 0.5, 0.3, 'call')
    logger.info(f"Greeks: {greeks}")
    
    # Forecast volatility
    forecast = forecaster.forecast_implied_volatility('AAPL', 30)
    logger.info(f"Volatility forecast: {forecast}") 