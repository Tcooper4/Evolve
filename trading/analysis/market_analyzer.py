import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas_ta as ta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import logging
from functools import wraps
import time

from trading.utils.common import normalize_indicator_name

# Try to import redis, but make it optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class MarketAnalysisError(Exception):
    """Custom exception for market analysis errors."""
    pass

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying operations on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
            raise last_error
        return wrapper
    return decorator

def validate_dataframe(data: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame structure.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        MarketAnalysisError: If validation fails
    """
    if not isinstance(data, pd.DataFrame):
        raise MarketAnalysisError("Data must be a pandas DataFrame")
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise MarketAnalysisError(f"Missing required columns: {missing_columns}")
    
    if data.empty:
        raise MarketAnalysisError("DataFrame is empty")
    
    if data.isnull().any().any():
        raise MarketAnalysisError("DataFrame contains null values")

class MarketAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the market analyzer.
        
        Args:
            config: Configuration dictionary containing:
                - redis_host: Redis host (default: localhost)
                - redis_port: Redis port (default: 6379)
                - redis_db: Redis database (default: 0)
                - redis_password: Redis password
                - redis_ssl: Whether to use SSL (default: false)
                - log_level: Logging level (default: INFO)
                - cache_ttl: Cache TTL in seconds (default: 3600)
                - results_dir: Directory for saving results (default: market_analysis)
                - max_symbols: Maximum number of symbols to analyze (default: 100)
                - min_data_points: Minimum data points required (default: 100)
        """
        # Load configuration from environment variables with defaults
        self.config = {
            'redis_host': os.getenv('REDIS_HOST', 'localhost'),
            'redis_port': int(os.getenv('REDIS_PORT', 6379)),
            'redis_db': int(os.getenv('REDIS_DB', 0)),
            'redis_password': os.getenv('REDIS_PASSWORD'),
            'redis_ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true',
            'log_level': os.getenv('MARKET_ANALYZER_LOG_LEVEL', 'INFO'),
            'cache_ttl': int(os.getenv('MARKET_ANALYZER_CACHE_TTL', 3600)),
            'results_dir': os.getenv('MARKET_ANALYZER_RESULTS_DIR', 'market_analysis'),
            'max_symbols': int(os.getenv('MARKET_ANALYZER_MAX_SYMBOLS', 100)),
            'min_data_points': int(os.getenv('MARKET_ANALYZER_MIN_DATA_POINTS', 100))
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config['log_level']))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Initialize Redis connection if available
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=self.config['redis_host'],
                    port=self.config['redis_port'],
                    db=self.config['redis_db'],
                    password=self.config['redis_password'],
                    ssl=self.config['redis_ssl']
                )
                # Test connection
                self.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {str(e)}")
                self.redis_client = None
        
        # Create results directory
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Create cache directory
        self.cache_dir = self.results_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize data storage
        self.data = {}
        self.indicators = {}
        
        # Initialize models
        self.scaler = StandardScaler()
        self.regime_model = KMeans(n_clusters=3, random_state=42)
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_columns = []
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache (Redis or file).
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data if available, None otherwise
        """
        # Try Redis first if available
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return pd.read_json(cached_data)
            except Exception as e:
                self.logger.warning(f"Redis cache error: {str(e)}")
        
        # Try file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                # Check if cache is still valid
                if time.time() - cache_file.stat().st_mtime < self.config['cache_ttl']:
                    return pd.read_json(cache_file)
            except Exception as e:
                self.logger.warning(f"File cache error: {str(e)}")
        
        return None
    
    def _set_cached_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Store data in cache (Redis and file).
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        # Try Redis first if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.config['cache_ttl'],
                    data.to_json()
                )
            except Exception as e:
                self.logger.warning(f"Redis cache error: {str(e)}")
        
        # Always store in file cache as backup
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            data.to_json(cache_file)
        except Exception as e:
            self.logger.warning(f"File cache error: {str(e)}")
    
    @retry_on_error(max_retries=3)
    def fetch_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """Fetch market data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: Data period (default: 1y)
            interval: Data interval (default: 1d)
            
        Returns:
            DataFrame with market data
            
        Raises:
            MarketAnalysisError: If data fetching fails
        """
        try:
            # Validate inputs
            if not isinstance(symbol, str) or not symbol:
                raise MarketAnalysisError("Invalid symbol")
            if not isinstance(period, str) or not period:
                raise MarketAnalysisError("Invalid period")
            if not isinstance(interval, str) or not interval:
                raise MarketAnalysisError("Invalid interval")
            
            # Check cache first
            cache_key = f"market_data:{symbol}:{period}:{interval}"
            cached_data = self._get_cached_data(cache_key)
            
            if cached_data is not None:
                self.logger.info(f"Using cached data for {symbol}")
                return cached_data
            
            # Fetch new data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Validate data
            validate_dataframe(data, ['Open', 'High', 'Low', 'Close', 'Volume'])
            
            if len(data) < self.config['min_data_points']:
                raise MarketAnalysisError(f"Insufficient data points for {symbol}")
            
            # Store in cache
            self._set_cached_data(cache_key, data)
            
            self.data[symbol] = data
            self.logger.info(f"Fetched data for {symbol}")
            return data
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def calculate_technical_indicators(self, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with technical indicators
            
        Raises:
            MarketAnalysisError: If indicator calculation fails
        """
        try:
            if symbol not in self.data:
                raise MarketAnalysisError(f"No data available for {symbol}")
            
            data = self.data[symbol].copy()
            
            # Create a custom strategy
            custom_strategy = ta.Strategy(
                name="custom_strategy",
                description="Custom technical analysis strategy",
                ta=[
                    # Trend Indicators
                    {"kind": "sma", "length": 20},
                    {"kind": "sma", "length": 50},
                    {"kind": "sma", "length": 200},
                    {"kind": "ema", "length": 20},
                    {"kind": "ema", "length": 50},
                    {"kind": "ema", "length": 200},
                    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "adx", "length": 14},
                    {"kind": "ichimoku", "tenkan": 9, "kijun": 26, "senkou": 52},
                    {"kind": "psar", "af0": 0.02, "af": 0.02, "max_af": 0.2},
                    {"kind": "supertrend", "length": 10, "multiplier": 3},
                    
                    # Momentum Indicators
                    {"kind": "rsi", "length": 14},
                    {"kind": "stoch", "k": 14, "d": 3},
                    {"kind": "cci", "length": 14},
                    {"kind": "willr", "length": 14},
                    {"kind": "mom", "length": 10},
                    {"kind": "roc", "length": 10},
                    {"kind": "mfi", "length": 14},
                    {"kind": "trix", "length": 18, "signal": 9},
                    {"kind": "massi", "length": 9},
                    {"kind": "dpo", "length": 20},
                    {"kind": "kst", "roc1": 10, "roc2": 15, "roc3": 20, "roc4": 30, "sma1": 10, "sma2": 10, "sma3": 10, "sma4": 15},
                    
                    # Volatility Indicators
                    {"kind": "bbands", "length": 20, "std": 2},
                    {"kind": "atr", "length": 14},
                    {"kind": "natr", "length": 14},
                    {"kind": "tr"},
                    {"kind": "true_range"},
                    
                    # Volume Indicators
                    {"kind": "obv"},
                    {"kind": "vwap"},
                    {"kind": "pvt"},
                    {"kind": "efi", "length": 13},
                    {"kind": "cfi", "length": 14},
                    {"kind": "ebsw", "length": 10},
                    
                    # Custom Indicators
                    {"kind": "tsi", "fast": 13, "slow": 25},
                    {"kind": "uo", "fast": 7, "medium": 14, "slow": 28},
                    {"kind": "ao", "fast": 5, "slow": 34},
                    {"kind": "bop"},
                    {"kind": "cmo", "length": 14},
                    {"kind": "ppo", "fast": 12, "slow": 26, "signal": 9}
                ]
            )

            # Add the strategy to the DataFrame
            data.ta.strategy(custom_strategy)
            
            # Get all the technical indicators
            indicators = data.ta.indicators()
            indicators.rename(columns=lambda c: normalize_indicator_name(c), inplace=True)
            
            # Add basic price-based features
            indicators['returns'] = data['Close'].pct_change()
            indicators['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            indicators['volatility'] = indicators['returns'].rolling(window=20).std()
            
            # Store indicators
            self.indicators[symbol] = indicators
            
            # Cache indicators
            cache_key = f"indicators:{symbol}"
            self._set_cached_data(cache_key, indicators)
            
            self.logger.info(f"Calculated indicators for {symbol}")
            return indicators
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to calculate indicators for {symbol}: {str(e)}")
    
    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Calculate market sentiment indicators.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of sentiment indicators
            
        Raises:
            MarketAnalysisError: If sentiment calculation fails
        """
        try:
            if symbol not in self.indicators:
                raise MarketAnalysisError(f"No indicators available for {symbol}")
            
            data = self.indicators[symbol]
            latest = data.iloc[-1]
            
            # Calculate sentiment scores
            trend_score = 1 if latest['Close'] > latest['SMA_200'] else -1
            rsi_score = -1 if latest['RSI_14'] < 30 else 1 if latest['RSI_14'] > 70 else 0
            macd_score = 1 if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] else -1
            bb_score = -1 if latest['Close'] < latest['BBL_20_2.0'] else 1 if latest['Close'] > latest['BBU_20_2.0'] else 0
            
            # Calculate overall sentiment
            overall_score = (trend_score + rsi_score + macd_score + bb_score) / 4
            
            sentiment = {
                'trend': 'bullish' if trend_score > 0 else 'bearish',
                'rsi_signal': 'oversold' if rsi_score < 0 else 'overbought' if rsi_score > 0 else 'neutral',
                'macd_signal': 'bullish' if macd_score > 0 else 'bearish',
                'bb_signal': 'oversold' if bb_score < 0 else 'overbought' if bb_score > 0 else 'neutral',
                'overall_score': overall_score,
                'overall_sentiment': 'bullish' if overall_score > 0.2 else 'bearish' if overall_score < -0.2 else 'neutral'
            }
            
            # Cache sentiment
            cache_key = f"sentiment:{symbol}"
            self._set_cached_data(cache_key, json.dumps(sentiment))
            
            self.logger.info(f"Calculated sentiment for {symbol}")
            return sentiment
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to calculate sentiment for {symbol}: {str(e)}")
    
    def get_support_resistance(self, symbol: str) -> Dict[str, List[float]]:
        """Calculate support and resistance levels.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of support and resistance levels
            
        Raises:
            MarketAnalysisError: If level calculation fails
        """
        try:
            if symbol not in self.data:
                raise MarketAnalysisError(f"No data available for {symbol}")
            
            data = self.data[symbol]
            
            # Find local extrema
            highs = self._find_local_extrema(data['High'], 'max')
            lows = self._find_local_extrema(data['Low'], 'min')
            
            # Calculate support and resistance levels
            support_levels = self._calculate_support_resistance(lows)
            resistance_levels = self._calculate_support_resistance(highs)
            
            levels = {
                'support': sorted(support_levels)[:3],  # Top 3 support levels
                'resistance': sorted(resistance_levels, reverse=True)[:3]  # Top 3 resistance levels
            }
            
            # Cache levels
            cache_key = f"levels:{symbol}"
            self._set_cached_data(cache_key, json.dumps(levels))
            
            self.logger.info(f"Calculated support/resistance levels for {symbol}")
            return levels
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to calculate support/resistance levels for {symbol}: {str(e)}")
    
    def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market analysis summary.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing market summary
            
        Raises:
            MarketAnalysisError: If summary generation fails
        """
        try:
            if symbol not in self.data:
                raise MarketAnalysisError(f"No data available for {symbol}")
            
            data = self.data[symbol]

            if data.empty:
                return None

            latest = data.iloc[-1]

            def _calc_change(lookback: int) -> Tuple[Optional[float], Optional[float]]:
                if len(data) < 2:
                    return None, None

                index = -lookback if len(data) >= lookback else 0
                prev_close = data.iloc[index]['Close']
                change = latest['Close'] - prev_close
                pct = (change / prev_close) * 100
                return float(change), float(pct)

            daily_change, daily_change_pct = _calc_change(2)
            weekly_change, weekly_change_pct = _calc_change(6)
            monthly_change, monthly_change_pct = _calc_change(21)
            
            summary = {
                'price': {
                    'current': latest['Close'],
                    'daily_change': daily_change,
                    'daily_change_pct': daily_change_pct,
                    'weekly_change': weekly_change,
                    'weekly_change_pct': weekly_change_pct,
                    'monthly_change': monthly_change,
                    'monthly_change_pct': monthly_change_pct
                },
                'volume': {
                    'current': latest['Volume'],
                    'avg_20d': data['Volume'].rolling(window=20).mean().iloc[-1],
                    'volume_ratio': latest['Volume'] / data['Volume'].rolling(window=20).mean().iloc[-1]
                },
                'sentiment': self.get_market_sentiment(symbol),
                'support_resistance': self.get_support_resistance(symbol),
                'volatility': {
                    'current': data['Close'].pct_change().rolling(window=20).std().iloc[-1],
                    'historical_avg': data['Close'].pct_change().std()
                },
                'trend': {
                    'sma_20': latest['SMA_20'] if 'SMA_20' in latest else None,
                    'sma_50': latest['SMA_50'] if 'SMA_50' in latest else None,
                    'sma_200': latest['SMA_200'] if 'SMA_200' in latest else None,
                    'ema_20': latest['EMA_20'] if 'EMA_20' in latest else None,
                    'ema_50': latest['EMA_50'] if 'EMA_50' in latest else None,
                    'ema_200': latest['EMA_200'] if 'EMA_200' in latest else None
                },
                'momentum': {
                    'rsi': latest['RSI_14'] if 'RSI_14' in latest else None,
                    'macd': latest['MACD_12_26_9'] if 'MACD_12_26_9' in latest else None,
                    'macd_signal': latest['MACDs_12_26_9'] if 'MACDs_12_26_9' in latest else None,
                    'macd_hist': latest['MACDh_12_26_9'] if 'MACDh_12_26_9' in latest else None,
                    'stoch_k': latest['STOCHk_14_3_3'] if 'STOCHk_14_3_3' in latest else None,
                    'stoch_d': latest['STOCHd_14_3_3'] if 'STOCHd_14_3_3' in latest else None
                },
                'volatility_indicators': {
                    'bb_upper': latest['BBU_20_2.0'] if 'BBU_20_2.0' in latest else None,
                    'bb_middle': latest['BBM_20_2.0'] if 'BBM_20_2.0' in latest else None,
                    'bb_lower': latest['BBL_20_2.0'] if 'BBL_20_2.0' in latest else None,
                    'atr': latest['ATR_14'] if 'ATR_14' in latest else None,
                    'natr': latest['NATR_14'] if 'NATR_14' in latest else None
                },
                'volume_indicators': {
                    'obv': latest['OBV'] if 'OBV' in latest else None,
                    'vwap': latest['VWAP'] if 'VWAP' in latest else None,
                    'mfi': latest['MFI_14'] if 'MFI_14' in latest else None
                }
            }
            
            # Cache summary
            cache_key = f"summary:{symbol}"
            self._set_cached_data(cache_key, json.dumps(summary))
            
            self.logger.info(f"Generated market summary for {symbol}")
            return summary
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to generate market summary for {symbol}: {str(e)}")
    
    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive market analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            MarketAnalysisError: If analysis fails
        """
        try:
            if len(data) < self.config['min_data_points']:
                raise MarketAnalysisError("Insufficient data points for analysis")
            
            analysis = {}
            
            # Technical analysis
            analysis['technical'] = self._perform_technical_analysis(data)
            
            # Market regime
            analysis['regime'] = self._detect_market_regime(data)
            
            # Correlation analysis
            analysis['correlation'] = self._analyze_correlations(data)
            
            # Volatility analysis
            analysis['volatility'] = self._analyze_volatility(data)
            
            # Volume analysis
            analysis['volume'] = self._analyze_volume(data)
            
            # Market structure
            analysis['structure'] = self._analyze_market_structure(data)
            
            # Pattern recognition
            analysis['patterns'] = self._identify_chart_patterns(data)
            
            self.logger.info("Completed market analysis")
            return analysis
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to analyze market: {str(e)}")
    
    def _perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform technical analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of technical indicators
            
        Raises:
            MarketAnalysisError: If technical analysis fails
        """
        try:
            # Create a custom strategy
            custom_strategy = ta.Strategy(
                name="custom_strategy",
                description="Custom technical analysis strategy",
                ta=[
                    # Trend Indicators
                    {"kind": "sma", "length": 20},
                    {"kind": "sma", "length": 50},
                    {"kind": "sma", "length": 200},
                    {"kind": "ema", "length": 20},
                    {"kind": "ema", "length": 50},
                    {"kind": "ema", "length": 200},
                    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                    {"kind": "adx", "length": 14},
                    {"kind": "ichimoku", "tenkan": 9, "kijun": 26, "senkou": 52},
                    {"kind": "psar", "af0": 0.02, "af": 0.02, "max_af": 0.2},
                    {"kind": "supertrend", "length": 10, "multiplier": 3},
                    
                    # Momentum Indicators
                    {"kind": "rsi", "length": 14},
                    {"kind": "stoch", "k": 14, "d": 3},
                    {"kind": "cci", "length": 14},
                    {"kind": "willr", "length": 14},
                    {"kind": "mom", "length": 10},
                    {"kind": "roc", "length": 10},
                    {"kind": "mfi", "length": 14},
                    {"kind": "trix", "length": 18, "signal": 9},
                    {"kind": "massi", "length": 9},
                    {"kind": "dpo", "length": 20},
                    {"kind": "kst", "roc1": 10, "roc2": 15, "roc3": 20, "roc4": 30, "sma1": 10, "sma2": 10, "sma3": 10, "sma4": 15},
                    
                    # Volatility Indicators
                    {"kind": "bbands", "length": 20, "std": 2},
                    {"kind": "atr", "length": 14},
                    {"kind": "natr", "length": 14},
                    {"kind": "tr"},
                    {"kind": "true_range"},
                    
                    # Volume Indicators
                    {"kind": "obv"},
                    {"kind": "vwap"},
                    {"kind": "pvt"},
                    {"kind": "efi", "length": 13},
                    {"kind": "cfi", "length": 14},
                    {"kind": "ebsw", "length": 10},
                    
                    # Custom Indicators
                    {"kind": "tsi", "fast": 13, "slow": 25},
                    {"kind": "uo", "fast": 7, "medium": 14, "slow": 28},
                    {"kind": "ao", "fast": 5, "slow": 34},
                    {"kind": "bop"},
                    {"kind": "cmo", "length": 14},
                    {"kind": "ppo", "fast": 12, "slow": 26, "signal": 9}
                ]
            )

            # Add the strategy to the DataFrame
            data.ta.strategy(custom_strategy)
            
            # Get all the technical indicators
            indicators = data.ta.indicators()
            
            # Organize indicators by category
            analysis = {
                'trend': {
                    'sma_20': indicators['SMA_20'],
                    'sma_50': indicators['SMA_50'],
                    'sma_200': indicators['SMA_200'],
                    'ema_20': indicators['EMA_20'],
                    'ema_50': indicators['EMA_50'],
                    'ema_200': indicators['EMA_200'],
                    'macd': indicators['MACD_12_26_9'],
                    'macd_signal': indicators['MACDs_12_26_9'],
                    'macd_hist': indicators['MACDh_12_26_9'],
                    'adx': indicators['ADX_14']
                },
                'momentum': {
                    'rsi': indicators['RSI_14'],
                    'stoch_k': indicators['STOCHk_14_3_3'],
                    'stoch_d': indicators['STOCHd_14_3_3'],
                    'cci': indicators['CCI_14'],
                    'willr': indicators['WILLR_14'],
                    'mom': indicators['MOM_10'],
                    'roc': indicators['ROC_10'],
                    'mfi': indicators['MFI_14']
                },
                'volatility': {
                    'bb_upper': indicators['BBU_20_2.0'],
                    'bb_middle': indicators['BBM_20_2.0'],
                    'bb_lower': indicators['BBL_20_2.0'],
                    'atr': indicators['ATR_14'],
                    'natr': indicators['NATR_14'],
                    'tr': indicators['TRUERANGE_1']
                },
                'volume': {
                    'obv': indicators['OBV'],
                    'vwap': indicators['VWAP'],
                    'pvt': indicators['PVT'],
                    'efi': indicators['EFI_13'],
                    'cfi': indicators['CFI_14'],
                    'ebsw': indicators['EBSW_10']
                },
                'custom': {
                    'tsi': indicators['TSI_13_25'],
                    'uo': indicators['UO_7_14_28'],
                    'ao': indicators['AO_5_34'],
                    'bop': indicators['BOP'],
                    'cmo': indicators['CMO_14'],
                    'ppo': indicators['PPO_12_26_9']
                }
            }
            
            return analysis
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to perform technical analysis: {str(e)}")
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime using clustering.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of regime statistics
            
        Raises:
            MarketAnalysisError: If regime detection fails
        """
        try:
            # Prepare features for regime detection
            features = pd.DataFrame({
                'returns': data['Close'].pct_change(),
                'volatility': data['Close'].pct_change().rolling(window=20).std(),
                'volume_ma_ratio': data['Volume'] / data['Volume'].rolling(window=20).mean(),
                'rsi': ta.rsi(data['Close'])
            }).fillna(0)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Fit regime model
            self.regime_model.fit(scaled_features)
            
            # Get regime labels
            regimes = self.regime_model.predict(scaled_features)
            
            # Calculate regime statistics
            regime_stats = {
                'current_regime': int(regimes[-1]),
                'regime_duration': self._calculate_regime_duration(regimes),
                'regime_probability': self._calculate_regime_probability(scaled_features[-1]),
                'regime_volatility': features['volatility'].iloc[-1],
                'regime_volume': features['volume_ma_ratio'].iloc[-1]
            }
            
            return regime_stats
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to detect market regime: {str(e)}")
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different market aspects.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of correlation analysis
            
        Raises:
            MarketAnalysisError: If correlation analysis fails
        """
        try:
            # Calculate returns for different timeframes
            returns = pd.DataFrame({
                'daily': data['Close'].pct_change(),
                'weekly': data['Close'].pct_change(5),
                'monthly': data['Close'].pct_change(20)
            })
            
            # Calculate correlations
            correlation_matrix = returns.corr()
            
            # Calculate rolling correlations
            rolling_corr = returns['daily'].rolling(window=20).corr(returns['weekly'])
            
            # Calculate autocorrelation
            autocorr = pd.Series(returns['daily']).autocorr()
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'rolling_correlation': rolling_corr.to_dict(),
                'autocorrelation': autocorr
            }
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to analyze correlations: {str(e)}")
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of volatility analysis
            
        Raises:
            MarketAnalysisError: If volatility analysis fails
        """
        try:
            returns = data['Close'].pct_change()
            
            # Calculate different volatility measures
            volatility = {
                'historical': returns.rolling(window=20).std(),
                'parkinson': np.sqrt(1 / (4 * np.log(2)) * 
                                   (np.log(data['High'] / data['Low'])**2).rolling(window=20).mean()),
                'garman_klass': np.sqrt(0.5 * np.log(data['High'] / data['Low'])**2 - 
                                      (2 * np.log(2) - 1) * np.log(data['Close'] / data['Open'])**2)
            }
            
            # Volatility regime
            vol_regime = pd.cut(volatility['historical'], 
                              bins=[-np.inf, 0.01, 0.02, np.inf],
                              labels=['Low', 'Medium', 'High'])
            
            # Volatility term structure
            term_structure = self._calculate_volatility_term_structure(returns)
            
            return {
                'current_volatility': volatility['historical'].iloc[-1],
                'volatility_regime': vol_regime.iloc[-1],
                'parkinson_volatility': volatility['parkinson'].iloc[-1],
                'garman_klass_volatility': volatility['garman_klass'].iloc[-1],
                'term_structure': term_structure.to_dict()
            }
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to analyze volatility: {str(e)}")
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading volume.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of volume analysis
            
        Raises:
            MarketAnalysisError: If volume analysis fails
        """
        try:
            # Calculate volume metrics
            volume_ma = data['Volume'].rolling(window=20).mean()
            volume_std = data['Volume'].rolling(window=20).std()
            
            # Volume profile
            volume_profile = self._calculate_volume_profile(data)
            
            # Volume trend
            volume_trend = (data['Volume'] > volume_ma).rolling(window=5).mean()
            
            return {
                'current_volume': data['Volume'].iloc[-1],
                'volume_ma': volume_ma.iloc[-1],
                'volume_std': volume_std.iloc[-1],
                'volume_trend': volume_trend.iloc[-1],
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to analyze volume: {str(e)}")
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of market structure analysis
            
        Raises:
            MarketAnalysisError: If structure analysis fails
        """
        try:
            # Find local extrema
            highs = self._find_local_extrema(data['High'], 'max')
            lows = self._find_local_extrema(data['Low'], 'min')
            
            # Calculate support and resistance levels
            support_levels = self._calculate_support_resistance(lows)
            resistance_levels = self._calculate_support_resistance(highs)
            
            # Identify chart patterns
            patterns = self._identify_chart_patterns(data)
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'patterns': patterns,
                'trend_strength': ta.adx(data['High'], data['Low'], data['Close']).iloc[-1]
            }
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to analyze market structure: {str(e)}")
    
    def _calculate_regime_duration(self, regimes: np.ndarray) -> int:
        """Calculate current regime duration.
        
        Args:
            regimes: Array of regime labels
            
        Returns:
            Duration of current regime
        """
        current_regime = regimes[-1]
        duration = 1
        
        for i in range(len(regimes) - 2, -1, -1):
            if regimes[i] == current_regime:
                duration += 1
            else:
                break
                
        return duration
    
    def _calculate_regime_probability(self, features: np.ndarray) -> float:
        """Calculate probability of current regime.
        
        Args:
            features: Scaled feature vector
            
        Returns:
            Probability of current regime
        """
        distances = self.regime_model.transform(features.reshape(1, -1))
        probabilities = 1 / (1 + distances)
        return float(probabilities[0][self.regime_model.predict(features.reshape(1, -1))[0]])
    
    def _calculate_volatility_term_structure(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate volatility term structure.
        
        Args:
            returns: Series of returns
            
        Returns:
            DataFrame of volatility term structure
        """
        windows = [5, 10, 20, 60, 120]
        term_structure = pd.DataFrame()
        
        for window in windows:
            term_structure[f'{window}d'] = returns.rolling(window=window).std()
            
        return term_structure
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of volume profile
        """
        price_bins = pd.qcut(data['Close'], q=10)
        volume_profile = data.groupby(price_bins)['Volume'].sum()
        
        return {
            'price_levels': volume_profile.index.tolist(),
            'volumes': volume_profile.values.tolist()
        }
    
    def _find_local_extrema(self, series: pd.Series, kind: str) -> pd.Series:
        """Find local extrema in a series.
        
        Args:
            series: Series to analyze
            kind: Type of extrema ('max' or 'min')
            
        Returns:
            Series of extrema
        """
        if kind == 'max':
            return series[series == series.rolling(window=5, center=True).max()]
        else:
            return series[series == series.rolling(window=5, center=True).min()]
    
    def _calculate_support_resistance(self, extrema: pd.Series) -> List[float]:
        """Calculate support/resistance levels from extrema.
        
        Args:
            extrema: Series of extrema
            
        Returns:
            List of support/resistance levels
        """
        # Cluster extrema to find significant levels
        if len(extrema) < 2:
            return []
            
        kmeans = KMeans(n_clusters=min(5, len(extrema)), random_state=42)
        clusters = kmeans.fit_predict(extrema.values.reshape(-1, 1))
        
        levels = []
        for i in range(kmeans.n_clusters):
            cluster_points = extrema[clusters == i]
            if len(cluster_points) > 0:
                levels.append(float(cluster_points.mean()))
                
        return sorted(levels)
    
    def _identify_chart_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify chart patterns using both traditional and ML-based approaches.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of identified patterns
            
        Raises:
            MarketAnalysisError: If pattern identification fails
        """
        try:
            patterns = {}
            
            # Traditional pattern recognition
            patterns['traditional'] = {
                'double_top': ta.cdl_double_top(data['Open'], data['High'], data['Low'], data['Close']),
                'double_bottom': ta.cdl_double_bottom(data['Open'], data['High'], data['Low'], data['Close']),
                'head_shoulders': ta.cdl_harami(data['Open'], data['High'], data['Low'], data['Close']),
                'bullish_engulfing': ta.cdl_engulfing(data['Open'], data['High'], data['Low'], data['Close']),
                'bearish_engulfing': ta.cdl_engulfing(data['Open'], data['High'], data['Low'], data['Close']),
                'doji': ta.cdl_doji(data['Open'], data['High'], data['Low'], data['Close']),
                'hammer': ta.cdl_hammer(data['Open'], data['High'], data['Low'], data['Close']),
                'shooting_star': ta.cdl_shooting_star(data['Open'], data['High'], data['Low'], data['Close']),
                'morning_star': ta.cdl_morning_star(data['Open'], data['High'], data['Low'], data['Close']),
                'evening_star': ta.cdl_evening_star(data['Open'], data['High'], data['Low'], data['Close'])
            }
            
            # ML-based pattern recognition
            features = self._extract_pattern_features(data)
            pattern_probs = self._predict_patterns(features)
            
            patterns['ml_based'] = {
                'pattern_probabilities': pattern_probs,
                'dominant_pattern': max(pattern_probs.items(), key=lambda x: x[1])[0]
            }
            
            # Combine traditional and ML-based patterns
            patterns['combined'] = self._combine_pattern_signals(patterns['traditional'], patterns['ml_based'])
            
            return patterns
            
        except Exception as e:
            raise MarketAnalysisError(f"Failed to identify patterns: {str(e)}")
    
    def _extract_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for pattern recognition.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame of pattern features
        """
        features = pd.DataFrame()
        
        # Price action features
        features['returns'] = data['Close'].pct_change()
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        
        # Volume features
        features['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        features['volume_trend'] = data['Volume'].pct_change()
        
        # Technical indicators
        features['rsi'] = ta.rsi(data['Close'])
        features['macd'] = ta.macd(data['Close'])[0]
        features['macd_signal'] = ta.macd(data['Close'])[1]
        features['adx'] = ta.adx(data['High'], data['Low'], data['Close'])
        
        # Pattern-specific features
        features['body_size'] = abs(data['Close'] - data['Open'])
        features['upper_shadow'] = data['High'] - data[['Open', 'Close']].max(axis=1)
        features['lower_shadow'] = data[['Open', 'Close']].min(axis=1) - data['Low']
        
        return features.fillna(0)
    
    def _predict_patterns(self, features: pd.DataFrame) -> Dict[str, float]:
        """Predict pattern probabilities using ML model.
        
        Args:
            features: DataFrame of pattern features
            
        Returns:
            Dictionary of pattern probabilities
        """
        # Initialize pattern classifier if not exists
        if not hasattr(self, 'pattern_classifier'):
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            # Note: In a real implementation, you would train this model
            # with labeled pattern data
        
        # Get predictions
        predictions = self.pattern_classifier.predict_proba(features.iloc[-1:])
        
        # Map probabilities to pattern names
        pattern_names = ['double_top', 'double_bottom', 'head_shoulders',
                        'bullish_engulfing', 'bearish_engulfing', 'doji']
        
        return dict(zip(pattern_names, predictions[0]))
    
    def _combine_pattern_signals(self, traditional: Dict[str, Any],
                               ml_based: Dict[str, Any]) -> Dict[str, Any]:
        """Combine traditional and ML-based pattern signals.
        
        Args:
            traditional: Dictionary of traditional pattern signals
            ml_based: Dictionary of ML-based pattern signals
            
        Returns:
            Dictionary of combined pattern signals
        """
        combined = {}
        
        # Convert traditional signals to probabilities
        for pattern, signal in traditional.items():
            if signal.iloc[-1] != 0:
                combined[pattern] = {
                    'signal': signal.iloc[-1],
                    'confidence': abs(signal.iloc[-1]) / 100,
                    'source': 'traditional'
                }
        
        # Add ML-based predictions
        for pattern, prob in ml_based['pattern_probabilities'].items():
            if pattern in combined:
                # Combine signals if pattern exists in both
                combined[pattern]['ml_confidence'] = prob
                combined[pattern]['combined_confidence'] = (
                    combined[pattern]['confidence'] + prob
                ) / 2
            else:
                # Add new pattern from ML
                combined[pattern] = {
                    'signal': 1 if prob > 0.5 else -1,
                    'confidence': prob,
                    'source': 'ml'
                }
        
        return combined 