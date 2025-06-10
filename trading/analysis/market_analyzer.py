import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import talib
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import yfinance as yf
from datetime import datetime, timedelta

class MarketAnalyzer:
    def __init__(self):
        """Initialize the market analyzer."""
        self.data = {}
        self.indicators = {}
    
    def fetch_data(self, symbol: str, period: str = '1y', interval: str = '1d'):
        """Fetch market data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            self.data[symbol] = data
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, symbol: str):
        """Calculate technical indicators for a symbol."""
        if symbol not in self.data:
            return None
        
        data = self.data[symbol]
        
        # Calculate moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
        
        self.indicators[symbol] = data
        return data
    
    def get_market_sentiment(self, symbol: str) -> Dict[str, float]:
        """Calculate market sentiment indicators."""
        if symbol not in self.indicators:
            return {}
        
        data = self.indicators[symbol]
        latest = data.iloc[-1]
        
        sentiment = {
            'trend': 'bullish' if latest['Close'] > latest['SMA_200'] else 'bearish',
            'rsi_signal': 'oversold' if latest['RSI'] < 30 else 'overbought' if latest['RSI'] > 70 else 'neutral',
            'macd_signal': 'bullish' if latest['MACD'] > latest['Signal_Line'] else 'bearish',
            'bb_signal': 'oversold' if latest['Close'] < latest['BB_Lower'] else 'overbought' if latest['Close'] > latest['BB_Upper'] else 'neutral'
        }
        
        return sentiment
    
    def get_support_resistance(self, symbol: str) -> Dict[str, List[float]]:
        """Calculate support and resistance levels."""
        if symbol not in self.data:
            return {'support': [], 'resistance': []}
        
        data = self.data[symbol]
        
        # Simple support/resistance calculation
        recent_lows = data['Low'].rolling(window=20).min()
        recent_highs = data['High'].rolling(window=20).max()
        
        support_levels = recent_lows.unique()
        resistance_levels = recent_highs.unique()
        
        return {
            'support': sorted(support_levels)[:3],  # Top 3 support levels
            'resistance': sorted(resistance_levels, reverse=True)[:3]  # Top 3 resistance levels
        }
    
    def get_market_summary(self, symbol: str) -> Dict:
        """Get comprehensive market analysis summary."""
        if symbol not in self.data:
            return {}
        
        data = self.data[symbol]
        latest = data.iloc[-1]
        
        return {
            'price': latest['Close'],
            'change': latest['Close'] - data.iloc[-2]['Close'],
            'change_pct': (latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100,
            'volume': latest['Volume'],
            'sentiment': self.get_market_sentiment(symbol),
            'support_resistance': self.get_support_resistance(symbol)
        }

    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """Perform comprehensive market analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing analysis results
        """
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
        
        return analysis
    
    def _perform_technical_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform technical analysis."""
        analysis = {}
        
        # Trend indicators
        analysis['trend'] = {
            'sma_20': talib.SMA(data['close'], timeperiod=20),
            'sma_50': talib.SMA(data['close'], timeperiod=50),
            'sma_200': talib.SMA(data['close'], timeperiod=200),
            'adx': talib.ADX(data['high'], data['low'], data['close']),
            'macd': talib.MACD(data['close'])[0],
            'macd_signal': talib.MACD(data['close'])[1]
        }
        
        # Momentum indicators
        analysis['momentum'] = {
            'rsi': talib.RSI(data['close']),
            'stoch_k': talib.STOCH(data['high'], data['low'], data['close'])[0],
            'stoch_d': talib.STOCH(data['high'], data['low'], data['close'])[1],
            'cci': talib.CCI(data['high'], data['low'], data['close'])
        }
        
        # Volatility indicators
        analysis['volatility'] = {
            'atr': talib.ATR(data['high'], data['low'], data['close']),
            'natr': talib.NATR(data['high'], data['low'], data['close']),
            'bollinger_upper': talib.BBANDS(data['close'])[0],
            'bollinger_middle': talib.BBANDS(data['close'])[1],
            'bollinger_lower': talib.BBANDS(data['close'])[2]
        }
        
        # Volume indicators
        analysis['volume'] = {
            'obv': talib.OBV(data['close'], data['volume']),
            'ad': talib.AD(data['high'], data['low'], data['close'], data['volume']),
            'adosc': talib.ADOSC(data['high'], data['low'], data['close'], data['volume'])
        }
        
        return analysis
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """Detect market regime using clustering."""
        # Prepare features for regime detection
        features = pd.DataFrame({
            'returns': data['close'].pct_change(),
            'volatility': data['close'].pct_change().rolling(window=20).std(),
            'volume_ma_ratio': data['volume'] / data['volume'].rolling(window=20).mean(),
            'rsi': talib.RSI(data['close'])
        }).fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit regime model
        self.regime_model.fit(scaled_features)
        
        # Get regime labels
        regimes = self.regime_model.predict(scaled_features)
        
        # Calculate regime statistics
        regime_stats = {
            'current_regime': regimes[-1],
            'regime_duration': self._calculate_regime_duration(regimes),
            'regime_probability': self._calculate_regime_probability(scaled_features[-1])
        }
        
        return regime_stats
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze correlations between different market aspects."""
        # Calculate returns for different timeframes
        returns = pd.DataFrame({
            'daily': data['close'].pct_change(),
            'weekly': data['close'].pct_change(5),
            'monthly': data['close'].pct_change(20)
        })
        
        # Calculate correlations
        correlation_matrix = returns.corr()
        
        # Calculate rolling correlations
        rolling_corr = returns['daily'].rolling(window=20).corr(returns['weekly'])
        
        return {
            'correlation_matrix': correlation_matrix,
            'rolling_correlation': rolling_corr
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analyze market volatility."""
        returns = data['close'].pct_change()
        
        # Calculate different volatility measures
        volatility = {
            'historical': returns.rolling(window=20).std(),
            'parkinson': np.sqrt(1 / (4 * np.log(2)) * 
                               (np.log(data['high'] / data['low'])**2).rolling(window=20).mean()),
            'garman_klass': np.sqrt(0.5 * np.log(data['high'] / data['low'])**2 - 
                                  (2 * np.log(2) - 1) * np.log(data['close'] / data['open'])**2)
        }
        
        # Volatility regime
        vol_regime = pd.cut(volatility['historical'], 
                          bins=[-np.inf, 0.01, 0.02, np.inf],
                          labels=['Low', 'Medium', 'High'])
        
        return {
            'measures': volatility,
            'regime': vol_regime,
            'term_structure': self._calculate_volatility_term_structure(returns)
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze trading volume patterns."""
        volume_analysis = {
            'volume_ma': data['volume'].rolling(window=20).mean(),
            'volume_std': data['volume'].rolling(window=20).std(),
            'volume_trend': data['volume'].rolling(window=20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            ),
            'volume_profile': self._calculate_volume_profile(data)
        }
        
        return volume_analysis
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure including support/resistance levels."""
        # Find local extrema
        highs = self._find_local_extrema(data['high'], 'max')
        lows = self._find_local_extrema(data['low'], 'min')
        
        # Calculate support and resistance levels
        support_levels = self._calculate_support_resistance(lows)
        resistance_levels = self._calculate_support_resistance(highs)
        
        # Identify chart patterns
        patterns = self._identify_chart_patterns(data)
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'patterns': patterns
        }
    
    def _calculate_regime_duration(self, regimes: np.ndarray) -> int:
        """Calculate duration of current regime."""
        current_regime = regimes[-1]
        duration = 1
        for regime in reversed(regimes[:-1]):
            if regime == current_regime:
                duration += 1
            else:
                break
        return duration
    
    def _calculate_regime_probability(self, features: np.ndarray) -> float:
        """Calculate probability of current regime."""
        distances = self.regime_model.transform(features.reshape(1, -1))
        probabilities = 1 / (1 + distances)
        return probabilities[0][self.regime_model.predict(features.reshape(1, -1))[0]]
    
    def _calculate_volatility_term_structure(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate volatility term structure."""
        term_structure = pd.DataFrame()
        for window in [5, 10, 20, 50, 100]:
            term_structure[f'{window}d'] = returns.rolling(window=window).std()
        return term_structure
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict:
        """Calculate volume profile."""
        price_bins = pd.qcut(data['close'], q=10)
        volume_profile = data.groupby(price_bins)['volume'].sum()
        return {
            'profile': volume_profile,
            'poc_price': volume_profile.idxmax().left  # Point of Control
        }
    
    def _find_local_extrema(self, series: pd.Series, kind: str) -> pd.Series:
        """Find local extrema in price series."""
        if kind == 'max':
            return series[series.shift(1) < series][series > series.shift(-1)]
        else:
            return series[series.shift(1) > series][series < series.shift(-1)]
    
    def _calculate_support_resistance(self, extrema: pd.Series) -> List[float]:
        """Calculate support/resistance levels using clustering."""
        if len(extrema) < 2:
            return []
        
        # Cluster price levels
        kmeans = KMeans(n_clusters=min(5, len(extrema)))
        clusters = kmeans.fit(extrema.values.reshape(-1, 1))
        
        return sorted(clusters.cluster_centers_.flatten())
    
    def _identify_chart_patterns(self, data: pd.DataFrame) -> Dict:
        """Identify common chart patterns."""
        patterns = {}
        
        # Double top/bottom
        patterns['double_top'] = talib.CDLDOUBLESTOP(data['open'], data['high'], 
                                                    data['low'], data['close'])
        patterns['double_bottom'] = talib.CDLDOUBLEBOTTOM(data['open'], data['high'], 
                                                         data['low'], data['close'])
        
        # Head and shoulders
        patterns['head_shoulders'] = talib.CDLHARAMI(data['open'], data['high'], 
                                                    data['low'], data['close'])
        
        # Engulfing patterns
        patterns['bullish_engulfing'] = talib.CDLENGULFING(data['open'], data['high'], 
                                                          data['low'], data['close'])
        
        return patterns 