import unittest
import pandas as pd
import numpy as np
from src.analysis.market_analysis import MarketAnalysis, MarketRegime, MarketCondition

class TestMarketAnalysis(unittest.TestCase):
    """Test suite for market analysis module"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        self.data = pd.DataFrame({
            'open': np.random.normal(100, 2, len(dates)),
            'high': np.random.normal(102, 2, len(dates)),
            'low': np.random.normal(98, 2, len(dates)),
            'close': np.random.normal(100, 2, len(dates)),
            'volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        # Ensure high is highest and low is lowest
        self.data['high'] = self.data[['open', 'high', 'close']].max(axis=1)
        self.data['low'] = self.data[['open', 'low', 'close']].min(axis=1)
        
        # Initialize market analysis
        self.market_analysis = MarketAnalysis()
    
    def test_market_regime_classification(self):
        """Test market regime classification"""
        regime = self.market_analysis._analyze_market_regime(self.data, {})
        self.assertIsInstance(regime, MarketRegime)
        self.assertIsInstance(regime.name, str)
        self.assertIsInstance(regime.description, str)
        self.assertIsInstance(regime.conditions, dict)
        self.assertIsInstance(regime.metrics, dict)
        self.assertIsInstance(regime.confidence, float)
        self.assertTrue(0 <= regime.confidence <= 1)
    
    def test_market_condition_analysis(self):
        """Test market condition analysis"""
        conditions = self.market_analysis._analyze_market_conditions(self.data, {})
        self.assertIsInstance(conditions, list)
        for condition in conditions:
            self.assertIsInstance(condition, MarketCondition)
            self.assertIsInstance(condition.name, str)
            self.assertIsInstance(condition.description, str)
            self.assertIsInstance(condition.indicators, dict)
            self.assertIsInstance(condition.signals, dict)
            self.assertIsInstance(condition.strength, float)
            self.assertTrue(0 <= condition.strength <= 1)
    
    def test_trend_analysis(self):
        """Test trend analysis"""
        trend_condition = self.market_analysis._analyze_trend_condition(self.data, {})
        if trend_condition:
            self.assertIsInstance(trend_condition, MarketCondition)
            self.assertIn('trend', trend_condition.signals)
            self.assertIn('strength', trend_condition.signals)
    
    def test_momentum_analysis(self):
        """Test momentum analysis"""
        momentum_condition = self.market_analysis._analyze_momentum_condition(self.data, {})
        if momentum_condition:
            self.assertIsInstance(momentum_condition, MarketCondition)
            self.assertIn('momentum', momentum_condition.signals)
    
    def test_volatility_analysis(self):
        """Test volatility analysis"""
        volatility_condition = self.market_analysis._analyze_volatility_condition(self.data, {})
        if volatility_condition:
            self.assertIsInstance(volatility_condition, MarketCondition)
            self.assertIn('volatility', volatility_condition.signals)
    
    def test_volume_analysis(self):
        """Test volume analysis"""
        volume_condition = self.market_analysis._analyze_volume_condition(self.data, {})
        if volume_condition:
            self.assertIsInstance(volume_condition, MarketCondition)
            self.assertIn('volume', volume_condition.signals)
            self.assertIn('flow', volume_condition.signals)
    
    def test_support_resistance_analysis(self):
        """Test support/resistance analysis"""
        sr_condition = self.market_analysis._analyze_support_resistance_condition(self.data, {})
        if sr_condition:
            self.assertIsInstance(sr_condition, MarketCondition)
            self.assertIn('position', sr_condition.signals)
    
    def test_trend_signals(self):
        """Test trend signal generation"""
        signals = self.market_analysis._generate_trend_signals(self.data, {}, None)
        self.assertIsInstance(signals, dict)
        self.assertIn('signal', signals)
        self.assertIn('strength', signals)
        self.assertTrue(0 <= signals['strength'] <= 1)
    
    def test_momentum_signals(self):
        """Test momentum signal generation"""
        signals = self.market_analysis._generate_momentum_signals(self.data, {}, None)
        self.assertIsInstance(signals, dict)
        self.assertIn('signal', signals)
        self.assertIn('strength', signals)
        self.assertTrue(0 <= signals['strength'] <= 1)
    
    def test_volatility_signals(self):
        """Test volatility signal generation"""
        signals = self.market_analysis._generate_volatility_signals(self.data, {}, None)
        self.assertIsInstance(signals, dict)
        self.assertIn('signal', signals)
        self.assertIn('strength', signals)
        self.assertTrue(0 <= signals['strength'] <= 1)
    
    def test_volume_signals(self):
        """Test volume signal generation"""
        signals = self.market_analysis._generate_volume_signals(self.data, {}, None)
        self.assertIsInstance(signals, dict)
        self.assertIn('signal', signals)
        self.assertIn('strength', signals)
        self.assertTrue(0 <= signals['strength'] <= 1)
    
    def test_support_resistance_signals(self):
        """Test support/resistance signal generation"""
        signals = self.market_analysis._generate_support_resistance_signals(self.data, {}, None)
        self.assertIsInstance(signals, dict)
        self.assertIn('signal', signals)
        self.assertIn('strength', signals)
        self.assertTrue(0 <= signals['strength'] <= 1)
    
    def test_signal_combination(self):
        """Test signal combination"""
        signals = {
            'trend': {'signal': 'buy', 'strength': 0.8},
            'momentum': {'signal': 'buy', 'strength': 0.7},
            'volatility': {'signal': 'neutral', 'strength': 0.5},
            'volume': {'signal': 'buy', 'strength': 0.6},
            'support_resistance': {'signal': 'buy', 'strength': 0.9}
        }
        
        regime = MarketRegime(
            name='Bull',
            description='Bullish market',
            conditions={'trend': 'up'},
            metrics={'trend_strength': 0.8},
            confidence=0.8
        )
        
        conditions = [
            MarketCondition(
                name='Strong Trend',
                description='Strong uptrend',
                indicators={'adx': 30},
                signals={'trend': 'up'},
                strength=0.8
            )
        ]
        
        combined_signal = self.market_analysis._combine_signals(signals, regime, conditions)
        self.assertIsInstance(combined_signal, dict)
        self.assertIn('signal', combined_signal)
        self.assertIn('strength', combined_signal)
        self.assertIn('confidence', combined_signal)
        self.assertTrue(0 <= combined_signal['strength'] <= 1)
        self.assertTrue(0 <= combined_signal['confidence'] <= 1)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive market analysis"""
        analysis = self.market_analysis.analyze_market(self.data)
        self.assertIsInstance(analysis, dict)
        self.assertIn('indicators', analysis)
        self.assertIn('regime', analysis)
        self.assertIn('conditions', analysis)
        self.assertIn('signals', analysis)
        
        self.assertIsInstance(analysis['regime'], MarketRegime)
        self.assertIsInstance(analysis['conditions'], list)
        self.assertIsInstance(analysis['signals'], dict)
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        # Test SMA
        sma = self.market_analysis._calculate_sma(self.data)
        self.assertIsInstance(sma, pd.Series)
        self.assertEqual(len(sma), len(self.data))
        
        # Test EMA
        ema = self.market_analysis._calculate_ema(self.data)
        self.assertIsInstance(ema, pd.Series)
        self.assertEqual(len(ema), len(self.data))
        
        # Test MACD
        macd, signal, hist = self.market_analysis._calculate_macd(self.data)
        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(signal, pd.Series)
        self.assertIsInstance(hist, pd.Series)
        self.assertEqual(len(macd), len(self.data))
        
        # Test RSI
        rsi = self.market_analysis._calculate_rsi(self.data)
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(self.data))
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())
        
        # Test Bollinger Bands
        bb = self.market_analysis._calculate_bollinger_bands(self.data)
        self.assertIsInstance(bb, dict)
        self.assertTrue(all(key in bb for key in ['middle_band', 'upper_band', 'lower_band']))
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with empty data
        empty_data = pd.DataFrame()
        with self.assertRaises(Exception):
            self.market_analysis.analyze_market(empty_data)
        
        # Test with invalid data
        invalid_data = pd.DataFrame({'close': [np.nan] * 10})
        with self.assertRaises(Exception):
            self.market_analysis.analyze_market(invalid_data)
        
        # Test with missing columns
        missing_data = pd.DataFrame({'open': [1, 2, 3]})
        with self.assertRaises(Exception):
            self.market_analysis.analyze_market(missing_data)

if __name__ == '__main__':
    unittest.main()
