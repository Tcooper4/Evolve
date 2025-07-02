"""
Test suite for the MarketAnalyzer class.

This module contains tests for all major functionality of the MarketAnalyzer,
including data fetching, caching, PCA, KMeans, and batch processing.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.market.market_analyzer import MarketAnalyzer, MarketAnalysisError

class TestMarketAnalyzer(unittest.TestCase):
    """Test cases for MarketAnalyzer."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Test configuration
        self.config = {
            'results_dir': self.test_dir,
            'debug_mode': True,
            'skip_pca': False,
            'use_alpha_vantage': True,
            'alpha_vantage_key': 'test_key',
            'min_data_points': 10
        }
        
        # Create test data
        self.test_data = pd.DataFrame({
            'Open': np.random.randn(100),
            'High': np.random.randn(100),
            'Low': np.random.randn(100),
            'Close': np.random.randn(100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=pd.date_range(start='2020-01-01', periods=100))
        
        # Initialize analyzer
        self.analyzer = MarketAnalyzer(config=self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        
    def test_initialization(self):
        """Test MarketAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.config['debug_mode'], True)
        self.assertEqual(self.analyzer.config['skip_pca'], False)
        
    @patch('yfinance.Ticker')
    def test_fetch_data_yfinance(self, mock_ticker):
        """Test data fetching from yfinance."""
        # Mock yfinance response
        mock_ticker.return_value.history.return_value = self.test_data
        
        # Test successful fetch
        data = self.analyzer.fetch_data('AAPL', period='1y', interval='1d')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        
        # Test invalid symbol
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.fetch_data('', period='1y', interval='1d')
            
        # Test invalid period
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.fetch_data('AAPL', period='', interval='1d')
            
        # Test invalid interval
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.fetch_data('AAPL', period='1y', interval='')
            
    @patch('alpha_vantage.timeseries.TimeSeries')
    def test_fetch_data_alpha_vantage(self, mock_ts):
        """Test data fetching from Alpha Vantage."""
        # Mock Alpha Vantage response
        mock_ts.return_value.get_daily.return_value = (self.test_data, None)
        
        # Test fallback to Alpha Vantage
        with patch('yfinance.Ticker') as mock_yf:
            mock_yf.return_value.history.side_effect = Exception("yfinance failed")
            data = self.analyzer.fetch_data('AAPL', period='1y', interval='1d')
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 100)
            
    def test_caching(self):
        """Test data caching functionality."""
        # Test file caching
        cache_key = "test_cache"
        self.analyzer._set_cached_data(cache_key, self.test_data)
        cached_data = self.analyzer._get_cached_data(cache_key)
        self.assertIsNotNone(cached_data)
        pd.testing.assert_frame_equal(cached_data, self.test_data)
        
        # Test malformed cache
        cache_file = Path(self.test_dir) / 'cache' / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            f.write("invalid json")
        with self.assertRaises(MarketAnalysisError):
            self.analyzer._get_cached_data(cache_key)
            
    def test_pca_analysis(self):
        """Test PCA analysis functionality."""
        # Test PCA with data
        results = self.analyzer.analyze('AAPL', period='1y', interval='1d')
        self.assertIn('pca', results)
        self.assertIn('n_components', results['pca'])
        self.assertIn('explained_variance_ratio', results['pca'])
        
        # Test PCA skipping
        self.analyzer.config['skip_pca'] = True
        results = self.analyzer.analyze('AAPL', period='1y', interval='1d')
        self.assertNotIn('pca', results)
        
    def test_kmeans_analysis(self):
        """Test KMeans analysis functionality."""
        results = self.analyzer.analyze('AAPL', period='1y', interval='1d')
        self.assertIn('regime', results)
        self.assertIn('labels', results['regime'])
        self.assertIn('inertia', results['regime'])
        self.assertIn('silhouette_score', results['regime'])
        self.assertIn('regime_counts', results['regime'])
        
    def test_model_persistence(self):
        """Test model persistence functionality."""
        # Run analysis to create models
        self.analyzer.analyze('AAPL', period='1y', interval='1d')
        
        # Check if models were saved
        pca_path = Path(self.test_dir) / 'models' / 'pca_model.joblib'
        kmeans_path = Path(self.test_dir) / 'models' / 'kmeans_model.joblib'
        self.assertTrue(pca_path.exists())
        self.assertTrue(kmeans_path.exists())
        
        # Test model loading
        new_analyzer = MarketAnalyzer(config=self.config)
        self.assertIsNotNone(new_analyzer.pca)
        self.assertIsNotNone(new_analyzer.regime_model)
        
    def test_batch_processing(self):
        """Test batch processing functionality."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        results = self.analyzer.analyze_batch(symbols, period='1y', interval='1d')
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, results)
            self.assertIsInstance(results[symbol], dict)
            
    def test_error_handling(self):
        """Test error handling and recovery."""
        # Test invalid data
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.analyze('INVALID', period='1y', interval='1d')
            
        # Test insufficient data
        small_data = self.test_data.iloc[:5]
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = small_data
            with self.assertRaises(MarketAnalysisError):
                self.analyzer.fetch_data('AAPL', period='1y', interval='1d')
                
    def test_debug_mode(self):
        """Test debug mode functionality."""
        # Test with debug mode on
        self.analyzer.config['debug_mode'] = True
        with self.assertRaises(Exception):
            self.analyzer.analyze('INVALID', period='1y', interval='1d')
            
        # Test with debug mode off
        self.analyzer.config['debug_mode'] = False
        results = self.analyzer.analyze('INVALID', period='1y', interval='1d')
        self.assertIn('error', results.get('pca', {}))
        
    def test_analyze_trend(self):
        """Test trend analysis functionality."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100)
        data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100)
        }, index=dates)
        
        # Test trend analysis
        result = self.analyzer.analyze_trend(data)
        
        # Verify result structure
        self.assertIn('trend_direction', result)
        self.assertIn('trend_strength', result)
        self.assertIn('trend_duration', result)
        self.assertIn('ma_short', result)
        self.assertIn('ma_long', result)
        
    def test_analyze_volatility(self):
        """Test volatility analysis functionality."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100)
        data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100)
        }, index=dates)
        
        # Test volatility analysis
        result = self.analyzer.analyze_volatility(data)
        
        # Verify result structure
        self.assertIn('current_volatility', result)
        self.assertIn('volatility_rank', result)
        self.assertIn('volatility_trend', result)
        self.assertIn('historical_volatility', result)
        
    def test_analyze_correlation(self):
        """Test correlation analysis functionality."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100)
        data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100)
        }, index=dates)
        market_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100)
        }, index=dates)
        
        # Test correlation analysis
        result = self.analyzer.analyze_correlation(data, market_data)
        
        # Verify result structure
        self.assertIn('correlation', result)
        self.assertIn('correlation_trend', result)
        self.assertIn('rolling_correlation', result)
        
    def test_analyze_market_conditions(self):
        """Test overall market conditions analysis."""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100)
        data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100)
        }, index=dates)
        market_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100)
        }, index=dates)
        
        # Test market conditions analysis
        result = self.analyzer.analyze_market_conditions(data, market_data)
        
        # Verify result structure
        self.assertIn('trend', result)
        self.assertIn('volatility', result)
        self.assertIn('correlation', result)
        self.assertIn('timestamp', result)
        
    def test_invalid_data(self):
        """Test handling of invalid data."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(KeyError):
            self.analyzer.analyze_trend(empty_df)
            
        # Test with missing required column
        invalid_df = pd.DataFrame({'Open': [1, 2, 3]})
        with self.assertRaises(KeyError):
            self.analyzer.analyze_trend(invalid_df)
            
        # Test with insufficient data
        small_df = pd.DataFrame({'Close': [1]})
        with self.assertRaises(ValueError):
            self.analyzer.analyze_trend(small_df)

if __name__ == '__main__':
    unittest.main() 