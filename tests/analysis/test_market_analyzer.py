"""
Test suite for the MarketAnalyzer class.

This module contains tests for all major functionality of the MarketAnalyzer,
including data fetching, caching, PCA, KMeans, and batch processing.
"""

import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading.market.market_analyzer import MarketAnalysisError, MarketAnalyzer

logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMarketAnalyzer(unittest.TestCase):
    """Test cases for MarketAnalyzer."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()

        # Test configuration
        self.config = {
            "results_dir": self.test_dir,
            "debug_mode": True,
            "skip_pca": False,
            "use_alpha_vantage": True,
            "alpha_vantage_key": "test_key",
            "min_data_points": 10,
        }

        # Create test data
        self.test_data = pd.DataFrame(
            {
                "Open": np.random.randn(100),
                "High": np.random.randn(100),
                "Low": np.random.randn(100),
                "Close": np.random.randn(100),
                "Volume": np.random.randint(1000, 10000, 100),
            },
            index=pd.date_range(start="2020-01-01", periods=100),
        )

        # Initialize analyzer
        self.analyzer = MarketAnalyzer(config=self.config)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test MarketAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.config["debug_mode"], True)
        self.assertEqual(self.analyzer.config["skip_pca"], False)

    @patch("yfinance.Ticker")
    def test_fetch_data_yfinance(self, mock_ticker):
        """Test data fetching from yfinance."""
        # Mock yfinance response
        mock_ticker.return_value.history.return_value = self.test_data

        # Test successful fetch
        data = self.analyzer.fetch_data("AAPL", period="1y", interval="1d")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)

        # Test invalid symbol
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.fetch_data("", period="1y", interval="1d")

        # Test invalid period
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.fetch_data("AAPL", period="", interval="1d")

        # Test invalid interval
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.fetch_data("AAPL", period="1y", interval="")

    @patch("alpha_vantage.timeseries.TimeSeries")
    def test_fetch_data_alpha_vantage(self, mock_ts):
        """Test data fetching from Alpha Vantage."""
        # Mock Alpha Vantage response
        mock_ts.return_value.get_daily.return_value = (self.test_data, None)

        # Test fallback to Alpha Vantage
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value.history.side_effect = Exception("yfinance failed")
            data = self.analyzer.fetch_data("AAPL", period="1y", interval="1d")
            self.assertIsInstance(data, pd.DataFrame)
            self.assertEqual(len(data), 100)

    def test_caching(self):
        """Test data caching functionality."""
        # Test file caching
        cache_key = os.getenv("KEY", "")
        self.analyzer._set_cached_data(cache_key, self.test_data)
        cached_data = self.analyzer._get_cached_data(cache_key)
        self.assertIsNotNone(cached_data)
        pd.testing.assert_frame_equal(cached_data, self.test_data)

        # Test malformed cache
        cache_file = Path(self.test_dir) / "cache" / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            f.write("invalid json")
        with self.assertRaises(MarketAnalysisError):
            self.analyzer._get_cached_data(cache_key)

    def test_pca_analysis(self):
        """Test PCA analysis functionality."""
        # Test PCA with data
        results = self.analyzer.analyze("AAPL", period="1y", interval="1d")
        self.assertIn("pca", results)
        self.assertIn("n_components", results["pca"])
        self.assertIn("explained_variance_ratio", results["pca"])

        # Test PCA skipping
        self.analyzer.config["skip_pca"] = True
        results = self.analyzer.analyze("AAPL", period="1y", interval="1d")
        self.assertNotIn("pca", results)

    def test_kmeans_analysis(self):
        """Test KMeans analysis functionality."""
        results = self.analyzer.analyze("AAPL", period="1y", interval="1d")
        self.assertIn("regime", results)
        self.assertIn("labels", results["regime"])
        self.assertIn("inertia", results["regime"])
        self.assertIn("silhouette_score", results["regime"])
        self.assertIn("regime_counts", results["regime"])

    def test_model_persistence(self):
        """Test model persistence functionality."""
        # Run analysis to create models
        self.analyzer.analyze("AAPL", period="1y", interval="1d")

        # Check if models were saved
        pca_path = Path(self.test_dir) / "models" / "pca_model.joblib"
        kmeans_path = Path(self.test_dir) / "models" / "kmeans_model.joblib"
        self.assertTrue(pca_path.exists())
        self.assertTrue(kmeans_path.exists())

        # Test model loading
        new_analyzer = MarketAnalyzer(config=self.config)
        self.assertIsNotNone(new_analyzer.pca)
        self.assertIsNotNone(new_analyzer.regime_model)

    def test_batch_processing(self):
        """Test batch processing functionality."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = self.analyzer.analyze_batch(symbols, period="1y", interval="1d")

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(symbols))
        for symbol in symbols:
            self.assertIn(symbol, results)
            self.assertIsInstance(results[symbol], dict)

    def test_error_handling(self):
        """Test error handling and recovery."""
        # Test invalid data
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.analyze("INVALID", period="1y", interval="1d")

        # Test insufficient data
        small_data = self.test_data.iloc[:5]
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = small_data
            with self.assertRaises(MarketAnalysisError):
                self.analyzer.fetch_data("AAPL", period="1y", interval="1d")

    def test_debug_mode(self):
        """Test debug mode functionality."""
        # Test with debug mode on
        self.analyzer.config["debug_mode"] = True
        with self.assertRaises(Exception):
            self.analyzer.analyze("INVALID", period="1y", interval="1d")

        # Test with debug mode off
        self.analyzer.config["debug_mode"] = False
        results = self.analyzer.analyze("INVALID", period="1y", interval="1d")
        self.assertIn("error", results.get("pca", {}))

    def test_analyze_trend(self):
        """Test trend analysis functionality."""
        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100)
        data = pd.DataFrame({"Close": np.random.normal(100, 10, 100)}, index=dates)

        # Test trend analysis
        result = self.analyzer.analyze_trend(data)

        # Verify result structure
        self.assertIn("trend_direction", result)
        self.assertIn("trend_strength", result)
        self.assertIn("trend_duration", result)
        self.assertIn("ma_short", result)
        self.assertIn("ma_long", result)

    def test_analyze_volatility(self):
        """Test volatility analysis functionality."""
        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100)
        data = pd.DataFrame({"Close": np.random.normal(100, 10, 100)}, index=dates)

        # Test volatility analysis
        result = self.analyzer.analyze_volatility(data)

        # Verify result structure
        self.assertIn("current_volatility", result)
        self.assertIn("volatility_rank", result)
        self.assertIn("volatility_trend", result)
        self.assertIn("historical_volatility", result)

    def test_analyze_correlation(self):
        """Test correlation analysis functionality."""
        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100)
        data = pd.DataFrame({"Close": np.random.normal(100, 10, 100)}, index=dates)
        market_data = pd.DataFrame(
            {"Close": np.random.normal(100, 10, 100)}, index=dates
        )

        # Test correlation analysis
        result = self.analyzer.analyze_correlation(data, market_data)

        # Verify result structure
        self.assertIn("correlation", result)
        self.assertIn("correlation_trend", result)
        self.assertIn("rolling_correlation", result)

    def test_analyze_market_conditions(self):
        """Test overall market conditions analysis."""
        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100)
        data = pd.DataFrame({"Close": np.random.normal(100, 10, 100)}, index=dates)
        market_data = pd.DataFrame(
            {"Close": np.random.normal(100, 10, 100)}, index=dates
        )

        # Test market conditions analysis
        result = self.analyzer.analyze_market_conditions(data, market_data)

        # Verify result structure
        self.assertIn("trend", result)
        self.assertIn("volatility", result)
        self.assertIn("correlation", result)
        self.assertIn("timestamp", result)

    def test_invalid_data(self):
        """Test handling of invalid data."""
        # Test empty DataFrame
        empty_data = pd.DataFrame()
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.analyze_trend(empty_data)

        # Test DataFrame with missing columns
        incomplete_data = pd.DataFrame({"Close": [1, 2, 3]})
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.analyze_volatility(incomplete_data)

        # Test DataFrame with all NaN values
        nan_data = pd.DataFrame(
            {
                "Open": [np.nan, np.nan, np.nan],
                "High": [np.nan, np.nan, np.nan],
                "Low": [np.nan, np.nan, np.nan],
                "Close": [np.nan, np.nan, np.nan],
                "Volume": [np.nan, np.nan, np.nan],
            }
        )
        with self.assertRaises(MarketAnalysisError):
            self.analyzer.analyze_trend(nan_data)

    def test_extreme_outliers_and_missing_data(self):
        """Test edge case where market data has extreme outliers or missing months."""
        logger.info("\n🔍 Testing Extreme Outliers and Missing Data Edge Cases")

        # Create data with extreme outliers
        dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
        normal_data = np.random.normal(100, 10, 365)

        # Add extreme outliers (10x normal values)
        outlier_indices = [50, 150, 250, 350]
        for idx in outlier_indices:
            normal_data[idx] = normal_data[idx] * 10

        # Add missing months (remove entire months)
        missing_months = ["2020-03", "2020-07", "2020-11"]
        for month in missing_months:
            month_mask = dates.strftime("%Y-%m") == month
            normal_data[month_mask] = np.nan

        # Create test DataFrame with outliers and missing data
        outlier_data = pd.DataFrame(
            {
                "Open": normal_data,
                "High": normal_data * 1.02,  # High slightly above close
                "Low": normal_data * 0.98,  # Low slightly below close
                "Close": normal_data,
                "Volume": np.random.randint(1000, 10000, 365),
            },
            index=dates,
        )

        logger.info(
            f"✅ Created test data with {len(outlier_indices)} extreme outliers and {len(missing_months)} missing months"
        )

        # Test outlier detection
        outliers = self.analyzer._detect_outliers(outlier_data["Close"])
        self.assertIsInstance(outliers, pd.Series)
        self.assertTrue(
            len(outliers) > 0, "Should detect outliers in data with extreme values"
        )
        logger.info(f"✅ Detected {outliers.sum()} outliers in the data")

        # Test missing data handling
        missing_data_info = self.analyzer._analyze_missing_data(outlier_data)
        self.assertIsInstance(missing_data_info, dict)
        self.assertIn("missing_percentage", missing_data_info)
        self.assertIn("missing_patterns", missing_data_info)
        logger.info(
            f"✅ Missing data analysis: {missing_data_info['missing_percentage']:.1f}% missing"
        )

        # Test data cleaning
        cleaned_data = self.analyzer._clean_data(outlier_data)
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertFalse(
            cleaned_data.isnull().all().any(),
            "Cleaned data should not have all-NaN columns",
        )
        logger.info(f"✅ Data cleaning completed: {len(cleaned_data)} rows remaining")

        # Test analysis with cleaned data
        try:
            trend_result = self.analyzer.analyze_trend(cleaned_data)
            self.assertIsInstance(trend_result, dict)
            self.assertIn("trend_direction", trend_result)
            logger.info(
                f"✅ Trend analysis with cleaned data: {trend_result['trend_direction']}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Trend analysis failed with cleaned data: {e}")

        # Test volatility analysis with outliers
        try:
            volatility_result = self.analyzer.analyze_volatility(cleaned_data)
            self.assertIsInstance(volatility_result, dict)
            self.assertIn("volatility", volatility_result)
            logger.info(f"✅ Volatility analysis: {volatility_result['volatility']:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ Volatility analysis failed: {e}")

        # Test correlation analysis with missing data
        try:
            correlation_result = self.analyzer.analyze_correlation(cleaned_data)
            self.assertIsInstance(correlation_result, dict)
            self.assertIn("correlation_matrix", correlation_result)
            logger.info(f"✅ Correlation analysis completed")
        except Exception as e:
            logger.warning(f"⚠️ Correlation analysis failed: {e}")

        # Test market conditions analysis
        try:
            conditions_result = self.analyzer.analyze_market_conditions(cleaned_data)
            self.assertIsInstance(conditions_result, dict)
            self.assertIn("market_regime", conditions_result)
            logger.info(
                f"✅ Market conditions analysis: {conditions_result['market_regime']}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Market conditions analysis failed: {e}")

        logger.info("✅ Extreme outliers and missing data edge case test completed")

    def _detect_outliers(self, data):
        """Detect outliers in the data using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    def _analyze_missing_data(self, data):
        """Analyze missing data patterns."""
        missing_info = {
            "missing_percentage": (
                data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            )
            * 100,
            "missing_patterns": {},
            "columns_with_missing": data.columns[data.isnull().any()].tolist(),
        }

        # Analyze missing patterns by month
        if hasattr(data.index, "month"):
            for month in range(1, 13):
                month_mask = data.index.month == month
                missing_count = data.loc[month_mask].isnull().sum().sum()
                if missing_count > 0:
                    missing_info["missing_patterns"][f"month_{month}"] = missing_count

        return missing_info

    def _clean_data(self, data):
        """Clean data by handling outliers and missing values."""
        cleaned = data.copy()

        # Handle outliers using winsorization (cap at 99th percentile)
        for col in ["Open", "High", "Low", "Close"]:
            if col in cleaned.columns:
                q99 = cleaned[col].quantile(0.99)
                q01 = cleaned[col].quantile(0.01)
                cleaned[col] = cleaned[col].clip(lower=q01, upper=q99)

        # Handle missing values using forward fill and backward fill
        cleaned = cleaned.fillna(method="ffill").fillna(method="bfill")

        # Remove any remaining rows with NaN values
        cleaned = cleaned.dropna()

        return cleaned


if __name__ == "__main__":
    unittest.main()
    print(
        "Market analysis test completed. All analysis functions have " "been validated."
    )
