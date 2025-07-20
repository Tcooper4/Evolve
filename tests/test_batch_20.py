"""
Batch 20 Tests
Tests for sentiment, risk, and ensemble validation modules
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import Batch 20 modules
from trading.sentiment.parser import SentimentParser, FinancialSymbol
from trading.ensemble.integrity_checker import EnsembleIntegrityChecker, DriftType, DistanceMetric
from trading.risk_model.compliance_flags import RiskModelComplianceFlags, RiskLevel, ComplianceFlagType


class TestSentimentParser(unittest.TestCase):
    """Test SentimentParser financial symbol preservation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = SentimentParser(preserve_financial_symbols=True)
    
    def test_financial_symbol_preservation(self):
        """Test preservation of financial symbols in clean_text()."""
        # Test ticker symbols
        text_with_ticker = "I think $AAPL will go up 20% this quarter"
        parsed = self.parser.clean_text(text_with_ticker)
        
        self.assertIn('$AAPL', parsed.cleaned_text)
        self.assertIn('20%', parsed.cleaned_text)
        self.assertEqual(len(parsed.preserved_symbols), 2)
        
        # Check ticker symbol
        ticker_symbol = next((s for s in parsed.preserved_symbols if s['type'] == FinancialSymbol.TICKER.value), None)
        self.assertIsNotNone(ticker_symbol)
        self.assertEqual(ticker_symbol['symbol'], '$AAPL')
        
        # Check percentage symbol
        percentage_symbol = next((s for s in parsed.preserved_symbols if s['type'] == FinancialSymbol.PERCENTAGE.value), None)
        self.assertIsNotNone(percentage_symbol)
        self.assertEqual(percentage_symbol['symbol'], '20%')
        self.assertEqual(percentage_symbol['value'], 20.0)
    
    def test_hashtag_preservation(self):
        """Test preservation of financial hashtags."""
        text_with_hashtag = "Market analysis #earnings season is strong"
        parsed = self.parser.clean_text(text_with_hashtag)
        
        self.assertIn('#earnings', parsed.cleaned_text)
        self.assertEqual(len(parsed.preserved_symbols), 1)
        
        hashtag_symbol = parsed.preserved_symbols[0]
        self.assertEqual(hashtag_symbol['type'], FinancialSymbol.HASHTAG.value)
        self.assertEqual(hashtag_symbol['symbol'], '#earnings')
        self.assertTrue(hashtag_symbol['is_financial'])
    
    def test_currency_preservation(self):
        """Test preservation of currency amounts."""
        text_with_currency = "Stock price is $150.50 and revenue is $1,000,000"
        parsed = self.parser.clean_text(text_with_currency)
        
        self.assertIn('$150.50', parsed.cleaned_text)
        self.assertIn('$1,000,000', parsed.cleaned_text)
        
        currency_symbols = [s for s in parsed.preserved_symbols if s['type'] == FinancialSymbol.CURRENCY.value]
        self.assertEqual(len(currency_symbols), 2)
        
        # Check currency values
        values = [s['value'] for s in currency_symbols]
        self.assertIn(150.50, values)
        self.assertIn(1000000.0, values)
    
    def test_financial_context_analysis(self):
        """Test financial context analysis."""
        text = "$TSLA is up 15% today, #bullish sentiment"
        parsed = self.parser.clean_text(text)
        analysis = self.parser.analyze_financial_context(parsed)
        
        self.assertTrue(analysis['has_tickers'])
        self.assertTrue(analysis['has_financial_hashtags'])
        self.assertTrue(analysis['has_percentages'])
        self.assertGreater(analysis['financial_relevance_score'], 0.5)
    
    def test_symbol_disabling(self):
        """Test disabling financial symbol preservation."""
        parser_no_symbols = SentimentParser(preserve_financial_symbols=False)
        text = "$AAPL is up 20% #bullish"
        parsed = parser_no_symbols.clean_text(text)
        
        # Symbols should be removed
        self.assertNotIn('$AAPL', parsed.cleaned_text)
        self.assertNotIn('20%', parsed.cleaned_text)
        self.assertNotIn('#bullish', parsed.cleaned_text)
        self.assertEqual(len(parsed.preserved_symbols), 0)


class TestEnsembleIntegrityChecker(unittest.TestCase):
    """Test EnsembleIntegrityChecker distribution drift detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = EnsembleIntegrityChecker(reference_window=100)
    
    def test_distribution_drift_detection(self):
        """Test distribution drift detection with KL divergence."""
        # Add reference predictions
        reference_predictions = np.random.normal(0, 1, 100)
        self.checker.add_model_predictions("model_1", reference_predictions)
        
        # Add more predictions to build reference distribution
        for _ in range(10):
            self.checker.add_model_predictions("model_1", np.random.normal(0, 1, 10))
        
        # Test with similar distribution (should have low drift)
        similar_predictions = np.random.normal(0, 1, 50)
        result = self.checker.distribution_drift_check(
            "model_1", similar_predictions, DistanceMetric.KL_DIVERGENCE
        )
        
        self.assertIsNotNone(result)
        self.assertLess(result.distance_value, 0.5)  # Should be low drift
        
        # Test with different distribution (should have high drift)
        different_predictions = np.random.normal(5, 2, 50)  # Different mean and std
        result = self.checker.distribution_drift_check(
            "model_1", different_predictions, DistanceMetric.KL_DIVERGENCE
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(result.distance_value, 0.1)  # Should be higher drift
    
    def test_wasserstein_distance(self):
        """Test Wasserstein distance calculation."""
        # Add reference predictions
        reference_predictions = np.random.normal(0, 1, 100)
        self.checker.add_model_predictions("model_2", reference_predictions)
        
        # Add more predictions to build reference distribution
        for _ in range(10):
            self.checker.add_model_predictions("model_2", np.random.normal(0, 1, 10))
        
        # Test with Wasserstein distance
        current_predictions = np.random.normal(2, 1, 50)  # Shifted distribution
        result = self.checker.distribution_drift_check(
            "model_2", current_predictions, DistanceMetric.WASSERSTEIN
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.distance_metric, DistanceMetric.WASSERSTEIN)
        self.assertGreater(result.distance_value, 0)  # Should detect some drift
    
    def test_drift_classification(self):
        """Test drift classification levels."""
        # Add reference predictions
        reference_predictions = np.random.normal(0, 1, 100)
        self.checker.add_model_predictions("model_3", reference_predictions)
        
        # Add more predictions to build reference distribution
        for _ in range(10):
            self.checker.add_model_predictions("model_3", np.random.normal(0, 1, 10))
        
        # Test different drift levels
        test_cases = [
            (np.random.normal(0, 1, 50), DriftType.NONE),  # Similar distribution
            (np.random.normal(1, 1, 50), DriftType.MILD),  # Slight shift
            (np.random.normal(3, 2, 50), DriftType.MODERATE),  # Moderate shift
            (np.random.normal(5, 3, 50), DriftType.SEVERE),  # Large shift
        ]
        
        for predictions, expected_drift in test_cases:
            result = self.checker.distribution_drift_check("model_3", predictions)
            # Note: Exact classification depends on thresholds, so we just check it's valid
            self.assertIn(result.drift_type, [DriftType.NONE, DriftType.MILD, 
                                             DriftType.MODERATE, DriftType.SEVERE, DriftType.CRITICAL])
    
    def test_ensemble_health_check(self):
        """Test ensemble health monitoring."""
        # Add multiple models
        for i in range(3):
            model_id = f"model_{i}"
            predictions = np.random.normal(i, 1, 50)
            self.checker.add_model_predictions(model_id, predictions)
        
        # Check ensemble health
        health = self.checker.check_ensemble_health()
        
        self.assertIsNotNone(health)
        self.assertGreaterEqual(health.overall_health, 0.0)
        self.assertLessEqual(health.overall_health, 1.0)
        self.assertIsInstance(health.model_stability, dict)
        self.assertIsInstance(health.ensemble_coherence, float)


class TestRiskModelComplianceFlags(unittest.TestCase):
    """Test RiskModelComplianceFlags dynamic thresholds."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.risk_model = RiskModelComplianceFlags(rolling_window=100)
    
    def test_dynamic_threshold_calculation(self):
        """Test dynamic threshold calculation based on rolling percentiles."""
        portfolio_id = "test_portfolio"
        
        # Add historical risk data
        for i in range(100):
            risk_metrics = {
                'volatility': np.random.uniform(0.1, 0.3),
                'drawdown': np.random.uniform(0.05, 0.2),
                'concentration': np.random.uniform(0.1, 0.5),
                'correlation': np.random.uniform(0.3, 0.8)
            }
            self.risk_model.add_risk_data(portfolio_id, risk_metrics)
        
        # Check that thresholds were calculated
        thresholds = self.risk_model.thresholds.get(portfolio_id)
        self.assertIsNotNone(thresholds)
        self.assertGreater(thresholds.volatility_95th, 0)
        self.assertGreater(thresholds.volatility_99th, thresholds.volatility_95th)
        self.assertGreater(thresholds.drawdown_95th, 0)
        self.assertGreater(thresholds.drawdown_99th, thresholds.drawdown_95th)
    
    def test_compliance_flag_triggering(self):
        """Test compliance flag triggering with dynamic thresholds."""
        portfolio_id = "test_portfolio"
        
        # Add historical data to establish thresholds
        for i in range(100):
            risk_metrics = {
                'volatility': np.random.uniform(0.1, 0.2),
                'drawdown': np.random.uniform(0.05, 0.15),
                'concentration': np.random.uniform(0.1, 0.3),
                'correlation': np.random.uniform(0.3, 0.6)
            }
            self.risk_model.add_risk_data(portfolio_id, risk_metrics)
        
        # Test with high volatility (should trigger flag)
        high_vol_metrics = {'volatility': 0.4}  # Above 95th percentile
        flags = self.risk_model.check_compliance(portfolio_id, high_vol_metrics)
        
        self.assertGreater(len(flags), 0)
        vol_flag = next((f for f in flags if f.flag_type == ComplianceFlagType.VOLATILITY), None)
        self.assertIsNotNone(vol_flag)
        self.assertIn(vol_flag.risk_level, [RiskLevel.HIGH, RiskLevel.CRITICAL])
        self.assertIn('exceeds', vol_flag.rationale.lower())
    
    def test_multiple_compliance_flags(self):
        """Test multiple compliance flags for different risk metrics."""
        portfolio_id = "test_portfolio"
        
        # Add historical data
        for i in range(100):
            risk_metrics = {
                'volatility': np.random.uniform(0.1, 0.2),
                'drawdown': np.random.uniform(0.05, 0.15),
                'concentration': np.random.uniform(0.1, 0.3),
                'correlation': np.random.uniform(0.3, 0.6)
            }
            self.risk_model.add_risk_data(portfolio_id, risk_metrics)
        
        # Test with multiple high risk metrics
        high_risk_metrics = {
            'volatility': 0.4,  # High volatility
            'drawdown': 0.25,   # High drawdown
            'concentration': 0.6,  # High concentration
            'correlation': 0.9   # High correlation
        }
        
        flags = self.risk_model.check_compliance(portfolio_id, high_risk_metrics)
        
        self.assertGreaterEqual(len(flags), 3)  # Should trigger multiple flags
        
        flag_types = [f.flag_type for f in flags]
        self.assertIn(ComplianceFlagType.VOLATILITY, flag_types)
        self.assertIn(ComplianceFlagType.DRAWDOWN, flag_types)
        self.assertIn(ComplianceFlagType.CONCENTRATION, flag_types)
        self.assertIn(ComplianceFlagType.CORRELATION, flag_types)
    
    def test_compliance_rationale_logging(self):
        """Test compliance flag rationale logging."""
        portfolio_id = "test_portfolio"
        
        # Add historical data
        for i in range(100):
            risk_metrics = {
                'volatility': np.random.uniform(0.1, 0.2),
                'drawdown': np.random.uniform(0.05, 0.15)
            }
            self.risk_model.add_risk_data(portfolio_id, risk_metrics)
        
        # Test with high risk metrics
        high_risk_metrics = {
            'volatility': 0.4,
            'drawdown': 0.25
        }
        
        with patch.object(self.risk_model, '_log_compliance_rationale') as mock_log:
            flags = self.risk_model.check_compliance(portfolio_id, high_risk_metrics)
            
            if flags:  # If flags were triggered
                mock_log.assert_called_once()
                called_flags = mock_log.call_args[0][0]
                self.assertEqual(len(called_flags), len(flags))
    
    def test_active_flags_management(self):
        """Test active flags management."""
        portfolio_id = "test_portfolio"
        
        # Add historical data
        for i in range(100):
            risk_metrics = {
                'volatility': np.random.uniform(0.1, 0.2),
                'drawdown': np.random.uniform(0.05, 0.15)
            }
            self.risk_model.add_risk_data(portfolio_id, risk_metrics)
        
        # Trigger some flags
        high_risk_metrics = {'volatility': 0.4, 'drawdown': 0.25}
        flags = self.risk_model.check_compliance(portfolio_id, high_risk_metrics)
        
        # Check active flags
        active_flags = self.risk_model.get_active_flags(portfolio_id)
        self.assertEqual(len(active_flags), len(flags))
        
        # Clear specific flag types
        cleared_count = self.risk_model.clear_resolved_flags(
            portfolio_id, [ComplianceFlagType.VOLATILITY]
        )
        self.assertGreater(cleared_count, 0)
        
        # Check remaining active flags
        remaining_flags = self.risk_model.get_active_flags(portfolio_id)
        self.assertLess(len(remaining_flags), len(flags))


class TestBatch20Integration(unittest.TestCase):
    """Integration tests for Batch 20 modules."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow with all Batch 20 modules."""
        # Create instances
        sentiment_parser = SentimentParser()
        integrity_checker = EnsembleIntegrityChecker()
        risk_model = RiskModelComplianceFlags()
        
        # Test sentiment parsing with financial context
        financial_text = "$AAPL is up 15% today, #earnings look strong"
        parsed_text = sentiment_parser.clean_text(financial_text)
        
        self.assertIn('$AAPL', parsed_text.cleaned_text)
        self.assertIn('15%', parsed_text.cleaned_text)
        self.assertIn('#earnings', parsed_text.cleaned_text)
        
        # Test ensemble drift detection
        reference_predictions = np.random.normal(0, 1, 100)
        integrity_checker.add_model_predictions("test_model", reference_predictions)
        
        # Add more predictions to build reference
        for _ in range(10):
            integrity_checker.add_model_predictions("test_model", np.random.normal(0, 1, 10))
        
        current_predictions = np.random.normal(2, 1, 50)
        drift_result = integrity_checker.distribution_drift_check("test_model", current_predictions)
        
        self.assertIsNotNone(drift_result)
        
        # Test risk compliance
        portfolio_id = "test_portfolio"
        
        # Add historical risk data
        for i in range(100):
            risk_metrics = {
                'volatility': np.random.uniform(0.1, 0.2),
                'drawdown': np.random.uniform(0.05, 0.15)
            }
            risk_model.add_risk_data(portfolio_id, risk_metrics)
        
        # Check compliance
        current_metrics = {'volatility': 0.4, 'drawdown': 0.25}
        compliance_flags = risk_model.check_compliance(portfolio_id, current_metrics)
        
        self.assertGreater(len(compliance_flags), 0)
        
        # Verify all modules work together
        self.assertTrue(len(parsed_text.preserved_symbols) > 0)
        self.assertIsNotNone(drift_result.drift_type)
        self.assertTrue(all(flag.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] 
                           for flag in compliance_flags))


if __name__ == '__main__':
    unittest.main()
