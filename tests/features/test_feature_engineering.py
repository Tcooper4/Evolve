import unittest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureGenerator, FeatureConfig, FeatureVerificationError

class TestFeatureEngineering(unittest.TestCase):
    """Test suite for feature engineering module"""
    
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
        
        # Initialize feature generator
        self.feature_generator = FeatureGenerator()
        
        # Register test features
        self.feature_generator.register_feature(FeatureConfig(
            name='returns',
            description='Price returns',
            category='price',
            dependencies=['close'],
            parameters={'window': 20},
            validation_rules={'not_null': lambda x: not x.isnull().any()},
            is_required=True
        ))
    
    def test_returns_calculation(self):
        """Test returns calculation"""
        returns = self.feature_generator.generate_feature(self.data, 'returns')
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(self.data))
        self.assertTrue(not returns.isnull().all())
    
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        volatility = self.feature_generator._calculate_volatility(self.data)
        self.assertIsInstance(volatility, pd.Series)
        self.assertEqual(len(volatility), len(self.data))
        self.assertTrue(not volatility.isnull().all())
        self.assertTrue((volatility >= 0).all())
    
    def test_momentum_calculation(self):
        """Test momentum calculation"""
        momentum = self.feature_generator._calculate_momentum(self.data)
        self.assertIsInstance(momentum, pd.Series)
        self.assertEqual(len(momentum), len(self.data))
        self.assertTrue(not momentum.isnull().all())
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.feature_generator._calculate_rsi(self.data)
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(self.data))
        self.assertTrue(not rsi.isnull().all())
        self.assertTrue((rsi >= 0).all() and (rsi <= 100).all())
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd = self.feature_generator._calculate_macd(self.data)
        self.assertIsInstance(macd, pd.Series)
        self.assertEqual(len(macd), len(self.data))
        self.assertTrue(not macd.isnull().all())
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        bb = self.feature_generator._calculate_bollinger_bands(self.data)
        self.assertIsInstance(bb, dict)
        self.assertTrue(all(key in bb for key in ['middle_band', 'upper_band', 'lower_band']))
        self.assertTrue(all(len(bb[key]) == len(self.data) for key in bb))
        self.assertTrue(all(not bb[key].isnull().all() for key in bb))
    
    def test_volume_profile(self):
        """Test volume profile calculation"""
        vp = self.feature_generator._calculate_volume_profile(self.data)
        self.assertIsInstance(vp, pd.Series)
        self.assertEqual(len(vp), len(self.data))
        self.assertTrue(not vp.isnull().all())
    
    def test_support_resistance(self):
        """Test support/resistance calculation"""
        sr = self.feature_generator._calculate_support_resistance(self.data)
        self.assertIsInstance(sr, pd.Series)
        self.assertEqual(len(sr), len(self.data))
        self.assertTrue(not sr.isnull().all())
        self.assertTrue((sr >= 0).all() and (sr <= 1).all())
    
    def test_trend_strength(self):
        """Test trend strength calculation"""
        ts = self.feature_generator._calculate_trend_strength(self.data)
        self.assertIsInstance(ts, pd.Series)
        self.assertEqual(len(ts), len(self.data))
        self.assertTrue(not ts.isnull().all())
    
    def test_market_regime(self):
        """Test market regime calculation"""
        mr = self.feature_generator._calculate_market_regime(self.data)
        self.assertIsInstance(mr, pd.Series)
        self.assertEqual(len(mr), len(self.data))
        self.assertTrue(not mr.isnull().all())
        self.assertTrue(mr.isin([-1, 0, 1]).all())
    
    def test_volatility_regime(self):
        """Test volatility regime calculation"""
        vr = self.feature_generator._calculate_volatility_regime(self.data)
        self.assertIsInstance(vr, pd.Series)
        self.assertEqual(len(vr), len(self.data))
        self.assertTrue(not vr.isnull().all())
        self.assertTrue(vr.isin([-1, 0, 1]).all())
    
    def test_correlation(self):
        """Test correlation calculation"""
        market_returns = self.data['close'].pct_change()
        corr = self.feature_generator._calculate_correlation(self.data, market_returns=market_returns)
        self.assertIsInstance(corr, pd.Series)
        self.assertEqual(len(corr), len(self.data))
        self.assertTrue(not corr.isnull().all())
        self.assertTrue((corr >= -1).all() and (corr <= 1).all())
    
    def test_beta(self):
        """Test beta calculation"""
        market_returns = self.data['close'].pct_change()
        beta = self.feature_generator._calculate_beta(self.data, market_returns=market_returns)
        self.assertIsInstance(beta, pd.Series)
        self.assertEqual(len(beta), len(self.data))
        self.assertTrue(not beta.isnull().all())
    
    def test_alpha(self):
        """Test alpha calculation"""
        market_returns = self.data['close'].pct_change()
        alpha = self.feature_generator._calculate_alpha(self.data, market_returns=market_returns)
        self.assertIsInstance(alpha, pd.Series)
        self.assertEqual(len(alpha), len(self.data))
        self.assertTrue(not alpha.isnull().all())
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        sharpe = self.feature_generator._calculate_sharpe_ratio(self.data)
        self.assertIsInstance(sharpe, pd.Series)
        self.assertEqual(len(sharpe), len(self.data))
        self.assertTrue(not sharpe.isnull().all())
    
    def test_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        sortino = self.feature_generator._calculate_sortino_ratio(self.data)
        self.assertIsInstance(sortino, pd.Series)
        self.assertEqual(len(sortino), len(self.data))
        self.assertTrue(not sortino.isnull().all())
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        max_dd = self.feature_generator._calculate_max_drawdown(self.data)
        self.assertIsInstance(max_dd, pd.Series)
        self.assertEqual(len(max_dd), len(self.data))
        self.assertTrue(not max_dd.isnull().all())
        self.assertTrue((max_dd <= 0).all())
    
    def test_var(self):
        """Test Value at Risk calculation"""
        var = self.feature_generator._calculate_var(self.data)
        self.assertIsInstance(var, pd.Series)
        self.assertEqual(len(var), len(self.data))
        self.assertTrue(not var.isnull().all())
    
    def test_cvar(self):
        """Test Conditional Value at Risk calculation"""
        cvar = self.feature_generator._calculate_cvar(self.data)
        self.assertIsInstance(cvar, pd.Series)
        self.assertEqual(len(cvar), len(self.data))
        self.assertTrue(not cvar.isnull().all())
    
    def test_position_size(self):
        """Test position size calculation"""
        pos_size = self.feature_generator._calculate_position_size(self.data)
        self.assertIsInstance(pos_size, pd.Series)
        self.assertEqual(len(pos_size), len(self.data))
        self.assertTrue(not pos_size.isnull().all())
        self.assertTrue((pos_size > 0).all())
    
    def test_risk_parity(self):
        """Test risk parity calculation"""
        rp = self.feature_generator._calculate_risk_parity(self.data)
        self.assertIsInstance(rp, pd.Series)
        self.assertEqual(len(rp), len(self.data))
        self.assertTrue(not rp.isnull().all())
        self.assertTrue((rp > 0).all())
    
    def test_mean_variance(self):
        """Test mean-variance optimization calculation"""
        mv = self.feature_generator._calculate_mean_variance(self.data)
        self.assertIsInstance(mv, pd.Series)
        self.assertEqual(len(mv), len(self.data))
        self.assertTrue(not mv.isnull().all())
    
    def test_black_litterman(self):
        """Test Black-Litterman model calculation"""
        bl = self.feature_generator._calculate_black_litterman(self.data)
        self.assertIsInstance(bl, pd.Series)
        self.assertEqual(len(bl), len(self.data))
        self.assertTrue(not bl.isnull().all())
    
    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        rm = self.feature_generator._calculate_risk_metrics(self.data)
        self.assertIsInstance(rm, pd.Series)
        self.assertEqual(len(rm), len(self.data))
        self.assertTrue(not rm.isnull().all())
        self.assertTrue((rm >= 0).all())
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        pm = self.feature_generator._calculate_performance_metrics(self.data)
        self.assertIsInstance(pm, pd.Series)
        self.assertEqual(len(pm), len(self.data))
        self.assertTrue(not pm.isnull().all())
    
    def test_execution_metrics(self):
        """Test execution metrics calculation"""
        em = self.feature_generator._calculate_execution_metrics(self.data)
        self.assertIsInstance(em, pd.Series)
        self.assertEqual(len(em), len(self.data))
        self.assertTrue(not em.isnull().all())
        self.assertTrue((em >= 0).all())
    
    def test_signal_metrics(self):
        """Test signal metrics calculation"""
        sm = self.feature_generator._calculate_signal_metrics(self.data)
        self.assertIsInstance(sm, pd.Series)
        self.assertEqual(len(sm), len(self.data))
        self.assertTrue(not sm.isnull().all())
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        pm = self.feature_generator._calculate_portfolio_metrics(self.data)
        self.assertIsInstance(pm, pd.Series)
        self.assertEqual(len(pm), len(self.data))
        self.assertTrue(not pm.isnull().all())
    
    def test_regime_metrics(self):
        """Test regime metrics calculation"""
        rm = self.feature_generator._calculate_regime_metrics(self.data)
        self.assertIsInstance(rm, pd.Series)
        self.assertEqual(len(rm), len(self.data))
        self.assertTrue(not rm.isnull().all())
    
    def test_feature_verification(self):
        """Test feature verification"""
        # Test with invalid data
        invalid_data = pd.DataFrame({'close': [np.nan] * 10})
        with self.assertRaises(FeatureVerificationError):
            self.feature_generator.generate_feature(invalid_data, 'returns')
    
    def test_feature_caching(self):
        """Test feature caching"""
        # Generate feature twice
        returns1 = self.feature_generator.generate_feature(self.data, 'returns')
        returns2 = self.feature_generator.generate_feature(self.data, 'returns')
        
        # Check if cached
        self.assertTrue('returns' in self.feature_generator.feature_cache)
        pd.testing.assert_series_equal(returns1, returns2)
    
    def test_custom_feature(self):
        """Test custom feature generation"""
        def custom_func(data):
            return data['close'] * 2
        
        config = FeatureConfig(
            name='custom',
            description='Custom feature',
            category='custom',
            dependencies=['close'],
            parameters={},
            validation_rules={'not_null': lambda x: not x.isnull().any()},
            is_custom=True
        )
        
        self.feature_generator.register_feature(config)
        custom_feature = self.feature_generator.generate_feature(
            self.data, 'custom', custom_func=custom_func
        )
        
        self.assertIsInstance(custom_feature, pd.Series)
        self.assertEqual(len(custom_feature), len(self.data))
        self.assertTrue(not custom_feature.isnull().all())
        self.assertTrue((custom_feature == self.data['close'] * 2).all())

if __name__ == '__main__':
    unittest.main()
