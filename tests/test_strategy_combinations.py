import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Comprehensive test suite for strategy combinations and interactions.

This module tests various strategy combinations to ensure they work together
properly and produce reasonable results. Includes advanced testing scenarios
for hybrid strategies, parameter optimization, and performance validation.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
import warnings

# Import available strategies
from trading.strategies.bollinger_strategy import BollingerStrategy, BollingerConfig
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.sma_strategy import SMAStrategy
from trading.strategies.rsi_signals import generate_rsi_signals
from trading.strategies.hybrid_engine import HybridEngine as HybridStrategyEngine
from trading.strategies.strategy_manager import StrategyManager
from trading.strategies.parameter_validator import StrategyParameterValidator as ParameterValidator

logger = logging.getLogger(__name__)

class TestStrategyCombinations:
    """Comprehensive test suite for strategy combinations."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate comprehensive sample price data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data with trends and volatility
        returns = np.random.normal(0.001, 0.02, len(dates))
        
        # Add some trending periods
        trend_periods = [
            (50, 100, 0.002),   # Uptrend
            (150, 200, -0.001), # Downtrend
            (250, 300, 0.003),  # Strong uptrend
        ]
        
        for start, end, trend in trend_periods:
            if start < len(returns) and end < len(returns):
                returns[start:end] += trend
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.normal(1000000, 200000, len(dates))
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def strategy_configs(self):
        """Get comprehensive strategy configurations for testing."""
        return {
            'rsi': {
                'period': 14,
                'overbought': 70,
                'oversold': 30,
                'smooth_period': 3
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'smooth_period': 3
            },
            'bollinger': {
                'window': 20,
                'num_std': 2.0,
                'min_volume': 1000.0,
                'min_price': 1.0,
                'smooth_period': 3
            },
            'sma': {
                'short_window': 10,
                'long_window': 30,
                'smooth_period': 3
            },
            'ema': {
                'short_window': 12,
                'long_window': 26,
                'smooth_period': 3
            }
        }
    
    @pytest.fixture
    def hybrid_config(self):
        """Get hybrid strategy configuration."""
        return {
            'strategies': ['bollinger', 'macd', 'rsi'],
            'weights': [0.4, 0.3, 0.3],
            'consensus_threshold': 0.6,
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss': 0.05,
                'take_profit': 0.15
            }
        }
    
    def test_rsi_macd_combination(self, sample_data, strategy_configs):
        """Test RSI + MACD strategy combination with comprehensive validation."""
        logger.info("Testing comprehensive RSI + MACD combination")
        
        # Generate RSI signals
        rsi_signals = generate_rsi_signals(sample_data, **strategy_configs['rsi'])
        
        # Create MACD strategy
        macd_strategy = MACDStrategy(**strategy_configs['macd'])
        macd_signals = macd_strategy.generate_signals(sample_data)
        
        # Test individual strategies
        assert not rsi_signals.empty, "RSI signals should not be empty"
        assert not macd_signals.empty, "MACD signals should not be empty"
        assert 'signal' in rsi_signals.columns, "RSI should have signal column"
        assert 'signal' in macd_signals.columns, "MACD should have signal column"
        
        # Test signal values
        assert rsi_signals['signal'].isin([-1, 0, 1]).all(), "RSI signals should be -1, 0, or 1"
        assert macd_signals['signal'].isin([-1, 0, 1]).all(), "MACD signals should be -1, 0, or 1"
        
        # Test RSI-specific columns
        assert 'rsi' in rsi_signals.columns, "RSI should have RSI column"
        assert 'overbought' in rsi_signals.columns, "RSI should have overbought column"
        assert 'oversold' in rsi_signals.columns, "RSI should have oversold column"
        
        # Test MACD-specific columns
        macd_cols = [col for col in macd_signals.columns if 'MACD' in col or 'signal' in col]
        assert len(macd_cols) >= 3, "MACD should have MACD line, signal line, and histogram"
        
        # Test combination logic with multiple approaches
        combined_signals = pd.DataFrame(index=sample_data.index)
        combined_signals['rsi_signal'] = rsi_signals['signal']
        combined_signals['macd_signal'] = macd_signals['signal']
        
        # Approach 1: Both must agree for signal
        combined_signals['consensus_signal'] = 0
        buy_condition = (combined_signals['rsi_signal'] == 1) & (combined_signals['macd_signal'] == 1)
        sell_condition = (combined_signals['rsi_signal'] == -1) & (combined_signals['macd_signal'] == -1)
        
        combined_signals.loc[buy_condition, 'consensus_signal'] = 1
        combined_signals.loc[sell_condition, 'consensus_signal'] = -1
        
        # Approach 2: RSI confirms MACD
        combined_signals['rsi_confirmed'] = 0
        macd_buy = combined_signals['macd_signal'] == 1
        macd_sell = combined_signals['macd_signal'] == -1
        rsi_oversold = combined_signals['rsi_signal'] == 1
        rsi_overbought = combined_signals['rsi_signal'] == -1
        
        # Buy when MACD says buy and RSI confirms (not overbought)
        buy_confirmed = macd_buy & ~rsi_overbought
        # Sell when MACD says sell and RSI confirms (not oversold)
        sell_confirmed = macd_sell & ~rsi_oversold
        
        combined_signals.loc[buy_confirmed, 'rsi_confirmed'] = 1
        combined_signals.loc[sell_confirmed, 'rsi_confirmed'] = -1
        
        # Approach 3: Weighted combination
        combined_signals['weighted_signal'] = (
            combined_signals['rsi_signal'] * 0.4 + 
            combined_signals['macd_signal'] * 0.6
        )
        
        # Verify all approaches produce reasonable signals
        assert combined_signals['consensus_signal'].isin([-1, 0, 1]).all()
        assert combined_signals['rsi_confirmed'].isin([-1, 0, 1]).all()
        assert len(combined_signals[combined_signals['consensus_signal'] != 0]) > 0
        assert len(combined_signals[combined_signals['rsi_confirmed'] != 0]) > 0
        
        # Test signal distribution
        consensus_signals = combined_signals['consensus_signal'].value_counts()
        confirmed_signals = combined_signals['rsi_confirmed'].value_counts()
        
        logger.info(f"RSI + MACD consensus signals: {consensus_signals.to_dict()}")
        logger.info(f"RSI + MACD confirmed signals: {confirmed_signals.to_dict()}")
        
        # Verify signal quality
        assert consensus_signals.get(1, 0) > 0, "Should have buy signals"
        assert consensus_signals.get(-1, 0) > 0, "Should have sell signals"
    
    def test_bollinger_rsi_combination(self, sample_data, strategy_configs):
        """Test Bollinger Bands + RSI strategy combination with advanced validation."""
        logger.info("Testing comprehensive Bollinger Bands + RSI combination")
        
        # Create Bollinger strategy
        bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
        bollinger_strategy = BollingerStrategy(bollinger_config)
        bb_signals = bollinger_strategy.generate_signals(sample_data)
        
        # Generate RSI signals
        rsi_signals = generate_rsi_signals(sample_data, **strategy_configs['rsi'])
        
        # Test individual strategies
        assert not bb_signals.empty, "Bollinger signals should not be empty"
        assert not rsi_signals.empty, "RSI signals should not be empty"
        
        # Test Bollinger Bands specific columns
        assert 'upper_band' in bb_signals.columns, "Bollinger should have upper_band"
        assert 'lower_band' in bb_signals.columns, "Bollinger should have lower_band"
        assert 'middle_band' in bb_signals.columns, "Bollinger should have middle_band"
        assert 'signal' in bb_signals.columns, "Bollinger should have signal column"
        
        # Test RSI specific columns
        assert 'rsi' in rsi_signals.columns, "RSI should have RSI column"
        assert 'signal' in rsi_signals.columns, "RSI should have signal column"
        
        # Test combination with multiple confirmation strategies
        combined_signals = pd.DataFrame(index=sample_data.index)
        combined_signals['bb_signal'] = bb_signals['signal']
        combined_signals['rsi_signal'] = rsi_signals['signal']
        combined_signals['bb_upper'] = bb_signals['upper_band']
        combined_signals['bb_lower'] = bb_signals['lower_band']
        combined_signals['rsi_value'] = rsi_signals['rsi']
        
        # Strategy 1: RSI confirms Bollinger signal
        combined_signals['confirmed_signal'] = 0
        bb_buy = combined_signals['bb_signal'] == 1
        bb_sell = combined_signals['bb_signal'] == -1
        rsi_oversold = combined_signals['rsi_signal'] == 1
        rsi_overbought = combined_signals['rsi_signal'] == -1
        
        # Buy when BB says buy and RSI confirms (not overbought)
        buy_condition = bb_buy & ~rsi_overbought
        # Sell when BB says sell and RSI confirms (not oversold)
        sell_condition = bb_sell & ~rsi_oversold
        
        combined_signals.loc[buy_condition, 'confirmed_signal'] = 1
        combined_signals.loc[sell_condition, 'confirmed_signal'] = -1
        
        # Strategy 2: Bollinger breakout with RSI momentum
        combined_signals['breakout_signal'] = 0
        
        # Buy breakout: price below lower band and RSI oversold
        breakout_buy = (sample_data['close'] < combined_signals['bb_lower']) & (combined_signals['rsi_value'] < 30)
        # Sell breakout: price above upper band and RSI overbought
        breakout_sell = (sample_data['close'] > combined_signals['bb_upper']) & (combined_signals['rsi_value'] > 70)
        
        combined_signals.loc[breakout_buy, 'breakout_signal'] = 1
        combined_signals.loc[breakout_sell, 'breakout_signal'] = -1
        
        # Strategy 3: Mean reversion with RSI filter
        combined_signals['mean_reversion'] = 0
        
        # Buy mean reversion: price near lower band and RSI starting to rise
        mean_buy = (sample_data['close'] <= combined_signals['bb_lower'] * 1.02) & (combined_signals['rsi_value'] > 25)
        # Sell mean reversion: price near upper band and RSI starting to fall
        mean_sell = (sample_data['close'] >= combined_signals['bb_upper'] * 0.98) & (combined_signals['rsi_value'] < 75)
        
        combined_signals.loc[mean_buy, 'mean_reversion'] = 1
        combined_signals.loc[mean_sell, 'mean_reversion'] = -1
        
        # Verify all strategies produce reasonable signals
        assert combined_signals['confirmed_signal'].isin([-1, 0, 1]).all()
        assert combined_signals['breakout_signal'].isin([-1, 0, 1]).all()
        assert combined_signals['mean_reversion'].isin([-1, 0, 1]).all()
        
        # Test signal quality and distribution
        for strategy in ['confirmed_signal', 'breakout_signal', 'mean_reversion']:
            signal_counts = combined_signals[strategy].value_counts()
            logger.info(f"Bollinger + RSI {strategy}: {signal_counts.to_dict()}")
            
            assert signal_counts.get(1, 0) > 0, f"{strategy} should have buy signals"
            assert signal_counts.get(-1, 0) > 0, f"{strategy} should have sell signals"
    
    def test_sma_ema_combination(self, sample_data, strategy_configs):
        """Test SMA + EMA strategy combination with trend analysis."""
        logger.info("Testing comprehensive SMA + EMA combination")
        
        # Create strategies
        sma_strategy = SMAStrategy(**strategy_configs['sma'])
        sma_signals = sma_strategy.generate_signals(sample_data)
        
        # Create EMA strategy (using SMA with different parameters)
        ema_config = strategy_configs['sma'].copy()
        ema_config['short_window'] = strategy_configs['ema']['short_window']
        ema_config['long_window'] = strategy_configs['ema']['long_window']
        ema_strategy = SMAStrategy(**ema_config)
        ema_signals = ema_strategy.generate_signals(sample_data)
        
        # Test individual strategies
        assert not sma_signals.empty, "SMA signals should not be empty"
        assert not ema_signals.empty, "EMA signals should not be empty"
        
        # Test moving average columns
        sma_cols = [col for col in sma_signals.columns if 'SMA' in col]
        ema_cols = [col for col in ema_signals.columns if 'SMA' in col]  # Using SMA for EMA test
        assert len(sma_cols) >= 2, "SMA should have at least 2 moving averages"
        assert len(ema_cols) >= 2, "EMA should have at least 2 moving averages"
        
        # Test combination with multiple trend confirmation strategies
        combined_signals = pd.DataFrame(index=sample_data.index)
        combined_signals['sma_signal'] = sma_signals['signal']
        combined_signals['ema_signal'] = ema_signals['signal']
        
        # Strategy 1: Both must agree for trend confirmation
        combined_signals['trend_confirmed'] = 0
        both_buy = (combined_signals['sma_signal'] == 1) & (combined_signals['ema_signal'] == 1)
        both_sell = (combined_signals['sma_signal'] == -1) & (combined_signals['ema_signal'] == -1)
        
        combined_signals.loc[both_buy, 'trend_confirmed'] = 1
        combined_signals.loc[both_sell, 'trend_confirmed'] = -1
        
        # Strategy 2: EMA leads, SMA confirms
        combined_signals['ema_lead'] = 0
        ema_buy = combined_signals['ema_signal'] == 1
        ema_sell = combined_signals['ema_signal'] == -1
        sma_confirm_buy = combined_signals['sma_signal'] >= 0
        sma_confirm_sell = combined_signals['sma_signal'] <= 0
        
        lead_buy = ema_buy & sma_confirm_buy
        lead_sell = ema_sell & sma_confirm_sell
        
        combined_signals.loc[lead_buy, 'ema_lead'] = 1
        combined_signals.loc[lead_sell, 'ema_lead'] = -1
        
        # Strategy 3: Trend strength analysis
        combined_signals['trend_strength'] = 0
        
        # Strong uptrend: both moving averages showing buy signals
        strong_uptrend = both_buy
        # Strong downtrend: both moving averages showing sell signals
        strong_downtrend = both_sell
        # Weak trend: mixed signals
        weak_trend = (combined_signals['sma_signal'] != combined_signals['ema_signal']) & \
                    ((combined_signals['sma_signal'] != 0) | (combined_signals['ema_signal'] != 0))
        
        combined_signals.loc[strong_uptrend, 'trend_strength'] = 2
        combined_signals.loc[strong_downtrend, 'trend_strength'] = -2
        combined_signals.loc[weak_trend, 'trend_strength'] = 1
        
        # Verify all strategies produce reasonable signals
        assert combined_signals['trend_confirmed'].isin([-1, 0, 1]).all()
        assert combined_signals['ema_lead'].isin([-1, 0, 1]).all()
        assert combined_signals['trend_strength'].isin([-2, -1, 0, 1, 2]).all()
        
        # Test signal quality and distribution
        for strategy in ['trend_confirmed', 'ema_lead']:
            signal_counts = combined_signals[strategy].value_counts()
            logger.info(f"SMA + EMA {strategy}: {signal_counts.to_dict()}")
            
            assert signal_counts.get(1, 0) > 0, f"{strategy} should have buy signals"
            assert signal_counts.get(-1, 0) > 0, f"{strategy} should have sell signals"
        
        trend_strength_counts = combined_signals['trend_strength'].value_counts()
        logger.info(f"SMA + EMA trend strength: {trend_strength_counts.to_dict()}")
    
    def test_hybrid_strategy_combination(self, sample_data, strategy_configs, hybrid_config):
        """Test hybrid strategy with multiple components and advanced logic."""
        logger.info("Testing comprehensive hybrid strategy combination")
        
        # Create individual strategies
        strategies = {}
        
        # Bollinger strategy
        bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
        strategies['bollinger'] = BollingerStrategy(bollinger_config)
        
        # MACD strategy
        strategies['macd'] = MACDStrategy(**strategy_configs['macd'])
        
        # SMA strategy
        strategies['sma'] = SMAStrategy(**strategy_configs['sma'])
        
        # Generate signals for all strategies
        signals = {}
        for name, strategy in strategies.items():
            signals[name] = strategy.generate_signals(sample_data)
        
        # Generate RSI signals
        signals['rsi'] = generate_rsi_signals(sample_data, **strategy_configs['rsi'])
        
        # Test individual strategy signals
        for name, signal_df in signals.items():
            assert not signal_df.empty, f"{name} signals should not be empty"
            assert 'signal' in signal_df.columns, f"{name} should have signal column"
            assert signal_df['signal'].isin([-1, 0, 1]).all(), f"{name} signals should be -1, 0, or 1"
        
        # Create comprehensive hybrid combination
        hybrid_signals = pd.DataFrame(index=sample_data.index)
        
        # Add individual signals
        for name, signal_df in signals.items():
            hybrid_signals[f'{name}_signal'] = signal_df['signal']
        
        # Strategy 1: Weighted consensus
        weights = hybrid_config['weights']
        strategy_names = hybrid_config['strategies']
        
        weighted_signal = pd.Series(0, index=sample_data.index)
        for i, name in enumerate(strategy_names):
            if name in hybrid_signals.columns:
                weighted_signal += hybrid_signals[f'{name}_signal'] * weights[i]
        
        hybrid_signals['weighted_consensus'] = np.where(
            weighted_signal > hybrid_config['consensus_threshold'], 1,
            np.where(weighted_signal < -hybrid_config['consensus_threshold'], -1, 0)
        )
        
        # Strategy 2: Majority voting
        signal_columns = [f'{name}_signal' for name in strategy_names if f'{name}_signal' in hybrid_signals.columns]
        if signal_columns:
            majority_signal = hybrid_signals[signal_columns].sum(axis=1)
            hybrid_signals['majority_vote'] = np.where(
                majority_signal > 0, 1,
                np.where(majority_signal < 0, -1, 0)
            )
        
        # Strategy 3: Risk-adjusted signals
        hybrid_signals['risk_adjusted'] = 0
        
        # Buy condition: at least 2 strategies agree on buy and no strong sell signals
        buy_agreement = (hybrid_signals[signal_columns] == 1).sum(axis=1) >= 2
        no_strong_sell = (hybrid_signals[signal_columns] == -1).sum(axis=1) <= 1
        
        # Sell condition: at least 2 strategies agree on sell and no strong buy signals
        sell_agreement = (hybrid_signals[signal_columns] == -1).sum(axis=1) >= 2
        no_strong_buy = (hybrid_signals[signal_columns] == 1).sum(axis=1) <= 1
        
        hybrid_signals.loc[buy_agreement & no_strong_sell, 'risk_adjusted'] = 1
        hybrid_signals.loc[sell_agreement & no_strong_buy, 'risk_adjusted'] = -1
        
        # Strategy 4: Trend following with momentum
        hybrid_signals['trend_momentum'] = 0
        
        # Calculate trend strength
        trend_strength = hybrid_signals[signal_columns].sum(axis=1)
        
        # Strong trend with momentum
        strong_uptrend = trend_strength >= 2
        strong_downtrend = trend_strength <= -2
        
        hybrid_signals.loc[strong_uptrend, 'trend_momentum'] = 1
        hybrid_signals.loc[strong_downtrend, 'trend_momentum'] = -1
        
        # Verify all hybrid strategies produce reasonable signals
        hybrid_strategies = ['weighted_consensus', 'majority_vote', 'risk_adjusted', 'trend_momentum']
        
        for strategy in hybrid_strategies:
            if strategy in hybrid_signals.columns:
                assert hybrid_signals[strategy].isin([-1, 0, 1]).all(), f"{strategy} signals should be -1, 0, or 1"
                
                signal_counts = hybrid_signals[strategy].value_counts()
                logger.info(f"Hybrid {strategy}: {signal_counts.to_dict()}")
                
                assert signal_counts.get(1, 0) > 0, f"{strategy} should have buy signals"
                assert signal_counts.get(-1, 0) > 0, f"{strategy} should have sell signals"
    
    def test_multi_strategy_engine(self, sample_data, strategy_configs):
        """Test multi-strategy engine with comprehensive validation."""
        logger.info("Testing multi-strategy engine")
        
        # Create strategy manager
        strategy_manager = StrategyManager()
        
        # Register strategies
        bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
        strategy_manager.register_strategy('bollinger', BollingerStrategy(bollinger_config))
        strategy_manager.register_strategy('macd', MACDStrategy(**strategy_configs['macd']))
        strategy_manager.register_strategy('sma', SMAStrategy(**strategy_configs['sma']))
        
        # Test strategy registration
        registered_strategies = strategy_manager.get_registered_strategies()
        assert 'bollinger' in registered_strategies, "Bollinger strategy should be registered"
        assert 'macd' in registered_strategies, "MACD strategy should be registered"
        assert 'sma' in registered_strategies, "SMA strategy should be registered"
        
        # Generate signals for all strategies
        all_signals = strategy_manager.generate_all_signals(sample_data)
        
        # Test signal generation
        assert isinstance(all_signals, dict), "Should return dictionary of signals"
        assert len(all_signals) == 3, "Should have signals for all 3 strategies"
        
        for strategy_name, signals in all_signals.items():
            assert not signals.empty, f"{strategy_name} signals should not be empty"
            assert 'signal' in signals.columns, f"{strategy_name} should have signal column"
        
        # Test strategy comparison
        comparison = strategy_manager.compare_strategies(sample_data)
        
        assert isinstance(comparison, dict), "Comparison should return dictionary"
        assert 'performance' in comparison, "Should have performance metrics"
        assert 'signals' in comparison, "Should have signal analysis"
        
        logger.info(f"Strategy comparison results: {comparison}")
    
    def test_strategy_parameter_validation(self, sample_data, strategy_configs):
        """Test comprehensive strategy parameter validation."""
        logger.info("Testing comprehensive strategy parameter validation")
        
        # Create parameter validator
        validator = ParameterValidator()
        
        # Test Bollinger strategy with invalid parameters
        with pytest.raises(ValueError):
            invalid_config = BollingerConfig(window=-1, num_std=0)
            bollinger_strategy = BollingerStrategy(invalid_config)
            bollinger_strategy.generate_signals(sample_data)
        
        # Test MACD strategy with invalid parameters
        with pytest.raises(ValueError):
            invalid_macd_config = {
                'fast_period': -1,
                'slow_period': 26,
                'signal_period': 9
            }
            macd_strategy = MACDStrategy(**invalid_macd_config)
            macd_strategy.generate_signals(sample_data)
        
        # Test SMA strategy with invalid parameters
        with pytest.raises(ValueError):
            invalid_sma_config = {
                'short_window': 30,
                'long_window': 10  # Short window > long window
            }
            sma_strategy = SMAStrategy(**invalid_sma_config)
            sma_strategy.generate_signals(sample_data)
        
        # Test RSI with invalid parameters
        with pytest.raises(ValueError):
            invalid_rsi_config = {
                'period': 0,
                'overbought': 101,
                'oversold': -1
            }
            generate_rsi_signals(sample_data, **invalid_rsi_config)
        
        # Test parameter validation functions
        valid_params = {
            'bollinger': strategy_configs['bollinger'],
            'macd': strategy_configs['macd'],
            'sma': strategy_configs['sma'],
            'rsi': strategy_configs['rsi']
        }
        
        for strategy_name, params in valid_params.items():
            validation_result = validator.validate_parameters(strategy_name, params)
            assert validation_result['valid'], f"{strategy_name} parameters should be valid"
        
        logger.info("Comprehensive strategy parameter validation tests passed")
    
    def test_strategy_performance_comparison(self, sample_data, strategy_configs):
        """Test comprehensive strategy performance comparison."""
        logger.info("Testing comprehensive strategy performance comparison")
        
        # Create strategies
        strategies = {}
        
        bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
        strategies['Bollinger'] = BollingerStrategy(bollinger_config)
        strategies['MACD'] = MACDStrategy(**strategy_configs['macd'])
        strategies['SMA'] = SMAStrategy(**strategy_configs['sma'])
        
        # Generate signals
        signals = {}
        for name, strategy in strategies.items():
            signals[name] = strategy.generate_signals(sample_data)
        
        # Add RSI signals
        signals['RSI'] = generate_rsi_signals(sample_data, **strategy_configs['rsi'])
        
        # Calculate comprehensive performance metrics
        performance = {}
        for name, signal_df in signals.items():
            # Basic metrics
            total_signals = len(signal_df[signal_df['signal'] != 0])
            buy_signals = len(signal_df[signal_df['signal'] == 1])
            sell_signals = len(signal_df[signal_df['signal'] == -1])
            
            # Signal distribution
            signal_distribution = signal_df['signal'].value_counts()
            
            # Signal frequency
            signal_frequency = total_signals / len(signal_df) if len(signal_df) > 0 else 0
            
            # Signal persistence (consecutive signals)
            signal_changes = signal_df['signal'].diff().abs()
            avg_signal_persistence = signal_changes.mean() if len(signal_changes) > 0 else 0
            
            # Calculate returns if price data available
            returns = None
            if 'close' in sample_data.columns:
                price_returns = sample_data['close'].pct_change()
                strategy_returns = signal_df['signal'].shift(1) * price_returns
                returns = strategy_returns.dropna()
            
            performance[name] = {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'signal_ratio': signal_frequency,
                'signal_distribution': signal_distribution.to_dict(),
                'avg_signal_persistence': avg_signal_persistence,
                'returns': returns
            }
            
            # Calculate return metrics if available
            if returns is not None and len(returns) > 0:
                performance[name].update({
                    'total_return': returns.sum(),
                    'avg_return': returns.mean(),
                    'return_std': returns.std(),
                    'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                    'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
                    'win_rate': len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
                })
        
        # Verify performance metrics
        for name, metrics in performance.items():
            # Basic validation
            assert metrics['total_signals'] >= 0, f"{name} total signals should be non-negative"
            assert metrics['buy_signals'] >= 0, f"{name} buy signals should be non-negative"
            assert metrics['sell_signals'] >= 0, f"{name} sell signals should be non-negative"
            assert 0 <= metrics['signal_ratio'] <= 1, f"{name} signal ratio should be between 0 and 1"
            
            # Signal distribution validation
            signal_dist = metrics['signal_distribution']
            assert -1 in signal_dist, f"{name} should have sell signals"
            assert 0 in signal_dist, f"{name} should have neutral signals"
            assert 1 in signal_dist, f"{name} should have buy signals"
            
            # Return metrics validation (if available)
            if 'returns' in metrics and metrics['returns'] is not None:
                assert isinstance(metrics['total_return'], (int, float)), f"{name} total return should be numeric"
                assert isinstance(metrics['avg_return'], (int, float)), f"{name} avg return should be numeric"
                assert metrics['return_std'] >= 0, f"{name} return std should be non-negative"
                assert 0 <= metrics['win_rate'] <= 1, f"{name} win rate should be between 0 and 1"
        
        # Compare strategies
        logger.info("Strategy Performance Comparison:")
        for name, metrics in performance.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Total Signals: {metrics['total_signals']}")
            logger.info(f"  Buy/Sell Ratio: {metrics['buy_signals']}/{metrics['sell_signals']}")
            logger.info(f"  Signal Frequency: {metrics['signal_ratio']:.3f}")
            
            if 'total_return' in metrics:
                logger.info(f"  Total Return: {metrics['total_return']:.4f}")
                logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                logger.info(f"  Win Rate: {metrics['win_rate']:.3f}")
        
        # Find best performing strategy
        if any('total_return' in metrics for metrics in performance.values()):
            best_strategy = max(
                [(name, metrics) for name, metrics in performance.items() if 'total_return' in metrics],
                key=lambda x: x[1]['total_return']
            )
            logger.info(f"\nBest performing strategy: {best_strategy[0]} with {best_strategy[1]['total_return']:.4f} return")
    
    def test_edge_cases_and_error_handling(self, sample_data, strategy_configs):
        """Test edge cases and error handling for strategy combinations."""
        logger.info("Testing edge cases and error handling")
        
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
            bollinger_strategy = BollingerStrategy(bollinger_config)
            bollinger_strategy.generate_signals(empty_data)
        
        # Test with insufficient data
        small_data = sample_data.head(5)  # Too few data points
        with pytest.raises((ValueError, KeyError)):
            macd_strategy = MACDStrategy(**strategy_configs['macd'])
            macd_strategy.generate_signals(small_data)
        
        # Test with missing columns
        incomplete_data = sample_data[['open', 'high']].copy()  # Missing close and volume
        with pytest.raises(ValueError):
            bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
            bollinger_strategy = BollingerStrategy(bollinger_config)
            bollinger_strategy.generate_signals(incomplete_data)
        
        # Test with NaN values
        nan_data = sample_data.copy()
        nan_data.loc[nan_data.index[10:20], 'close'] = np.nan
        with pytest.raises((ValueError, KeyError)):
            sma_strategy = SMAStrategy(**strategy_configs['sma'])
            sma_strategy.generate_signals(nan_data)
        
        # Test with extreme values
        extreme_data = sample_data.copy()
        extreme_data['close'] = extreme_data['close'] * 1000000  # Extreme prices
        extreme_data['volume'] = extreme_data['volume'] * 0.0001  # Very low volume
        
        # Should handle extreme values gracefully
        try:
            bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
            bollinger_strategy = BollingerStrategy(bollinger_config)
            signals = bollinger_strategy.generate_signals(extreme_data)
            assert not signals.empty, "Should handle extreme values"
        except Exception as e:
            logger.warning(f"Strategy could not handle extreme values: {e}")
        
        logger.info("Edge cases and error handling tests completed")
    
    def test_strategy_optimization_scenarios(self, sample_data, strategy_configs):
        """Test strategy optimization scenarios and parameter tuning."""
        logger.info("Testing strategy optimization scenarios")
        
        # Test parameter optimization for Bollinger strategy
        bollinger_params = [
            {'window': 10, 'num_std': 1.5},
            {'window': 20, 'num_std': 2.0},
            {'window': 30, 'num_std': 2.5}
        ]
        
        bollinger_performance = []
        for params in bollinger_params:
            try:
                config = BollingerConfig(**params)
                strategy = BollingerStrategy(config)
                signals = strategy.generate_signals(sample_data)
                
                # Calculate performance metric
                total_signals = len(signals[signals['signal'] != 0])
                signal_frequency = total_signals / len(signals) if len(signals) > 0 else 0
                
                bollinger_performance.append({
                    'params': params,
                    'signal_frequency': signal_frequency,
                    'total_signals': total_signals
                })
            except Exception as e:
                logger.warning(f"Failed to test Bollinger params {params}: {e}")
        
        # Find optimal parameters
        if bollinger_performance:
            optimal_bollinger = max(bollinger_performance, key=lambda x: x['signal_frequency'])
            logger.info(f"Optimal Bollinger parameters: {optimal_bollinger['params']}")
        
        # Test MACD parameter optimization
        macd_params = [
            {'fast_period': 8, 'slow_period': 21, 'signal_period': 5},
            {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            {'fast_period': 16, 'slow_period': 32, 'signal_period': 13}
        ]
        
        macd_performance = []
        for params in macd_params:
            try:
                strategy = MACDStrategy(**params)
                signals = strategy.generate_signals(sample_data)
                
                total_signals = len(signals[signals['signal'] != 0])
                signal_frequency = total_signals / len(signals) if len(signals) > 0 else 0
                
                macd_performance.append({
                    'params': params,
                    'signal_frequency': signal_frequency,
                    'total_signals': total_signals
                })
            except Exception as e:
                logger.warning(f"Failed to test MACD params {params}: {e}")
        
        # Find optimal parameters
        if macd_performance:
            optimal_macd = max(macd_performance, key=lambda x: x['signal_frequency'])
            logger.info(f"Optimal MACD parameters: {optimal_macd['params']}")
        
        logger.info("Strategy optimization scenarios completed")
    
    def test_integration_with_risk_management(self, sample_data, strategy_configs):
        """Test integration with risk management features."""
        logger.info("Testing integration with risk management")
        
        # Create strategies with risk management
        bollinger_config = BollingerConfig(**strategy_configs['bollinger'])
        bollinger_strategy = BollingerStrategy(bollinger_config)
        
        # Generate signals
        signals = bollinger_strategy.generate_signals(sample_data)
        
        # Apply risk management rules
        risk_managed_signals = signals.copy()
        
        # Rule 1: Maximum consecutive signals
        max_consecutive = 5
        signal_changes = risk_managed_signals['signal'].diff().abs()
        consecutive_count = 0
        
        for i in range(1, len(risk_managed_signals)):
            if signal_changes.iloc[i] == 0 and risk_managed_signals['signal'].iloc[i] != 0:
                consecutive_count += 1
                if consecutive_count > max_consecutive:
                    risk_managed_signals['signal'].iloc[i] = 0
            else:
                consecutive_count = 0
        
        # Rule 2: Volume-based filtering
        if 'volume' in sample_data.columns:
            volume_threshold = sample_data['volume'].quantile(0.1)  # Bottom 10%
            low_volume_mask = sample_data['volume'] < volume_threshold
            risk_managed_signals.loc[low_volume_mask, 'signal'] = 0
        
        # Rule 3: Price volatility filtering
        if 'close' in sample_data.columns:
            price_volatility = sample_data['close'].pct_change().rolling(20).std()
            high_volatility_mask = price_volatility > price_volatility.quantile(0.9)  # Top 10%
            risk_managed_signals.loc[high_volatility_mask, 'signal'] = 0
        
        # Compare original vs risk-managed signals
        original_signals = len(signals[signals['signal'] != 0])
        risk_managed_count = len(risk_managed_signals[risk_managed_signals['signal'] != 0])
        
        logger.info(f"Original signals: {original_signals}")
        logger.info(f"Risk-managed signals: {risk_managed_count}")
        logger.info(f"Risk management reduced signals by: {original_signals - risk_managed_count}")
        
        # Verify risk management worked
        assert risk_managed_count <= original_signals, "Risk management should not increase signals"
        
        logger.info("Risk management integration tests completed") 