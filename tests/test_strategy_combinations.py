"""
Test cases for new strategy combinations.

This module tests various combinations of trading strategies to ensure
they work together properly and produce expected results.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

# Import strategies
from trading.strategies.rsi_strategy import RSIStrategy
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.bollinger_strategy import BollingerStrategy
from trading.strategies.sma_strategy import SMAStrategy
from trading.strategies.ema_strategy import EMAStrategy
from trading.strategies.hybrid_strategy import HybridStrategy
from trading.strategies.multi_strategy_hybrid_engine import MultiStrategyHybridEngine

logger = logging.getLogger(__name__)

class TestStrategyCombinations:
    """Test various strategy combinations."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample price data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
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
        """Get strategy configurations for testing."""
        return {
            'rsi': {
                'period': 14,
                'overbought': 70,
                'oversold': 30
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'bollinger': {
                'period': 20,
                'std_dev': 2
            },
            'sma': {
                'short_window': 10,
                'long_window': 30
            },
            'ema': {
                'short_window': 12,
                'long_window': 26
            }
        }
    
    def test_rsi_macd_combination(self, sample_data, strategy_configs):
        """Test RSI + MACD strategy combination."""
        logger.info("Testing RSI + MACD combination")
        
        # Create strategies
        rsi_strategy = RSIStrategy(**strategy_configs['rsi'])
        macd_strategy = MACDStrategy(**strategy_configs['macd'])
        
        # Generate signals
        rsi_signals = rsi_strategy.generate_signals(sample_data)
        macd_signals = macd_strategy.generate_signals(sample_data)
        
        # Test individual strategies
        assert not rsi_signals.empty, "RSI signals should not be empty"
        assert not macd_signals.empty, "MACD signals should not be empty"
        assert 'signal' in rsi_signals.columns, "RSI should have signal column"
        assert 'signal' in macd_signals.columns, "MACD should have signal column"
        
        # Test signal values
        assert rsi_signals['signal'].isin([-1, 0, 1]).all(), "RSI signals should be -1, 0, or 1"
        assert macd_signals['signal'].isin([-1, 0, 1]).all(), "MACD signals should be -1, 0, or 1"
        
        # Test combination logic
        combined_signals = pd.DataFrame(index=sample_data.index)
        combined_signals['rsi_signal'] = rsi_signals['signal']
        combined_signals['macd_signal'] = macd_signals['signal']
        
        # Simple combination: both must agree for signal
        combined_signals['combined_signal'] = 0
        buy_condition = (combined_signals['rsi_signal'] == 1) & (combined_signals['macd_signal'] == 1)
        sell_condition = (combined_signals['rsi_signal'] == -1) & (combined_signals['macd_signal'] == -1)
        
        combined_signals.loc[buy_condition, 'combined_signal'] = 1
        combined_signals.loc[sell_condition, 'combined_signal'] = -1
        
        # Verify combination produces reasonable signals
        assert combined_signals['combined_signal'].isin([-1, 0, 1]).all()
        assert len(combined_signals[combined_signals['combined_signal'] != 0]) > 0, "Should have some combined signals"
        
        logger.info(f"RSI + MACD combination: {len(combined_signals[combined_signals['combined_signal'] != 0])} signals generated")
    
    def test_bollinger_rsi_combination(self, sample_data, strategy_configs):
        """Test Bollinger Bands + RSI strategy combination."""
        logger.info("Testing Bollinger Bands + RSI combination")
        
        # Create strategies
        bollinger_strategy = BollingerStrategy(**strategy_configs['bollinger'])
        rsi_strategy = RSIStrategy(**strategy_configs['rsi'])
        
        # Generate signals
        bb_signals = bollinger_strategy.generate_signals(sample_data)
        rsi_signals = rsi_strategy.generate_signals(sample_data)
        
        # Test individual strategies
        assert not bb_signals.empty, "Bollinger signals should not be empty"
        assert not rsi_signals.empty, "RSI signals should not be empty"
        
        # Test Bollinger Bands specific columns
        assert 'upper_band' in bb_signals.columns, "Bollinger should have upper_band"
        assert 'lower_band' in bb_signals.columns, "Bollinger should have lower_band"
        assert 'middle_band' in bb_signals.columns, "Bollinger should have middle_band"
        
        # Test combination with confirmation
        combined_signals = pd.DataFrame(index=sample_data.index)
        combined_signals['bb_signal'] = bb_signals['signal']
        combined_signals['rsi_signal'] = rsi_signals['signal']
        
        # RSI confirms Bollinger signal
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
        
        # Verify combination
        assert combined_signals['confirmed_signal'].isin([-1, 0, 1]).all()
        
        logger.info(f"Bollinger + RSI combination: {len(combined_signals[combined_signals['confirmed_signal'] != 0])} confirmed signals")
    
    def test_sma_ema_combination(self, sample_data, strategy_configs):
        """Test SMA + EMA strategy combination."""
        logger.info("Testing SMA + EMA combination")
        
        # Create strategies
        sma_strategy = SMAStrategy(**strategy_configs['sma'])
        ema_strategy = EMAStrategy(**strategy_configs['ema'])
        
        # Generate signals
        sma_signals = sma_strategy.generate_signals(sample_data)
        ema_signals = ema_strategy.generate_signals(sample_data)
        
        # Test individual strategies
        assert not sma_signals.empty, "SMA signals should not be empty"
        assert not ema_signals.empty, "EMA signals should not be empty"
        
        # Test moving average columns
        sma_cols = [col for col in sma_signals.columns if 'SMA' in col]
        ema_cols = [col for col in ema_signals.columns if 'EMA' in col]
        assert len(sma_cols) >= 2, "SMA should have at least 2 moving averages"
        assert len(ema_cols) >= 2, "EMA should have at least 2 moving averages"
        
        # Test combination with trend confirmation
        combined_signals = pd.DataFrame(index=sample_data.index)
        combined_signals['sma_signal'] = sma_signals['signal']
        combined_signals['ema_signal'] = ema_signals['signal']
        
        # Both must agree for trend confirmation
        combined_signals['trend_confirmed'] = 0
        both_buy = (combined_signals['sma_signal'] == 1) & (combined_signals['ema_signal'] == 1)
        both_sell = (combined_signals['sma_signal'] == -1) & (combined_signals['ema_signal'] == -1)
        
        combined_signals.loc[both_buy, 'trend_confirmed'] = 1
        combined_signals.loc[both_sell, 'trend_confirmed'] = -1
        
        # Verify combination
        assert combined_signals['trend_confirmed'].isin([-1, 0, 1]).all()
        
        logger.info(f"SMA + EMA combination: {len(combined_signals[combined_signals['trend_confirmed'] != 0])} trend-confirmed signals")
    
    def test_hybrid_strategy_combination(self, sample_data, strategy_configs):
        """Test hybrid strategy with multiple components."""
        logger.info("Testing hybrid strategy combination")
        
        # Create hybrid strategy
        hybrid_config = {
            'strategies': [
                {'name': 'rsi', 'config': strategy_configs['rsi'], 'weight': 0.3},
                {'name': 'macd', 'config': strategy_configs['macd'], 'weight': 0.3},
                {'name': 'bollinger', 'config': strategy_configs['bollinger'], 'weight': 0.4}
            ],
            'consensus_threshold': 0.6
        }
        
        hybrid_strategy = HybridStrategy(hybrid_config)
        
        # Generate signals
        hybrid_signals = hybrid_strategy.generate_signals(sample_data)
        
        # Test hybrid strategy
        assert not hybrid_signals.empty, "Hybrid signals should not be empty"
        assert 'signal' in hybrid_signals.columns, "Hybrid should have signal column"
        assert 'confidence' in hybrid_signals.columns, "Hybrid should have confidence column"
        
        # Test signal values
        assert hybrid_signals['signal'].isin([-1, 0, 1]).all(), "Hybrid signals should be -1, 0, or 1"
        assert (hybrid_signals['confidence'] >= 0).all(), "Confidence should be non-negative"
        assert (hybrid_signals['confidence'] <= 1).all(), "Confidence should be <= 1"
        
        # Test that high confidence signals exist
        high_confidence = hybrid_signals[hybrid_signals['confidence'] > 0.7]
        assert len(high_confidence) > 0, "Should have some high confidence signals"
        
        logger.info(f"Hybrid strategy: {len(hybrid_signals[hybrid_signals['signal'] != 0])} signals, "
                   f"{len(high_confidence)} high confidence")
    
    def test_multi_strategy_engine(self, sample_data, strategy_configs):
        """Test multi-strategy hybrid engine."""
        logger.info("Testing multi-strategy hybrid engine")
        
        # Create engine
        engine = MultiStrategyHybridEngine()
        
        # Add strategies
        strategies = {
            'rsi': RSIStrategy(**strategy_configs['rsi']),
            'macd': MACDStrategy(**strategy_configs['macd']),
            'bollinger': BollingerStrategy(**strategy_configs['bollinger']),
            'sma': SMAStrategy(**strategy_configs['sma'])
        }
        
        for name, strategy in strategies.items():
            engine.add_strategy(name, strategy)
        
        # Generate signals
        engine_signals = engine.generate_signals(sample_data)
        
        # Test engine output
        assert not engine_signals.empty, "Engine signals should not be empty"
        assert 'signal' in engine_signals.columns, "Engine should have signal column"
        assert 'confidence' in engine_signals.columns, "Engine should have confidence column"
        assert 'consensus_score' in engine_signals.columns, "Engine should have consensus_score column"
        
        # Test signal values
        assert engine_signals['signal'].isin([-1, 0, 1]).all(), "Engine signals should be -1, 0, or 1"
        assert (engine_signals['confidence'] >= 0).all(), "Confidence should be non-negative"
        assert (engine_signals['consensus_score'] >= 0).all(), "Consensus score should be non-negative"
        
        # Test performance metrics
        performance = engine.get_performance_metrics()
        assert isinstance(performance, dict), "Performance should be a dictionary"
        assert 'total_signals' in performance, "Performance should have total_signals"
        assert 'consensus_agreement' in performance, "Performance should have consensus_agreement"
        
        logger.info(f"Multi-strategy engine: {performance['total_signals']} signals, "
                   f"{performance['consensus_agreement']:.2f} consensus agreement")
    
    def test_strategy_parameter_validation(self, sample_data, strategy_configs):
        """Test strategy parameter validation in combinations."""
        logger.info("Testing strategy parameter validation")
        
        # Test invalid parameters
        invalid_configs = [
            {'period': -1},  # Negative period
            {'period': 0},   # Zero period
            {'fast_period': 30, 'slow_period': 10},  # Fast > slow
            {'short_window': 50, 'long_window': 20}  # Short > long
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                if 'period' in invalid_config:
                    RSIStrategy(**invalid_config)
                elif 'fast_period' in invalid_config:
                    MACDStrategy(**invalid_config)
                elif 'short_window' in invalid_config:
                    SMAStrategy(**invalid_config)
        
        # Test valid parameter combinations
        valid_combinations = [
            {'rsi': {'period': 14}, 'macd': {'fast_period': 12, 'slow_period': 26}},
            {'bollinger': {'period': 20}, 'sma': {'short_window': 10, 'long_window': 30}},
            {'rsi': {'period': 21}, 'ema': {'short_window': 12, 'long_window': 26}}
        ]
        
        for combo in valid_combinations:
            strategies = {}
            for name, config in combo.items():
                if name == 'rsi':
                    strategies[name] = RSIStrategy(**config)
                elif name == 'macd':
                    strategies[name] = MACDStrategy(**config)
                elif name == 'bollinger':
                    strategies[name] = BollingerStrategy(**config)
                elif name == 'sma':
                    strategies[name] = SMAStrategy(**config)
                elif name == 'ema':
                    strategies[name] = EMAStrategy(**config)
            
            # All strategies should be created successfully
            assert len(strategies) == len(combo), "All strategies should be created"
            
            # Test signal generation
            for name, strategy in strategies.items():
                signals = strategy.generate_signals(sample_data)
                assert not signals.empty, f"{name} should generate signals"
        
        logger.info("Strategy parameter validation completed successfully")
    
    def test_strategy_performance_comparison(self, sample_data, strategy_configs):
        """Test performance comparison between strategy combinations."""
        logger.info("Testing strategy performance comparison")
        
        # Create different strategy combinations
        combinations = {
            'rsi_only': RSIStrategy(**strategy_configs['rsi']),
            'macd_only': MACDStrategy(**strategy_configs['macd']),
            'rsi_macd': HybridStrategy({
                'strategies': [
                    {'name': 'rsi', 'config': strategy_configs['rsi'], 'weight': 0.5},
                    {'name': 'macd', 'config': strategy_configs['macd'], 'weight': 0.5}
                ],
                'consensus_threshold': 0.5
            }),
            'bollinger_rsi': HybridStrategy({
                'strategies': [
                    {'name': 'bollinger', 'config': strategy_configs['bollinger'], 'weight': 0.6},
                    {'name': 'rsi', 'config': strategy_configs['rsi'], 'weight': 0.4}
                ],
                'consensus_threshold': 0.6
            })
        }
        
        # Generate signals for each combination
        results = {}
        for name, strategy in combinations.items():
            signals = strategy.generate_signals(sample_data)
            
            # Calculate basic metrics
            total_signals = len(signals[signals['signal'] != 0])
            buy_signals = len(signals[signals['signal'] == 1])
            sell_signals = len(signals[signals['signal'] == -1])
            
            results[name] = {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'signal_ratio': buy_signals / total_signals if total_signals > 0 else 0
            }
        
        # Verify all combinations produce signals
        for name, result in results.items():
            assert result['total_signals'] > 0, f"{name} should produce signals"
            assert result['buy_signals'] >= 0, f"{name} should have non-negative buy signals"
            assert result['sell_signals'] >= 0, f"{name} should have non-negative sell signals"
        
        # Log comparison results
        for name, result in results.items():
            logger.info(f"{name}: {result['total_signals']} total, "
                       f"{result['buy_signals']} buy, {result['sell_signals']} sell, "
                       f"ratio: {result['signal_ratio']:.2f}")
        
        logger.info("Strategy performance comparison completed") 