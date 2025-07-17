#!/usr/bin/env python3
"""
Strategy Combo Example

This script demonstrates how to use the enhanced strategy pipeline
to create and test combinations of multiple trading strategies.

Features demonstrated:
- Creating strategy combinations with different modes
- Testing different combination methods
- Performance analysis and comparison
- Integration with existing strategy system
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample market data for testing."""
    logger.info(f"Generating {n_samples} samples of market data")
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(0, 0.2, n_samples)
    
    # Random walk with volatility clustering
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Add volatility clustering
    volatility = np.exp(np.random.AR(0.9).generate_sample(n_samples) * 0.5)
    returns = returns * volatility
    
    # Add trend
    returns = returns + trend * 0.001
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from price
        daily_volatility = np.random.normal(0, 0.01)
        
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(daily_volatility))
        low_price = min(open_price, price) * (1 - abs(daily_volatility))
        close_price = price
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    logger.info(f"Generated data with shape: {df.shape}")
    return df


def demonstrate_basic_combination():
    """Demonstrate basic strategy combination."""
    logger.info("=== Basic Strategy Combination Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    try:
        from strategies.strategy_pipeline import (
            StrategyPipeline, StrategyConfig, CombinationConfig,
            rsi_strategy, macd_strategy, bollinger_strategy
        )
        
        # Create strategy configurations
        strategies = [
            StrategyConfig(name="RSI", weight=1.0, parameters={'window': 14}),
            StrategyConfig(name="MACD", weight=1.0, parameters={'fast': 12, 'slow': 26}),
            StrategyConfig(name="Bollinger", weight=1.0, parameters={'window': 20})
        ]
        
        # Create combination configuration
        combination_config = CombinationConfig(
            mode='intersection',
            min_agreement=0.5,
            confidence_threshold=0.6
        )
        
        # Create pipeline
        pipeline = StrategyPipeline(strategies, combination_config)
        
        # Add strategy functions
        pipeline.strategy_functions = {
            "RSI": rsi_strategy,
            "MACD": macd_strategy,
            "Bollinger": bollinger_strategy
        }
        
        # Generate combined signals
        combined_signal, metadata = pipeline.generate_combined_signals(data)
        
        logger.info(f"Combined signal generated with {len(combined_signal[combined_signal != 0])} signals")
        logger.info(f"Signal agreement: {metadata['signal_agreement']:.3f}")
        
        return combined_signal, metadata
        
    except ImportError as e:
        logger.error(f"Could not import strategy pipeline: {e}")
        return None, None


def demonstrate_different_modes():
    """Demonstrate different combination modes."""
    logger.info("=== Different Combination Modes Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    try:
        from strategies.strategy_pipeline import (
            StrategyPipeline, StrategyConfig, CombinationConfig,
            rsi_strategy, macd_strategy, bollinger_strategy
        )
        
        # Test different combination modes
        modes = ['intersection', 'union', 'weighted', 'voting', 'confidence']
        results = {}
        
        for mode in modes:
            logger.info(f"Testing {mode} mode...")
            
            # Create pipeline with current mode
            strategies = [
                StrategyConfig(name="RSI", weight=1.0),
                StrategyConfig(name="MACD", weight=1.0),
                StrategyConfig(name="Bollinger", weight=1.0)
            ]
            
            combination_config = CombinationConfig(mode=mode)
            pipeline = StrategyPipeline(strategies, combination_config)
            
            # Add strategy functions
            pipeline.strategy_functions = {
                "RSI": rsi_strategy,
                "MACD": macd_strategy,
                "Bollinger": bollinger_strategy
            }
            
            # Generate signals
            combined_signal, metadata = pipeline.generate_combined_signals(data)
            
            # Calculate basic metrics
            signal_count = len(combined_signal[combined_signal != 0])
            buy_signals = len(combined_signal[combined_signal == 1])
            sell_signals = len(combined_signal[combined_signal == -1])
            
            results[mode] = {
                'signal': combined_signal,
                'metadata': metadata,
                'signal_count': signal_count,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'agreement': metadata['signal_agreement']
            }
            
            logger.info(f"{mode}: {signal_count} signals ({buy_signals} buy, {sell_signals} sell)")
        
        return results
        
    except ImportError as e:
        logger.error(f"Could not import strategy pipeline: {e}")
        return {}


def demonstrate_weighted_combination():
    """Demonstrate weighted strategy combination."""
    logger.info("=== Weighted Strategy Combination Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    try:
        from strategies.strategy_pipeline import (
            StrategyPipeline, StrategyConfig, CombinationConfig,
            rsi_strategy, macd_strategy, bollinger_strategy
        )
        
        # Test different weight combinations
        weight_combinations = [
            {"RSI": 0.5, "MACD": 0.3, "Bollinger": 0.2},
            {"RSI": 0.2, "MACD": 0.5, "Bollinger": 0.3},
            {"RSI": 0.3, "MACD": 0.2, "Bollinger": 0.5},
            {"RSI": 1.0, "MACD": 1.0, "Bollinger": 1.0}  # Equal weights
        ]
        
        results = {}
        
        for i, weights in enumerate(weight_combinations):
            logger.info(f"Testing weight combination {i+1}: {weights}")
            
            # Create strategy configurations with weights
            strategies = [
                StrategyConfig(name="RSI", weight=weights["RSI"]),
                StrategyConfig(name="MACD", weight=weights["MACD"]),
                StrategyConfig(name="Bollinger", weight=weights["Bollinger"])
            ]
            
            # Create combination configuration
            combination_config = CombinationConfig(mode='weighted')
            pipeline = StrategyPipeline(strategies, combination_config)
            
            # Add strategy functions
            pipeline.strategy_functions = {
                "RSI": rsi_strategy,
                "MACD": macd_strategy,
                "Bollinger": bollinger_strategy
            }
            
            # Generate signals
            combined_signal, metadata = pipeline.generate_combined_signals(data)
            
            # Calculate performance metrics
            performance = calculate_performance_metrics(data, combined_signal)
            
            results[f"weights_{i+1}"] = {
                'weights': weights,
                'signal': combined_signal,
                'metadata': metadata,
                'performance': performance
            }
            
            logger.info(f"Weights {i+1} - Sharpe: {performance['sharpe_ratio']:.3f}, "
                       f"Return: {performance['total_return']:.3f}")
        
        return results
        
    except ImportError as e:
        logger.error(f"Could not import strategy pipeline: {e}")
        return {}


def calculate_performance_metrics(data: pd.DataFrame, signal: pd.Series) -> Dict[str, float]:
    """Calculate performance metrics for a strategy signal."""
    try:
        # Calculate returns
        price_returns = data['close'].pct_change()
        strategy_returns = signal.shift(1) * price_returns
        
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }


def demonstrate_strategy_pipeline_integration():
    """Demonstrate integration with the strategy pipeline."""
    logger.info("=== Strategy Pipeline Integration Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    try:
        from strategies.strategy_pipeline import (
            StrategyPipeline, StrategyConfig, CombinationConfig,
            create_strategy_combo
        )
        
        # Create a strategy combo using the convenience function
        strategy_names = ["RSI", "MACD", "Bollinger"]
        pipeline = create_strategy_combo(
            strategy_names=strategy_names,
            mode='weighted',
            weights=[0.4, 0.4, 0.2]
        )
        
        # Generate signals
        combined_signal, metadata = pipeline.generate_combined_signals(data, strategy_names)
        
        # Calculate performance
        performance = calculate_performance_metrics(data, combined_signal)
        
        logger.info("Strategy pipeline integration successful!")
        logger.info(f"Performance - Sharpe: {performance['sharpe_ratio']:.3f}, "
                   f"Return: {performance['total_return']:.3f}")
        
        # Get performance summary
        summary = pipeline.get_performance_summary()
        logger.info(f"Pipeline summary: {summary}")
        
        return {
            'pipeline': pipeline,
            'signal': combined_signal,
            'metadata': metadata,
            'performance': performance,
            'summary': summary
        }
        
    except ImportError as e:
        logger.error(f"Could not import strategy pipeline: {e}")
        return {}


def demonstrate_backward_compatibility():
    """Demonstrate backward compatibility with existing code."""
    logger.info("=== Backward Compatibility Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    try:
        from strategies.strategy_pipeline import (
            combine_signals, rsi_strategy, macd_strategy, bollinger_strategy
        )
        
        # Generate individual signals using existing functions
        rsi_signals = rsi_strategy(data)
        macd_signals = macd_strategy(data)
        bollinger_signals = bollinger_strategy(data)
        
        # Combine using the backward-compatible function
        combined = combine_signals(
            signals_list=[rsi_signals, macd_signals, bollinger_signals],
            mode='intersection'
        )
        
        logger.info(f"Backward compatibility test successful!")
        logger.info(f"Combined {len(combined[combined != 0])} signals using intersection mode")
        
        return combined
        
    except ImportError as e:
        logger.error(f"Could not import strategy pipeline: {e}")
        return None


def main():
    """Run all demonstration functions."""
    logger.info("Starting Strategy Combo Examples")
    logger.info("=" * 50)
    
    try:
        # Run basic combination demo
        basic_signal, basic_metadata = demonstrate_basic_combination()
        logger.info("-" * 30)
        
        # Run different modes demo
        modes_results = demonstrate_different_modes()
        logger.info("-" * 30)
        
        # Run weighted combination demo
        weighted_results = demonstrate_weighted_combination()
        logger.info("-" * 30)
        
        # Run pipeline integration demo
        integration_results = demonstrate_strategy_pipeline_integration()
        logger.info("-" * 30)
        
        # Run backward compatibility demo
        backward_signal = demonstrate_backward_compatibility()
        logger.info("-" * 30)
        
        # Summary
        logger.info("=== SUMMARY ===")
        logger.info(f"Basic combination: {len(basic_signal[basic_signal != 0]) if basic_signal is not None else 0} signals")
        logger.info(f"Different modes tested: {len(modes_results)}")
        logger.info(f"Weighted combinations tested: {len(weighted_results)}")
        logger.info(f"Pipeline integration: {'Success' if integration_results else 'Failed'}")
        logger.info(f"Backward compatibility: {'Success' if backward_signal is not None else 'Failed'}")
        
        logger.info("All examples completed successfully!")
        
        return {
            'basic_combination': (basic_signal, basic_metadata),
            'modes_results': modes_results,
            'weighted_results': weighted_results,
            'integration_results': integration_results,
            'backward_compatibility': backward_signal
        }
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    results = main()
    print("\nExample execution completed!") 