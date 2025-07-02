"""
Strategy Comparison and Stacking Module

This module provides comprehensive strategy comparison and stacking capabilities
for the Evolve trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json

from trading.utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from trading.strategies.registry import RSIStrategy, MACDStrategy, BollingerBandsStrategy

logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_name: str
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_return: float
    volatility: float
    profit_factor: float
    avg_trade: float
    num_trades: int
    avg_holding_period: float
    timestamp: datetime

@dataclass
class StrategyComparison:
    """Comparison results between strategies."""
    strategy_a: str
    strategy_b: str
    sharpe_diff: float
    win_rate_diff: float
    drawdown_diff: float
    return_diff: float
    correlation: float
    combined_sharpe: float
    timestamp: datetime

class StrategyComparisonMatrix:
    """Multi-strategy comparison matrix."""
    
    def __init__(self):
        """Initialize the strategy comparison matrix."""
        self.strategies = {
            'rsi': RSIStrategy(),
            'macd': MACDStrategy(),
            'bollinger': BollingerBandsStrategy()
        }
        self.comparison_history = []
        self.performance_cache = {}
        
    def generate_comparison_matrix(self, 
                                 data: pd.DataFrame,
                                 strategies: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate a comprehensive strategy comparison matrix.
        
        Args:
            data: Market data
            strategies: List of strategy names to compare (default: all)
            
        Returns:
            DataFrame with comparison matrix
        """
        try:
            if strategies is None:
                strategies = list(self.strategies.keys())
            
            # Calculate performance for each strategy
            performances = {}
            for strategy_name in strategies:
                if strategy_name in self.strategies:
                    performance = self._calculate_strategy_performance(
                        strategy_name, data
                    )
                    performances[strategy_name] = performance
            
            # Create comparison matrix
            matrix_data = []
            metrics = ['sharpe_ratio', 'win_rate', 'max_drawdown', 'total_return', 
                      'volatility', 'profit_factor']
            
            for strategy_name, performance in performances.items():
                row = {'Strategy': strategy_name}
                for metric in metrics:
                    row[metric] = getattr(performance, metric, 0.0)
                matrix_data.append(row)
            
            matrix_df = pd.DataFrame(matrix_data)
            
            # Add ranking columns
            for metric in metrics:
                matrix_df[f'{metric}_rank'] = matrix_df[metric].rank(ascending=False)
            
            # Cache results
            self.performance_cache = performances
            
            return matrix_df
            
        except Exception as e:
            logger.error(f"Error generating comparison matrix: {e}")
            return pd.DataFrame()
    
    def _calculate_strategy_performance(self, 
                                      strategy_name: str, 
                                      data: pd.DataFrame) -> StrategyPerformance:
        """Calculate performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            data: Market data
            
        Returns:
            StrategyPerformance object
        """
        try:
            strategy = self.strategies[strategy_name]
            signals = strategy.generate_signals(data)
            
            # Calculate returns
            returns = self._calculate_strategy_returns(data, signals)
            
            # Calculate metrics
            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(returns)
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate trade-based metrics
            trade_metrics = self._calculate_trade_metrics(returns, signals)
            
            return StrategyPerformance(
                strategy_name=strategy_name,
                sharpe_ratio=sharpe,
                win_rate=trade_metrics['win_rate'],
                max_drawdown=max_dd,
                total_return=total_return,
                volatility=volatility,
                profit_factor=trade_metrics['profit_factor'],
                avg_trade=trade_metrics['avg_trade'],
                num_trades=trade_metrics['num_trades'],
                avg_holding_period=trade_metrics['avg_holding_period'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance for {strategy_name}: {e}")
            return StrategyPerformance(
                strategy_name=strategy_name,
                sharpe_ratio=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                total_return=0.0,
                volatility=0.0,
                profit_factor=0.0,
                avg_trade=0.0,
                num_trades=0,
                avg_holding_period=0.0,
                timestamp=datetime.now()
            )
    
    def _calculate_strategy_returns(self, 
                                  data: pd.DataFrame, 
                                  signals: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from signals.
        
        Args:
            data: Market data
            signals: Strategy signals
            
        Returns:
            Series of returns
        """
        try:
            # Calculate price returns
            price_returns = data['Close'].pct_change()
            
            # Apply signals (1 = buy, -1 = sell, 0 = hold)
            strategy_returns = signals['signal'].shift(1) * price_returns
            
            # Remove NaN values
            strategy_returns = strategy_returns.dropna()
            
            return strategy_returns
            
        except Exception as e:
            logger.error(f"Error calculating strategy returns: {e}")
            return pd.Series([0.0] * len(data))
    
    def _calculate_trade_metrics(self, 
                               returns: pd.Series, 
                               signals: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based performance metrics.
        
        Args:
            returns: Strategy returns
            signals: Strategy signals
            
        Returns:
            Dictionary of trade metrics
        """
        try:
            # Identify trades
            signal_changes = signals['signal'].diff().fillna(0)
            trade_starts = signal_changes != 0
            
            if not trade_starts.any():
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_trade': 0.0,
                    'num_trades': 0,
                    'avg_holding_period': 0.0
                }
            
            # Calculate trade returns
            trade_returns = []
            holding_periods = []
            
            start_idx = None
            for i, is_start in enumerate(trade_starts):
                if is_start:
                    if start_idx is not None:
                        # End previous trade
                        trade_return = (1 + returns.iloc[start_idx:i]).prod() - 1
                        trade_returns.append(trade_return)
                        holding_periods.append(i - start_idx)
                    start_idx = i
            
            # End last trade if still open
            if start_idx is not None and start_idx < len(returns):
                trade_return = (1 + returns.iloc[start_idx:]).prod() - 1
                trade_returns.append(trade_return)
                holding_periods.append(len(returns) - start_idx)
            
            if not trade_returns:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_trade': 0.0,
                    'num_trades': 0,
                    'avg_holding_period': 0.0
                }
            
            # Calculate metrics
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]
            
            win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0.0
            
            if losing_trades:
                profit_factor = abs(sum(winning_trades)) / abs(sum(losing_trades))
            else:
                profit_factor = float('inf') if winning_trades else 0.0
            
            avg_trade = np.mean(trade_returns)
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0.0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade': avg_trade,
                'num_trades': len(trade_returns),
                'avg_holding_period': avg_holding_period
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'num_trades': 0,
                'avg_holding_period': 0.0
            }
    
    def get_best_strategy(self, 
                         data: pd.DataFrame, 
                         metric: str = 'sharpe_ratio') -> Tuple[str, float]:
        """Get the best performing strategy based on a metric.
        
        Args:
            data: Market data
            metric: Performance metric to optimize
            
        Returns:
            Tuple of (strategy_name, metric_value)
        """
        try:
            matrix = self.generate_comparison_matrix(data)
            
            if matrix.empty:
                return None, 0.0
            
            best_idx = matrix[metric].idxmax()
            best_strategy = matrix.loc[best_idx, 'Strategy']
            best_value = matrix.loc[best_idx, metric]
            
            return best_strategy, best_value
            
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            return None, 0.0
    
    def compare_strategies(self, 
                          strategy_a: str, 
                          strategy_b: str, 
                          data: pd.DataFrame) -> StrategyComparison:
        """Compare two specific strategies.
        
        Args:
            strategy_a: First strategy name
            strategy_b: Second strategy name
            data: Market data
            
        Returns:
            StrategyComparison object
        """
        try:
            # Calculate performance for both strategies
            perf_a = self._calculate_strategy_performance(strategy_a, data)
            perf_b = self._calculate_strategy_performance(strategy_b, data)
            
            # Calculate returns for correlation
            returns_a = self._calculate_strategy_returns(data, 
                self.strategies[strategy_a].generate_signals(data))
            returns_b = self._calculate_strategy_returns(data, 
                self.strategies[strategy_b].generate_signals(data))
            
            # Align returns
            aligned_returns = pd.concat([returns_a, returns_b], axis=1).dropna()
            correlation = aligned_returns.corr().iloc[0, 1] if len(aligned_returns) > 1 else 0.0
            
            # Calculate combined performance (equal weight)
            combined_returns = (aligned_returns.iloc[:, 0] + aligned_returns.iloc[:, 1]) / 2
            combined_sharpe = calculate_sharpe_ratio(combined_returns)
            
            comparison = StrategyComparison(
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                sharpe_diff=perf_a.sharpe_ratio - perf_b.sharpe_ratio,
                win_rate_diff=perf_a.win_rate - perf_b.win_rate,
                drawdown_diff=perf_a.max_drawdown - perf_b.max_drawdown,
                return_diff=perf_a.total_return - perf_b.total_return,
                correlation=correlation,
                combined_sharpe=combined_sharpe,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.comparison_history.append(comparison)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            return None

class StrategyStacker:
    """Strategy stacking implementation."""
    
    def __init__(self, max_strategies: int = 3):
        """Initialize strategy stacker.
        
        Args:
            max_strategies: Maximum number of strategies to stack
        """
        self.max_strategies = max_strategies
        self.comparison_matrix = StrategyComparisonMatrix()
        self.stacking_history = []
        
    def create_strategy_stack(self, 
                            data: pd.DataFrame,
                            strategy_names: Optional[List[str]] = None,
                            method: str = 'performance_weighted') -> Dict[str, Any]:
        """Create a stacked strategy combining multiple strategies.
        
        Args:
            data: Market data
            strategy_names: List of strategies to stack (default: best performers)
            method: Stacking method ('equal', 'performance_weighted', 'correlation_weighted')
            
        Returns:
            Dictionary with stacked strategy results
        """
        try:
            if strategy_names is None:
                # Select best strategies based on Sharpe ratio
                matrix = self.comparison_matrix.generate_comparison_matrix(data)
                if matrix.empty:
                    return self._create_fallback_stack(data)
                
                # Get top strategies
                top_strategies = matrix.nlargest(self.max_strategies, 'sharpe_ratio')['Strategy'].tolist()
                strategy_names = top_strategies
            
            # Limit to max strategies
            strategy_names = strategy_names[:self.max_strategies]
            
            # Generate signals for each strategy
            strategy_signals = {}
            strategy_returns = {}
            
            for strategy_name in strategy_names:
                if strategy_name in self.comparison_matrix.strategies:
                    strategy = self.comparison_matrix.strategies[strategy_name]
                    signals = strategy.generate_signals(data)
                    returns = self.comparison_matrix._calculate_strategy_returns(data, signals)
                    
                    strategy_signals[strategy_name] = signals
                    strategy_returns[strategy_name] = returns
            
            # Calculate weights based on method
            weights = self._calculate_stack_weights(strategy_returns, method)
            
            # Create stacked signals
            stacked_signals = self._combine_signals(strategy_signals, weights)
            
            # Calculate stacked performance
            stacked_returns = self.comparison_matrix._calculate_strategy_returns(data, stacked_signals)
            stacked_performance = self.comparison_matrix._calculate_strategy_performance(
                'stacked', data
            )
            
            # Store stacking result
            stacking_result = {
                'strategies': strategy_names,
                'weights': weights,
                'method': method,
                'performance': stacked_performance,
                'signals': stacked_signals,
                'returns': stacked_returns,
                'timestamp': datetime.now()
            }
            
            self.stacking_history.append(stacking_result)
            
            return stacking_result
            
        except Exception as e:
            logger.error(f"Error creating strategy stack: {e}")
            return self._create_fallback_stack(data)
    
    def _calculate_stack_weights(self, 
                               strategy_returns: Dict[str, pd.Series], 
                               method: str) -> Dict[str, float]:
        """Calculate weights for strategy stacking.
        
        Args:
            strategy_returns: Dictionary of strategy returns
            method: Weighting method
            
        Returns:
            Dictionary of strategy weights
        """
        try:
            if method == 'equal':
                # Equal weights
                n_strategies = len(strategy_returns)
                return {name: 1.0 / n_strategies for name in strategy_returns.keys()}
            
            elif method == 'performance_weighted':
                # Weight by Sharpe ratio
                sharpe_ratios = {}
                for name, returns in strategy_returns.items():
                    sharpe_ratios[name] = calculate_sharpe_ratio(returns)
                
                total_sharpe = sum(max(0, sr) for sr in sharpe_ratios.values())
                if total_sharpe > 0:
                    return {name: max(0, sr) / total_sharpe 
                           for name, sr in sharpe_ratios.items()}
                else:
                    return {name: 1.0 / len(strategy_returns) 
                           for name in strategy_returns.keys()}
            
            elif method == 'correlation_weighted':
                # Weight by inverse correlation
                returns_df = pd.DataFrame(strategy_returns)
                corr_matrix = returns_df.corr()
                
                # Calculate average correlation for each strategy
                avg_correlations = {}
                for strategy in strategy_returns.keys():
                    correlations = corr_matrix[strategy].drop(strategy)
                    avg_correlations[strategy] = correlations.mean()
                
                # Weight inversely to correlation
                inverse_correlations = {name: 1 - corr 
                                      for name, corr in avg_correlations.items()}
                total_inverse = sum(inverse_correlations.values())
                
                if total_inverse > 0:
                    return {name: inv_corr / total_inverse 
                           for name, inv_corr in inverse_correlations.items()}
                else:
                    return {name: 1.0 / len(strategy_returns) 
                           for name in strategy_returns.keys()}
            
            else:
                # Default to equal weights
                n_strategies = len(strategy_returns)
                return {name: 1.0 / n_strategies for name in strategy_returns.keys()}
                
        except Exception as e:
            logger.error(f"Error calculating stack weights: {e}")
            n_strategies = len(strategy_returns)
            return {name: 1.0 / n_strategies for name in strategy_returns.keys()}
    
    def _combine_signals(self, 
                        strategy_signals: Dict[str, pd.DataFrame], 
                        weights: Dict[str, float]) -> pd.DataFrame:
        """Combine signals from multiple strategies.
        
        Args:
            strategy_signals: Dictionary of strategy signals
            weights: Strategy weights
            
        Returns:
            Combined signals DataFrame
        """
        try:
            # Initialize combined signals
            combined_signals = pd.DataFrame(index=next(iter(strategy_signals.values())).index)
            combined_signals['signal'] = 0.0
            
            # Weight and combine signals
            for strategy_name, signals in strategy_signals.items():
                weight = weights.get(strategy_name, 0.0)
                combined_signals['signal'] += weight * signals['signal']
            
            # Convert to discrete signals
            combined_signals['signal'] = np.where(combined_signals['signal'] > 0.5, 1,
                                                np.where(combined_signals['signal'] < -0.5, -1, 0))
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            # Return neutral signals
            return pd.DataFrame({'signal': 0}, index=next(iter(strategy_signals.values())).index)
    
    def _create_fallback_stack(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create a fallback stacked strategy when stacking fails.
        
        Args:
            data: Market data
            
        Returns:
            Fallback stacking result
        """
        try:
            # Use simple RSI strategy as fallback
            rsi_strategy = RSIStrategy()
            signals = rsi_strategy.generate_signals(data)
            returns = self.comparison_matrix._calculate_strategy_returns(data, signals)
            
            return {
                'strategies': ['rsi_fallback'],
                'weights': {'rsi_fallback': 1.0},
                'method': 'fallback',
                'performance': self.comparison_matrix._calculate_strategy_performance(
                    'rsi_fallback', data
                ),
                'signals': signals,
                'returns': returns,
                'timestamp': datetime.now(),
                'warning': 'Fallback strategy used due to stacking errors'
            }
            
        except Exception as e:
            logger.error(f"Error creating fallback stack: {e}")
            return {
                'strategies': [],
                'weights': {},
                'method': 'error',
                'performance': None,
                'signals': pd.DataFrame(),
                'returns': pd.Series(),
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def get_stacking_summary(self) -> Dict[str, Any]:
        """Get summary of stacking history.
        
        Returns:
            Dictionary with stacking summary
        """
        try:
            if not self.stacking_history:
                return {'message': 'No stacking history available'}
            
            recent_stacks = self.stacking_history[-10:]  # Last 10 stacks
            
            summary = {
                'total_stacks': len(self.stacking_history),
                'recent_stacks': len(recent_stacks),
                'methods_used': list(set(stack['method'] for stack in recent_stacks)),
                'avg_strategies_per_stack': np.mean([
                    len(stack['strategies']) for stack in recent_stacks
                ]),
                'best_performing_method': self._get_best_method(recent_stacks),
                'last_stack_timestamp': recent_stacks[-1]['timestamp'] if recent_stacks else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting stacking summary: {e}")
            return {'error': str(e)}
    
    def _get_best_method(self, stacks: List[Dict[str, Any]]) -> str:
        """Get the best performing stacking method.
        
        Args:
            stacks: List of stacking results
            
        Returns:
            Best method name
        """
        try:
            method_performance = {}
            
            for stack in stacks:
                method = stack['method']
                if stack['performance'] and hasattr(stack['performance'], 'sharpe_ratio'):
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(stack['performance'].sharpe_ratio)
            
            if not method_performance:
                return 'unknown'
            
            # Calculate average Sharpe ratio for each method
            avg_performance = {
                method: np.mean(perfs) for method, perfs in method_performance.items()
            }
            
            return max(avg_performance, key=avg_performance.get)
            
        except Exception as e:
            logger.error(f"Error getting best method: {e}")
            return 'unknown'

# Global instances
strategy_comparison = StrategyComparisonMatrix()
strategy_stacker = StrategyStacker()

def get_strategy_comparison() -> StrategyComparisonMatrix:
    """Get the global strategy comparison instance."""
    return strategy_comparison

def get_strategy_stacker() -> StrategyStacker:
    """Get the global strategy stacker instance."""
    return strategy_stacker 