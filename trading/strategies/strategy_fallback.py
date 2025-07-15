"""
Strategy Fallback - Batch 18
Enhanced strategy fallback with ranked fallback pool based on historical performance
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class FallbackStrategy(Enum):
    """Available fallback strategies."""
    RSI = "RSI"
    SMA = "SMA"
    MACD = "MACD"
    BOLLINGER = "Bollinger"
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "MeanReversion"

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_name: str
    win_rate: float
    total_trades: int
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime
    confidence_score: float = 0.0

@dataclass
class FallbackResult:
    """Result of fallback strategy execution."""
    strategy_name: str
    signal: str
    confidence: float
    performance_metrics: Dict[str, float]
    execution_time: float
    fallback_rank: int
    timestamp: datetime = field(default_factory=datetime.now)

class StrategyFallback:
    """
    Enhanced strategy fallback with ranked fallback pool.
    
    Features:
    - Ranked fallback strategies based on historical win rates
    - Dynamic performance tracking
    - Multiple fallback strategies (RSI, SMA, MACD)
    - Performance-based strategy selection
    """
    
    def __init__(self, 
                 fallback_pool: Optional[List[str]] = None,
                 performance_window: int = 30,
                 min_trades_for_ranking: int = 5):
        """
        Initialize strategy fallback.
        
        Args:
            fallback_pool: List of fallback strategy names
            performance_window: Days to look back for performance
            min_trades_for_ranking: Minimum trades required for ranking
        """
        self.fallback_pool = fallback_pool or ["RSI", "SMA", "MACD"]
        self.performance_window = performance_window
        self.min_trades_for_ranking = min_trades_for_ranking
        
        # Performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Initialize default performance for fallback strategies
        self._initialize_default_performance()
        
        logger.info(f"StrategyFallback initialized with pool: {self.fallback_pool}")
    
    def _initialize_default_performance(self):
        """Initialize default performance metrics for fallback strategies."""
        default_performance = {
            "RSI": {"win_rate": 0.55, "avg_return": 0.02, "sharpe": 0.8},
            "SMA": {"win_rate": 0.52, "avg_return": 0.015, "sharpe": 0.6},
            "MACD": {"win_rate": 0.58, "avg_return": 0.025, "sharpe": 0.9},
            "Bollinger": {"win_rate": 0.53, "avg_return": 0.018, "sharpe": 0.7},
            "Momentum": {"win_rate": 0.56, "avg_return": 0.022, "sharpe": 0.85},
            "MeanReversion": {"win_rate": 0.54, "avg_return": 0.019, "sharpe": 0.75}
        }
        
        for strategy_name in self.fallback_pool:
            if strategy_name in default_performance:
                perf = default_performance[strategy_name]
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    win_rate=perf["win_rate"],
                    total_trades=10,  # Default trade count
                    avg_return=perf["avg_return"],
                    sharpe_ratio=perf["sharpe"],
                    max_drawdown=0.05,
                    last_updated=datetime.now(),
                    confidence_score=0.7
                )
            else:
                # Initialize with conservative defaults
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    win_rate=0.5,
                    total_trades=0,
                    avg_return=0.01,
                    sharpe_ratio=0.5,
                    max_drawdown=0.1,
                    last_updated=datetime.now(),
                    confidence_score=0.5
                )
    
    def get_ranked_fallbacks(self) -> List[Tuple[str, float]]:
        """
        Get ranked fallback strategies based on historical performance.
        
        Returns:
            List of (strategy_name, score) tuples ranked by performance
        """
        ranked_strategies = []
        
        for strategy_name in self.fallback_pool:
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                
                # Calculate composite score
                score = self._calculate_strategy_score(perf)
                ranked_strategies.append((strategy_name, score))
        
        # Sort by score (descending)
        ranked_strategies.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Ranked fallbacks: {ranked_strategies}")
        return ranked_strategies
    
    def _calculate_strategy_score(self, performance: StrategyPerformance) -> float:
        """
        Calculate composite score for strategy ranking.
        
        Args:
            performance: Strategy performance metrics
            
        Returns:
            Composite score (higher is better)
        """
        # Weighted combination of metrics
        win_rate_weight = 0.4
        sharpe_weight = 0.3
        return_weight = 0.2
        confidence_weight = 0.1
        
        # Normalize metrics
        normalized_win_rate = min(performance.win_rate, 0.8) / 0.8  # Cap at 80%
        normalized_sharpe = max(0, min(performance.sharpe_ratio, 2.0)) / 2.0
        normalized_return = max(0, min(performance.avg_return, 0.05)) / 0.05
        
        # Penalty for insufficient data
        data_penalty = 1.0
        if performance.total_trades < self.min_trades_for_ranking:
            data_penalty = 0.7
        
        score = (
            win_rate_weight * normalized_win_rate +
            sharpe_weight * normalized_sharpe +
            return_weight * normalized_return +
            confidence_weight * performance.confidence_score
        ) * data_penalty
        
        return score
    
    def execute_fallback(self, 
                        market_data: pd.DataFrame,
                        context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """
        Execute the best fallback strategy.
        
        Args:
            market_data: Market data for strategy execution
            context: Additional context
            
        Returns:
            FallbackResult with execution details
        """
        start_time = datetime.now()
        
        # Get ranked fallbacks
        ranked_fallbacks = self.get_ranked_fallbacks()
        
        if not ranked_fallbacks:
            logger.error("No fallback strategies available")
            return self._create_error_result("No fallbacks available")
        
        # Try strategies in order of ranking
        for rank, (strategy_name, score) in enumerate(ranked_fallbacks):
            try:
                result = self._execute_strategy(strategy_name, market_data, context)
                if result:
                    result.fallback_rank = rank + 1
                    execution_time = (datetime.now() - start_time).total_seconds()
                    result.execution_time = execution_time
                    
                    logger.info(f"Executed fallback strategy: {strategy_name} (rank {rank + 1}, score: {score:.3f})")
                    return result
                    
            except Exception as e:
                logger.warning(f"Failed to execute fallback strategy {strategy_name}: {e}")
                continue
        
        # If all strategies fail, return error result
        logger.error("All fallback strategies failed")
        return self._create_error_result("All fallbacks failed")
    
    def _execute_strategy(self, 
                         strategy_name: str, 
                         market_data: pd.DataFrame,
                         context: Optional[Dict[str, Any]] = None) -> Optional[FallbackResult]:
        """
        Execute a specific fallback strategy.
        
        Args:
            strategy_name: Name of the strategy
            market_data: Market data
            context: Execution context
            
        Returns:
            FallbackResult or None if execution fails
        """
        try:
            if strategy_name == "RSI":
                return self._execute_rsi_strategy(market_data, context)
            elif strategy_name == "SMA":
                return self._execute_sma_strategy(market_data, context)
            elif strategy_name == "MACD":
                return self._execute_macd_strategy(market_data, context)
            elif strategy_name == "Bollinger":
                return self._execute_bollinger_strategy(market_data, context)
            elif strategy_name == "Momentum":
                return self._execute_momentum_strategy(market_data, context)
            elif strategy_name == "MeanReversion":
                return self._execute_mean_reversion_strategy(market_data, context)
            else:
                logger.warning(f"Unknown fallback strategy: {strategy_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing {strategy_name}: {e}")
            return None
    
    def _execute_rsi_strategy(self, 
                            market_data: pd.DataFrame,
                            context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """Execute RSI strategy."""
        # Simulate RSI calculation
        rsi_value = context.get('rsi', 50) if context else 50
        
        if rsi_value < 30:
            signal = "BUY"
            confidence = 0.8
        elif rsi_value > 70:
            signal = "SELL"
            confidence = 0.8
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return FallbackResult(
            strategy_name="RSI",
            signal=signal,
            confidence=confidence,
            performance_metrics={"rsi_value": rsi_value},
            execution_time=0.0,
            fallback_rank=0
        )
    
    def _execute_sma_strategy(self, 
                            market_data: pd.DataFrame,
                            context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """Execute SMA strategy."""
        # Simulate SMA calculation
        current_price = context.get('current_price', 100) if context else 100
        sma_short = context.get('sma_short', 98) if context else 98
        sma_long = context.get('sma_long', 102) if context else 102
        
        if current_price > sma_short > sma_long:
            signal = "BUY"
            confidence = 0.7
        elif current_price < sma_short < sma_long:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "HOLD"
            confidence = 0.4
        
        return FallbackResult(
            strategy_name="SMA",
            signal=signal,
            confidence=confidence,
            performance_metrics={"current_price": current_price, "sma_short": sma_short, "sma_long": sma_long},
            execution_time=0.0,
            fallback_rank=0
        )
    
    def _execute_macd_strategy(self, 
                             market_data: pd.DataFrame,
                             context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """Execute MACD strategy."""
        # Simulate MACD calculation
        macd_line = context.get('macd_line', 0.5) if context else 0.5
        signal_line = context.get('signal_line', 0.3) if context else 0.3
        
        if macd_line > signal_line and macd_line > 0:
            signal = "BUY"
            confidence = 0.75
        elif macd_line < signal_line and macd_line < 0:
            signal = "SELL"
            confidence = 0.75
        else:
            signal = "HOLD"
            confidence = 0.4
        
        return FallbackResult(
            strategy_name="MACD",
            signal=signal,
            confidence=confidence,
            performance_metrics={"macd_line": macd_line, "signal_line": signal_line},
            execution_time=0.0,
            fallback_rank=0
        )
    
    def _execute_bollinger_strategy(self, 
                                  market_data: pd.DataFrame,
                                  context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """Execute Bollinger Bands strategy."""
        # Simulate Bollinger Bands calculation
        current_price = context.get('current_price', 100) if context else 100
        upper_band = context.get('upper_band', 105) if context else 105
        lower_band = context.get('lower_band', 95) if context else 95
        
        if current_price <= lower_band:
            signal = "BUY"
            confidence = 0.8
        elif current_price >= upper_band:
            signal = "SELL"
            confidence = 0.8
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return FallbackResult(
            strategy_name="Bollinger",
            signal=signal,
            confidence=confidence,
            performance_metrics={"current_price": current_price, "upper_band": upper_band, "lower_band": lower_band},
            execution_time=0.0,
            fallback_rank=0
        )
    
    def _execute_momentum_strategy(self, 
                                 market_data: pd.DataFrame,
                                 context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """Execute Momentum strategy."""
        # Simulate momentum calculation
        momentum = context.get('momentum', 0.02) if context else 0.02
        
        if momentum > 0.01:
            signal = "BUY"
            confidence = 0.7
        elif momentum < -0.01:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "HOLD"
            confidence = 0.4
        
        return FallbackResult(
            strategy_name="Momentum",
            signal=signal,
            confidence=confidence,
            performance_metrics={"momentum": momentum},
            execution_time=0.0,
            fallback_rank=0
        )
    
    def _execute_mean_reversion_strategy(self, 
                                       market_data: pd.DataFrame,
                                       context: Optional[Dict[str, Any]] = None) -> FallbackResult:
        """Execute Mean Reversion strategy."""
        # Simulate mean reversion calculation
        deviation = context.get('deviation', 0.05) if context else 0.05
        
        if deviation > 0.1:
            signal = "SELL"
            confidence = 0.7
        elif deviation < -0.1:
            signal = "BUY"
            confidence = 0.7
        else:
            signal = "HOLD"
            confidence = 0.4
        
        return FallbackResult(
            strategy_name="MeanReversion",
            signal=signal,
            confidence=confidence,
            performance_metrics={"deviation": deviation},
            execution_time=0.0,
            fallback_rank=0
        )
    
    def _create_error_result(self, error_message: str) -> FallbackResult:
        """Create error result."""
        return FallbackResult(
            strategy_name="ERROR",
            signal="HOLD",
            confidence=0.0,
            performance_metrics={"error": error_message},
            execution_time=0.0,
            fallback_rank=0
        )
    
    def update_strategy_performance(self, 
                                  strategy_name: str,
                                  trade_result: Dict[str, Any]):
        """
        Update strategy performance with trade result.
        
        Args:
            strategy_name: Name of the strategy
            trade_result: Trade result data
        """
        if strategy_name not in self.strategy_performance:
            return
        
        # Add to trade history
        trade_record = {
            'strategy_name': strategy_name,
            'timestamp': datetime.now().isoformat(),
            'result': trade_result
        }
        self.trade_history.append(trade_record)
        
        # Update performance metrics
        self._recalculate_performance(strategy_name)
        
        logger.debug(f"Updated performance for {strategy_name}")
    
    def _recalculate_performance(self, strategy_name: str):
        """Recalculate performance metrics for a strategy."""
        # Get recent trades for this strategy
        recent_trades = [
            trade for trade in self.trade_history[-100:]  # Last 100 trades
            if trade['strategy_name'] == strategy_name
        ]
        
        if len(recent_trades) < 3:
            return
        
        # Calculate metrics
        wins = sum(1 for trade in recent_trades if trade['result'].get('pnl', 0) > 0)
        total_trades = len(recent_trades)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        returns = [trade['result'].get('pnl', 0) for trade in recent_trades]
        avg_return = np.mean(returns) if returns else 0.0
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0
        
        # Update performance
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            perf.win_rate = win_rate
            perf.total_trades = total_trades
            perf.avg_return = avg_return
            perf.sharpe_ratio = sharpe_ratio
            perf.last_updated = datetime.now()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies."""
        summary = {
            'total_strategies': len(self.strategy_performance),
            'ranked_strategies': self.get_ranked_fallbacks(),
            'performance_metrics': {}
        }
        
        for strategy_name, perf in self.strategy_performance.items():
            summary['performance_metrics'][strategy_name] = {
                'win_rate': perf.win_rate,
                'total_trades': perf.total_trades,
                'avg_return': perf.avg_return,
                'sharpe_ratio': perf.sharpe_ratio,
                'last_updated': perf.last_updated.isoformat()
            }
        
        return summary

def create_strategy_fallback(fallback_pool: Optional[List[str]] = None) -> StrategyFallback:
    """Factory function to create strategy fallback."""
    return StrategyFallback(fallback_pool=fallback_pool) 