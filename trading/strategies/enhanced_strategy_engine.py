"""
Enhanced Strategy Engine for Evolve Trading Platform

This module provides institutional-level strategy capabilities:
- Dynamic strategy chaining based on market regime
- Automatic strategy combination and optimization
- Continuous performance monitoring and improvement
- Meta-agent loop for strategy retirement and tuning
- Confidence scoring and edge calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    VOLATILITY = "volatility"
    DEFENSIVE = "defensive"
    CASH_HEAVY = "cash_heavy"
    OPTIONS_INCOME = "options_income"
    LEVERAGE = "leverage"
    SHORT_MOMENTUM = "short_momentum"
    RANGE_TRADING = "range_trading"
    FALLBACK = "fallback"

class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    NORMAL = "normal"

@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    confidence: float
    expected_sharpe: float
    max_drawdown: float
    win_rate: float
    regime_compatibility: List[MarketRegime]
    risk_level: str  # low, medium, high
    min_volatility: float
    max_volatility: float

@dataclass
class StrategyResult:
    """Strategy execution result."""
    strategy_name: str
    signals: pd.DataFrame
    performance: Dict[str, float]
    confidence: float
    regime: MarketRegime
    parameters_used: Dict[str, Any]
    execution_time: float
    timestamp: datetime

class EnhancedStrategyEngine:
    """Enhanced strategy engine with institutional-level capabilities."""
    
    def __init__(self):
        """Initialize the enhanced strategy engine."""
        self.strategies = self._initialize_strategies()
        self.performance_history = []
        self.strategy_weights = {}
        self.regime_classifier = None
        self.meta_agent = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Enhanced Strategy Engine initialized")
    
    def _initialize_components(self):
        """Initialize strategy engine components."""
        try:
            # Initialize regime classifier
            from trading.agents.market_regime_agent import MarketRegimeAgent
            self.regime_classifier = MarketRegimeAgent()
        except ImportError:
            logger.warning("MarketRegimeAgent not available - using fallback")
            self.regime_classifier = self._create_fallback_regime_classifier()
        
        try:
            # Initialize meta-agent for continuous improvement
            from trading.meta_agents.agents.performance_checker import PerformanceChecker
            self.meta_agent = PerformanceChecker()
        except ImportError:
            logger.warning("PerformanceChecker not available - using fallback")
            self.meta_agent = self._create_fallback_meta_agent()
    
    def _create_fallback_regime_classifier(self):
        """Create fallback regime classifier."""
        class FallbackRegimeClassifier:
            def classify_regime(self, data: pd.DataFrame) -> MarketRegime:
                # Simple regime classification based on returns
                returns = data['Close'].pct_change().dropna()
                mean_return = returns.mean()
                volatility = returns.std()
                
                if mean_return > 0.001 and volatility < 0.02:
                    return MarketRegime.BULL
                elif mean_return < -0.001 and volatility < 0.02:
                    return MarketRegime.BEAR
                elif volatility > 0.03:
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.SIDEWAYS
            
            def get_regime_confidence(self) -> float:
                return 0.7
        
        return FallbackRegimeClassifier()
    
    def _create_fallback_meta_agent(self):
        """Create fallback meta-agent."""
        class FallbackMetaAgent:
            def check_strategy_performance(self, strategy_name: str, performance: Dict[str, float]) -> Dict[str, Any]:
                return {
                    'should_retire': False,
                    'should_tune': False,
                    'confidence': 0.5,
                    'recommendations': []
                }
            
            def suggest_improvements(self, strategy_name: str, performance: Dict[str, float]) -> List[str]:
                return ["Consider parameter tuning", "Monitor performance closely"]
        
        return FallbackMetaAgent()
    
    def _initialize_strategies(self) -> Dict[str, StrategyConfig]:
        """Initialize strategy configurations."""
        strategies = {
            'momentum': StrategyConfig(
                name='Momentum',
                strategy_type=StrategyType.MOMENTUM,
                parameters={'lookback': 20, 'threshold': 0.02},
                confidence=0.8,
                expected_sharpe=1.2,
                max_drawdown=0.15,
                win_rate=0.65,
                regime_compatibility=[MarketRegime.BULL, MarketRegime.VOLATILE],
                risk_level='medium',
                min_volatility=0.01,
                max_volatility=0.05
            ),
            'mean_reversion': StrategyConfig(
                name='Mean Reversion',
                strategy_type=StrategyType.MEAN_REVERSION,
                parameters={'lookback': 50, 'std_threshold': 2.0},
                confidence=0.75,
                expected_sharpe=0.9,
                max_drawdown=0.12,
                win_rate=0.58,
                regime_compatibility=[MarketRegime.SIDEWAYS, MarketRegime.NORMAL],
                risk_level='low',
                min_volatility=0.005,
                max_volatility=0.03
            ),
            'breakout': StrategyConfig(
                name='Breakout',
                strategy_type=StrategyType.BREAKOUT,
                parameters={'lookback': 30, 'breakout_threshold': 0.03},
                confidence=0.7,
                expected_sharpe=1.0,
                max_drawdown=0.18,
                win_rate=0.45,
                regime_compatibility=[MarketRegime.BULL, MarketRegime.VOLATILE],
                risk_level='high',
                min_volatility=0.02,
                max_volatility=0.06
            ),
            'trend_following': StrategyConfig(
                name='Trend Following',
                strategy_type=StrategyType.TREND_FOLLOWING,
                parameters={'short_window': 10, 'long_window': 50},
                confidence=0.85,
                expected_sharpe=1.1,
                max_drawdown=0.14,
                win_rate=0.62,
                regime_compatibility=[MarketRegime.BULL, MarketRegime.NORMAL],
                risk_level='medium',
                min_volatility=0.01,
                max_volatility=0.04
            ),
            'volatility_trading': StrategyConfig(
                name='Volatility Trading',
                strategy_type=StrategyType.VOLATILITY,
                parameters={'vol_window': 20, 'vol_threshold': 0.025},
                confidence=0.65,
                expected_sharpe=0.8,
                max_drawdown=0.20,
                win_rate=0.40,
                regime_compatibility=[MarketRegime.VOLATILE],
                risk_level='high',
                min_volatility=0.03,
                max_volatility=0.08
            ),
            'defensive': StrategyConfig(
                name='Defensive',
                strategy_type=StrategyType.DEFENSIVE,
                parameters={'stop_loss': 0.05, 'position_size': 0.1},
                confidence=0.9,
                expected_sharpe=0.6,
                max_drawdown=0.08,
                win_rate=0.70,
                regime_compatibility=[MarketRegime.BEAR, MarketRegime.VOLATILE],
                risk_level='low',
                min_volatility=0.005,
                max_volatility=0.04
            ),
            'cash_heavy': StrategyConfig(
                name='Cash Heavy',
                strategy_type=StrategyType.CASH_HEAVY,
                parameters={'cash_allocation': 0.8, 'bond_allocation': 0.2},
                confidence=0.95,
                expected_sharpe=0.4,
                max_drawdown=0.05,
                win_rate=0.80,
                regime_compatibility=[MarketRegime.BEAR, MarketRegime.VOLATILE],
                risk_level='low',
                min_volatility=0.001,
                max_volatility=0.02
            ),
            'options_income': StrategyConfig(
                name='Options Income',
                strategy_type=StrategyType.OPTIONS_INCOME,
                parameters={'delta_target': 0.3, 'days_to_expiry': 30},
                confidence=0.6,
                expected_sharpe=0.7,
                max_drawdown=0.25,
                win_rate=0.55,
                regime_compatibility=[MarketRegime.SIDEWAYS, MarketRegime.NORMAL],
                risk_level='medium',
                min_volatility=0.01,
                max_volatility=0.05
            )
        }
        
        return strategies
    
    def get_strategy_chain(self, regime: MarketRegime, risk_tolerance: str) -> List[Dict[str, Any]]:
        """Get dynamic strategy chain based on regime and risk tolerance."""
        compatible_strategies = []
        
        for name, config in self.strategies.items():
            if regime in config.regime_compatibility:
                # Check risk level compatibility
                risk_compatible = (
                    (risk_tolerance == 'low' and config.risk_level in ['low']) or
                    (risk_tolerance == 'medium' and config.risk_level in ['low', 'medium']) or
                    (risk_tolerance == 'high' and config.risk_level in ['low', 'medium', 'high'])
                )
                
                if risk_compatible:
                    compatible_strategies.append({
                        'name': name,
                        'config': config,
                        'weight': 1.0,  # Will be normalized
                        'reason': f"Compatible with {regime.value} regime and {risk_tolerance} risk"
                    })
        
        if not compatible_strategies:
            # Fallback to defensive strategy
            fallback_config = self.strategies['defensive']
            compatible_strategies.append({
                'name': 'defensive',
                'config': fallback_config,
                'weight': 1.0,
                'reason': 'Fallback strategy - no compatible strategies found'
            })
        
        # Normalize weights
        total_weight = sum(s['weight'] for s in compatible_strategies)
        for strategy in compatible_strategies:
            strategy['weight'] /= total_weight
        
        return compatible_strategies
    
    def execute_strategy_chain(self, data: pd.DataFrame, regime: MarketRegime, 
                             risk_tolerance: str) -> Dict[str, Any]:
        """Execute a strategy chain and combine results."""
        start_time = datetime.now()
        
        # Get strategy chain
        strategy_chain = self.get_strategy_chain(regime, risk_tolerance)
        
        # Execute each strategy
        strategy_results = []
        combined_signals = pd.DataFrame(index=data.index)
        combined_performance = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'volatility': 0.0
        }
        
        for strategy_info in strategy_chain:
            strategy_name = strategy_info['name']
            config = strategy_info['config']
            weight = strategy_info['weight']
            
            # Execute strategy
            result = self._execute_single_strategy(data, config)
            
            if result:
                # Weight the results
                weighted_signals = result.signals * weight
                weighted_performance = {k: v * weight for k, v in result.performance.items()}
                
                strategy_results.append({
                    'strategy_name': strategy_name,
                    'weight': weight,
                    'signals': weighted_signals,
                    'performance': weighted_performance,
                    'confidence': result.confidence,
                    'reason': strategy_info['reason']
                })
                
                # Combine signals and performance
                if combined_signals.empty:
                    combined_signals = weighted_signals
                else:
                    combined_signals += weighted_signals
                
                for key in combined_performance:
                    combined_performance[key] += weighted_performance.get(key, 0.0)
        
        # Create combined result
        execution_time = (datetime.now() - start_time).total_seconds()
        
        combined_result = StrategyResult(
            strategy_name='strategy_chain',
            signals=combined_signals,
            performance=combined_performance,
            confidence=np.mean([r['confidence'] for r in strategy_results]),
            regime=regime,
            parameters_used={'strategy_chain': strategy_chain},
            execution_time=execution_time,
            timestamp=datetime.now()
        )
        
        # Log performance for meta-agent
        self._log_strategy_performance(combined_result, strategy_results)
        
        return {
            'success': True,
            'combined_result': combined_result,
            'strategy_results': strategy_results,
            'regime': regime.value,
            'risk_tolerance': risk_tolerance,
            'execution_time': execution_time
        }
    
    def _execute_single_strategy(self, data: pd.DataFrame, config: StrategyConfig) -> Optional[StrategyResult]:
        """Execute a single strategy."""
        try:
            start_time = datetime.now()
            
            # Generate signals based on strategy type
            signals = self._generate_signals(data, config)
            
            # Calculate performance metrics
            performance = self._calculate_performance(signals, data)
            
            # Adjust confidence based on performance
            confidence = self._adjust_confidence(config.confidence, performance)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StrategyResult(
                strategy_name=config.name,
                signals=signals,
                performance=performance,
                confidence=confidence,
                regime=MarketRegime.NORMAL,  # Will be set by caller
                parameters_used=config.parameters,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Strategy execution failed for {config.name}: {e}")
            return None
    
    def _generate_signals(self, data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """Generate trading signals based on strategy configuration."""
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0.0
        signals['signal'] = 0.0
        
        if config.strategy_type == StrategyType.MOMENTUM:
            lookback = config.parameters['lookback']
            threshold = config.parameters['threshold']
            
            # Calculate momentum
            returns = data['Close'].pct_change(lookback)
            
            # Generate signals
            signals.loc[returns > threshold, 'signal'] = 1.0
            signals.loc[returns < -threshold, 'signal'] = -1.0
            
            # Calculate position size
            signals['position'] = signals['signal'] * 0.1  # 10% position size
        
        elif config.strategy_type == StrategyType.MEAN_REVERSION:
            lookback = config.parameters['lookback']
            std_threshold = config.parameters['std_threshold']
            
            # Calculate z-score
            rolling_mean = data['Close'].rolling(lookback).mean()
            rolling_std = data['Close'].rolling(lookback).std()
            z_score = (data['Close'] - rolling_mean) / rolling_std
            
            # Generate signals
            signals.loc[z_score > std_threshold, 'signal'] = -1.0  # Sell overvalued
            signals.loc[z_score < -std_threshold, 'signal'] = 1.0   # Buy undervalued
            
            # Calculate position size
            signals['position'] = signals['signal'] * 0.1
        
        elif config.strategy_type == StrategyType.BREAKOUT:
            lookback = config.parameters['lookback']
            threshold = config.parameters['breakout_threshold']
            
            # Calculate breakout levels
            rolling_high = data['High'].rolling(lookback).max()
            rolling_low = data['Low'].rolling(lookback).min()
            
            # Generate signals
            signals.loc[data['Close'] > rolling_high * (1 + threshold), 'signal'] = 1.0
            signals.loc[data['Close'] < rolling_low * (1 - threshold), 'signal'] = -1.0
            
            # Calculate position size
            signals['position'] = signals['signal'] * 0.15
        
        elif config.strategy_type == StrategyType.TREND_FOLLOWING:
            short_window = config.parameters['short_window']
            long_window = config.parameters['long_window']
            
            # Calculate moving averages
            short_ma = data['Close'].rolling(short_window).mean()
            long_ma = data['Close'].rolling(long_window).mean()
            
            # Generate signals
            signals.loc[short_ma > long_ma, 'signal'] = 1.0
            signals.loc[short_ma < long_ma, 'signal'] = -1.0
            
            # Calculate position size
            signals['position'] = signals['signal'] * 0.12
        
        elif config.strategy_type == StrategyType.VOLATILITY:
            vol_window = config.parameters['vol_window']
            vol_threshold = config.parameters['vol_threshold']
            
            # Calculate volatility
            returns = data['Close'].pct_change()
            volatility = returns.rolling(vol_window).std()
            
            # Generate signals
            signals.loc[volatility > vol_threshold, 'signal'] = 1.0
            signals.loc[volatility < vol_threshold * 0.5, 'signal'] = -1.0
            
            # Calculate position size
            signals['position'] = signals['signal'] * 0.08
        
        elif config.strategy_type == StrategyType.DEFENSIVE:
            stop_loss = config.parameters['stop_loss']
            position_size = config.parameters['position_size']
            
            # Simple defensive strategy - small long position with stop loss
            signals['signal'] = 1.0
            signals['position'] = position_size
            
            # Apply stop loss (simplified)
            returns = data['Close'].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            stop_loss_triggered = cumulative_returns < (1 - stop_loss)
            signals.loc[stop_loss_triggered, 'position'] = 0.0
        
        elif config.strategy_type == StrategyType.CASH_HEAVY:
            cash_allocation = config.parameters['cash_allocation']
            bond_allocation = config.parameters['bond_allocation']
            
            # Cash-heavy strategy - mostly cash with small bond allocation
            signals['signal'] = 0.0
            signals['position'] = bond_allocation
        
        else:
            # Fallback strategy
            signals['signal'] = 0.0
            signals['position'] = 0.0
        
        return signals
    
    def _calculate_performance(self, signals: pd.DataFrame, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for signals."""
        try:
            # Calculate returns
            price_returns = data['Close'].pct_change()
            strategy_returns = signals['position'].shift(1) * price_returns
            
            # Remove NaN values
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) == 0:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'volatility': 0.0
                }
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate win rate
            win_rate = (strategy_returns > 0).mean()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0
            }
    
    def _adjust_confidence(self, base_confidence: float, performance: Dict[str, float]) -> float:
        """Adjust confidence based on performance metrics."""
        # Adjust based on Sharpe ratio
        sharpe_adjustment = min(performance['sharpe_ratio'] / 2.0, 0.2)  # Max 20% adjustment
        
        # Adjust based on win rate
        win_rate_adjustment = (performance['win_rate'] - 0.5) * 0.3  # Max 15% adjustment
        
        # Adjust based on drawdown
        drawdown_adjustment = max(performance['max_drawdown'] * 2, -0.2)  # Max 20% penalty
        
        adjusted_confidence = base_confidence + sharpe_adjustment + win_rate_adjustment + drawdown_adjustment
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _log_strategy_performance(self, combined_result: StrategyResult, 
                                strategy_results: List[Dict[str, Any]]):
        """Log strategy performance for meta-agent analysis."""
        try:
            # Store performance history
            performance_record = {
                'timestamp': combined_result.timestamp.isoformat(),
                'strategy_name': combined_result.strategy_name,
                'performance': combined_result.performance,
                'confidence': combined_result.confidence,
                'regime': combined_result.regime.value,
                'execution_time': combined_result.execution_time,
                'strategy_breakdown': [
                    {
                        'name': r['strategy_name'],
                        'weight': r['weight'],
                        'performance': r['performance'],
                        'confidence': r['confidence']
                    }
                    for r in strategy_results
                ]
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
            # Check with meta-agent
            if self.meta_agent:
                meta_analysis = self.meta_agent.check_strategy_performance(
                    combined_result.strategy_name,
                    combined_result.performance
                )
                
                if meta_analysis.get('should_retire', False):
                    logger.warning(f"Meta-agent suggests retiring strategy: {combined_result.strategy_name}")
                
                if meta_analysis.get('should_tune', False):
                    logger.info(f"Meta-agent suggests tuning strategy: {combined_result.strategy_name}")
                    
        except Exception as e:
            logger.error(f"Performance logging failed: {e}")
    
    def get_strategy_performance_history(self, strategy_name: str = None, 
                                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get strategy performance history."""
        if strategy_name:
            return [
                record for record in self.performance_history[-limit:]
                if record['strategy_name'] == strategy_name
            ]
        else:
            return self.performance_history[-limit:]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get strategy engine health information."""
        try:
            # Calculate recent performance
            recent_performance = self.performance_history[-50:] if self.performance_history else []
            
            if recent_performance:
                avg_sharpe = np.mean([p['performance']['sharpe_ratio'] for p in recent_performance])
                avg_confidence = np.mean([p['confidence'] for p in recent_performance])
                success_rate = len([p for p in recent_performance if p['performance']['total_return'] > 0]) / len(recent_performance)
            else:
                avg_sharpe = 0.0
                avg_confidence = 0.0
                success_rate = 0.0
            
            return {
                'status': 'healthy' if avg_sharpe > 0.5 else 'degraded',
                'active_strategies': len(self.strategies),
                'recent_performance_count': len(recent_performance),
                'average_sharpe': avg_sharpe,
                'average_confidence': avg_confidence,
                'success_rate': success_rate,
                'last_execution': self.performance_history[-1]['timestamp'] if self.performance_history else None
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# Global instance
enhanced_strategy_engine = EnhancedStrategyEngine()

def get_enhanced_strategy_engine() -> EnhancedStrategyEngine:
    """Get the global enhanced strategy engine instance."""
    return enhanced_strategy_engine 