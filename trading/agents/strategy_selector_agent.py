# -*- coding: utf-8 -*-
"""
Strategy Selector Agent for dynamic strategy selection and parameter optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from trading.strategies.strategy_manager import StrategyManager
from trading.strategies.rsi_signals import generate_rsi_signals
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.bollinger_strategy import BollingerStrategy
from trading.strategies.sma_strategy import SMAStrategy
from trading.optimization.genetic_optimizer import GeneticOptimizer
from trading.market.market_analyzer import MarketAnalyzer
from trading.utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown
from trading.memory.agent_memory import AgentMemory


class StrategyType(str, Enum):
    """Strategy type classifications."""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    SMA = "sma"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    PAIRS = "pairs"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    timestamp: datetime
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    market_regime: str
    confidence_score: float


@dataclass
class StrategyRecommendation:
    """Strategy recommendation with parameters."""
    strategy_name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    confidence_score: float
    expected_sharpe: float
    expected_drawdown: float
    market_regime: str
    reasoning: str


class StrategySelectorAgent:
    """
    Agent responsible for:
    - Detecting best-fit strategies based on market conditions
    - Adjusting strategy parameters using genetic optimization
    - Cross-validating strategies over multiple market regimes
    - Providing strategy recommendations with confidence scores
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Strategy Selector Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.strategy_manager = StrategyManager()
        self.market_analyzer = MarketAnalyzer()
        self.genetic_optimizer = GeneticOptimizer()
        
        # Performance tracking
        self.strategy_performance: Dict[str, List[StrategyPerformance]] = {}
        self.parameter_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configuration
        self.performance_window = self.config.get('performance_window', 30)
        self.optimization_frequency = self.config.get('optimization_frequency', 'weekly')
        self.min_performance_threshold = self.config.get('min_performance_threshold', 0.5)
        self.cross_validation_periods = self.config.get('cross_validation_periods', 5)
        
        # Strategy mappings
        self.strategy_mappings = {
            'trending_up': [StrategyType.MACD, StrategyType.SMA, StrategyType.TREND_FOLLOWING],
            'trending_down': [StrategyType.MACD, StrategyType.SMA, StrategyType.TREND_FOLLOWING],
            'sideways': [StrategyType.RSI, StrategyType.BOLLINGER, StrategyType.MEAN_REVERSION],
            'volatile': [StrategyType.VOLATILITY, StrategyType.BREAKOUT, StrategyType.BOLLINGER],
            'low_volatility': [StrategyType.RSI, StrategyType.MEAN_REVERSION, StrategyType.PAIRS]
        }
        
        # Parameter spaces for each strategy
        self.parameter_spaces = {
            StrategyType.RSI: {
                'period': [10, 14, 20, 30],
                'overbought': [60, 70, 80],
                'oversold': [20, 30, 40],
                'smoothing': [1, 3, 5]
            },
            StrategyType.MACD: {
                'fast_period': [8, 12, 16],
                'slow_period': [20, 26, 32],
                'signal_period': [7, 9, 11],
                'smoothing': [1, 3, 5]
            },
            StrategyType.BOLLINGER: {
                'period': [15, 20, 25, 30],
                'std_dev': [1.5, 2.0, 2.5, 3.0],
                'smoothing': [1, 3, 5]
            },
            StrategyType.SMA: {
                'short_period': [5, 10, 15, 20],
                'long_period': [30, 50, 100, 200],
                'smoothing': [1, 3, 5]
            },
            StrategyType.BREAKOUT: {
                'period': [10, 20, 30, 50],
                'multiplier': [1.5, 2.0, 2.5, 3.0],
                'volume_threshold': [1.2, 1.5, 2.0, 2.5]
            },
            StrategyType.VOLATILITY: {
                'period': [10, 20, 30],
                'threshold': [0.01, 0.02, 0.03, 0.05],
                'smoothing': [1, 3, 5]
            }
        }
        
        # Load existing data
        self._load_strategy_performance()
        
    def select_strategy(self, 
                       market_data: pd.DataFrame,
                       asset_symbol: str,
                       forecast_horizon: int,
                       risk_tolerance: str = 'medium') -> StrategyRecommendation:
        """
        Select the best strategy for the given market conditions.
        
        Args:
            market_data: Historical market data
            asset_symbol: Asset symbol
            forecast_horizon: Forecast horizon
            risk_tolerance: Risk tolerance level (low/medium/high)
            
        Returns:
            StrategyRecommendation with optimal strategy and parameters
        """
        try:
            # Analyze market conditions
            market_regime = self._detect_market_regime(market_data)
            volatility = self._calculate_volatility(market_data)
            trend_strength = self._calculate_trend_strength(market_data)
            
            # Get compatible strategies
            compatible_strategies = self._get_compatible_strategies(
                market_regime, volatility, trend_strength, risk_tolerance
            )
            
            if not compatible_strategies:
                self.logger.warning("No compatible strategies found, using default")
                return self._get_default_strategy(market_regime)
            
            # Optimize parameters for each strategy
            strategy_recommendations = []
            for strategy_type in compatible_strategies:
                optimized_params = self._optimize_strategy_parameters(
                    strategy_type, market_data, forecast_horizon
                )
                
                # Evaluate strategy with optimized parameters
                performance = self._evaluate_strategy(
                    strategy_type, optimized_params, market_data, forecast_horizon
                )
                
                if performance:
                    recommendation = StrategyRecommendation(
                        strategy_name=f"{strategy_type.value}_strategy",
                        strategy_type=strategy_type,
                        parameters=optimized_params,
                        confidence_score=performance.confidence_score,
                        expected_sharpe=performance.sharpe_ratio,
                        expected_drawdown=performance.max_drawdown,
                        market_regime=market_regime,
                        reasoning=self._generate_reasoning(strategy_type, market_regime, performance)
                    )
                    strategy_recommendations.append(recommendation)
            
            if not strategy_recommendations:
                return self._get_default_strategy(market_regime)
            
            # Select best strategy
            best_recommendation = max(
                strategy_recommendations, 
                key=lambda x: x.confidence_score
            )
            
            # Log selection
            self.logger.info(f"Selected strategy: {best_recommendation.strategy_name}")
            self.logger.info(f"Parameters: {best_recommendation.parameters}")
            self.logger.info(f"Confidence: {best_recommendation.confidence_score:.3f}")
            
            # Store selection
            self._store_strategy_selection(best_recommendation, market_data)
            
            return best_recommendation
            
        except Exception as e:
            self.logger.error(f"Error in strategy selection: {str(e)}")
            return self._get_default_strategy('sideways')
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect market regime from price data."""
        try:
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # Calculate trend
            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=50).mean()
            trend = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            
            # Determine regime
            if volatility > 0.03:
                return 'volatile'
            elif volatility < 0.01:
                return 'low_volatility'
            elif trend > 0.02:
                return 'trending_up'
            elif trend < -0.02:
                return 'trending_down'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return 'sideways'
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate current volatility."""
        try:
            returns = market_data['close'].pct_change().dropna()
            return returns.rolling(window=20).std().iloc[-1]
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {str(e)}")
            return 0.02
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength."""
        try:
            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=50).mean()
            return abs((sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1])
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.01
    
    def _get_compatible_strategies(self, 
                                 market_regime: str,
                                 volatility: float,
                                 trend_strength: float,
                                 risk_tolerance: str) -> List[StrategyType]:
        """Get list of strategies compatible with market conditions."""
        try:
            # Get base strategies for market regime
            base_strategies = self.strategy_mappings.get(market_regime, [StrategyType.RSI])
            
            # Filter based on volatility
            if volatility > 0.03:
                # High volatility - prefer volatility and breakout strategies
                compatible = [StrategyType.VOLATILITY, StrategyType.BREAKOUT, StrategyType.BOLLINGER]
            elif volatility < 0.01:
                # Low volatility - prefer mean reversion strategies
                compatible = [StrategyType.RSI, StrategyType.MEAN_REVERSION, StrategyType.PAIRS]
            else:
                # Medium volatility - use base strategies
                compatible = base_strategies
            
            # Filter based on trend strength
            if trend_strength > 0.03:
                # Strong trend - add trend following strategies
                compatible.extend([StrategyType.MACD, StrategyType.SMA, StrategyType.TREND_FOLLOWING])
            
            # Filter based on risk tolerance
            if risk_tolerance == 'low':
                # Low risk - prefer conservative strategies
                compatible = [s for s in compatible if s in [StrategyType.SMA, StrategyType.MEAN_REVERSION]]
            elif risk_tolerance == 'high':
                # High risk - prefer aggressive strategies
                compatible.extend([StrategyType.BREAKOUT, StrategyType.VOLATILITY])
            
            return list(set(compatible))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error getting compatible strategies: {str(e)}")
            return [StrategyType.RSI]
    
    def _optimize_strategy_parameters(self, 
                                    strategy_type: StrategyType,
                                    market_data: pd.DataFrame,
                                    forecast_horizon: int) -> Dict[str, Any]:
        """Optimize parameters for a specific strategy."""
        try:
            # Get parameter space
            param_space = self.parameter_spaces.get(strategy_type, {})
            if not param_space:
                return self._get_default_parameters(strategy_type)
            
            # Define optimization objective
            def objective(parameters):
                try:
                    # Evaluate strategy with given parameters
                    performance = self._evaluate_strategy(
                        strategy_type, parameters, market_data, forecast_horizon
                    )
                    
                    if performance:
                        # Return negative score (minimize)
                        return -(performance.sharpe_ratio - 0.5 * performance.max_drawdown)
                    else:
                        return 0.0
                        
                except Exception as e:
                    self.logger.error(f"Error in optimization objective: {str(e)}")
                    return 0.0
            
            # Run genetic optimization
            best_params = self.genetic_optimizer.optimize(
                objective=objective,
                param_space=param_space,
                population_size=30,
                generations=15
            )
            
            return best_params or self._get_default_parameters(strategy_type)
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {str(e)}")
            return self._get_default_parameters(strategy_type)
    
    def _evaluate_strategy(self, 
                          strategy_type: StrategyType,
                          parameters: Dict[str, Any],
                          market_data: pd.DataFrame,
                          forecast_horizon: int) -> Optional[StrategyPerformance]:
        """Evaluate strategy performance with given parameters."""
        try:
            # Generate signals using the strategy
            signals = self._generate_signals(strategy_type, parameters, market_data)
            
            if signals is None or len(signals) < 20:
                return None
            
            # Calculate performance metrics
            returns = market_data['close'].pct_change().dropna()
            strategy_returns = signals.shift(1) * returns
            
            # Remove NaN values
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < 10:
                return None
            
            # Calculate metrics
            sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
            max_drawdown = calculate_max_drawdown(strategy_returns)
            win_rate = (strategy_returns > 0).mean()
            total_return = strategy_returns.sum()
            
            # Calculate profit factor
            positive_returns = strategy_returns[strategy_returns > 0].sum()
            negative_returns = abs(strategy_returns[strategy_returns < 0].sum())
            profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                sharpe_ratio, max_drawdown, win_rate, profit_factor
            )
            
            return StrategyPerformance(
                strategy_name=f"{strategy_type.value}_strategy",
                strategy_type=strategy_type,
                parameters=parameters,
                timestamp=datetime.now(),
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_return=total_return,
                market_regime=self._detect_market_regime(market_data),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy: {str(e)}")
            return None
    
    def _generate_signals(self, 
                         strategy_type: StrategyType,
                         parameters: Dict[str, Any],
                         market_data: pd.DataFrame) -> Optional[pd.Series]:
        """Generate trading signals for a strategy."""
        try:
            if strategy_type == StrategyType.RSI:
                return generate_rsi_signals(
                    market_data['close'],
                    period=parameters.get('period', 14),
                    overbought=parameters.get('overbought', 70),
                    oversold=parameters.get('oversold', 30)
                )
            
            elif strategy_type == StrategyType.MACD:
                strategy = MACDStrategy()
                return strategy.generate_signals(
                    market_data,
                    fast_period=parameters.get('fast_period', 12),
                    slow_period=parameters.get('slow_period', 26),
                    signal_period=parameters.get('signal_period', 9)
                )
            
            elif strategy_type == StrategyType.BOLLINGER:
                strategy = BollingerStrategy()
                return strategy.generate_signals(
                    market_data,
                    period=parameters.get('period', 20),
                    std_dev=parameters.get('std_dev', 2.0)
                )
            
            elif strategy_type == StrategyType.SMA:
                strategy = SMAStrategy()
                return strategy.generate_signals(
                    market_data,
                    short_period=parameters.get('short_period', 10),
                    long_period=parameters.get('long_period', 50)
                )
            
            elif strategy_type == StrategyType.BREAKOUT:
                return self._generate_breakout_signals(market_data, parameters)
            
            elif strategy_type == StrategyType.VOLATILITY:
                return self._generate_volatility_signals(market_data, parameters)
            
            else:
                # Default to RSI
                return generate_rsi_signals(market_data['close'])
                
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return None
    
    def _generate_breakout_signals(self, 
                                 market_data: pd.DataFrame,
                                 parameters: Dict[str, Any]) -> pd.Series:
        """Generate breakout strategy signals."""
        try:
            period = parameters.get('period', 20)
            multiplier = parameters.get('multiplier', 2.0)
            volume_threshold = parameters.get('volume_threshold', 1.5)
            
            # Calculate upper and lower bands
            high_band = market_data['high'].rolling(window=period).max()
            low_band = market_data['low'].rolling(window=period).min()
            
            # Calculate volume spike
            volume_ma = market_data['volume'].rolling(window=period).mean()
            volume_spike = market_data['volume'] > (volume_ma * volume_threshold)
            
            # Generate signals
            signals = pd.Series(0, index=market_data.index)
            
            # Buy signal: price breaks above upper band with volume spike
            buy_signal = (market_data['close'] > high_band.shift(1)) & volume_spike
            signals[buy_signal] = 1
            
            # Sell signal: price breaks below lower band with volume spike
            sell_signal = (market_data['close'] < low_band.shift(1)) & volume_spike
            signals[sell_signal] = -1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating breakout signals: {str(e)}")
            return pd.Series(0, index=market_data.index)
    
    def _generate_volatility_signals(self, 
                                   market_data: pd.DataFrame,
                                   parameters: Dict[str, Any]) -> pd.Series:
        """Generate volatility-based strategy signals."""
        try:
            period = parameters.get('period', 20)
            threshold = parameters.get('threshold', 0.02)
            
            # Calculate rolling volatility
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(window=period).std()
            
            # Generate signals based on volatility regime
            signals = pd.Series(0, index=market_data.index)
            
            # High volatility: reduce position or hedge
            high_vol = volatility > threshold
            signals[high_vol] = -0.5  # Reduce position
            
            # Low volatility: increase position
            low_vol = volatility < (threshold * 0.5)
            signals[low_vol] = 0.5  # Increase position
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating volatility signals: {str(e)}")
            return pd.Series(0, index=market_data.index)
    
    def _calculate_confidence_score(self, 
                                  sharpe_ratio: float,
                                  max_drawdown: float,
                                  win_rate: float,
                                  profit_factor: float) -> float:
        """Calculate confidence score for strategy performance."""
        try:
            # Normalize metrics
            sharpe_score = max(0, min(1, (sharpe_ratio + 2) / 4))
            drawdown_score = max(0, min(1, 1 - abs(max_drawdown)))
            win_rate_score = win_rate
            profit_factor_score = min(1, profit_factor / 3) if profit_factor < float('inf') else 1
            
            # Weighted average
            confidence = (
                0.3 * sharpe_score +
                0.2 * drawdown_score +
                0.3 * win_rate_score +
                0.2 * profit_factor_score
            )
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _generate_reasoning(self, 
                          strategy_type: StrategyType,
                          market_regime: str,
                          performance: StrategyPerformance) -> str:
        """Generate reasoning for strategy selection."""
        try:
            reasoning = f"Selected {strategy_type.value} strategy for {market_regime} market regime. "
            reasoning += f"Expected Sharpe: {performance.sharpe_ratio:.3f}, "
            reasoning += f"Max Drawdown: {performance.max_drawdown:.3f}, "
            reasoning += f"Win Rate: {performance.win_rate:.3f}. "
            
            if performance.sharpe_ratio > 1.0:
                reasoning += "Strategy shows strong risk-adjusted returns. "
            elif performance.sharpe_ratio > 0.5:
                reasoning += "Strategy shows moderate risk-adjusted returns. "
            else:
                reasoning += "Strategy shows weak risk-adjusted returns but fits market conditions. "
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {str(e)}")
            return f"Selected {strategy_type.value} strategy based on market conditions."
    
    def _get_default_strategy(self, market_regime: str) -> StrategyRecommendation:
        """Get default strategy when selection fails."""
        default_type = StrategyType.RSI
        default_params = self._get_default_parameters(default_type)
        
        return StrategyRecommendation(
            strategy_name=f"{default_type.value}_strategy",
            strategy_type=default_type,
            parameters=default_params,
            confidence_score=0.5,
            expected_sharpe=0.0,
            expected_drawdown=0.0,
            market_regime=market_regime,
            reasoning=f"Using default {default_type.value} strategy"
        )
    
    def _get_default_parameters(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get default parameters for a strategy type."""
        try:
            if strategy_type == StrategyType.RSI:
                return {'period': 14, 'overbought': 70, 'oversold': 30}
            elif strategy_type == StrategyType.MACD:
                return {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            elif strategy_type == StrategyType.BOLLINGER:
                return {'period': 20, 'std_dev': 2.0}
            elif strategy_type == StrategyType.SMA:
                return {'short_period': 10, 'long_period': 50}
            elif strategy_type == StrategyType.BREAKOUT:
                return {'period': 20, 'multiplier': 2.0, 'volume_threshold': 1.5}
            elif strategy_type == StrategyType.VOLATILITY:
                return {'period': 20, 'threshold': 0.02}
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting default parameters: {str(e)}")
            return {}
    
    def _store_strategy_selection(self, 
                                recommendation: StrategyRecommendation,
                                market_data: pd.DataFrame):
        """Store strategy selection in memory."""
        try:
            selection_data = {
                'strategy_name': recommendation.strategy_name,
                'strategy_type': recommendation.strategy_type.value,
                'parameters': recommendation.parameters,
                'confidence_score': recommendation.confidence_score,
                'market_regime': recommendation.market_regime,
                'timestamp': datetime.now().isoformat()
            }
            
            self.memory.store('strategy_selections', selection_data)
            
        except Exception as e:
            self.logger.error(f"Error storing strategy selection: {str(e)}")
    
    def update_strategy_performance(self, performance: StrategyPerformance):
        """Update strategy performance after execution."""
        try:
            strategy_name = performance.strategy_name
            
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = []
            
            self.strategy_performance[strategy_name].append(performance)
            
            # Keep only recent performance
            cutoff_date = datetime.now() - timedelta(days=self.performance_window)
            self.strategy_performance[strategy_name] = [
                p for p in self.strategy_performance[strategy_name]
                if p.timestamp > cutoff_date
            ]
            
            # Store in memory
            memory_key = f"strategy_performance_{strategy_name}"
            self.memory.store(memory_key, {
                'performance': [p.__dict__ for p in self.strategy_performance[strategy_name]],
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Updated performance for strategy: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Error updating strategy performance: {str(e)}")
    
    def get_strategy_recommendations(self, 
                                   market_regime: str,
                                   risk_tolerance: str = 'medium') -> List[StrategyRecommendation]:
        """Get strategy recommendations for given market conditions."""
        try:
            # Create sample market data for evaluation
            sample_data = self._create_sample_market_data(market_regime)
            
            # Get compatible strategies
            compatible_strategies = self._get_compatible_strategies(
                market_regime, 0.02, 0.01, risk_tolerance
            )
            
            recommendations = []
            for strategy_type in compatible_strategies[:3]:  # Top 3 strategies
                optimized_params = self._optimize_strategy_parameters(
                    strategy_type, sample_data, 30
                )
                
                performance = self._evaluate_strategy(
                    strategy_type, optimized_params, sample_data, 30
                )
                
                if performance:
                    recommendation = StrategyRecommendation(
                        strategy_name=f"{strategy_type.value}_strategy",
                        strategy_type=strategy_type,
                        parameters=optimized_params,
                        confidence_score=performance.confidence_score,
                        expected_sharpe=performance.sharpe_ratio,
                        expected_drawdown=performance.max_drawdown,
                        market_regime=market_regime,
                        reasoning=self._generate_reasoning(strategy_type, market_regime, performance)
                    )
                    recommendations.append(recommendation)
            
            return sorted(recommendations, key=lambda x: x.confidence_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting strategy recommendations: {str(e)}")
            return []
    
    def _create_sample_market_data(self, market_regime: str) -> pd.DataFrame:
        """Create sample market data for strategy evaluation."""
        try:
            # Generate sample data based on market regime
            np.random.seed(42)
            n_periods = 100
            
            if market_regime == 'trending_up':
                trend = 0.001
                volatility = 0.02
            elif market_regime == 'trending_down':
                trend = -0.001
                volatility = 0.02
            elif market_regime == 'volatile':
                trend = 0.0
                volatility = 0.04
            else:  # sideways or low_volatility
                trend = 0.0
                volatility = 0.01
            
            # Generate price series
            returns = np.random.normal(trend, volatility, n_periods)
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create DataFrame
            data = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
                'high': prices * (1 + abs(np.random.normal(0, 0.002, n_periods))),
                'low': prices * (1 - abs(np.random.normal(0, 0.002, n_periods))),
                'close': prices,
                'volume': np.random.randint(1000, 10000, n_periods)
            })
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating sample market data: {str(e)}")
            return pd.DataFrame()
    
    def _load_strategy_performance(self):
        """Load strategy performance from memory."""
        try:
            performance_file = Path("memory/strategy_performance.json")
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    data = json.load(f)
                    for strategy_name, performance_list in data.items():
                        self.strategy_performance[strategy_name] = [
                            StrategyPerformance(**p) for p in performance_list
                        ]
                        
        except Exception as e:
            self.logger.error(f"Error loading strategy performance: {str(e)}")
    
    def save_strategy_performance(self):
        """Save strategy performance to file."""
        try:
            performance_file = Path("memory/strategy_performance.json")
            performance_file.parent.mkdir(exist_ok=True)
            
            data = {}
            for strategy_name, performance_list in self.strategy_performance.items():
                data[strategy_name] = [p.__dict__ for p in performance_list]
            
            with open(performance_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving strategy performance: {str(e)}") 