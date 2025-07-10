"""
Adaptive Strategy and Model Selector

This module provides intelligent selection of models and strategies based on market conditions,
volatility regimes, and performance history. It automatically adapts to changing market
conditions and optimizes model weights in hybrid ensembles.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Market volatility regimes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class MarketTrend(Enum):
    """Market trend directions."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    rmse: float
    mae: float
    mape: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    volatility_score: float
    trend_score: float
    overall_score: float
    last_updated: datetime
    regime_performance: Dict[str, float] = None

@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_name: str
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    volatility_score: float
    trend_score: float
    overall_score: float
    last_updated: datetime
    regime_performance: Dict[str, float] = None

@dataclass
class MarketConditions:
    """Current market conditions."""
    volatility_regime: VolatilityRegime
    market_trend: MarketTrend
    volatility_score: float
    trend_score: float
    volume_score: float
    momentum_score: float
    timestamp: datetime

class VolatilityAnalyzer:
    """Analyzes market volatility and determines regime."""
    
    def __init__(self, lookback_period: int = 30):
        """Initialize volatility analyzer.
        
        Args:
            lookback_period: Number of days to look back for volatility calculation
        """
        self.lookback_period = lookback_period
        self.volatility_thresholds = {
            VolatilityRegime.LOW: 0.01,
            VolatilityRegime.MEDIUM: 0.02,
            VolatilityRegime.HIGH: 0.04,
            VolatilityRegime.EXTREME: 0.08
        }
    
    def analyze_volatility(self, price_data: pd.Series) -> VolatilityRegime:
        """Analyze price data and determine volatility regime.
        
        Args:
            price_data: Price series data
            
        Returns:
            Volatility regime
        """
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=self.lookback_period).std()
            current_vol = rolling_vol.iloc[-1]
            
            # Determine regime
            if current_vol <= self.volatility_thresholds[VolatilityRegime.LOW]:
                return VolatilityRegime.LOW
            elif current_vol <= self.volatility_thresholds[VolatilityRegime.MEDIUM]:
                return VolatilityRegime.MEDIUM
            elif current_vol <= self.volatility_thresholds[VolatilityRegime.HIGH]:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.EXTREME
                
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return VolatilityRegime.MEDIUM
    
    def calculate_volatility_score(self, price_data: pd.Series) -> float:
        """Calculate normalized volatility score (0-1).
        
        Args:
            price_data: Price series data
            
        Returns:
            Volatility score between 0 and 1
        """
        try:
            returns = price_data.pct_change().dropna()
            rolling_vol = returns.rolling(window=self.lookback_period).std()
            current_vol = rolling_vol.iloc[-1]
            
            # Normalize to 0-1 scale
            max_vol = self.volatility_thresholds[VolatilityRegime.EXTREME]
            volatility_score = min(1.0, current_vol / max_vol)
            
            return volatility_score
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {e}")
            return 0.5

class TrendAnalyzer:
    """Analyzes market trends and momentum."""
    
    def __init__(self, short_period: int = 10, long_period: int = 30):
        """Initialize trend analyzer.
        
        Args:
            short_period: Short-term moving average period
            long_period: Long-term moving average period
        """
        self.short_period = short_period
        self.long_period = long_period
    
    def analyze_trend(self, price_data: pd.Series) -> MarketTrend:
        """Analyze price data and determine market trend.
        
        Args:
            price_data: Price series data
            
        Returns:
            Market trend
        """
        try:
            # Calculate moving averages
            short_ma = price_data.rolling(window=self.short_period).mean()
            long_ma = price_data.rolling(window=self.long_period).mean()
            
            # Calculate momentum indicators
            rsi = self._calculate_rsi(price_data)
            momentum = self._calculate_momentum(price_data)
            
            # Determine trend based on multiple indicators
            ma_trend = short_ma.iloc[-1] > long_ma.iloc[-1]
            rsi_trend = rsi > 50
            momentum_trend = momentum > 0
            
            # Combine signals
            bullish_signals = sum([ma_trend, rsi_trend, momentum_trend])
            
            if bullish_signals >= 2:
                return MarketTrend.BULL
            elif bullish_signals <= 1:
                return MarketTrend.BEAR
            else:
                return MarketTrend.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return MarketTrend.NEUTRAL
    
    def calculate_trend_score(self, price_data: pd.Series) -> float:
        """Calculate normalized trend score (-1 to 1).
        
        Args:
            price_data: Price series data
            
        Returns:
            Trend score between -1 and 1
        """
        try:
            # Calculate multiple trend indicators
            short_ma = price_data.rolling(window=self.short_period).mean()
            long_ma = price_data.rolling(window=self.long_period).mean()
            
            # MA trend
            ma_score = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            
            # RSI trend
            rsi = self._calculate_rsi(price_data)
            rsi_score = (rsi - 50) / 50
            
            # Momentum trend
            momentum = self._calculate_momentum(price_data)
            momentum_score = np.tanh(momentum / 0.01)  # Normalize with tanh
            
            # Combine scores
            trend_score = (ma_score + rsi_score + momentum_score) / 3
            return max(-1.0, min(1.0, trend_score))
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {e}")
            return 0.0
    
    def _calculate_rsi(self, price_data: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            delta = price_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0
    
    def _calculate_momentum(self, price_data: pd.Series, period: int = 10) -> float:
        """Calculate momentum indicator."""
        try:
            return (price_data.iloc[-1] - price_data.iloc[-period]) / price_data.iloc[-period]
        except:
            return 0.0

class ModelSelector:
    """Intelligent model selector based on market conditions."""
    
    def __init__(self):
        """Initialize model selector."""
        self.model_registry = {
            'LSTM': {
                'preferred_regimes': [VolatilityRegime.LOW, VolatilityRegime.MEDIUM],
                'preferred_trends': [MarketTrend.BULL, MarketTrend.BEAR],
                'volatility_range': [0.005, 0.03],
                'trend_range': [-0.5, 0.5]
            },
            'XGBoost': {
                'preferred_regimes': [VolatilityRegime.MEDIUM, VolatilityRegime.HIGH],
                'preferred_trends': [MarketTrend.VOLATILE, MarketTrend.NEUTRAL],
                'volatility_range': [0.02, 0.06],
                'trend_range': [-1.0, 1.0]
            },
            'Transformer': {
                'preferred_regimes': [VolatilityRegime.HIGH, VolatilityRegime.EXTREME],
                'preferred_trends': [MarketTrend.VOLATILE, MarketTrend.NEUTRAL],
                'volatility_range': [0.04, 0.10],
                'trend_range': [-1.0, 1.0]
            },
            'ARIMA': {
                'preferred_regimes': [VolatilityRegime.LOW, VolatilityRegime.MEDIUM],
                'preferred_trends': [MarketTrend.BULL, MarketTrend.BEAR],
                'volatility_range': [0.005, 0.025],
                'trend_range': [-0.3, 0.3]
            },
            'Prophet': {
                'preferred_regimes': [VolatilityRegime.LOW, VolatilityRegime.MEDIUM],
                'preferred_trends': [MarketTrend.BULL, MarketTrend.BEAR],
                'preferred_trends': [MarketTrend.BULL, MarketTrend.BEAR],
                'volatility_range': [0.005, 0.025],
                'trend_range': [-0.3, 0.3]
            }
        }
        
        self.performance_history = {}
    
    def select_optimal_model(self, market_conditions: MarketConditions, 
                           available_models: List[str]) -> str:
        """Select optimal model based on market conditions.
        
        Args:
            market_conditions: Current market conditions
            available_models: List of available model names
            
        Returns:
            Selected model name
        """
        try:
            model_scores = {}
            
            for model_name in available_models:
                if model_name in self.model_registry:
                    score = self._calculate_model_score(model_name, market_conditions)
                    model_scores[model_name] = score
            
            # Select model with highest score
            if model_scores:
                best_model = max(model_scores, key=model_scores.get)
                logger.info(f"Selected model: {best_model} (score: {model_scores[best_model]:.3f})")
                return best_model
            else:
                # Fallback to LSTM
                logger.warning("No suitable model found, using LSTM as fallback")
                return 'LSTM'
                
        except Exception as e:
            logger.error(f"Error selecting optimal model: {e}")
            return 'LSTM'
    
    def _calculate_model_score(self, model_name: str, 
                             market_conditions: MarketConditions) -> float:
        """Calculate suitability score for a model.
        
        Args:
            model_name: Name of the model
            market_conditions: Current market conditions
            
        Returns:
            Suitability score (0-1)
        """
        try:
            model_config = self.model_registry[model_name]
            
            # Regime compatibility
            regime_score = 1.0 if market_conditions.volatility_regime in model_config['preferred_regimes'] else 0.3
            
            # Trend compatibility
            trend_score = 1.0 if market_conditions.market_trend in model_config['preferred_trends'] else 0.3
            
            # Volatility range compatibility
            vol_min, vol_max = model_config['volatility_range']
            vol_compatibility = 1.0 if vol_min <= market_conditions.volatility_score <= vol_max else 0.5
            
            # Trend range compatibility
            trend_min, trend_max = model_config['trend_range']
            trend_compatibility = 1.0 if trend_min <= market_conditions.trend_score <= trend_max else 0.5
            
            # Historical performance (if available)
            performance_score = self._get_historical_performance(model_name, market_conditions)
            
            # Calculate weighted score
            weights = {
                'regime': 0.25,
                'trend': 0.25,
                'volatility': 0.2,
                'trend_range': 0.15,
                'performance': 0.15
            }
            
            total_score = (
                regime_score * weights['regime'] +
                trend_score * weights['trend'] +
                vol_compatibility * weights['volatility'] +
                trend_compatibility * weights['trend_range'] +
                performance_score * weights['performance']
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating model score: {e}")
            return 0.5
    
    def _get_historical_performance(self, model_name: str, 
                                  market_conditions: MarketConditions) -> float:
        """Get historical performance for model in similar conditions.
        
        Args:
            model_name: Name of the model
            market_conditions: Current market conditions
            
        Returns:
            Performance score (0-1)
        """
        try:
            if model_name not in self.performance_history:
                return 0.5
            
            # Find similar market conditions in history
            similar_performances = []
            
            for performance in self.performance_history[model_name]:
                # Check if conditions are similar
                regime_match = performance.regime_performance.get(str(market_conditions.volatility_regime.value), 0.5)
                similar_performances.append(regime_match)
            
            if similar_performances:
                return np.mean(similar_performances)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return 0.5

class StrategySelector:
    """Intelligent strategy selector based on market conditions."""
    
    def __init__(self):
        """Initialize strategy selector."""
        self.strategy_registry = {
            'RSI Mean Reversion': {
                'preferred_regimes': [VolatilityRegime.MEDIUM, VolatilityRegime.HIGH],
                'preferred_trends': [MarketTrend.NEUTRAL, MarketTrend.VOLATILE],
                'volatility_range': [0.015, 0.05],
                'trend_range': [-0.3, 0.3]
            },
            'Moving Average Crossover': {
                'preferred_regimes': [VolatilityRegime.LOW, VolatilityRegime.MEDIUM],
                'preferred_trends': [MarketTrend.BULL, MarketTrend.BEAR],
                'volatility_range': [0.005, 0.025],
                'trend_range': [-0.8, 0.8]
            },
            'Bollinger Bands': {
                'preferred_regimes': [VolatilityRegime.MEDIUM, VolatilityRegime.HIGH],
                'preferred_trends': [MarketTrend.NEUTRAL, MarketTrend.VOLATILE],
                'volatility_range': [0.02, 0.06],
                'trend_range': [-0.5, 0.5]
            },
            'MACD': {
                'preferred_regimes': [VolatilityRegime.LOW, VolatilityRegime.MEDIUM],
                'preferred_trends': [MarketTrend.BULL, MarketTrend.BEAR],
                'volatility_range': [0.005, 0.03],
                'trend_range': [-0.6, 0.6]
            },
            'GARCH Volatility': {
                'preferred_regimes': [VolatilityRegime.HIGH, VolatilityRegime.EXTREME],
                'preferred_trends': [MarketTrend.VOLATILE, MarketTrend.NEUTRAL],
                'volatility_range': [0.04, 0.10],
                'trend_range': [-1.0, 1.0]
            }
        }
        
        self.performance_history = {}
    
    def select_optimal_strategy(self, market_conditions: MarketConditions,
                              available_strategies: List[str]) -> str:
        """Select optimal strategy based on market conditions.
        
        Args:
            market_conditions: Current market conditions
            available_strategies: List of available strategy names
            
        Returns:
            Selected strategy name
        """
        try:
            strategy_scores = {}
            
            for strategy_name in available_strategies:
                if strategy_name in self.strategy_registry:
                    score = self._calculate_strategy_score(strategy_name, market_conditions)
                    strategy_scores[strategy_name] = score
            
            # Select strategy with highest score
            if strategy_scores:
                best_strategy = max(strategy_scores, key=strategy_scores.get)
                logger.info(f"Selected strategy: {best_strategy} (score: {strategy_scores[best_strategy]:.3f})")
                return best_strategy
            else:
                # Fallback to RSI
                logger.warning("No suitable strategy found, using RSI as fallback")
                return 'RSI Mean Reversion'
                
        except Exception as e:
            logger.error(f"Error selecting optimal strategy: {e}")
            return 'RSI Mean Reversion'
    
    def _calculate_strategy_score(self, strategy_name: str,
                                market_conditions: MarketConditions) -> float:
        """Calculate suitability score for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            market_conditions: Current market conditions
            
        Returns:
            Suitability score (0-1)
        """
        try:
            strategy_config = self.strategy_registry[strategy_name]
            
            # Regime compatibility
            regime_score = 1.0 if market_conditions.volatility_regime in strategy_config['preferred_regimes'] else 0.3
            
            # Trend compatibility
            trend_score = 1.0 if market_conditions.market_trend in strategy_config['preferred_trends'] else 0.3
            
            # Volatility range compatibility
            vol_min, vol_max = strategy_config['volatility_range']
            vol_compatibility = 1.0 if vol_min <= market_conditions.volatility_score <= vol_max else 0.5
            
            # Trend range compatibility
            trend_min, trend_max = strategy_config['trend_range']
            trend_compatibility = 1.0 if trend_min <= market_conditions.trend_score <= trend_max else 0.5
            
            # Historical performance
            performance_score = self._get_historical_performance(strategy_name, market_conditions)
            
            # Calculate weighted score
            weights = {
                'regime': 0.25,
                'trend': 0.25,
                'volatility': 0.2,
                'trend_range': 0.15,
                'performance': 0.15
            }
            
            total_score = (
                regime_score * weights['regime'] +
                trend_score * weights['trend'] +
                vol_compatibility * weights['volatility'] +
                trend_compatibility * weights['trend_range'] +
                performance_score * weights['performance']
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating strategy score: {e}")
            return 0.5
    
    def _get_historical_performance(self, strategy_name: str,
                                  market_conditions: MarketConditions) -> float:
        """Get historical performance for strategy in similar conditions."""
        try:
            if strategy_name not in self.performance_history:
                return 0.5
            
            similar_performances = []
            
            for performance in self.performance_history[strategy_name]:
                regime_match = performance.regime_performance.get(str(market_conditions.volatility_regime.value), 0.5)
                similar_performances.append(regime_match)
            
            if similar_performances:
                return np.mean(similar_performances)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return 0.5

class HybridEnsembleOptimizer:
    """Optimizes weights for hybrid ensemble models."""
    
    def __init__(self, reweight_period: int = 30):
        """Initialize hybrid ensemble optimizer.
        
        Args:
            reweight_period: Days to look back for performance calculation
        """
        self.reweight_period = reweight_period
        self.weight_history = {}
    
    def optimize_weights(self, models: List[str], 
                        performance_data: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Optimize model weights based on recent performance.
        
        Args:
            models: List of model names
            performance_data: Performance data for each model
            
        Returns:
            Dictionary of optimized weights
        """
        try:
            weights = {}
            total_score = 0
            
            for model_name in models:
                if model_name in performance_data:
                    # Calculate performance score
                    performance = performance_data[model_name]
                    score = self._calculate_performance_score(performance)
                    weights[model_name] = score
                    total_score += score
                else:
                    # Default weight for new models
                    weights[model_name] = 0.5
                    total_score += 0.5
            
            # Normalize weights
            if total_score > 0:
                for model_name in weights:
                    weights[model_name] /= total_score
            else:
                # Equal weights if no performance data
                equal_weight = 1.0 / len(models)
                for model_name in models:
                    weights[model_name] = equal_weight
            
            # Store weight history
            self.weight_history[datetime.now()] = weights
            
            logger.info(f"Optimized weights: {weights}")
            return weights
            
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
            # Return equal weights as fallback
            equal_weight = 1.0 / len(models)
            return {model: equal_weight for model in models}
    
    def _calculate_performance_score(self, performance: ModelPerformance) -> float:
        """Calculate performance score for weight optimization.
        
        Args:
            performance: Model performance data
            
        Returns:
            Performance score
        """
        try:
            # Weight different metrics
            weights = {
                'sharpe_ratio': 0.3,
                'win_rate': 0.25,
                'profit_factor': 0.2,
                'max_drawdown': 0.15,
                'rmse': 0.1
            }
            
            # Normalize metrics
            sharpe_score = max(0, min(1, performance.sharpe_ratio / 2))  # Normalize to 0-1
            win_rate_score = performance.win_rate
            profit_factor_score = min(1, performance.profit_factor / 3)
            drawdown_score = max(0, 1 - performance.max_drawdown / 0.3)  # Lower is better
            rmse_score = max(0, 1 - performance.rmse / 0.1)  # Lower is better
            
            # Calculate weighted score
            total_score = (
                sharpe_score * weights['sharpe_ratio'] +
                win_rate_score * weights['win_rate'] +
                profit_factor_score * weights['profit_factor'] +
                drawdown_score * weights['max_drawdown'] +
                rmse_score * weights['rmse']
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    def get_weight_history(self, model_name: str, days_back: int = 30) -> List[Tuple[datetime, float]]:
        """Get weight history for a specific model.
        
        Args:
            model_name: Name of the model
            days_back: Number of days to look back
            
        Returns:
            List of (timestamp, weight) tuples
        """
        try:
            history = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for timestamp, weights in self.weight_history.items():
                if timestamp >= cutoff_date and model_name in weights:
                    history.append((timestamp, weights[model_name]))
            
            return sorted(history, key=lambda x: x[0])
            
        except Exception as e:
            logger.error(f"Error getting weight history: {e}")
            return []

class AdaptiveSelector:
    """Main adaptive selector that coordinates model and strategy selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.volatility_analyzer = VolatilityAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.model_selector = ModelSelector()
        self.strategy_selector = StrategySelector()
        self.ensemble_optimizer = HybridEnsembleOptimizer()
        
        # Market conditions cache
        self.market_conditions_cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        logger.info("AdaptiveSelector initialized successfully")
    
    def analyze_market_conditions(self, price_data: pd.Series) -> MarketConditions:
        """Analyze current market conditions.
        
        Args:
            price_data: Price series data
            
        Returns:
            Market conditions
        """
        try:
            # Check cache first
            cache_key = price_data.index[-1]
            if cache_key in self.market_conditions_cache:
                cache_time, conditions = self.market_conditions_cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    return conditions
            
            # Analyze volatility
            volatility_regime = self.volatility_analyzer.analyze_volatility(price_data)
            volatility_score = self.volatility_analyzer.calculate_volatility_score(price_data)
            
            # Analyze trend
            market_trend = self.trend_analyzer.analyze_trend(price_data)
            trend_score = self.trend_analyzer.calculate_trend_score(price_data)
            
            # Calculate additional scores
            volume_score = self._calculate_volume_score(price_data)
            momentum_score = self._calculate_momentum_score(price_data)
            
            # Create market conditions
            conditions = MarketConditions(
                volatility_regime=volatility_regime,
                market_trend=market_trend,
                volatility_score=volatility_score,
                trend_score=trend_score,
                volume_score=volume_score,
                momentum_score=momentum_score,
                timestamp=datetime.now()
            )
            
            # Cache results
            self.market_conditions_cache[cache_key] = (datetime.now(), conditions)
            
            logger.info(f"Market conditions: {volatility_regime.value} volatility, {market_trend.value} trend")
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            # Return default conditions
            return MarketConditions(
                volatility_regime=VolatilityRegime.MEDIUM,
                market_trend=MarketTrend.NEUTRAL,
                volatility_score=0.5,
                trend_score=0.0,
                volume_score=0.5,
                momentum_score=0.0,
                timestamp=datetime.now()
            )
    
    def select_optimal_configuration(self, price_data: pd.Series,
                                   available_models: List[str],
                                   available_strategies: List[str]) -> Dict[str, Any]:
        """Select optimal model and strategy configuration.
        
        Args:
            price_data: Price series data
            available_models: List of available models
            available_strategies: List of available strategies
            
        Returns:
            Configuration dictionary
        """
        try:
            # Analyze market conditions
            market_conditions = self.analyze_market_conditions(price_data)
            
            # Select optimal model
            optimal_model = self.model_selector.select_optimal_model(
                market_conditions, available_models
            )
            
            # Select optimal strategy
            optimal_strategy = self.strategy_selector.select_optimal_strategy(
                market_conditions, available_strategies
            )
            
            # Determine if hybrid ensemble is needed
            use_hybrid = self._should_use_hybrid(market_conditions, available_models)
            
            configuration = {
                'market_conditions': market_conditions,
                'selected_model': optimal_model,
                'selected_strategy': optimal_strategy,
                'use_hybrid': use_hybrid,
                'confidence': self._calculate_selection_confidence(market_conditions)
            }
            
            if use_hybrid:
                # Get performance data for weight optimization
                performance_data = self._get_model_performance_data(available_models)
                optimized_weights = self.ensemble_optimizer.optimize_weights(
                    available_models, performance_data
                )
                configuration['ensemble_weights'] = optimized_weights
            
            logger.info(f"Selected configuration: {configuration['selected_model']} + {configuration['selected_strategy']}")
            
            return configuration
            
        except Exception as e:
            logger.error(f"Error selecting optimal configuration: {e}")
            # Return default configuration
            return {
                'market_conditions': MarketConditions(
                    VolatilityRegime.MEDIUM, MarketTrend.NEUTRAL, 0.5, 0.0, 0.5, 0.0, datetime.now()
                ),
                'selected_model': 'LSTM',
                'selected_strategy': 'RSI Mean Reversion',
                'use_hybrid': False,
                'confidence': 0.5
            }
    
    def _should_use_hybrid(self, market_conditions: MarketConditions, 
                          available_models: List[str]) -> bool:
        """Determine if hybrid ensemble should be used.
        
        Args:
            market_conditions: Current market conditions
            available_models: Available models
            
        Returns:
            True if hybrid ensemble should be used
        """
        try:
            # Use hybrid for high volatility or uncertain conditions
            if market_conditions.volatility_regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
                return True
            
            # Use hybrid if multiple models are available
            if len(available_models) >= 3:
                return True
            
            # Use hybrid for neutral/volatile trends
            if market_conditions.market_trend in [MarketTrend.NEUTRAL, MarketTrend.VOLATILE]:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining hybrid usage: {e}")
            return False
    
    def _calculate_selection_confidence(self, market_conditions: MarketConditions) -> float:
        """Calculate confidence in the selection.
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Higher confidence for clear market conditions
            confidence = 0.5
            
            # Adjust based on volatility regime clarity
            if market_conditions.volatility_regime in [VolatilityRegime.LOW, VolatilityRegime.EXTREME]:
                confidence += 0.2
            
            # Adjust based on trend clarity
            if market_conditions.market_trend in [MarketTrend.BULL, MarketTrend.BEAR]:
                confidence += 0.2
            
            # Adjust based on volume and momentum
            if market_conditions.volume_score > 0.7:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating selection confidence: {e}")
            return 0.5
    
    def _calculate_volume_score(self, price_data: pd.Series) -> float:
        """Calculate volume score."""
        try:
            # This would use actual volume data
            # For now, return a default score
            return 0.5
        except:
            return 0.5
    
    def _calculate_momentum_score(self, price_data: pd.Series) -> float:
        """Calculate momentum score."""
        try:
            # Calculate momentum over different periods
            momentum_5 = (price_data.iloc[-1] - price_data.iloc[-5]) / price_data.iloc[-5]
            momentum_10 = (price_data.iloc[-1] - price_data.iloc[-10]) / price_data.iloc[-10]
            
            # Combine momentum signals
            combined_momentum = (momentum_5 + momentum_10) / 2
            return np.tanh(combined_momentum / 0.01)
        except:
            return 0.0
    
    def _get_model_performance_data(self, models: List[str]) -> Dict[str, ModelPerformance]:
        """Get performance data for models."""
        try:
            # This would fetch actual performance data
            # For now, return simulated data
            performance_data = {}
            
            for model in models:
                performance_data[model] = ModelPerformance(
                    model_name=model,
                    rmse=0.02 + np.random.random() * 0.03,
                    mae=0.015 + np.random.random() * 0.025,
                    mape=2.0 + np.random.random() * 3.0,
                    sharpe_ratio=0.5 + np.random.random() * 1.5,
                    max_drawdown=0.05 + np.random.random() * 0.15,
                    win_rate=0.45 + np.random.random() * 0.3,
                    profit_factor=1.0 + np.random.random() * 2.0,
                    volatility_score=0.5,
                    trend_score=0.0,
                    overall_score=0.5 + np.random.random() * 0.5,
                    last_updated=datetime.now(),
                    regime_performance={}
                )
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting model performance data: {e}")
            return {}

def get_adaptive_selector(config: Optional[Dict[str, Any]] = None) -> AdaptiveSelector:
    """Get the adaptive selector instance."""
    return AdaptiveSelector(config) 