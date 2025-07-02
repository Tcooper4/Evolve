"""
Multi-Strategy Hybrid Engine

Combines multiple strategies with conditional filters and confidence scoring.
Provides ensemble predictions and risk-adjusted position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal types for strategy outputs."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class StrategySignal:
    """Individual strategy signal."""
    strategy_name: str
    signal_type: SignalType
    confidence: float
    predicted_return: float
    position_size: float
    risk_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class HybridSignal:
    """Combined hybrid signal."""
    signal_type: SignalType
    confidence: float
    predicted_return: float
    position_size: float
    risk_score: float
    strategy_weights: Dict[str, float]
    individual_signals: List[StrategySignal]
    timestamp: datetime
    metadata: Dict[str, Any]

class MultiStrategyHybridEngine:
    """Advanced multi-strategy hybrid engine with ensemble learning."""
    
    def __init__(self, 
                 strategies: Optional[Dict[str, Callable]] = None,
                 ensemble_method: str = "weighted_average",
                 confidence_threshold: float = 0.6,
                 max_position_size: float = 1.0,
                 risk_free_rate: float = 0.02):
        """Initialize the hybrid engine.
        
        Args:
            strategies: Dictionary of strategy functions
            ensemble_method: Method for combining strategies
            confidence_threshold: Minimum confidence for signal generation
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.strategies = strategies or {}
        self.ensemble_method = ensemble_method
        self.confidence_threshold = confidence_threshold
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        self.scaler = StandardScaler()
        self.ensemble_model = None
        self.strategy_weights = {}
        self.signal_history = []
        self.performance_history = []
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        # Initialize ensemble model
        self._initialize_ensemble_model()
        
        logger.info("Multi-Strategy Hybrid Engine initialized successfully")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _initialize_default_strategies(self):
        """Initialize default trading strategies."""
        if not self.strategies:
            self.strategies = {
                'momentum': self._momentum_strategy,
                'mean_reversion': self._mean_reversion_strategy,
                'volatility_breakout': self._volatility_breakout_strategy,
                'trend_following': self._trend_following_strategy,
                'volume_price': self._volume_price_strategy
            }
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _initialize_ensemble_model(self):
        """Initialize the ensemble model."""
        try:
            if self.ensemble_method == "voting":
                # Create voting regressor with individual strategy models
                estimators = []
                for strategy_name in self.strategies.keys():
                    estimator = (strategy_name, LinearRegression())
                    estimators.append(estimator)
                
                self.ensemble_model = VotingRegressor(estimators=estimators)
            
            elif self.ensemble_method == "weighted_average":
                # Initialize equal weights
                n_strategies = len(self.strategies)
                self.strategy_weights = {
                    name: 1.0 / n_strategies 
                    for name in self.strategies.keys()
                }
            
            logger.info(f"Ensemble model initialized with method: {self.ensemble_method}")
            
        except Exception as e:
            logger.error(f"Error initializing ensemble model: {e}")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _momentum_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Momentum-based strategy."""
        try:
            # Calculate momentum indicators
            returns = data['Close'].pct_change()
            momentum_5 = returns.rolling(5).mean()
            momentum_20 = returns.rolling(20).mean()
            
            # Current momentum
            current_momentum = momentum_5.iloc[-1]
            long_momentum = momentum_20.iloc[-1]
            
            # Signal generation
            if current_momentum > 0 and long_momentum > 0:
                signal_type = SignalType.BUY
                confidence = min(0.9, abs(current_momentum) * 10)
                predicted_return = current_momentum * 252  # Annualized
            elif current_momentum < 0 and long_momentum < 0:
                signal_type = SignalType.SELL
                confidence = min(0.9, abs(current_momentum) * 10)
                predicted_return = current_momentum * 252
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
                predicted_return = 0.0
            
            # Risk score based on volatility
            volatility = returns.rolling(20).std().iloc[-1]
            risk_score = min(1.0, volatility * np.sqrt(252))
            
            # Position sizing
            position_size = self._calculate_position_size(confidence, risk_score)
            
            return StrategySignal(
                strategy_name='momentum',
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=predicted_return,
                position_size=position_size,
                risk_score=risk_score,
                timestamp=datetime.now(),
                metadata={
                    'current_momentum': current_momentum,
                    'long_momentum': long_momentum,
                    'volatility': volatility
                }
            )
            
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return self._create_fallback_signal('momentum')
    
    def _mean_reversion_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Mean reversion strategy."""
        try:
            # Calculate Bollinger Bands
            sma = data['Close'].rolling(20).mean()
            std = data['Close'].rolling(20).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            current_price = data['Close'].iloc[-1]
            current_sma = sma.iloc[-1]
            
            # Position within bands
            band_position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            
            # Signal generation
            if band_position < 0.2:  # Near lower band
                signal_type = SignalType.BUY
                confidence = 0.8
                predicted_return = 0.05  # 5% expected return
            elif band_position > 0.8:  # Near upper band
                signal_type = SignalType.SELL
                confidence = 0.8
                predicted_return = -0.05
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
                predicted_return = 0.0
            
            # Risk score
            volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]
            risk_score = min(1.0, volatility * np.sqrt(252))
            
            # Position sizing
            position_size = self._calculate_position_size(confidence, risk_score)
            
            return StrategySignal(
                strategy_name='mean_reversion',
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=predicted_return,
                position_size=position_size,
                risk_score=risk_score,
                timestamp=datetime.now(),
                metadata={
                    'band_position': band_position,
                    'current_sma': current_sma,
                    'volatility': volatility
                }
            )
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return self._create_fallback_signal('mean_reversion')
    
    def _volatility_breakout_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Volatility breakout strategy."""
        try:
            # Calculate volatility
            returns = data['Close'].pct_change()
            current_vol = returns.rolling(5).std().iloc[-1]
            historical_vol = returns.rolling(20).std().iloc[-1]
            
            # Volatility ratio
            vol_ratio = current_vol / historical_vol
            
            # Price action
            current_return = returns.iloc[-1]
            recent_returns = returns.tail(5)
            
            # Signal generation
            if vol_ratio > 1.5 and current_return > 0:
                signal_type = SignalType.BUY
                confidence = min(0.9, vol_ratio * 0.3)
                predicted_return = current_return * 252
            elif vol_ratio > 1.5 and current_return < 0:
                signal_type = SignalType.SELL
                confidence = min(0.9, vol_ratio * 0.3)
                predicted_return = current_return * 252
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
                predicted_return = 0.0
            
            # Risk score (higher for volatility breakouts)
            risk_score = min(1.0, vol_ratio * 0.5)
            
            # Position sizing
            position_size = self._calculate_position_size(confidence, risk_score)
            
            return StrategySignal(
                strategy_name='volatility_breakout',
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=predicted_return,
                position_size=position_size,
                risk_score=risk_score,
                timestamp=datetime.now(),
                metadata={
                    'volatility_ratio': vol_ratio,
                    'current_volatility': current_vol,
                    'historical_volatility': historical_vol
                }
            )
            
        except Exception as e:
            logger.error(f"Error in volatility breakout strategy: {e}")
            return self._create_fallback_signal('volatility_breakout')
    
    def _trend_following_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Trend following strategy."""
        try:
            # Calculate moving averages
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            sma_200 = data['Close'].rolling(200).mean()
            
            current_price = data['Close'].iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            current_sma_200 = sma_200.iloc[-1]
            
            # Trend strength
            trend_strength = (current_sma_20 - current_sma_200) / current_sma_200
            
            # Signal generation
            if current_price > current_sma_20 > current_sma_50 > current_sma_200:
                signal_type = SignalType.STRONG_BUY
                confidence = min(0.9, abs(trend_strength) * 5 + 0.5)
                predicted_return = trend_strength * 252
            elif current_price < current_sma_20 < current_sma_50 < current_sma_200:
                signal_type = SignalType.STRONG_SELL
                confidence = min(0.9, abs(trend_strength) * 5 + 0.5)
                predicted_return = trend_strength * 252
            elif current_price > current_sma_20 and current_sma_20 > current_sma_50:
                signal_type = SignalType.BUY
                confidence = 0.7
                predicted_return = trend_strength * 252
            elif current_price < current_sma_20 and current_sma_20 < current_sma_50:
                signal_type = SignalType.SELL
                confidence = 0.7
                predicted_return = trend_strength * 252
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
                predicted_return = 0.0
            
            # Risk score
            volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]
            risk_score = min(1.0, volatility * np.sqrt(252))
            
            # Position sizing
            position_size = self._calculate_position_size(confidence, risk_score)
            
            return StrategySignal(
                strategy_name='trend_following',
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=predicted_return,
                position_size=position_size,
                risk_score=risk_score,
                timestamp=datetime.now(),
                metadata={
                    'trend_strength': trend_strength,
                    'sma_20': current_sma_20,
                    'sma_50': current_sma_50,
                    'sma_200': current_sma_200
                }
            )
            
        except Exception as e:
            logger.error(f"Error in trend following strategy: {e}")
            return self._create_fallback_signal('trend_following')
    
    def _volume_price_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Volume-price relationship strategy."""
        try:
            if 'Volume' not in data.columns:
                return self._create_fallback_signal('volume_price')
            
            # Calculate volume indicators
            volume_ma = data['Volume'].rolling(20).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma.iloc[-1]
            
            # Price action
            returns = data['Close'].pct_change()
            current_return = returns.iloc[-1]
            
            # Volume-price divergence
            price_trend = data['Close'].rolling(5).mean().diff().iloc[-1]
            volume_trend = data['Volume'].rolling(5).mean().diff().iloc[-1]
            
            # Signal generation
            if volume_ratio > 1.5 and current_return > 0:
                signal_type = SignalType.BUY
                confidence = min(0.9, volume_ratio * 0.2)
                predicted_return = current_return * 252
            elif volume_ratio > 1.5 and current_return < 0:
                signal_type = SignalType.SELL
                confidence = min(0.9, volume_ratio * 0.2)
                predicted_return = current_return * 252
            elif volume_ratio < 0.5:
                signal_type = SignalType.HOLD
                confidence = 0.6
                predicted_return = 0.0
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
                predicted_return = 0.0
            
            # Risk score
            volatility = returns.rolling(20).std().iloc[-1]
            risk_score = min(1.0, volatility * np.sqrt(252))
            
            # Position sizing
            position_size = self._calculate_position_size(confidence, risk_score)
            
            return StrategySignal(
                strategy_name='volume_price',
                signal_type=signal_type,
                confidence=confidence,
                predicted_return=predicted_return,
                position_size=position_size,
                risk_score=risk_score,
                timestamp=datetime.now(),
                metadata={
                    'volume_ratio': volume_ratio,
                    'price_trend': price_trend,
                    'volume_trend': volume_trend
                }
            )
            
        except Exception as e:
            logger.error(f"Error in volume price strategy: {e}")
            return self._create_fallback_signal('volume_price')
    
    def _create_fallback_signal(self, strategy_name: str) -> StrategySignal:
        """Create a fallback signal when strategy fails."""
        return StrategySignal(
            strategy_name=strategy_name,
            signal_type=SignalType.HOLD,
            confidence=0.3,
            predicted_return=0.0,
            position_size=0.0,
            risk_score=0.5,
            timestamp=datetime.now(),
            metadata={'error': 'Strategy failed, using fallback'}
        )
    
    def _calculate_position_size(self, confidence: float, risk_score: float) -> float:
        """Calculate position size based on confidence and risk."""
        try:
            # Base position size from confidence
            base_size = confidence * self.max_position_size
            
            # Adjust for risk
            risk_adjustment = 1.0 - (risk_score * 0.5)
            adjusted_size = base_size * risk_adjustment
            
            # Ensure within bounds
            return max(0.0, min(self.max_position_size, adjusted_size))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def generate_hybrid_signal(self, data: pd.DataFrame) -> HybridSignal:
        """Generate hybrid signal by combining all strategies."""
        try:
            # Generate individual strategy signals
            individual_signals = []
            for strategy_name, strategy_func in self.strategies.items():
                signal = strategy_func(data)
                individual_signals.append(signal)
            
            # Combine signals based on ensemble method
            if self.ensemble_method == "weighted_average":
                hybrid_signal = self._combine_weighted_average(individual_signals)
            elif self.ensemble_method == "voting":
                hybrid_signal = self._combine_voting(individual_signals)
            else:
                hybrid_signal = self._combine_weighted_average(individual_signals)
            
            # Store signal history
            self.signal_history.append({
                'timestamp': hybrid_signal.timestamp.isoformat(),
                'signal_type': hybrid_signal.signal_type.value,
                'confidence': hybrid_signal.confidence,
                'predicted_return': hybrid_signal.predicted_return,
                'position_size': hybrid_signal.position_size,
                'risk_score': hybrid_signal.risk_score
            })
            
            # Keep only last 1000 signals
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            logger.info(f"Hybrid signal generated: {hybrid_signal.signal_type.value} "
                       f"(confidence: {hybrid_signal.confidence:.2f})")
            
            return hybrid_signal
            
        except Exception as e:
            logger.error(f"Error generating hybrid signal: {e}")
            return self._create_fallback_hybrid_signal()
    
    def _combine_weighted_average(self, signals: List[StrategySignal]) -> HybridSignal:
        """Combine signals using weighted average."""
        try:
            # Calculate weighted average of predicted returns
            weighted_return = 0.0
            weighted_confidence = 0.0
            weighted_risk = 0.0
            total_weight = 0.0
            
            for signal in signals:
                weight = self.strategy_weights.get(signal.strategy_name, 1.0 / len(signals))
                weighted_return += signal.predicted_return * weight
                weighted_confidence += signal.confidence * weight
                weighted_risk += signal.risk_score * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_return = weighted_return / total_weight
                avg_confidence = weighted_confidence / total_weight
                avg_risk = weighted_risk / total_weight
            else:
                avg_return = 0.0
                avg_confidence = 0.5
                avg_risk = 0.5
            
            # Determine signal type
            if avg_confidence >= self.confidence_threshold:
                if avg_return > 0.05:  # 5% annualized return threshold
                    signal_type = SignalType.BUY
                elif avg_return < -0.05:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.HOLD
            else:
                signal_type = SignalType.HOLD
            
            # Calculate position size
            position_size = self._calculate_position_size(avg_confidence, avg_risk)
            
            return HybridSignal(
                signal_type=signal_type,
                confidence=avg_confidence,
                predicted_return=avg_return,
                position_size=position_size,
                risk_score=avg_risk,
                strategy_weights=self.strategy_weights.copy(),
                individual_signals=signals,
                timestamp=datetime.now(),
                metadata={
                    'ensemble_method': self.ensemble_method,
                    'n_strategies': len(signals)
                }
            )
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return self._create_fallback_hybrid_signal()
    
    def _combine_voting(self, signals: List[StrategySignal]) -> HybridSignal:
        """Combine signals using voting mechanism."""
        try:
            # Count votes for each signal type
            vote_counts = {
                SignalType.STRONG_BUY: 0,
                SignalType.BUY: 0,
                SignalType.HOLD: 0,
                SignalType.SELL: 0,
                SignalType.STRONG_SELL: 0
            }
            
            for signal in signals:
                vote_counts[signal.signal_type] += 1
            
            # Determine majority signal
            signal_type = max(vote_counts, key=vote_counts.get)
            
            # Calculate average metrics
            avg_confidence = np.mean([s.confidence for s in signals])
            avg_return = np.mean([s.predicted_return for s in signals])
            avg_risk = np.mean([s.risk_score for s in signals])
            
            # Calculate position size
            position_size = self._calculate_position_size(avg_confidence, avg_risk)
            
            return HybridSignal(
                signal_type=signal_type,
                confidence=avg_confidence,
                predicted_return=avg_return,
                position_size=position_size,
                risk_score=avg_risk,
                strategy_weights=self.strategy_weights.copy(),
                individual_signals=signals,
                timestamp=datetime.now(),
                metadata={
                    'ensemble_method': self.ensemble_method,
                    'vote_counts': {k.value: v for k, v in vote_counts.items()}
                }
            )
            
        except Exception as e:
            logger.error(f"Error in voting combination: {e}")
            return self._create_fallback_hybrid_signal()
    
    def _create_fallback_hybrid_signal(self) -> HybridSignal:
        """Create fallback hybrid signal."""
        return HybridSignal(
            signal_type=SignalType.HOLD,
            confidence=0.3,
            predicted_return=0.0,
            position_size=0.0,
            risk_score=0.5,
            strategy_weights={},
            individual_signals=[],
            timestamp=datetime.now(),
            metadata={'error': 'Hybrid signal generation failed'}
        )
    
    def update_strategy_weights(self, performance_data: Dict[str, float]):
        """Update strategy weights based on recent performance."""
        try:
            if not performance_data:
                return

            # Calculate new weights based on performance
            total_performance = sum(performance_data.values())
            
            if total_performance > 0:
                new_weights = {}
                for strategy_name, performance in performance_data.items():
                    if strategy_name in self.strategy_weights:
                        new_weights[strategy_name] = performance / total_performance
                
                # Update weights with smoothing
                alpha = 0.1  # Learning rate
                for strategy_name in self.strategy_weights:
                    if strategy_name in new_weights:
                        self.strategy_weights[strategy_name] = (
                            (1 - alpha) * self.strategy_weights[strategy_name] +
                            alpha * new_weights[strategy_name]
                        )
                
                # Normalize weights
                total_weight = sum(self.strategy_weights.values())
                for strategy_name in self.strategy_weights:
                    self.strategy_weights[strategy_name] /= total_weight
                
                logger.info("Strategy weights updated based on performance")
            
        except Exception as e:
            logger.error(f"Error updating strategy weights: {e}")
    
    def update_weights_based_on_error(self, error_history: Dict[str, List[float]]):
        """Update strategy weights based on model error over time.
        
        Args:
            error_history: Dictionary mapping strategy names to lists of recent errors
        """
        try:
            if not error_history:
                return
            
            # Calculate average error for each strategy
            strategy_errors = {}
            for strategy_name, errors in error_history.items():
                if errors and strategy_name in self.strategy_weights:
                    # Use inverse of mean error (lower error = higher weight)
                    mean_error = np.mean(errors)
                    if mean_error > 0:
                        strategy_errors[strategy_name] = 1.0 / mean_error
                    else:
                        strategy_errors[strategy_name] = 1.0
            
            if strategy_errors:
                # Normalize weights based on inverse errors
                total_inverse_error = sum(strategy_errors.values())
                if total_inverse_error > 0:
                    new_weights = {}
                    for strategy_name, inverse_error in strategy_errors.items():
                        new_weights[strategy_name] = inverse_error / total_inverse_error
                    
                    # Update weights with exponential moving average
                    alpha = 0.05  # Slower learning rate for error-based updates
                    for strategy_name in self.strategy_weights:
                        if strategy_name in new_weights:
                            self.strategy_weights[strategy_name] = (
                                (1 - alpha) * self.strategy_weights[strategy_name] +
                                alpha * new_weights[strategy_name]
                            )
                    
                    # Normalize weights
                    total_weight = sum(self.strategy_weights.values())
                    for strategy_name in self.strategy_weights:
                        self.strategy_weights[strategy_name] /= total_weight
                    
                    logger.info("Strategy weights updated based on error history")
        
        except Exception as e:
            logger.error(f"Error updating weights based on error: {e}")
    
    def auto_update_weights(self, recent_performance: Dict[str, float] = None, 
                          error_history: Dict[str, List[float]] = None):
        """Automatically update weights based on both performance and error history."""
        try:
            # Update based on performance if available
            if recent_performance:
                self.update_strategy_weights(recent_performance)
            
            # Update based on error history if available
            if error_history:
                self.update_weights_based_on_error(error_history)
            
            # Log current weights
            logger.info(f"Current strategy weights: {self.strategy_weights}")
            
        except Exception as e:
            logger.error(f"Error in auto weight update: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the hybrid engine."""
        try:
            if not self.signal_history:
                return {'message': 'No signal history available'}
            
            # Calculate performance metrics
            recent_signals = self.signal_history[-100:]  # Last 100 signals
            
            signal_types = [s['signal_type'] for s in recent_signals]
            confidences = [s['confidence'] for s in recent_signals]
            predicted_returns = [s['predicted_return'] for s in recent_signals]
            
            # Signal distribution
            signal_distribution = {}
            for signal_type in set(signal_types):
                signal_distribution[signal_type] = signal_types.count(signal_type) / len(signal_types)
            
            # Performance metrics
            avg_confidence = np.mean(confidences)
            avg_predicted_return = np.mean(predicted_returns)
            signal_volatility = np.std(predicted_returns)
            
            return {
                'total_signals': len(self.signal_history),
                'recent_signals': len(recent_signals),
                'signal_distribution': signal_distribution,
                'avg_confidence': avg_confidence,
                'avg_predicted_return': avg_predicted_return,
                'signal_volatility': signal_volatility,
                'strategy_weights': self.strategy_weights,
                'ensemble_method': self.ensemble_method,
                'confidence_threshold': self.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def export_signals(self, filepath: str = "logs/hybrid_signals.json"):
        """Export signal history to file."""
        try:
            export_data = {
                'signal_history': self.signal_history,
                'strategy_weights': self.strategy_weights,
                'performance_summary': self.get_performance_summary(),
                'export_date': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Hybrid signals exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting signals: {e}") 
