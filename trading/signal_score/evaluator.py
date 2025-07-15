"""
Signal Score Evaluator - Batch 19
Enhanced signal evaluation with multiple strategy support and numerical safety
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class StrategyType(Enum):
    """Supported strategy types."""
    RSI = "RSI"
    SMA = "SMA"
    MACD = "MACD"
    BB = "BB"  # Bollinger Bands
    CUSTOM = "custom"

@dataclass
class SignalScore:
    """Signal score with metadata."""
    signal_type: SignalType
    score: float
    confidence: float
    strategy: str
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationResult:
    """Result of signal evaluation."""
    signal_scores: List[SignalScore]
    composite_score: float
    recommended_action: SignalType
    confidence: float
    evaluation_time: float
    warnings: List[str] = field(default_factory=list)

class SignalScoreEvaluator:
    """
    Enhanced signal score evaluator with multiple strategy support.
    
    Features:
    - Support for SMA, BB, and custom strategies
    - Numerical safety with np.nan_to_num()
    - Composite scoring across multiple signals
    - Confidence-weighted aggregation
    """
    
    def __init__(self, 
                 enable_nan_protection: bool = True,
                 default_confidence_threshold: float = 0.3,
                 max_score: float = 1.0,
                 min_score: float = -1.0):
        """
        Initialize signal score evaluator.
        
        Args:
            enable_nan_protection: Enable NaN protection with np.nan_to_num()
            default_confidence_threshold: Minimum confidence for signal consideration
            max_score: Maximum allowed score
            min_score: Minimum allowed score
        """
        self.enable_nan_protection = enable_nan_protection
        self.default_confidence_threshold = default_confidence_threshold
        self.max_score = max_score
        self.min_score = min_score
        
        # Strategy evaluators
        self.strategy_evaluators = self._initialize_strategy_evaluators()
        
        # Custom strategy registry
        self.custom_strategies: Dict[str, Callable] = {}
        
        # Evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        
        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'nan_detected_count': 0,
            'strategy_usage': {},
            'avg_composite_score': 0.0
        }
        
        logger.info(f"SignalScoreEvaluator initialized with NaN protection: {enable_nan_protection}")
    
    def _initialize_strategy_evaluators(self) -> Dict[str, Callable]:
        """Initialize built-in strategy evaluators."""
        return {
            StrategyType.RSI.value: self._evaluate_rsi_signal,
            StrategyType.SMA.value: self._evaluate_sma_signal,
            StrategyType.MACD.value: self._evaluate_macd_signal,
            StrategyType.BB.value: self._evaluate_bollinger_signal
        }
    
    def evaluate_signal(self, 
                       signal_data: Dict[str, Any],
                       strategy_type: str,
                       parameters: Optional[Dict[str, Any]] = None) -> SignalScore:
        """
        Evaluate a single signal.
        
        Args:
            signal_data: Signal data dictionary
            strategy_type: Type of strategy
            parameters: Strategy parameters
            
        Returns:
            SignalScore object
        """
        try:
            # Get evaluator function
            if strategy_type in self.strategy_evaluators:
                evaluator = self.strategy_evaluators[strategy_type]
            elif strategy_type in self.custom_strategies:
                evaluator = self.custom_strategies[strategy_type]
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
                return self._create_default_score(strategy_type)
            
            # Evaluate signal
            score, confidence, signal_type = evaluator(signal_data, parameters or {})
            
            # Apply NaN protection
            if self.enable_nan_protection:
                score = self._apply_nan_protection(score)
                confidence = self._apply_nan_protection(confidence)
            
            # Create signal score
            signal_score = SignalScore(
                signal_type=signal_type,
                score=score,
                confidence=confidence,
                strategy=strategy_type,
                timestamp=datetime.now(),
                parameters=parameters or {}
            )
            
            # Update statistics
            self._update_strategy_usage(strategy_type)
            
            return signal_score
            
        except Exception as e:
            logger.error(f"Error evaluating {strategy_type} signal: {e}")
            return self._create_error_score(strategy_type, str(e))
    
    def _evaluate_rsi_signal(self, 
                           signal_data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> Tuple[float, float, SignalType]:
        """Evaluate RSI signal."""
        rsi_value = signal_data.get('rsi', 50.0)
        oversold_threshold = parameters.get('oversold', 30.0)
        overbought_threshold = parameters.get('overbought', 70.0)
        
        # Apply NaN protection
        if self.enable_nan_protection:
            rsi_value = self._apply_nan_protection(rsi_value)
        
        # Calculate score
        if rsi_value <= oversold_threshold:
            score = -0.8 + (oversold_threshold - rsi_value) / oversold_threshold * 0.2
            signal_type = SignalType.STRONG_BUY
            confidence = 0.9
        elif rsi_value >= overbought_threshold:
            score = 0.8 + (rsi_value - overbought_threshold) / (100 - overbought_threshold) * 0.2
            signal_type = SignalType.STRONG_SELL
            confidence = 0.9
        else:
            # Neutral zone
            neutral_center = (oversold_threshold + overbought_threshold) / 2
            distance_from_center = abs(rsi_value - neutral_center)
            max_distance = (overbought_threshold - oversold_threshold) / 2
            
            if rsi_value < neutral_center:
                score = -0.3 * (distance_from_center / max_distance)
                signal_type = SignalType.BUY
            else:
                score = 0.3 * (distance_from_center / max_distance)
                signal_type = SignalType.SELL
            
            confidence = 0.6
        
        return score, confidence, signal_type
    
    def _evaluate_sma_signal(self, 
                           signal_data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> Tuple[float, float, SignalType]:
        """Evaluate SMA signal."""
        current_price = signal_data.get('current_price', 100.0)
        sma_short = signal_data.get('sma_short', 100.0)
        sma_long = signal_data.get('sma_long', 100.0)
        
        # Apply NaN protection
        if self.enable_nan_protection:
            current_price = self._apply_nan_protection(current_price)
            sma_short = self._apply_nan_protection(sma_short)
            sma_long = self._apply_nan_protection(sma_long)
        
        # Calculate price position relative to SMAs
        short_ratio = current_price / sma_short if sma_short > 0 else 1.0
        long_ratio = current_price / sma_long if sma_long > 0 else 1.0
        
        # Determine signal
        if short_ratio > 1.02 and long_ratio > 1.02:
            # Strong uptrend
            score = 0.8
            signal_type = SignalType.STRONG_BUY
            confidence = 0.85
        elif short_ratio < 0.98 and long_ratio < 0.98:
            # Strong downtrend
            score = -0.8
            signal_type = SignalType.STRONG_SELL
            confidence = 0.85
        elif short_ratio > 1.01 and long_ratio > 1.0:
            # Weak uptrend
            score = 0.4
            signal_type = SignalType.BUY
            confidence = 0.7
        elif short_ratio < 0.99 and long_ratio < 1.0:
            # Weak downtrend
            score = -0.4
            signal_type = SignalType.SELL
            confidence = 0.7
        else:
            # No clear trend
            score = 0.0
            signal_type = SignalType.HOLD
            confidence = 0.5
        
        return score, confidence, signal_type
    
    def _evaluate_macd_signal(self, 
                            signal_data: Dict[str, Any], 
                            parameters: Dict[str, Any]) -> Tuple[float, float, SignalType]:
        """Evaluate MACD signal."""
        macd_line = signal_data.get('macd_line', 0.0)
        signal_line = signal_data.get('signal_line', 0.0)
        histogram = signal_data.get('histogram', 0.0)
        
        # Apply NaN protection
        if self.enable_nan_protection:
            macd_line = self._apply_nan_protection(macd_line)
            signal_line = self._apply_nan_protection(signal_line)
            histogram = self._apply_nan_protection(histogram)
        
        # Calculate signal strength
        macd_diff = macd_line - signal_line
        histogram_abs = abs(histogram)
        
        # Determine signal
        if macd_diff > 0 and histogram > 0:
            # Bullish MACD
            if histogram_abs > 0.1:
                score = 0.8
                signal_type = SignalType.STRONG_BUY
                confidence = 0.9
            else:
                score = 0.5
                signal_type = SignalType.BUY
                confidence = 0.75
        elif macd_diff < 0 and histogram < 0:
            # Bearish MACD
            if histogram_abs > 0.1:
                score = -0.8
                signal_type = SignalType.STRONG_SELL
                confidence = 0.9
            else:
                score = -0.5
                signal_type = SignalType.SELL
                confidence = 0.75
        else:
            # Neutral or conflicting signals
            score = 0.0
            signal_type = SignalType.HOLD
            confidence = 0.5
        
        return score, confidence, signal_type
    
    def _evaluate_bollinger_signal(self, 
                                 signal_data: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> Tuple[float, float, SignalType]:
        """Evaluate Bollinger Bands signal."""
        current_price = signal_data.get('current_price', 100.0)
        upper_band = signal_data.get('upper_band', 105.0)
        lower_band = signal_data.get('lower_band', 95.0)
        middle_band = signal_data.get('middle_band', 100.0)
        
        # Apply NaN protection
        if self.enable_nan_protection:
            current_price = self._apply_nan_protection(current_price)
            upper_band = self._apply_nan_protection(upper_band)
            lower_band = self._apply_nan_protection(lower_band)
            middle_band = self._apply_nan_protection(middle_band)
        
        # Calculate position within bands
        band_width = upper_band - lower_band
        if band_width <= 0:
            return 0.0, 0.3, SignalType.HOLD
        
        position = (current_price - lower_band) / band_width
        
        # Determine signal
        if position <= 0.1:
            # Near lower band - potential buy
            score = -0.8
            signal_type = SignalType.STRONG_BUY
            confidence = 0.85
        elif position >= 0.9:
            # Near upper band - potential sell
            score = 0.8
            signal_type = SignalType.STRONG_SELL
            confidence = 0.85
        elif position <= 0.3:
            # Below middle - weak buy
            score = -0.4
            signal_type = SignalType.BUY
            confidence = 0.7
        elif position >= 0.7:
            # Above middle - weak sell
            score = 0.4
            signal_type = SignalType.SELL
            confidence = 0.7
        else:
            # Middle range - hold
            score = 0.0
            signal_type = SignalType.HOLD
            confidence = 0.5
        
        return score, confidence, signal_type
    
    def _apply_nan_protection(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Apply NaN protection to numerical values.
        
        Args:
            value: Input value or array
            
        Returns:
            Cleaned value with NaN replaced
        """
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                self.stats['nan_detected_count'] += 1
                return 0.0
            return value
        elif isinstance(value, np.ndarray):
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                self.stats['nan_detected_count'] += 1
                return np.nan_to_num(value, nan=0.0, posinf=self.max_score, neginf=self.min_score)
            return value
        else:
            return value
    
    def _create_default_score(self, strategy_type: str) -> SignalScore:
        """Create default score for unknown strategy."""
        return SignalScore(
            signal_type=SignalType.HOLD,
            score=0.0,
            confidence=0.3,
            strategy=strategy_type,
            timestamp=datetime.now(),
            parameters={},
            metadata={'error': 'Unknown strategy type'}
        )
    
    def _create_error_score(self, strategy_type: str, error_message: str) -> SignalScore:
        """Create error score for failed evaluation."""
        return SignalScore(
            signal_type=SignalType.HOLD,
            score=0.0,
            confidence=0.1,
            strategy=strategy_type,
            timestamp=datetime.now(),
            parameters={},
            metadata={'error': error_message}
        )
    
    def evaluate_multiple_signals(self, 
                                signals: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate multiple signals and create composite score.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            EvaluationResult with composite analysis
        """
        start_time = datetime.now()
        warnings = []
        
        # Evaluate individual signals
        signal_scores = []
        for signal in signals:
            strategy_type = signal.get('strategy_type', 'unknown')
            signal_data = signal.get('signal_data', {})
            parameters = signal.get('parameters', {})
            
            score = self.evaluate_signal(signal_data, strategy_type, parameters)
            signal_scores.append(score)
        
        # Calculate composite score
        composite_score, confidence, recommended_action = self._calculate_composite_score(signal_scores)
        
        # Apply NaN protection to composite score
        if self.enable_nan_protection:
            composite_score = self._apply_nan_protection(composite_score)
            confidence = self._apply_nan_protection(confidence)
        
        # Check for warnings
        if self.stats['nan_detected_count'] > 0:
            warnings.append(f"NaN values detected and corrected: {self.stats['nan_detected_count']}")
        
        low_confidence_signals = [s for s in signal_scores if s.confidence < self.default_confidence_threshold]
        if low_confidence_signals:
            warnings.append(f"Low confidence signals detected: {len(low_confidence_signals)}")
        
        # Create result
        evaluation_time = (datetime.now() - start_time).total_seconds()
        result = EvaluationResult(
            signal_scores=signal_scores,
            composite_score=composite_score,
            recommended_action=recommended_action,
            confidence=confidence,
            evaluation_time=evaluation_time,
            warnings=warnings
        )
        
        # Store in history
        self.evaluation_history.append(result)
        self.stats['total_evaluations'] += 1
        
        # Update average composite score
        self._update_average_score(composite_score)
        
        logger.info(f"Evaluated {len(signals)} signals, composite score: {composite_score:.3f}")
        return result
    
    def _calculate_composite_score(self, 
                                 signal_scores: List[SignalScore]) -> Tuple[float, float, SignalType]:
        """
        Calculate composite score from multiple signal scores.
        
        Args:
            signal_scores: List of individual signal scores
            
        Returns:
            Tuple of (composite_score, confidence, recommended_action)
        """
        if not signal_scores:
            return 0.0, 0.0, SignalType.HOLD
        
        # Filter by confidence threshold
        valid_scores = [s for s in signal_scores if s.confidence >= self.default_confidence_threshold]
        
        if not valid_scores:
            return 0.0, 0.0, SignalType.HOLD
        
        # Calculate confidence-weighted average
        total_weight = sum(s.confidence for s in valid_scores)
        if total_weight <= 0:
            return 0.0, 0.0, SignalType.HOLD
        
        weighted_score = sum(s.score * s.confidence for s in valid_scores) / total_weight
        avg_confidence = total_weight / len(valid_scores)
        
        # Determine recommended action
        if weighted_score >= 0.6:
            recommended_action = SignalType.STRONG_BUY
        elif weighted_score >= 0.2:
            recommended_action = SignalType.BUY
        elif weighted_score <= -0.6:
            recommended_action = SignalType.STRONG_SELL
        elif weighted_score <= -0.2:
            recommended_action = SignalType.SELL
        else:
            recommended_action = SignalType.HOLD
        
        return weighted_score, avg_confidence, recommended_action
    
    def register_custom_strategy(self, 
                               strategy_name: str, 
                               evaluator_function: Callable) -> bool:
        """
        Register a custom strategy evaluator.
        
        Args:
            strategy_name: Name of the custom strategy
            evaluator_function: Function that evaluates signals
            
        Returns:
            Success status
        """
        try:
            self.custom_strategies[strategy_name] = evaluator_function
            logger.info(f"Registered custom strategy: {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register custom strategy {strategy_name}: {e}")
            return False
    
    def _update_strategy_usage(self, strategy_type: str):
        """Update strategy usage statistics."""
        self.stats['strategy_usage'][strategy_type] = self.stats['strategy_usage'].get(strategy_type, 0) + 1
    
    def _update_average_score(self, new_score: float):
        """Update average composite score."""
        total_evaluations = self.stats['total_evaluations']
        current_avg = self.stats['avg_composite_score']
        
        if total_evaluations == 1:
            self.stats['avg_composite_score'] = new_score
        else:
            self.stats['avg_composite_score'] = (current_avg * (total_evaluations - 1) + new_score) / total_evaluations
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        stats = self.stats.copy()
        
        # Add recent performance metrics
        if self.evaluation_history:
            recent_results = self.evaluation_history[-10:]  # Last 10 evaluations
            stats['recent_avg_score'] = np.mean([r.composite_score for r in recent_results])
            stats['recent_avg_confidence'] = np.mean([r.confidence for r in recent_results])
            stats['recent_warning_count'] = sum(len(r.warnings) for r in recent_results)
        
        return stats
    
    def enable_nan_protection(self, enable: bool = True):
        """Enable or disable NaN protection."""
        self.enable_nan_protection = enable
        logger.info(f"NaN protection {'enabled' if enable else 'disabled'}")

def create_signal_score_evaluator(enable_nan_protection: bool = True) -> SignalScoreEvaluator:
    """Factory function to create signal score evaluator."""
    return SignalScoreEvaluator(enable_nan_protection=enable_nan_protection) 