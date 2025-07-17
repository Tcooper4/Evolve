"""
Strategy Synthesizer for combining multiple technical indicators.

This module provides a framework for combining signals from different technical
indicators (RSI, MACD, Bollinger Bands) using weighted averaging and conflict
resolution mechanisms.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .rsi_signals import generate_rsi_signals
from .macd_strategy import MACDStrategy
from .bollinger_strategy import BollingerStrategy

logger = logging.getLogger(__name__)


class StrategySynthesizer:
    """Combines signals from multiple technical indicators with conflict resolution."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy synthesizer.
        
        Args:
            config: Configuration dictionary with weights and settings
        """
        self.config = config or {}
        
        # Default weights for each strategy
        self.weights = self.config.get('weights', {
            'rsi': 0.3,
            'macd': 0.4,
            'bollinger': 0.3
        })
        
        # Conflict resolution method
        self.conflict_resolution = self.config.get('conflict_resolution', 'majority_vote')
        
        # Confidence thresholds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.min_agreement = self.config.get('min_agreement', 0.5)
        
        # Initialize individual strategies
        self.rsi_strategy = RSISignals()
        self.macd_strategy = MACDStrategy()
        self.bollinger_strategy = BollingerStrategy()
        
        # Strategy instances
        self.strategies = {
            'rsi': self.rsi_strategy,
            'macd': self.macd_strategy,
            'bollinger': self.bollinger_strategy
        }
        
        logger.info(f"Strategy synthesizer initialized with weights: {self.weights}")

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate signals from all individual strategies.
        
        Args:
            data: Price data with OHLC columns
            
        Returns:
            Dictionary of signals from each strategy
        """
        try:
            signals = {}
            
            # Generate RSI signals
            try:
                rsi_signals = self.rsi_strategy.generate_signals(data)
                signals['rsi'] = rsi_signals
                logger.debug("RSI signals generated successfully")
            except Exception as e:
                logger.error(f"Error generating RSI signals: {e}")
                signals['rsi'] = pd.Series(0, index=data.index)
            
            # Generate MACD signals
            try:
                macd_signals = self.macd_strategy.generate_signals(data)
                signals['macd'] = macd_signals
                logger.debug("MACD signals generated successfully")
            except Exception as e:
                logger.error(f"Error generating MACD signals: {e}")
                signals['macd'] = pd.Series(0, index=data.index)
            
            # Generate Bollinger signals
            try:
                bollinger_signals = self.bollinger_strategy.generate_signals(data)
                signals['bollinger'] = bollinger_signals
                logger.debug("Bollinger signals generated successfully")
            except Exception as e:
                logger.error(f"Error generating Bollinger signals: {e}")
                signals['bollinger'] = pd.Series(0, index=data.index)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise

    def combine_signals(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Combine signals using the specified method.
        
        Args:
            signals: Dictionary of signals from individual strategies
            
        Returns:
            Combined signal series
        """
        try:
            if self.conflict_resolution == 'majority_vote':
                return self._majority_vote_combine(signals)
            elif self.conflict_resolution == 'weighted_average':
                return self._weighted_average_combine(signals)
            elif self.conflict_resolution == 'confidence_based':
                return self._confidence_based_combine(signals)
            else:
                logger.warning(f"Unknown conflict resolution method: {self.conflict_resolution}")
                return self._weighted_average_combine(signals)
                
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            raise

    def _majority_vote_combine(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Combine signals using majority vote with confidence-based tie-breaking.
        
        Args:
            signals: Dictionary of signals from individual strategies
            
        Returns:
            Combined signal series
        """
        try:
            # Convert signals to directional values (-1, 0, 1)
            directional_signals = {}
            for strategy, signal in signals.items():
                # Convert to directional signals
                directional = np.where(signal > 0, 1, np.where(signal < 0, -1, 0))
                directional_signals[strategy] = pd.Series(directional, index=signal.index)
            
            # Calculate weighted votes
            weighted_votes = pd.Series(0.0, index=signals[list(signals.keys())[0]].index)
            
            for strategy, signal in directional_signals.items():
                weight = self.weights.get(strategy, 1.0)
                weighted_votes += weight * signal
            
            # Determine final signal based on majority
            final_signal = np.where(weighted_votes > self.min_agreement, 1,
                                  np.where(weighted_votes < -self.min_agreement, -1, 0))
            
            # Convert back to original scale
            avg_magnitude = np.mean([np.abs(signal).mean() for signal in signals.values()])
            final_signal = final_signal * avg_magnitude
            
            logger.info(f"Majority vote combination completed. Agreement level: {np.abs(weighted_votes).mean():.2f}")
            return pd.Series(final_signal, index=weighted_votes.index)
            
        except Exception as e:
            logger.error(f"Error in majority vote combination: {e}")
            raise

    def _weighted_average_combine(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Combine signals using weighted average.
        
        Args:
            signals: Dictionary of signals from individual strategies
            
        Returns:
            Combined signal series
        """
        try:
            combined_signal = pd.Series(0.0, index=signals[list(signals.keys())[0]].index)
            total_weight = 0
            
            for strategy, signal in signals.items():
                weight = self.weights.get(strategy, 1.0)
                combined_signal += weight * signal
                total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                combined_signal = combined_signal / total_weight
            
            logger.info(f"Weighted average combination completed using {len(signals)} strategies")
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error in weighted average combination: {e}")
            raise

    def _confidence_based_combine(self, signals: Dict[str, pd.Series]) -> pd.Series:
        """Combine signals using confidence-based weighting.
        
        Args:
            signals: Dictionary of signals from individual strategies
            
        Returns:
            Combined signal series
        """
        try:
            # Calculate confidence for each strategy
            confidences = {}
            for strategy, signal in signals.items():
                # Simple confidence based on signal strength and consistency
                signal_strength = np.abs(signal).mean()
                signal_consistency = 1 - np.std(signal) / (np.abs(signal).mean() + 1e-8)
                confidence = signal_strength * signal_consistency
                confidences[strategy] = confidence
            
            # Normalize confidences
            total_confidence = sum(confidences.values())
            if total_confidence > 0:
                confidences = {k: v / total_confidence for k, v in confidences.items()}
            
            # Combine signals using confidence weights
            combined_signal = pd.Series(0.0, index=signals[list(signals.keys())[0]].index)
            
            for strategy, signal in signals.items():
                confidence = confidences.get(strategy, 0)
                combined_signal += confidence * signal
            
            logger.info(f"Confidence-based combination completed. Confidences: {confidences}")
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error in confidence-based combination: {e}")
            raise

    def resolve_conflicts(self, signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Resolve conflicts between different strategy signals.
        
        Args:
            signals: Dictionary of signals from individual strategies
            
        Returns:
            Dictionary with conflict resolution information
        """
        try:
            conflict_info = {}
            
            # Calculate agreement level
            directional_signals = {}
            for strategy, signal in signals.items():
                directional = np.where(signal > 0, 1, np.where(signal < 0, -1, 0))
                directional_signals[strategy] = pd.Series(directional, index=signal.index)
            
            # Calculate pairwise agreement
            strategies = list(signals.keys())
            agreement_matrix = {}
            
            for i, strategy1 in enumerate(strategies):
                for j, strategy2 in enumerate(strategies[i+1:], i+1):
                    agreement = (directional_signals[strategy1] == directional_signals[strategy2]).mean()
                    agreement_matrix[f"{strategy1}_{strategy2}"] = agreement
            
            # Identify conflicts
            conflicts = []
            for pair, agreement in agreement_matrix.items():
                if agreement < self.confidence_threshold:
                    conflicts.append(pair)
            
            conflict_info['agreement_matrix'] = agreement_matrix
            conflict_info['conflicts'] = conflicts
            conflict_info['overall_agreement'] = np.mean(list(agreement_matrix.values()))
            
            logger.info(f"Conflict resolution completed. Overall agreement: {conflict_info['overall_agreement']:.2f}")
            return conflict_info
            
        except Exception as e:
            logger.error(f"Error in conflict resolution: {e}")
            raise

    def get_synthesized_signal(self, data: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Get the final synthesized signal with metadata.
        
        Args:
            data: Price data with OHLC columns
            
        Returns:
            Tuple of (synthesized_signal, metadata)
        """
        try:
            # Generate individual signals
            individual_signals = self.generate_signals(data)
            
            # Resolve conflicts
            conflict_info = self.resolve_conflicts(individual_signals)
            
            # Combine signals
            synthesized_signal = self.combine_signals(individual_signals)
            
            # Add smoothing if enabled
            if self.config.get('smoothing', False):
                window = self.config.get('smoothing_window', 3)
                synthesized_signal = synthesized_signal.rolling(window=window, center=True).mean()
            
            # Prepare metadata
            metadata = {
                'individual_signals': individual_signals,
                'conflict_resolution': conflict_info,
                'weights_used': self.weights,
                'method_used': self.conflict_resolution,
                'signal_stats': {
                    'mean': synthesized_signal.mean(),
                    'std': synthesized_signal.std(),
                    'min': synthesized_signal.min(),
                    'max': synthesized_signal.max()
                }
            }
            
            logger.info(f"Synthesized signal generated successfully. "
                       f"Mean signal strength: {metadata['signal_stats']['mean']:.4f}")
            
            return synthesized_signal, metadata
            
        except Exception as e:
            logger.error(f"Error generating synthesized signal: {e}")
            raise

    def update_weights(self, performance_metrics: Dict[str, float]):
        """Update strategy weights based on recent performance.
        
        Args:
            performance_metrics: Dictionary of performance metrics for each strategy
        """
        try:
            # Normalize performance metrics
            total_performance = sum(performance_metrics.values())
            if total_performance > 0:
                new_weights = {k: v / total_performance for k, v in performance_metrics.items()}
                
                # Apply learning rate
                learning_rate = self.config.get('weight_learning_rate', 0.1)
                for strategy in self.weights:
                    if strategy in new_weights:
                        self.weights[strategy] = (1 - learning_rate) * self.weights[strategy] + \
                                                learning_rate * new_weights[strategy]
                
                # Renormalize
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    self.weights = {k: v / total_weight for k, v in self.weights.items()}
                
                logger.info(f"Weights updated: {self.weights}")
            
        except Exception as e:
            logger.error(f"Error updating weights: {e}")

    def get_strategy_summary(self) -> str:
        """Get a summary of the strategy synthesizer configuration.
        
        Returns:
            Summary string
        """
        summary = [
            "Strategy Synthesizer Summary",
            "=" * 30,
            f"Conflict Resolution: {self.conflict_resolution}",
            f"Confidence Threshold: {self.confidence_threshold}",
            f"Minimum Agreement: {self.min_agreement}",
            "",
            "Strategy Weights:"
        ]
        
        for strategy, weight in self.weights.items():
            summary.append(f"  {strategy}: {weight:.3f}")
        
        return "\n".join(summary) 