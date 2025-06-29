# -*- coding: utf-8 -*-
"""
Breakout Strategy Engine for detecting consolidation ranges and breakout signals.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import talib

from trading.strategies.rsi_signals import calculate_rsi
from trading.utils.performance_metrics import calculate_volatility


class BreakoutType(str, Enum):
    """Types of breakouts."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    FALSE = "false"


@dataclass
class ConsolidationRange:
    """Consolidation range information."""
    start_date: datetime
    end_date: datetime
    upper_bound: float
    lower_bound: float
    range_width: float
    range_percentage: float
    volume_profile: Dict[str, float]
    duration_days: int
    confidence: float


@dataclass
class BreakoutSignal:
    """Breakout trading signal."""
    timestamp: datetime
    symbol: str
    breakout_type: BreakoutType
    price_level: float
    volume_spike: float
    rsi_divergence: bool
    consolidation_range: ConsolidationRange
    confidence: float
    stop_loss: float
    take_profit: float
    position_size: float


class BreakoutStrategyEngine:
    """
    Breakout Strategy Engine with:
    - Consolidation range detection
    - Volume spike confirmation
    - RSI divergence analysis
    - False breakout filtering
    - Risk management with stop-loss and take-profit
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Breakout Strategy Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.min_consolidation_days = self.config.get('min_consolidation_days', 10)
        self.max_consolidation_days = self.config.get('max_consolidation_days', 60)
        self.range_threshold = self.config.get('range_threshold', 0.05)  # 5% range
        self.volume_spike_threshold = self.config.get('volume_spike_threshold', 2.0)
        self.breakout_confirmation_days = self.config.get('breakout_confirmation_days', 3)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.stop_loss_multiplier = self.config.get('stop_loss_multiplier', 1.5)
        self.take_profit_multiplier = self.config.get('take_profit_multiplier', 2.0)
        
        # Storage
        self.consolidation_ranges: Dict[str, List[ConsolidationRange]] = {}
        self.breakout_signals: Dict[str, List[BreakoutSignal]] = {}
        self.active_breakouts: Dict[str, Dict[str, Any]] = {}
        
    def detect_consolidation_ranges(self, 
                                  data: pd.DataFrame,
                                  symbol: str) -> List[ConsolidationRange]:
        """
        Detect consolidation ranges in price data.
        
        Args:
            data: Market data DataFrame
            symbol: Asset symbol
            
        Returns:
            List of detected consolidation ranges
        """
        try:
            self.logger.info(f"Detecting consolidation ranges for {symbol}")
            
            if data.empty or 'close' not in data.columns:
                return []
            
            consolidation_ranges = []
            close_prices = data['close']
            
            # Calculate rolling volatility
            volatility = calculate_volatility(close_prices, window=20)
            
            # Find periods of low volatility (consolidation)
            low_volatility_periods = volatility < volatility.quantile(0.3)
            
            # Group consecutive low volatility periods
            consolidation_groups = self._group_consecutive_periods(low_volatility_periods)
            
            for start_idx, end_idx in consolidation_groups:
                if end_idx - start_idx < self.min_consolidation_days:
                    continue
                
                if end_idx - start_idx > self.max_consolidation_days:
                    continue
                
                # Extract consolidation period
                consolidation_data = data.iloc[start_idx:end_idx+1]
                consolidation_prices = close_prices.iloc[start_idx:end_idx+1]
                
                # Calculate range bounds
                upper_bound = consolidation_prices.max()
                lower_bound = consolidation_prices.min()
                range_width = upper_bound - lower_bound
                range_percentage = range_width / lower_bound
                
                # Check if range is within threshold
                if range_percentage > self.range_threshold:
                    continue
                
                # Calculate volume profile
                volume_profile = self._calculate_volume_profile(consolidation_data)
                
                # Calculate confidence
                confidence = self._calculate_consolidation_confidence(
                    consolidation_data, range_percentage, volatility.iloc[start_idx:end_idx+1]
                )
                
                # Create consolidation range
                consolidation_range = ConsolidationRange(
                    start_date=consolidation_data.index[0],
                    end_date=consolidation_data.index[-1],
                    upper_bound=upper_bound,
                    lower_bound=lower_bound,
                    range_width=range_width,
                    range_percentage=range_percentage,
                    volume_profile=volume_profile,
                    duration_days=end_idx - start_idx + 1,
                    confidence=confidence
                )
                
                consolidation_ranges.append(consolidation_range)
                
                self.logger.info(f"Detected consolidation range: {consolidation_range.start_date} to "
                               f"{consolidation_range.end_date}, range: {range_percentage:.2%}")
            
            # Store consolidation ranges
            if symbol not in self.consolidation_ranges:
                self.consolidation_ranges[symbol] = []
            
            self.consolidation_ranges[symbol].extend(consolidation_ranges)
            
            return consolidation_ranges
            
        except Exception as e:
            self.logger.error(f"Error detecting consolidation ranges: {str(e)}")
            return []
    
    def _group_consecutive_periods(self, boolean_series: pd.Series) -> List[Tuple[int, int]]:
        """Group consecutive True periods in a boolean series."""
        try:
            groups = []
            start_idx = None
            
            for i, value in enumerate(boolean_series):
                if value and start_idx is None:
                    start_idx = i
                elif not value and start_idx is not None:
                    groups.append((start_idx, i - 1))
                    start_idx = None
            
            # Handle case where series ends with True
            if start_idx is not None:
                groups.append((start_idx, len(boolean_series) - 1))
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error grouping consecutive periods: {str(e)}")
            return []
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume profile for consolidation period."""
        try:
            if 'volume' not in data.columns:
                return {}
            
            volume = data['volume']
            
            return {
                'mean_volume': volume.mean(),
                'std_volume': volume.std(),
                'min_volume': volume.min(),
                'max_volume': volume.max(),
                'volume_trend': self._calculate_volume_trend(volume)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {str(e)}")
            return {}
    
    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend (positive = increasing, negative = decreasing)."""
        try:
            if len(volume) < 2:
                return 0.0
            
            # Linear regression slope
            x = np.arange(len(volume))
            slope = np.polyfit(x, volume, 1)[0]
            
            # Normalize by mean volume
            mean_volume = volume.mean()
            if mean_volume > 0:
                return slope / mean_volume
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating volume trend: {str(e)}")
            return 0.0
    
    def _calculate_consolidation_confidence(self, 
                                          data: pd.DataFrame,
                                          range_percentage: float,
                                          volatility: pd.Series) -> float:
        """Calculate confidence score for consolidation range."""
        try:
            # Base confidence from range percentage (smaller range = higher confidence)
            range_confidence = max(0.0, 1.0 - (range_percentage / self.range_threshold))
            
            # Volatility confidence (lower volatility = higher confidence)
            vol_confidence = max(0.0, 1.0 - (volatility.mean() / volatility.quantile(0.5)))
            
            # Duration confidence (optimal duration = higher confidence)
            duration = len(data)
            duration_confidence = 1.0
            if duration < self.min_consolidation_days:
                duration_confidence = duration / self.min_consolidation_days
            elif duration > self.max_consolidation_days:
                duration_confidence = self.max_consolidation_days / duration
            
            # Volume consistency confidence
            volume_confidence = 1.0
            if 'volume' in data.columns:
                volume_cv = data['volume'].std() / data['volume'].mean()
                volume_confidence = max(0.0, 1.0 - volume_cv)
            
            # Weighted average
            confidence = (
                0.3 * range_confidence +
                0.3 * vol_confidence +
                0.2 * duration_confidence +
                0.2 * volume_confidence
            )
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating consolidation confidence: {str(e)}")
            return 0.5
    
    def detect_breakouts(self, 
                        data: pd.DataFrame,
                        symbol: str) -> List[BreakoutSignal]:
        """
        Detect breakout signals from consolidation ranges.
        
        Args:
            data: Market data DataFrame
            symbol: Asset symbol
            
        Returns:
            List of breakout signals
        """
        try:
            self.logger.info(f"Detecting breakouts for {symbol}")
            
            if data.empty or 'close' not in data.columns:
                return []
            
            breakout_signals = []
            current_price = data['close'].iloc[-1]
            
            # Check each consolidation range for breakouts
            if symbol in self.consolidation_ranges:
                for consolidation_range in self.consolidation_ranges[symbol]:
                    # Check if price has broken out of the range
                    breakout_type = self._check_breakout_type(
                        current_price, consolidation_range
                    )
                    
                    if breakout_type != BreakoutType.FALSE:
                        # Confirm breakout with volume and RSI
                        if self._confirm_breakout(data, consolidation_range, breakout_type):
                            signal = self._create_breakout_signal(
                                data, symbol, consolidation_range, breakout_type
                            )
                            
                            if signal:
                                breakout_signals.append(signal)
                                self.logger.info(f"Detected {breakout_type.value} breakout for {symbol}")
            
            # Store breakout signals
            if symbol not in self.breakout_signals:
                self.breakout_signals[symbol] = []
            
            self.breakout_signals[symbol].extend(breakout_signals)
            
            return breakout_signals
            
        except Exception as e:
            self.logger.error(f"Error detecting breakouts: {str(e)}")
            return []
    
    def _check_breakout_type(self, 
                           current_price: float,
                           consolidation_range: ConsolidationRange) -> BreakoutType:
        """Check if price has broken out of consolidation range."""
        try:
            # Check for bullish breakout
            if current_price > consolidation_range.upper_bound:
                return BreakoutType.BULLISH
            
            # Check for bearish breakout
            elif current_price < consolidation_range.lower_bound:
                return BreakoutType.BEARISH
            
            else:
                return BreakoutType.FALSE
                
        except Exception as e:
            self.logger.error(f"Error checking breakout type: {str(e)}")
            return BreakoutType.FALSE
    
    def _confirm_breakout(self, 
                         data: pd.DataFrame,
                         consolidation_range: ConsolidationRange,
                         breakout_type: BreakoutType) -> bool:
        """Confirm breakout with volume spike and RSI divergence."""
        try:
            # Check volume spike
            volume_spike = self._check_volume_spike(data, consolidation_range)
            
            # Check RSI divergence
            rsi_divergence = self._check_rsi_divergence(data, consolidation_range, breakout_type)
            
            # Check price confirmation (multiple closes beyond breakout level)
            price_confirmation = self._check_price_confirmation(
                data, consolidation_range, breakout_type
            )
            
            # All conditions must be met for confirmation
            return volume_spike and rsi_divergence and price_confirmation
            
        except Exception as e:
            self.logger.error(f"Error confirming breakout: {str(e)}")
            return False
    
    def _check_volume_spike(self, 
                           data: pd.DataFrame,
                           consolidation_range: ConsolidationRange) -> bool:
        """Check for volume spike during breakout."""
        try:
            if 'volume' not in data.columns:
                return False
            
            # Get recent volume data
            recent_volume = data['volume'].tail(5)
            consolidation_volume = consolidation_range.volume_profile.get('mean_volume', 0)
            
            if consolidation_volume == 0:
                return False
            
            # Check if recent volume is significantly higher than consolidation average
            volume_ratio = recent_volume.mean() / consolidation_volume
            
            return volume_ratio > self.volume_spike_threshold
            
        except Exception as e:
            self.logger.error(f"Error checking volume spike: {str(e)}")
            return False
    
    def _check_rsi_divergence(self, 
                             data: pd.DataFrame,
                             consolidation_range: ConsolidationRange,
                             breakout_type: BreakoutType) -> bool:
        """Check for RSI divergence during breakout."""
        try:
            if 'close' not in data.columns:
                return False
            
            # Calculate RSI
            rsi = calculate_rsi(data['close'], period=self.rsi_period)
            
            if rsi is None or len(rsi) < 10:
                return False
            
            # Get recent RSI values
            recent_rsi = rsi.tail(5)
            
            if breakout_type == BreakoutType.BULLISH:
                # For bullish breakout, RSI should be strong (not overbought)
                return recent_rsi.mean() > 50 and recent_rsi.iloc[-1] < self.rsi_overbought
            
            elif breakout_type == BreakoutType.BEARISH:
                # For bearish breakout, RSI should be weak (not oversold)
                return recent_rsi.mean() < 50 and recent_rsi.iloc[-1] > self.rsi_oversold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking RSI divergence: {str(e)}")
            return False
    
    def _check_price_confirmation(self, 
                                 data: pd.DataFrame,
                                 consolidation_range: ConsolidationRange,
                                 breakout_type: BreakoutType) -> bool:
        """Check for price confirmation over multiple periods."""
        try:
            if 'close' not in data.columns:
                return False
            
            recent_prices = data['close'].tail(self.breakout_confirmation_days)
            
            if breakout_type == BreakoutType.BULLISH:
                # Check if prices stay above upper bound
                return all(price > consolidation_range.upper_bound for price in recent_prices)
            
            elif breakout_type == BreakoutType.BEARISH:
                # Check if prices stay below lower bound
                return all(price < consolidation_range.lower_bound for price in recent_prices)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking price confirmation: {str(e)}")
            return False
    
    def _create_breakout_signal(self, 
                               data: pd.DataFrame,
                               symbol: str,
                               consolidation_range: ConsolidationRange,
                               breakout_type: BreakoutType) -> Optional[BreakoutSignal]:
        """Create breakout signal with risk management levels."""
        try:
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 0
            
            # Calculate volume spike ratio
            volume_spike = current_volume / consolidation_range.volume_profile.get('mean_volume', 1)
            
            # Check RSI divergence
            rsi = calculate_rsi(data['close'], period=self.rsi_period)
            rsi_divergence = False
            if rsi is not None and len(rsi) > 0:
                if breakout_type == BreakoutType.BULLISH:
                    rsi_divergence = rsi.iloc[-1] > 50 and rsi.iloc[-1] < self.rsi_overbought
                elif breakout_type == BreakoutType.BEARISH:
                    rsi_divergence = rsi.iloc[-1] < 50 and rsi.iloc[-1] > self.rsi_oversold
            
            # Calculate confidence
            confidence = self._calculate_breakout_confidence(
                consolidation_range, volume_spike, rsi_divergence
            )
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_levels(
                current_price, consolidation_range, breakout_type
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(confidence, volume_spike)
            
            return BreakoutSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                breakout_type=breakout_type,
                price_level=current_price,
                volume_spike=volume_spike,
                rsi_divergence=rsi_divergence,
                consolidation_range=consolidation_range,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size
            )
            
        except Exception as e:
            self.logger.error(f"Error creating breakout signal: {str(e)}")
            return None
    
    def _calculate_breakout_confidence(self, 
                                     consolidation_range: ConsolidationRange,
                                     volume_spike: float,
                                     rsi_divergence: bool) -> float:
        """Calculate confidence score for breakout signal."""
        try:
            # Base confidence from consolidation range
            base_confidence = consolidation_range.confidence
            
            # Volume spike confidence
            volume_confidence = min(1.0, volume_spike / self.volume_spike_threshold)
            
            # RSI divergence confidence
            rsi_confidence = 1.0 if rsi_divergence else 0.5
            
            # Duration confidence (longer consolidation = higher confidence)
            duration_confidence = min(1.0, consolidation_range.duration_days / 30)
            
            # Weighted average
            confidence = (
                0.3 * base_confidence +
                0.3 * volume_confidence +
                0.2 * rsi_confidence +
                0.2 * duration_confidence
            )
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout confidence: {str(e)}")
            return 0.5
    
    def _calculate_risk_levels(self, 
                             current_price: float,
                             consolidation_range: ConsolidationRange,
                             breakout_type: BreakoutType) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        try:
            range_width = consolidation_range.range_width
            
            if breakout_type == BreakoutType.BULLISH:
                # Stop loss below the consolidation range
                stop_loss = consolidation_range.lower_bound - (range_width * self.stop_loss_multiplier)
                # Take profit above the breakout level
                take_profit = current_price + (range_width * self.take_profit_multiplier)
            
            elif breakout_type == BreakoutType.BEARISH:
                # Stop loss above the consolidation range
                stop_loss = consolidation_range.upper_bound + (range_width * self.stop_loss_multiplier)
                # Take profit below the breakout level
                take_profit = current_price - (range_width * self.take_profit_multiplier)
            
            else:
                stop_loss = current_price
                take_profit = current_price
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating risk levels: {str(e)}")
            return current_price, current_price
    
    def _calculate_position_size(self, confidence: float, volume_spike: float) -> float:
        """Calculate position size based on confidence and volume spike."""
        try:
            # Base position size from confidence
            base_size = confidence
            
            # Adjust for volume spike
            volume_multiplier = min(1.5, volume_spike / self.volume_spike_threshold)
            
            # Risk adjustment
            risk_multiplier = 0.7  # Conservative position sizing
            
            position_size = base_size * volume_multiplier * risk_multiplier
            
            return max(0.1, min(1.0, position_size))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.5
    
    def filter_false_breakouts(self, 
                              data: pd.DataFrame,
                              symbol: str,
                              lookback_days: int = 30) -> List[BreakoutSignal]:
        """Filter out false breakouts by checking historical performance."""
        try:
            if symbol not in self.breakout_signals:
                return []
            
            # Get recent breakout signals
            recent_signals = [
                signal for signal in self.breakout_signals[symbol]
                if (datetime.now() - signal.timestamp).days <= lookback_days
            ]
            
            filtered_signals = []
            
            for signal in recent_signals:
                # Check if breakout was successful
                success = self._check_breakout_success(data, signal)
                
                if success:
                    filtered_signals.append(signal)
                else:
                    self.logger.info(f"Filtered out false breakout for {symbol}")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering false breakouts: {str(e)}")
            return []
    
    def _check_breakout_success(self, 
                               data: pd.DataFrame,
                               signal: BreakoutSignal) -> bool:
        """Check if a breakout signal was successful."""
        try:
            if signal.breakout_type == BreakoutType.BULLISH:
                # Check if price reached take profit level
                return data['close'].max() >= signal.take_profit
            
            elif signal.breakout_type == BreakoutType.BEARISH:
                # Check if price reached take profit level
                return data['close'].min() <= signal.take_profit
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking breakout success: {str(e)}")
            return False
    
    def get_consolidation_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get summary of consolidation ranges for a symbol."""
        try:
            if symbol not in self.consolidation_ranges:
                return None
            
            ranges = self.consolidation_ranges[symbol]
            if not ranges:
                return None
            
            # Calculate statistics
            durations = [r.duration_days for r in ranges]
            confidences = [r.confidence for r in ranges]
            range_percentages = [r.range_percentage for r in ranges]
            
            return {
                'symbol': symbol,
                'total_ranges': len(ranges),
                'avg_duration': np.mean(durations),
                'avg_confidence': np.mean(confidences),
                'avg_range_percentage': np.mean(range_percentages),
                'latest_range': {
                    'start_date': ranges[-1].start_date.isoformat(),
                    'end_date': ranges[-1].end_date.isoformat(),
                    'upper_bound': ranges[-1].upper_bound,
                    'lower_bound': ranges[-1].lower_bound,
                    'confidence': ranges[-1].confidence
                } if ranges else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting consolidation summary: {str(e)}")
            return None
    
    def get_breakout_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get summary of breakout signals for a symbol."""
        try:
            if symbol not in self.breakout_signals:
                return None
            
            signals = self.breakout_signals[symbol]
            if not signals:
                return None
            
            # Calculate statistics
            bullish_signals = [s for s in signals if s.breakout_type == BreakoutType.BULLISH]
            bearish_signals = [s for s in signals if s.breakout_type == BreakoutType.BEARISH]
            
            confidences = [s.confidence for s in signals]
            volume_spikes = [s.volume_spike for s in signals]
            
            return {
                'symbol': symbol,
                'total_signals': len(signals),
                'bullish_signals': len(bullish_signals),
                'bearish_signals': len(bearish_signals),
                'avg_confidence': np.mean(confidences),
                'avg_volume_spike': np.mean(volume_spikes),
                'latest_signal': {
                    'timestamp': signals[-1].timestamp.isoformat(),
                    'breakout_type': signals[-1].breakout_type.value,
                    'confidence': signals[-1].confidence,
                    'volume_spike': signals[-1].volume_spike
                } if signals else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting breakout summary: {str(e)}")
            return None 