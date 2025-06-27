"""
Position Sizer

This module provides dynamic position sizing based on risk tolerance,
confidence scores, and forecast certainty. Supports multiple sizing strategies
including fixed percentage, Kelly Criterion, and volatility-based sizing.
"""

import math
import logging
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from trading.portfolio.portfolio_manager import Position, TradeDirection


class SizingStrategy(Enum):
    """Position sizing strategy enum."""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"
    CONFIDENCE_BASED = "confidence_based"
    FORECAST_CERTAINTY = "forecast_certainty"
    OPTIMAL_F = "optimal_f"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"


@dataclass
class SizingParameters:
    """Position sizing parameters."""
    strategy: SizingStrategy
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.2  # 20% max position size
    confidence_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    kelly_fraction: float = 0.25  # Conservative Kelly fraction
    optimal_f_risk: float = 0.02  # 2% risk for Optimal F
    base_position_size: float = 0.1  # 10% base position size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy': self.strategy.value,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'confidence_multiplier': self.confidence_multiplier,
            'volatility_multiplier': self.volatility_multiplier,
            'kelly_fraction': self.kelly_fraction,
            'optimal_f_risk': self.optimal_f_risk,
            'base_position_size': self.base_position_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SizingParameters':
        """Create from dictionary."""
        data['strategy'] = SizingStrategy(data['strategy'])
        return cls(**data)


@dataclass
class MarketContext:
    """Market context for position sizing."""
    symbol: str
    current_price: float
    volatility: float
    volume: float
    market_regime: str = "normal"
    correlation: float = 0.0
    liquidity_score: float = 1.0
    bid_ask_spread: float = 0.001
    price_history: Optional[pd.Series] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'volatility': self.volatility,
            'volume': self.volume,
            'market_regime': self.market_regime,
            'correlation': self.correlation,
            'liquidity_score': self.liquidity_score,
            'bid_ask_spread': self.bid_ask_spread
        }


@dataclass
class SignalContext:
    """Signal context for position sizing."""
    confidence: float
    forecast_certainty: float
    strategy_performance: float
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: float
    max_drawdown: float
    signal_strength: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'confidence': self.confidence,
            'forecast_certainty': self.forecast_certainty,
            'strategy_performance': self.strategy_performance,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'signal_strength': self.signal_strength
        }


@dataclass
class PortfolioContext:
    """Portfolio context for position sizing."""
    total_capital: float
    available_capital: float
    current_exposure: float
    open_positions: int
    daily_pnl: float
    portfolio_volatility: float
    correlation_matrix: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'current_exposure': self.current_exposure,
            'open_positions': self.open_positions,
            'daily_pnl': self.daily_pnl,
            'portfolio_volatility': self.portfolio_volatility
        }


class PositionSizer:
    """Dynamic position sizer with multiple strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the position sizer.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        default_config = {
            'default_strategy': SizingStrategy.FIXED_PERCENTAGE,
            'risk_per_trade': 0.02,
            'max_position_size': 0.2,
            'confidence_multiplier': 1.0,
            'volatility_multiplier': 1.0,
            'kelly_fraction': 0.25,
            'optimal_f_risk': 0.02,
            'base_position_size': 0.1,
            'enable_risk_adjustment': True,
            'enable_correlation_adjustment': True,
            'enable_volatility_adjustment': True
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.sizing_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"PositionSizer initialized with strategy: {self.config['default_strategy'].value}")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        market_context: MarketContext,
        signal_context: SignalContext,
        portfolio_context: PortfolioContext,
        sizing_params: Optional[SizingParameters] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate optimal position size.
        
        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            market_context: Market context
            signal_context: Signal context
            portfolio_context: Portfolio context
            sizing_params: Optional sizing parameters override
            
        Returns:
            Tuple of (position_size, sizing_details)
        """
        try:
            # Use provided parameters or defaults
            params = sizing_params or SizingParameters(
                strategy=self.config['default_strategy'],
                risk_per_trade=self.config['risk_per_trade'],
                max_position_size=self.config['max_position_size']
            )
            
            # Calculate base position size based on strategy
            base_size = self._calculate_base_size(
                entry_price, stop_loss_price, market_context,
                signal_context, portfolio_context, params
            )
            
            # Apply adjustments
            adjusted_size = self._apply_adjustments(
                base_size, market_context, signal_context,
                portfolio_context, params
            )
            
            # Apply final constraints
            final_size = self._apply_constraints(
                adjusted_size, portfolio_context, params
            )
            
            # Create sizing details
            sizing_details = {
                'strategy': params.strategy.value,
                'base_size': base_size,
                'adjusted_size': adjusted_size,
                'final_size': final_size,
                'risk_amount': abs(entry_price - stop_loss_price) * final_size,
                'risk_percentage': (abs(entry_price - stop_loss_price) * final_size) / portfolio_context.total_capital,
                'position_value': entry_price * final_size,
                'position_percentage': (entry_price * final_size) / portfolio_context.total_capital,
                'market_context': market_context.to_dict(),
                'signal_context': signal_context.to_dict(),
                'portfolio_context': portfolio_context.to_dict(),
                'sizing_params': params.to_dict()
            }
            
            # Log sizing decision
            self._log_sizing_decision(sizing_details)
            
            return final_size, sizing_details
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            # Return conservative default
            return self._calculate_conservative_size(portfolio_context), {
                'error': str(e),
                'strategy': 'conservative_fallback'
            }
    
    def _calculate_base_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        market_context: MarketContext,
        signal_context: SignalContext,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Calculate base position size using the specified strategy.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            market_context: Market context
            signal_context: Signal context
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Base position size
        """
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if params.strategy == SizingStrategy.FIXED_PERCENTAGE:
            return self._fixed_percentage_size(portfolio_context, params)
        
        elif params.strategy == SizingStrategy.KELLY_CRITERION:
            return self._kelly_criterion_size(
                signal_context, risk_per_share, portfolio_context, params
            )
        
        elif params.strategy == SizingStrategy.VOLATILITY_BASED:
            return self._volatility_based_size(
                market_context, risk_per_share, portfolio_context, params
            )
        
        elif params.strategy == SizingStrategy.RISK_PARITY:
            return self._risk_parity_size(
                market_context, portfolio_context, params
            )
        
        elif params.strategy == SizingStrategy.CONFIDENCE_BASED:
            return self._confidence_based_size(
                signal_context, risk_per_share, portfolio_context, params
            )
        
        elif params.strategy == SizingStrategy.FORECAST_CERTAINTY:
            return self._forecast_certainty_size(
                signal_context, risk_per_share, portfolio_context, params
            )
        
        elif params.strategy == SizingStrategy.OPTIMAL_F:
            return self._optimal_f_size(
                signal_context, risk_per_share, portfolio_context, params
            )
        
        elif params.strategy == SizingStrategy.MARTINGALE:
            return self._martingale_size(portfolio_context, params)
        
        elif params.strategy == SizingStrategy.ANTI_MARTINGALE:
            return self._anti_martingale_size(portfolio_context, params)
        
        else:
            # Default to fixed percentage
            return self._fixed_percentage_size(portfolio_context, params)
    
    def _fixed_percentage_size(
        self, portfolio_context: PortfolioContext, params: SizingParameters
    ) -> float:
        """Calculate fixed percentage position size.
        
        Args:
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Position size as percentage of capital
        """
        return params.risk_per_trade
    
    def _kelly_criterion_size(
        self,
        signal_context: SignalContext,
        risk_per_share: float,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Calculate Kelly Criterion position size.
        
        Args:
            signal_context: Signal context
            risk_per_share: Risk per share
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Kelly Criterion position size
        """
        # Kelly formula: f = (bp - q) / b
        # where b = odds received, p = probability of win, q = probability of loss
        
        win_rate = signal_context.win_rate
        avg_win = signal_context.avg_win
        avg_loss = abs(signal_context.avg_loss)
        
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return params.base_position_size
        
        # Calculate Kelly fraction
        b = avg_win / avg_loss  # odds received
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative Kelly fraction
        kelly_fraction *= params.kelly_fraction
        
        # Convert to position size
        position_size = kelly_fraction * portfolio_context.total_capital / risk_per_share
        
        return max(0, min(position_size, params.max_position_size))
    
    def _volatility_based_size(
        self,
        market_context: MarketContext,
        risk_per_share: float,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Calculate volatility-based position size.
        
        Args:
            market_context: Market context
            risk_per_share: Risk per share
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Volatility-based position size
        """
        # Inverse relationship with volatility
        volatility_factor = 1.0 / (1.0 + market_context.volatility * params.volatility_multiplier)
        
        # Base size adjusted by volatility
        base_size = params.base_position_size * volatility_factor
        
        # Convert to actual position size
        position_size = base_size * portfolio_context.total_capital / risk_per_share
        
        return max(0, min(position_size, params.max_position_size))
    
    def _risk_parity_size(
        self,
        market_context: MarketContext,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Calculate risk parity position size.
        
        Args:
            market_context: Market context
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Risk parity position size
        """
        # Equal risk contribution across positions
        target_risk = params.risk_per_trade
        
        # Adjust for volatility
        volatility_adjustment = 1.0 / (1.0 + market_context.volatility)
        
        # Calculate position size for equal risk contribution
        position_size = target_risk * volatility_adjustment
        
        return max(0, min(position_size, params.max_position_size))
    
    def _confidence_based_size(
        self,
        signal_context: SignalContext,
        risk_per_share: float,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Calculate confidence-based position size.
        
        Args:
            signal_context: Signal context
            risk_per_share: Risk per share
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Confidence-based position size
        """
        # Scale position size by confidence
        confidence_factor = signal_context.confidence * params.confidence_multiplier
        
        # Base size adjusted by confidence
        base_size = params.base_position_size * confidence_factor
        
        # Convert to actual position size
        position_size = base_size * portfolio_context.total_capital / risk_per_share
        
        return max(0, min(position_size, params.max_position_size))
    
    def _forecast_certainty_size(
        self,
        signal_context: SignalContext,
        risk_per_share: float,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Calculate forecast certainty-based position size.
        
        Args:
            signal_context: Signal context
            risk_per_share: Risk per share
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Forecast certainty-based position size
        """
        # Scale position size by forecast certainty
        certainty_factor = signal_context.forecast_certainty * params.confidence_multiplier
        
        # Base size adjusted by certainty
        base_size = params.base_position_size * certainty_factor
        
        # Convert to actual position size
        position_size = base_size * portfolio_context.total_capital / risk_per_share
        
        return max(0, min(position_size, params.max_position_size))
    
    def _optimal_f_size(
        self,
        signal_context: SignalContext,
        risk_per_share: float,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Calculate Optimal F position size.
        
        Args:
            signal_context: Signal context
            risk_per_share: Risk per share
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Optimal F position size
        """
        # Optimal F formula: f = (W * (1 + W)) / (L * (1 + L))
        # where W = winning trades, L = losing trades
        
        if signal_context.avg_loss == 0:
            return params.base_position_size
        
        # Calculate Optimal F
        win_ratio = signal_context.avg_win / abs(signal_context.avg_loss)
        optimal_f = (win_ratio * (1 + win_ratio)) / (1 + win_ratio)
        
        # Apply risk adjustment
        optimal_f *= params.optimal_f_risk
        
        # Convert to position size
        position_size = optimal_f * portfolio_context.total_capital / risk_per_share
        
        return max(0, min(position_size, params.max_position_size))
    
    def _martingale_size(
        self, portfolio_context: PortfolioContext, params: SizingParameters
    ) -> float:
        """Calculate Martingale position size (increasing after losses).
        
        Args:
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Martingale position size
        """
        # Increase position size after losses
        loss_multiplier = 1.0 + abs(portfolio_context.daily_pnl) * 2.0
        
        base_size = params.base_position_size * loss_multiplier
        
        return max(0, min(base_size, params.max_position_size))
    
    def _anti_martingale_size(
        self, portfolio_context: PortfolioContext, params: SizingParameters
    ) -> float:
        """Calculate Anti-Martingale position size (increasing after wins).
        
        Args:
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Anti-Martingale position size
        """
        # Increase position size after wins
        win_multiplier = 1.0 + max(0, portfolio_context.daily_pnl) * 2.0
        
        base_size = params.base_position_size * win_multiplier
        
        return max(0, min(base_size, params.max_position_size))
    
    def _apply_adjustments(
        self,
        base_size: float,
        market_context: MarketContext,
        signal_context: SignalContext,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Apply various adjustments to base position size.
        
        Args:
            base_size: Base position size
            market_context: Market context
            signal_context: Signal context
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Adjusted position size
        """
        adjusted_size = base_size
        
        # Risk adjustment
        if self.config['enable_risk_adjustment']:
            adjusted_size = self._apply_risk_adjustment(
                adjusted_size, signal_context, portfolio_context
            )
        
        # Correlation adjustment
        if self.config['enable_correlation_adjustment']:
            adjusted_size = self._apply_correlation_adjustment(
                adjusted_size, market_context, portfolio_context
            )
        
        # Volatility adjustment
        if self.config['enable_volatility_adjustment']:
            adjusted_size = self._apply_volatility_adjustment(
                adjusted_size, market_context
            )
        
        # Liquidity adjustment
        adjusted_size = self._apply_liquidity_adjustment(
            adjusted_size, market_context
        )
        
        return adjusted_size
    
    def _apply_risk_adjustment(
        self,
        size: float,
        signal_context: SignalContext,
        portfolio_context: PortfolioContext
    ) -> float:
        """Apply risk-based adjustments.
        
        Args:
            size: Current position size
            signal_context: Signal context
            portfolio_context: Portfolio context
            
        Returns:
            Risk-adjusted position size
        """
        # Reduce size if portfolio is already losing
        if portfolio_context.daily_pnl < 0:
            loss_factor = 1.0 + portfolio_context.daily_pnl  # Reduces size
            size *= max(0.5, loss_factor)  # Minimum 50% of original size
        
        # Reduce size if signal quality is poor
        if signal_context.sharpe_ratio < 0:
            size *= 0.5  # Reduce by 50% for negative Sharpe ratio
        
        # Reduce size if max drawdown is high
        if signal_context.max_drawdown > 0.2:  # 20% drawdown
            size *= 0.7  # Reduce by 30%
        
        return size
    
    def _apply_correlation_adjustment(
        self,
        size: float,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> float:
        """Apply correlation-based adjustments.
        
        Args:
            size: Current position size
            market_context: Market context
            portfolio_context: Portfolio context
            
        Returns:
            Correlation-adjusted position size
        """
        # Reduce size if correlation is high
        if market_context.correlation > 0.7:
            correlation_factor = 1.0 - (market_context.correlation - 0.7) * 2.0
            size *= max(0.3, correlation_factor)  # Minimum 30% of original size
        
        return size
    
    def _apply_volatility_adjustment(
        self, size: float, market_context: MarketContext
    ) -> float:
        """Apply volatility-based adjustments.
        
        Args:
            size: Current position size
            market_context: Market context
            
        Returns:
            Volatility-adjusted position size
        """
        # Reduce size in high volatility
        if market_context.volatility > 0.3:  # 30% volatility
            volatility_factor = 1.0 - (market_context.volatility - 0.3) * 2.0
            size *= max(0.5, volatility_factor)  # Minimum 50% of original size
        
        return size
    
    def _apply_liquidity_adjustment(
        self, size: float, market_context: MarketContext
    ) -> float:
        """Apply liquidity-based adjustments.
        
        Args:
            size: Current position size
            market_context: Market context
            
        Returns:
            Liquidity-adjusted position size
        """
        # Reduce size for low liquidity
        if market_context.liquidity_score < 0.5:
            size *= market_context.liquidity_score
        
        # Reduce size for wide bid-ask spreads
        if market_context.bid_ask_spread > 0.005:  # 0.5% spread
            spread_factor = 1.0 - (market_context.bid_ask_spread - 0.005) * 100
            size *= max(0.3, spread_factor)  # Minimum 30% of original size
        
        return size
    
    def _apply_constraints(
        self,
        size: float,
        portfolio_context: PortfolioContext,
        params: SizingParameters
    ) -> float:
        """Apply final constraints to position size.
        
        Args:
            size: Current position size
            portfolio_context: Portfolio context
            params: Sizing parameters
            
        Returns:
            Constrained position size
        """
        # Maximum position size constraint
        size = min(size, params.max_position_size)
        
        # Available capital constraint
        max_size_by_capital = portfolio_context.available_capital / portfolio_context.total_capital
        size = min(size, max_size_by_capital)
        
        # Minimum position size
        min_size = 0.001  # 0.1% minimum
        size = max(size, min_size)
        
        return size
    
    def _calculate_conservative_size(
        self, portfolio_context: PortfolioContext
    ) -> float:
        """Calculate conservative position size as fallback.
        
        Args:
            portfolio_context: Portfolio context
            
        Returns:
            Conservative position size
        """
        return min(0.01, portfolio_context.available_capital / portfolio_context.total_capital)
    
    def _log_sizing_decision(self, sizing_details: Dict[str, Any]) -> None:
        """Log position sizing decision.
        
        Args:
            sizing_details: Sizing details to log
        """
        self.sizing_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            **sizing_details
        })
        
        # Keep only recent history (last 1000)
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-1000:]
        
        self.logger.info(
            f"Position sizing: {sizing_details['strategy']} -> "
            f"{sizing_details['final_size']:.4f} "
            f"(risk: {sizing_details['risk_percentage']:.2%})"
        )
    
    def get_sizing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get position sizing history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of sizing history entries
        """
        return self.sizing_history[-limit:]
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get position sizing summary statistics.
        
        Returns:
            Sizing summary dictionary
        """
        if not self.sizing_history:
            return {}
        
        df = pd.DataFrame(self.sizing_history)
        
        summary = {
            'total_sizing_decisions': len(df),
            'average_position_size': df['final_size'].mean(),
            'average_risk_percentage': df['risk_percentage'].mean(),
            'strategy_usage': df['strategy'].value_counts().to_dict(),
            'size_distribution': {
                'min': df['final_size'].min(),
                'max': df['final_size'].max(),
                'std': df['final_size'].std()
            },
            'risk_distribution': {
                'min': df['risk_percentage'].min(),
                'max': df['risk_percentage'].max(),
                'std': df['risk_percentage'].std()
            }
        }
        
        return summary


def create_position_sizer(config: Optional[Dict[str, Any]] = None) -> PositionSizer:
    """Create a position sizer with default configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        PositionSizer instance
    """
    return PositionSizer(config) 