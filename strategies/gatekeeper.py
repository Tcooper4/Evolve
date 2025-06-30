"""Regime-Switching Strategy Gate for Evolve Trading Platform.

This module provides intelligent strategy selection and switching based on
market regimes, volatility conditions, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

# Technical analysis imports
try:
    import talib
    TA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"TA-Lib not available: {e}")
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str  # "bull", "bear", "neutral", "crisis", "volatile"
    confidence: float
    start_date: datetime
    end_date: Optional[datetime] = None
    duration: int = 0
    volatility: float = 0.0
    trend_strength: float = 0.0
    momentum: float = 0.0

@dataclass
class StrategyStatus:
    """Strategy status and performance."""
    strategy_name: str
    is_active: bool
    performance_score: float
    regime_fit: float
    volatility_fit: float
    momentum_fit: float
    last_switch: datetime
    switch_reason: str
    pnl_decay: float
    risk_score: float

@dataclass
class GatekeeperDecision:
    """Gatekeeper decision for strategy switching."""
    strategy_name: str
    action: str  # "activate", "deactivate", "maintain"
    confidence: float
    reason: str
    expected_performance: float
    risk_level: str
    timestamp: datetime

@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    strategy_name: str
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    last_updated: str
    is_active: bool
    performance_score: float
    decay_threshold: float = 0.5
    min_trades: int = 10

class RegimeClassifier:
    """Classifies market regimes based on multiple indicators."""
    
    def __init__(self, 
                 lookback_period: int = 60,
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.01):
        """Initialize regime classifier.
        
        Args:
            lookback_period: Period for regime classification
            volatility_threshold: Volatility threshold for regime changes
            trend_threshold: Trend strength threshold
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
        # Regime parameters
        self.bull_threshold = 0.02  # 2% positive trend
        self.bear_threshold = -0.02  # -2% negative trend
        self.crisis_threshold = -0.05  # -5% crisis threshold
        self.volatile_threshold = 0.03  # 3% volatility threshold
        
        logger.info(f"Initialized Regime Classifier with {lookback_period} day lookback")
    
    def classify_regime(self, 
                       prices: pd.Series,
                       volumes: Optional[pd.Series] = None) -> MarketRegime:
        """Classify current market regime.
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
            
        Returns:
            Market regime classification
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            if len(returns) < self.lookback_period:
                return MarketRegime(
                    regime="neutral",
                    confidence=0.5,
                    start_date=datetime.now(),
                    volatility=returns.std() if len(returns) > 0 else 0.0
                )
            
            # Calculate regime indicators
            volatility = returns.rolling(window=self.lookback_period).std().iloc[-1]
            trend = returns.rolling(window=self.lookback_period).mean().iloc[-1]
            momentum = returns.rolling(window=20).mean().iloc[-1]
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(prices)
            
            # Classify regime
            regime, confidence = self._determine_regime(volatility, trend, momentum, trend_strength)
            
            return MarketRegime(
                regime=regime,
                confidence=confidence,
                start_date=datetime.now(),
                volatility=volatility,
                trend_strength=trend_strength,
                momentum=momentum
            )
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return MarketRegime(
                regime="neutral",
                confidence=0.5,
                start_date=datetime.now()
            )
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using ADX or similar."""
        try:
            if TA_AVAILABLE and len(prices) > 14:
                # Use ADX for trend strength
                high = prices  # Simplified - use price as high/low
                low = prices
                close = prices
                
                adx = talib.ADX(high, low, close, timeperiod=14)
                return adx.iloc[-1] / 100.0 if not np.isnan(adx.iloc[-1]) else 0.5
            else:
                # Manual trend strength calculation
                returns = prices.pct_change().dropna()
                if len(returns) > 20:
                    # Calculate directional movement
                    positive_moves = (returns > 0).rolling(window=20).mean().iloc[-1]
                    negative_moves = (returns < 0).rolling(window=20).mean().iloc[-1]
                    trend_strength = abs(positive_moves - negative_moves)
                    return min(trend_strength, 1.0)
                else:
                    return 0.5
                    
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _determine_regime(self, 
                         volatility: float,
                         trend: float,
                         momentum: float,
                         trend_strength: float) -> Tuple[str, float]:
        """Determine market regime based on indicators."""
        confidence = 0.0
        
        # Crisis regime
        if trend < self.crisis_threshold and volatility > self.volatile_threshold:
            confidence = min(0.9, abs(trend) / self.crisis_threshold)
            return "crisis", confidence
        
        # Bear regime
        elif trend < self.bear_threshold:
            confidence = min(0.8, abs(trend) / self.bear_threshold)
            return "bear", confidence
        
        # Bull regime
        elif trend > self.bull_threshold and trend_strength > 0.6:
            confidence = min(0.8, trend / self.bull_threshold)
            return "bull", confidence
        
        # Volatile regime
        elif volatility > self.volatile_threshold:
            confidence = min(0.7, volatility / self.volatile_threshold)
            return "volatile", confidence
        
        # Neutral regime
        else:
            confidence = 0.6
            return "neutral", confidence

class StrategyGatekeeper:
    """Main strategy gatekeeper for regime-based switching."""
    
    def __init__(self, 
                 strategies: Dict[str, Dict[str, Any]],
                 regime_classifier: Optional[RegimeClassifier] = None):
        """Initialize strategy gatekeeper.
        
        Args:
            strategies: Dictionary of strategy configurations
            regime_classifier: Regime classifier instance
        """
        self.strategies = strategies
        self.regime_classifier = regime_classifier or RegimeClassifier()
        
        # Strategy status tracking
        self.strategy_status = {}
        self.regime_history = []
        self.switch_history = []
        
        # Performance tracking
        self.performance_window = 252  # 1 year
        self.decay_threshold = 0.1  # 10% performance decay threshold
        
        # Initialize strategy status
        self._initialize_strategy_status()
        
        # Create output directory
        self.output_dir = Path("strategies/gatekeeper")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Strategy Gatekeeper with {len(strategies)} strategies")
    
    def _initialize_strategy_status(self):
        """Initialize strategy status tracking."""
        for strategy_name, config in self.strategies.items():
            self.strategy_status[strategy_name] = StrategyStatus(
                strategy_name=strategy_name,
                is_active=config.get("default_active", False),
                performance_score=0.5,
                regime_fit=0.5,
                volatility_fit=0.5,
                momentum_fit=0.5,
                last_switch=datetime.now(),
                switch_reason="initialization",
                pnl_decay=0.0,
                risk_score=0.5
            )
    
    def update_market_data(self, 
                          prices: pd.Series,
                          volumes: Optional[pd.Series] = None) -> MarketRegime:
        """Update market data and classify regime.
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            Current market regime
        """
        # Classify current regime
        current_regime = self.regime_classifier.classify_regime(prices, volumes)
        
        # Update regime history
        if self.regime_history:
            # End previous regime
            self.regime_history[-1].end_date = current_regime.start_date
            self.regime_history[-1].duration = (
                current_regime.start_date - self.regime_history[-1].start_date
            ).days
        
        self.regime_history.append(current_regime)
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        logger.info(f"Market regime updated: {current_regime.regime} (confidence: {current_regime.confidence:.2f})")
        
        return current_regime
    
    def evaluate_strategy_fit(self, 
                            strategy_name: str,
                            current_regime: MarketRegime,
                            performance_data: Optional[Dict[str, float]] = None) -> float:
        """Evaluate how well a strategy fits the current regime.
        
        Args:
            strategy_name: Name of the strategy
            current_regime: Current market regime
            performance_data: Recent performance data
            
        Returns:
            Fit score (0-1)
        """
        if strategy_name not in self.strategies:
            return 0.0
        
        strategy_config = self.strategies[strategy_name]
        
        # Regime fit
        regime_fit = self._calculate_regime_fit(strategy_config, current_regime)
        
        # Volatility fit
        volatility_fit = self._calculate_volatility_fit(strategy_config, current_regime)
        
        # Momentum fit
        momentum_fit = self._calculate_momentum_fit(strategy_config, current_regime)
        
        # Performance fit
        performance_fit = self._calculate_performance_fit(performance_data) if performance_data else 0.5
        
        # Weighted combination
        fit_score = (
            regime_fit * 0.4 +
            volatility_fit * 0.3 +
            momentum_fit * 0.2 +
            performance_fit * 0.1
        )
        
        return min(max(fit_score, 0.0), 1.0)
    
    def _calculate_regime_fit(self, 
                            strategy_config: Dict[str, Any],
                            current_regime: MarketRegime) -> float:
        """Calculate regime fit score."""
        preferred_regimes = strategy_config.get("preferred_regimes", ["neutral"])
        regime_weights = strategy_config.get("regime_weights", {})
        
        if current_regime.regime in preferred_regimes:
            base_fit = 0.8
        else:
            base_fit = 0.3
        
        # Apply regime-specific weights
        weight = regime_weights.get(current_regime.regime, 1.0)
        
        return min(base_fit * weight, 1.0)
    
    def _calculate_volatility_fit(self, 
                                strategy_config: Dict[str, Any],
                                current_regime: MarketRegime) -> float:
        """Calculate volatility fit score."""
        preferred_volatility = strategy_config.get("preferred_volatility", "medium")
        volatility_range = strategy_config.get("volatility_range", [0.01, 0.03])
        
        current_vol = current_regime.volatility
        
        if preferred_volatility == "low" and current_vol < volatility_range[0]:
            return 0.9
        elif preferred_volatility == "medium" and volatility_range[0] <= current_vol <= volatility_range[1]:
            return 0.9
        elif preferred_volatility == "high" and current_vol > volatility_range[1]:
            return 0.9
        else:
            return 0.3
    
    def _calculate_momentum_fit(self, 
                              strategy_config: Dict[str, Any],
                              current_regime: MarketRegime) -> float:
        """Calculate momentum fit score."""
        momentum_requirement = strategy_config.get("momentum_requirement", "any")
        
        if momentum_requirement == "positive" and current_regime.momentum > 0:
            return 0.9
        elif momentum_requirement == "negative" and current_regime.momentum < 0:
            return 0.9
        elif momentum_requirement == "strong" and abs(current_regime.momentum) > 0.02:
            return 0.9
        elif momentum_requirement == "any":
            return 0.7
        else:
            return 0.3
    
    def _calculate_performance_fit(self, performance_data: Dict[str, float]) -> float:
        """Calculate performance fit score."""
        if not performance_data:
            return 0.5
        
        # Consider multiple performance metrics
        sharpe_ratio = performance_data.get("sharpe_ratio", 0.0)
        max_drawdown = performance_data.get("max_drawdown", 0.0)
        win_rate = performance_data.get("win_rate", 0.5)
        
        # Normalize metrics
        sharpe_score = min(max(sharpe_ratio / 2.0, 0.0), 1.0)  # Normalize to 0-1
        drawdown_score = max(0.0, 1.0 - abs(max_drawdown))  # Lower drawdown is better
        win_rate_score = win_rate
        
        # Weighted combination
        performance_fit = (
            sharpe_score * 0.4 +
            drawdown_score * 0.3 +
            win_rate_score * 0.3
        )
        
        return performance_fit
    
    def make_switching_decisions(self, 
                               current_regime: MarketRegime,
                               performance_data: Optional[Dict[str, Dict[str, float]]] = None) -> List[GatekeeperDecision]:
        """Make strategy switching decisions.
        
        Args:
            current_regime: Current market regime
            performance_data: Performance data for each strategy
            
        Returns:
            List of switching decisions
        """
        decisions = []
        
        for strategy_name in self.strategies.keys():
            # Evaluate strategy fit
            strategy_performance = performance_data.get(strategy_name, {}) if performance_data else {}
            fit_score = self.evaluate_strategy_fit(strategy_name, current_regime, strategy_performance)
            
            # Get current status
            current_status = self.strategy_status[strategy_name]
            
            # Calculate performance decay
            pnl_decay = self._calculate_pnl_decay(strategy_name, strategy_performance)
            
            # Make decision
            decision = self._make_strategy_decision(
                strategy_name,
                current_status,
                fit_score,
                pnl_decay,
                current_regime
            )
            
            if decision:
                decisions.append(decision)
                
                # Update strategy status
                self._update_strategy_status(strategy_name, decision, fit_score, pnl_decay)
        
        # Record decisions
        self.switch_history.extend(decisions)
        
        # Keep only recent history
        if len(self.switch_history) > 1000:
            self.switch_history = self.switch_history[-1000:]
        
        return decisions
    
    def _calculate_pnl_decay(self, 
                           strategy_name: str,
                           performance_data: Dict[str, float]) -> float:
        """Calculate PnL decay for a strategy."""
        if not performance_data:
            return 0.0
        
        # Get recent performance metrics
        recent_sharpe = performance_data.get("sharpe_ratio", 0.0)
        recent_drawdown = performance_data.get("max_drawdown", 0.0)
        
        # Compare with historical performance (simplified)
        historical_sharpe = 1.0  # Assume historical Sharpe of 1.0
        historical_drawdown = 0.1  # Assume historical max drawdown of 10%
        
        # Calculate decay
        sharpe_decay = max(0.0, (historical_sharpe - recent_sharpe) / historical_sharpe)
        drawdown_decay = max(0.0, (recent_drawdown - historical_drawdown) / historical_drawdown)
        
        # Combined decay
        total_decay = (sharpe_decay + drawdown_decay) / 2.0
        
        return min(total_decay, 1.0)
    
    def _make_strategy_decision(self, 
                              strategy_name: str,
                              current_status: StrategyStatus,
                              fit_score: float,
                              pnl_decay: float,
                              current_regime: MarketRegime) -> Optional[GatekeeperDecision]:
        """Make individual strategy decision."""
        strategy_config = self.strategies[strategy_name]
        
        # Decision thresholds
        activation_threshold = strategy_config.get("activation_threshold", 0.7)
        deactivation_threshold = strategy_config.get("deactivation_threshold", 0.3)
        decay_threshold = strategy_config.get("decay_threshold", 0.2)
        
        # Calculate decision confidence
        confidence = fit_score * (1.0 - pnl_decay)
        
        # Determine action
        if current_status.is_active:
            # Strategy is currently active
            if fit_score < deactivation_threshold or pnl_decay > decay_threshold:
                action = "deactivate"
                reason = f"Low fit score ({fit_score:.2f}) or high decay ({pnl_decay:.2f})"
            else:
                action = "maintain"
                reason = f"Strategy performing well (fit: {fit_score:.2f}, decay: {pnl_decay:.2f})"
        else:
            # Strategy is currently inactive
            if fit_score > activation_threshold and pnl_decay < decay_threshold:
                action = "activate"
                reason = f"High fit score ({fit_score:.2f}) and low decay ({pnl_decay:.2f})"
            else:
                action = "maintain"
                reason = f"Strategy not suitable for current conditions"
        
        # Calculate expected performance
        expected_performance = fit_score * (1.0 - pnl_decay)
        
        # Determine risk level
        if expected_performance > 0.7:
            risk_level = "low"
        elif expected_performance > 0.4:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return GatekeeperDecision(
            strategy_name=strategy_name,
            action=action,
            confidence=confidence,
            reason=reason,
            expected_performance=expected_performance,
            risk_level=risk_level,
            timestamp=datetime.now()
        )
    
    def _update_strategy_status(self, 
                              strategy_name: str,
                              decision: GatekeeperDecision,
                              fit_score: float,
                              pnl_decay: float):
        """Update strategy status based on decision."""
        status = self.strategy_status[strategy_name]
        
        if decision.action == "activate":
            status.is_active = True
        elif decision.action == "deactivate":
            status.is_active = False
        
        status.performance_score = decision.expected_performance
        status.regime_fit = fit_score
        status.pnl_decay = pnl_decay
        status.last_switch = decision.timestamp
        status.switch_reason = decision.reason
        status.risk_score = 1.0 - decision.expected_performance
    
    def get_active_strategies(self) -> List[str]:
        """Get list of currently active strategies."""
        return [name for name, status in self.strategy_status.items() if status.is_active]
    
    def get_strategy_status(self, strategy_name: str) -> Optional[StrategyStatus]:
        """Get status of a specific strategy."""
        return {'success': True, 'result': self.strategy_status.get(strategy_name), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime and strategy status."""
        if not self.regime_history:
            return {'success': True, 'result': {"error": "No regime history available"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        current_regime = self.regime_history[-1]
        active_strategies = self.get_active_strategies()
        
        return {
            "current_regime": {
                "regime": current_regime.regime,
                "confidence": current_regime.confidence,
                "volatility": current_regime.volatility,
                "trend_strength": current_regime.trend_strength,
                "momentum": current_regime.momentum
            },
            "active_strategies": active_strategies,
            "total_strategies": len(self.strategies),
            "regime_duration": current_regime.duration,
            "last_switch": max([status.last_switch for status in self.strategy_status.values()])
        }
    
    def save_gatekeeper_state(self, filepath: Optional[str] = None):
        """Save gatekeeper state to file."""
        if filepath is None:
            filepath = self.output_dir / f"gatekeeper_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        state = {
            "strategy_status": {
                name: {
                    "is_active": status.is_active,
                    "performance_score": status.performance_score,
                    "regime_fit": status.regime_fit,
                    "pnl_decay": status.pnl_decay,
                    "last_switch": status.last_switch.isoformat(),
                    "switch_reason": status.switch_reason
                }
                for name, status in self.strategy_status.items()
            },
            "regime_history": [
                {
                    "regime": regime.regime,
                    "confidence": regime.confidence,
                    "start_date": regime.start_date.isoformat(),
                    "end_date": regime.end_date.isoformat() if regime.end_date else None,
                    "duration": regime.duration,
                    "volatility": regime.volatility
                }
                for regime in self.regime_history[-10:]  # Save last 10 regimes
            ],
            "switch_history": [
                {
                    "strategy_name": decision.strategy_name,
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "timestamp": decision.timestamp.isoformat()
                }
                for decision in self.switch_history[-50:]  # Save last 50 switches
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Gatekeeper state saved to {filepath}")

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def create_strategy_gatekeeper(strategies_config: Dict[str, Dict[str, Any]]) -> StrategyGatekeeper:
    """Create strategy gatekeeper with default configurations.
    
    Args:
        strategies_config: Strategy configurations
        
    Returns:
        Strategy gatekeeper instance
    """
    # Add default configurations for strategies
    default_configs = {
        "momentum": {
            "default_active": True,
            "preferred_regimes": ["bull", "neutral"],
            "regime_weights": {"bull": 1.2, "neutral": 1.0, "bear": 0.5, "crisis": 0.2},
            "preferred_volatility": "medium",
            "volatility_range": [0.01, 0.03],
            "momentum_requirement": "positive",
            "activation_threshold": 0.7,
            "deactivation_threshold": 0.3,
            "decay_threshold": 0.2
        },
        "mean_reversion": {
            "default_active": True,
            "preferred_regimes": ["neutral", "volatile"],
            "regime_weights": {"neutral": 1.2, "volatile": 1.1, "bull": 0.8, "bear": 0.8},
            "preferred_volatility": "high",
            "volatility_range": [0.02, 0.05],
            "momentum_requirement": "any",
            "activation_threshold": 0.6,
            "deactivation_threshold": 0.4,
            "decay_threshold": 0.25
        },
        "trend_following": {
            "default_active": False,
            "preferred_regimes": ["bull", "bear"],
            "regime_weights": {"bull": 1.3, "bear": 1.1, "neutral": 0.7, "crisis": 0.3},
            "preferred_volatility": "medium",
            "volatility_range": [0.015, 0.04],
            "momentum_requirement": "strong",
            "activation_threshold": 0.75,
            "deactivation_threshold": 0.25,
            "decay_threshold": 0.15
        },
        "breakout": {
            "default_active": False,
            "preferred_regimes": ["bull", "volatile"],
            "regime_weights": {"bull": 1.2, "volatile": 1.1, "neutral": 0.6, "bear": 0.4},
            "preferred_volatility": "high",
            "volatility_range": [0.025, 0.06],
            "momentum_requirement": "positive",
            "activation_threshold": 0.8,
            "deactivation_threshold": 0.2,
            "decay_threshold": 0.2
        }
    }
    
    # Merge with provided configurations
    for strategy_name, config in strategies_config.items():
        if strategy_name in default_configs:
            default_configs[strategy_name].update(config)
        else:
            default_configs[strategy_name] = config
    
    return {'success': True, 'result': StrategyGatekeeper(default_configs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}