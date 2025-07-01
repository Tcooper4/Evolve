"""Strategy selection agent for intelligent strategy optimization."""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler('trading/optimization/logs/optimization_debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)

class MarketRegime(BaseModel):
    """Market regime classification."""
    
    regime: str = Field(..., description="Market regime type")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in regime classification")
    features: Dict[str, float] = Field(..., description="Market features used for classification")
    
    @classmethod
    def from_data(cls, data: pd.DataFrame) -> "MarketRegime":
        """Create market regime from data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            MarketRegime instance
        """
        # Calculate market features
        returns = data["price"].pct_change()
        volatility = returns.std() * np.sqrt(252)
        trend = (data["price"].iloc[-1] / data["price"].iloc[0]) - 1
        volume_trend = data["volume"].pct_change().mean()
        
        # Classify regime
        if volatility > 0.25:  # High volatility
            regime = "volatile"
            confidence = min(volatility / 0.5, 1.0)
        elif abs(trend) > 0.1:  # Strong trend
            regime = "trending"
            confidence = min(abs(trend) / 0.3, 1.0)
        elif volume_trend > 0.1:  # Increasing volume
            regime = "accumulation"
            confidence = min(volume_trend / 0.3, 1.0)
        else:  # Range-bound
            regime = "ranging"
            confidence = 0.7
            
        return cls(
            regime=regime,
            confidence=confidence,
            features={
                "volatility": volatility,
                "trend": trend,
                "volume_trend": volume_trend
            }
        )

class StrategyPerformance(BaseModel):
    """Strategy performance metrics."""
    
    strategy: str = Field(..., description="Strategy name")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate")
    max_drawdown: float = Field(..., ge=0, le=1, description="Maximum drawdown")
    mse: float = Field(..., ge=0, description="Mean squared error")
    alpha: float = Field(..., description="Strategy alpha")
    regime: str = Field(..., description="Market regime during performance")
    timestamp: datetime = Field(..., description="Performance timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "mse": self.mse,
            "alpha": self.alpha,
            "regime": self.regime,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyPerformance":
        """Create from dictionary.
        
        Args:
            data: Dictionary with performance data
            
        Returns:
            StrategyPerformance instance
        """
        return cls(
            strategy=data["strategy"],
            sharpe_ratio=data["sharpe_ratio"],
            win_rate=data["win_rate"],
            max_drawdown=data["max_drawdown"],
            mse=data["mse"],
            alpha=data["alpha"],
            regime=data["regime"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

class StrategySelectionAgent:
    """Agent for intelligent strategy selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.performance_history: List[StrategyPerformance] = []
        self.regime_history: List[MarketRegime] = []
        
        # Load performance history
        self._load_performance_history()
        
        logger.info("Initialized StrategySelectionAgent")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def select_strategy(self, data: pd.DataFrame,
                       available_strategies: List[str]) -> Tuple[str, float, str]:
        """Select best strategy for current market conditions.
        
        Args:
            data: DataFrame with price data
            available_strategies: List of available strategy names
            
        Returns:
            Tuple of (selected strategy, confidence, explanation)
        """
        try:
            # Classify current market regime
            current_regime = MarketRegime.from_data(data)
            self.regime_history.append(current_regime)
            
            # Get recent performance for each strategy
            recent_performance = self._get_recent_performance(
                available_strategies,
                current_regime.regime
            )
            
            if not recent_performance:
                # No recent performance data, use default strategy
                default_strategy = available_strategies[0]
                explanation = "No recent performance data available"
                logger.warning(explanation)
                return default_strategy, 0.5, explanation
            
            # Score strategies
            strategy_scores = self._score_strategies(recent_performance)
            
            # Select best strategy
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            
            # Generate explanation
            explanation = self._generate_explanation(
                best_strategy[0],
                current_regime,
                recent_performance[best_strategy[0]]
            )
            
            # Log decision
            self._log_decision(
                best_strategy[0],
                best_strategy[1],
                current_regime,
                explanation
            )
            
            return best_strategy[0], best_strategy[1], explanation
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            # Fallback to first available strategy
            return available_strategies[0], 0.5, f"Error: {str(e)}"
    
    def update_performance(self, performance: StrategyPerformance) -> None:
        """Update strategy performance history.
        
        Args:
            performance: StrategyPerformance instance
        """
        self.performance_history.append(performance)
        self._save_performance_history()
        
        logger.info(f"Updated performance for {performance.strategy}")

    def _load_performance_history(self) -> None:
        """Load performance history from file."""
        try:
            log_path = "trading/optimization/logs/optimization_metrics.jsonl"
            if not os.path.exists(log_path):
                pass
            with open(log_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "metrics" in data:
                        performance = StrategyPerformance(
                            strategy=data["strategy"],
                            sharpe_ratio=data["metrics"]["sharpe_ratio"],
                            win_rate=data["metrics"]["win_rate"],
                            max_drawdown=data["metrics"]["max_drawdown"],
                            mse=data["metrics"]["mse"],
                            alpha=data["metrics"]["alpha"],
                            regime=data.get("regime", "unknown"),
                            timestamp=datetime.fromisoformat(data["timestamp"])
                        )
                        self.performance_history.append(performance)
                        
            logger.info(f"Loaded {len(self.performance_history)} performance records")
            
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
    
    def _save_performance_history(self) -> None:
        """Save performance history to file."""
        try:
            log_path = "trading/optimization/logs/optimization_metrics.jsonl"
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory for log_path: {e}")
            
            with open(log_path, "a") as f:
                for performance in self.performance_history[-1:]:  # Save only latest
                    f.write(json.dumps(performance.to_dict()) + "\n")
                    
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")

    def _get_recent_performance(self, strategies: List[str],
                              regime: str) -> Dict[str, List[StrategyPerformance]]:
        """Get recent performance for strategies.
        
        Args:
            strategies: List of strategy names
            regime: Current market regime
            
        Returns:
            Dictionary mapping strategy names to lists of recent performance
        """
        recent_performance = {}
        cutoff_date = datetime.utcnow() - timedelta(days=30)  # Last 30 days
        
        for strategy in strategies:
            strategy_performance = [
                p for p in self.performance_history
                if p.strategy == strategy
                and p.timestamp > cutoff_date
                and p.regime == regime
            ]
            if strategy_performance:
                recent_performance[strategy] = strategy_performance
                
        return recent_performance
    
    def _score_strategies(self, recent_performance: Dict[str, List[StrategyPerformance]]) -> Dict[str, float]:
        """Score strategies based on recent performance.
        
        Args:
            recent_performance: Dictionary of recent performance by strategy
            
        Returns:
            Dictionary mapping strategy names to scores
        """
        scores = {}
        
        for strategy, performance_list in recent_performance.items():
            # Calculate average metrics
            avg_sharpe = np.mean([p.sharpe_ratio for p in performance_list])
            avg_win_rate = np.mean([p.win_rate for p in performance_list])
            avg_drawdown = np.mean([p.max_drawdown for p in performance_list])
            avg_alpha = np.mean([p.alpha for p in performance_list])
            
            # Calculate score (weighted average of metrics)
            score = (
                0.4 * avg_sharpe +
                0.3 * avg_win_rate +
                0.2 * (1 - avg_drawdown) +  # Invert drawdown
                0.1 * avg_alpha
            )
            
            scores[strategy] = score
            
        return scores
    
    def _generate_explanation(self, strategy: str, regime: MarketRegime,
                            performance: List[StrategyPerformance]) -> str:
        """Generate explanation for strategy selection.
        
        Args:
            strategy: Selected strategy name
            regime: Current market regime
            performance: Recent performance for strategy
            
        Returns:
            Explanation string
        """
        # Calculate average metrics
        avg_sharpe = np.mean([p.sharpe_ratio for p in performance])
        avg_win_rate = np.mean([p.win_rate for p in performance])
        
        return (
            f"Selected {strategy} for {regime.regime} market "
            f"(confidence: {regime.confidence:.2f}). "
            f"Recent performance: Sharpe={avg_sharpe:.2f}, "
            f"Win Rate={avg_win_rate:.2f}"
        )
    
    def _log_decision(self, strategy: str, confidence: float,
                     regime: MarketRegime, explanation: str) -> None:
        """Log strategy selection decision.
        
        Args:
            strategy: Selected strategy name
            confidence: Selection confidence
            regime: Current market regime
            explanation: Selection explanation
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": strategy,
            "confidence": confidence,
            "regime": regime.regime,
            "regime_confidence": regime.confidence,
            "explanation": explanation
        }
        
        log_path = "trading/optimization/logs/optimization_agent_decisions.jsonl"
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory for log_path: {e}")
        
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        logger.info(f"Logged decision: {explanation}") 
