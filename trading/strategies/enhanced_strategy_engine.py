"""
Enhanced Strategy Engine for Evolve Trading Platform

This module provides institutional-level strategy capabilities:
- Dynamic strategy chaining based on market regime
- Automatic strategy combination and optimization
- Continuous performance monitoring and improvement
- Meta-agent loop for strategy retirement and tuning
- Confidence scoring and edge calculation
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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


class PerformanceChecker:
    """Comprehensive meta-agent for strategy and model performance evaluation and improvement."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize PerformanceChecker with configurable thresholds.
        
        Thresholds can be provided directly or loaded from environment variables.
        """
        # Load thresholds from environment variables with defaults
        self.thresholds = thresholds or {
            "min_sharpe_ratio": float(os.getenv("SHARPE_THRESHOLD_POOR", "0.5")),
            "max_drawdown": float(os.getenv("DRAWDOWN_THRESHOLD", "-0.15")),
            "min_win_rate": float(os.getenv("WIN_RATE_THRESHOLD", "0.45")),
            "max_volatility": float(os.getenv("VOLATILITY_THRESHOLD", "0.25")),
            "min_calmar_ratio": float(os.getenv("CALMAR_THRESHOLD", "0.5")),
            "max_mse": float(os.getenv("MSE_THRESHOLD", "0.1")),
            "min_accuracy": float(os.getenv("ACCURACY_THRESHOLD", "0.55")),
        }
        
        # Override with provided thresholds if any
        if thresholds:
            self.thresholds.update(thresholds)

    def check_strategy_performance(
        self, strategy_name: str, performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate strategy performance and recommend actions."""
        sharpe = performance.get("sharpe_ratio", 0)
        drawdown = performance.get("max_drawdown", 0)
        win_rate = performance.get("win_rate", 0)
        volatility = performance.get("volatility", 0)
        performance.get("calmar_ratio", 0)
        total_return = performance.get("total_return", 0)

        # Load retirement thresholds from environment
        retire_sharpe = float(os.getenv("STRATEGY_RETIRE_SHARPE", "0.0"))
        retire_win_rate = float(os.getenv("STRATEGY_RETIRE_WIN_RATE", "0.2"))
        retire_return = float(os.getenv("STRATEGY_RETIRE_RETURN", "-0.2"))
        
        should_retire = (
            sharpe < retire_sharpe
            or drawdown < self.thresholds["max_drawdown"] * 2
            or win_rate < retire_win_rate
            or total_return < retire_return
        )
        should_tune = (
            sharpe < self.thresholds["min_sharpe_ratio"]
            or drawdown < self.thresholds["max_drawdown"]
            or win_rate < self.thresholds["min_win_rate"]
            or volatility > self.thresholds["max_volatility"]
        )
        confidence = max(
            0.0, min(1.0, 0.5 + 0.5 * (sharpe - self.thresholds["min_sharpe_ratio"]))
        )
        recommendations = []
        if should_retire:
            recommendations.append(
                "Retire or replace this strategy due to persistent underperformance."
            )
        elif should_tune:
            recommendations.append(
                "Tune parameters or retrain this strategy to improve performance."
            )
        else:
            recommendations.append(
                "Strategy is performing within acceptable thresholds."
            )
        if volatility > self.thresholds["max_volatility"]:
            recommendations.append(
                "Reduce position size or add risk controls due to high volatility."
            )
        if win_rate < self.thresholds["min_win_rate"]:
            recommendations.append("Review entry/exit logic to improve win rate.")
        return {
            "should_retire": should_retire,
            "should_tune": should_tune and not should_retire,
            "confidence": round(confidence, 3),
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

    def check_model_performance(
        self, model_name: str, performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate model performance and recommend actions."""
        mse = performance.get("mse", 1.0)
        accuracy = performance.get("accuracy", 0.0)
        sharpe = performance.get("sharpe_ratio", 0)
        should_retrain = (
            mse > self.thresholds["max_mse"]
            or accuracy < self.thresholds["min_accuracy"]
        )
        should_replace = sharpe < 0 or mse > self.thresholds["max_mse"] * 2
        recommendations = []
        if should_replace:
            recommendations.append(
                "Replace this model due to poor performance (negative Sharpe or high MSE)."
            )
        elif should_retrain:
            recommendations.append(
                "Retrain or tune this model to improve accuracy and reduce error."
            )
        else:
            recommendations.append("Model is performing within acceptable thresholds.")
        if accuracy < self.thresholds["min_accuracy"]:
            recommendations.append(
                "Collect more data or improve feature engineering to boost accuracy."
            )
        return {
            "should_retrain": should_retrain and not should_replace,
            "should_replace": should_replace,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

    def suggest_improvements(
        self, name: str, performance: Dict[str, float], is_model: bool = False
    ) -> List[str]:
        """Suggest concrete improvements for a strategy or model."""
        suggestions = []
        if is_model:
            if performance.get("mse", 0) > self.thresholds["max_mse"]:
                suggestions.append(
                    "Reduce model complexity or regularize to lower MSE."
                )
            if performance.get("accuracy", 1) < self.thresholds["min_accuracy"]:
                suggestions.append(
                    "Add more training data or features to improve accuracy."
                )
            if performance.get("sharpe_ratio", 0) < self.thresholds["min_sharpe_ratio"]:
                suggestions.append(
                    "Tune hyperparameters or try alternative model architectures."
                )
        else:
            if performance.get("sharpe_ratio", 0) < self.thresholds["min_sharpe_ratio"]:
                suggestions.append(
                    "Increase lookback window or adjust signal thresholds."
                )
            if performance.get("max_drawdown", 0) < self.thresholds["max_drawdown"]:
                suggestions.append(
                    "Add stop-loss or reduce leverage to control drawdown."
                )
            if performance.get("win_rate", 1) < self.thresholds["min_win_rate"]:
                suggestions.append("Refine entry/exit rules to improve win rate.")
            if performance.get("volatility", 0) > self.thresholds["max_volatility"]:
                suggestions.append("Reduce position size or add volatility filters.")
        if not suggestions:
            suggestions.append("No major improvements needed at this time.")
        return suggestions


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
            # from trading.meta_agents.agents.performance_checker import
            # PerformanceChecker  # Removed - meta_agents deleted
            self.meta_agent = PerformanceChecker()
        except ImportError:
            logger.warning("PerformanceChecker not available - using fallback")
            self.meta_agent = self._create_fallback_meta_agent()

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def _create_fallback_regime_classifier(self):
        """Create fallback regime classifier."""

        class FallbackRegimeClassifier:
            def classify_regime(self, data: pd.DataFrame) -> MarketRegime:
                # Simple regime classification based on returns
                returns = data["Close"].pct_change().dropna()
                mean_return = returns.mean()
                volatility = returns.std()

                if mean_return > 0.001 and volatility < 0.02:
                    return MarketRegime.BULL
                elif mean_return < -0.001 and volatility < 0.02:
                    return MarketRegime.BEAR
                elif volatility > 0.03:
                    return MarketRegime.VOLATILE
                else:
                    return {
                        "success": True,
                        "result": {
                            "success": True,
                            "result": MarketRegime.SIDEWAYS,
                            "message": "Operation completed successfully",
                            "timestamp": datetime.now().isoformat(),
                        },
                        "message": "Operation completed successfully",
                        "timestamp": datetime.now().isoformat(),
                    }

            def get_regime_confidence(self) -> float:
                return 0.7

        return FallbackRegimeClassifier()

    def _create_fallback_meta_agent(self):
        """Create fallback meta-agent."""

        class FallbackMetaAgent:
            def check_strategy_performance(
                self, strategy_name: str, performance: Dict[str, float]
            ) -> Dict[str, Any]:
                return {
                    "should_retire": False,
                    "should_tune": False,
                    "confidence": 0.5,
                    "recommendations": [],
                }

            def suggest_improvements(
                self, strategy_name: str, performance: Dict[str, float]
            ) -> List[str]:
                return {
                    "success": True,
                    "result": {
                        "success": True,
                        "result": [
                            "Consider parameter tuning",
                            "Monitor performance closely",
                        ],
                        "message": "Operation completed successfully",
                        "timestamp": datetime.now().isoformat(),
                    },
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                }

        return FallbackMetaAgent()

    def _initialize_strategies(self) -> Dict[str, StrategyConfig]:
        """Initialize strategy configurations."""
        strategies = {
            "momentum": StrategyConfig(
                name="Momentum",
                strategy_type=StrategyType.MOMENTUM,
                parameters={"lookback": 20, "threshold": 0.02},
                confidence=0.8,
                expected_sharpe=1.2,
                max_drawdown=0.15,
                win_rate=0.65,
                regime_compatibility=[MarketRegime.BULL, MarketRegime.VOLATILE],
                risk_level="medium",
                min_volatility=0.01,
                max_volatility=0.05,
            ),
            "mean_reversion": StrategyConfig(
                name="Mean Reversion",
                strategy_type=StrategyType.MEAN_REVERSION,
                parameters={"lookback": 50, "std_threshold": 2.0},
                confidence=0.75,
                expected_sharpe=0.9,
                max_drawdown=0.12,
                win_rate=0.58,
                regime_compatibility=[MarketRegime.SIDEWAYS, MarketRegime.NORMAL],
                risk_level="low",
                min_volatility=0.005,
                max_volatility=0.03,
            ),
            "breakout": StrategyConfig(
                name="Breakout",
                strategy_type=StrategyType.BREAKOUT,
                parameters={"lookback": 30, "breakout_threshold": 0.03},
                confidence=0.7,
                expected_sharpe=1.0,
                max_drawdown=0.18,
                win_rate=0.45,
                regime_compatibility=[MarketRegime.BULL, MarketRegime.VOLATILE],
                risk_level="high",
                min_volatility=0.02,
                max_volatility=0.06,
            ),
            "trend_following": StrategyConfig(
                name="Trend Following",
                strategy_type=StrategyType.TREND_FOLLOWING,
                parameters={"short_window": 10, "long_window": 50},
                confidence=0.85,
                expected_sharpe=1.1,
                max_drawdown=0.14,
                win_rate=0.62,
                regime_compatibility=[MarketRegime.BULL, MarketRegime.NORMAL],
                risk_level="medium",
                min_volatility=0.01,
                max_volatility=0.04,
            ),
            "volatility_trading": StrategyConfig(
                name="Volatility Trading",
                strategy_type=StrategyType.VOLATILITY,
                parameters={"vol_window": 20, "vol_threshold": 0.025},
                confidence=0.65,
                expected_sharpe=0.8,
                max_drawdown=0.20,
                win_rate=0.40,
                regime_compatibility=[MarketRegime.VOLATILE],
                risk_level="high",
                min_volatility=0.03,
                max_volatility=0.08,
            ),
            "defensive": StrategyConfig(
                name="Defensive",
                strategy_type=StrategyType.DEFENSIVE,
                parameters={"stop_loss": 0.05, "position_size": 0.1},
                confidence=0.9,
                expected_sharpe=0.6,
                max_drawdown=0.08,
                win_rate=0.70,
                regime_compatibility=[MarketRegime.BEAR, MarketRegime.VOLATILE],
                risk_level="low",
                min_volatility=0.005,
                max_volatility=0.04,
            ),
            "cash_heavy": StrategyConfig(
                name="Cash Heavy",
                strategy_type=StrategyType.CASH_HEAVY,
                parameters={"cash_allocation": 0.8, "bond_allocation": 0.2},
                confidence=0.95,
                expected_sharpe=0.4,
                max_drawdown=0.05,
                win_rate=0.80,
                regime_compatibility=[MarketRegime.BEAR, MarketRegime.VOLATILE],
                risk_level="low",
                min_volatility=0.001,
                max_volatility=0.02,
            ),
            "options_income": StrategyConfig(
                name="Options Income",
                strategy_type=StrategyType.OPTIONS_INCOME,
                parameters={"delta_target": 0.3, "days_to_expiry": 30},
                confidence=0.6,
                expected_sharpe=0.7,
                max_drawdown=0.25,
                win_rate=0.55,
                regime_compatibility=[MarketRegime.SIDEWAYS, MarketRegime.NORMAL],
                risk_level="medium",
                min_volatility=0.01,
                max_volatility=0.05,
            ),
        }

        return strategies

    def get_strategy_chain(
        self, regime: MarketRegime, risk_tolerance: str
    ) -> List[Dict[str, Any]]:
        """Get dynamic strategy chain based on regime and risk tolerance."""
        compatible_strategies = []

        for name, config in self.strategies.items():
            if regime in config.regime_compatibility:
                # Check risk level compatibility
                risk_compatible = (
                    (risk_tolerance == "low" and config.risk_level in ["low"])
                    or (
                        risk_tolerance == "medium"
                        and config.risk_level in ["low", "medium"]
                    )
                    or (
                        risk_tolerance == "high"
                        and config.risk_level in ["low", "medium", "high"]
                    )
                )

                if risk_compatible:
                    compatible_strategies.append(
                        {
                            "name": name,
                            "config": config,
                            "weight": 1.0,  # Will be normalized
                            "reason": f"Compatible with {regime.value} regime and {risk_tolerance} risk",
                        }
                    )

        if not compatible_strategies:
            # Fallback to defensive strategy
            fallback_config = self.strategies["defensive"]
            compatible_strategies.append(
                {
                    "name": "defensive",
                    "config": fallback_config,
                    "weight": 1.0,
                    "reason": "Fallback strategy - no compatible strategies found",
                }
            )

        # Normalize weights
        total_weight = sum(s["weight"] for s in compatible_strategies)
        for strategy in compatible_strategies:
            strategy["weight"] /= total_weight

        return compatible_strategies

    def execute_strategy_chain(
        self, data: pd.DataFrame, regime: MarketRegime, risk_tolerance: str
    ) -> Dict[str, Any]:
        """Execute a strategy chain and combine results."""
        start_time = datetime.now()

        # Get strategy chain
        strategy_chain = self.get_strategy_chain(regime, risk_tolerance)

        # Execute each strategy
        strategy_results = []
        combined_signals = pd.DataFrame(index=data.index)
        combined_performance = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "volatility": 0.0,
        }

        for strategy_info in strategy_chain:
            strategy_name = strategy_info["name"]
            config = strategy_info["config"]
            weight = strategy_info["weight"]

            # Execute strategy
            result = self._execute_single_strategy(data, config)

            if result:
                # Weight the results
                weighted_signals = result.signals * weight
                weighted_performance = {
                    k: v * weight for k, v in result.performance.items()
                }

                strategy_results.append(
                    {
                        "strategy_name": strategy_name,
                        "weight": weight,
                        "signals": weighted_signals,
                        "performance": weighted_performance,
                        "confidence": result.confidence,
                        "reason": strategy_info["reason"],
                    }
                )

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
            strategy_name="strategy_chain",
            signals=combined_signals,
            performance=combined_performance,
            confidence=np.mean([r["confidence"] for r in strategy_results]),
            regime=regime,
            parameters_used={"strategy_chain": strategy_chain},
            execution_time=execution_time,
            timestamp=datetime.now(),
        )

        # Log performance for meta-agent
        self._log_strategy_performance(combined_result, strategy_results)

        return {
            "success": True,
            "combined_result": combined_result,
            "strategy_results": strategy_results,
            "regime": regime.value,
            "risk_tolerance": risk_tolerance,
            "execution_time": execution_time,
        }

    def _execute_single_strategy(
        self, data: pd.DataFrame, config: StrategyConfig
    ) -> Optional[StrategyResult]:
        """Execute a single strategy with validation."""
        try:
            from trading.strategies.validation import StrategyExecutionValidator

            start_time = datetime.now()
            validator = StrategyExecutionValidator()

            # Validate execution context
            is_valid, error = validator.validate_execution_context(None, data, {"config": config})
            if not is_valid:
                logger.error(f"Execution context validation failed for {config.name}: {error}")
                return None

            # Generate signals based on strategy type
            try:
                signals = self._generate_signals(data, config)
            except Exception as e:
                logger.error(f"Signal generation failed for {config.name}: {e}")
                return None

            # Validate signals
            if isinstance(signals, pd.DataFrame):
                is_valid, error = validator.validate_signals_dataframe(signals, data)
                if not is_valid:
                    logger.error(f"Signals validation failed for {config.name}: {error}")
                    return None
            elif signals is None:
                logger.error(f"Strategy {config.name} returned None signals")
                return None

            # Calculate performance metrics
            performance = self._calculate_performance(signals, data)

            # Adjust confidence based on performance
            confidence = self._adjust_confidence(config.confidence, performance)

            execution_time = (datetime.now() - start_time).total_seconds()

            result = StrategyResult(
                strategy_name=config.name,
                signals=signals,
                performance=performance,
                confidence=confidence,
                regime=MarketRegime.NORMAL,  # Will be set by caller
                parameters_used=config.parameters,
                execution_time=execution_time,
                timestamp=datetime.now(),
            )

            # Validate result
            result_dict = {
                "signals": signals,
                "performance": performance,
            }
            is_valid, error = validator.validate_strategy_result(result_dict)
            if not is_valid:
                logger.error(f"Result validation failed for {config.name}: {error}")
                return None

            return result

        except Exception as e:
            logger.error(f"Strategy execution failed for {config.name}: {e}")
            return None

    def _generate_signals(
        self, data: pd.DataFrame, config: StrategyConfig
    ) -> pd.DataFrame:
        """Generate trading signals based on strategy configuration."""
        signals = pd.DataFrame(index=data.index)
        signals["position"] = 0.0
        signals["signal"] = 0.0

        if config.strategy_type == StrategyType.MOMENTUM:
            lookback = config.parameters["lookback"]
            threshold = config.parameters["threshold"]

            # Calculate momentum
            returns = data["Close"].pct_change(lookback)

            # Generate signals
            signals.loc[returns > threshold, "signal"] = 1.0
            signals.loc[returns < -threshold, "signal"] = -1.0

            # Calculate position size
            signals["position"] = signals["signal"] * 0.1  # 10% position size

        elif config.strategy_type == StrategyType.MEAN_REVERSION:
            lookback = config.parameters["lookback"]
            std_threshold = config.parameters["std_threshold"]

            # Calculate z-score
            rolling_mean = data["Close"].rolling(lookback).mean()
            rolling_std = data["Close"].rolling(lookback).std()
            z_score = (data["Close"] - rolling_mean) / rolling_std

            # Generate signals
            signals.loc[z_score > std_threshold, "signal"] = -1.0  # Sell overvalued
            signals.loc[z_score < -std_threshold, "signal"] = 1.0  # Buy undervalued

            # Calculate position size
            signals["position"] = signals["signal"] * 0.1

        elif config.strategy_type == StrategyType.BREAKOUT:
            lookback = config.parameters["lookback"]
            threshold = config.parameters["breakout_threshold"]

            # Calculate breakout levels
            rolling_high = data["High"].rolling(lookback).max()
            rolling_low = data["Low"].rolling(lookback).min()

            # Generate signals
            signals.loc[data["Close"] > rolling_high * (1 + threshold), "signal"] = 1.0
            signals.loc[data["Close"] < rolling_low * (1 - threshold), "signal"] = -1.0

            # Calculate position size
            signals["position"] = signals["signal"] * 0.15

        elif config.strategy_type == StrategyType.TREND_FOLLOWING:
            short_window = config.parameters["short_window"]
            long_window = config.parameters["long_window"]

            # Calculate moving averages
            short_ma = data["Close"].rolling(short_window).mean()
            long_ma = data["Close"].rolling(long_window).mean()

            # Generate signals
            signals.loc[short_ma > long_ma, "signal"] = 1.0
            signals.loc[short_ma < long_ma, "signal"] = -1.0

            # Calculate position size
            signals["position"] = signals["signal"] * 0.12

        elif config.strategy_type == StrategyType.VOLATILITY:
            vol_window = config.parameters["vol_window"]
            vol_threshold = config.parameters["vol_threshold"]

            # Calculate volatility
            returns = data["Close"].pct_change()
            volatility = returns.rolling(vol_window).std()

            # Generate signals
            signals.loc[volatility > vol_threshold, "signal"] = 1.0
            signals.loc[volatility < vol_threshold * 0.5, "signal"] = -1.0

            # Calculate position size
            signals["position"] = signals["signal"] * 0.08

        elif config.strategy_type == StrategyType.DEFENSIVE:
            stop_loss = config.parameters["stop_loss"]
            position_size = config.parameters["position_size"]

            # Simple defensive strategy - small long position with stop loss
            signals["signal"] = 1.0
            signals["position"] = position_size

            # Apply stop loss (simplified)
            returns = data["Close"].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            stop_loss_triggered = cumulative_returns < (1 - stop_loss)
            signals.loc[stop_loss_triggered, "position"] = 0.0

        elif config.strategy_type == StrategyType.CASH_HEAVY:
            config.parameters["cash_allocation"]
            bond_allocation = config.parameters["bond_allocation"]

            # Cash-heavy strategy - mostly cash with small bond allocation
            signals["signal"] = 0.0
            signals["position"] = bond_allocation

        else:
            # Fallback strategy
            signals["signal"] = 0.0
            signals["position"] = 0.0

        return signals

    def _calculate_performance(
        self, signals: pd.DataFrame, data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics for signals."""
        try:
            # Calculate returns
            price_returns = data["Close"].pct_change()
            strategy_returns = signals["position"].shift(1) * price_returns

            # Remove NaN values
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) == 0:
                return {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "volatility": 0.0,
                }

            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                if strategy_returns.std() > 0
                else 0
            )

            # Calculate max drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calculate win rate
            win_rate = (strategy_returns > 0).mean()

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "volatility": volatility,
            }

        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "volatility": 0.0,
            }

    def _adjust_confidence(
        self, base_confidence: float, performance: Dict[str, float]
    ) -> float:
        """Adjust confidence based on performance metrics."""
        # Adjust based on Sharpe ratio
        sharpe_adjustment = min(
            performance["sharpe_ratio"] / 2.0, 0.2
        )  # Max 20% adjustment

        # Adjust based on win rate
        win_rate_adjustment = (
            performance["win_rate"] - 0.5
        ) * 0.3  # Max 15% adjustment

        # Adjust based on drawdown
        drawdown_adjustment = max(
            performance["max_drawdown"] * 2, -0.2
        )  # Max 20% penalty

        adjusted_confidence = (
            base_confidence
            + sharpe_adjustment
            + win_rate_adjustment
            + drawdown_adjustment
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, adjusted_confidence))

    def _log_strategy_performance(
        self, combined_result: StrategyResult, strategy_results: List[Dict[str, Any]]
    ):
        """Log strategy performance for meta-agent analysis."""
        try:
            # Store performance history
            performance_record = {
                "timestamp": combined_result.timestamp.isoformat(),
                "strategy_name": combined_result.strategy_name,
                "performance": combined_result.performance,
                "confidence": combined_result.confidence,
                "regime": combined_result.regime.value,
                "execution_time": combined_result.execution_time,
                "strategy_breakdown": [
                    {
                        "name": r["strategy_name"],
                        "weight": r["weight"],
                        "performance": r["performance"],
                        "confidence": r["confidence"],
                    }
                    for r in strategy_results
                ],
            }

            self.performance_history.append(performance_record)

            # Keep only last 1000 records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

            # Check with meta-agent
            if self.meta_agent:
                meta_analysis = self.meta_agent.check_strategy_performance(
                    combined_result.strategy_name, combined_result.performance
                )

                if meta_analysis.get("should_retire", False):
                    logger.warning(
                        f"Meta-agent suggests retiring strategy: {combined_result.strategy_name}"
                    )

                if meta_analysis.get("should_tune", False):
                    logger.info(
                        f"Meta-agent suggests tuning strategy: {combined_result.strategy_name}"
                    )

        except Exception as e:
            logger.error(f"Performance logging failed: {e}")

    def get_strategy_performance_history(
        self, strategy_name: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get strategy performance history."""
        if strategy_name:
            return [
                record
                for record in self.performance_history[-limit:]
                if record["strategy_name"] == strategy_name
            ]
        else:
            return self.performance_history[-limit:]

    def get_system_health(self) -> Dict[str, Any]:
        """Get strategy engine health information."""
        try:
            # Calculate recent performance
            recent_performance = (
                self.performance_history[-50:] if self.performance_history else []
            )

            if recent_performance:
                avg_sharpe = np.mean(
                    [p["performance"]["sharpe_ratio"] for p in recent_performance]
                )
                avg_confidence = np.mean([p["confidence"] for p in recent_performance])
                success_rate = len(
                    [
                        p
                        for p in recent_performance
                        if p["performance"]["total_return"] > 0
                    ]
                ) / len(recent_performance)
            else:
                avg_sharpe = 0.0
                avg_confidence = 0.0
                success_rate = 0.0

            return {
                "status": "healthy" if avg_sharpe > 0.5 else "degraded",
                "active_strategies": len(self.strategies),
                "recent_performance_count": len(recent_performance),
                "average_sharpe": avg_sharpe,
                "average_confidence": avg_confidence,
                "success_rate": success_rate,
                "last_execution": (
                    self.performance_history[-1]["timestamp"]
                    if self.performance_history
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}


# Global instance
enhanced_strategy_engine = EnhancedStrategyEngine()


def get_enhanced_strategy_engine() -> EnhancedStrategyEngine:
    """Get the global enhanced strategy engine instance."""
    return enhanced_strategy_engine
