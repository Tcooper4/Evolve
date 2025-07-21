"""
Market Regime Detection Agent

Detects market regimes (bull, bear, sideways) and routes strategies accordingly.
Provides regime-specific strategy recommendations and risk management.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base_agent_interface import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"


@dataclass
class RegimeMetrics:
    """Metrics for regime classification."""

    volatility: float
    trend_strength: float
    momentum: float
    volume_trend: float
    correlation: float
    regime_confidence: float


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""

    current_regime: MarketRegime
    regime_confidence: float
    regime_duration: int
    regime_metrics: RegimeMetrics
    recommended_strategies: List[str]
    risk_level: str
    regime_transition_probability: float
    market_conditions: Dict[str, Any]


@dataclass
class MarketRegimeRequest:
    """Request for market regime analysis."""

    symbol: str = "SPY"
    period: str = "1y"
    analysis_type: str = "full"  # 'full', 'quick', 'detailed'
    include_strategies: bool = True
    include_risk_assessment: bool = True
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarketRegimeResult:
    """Result of market regime analysis."""

    success: bool
    symbol: str
    regime_analysis: Optional[RegimeAnalysis] = None
    market_data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MarketRegimeAgent(BaseAgent):
    """Advanced market regime detection and strategy routing agent."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        lookback_period: int = 252,
        regime_threshold: float = 0.7,
        model_path: Optional[str] = None,
    ):
        if config is None:
            config = AgentConfig(
                name="MarketRegimeAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={},
            )
        super().__init__(config)

        # Extract config from custom_config or use defaults
        custom_config = config.custom_config or {}
        self.lookback_period = custom_config.get("lookback_period", lookback_period)
        self.regime_threshold = custom_config.get("regime_threshold", regime_threshold)
        self.model_path = (
            custom_config.get("model_path", model_path)
            or "models/market_regime_classifier.pkl"
        )

        # Initialize components
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regime_history = []
        self.strategy_registry = self._initialize_strategy_registry()

        # Load or train model
        self._load_or_train_model()

        logger.info("Market Regime Agent initialized successfully")

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the market regime analysis logic.
        Args:
            **kwargs: symbol, action, data, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get("action", "analyze_regime")

            if action == "analyze_regime":
                symbol = kwargs.get("symbol", "SPY")
                analysis = self.analyze_regime(symbol)
                return AgentResult(
                    success=True,
                    data={
                        "regime_analysis": {
                            "current_regime": analysis.current_regime.value,
                            "regime_confidence": analysis.regime_confidence,
                            "regime_duration": analysis.regime_duration,
                            "recommended_strategies": analysis.recommended_strategies,
                            "risk_level": analysis.risk_level,
                            "regime_transition_probability": analysis.regime_transition_probability,
                        }
                    },
                )

            elif action == "get_market_data":
                symbol = kwargs.get("symbol", "SPY")
                period = kwargs.get("period", "1y")

                data = self.get_market_data(symbol, period)
                return AgentResult(
                    success=True,
                    data={
                        "market_data_shape": data.shape,
                        "market_data_columns": list(data.columns),
                        "data_range": {
                            "start": data.index[0].isoformat(),
                            "end": data.index[-1].isoformat(),
                        },
                    },
                )

            elif action == "calculate_regime_features":
                data = kwargs.get("data")

                if data is None:
                    return AgentResult(
                        success=False, error_message="Missing required parameter: data"
                    )

                metrics = self.calculate_regime_features(data)
                return AgentResult(
                    success=True,
                    data={
                        "regime_metrics": {
                            "volatility": metrics.volatility,
                            "trend_strength": metrics.trend_strength,
                            "momentum": metrics.momentum,
                            "volume_trend": metrics.volume_trend,
                            "correlation": metrics.correlation,
                            "regime_confidence": metrics.regime_confidence,
                        }
                    },
                )

            elif action == "get_recommended_strategies":
                regime_str = kwargs.get("regime")
                confidence = kwargs.get("confidence", 0.7)

                if regime_str is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: regime",
                    )

                try:
                    regime = MarketRegime(regime_str)
                    strategies = self.get_recommended_strategies(regime, confidence)
                    return AgentResult(
                        success=True, data={"recommended_strategies": strategies}
                    )
                except ValueError:
                    return AgentResult(
                        success=False, error_message=f"Invalid regime: {regime_str}"
                    )

            elif action == "get_regime_summary":
                summary = self.get_regime_summary()
                return AgentResult(success=True, data={"regime_summary": summary})

            elif action == "get_regime_confidence":
                confidence = self.get_regime_confidence()
                return AgentResult(success=True, data={"regime_confidence": confidence})

            else:
                return AgentResult(
                    success=False, error_message=f"Unknown action: {action}"
                )

        except Exception as e:
            return self.handle_error(e)

    def _initialize_strategy_registry(self) -> Dict[MarketRegime, List[Dict[str, Any]]]:
        """Initialize strategy registry for each regime."""
        return {
            MarketRegime.BULL: [
                {"name": "momentum_trend", "weight": 0.4, "risk_level": "medium"},
                {"name": "breakout", "weight": 0.3, "risk_level": "medium"},
                {"name": "mean_reversion", "weight": 0.2, "risk_level": "low"},
                {"name": "volatility_breakout", "weight": 0.1, "risk_level": "high"},
            ],
            MarketRegime.BEAR: [
                {"name": "short_momentum", "weight": 0.4, "risk_level": "high"},
                {"name": "defensive", "weight": 0.3, "risk_level": "low"},
                {"name": "volatility_short", "weight": 0.2, "risk_level": "high"},
                {"name": "cash_heavy", "weight": 0.1, "risk_level": "very_low"},
            ],
            MarketRegime.SIDEWAYS: [
                {"name": "mean_reversion", "weight": 0.4, "risk_level": "medium"},
                {"name": "range_bound", "weight": 0.3, "risk_level": "medium"},
                {"name": "volatility_breakout", "weight": 0.2, "risk_level": "medium"},
                {"name": "momentum_trend", "weight": 0.1, "risk_level": "medium"},
            ],
            MarketRegime.VOLATILE: [
                {"name": "volatility_breakout", "weight": 0.4, "risk_level": "high"},
                {"name": "mean_reversion", "weight": 0.3, "risk_level": "medium"},
                {"name": "defensive", "weight": 0.2, "risk_level": "low"},
                {"name": "cash_heavy", "weight": 0.1, "risk_level": "very_low"},
            ],
            MarketRegime.TRENDING: [
                {"name": "momentum_trend", "weight": 0.5, "risk_level": "medium"},
                {"name": "breakout", "weight": 0.3, "risk_level": "medium"},
                {"name": "mean_reversion", "weight": 0.2, "risk_level": "low"},
            ],
        }

    def _load_or_train_model(self):
        """Load existing model or train new one."""
        try:
            if os.path.exists(self.model_path):
                self.classifier = joblib.load(self.model_path)
                logger.info(f"Loaded existing regime classifier from {self.model_path}")
            else:
                logger.info("Training new regime classifier...")
                self._train_regime_classifier()
        except Exception as e:
            logger.warning(f"Error loading model: {e}. Training new classifier...")
            self._train_regime_classifier()

    def _train_regime_classifier(self):
        """Train the regime classification model."""
        try:
            # Generate synthetic training data
            X, y = self._generate_training_data()

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train classifier
            self.classifier.fit(X_scaled, y)

            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.classifier, self.model_path)

            logger.info("Regime classifier trained and saved successfully")

        except Exception as e:
            logger.error(f"Error training regime classifier: {e}")

    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for regime classification."""
        n_samples = 10000
        X = np.random.randn(n_samples, 6)  # 6 features
        y = np.random.choice([0, 1, 2, 3, 4], n_samples)  # 5 regimes

        # Add some structure to the data
        for i in range(n_samples):
            if y[i] == 0:  # Bull
                X[i, 1] += 2  # High trend strength
                X[i, 2] += 1.5  # High momentum
            elif y[i] == 1:  # Bear
                X[i, 1] -= 2  # Negative trend strength
                X[i, 2] -= 1.5  # Negative momentum
            elif y[i] == 2:  # Sideways
                X[i, 0] += 1  # High volatility
                X[i, 1] = np.random.normal(0, 0.5)  # Low trend strength
            elif y[i] == 3:  # Volatile
                X[i, 0] += 3  # Very high volatility
            elif y[i] == 4:  # Trending
                X[i, 1] += 1.5  # High trend strength
                X[i, 2] += 1  # High momentum

        return X, y

    def get_market_data(self, symbol: str = "SPY", period: str = "1y") -> pd.DataFrame:
        """Get market data for regime analysis."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                raise ValueError(f"No data received for {symbol}")

            return data

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            # Return synthetic data as fallback
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        data = pd.DataFrame(
            {
                "Open": prices * (1 + np.random.normal(0, 0.002, len(dates))),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
                "Close": prices,
                "Volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        return data

    def calculate_regime_features(self, data: pd.DataFrame) -> RegimeMetrics:
        """Calculate features for regime classification."""
        try:
            # Calculate returns
            returns = data["Close"].pct_change().dropna()

            # Volatility (rolling 20-day)
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

            # Trend strength (linear regression slope)
            prices = data["Close"].values
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            trend_strength = slope / np.mean(prices) * 252

            # Momentum (12-day vs 26-day moving average)
            ma12 = data["Close"].rolling(12).mean()
            ma26 = data["Close"].rolling(26).mean()
            momentum = (ma12.iloc[-1] - ma26.iloc[-1]) / ma26.iloc[-1]

            # Volume trend
            volume_ma = data["Volume"].rolling(20).mean()
            volume_trend = (
                data["Volume"].iloc[-1] - volume_ma.iloc[-1]
            ) / volume_ma.iloc[-1]

            # Correlation with market (using SPY as proxy)
            try:
                spy_data = yf.download(
                    "SPY", start=data.index[0], end=data.index[-1], progress=False
                )
                if not spy_data.empty:
                    spy_returns = spy_data["Close"].pct_change().dropna()
                    common_dates = returns.index.intersection(spy_returns.index)
                    if len(common_dates) > 10:
                        correlation = returns.loc[common_dates].corr(
                            spy_returns.loc[common_dates]
                        )
                    else:
                        correlation = 0.5
                else:
                    correlation = 0.5
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error calculating correlation, using default: {e}")
                correlation = 0.5
            except Exception as e:
                logger.error(f"Unexpected error calculating correlation: {e}")
                correlation = 0.5

            # Regime confidence (based on feature consistency)
            regime_confidence = min(
                1.0, abs(trend_strength) + abs(momentum) + abs(volume_trend)
            )

            return RegimeMetrics(
                volatility=volatility,
                trend_strength=trend_strength,
                momentum=momentum,
                volume_trend=volume_trend,
                correlation=correlation,
                regime_confidence=regime_confidence,
            )

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error calculating regime features: {e}")
            return RegimeMetrics(
                volatility=0.2,
                trend_strength=0.0,
                momentum=0.0,
                volume_trend=0.0,
                correlation=0.5,
                regime_confidence=0.5,
            )
        except Exception as e:
            logger.error(f"Unexpected error calculating regime features: {e}")
            return RegimeMetrics(
                volatility=0.2,
                trend_strength=0.0,
                momentum=0.0,
                volume_trend=0.0,
                correlation=0.5,
                regime_confidence=0.5,
            )

    def classify_regime(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """Classify market regime based on metrics."""
        try:
            # Prepare features for classification
            features = np.array(
                [
                    metrics.volatility,
                    metrics.trend_strength,
                    metrics.momentum,
                    metrics.volume_trend,
                    metrics.correlation,
                    metrics.regime_confidence,
                ]
            ).reshape(1, -1)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict regime
            regime_idx = self.classifier.predict(features_scaled)[0]
            confidence = np.max(self.classifier.predict_proba(features_scaled))

            # Map index to regime
            regime_map = {
                0: MarketRegime.BULL,
                1: MarketRegime.BEAR,
                2: MarketRegime.SIDEWAYS,
                3: MarketRegime.VOLATILE,
                4: MarketRegime.TRENDING,
            }

            return regime_map[regime_idx], confidence

        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return MarketRegime.SIDEWAYS, 0.5

    def get_recommended_strategies(
        self, regime: MarketRegime, confidence: float
    ) -> List[Dict[str, Any]]:
        """Get recommended strategies for the current regime."""
        if regime not in self.strategy_registry:
            return []

        strategies = self.strategy_registry[regime].copy()

        # Adjust weights based on confidence
        if confidence < self.regime_threshold:
            # Reduce risk in uncertain regimes
            for strategy in strategies:
                if strategy["risk_level"] in ["high", "very_high"]:
                    strategy["weight"] *= 0.5

        # Normalize weights
        total_weight = sum(s["weight"] for s in strategies)
        for strategy in strategies:
            strategy["weight"] /= total_weight

        return strategies

    def analyze_regime(self, symbol: str = "SPY") -> RegimeAnalysis:
        """Perform comprehensive regime analysis."""
        try:
            # Get market data
            data = self.get_market_data(symbol)

            # Calculate regime features
            metrics = self.calculate_regime_features(data)

            # Classify regime
            regime, confidence = self.classify_regime(metrics)

            # Get recommended strategies
            strategies = self.get_recommended_strategies(regime, confidence)

            # Calculate regime duration
            regime_duration = self._calculate_regime_duration(regime)

            # Determine risk level
            risk_level = self._determine_risk_level(regime, confidence, metrics)

            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(regime)

            # Market conditions summary
            market_conditions = {
                "volatility_regime": "high" if metrics.volatility > 0.25 else "low",
                "trend_direction": "up" if metrics.trend_strength > 0 else "down",
                "momentum_status": "positive" if metrics.momentum > 0 else "negative",
                "volume_status": (
                    "above_average" if metrics.volume_trend > 0 else "below_average"
                ),
                "market_correlation": (
                    "high" if abs(metrics.correlation) > 0.7 else "low"
                ),
            }

            # Create analysis result
            analysis = RegimeAnalysis(
                current_regime=regime,
                regime_confidence=confidence,
                regime_duration=regime_duration,
                regime_metrics=metrics,
                recommended_strategies=strategies,
                risk_level=risk_level,
                regime_transition_probability=transition_prob,
                market_conditions=market_conditions,
            )

            # Update regime history
            self.regime_history.append(
                {
                    "timestamp": datetime.now(),
                    "regime": regime.value,
                    "confidence": confidence,
                    "symbol": symbol,
                }
            )

            # Keep only last 1000 entries
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]

            logger.info(
                f"Regime analysis completed: {regime.value} (confidence: {confidence:.2f})"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error in regime analysis: {e}")
            return self._create_fallback_analysis()

    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long the current regime has been active."""
        if not self.regime_history:
            return 1

        duration = 1
        for entry in reversed(self.regime_history[:-1]):
            if entry["regime"] == current_regime.value:
                duration += 1
            else:
                break

        return duration

    def _determine_risk_level(
        self, regime: MarketRegime, confidence: float, metrics: RegimeMetrics
    ) -> str:
        """Determine overall risk level based on regime and metrics."""
        base_risk = {
            MarketRegime.BULL: "medium",
            MarketRegime.BEAR: "high",
            MarketRegime.SIDEWAYS: "low",
            MarketRegime.VOLATILE: "high",
            MarketRegime.TRENDING: "medium",
        }

        risk = base_risk[regime]

        # Adjust based on confidence
        if confidence < 0.6:
            risk = "high" if risk != "very_high" else "very_high"

        # Adjust based on volatility
        if metrics.volatility > 0.3:
            risk = "high" if risk != "very_high" else "very_high"

        return risk

    def _calculate_transition_probability(self, current_regime: MarketRegime) -> float:
        """Calculate probability of regime transition."""
        if len(self.regime_history) < 10:
            return 0.1

        # Simple heuristic based on regime duration
        duration = self._calculate_regime_duration(current_regime)

        # Longer regimes are more likely to transition
        if duration > 50:
            return 0.3
        elif duration > 20:
            return 0.2
        else:
            return 0.1

    def _create_fallback_analysis(self) -> RegimeAnalysis:
        """Create fallback analysis when main analysis fails."""
        return RegimeAnalysis(
            current_regime=MarketRegime.SIDEWAYS,
            regime_confidence=0.5,
            regime_duration=1,
            regime_metrics=RegimeMetrics(
                volatility=0.2,
                trend_strength=0.0,
                momentum=0.0,
                volume_trend=0.0,
                correlation=0.5,
                regime_confidence=0.5,
            ),
            recommended_strategies=self.strategy_registry[MarketRegime.SIDEWAYS],
            risk_level="medium",
            regime_transition_probability=0.1,
            market_conditions={
                "volatility_regime": "low",
                "trend_direction": "sideways",
                "momentum_status": "neutral",
                "volume_status": "average",
                "market_correlation": "medium",
            },
        )

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime analysis."""
        try:
            analysis = self.analyze_regime()

            return {
                "current_regime": analysis.current_regime.value,
                "confidence": analysis.regime_confidence,
                "risk_level": analysis.risk_level,
                "recommended_strategies": [
                    {
                        "name": s["name"],
                        "weight": s["weight"],
                        "risk_level": s["risk_level"],
                    }
                    for s in analysis.recommended_strategies
                ],
                "market_conditions": analysis.market_conditions,
                "regime_duration": analysis.regime_duration,
                "transition_probability": analysis.regime_transition_probability,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting regime summary: {e}")
            return {
                "current_regime": "sideways",
                "confidence": 0.5,
                "risk_level": "medium",
                "recommended_strategies": [],
                "market_conditions": {},
                "regime_duration": 1,
                "transition_probability": 0.1,
                "timestamp": datetime.now().isoformat(),
            }

    def save_regime_history(self, filepath: str = "logs/regime_history.json"):
        """Save regime history to file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Convert datetime objects to strings
            history_data = []
            for entry in self.regime_history:
                entry_copy = entry.copy()
                entry_copy["timestamp"] = entry_copy["timestamp"].isoformat()
                history_data.append(entry_copy)

            with open(filepath, "w") as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"Regime history saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving regime history: {e}")

    def load_regime_history(self, filepath: str = "logs/regime_history.json"):
        """Load regime history from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    history_data = json.load(f)

                # Convert string timestamps back to datetime objects
                for entry in history_data:
                    entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])

                self.regime_history = history_data
                logger.info(f"Regime history loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading regime history: {e}")

    def get_regime_confidence(self) -> float:
        """Get confidence level for the current regime classification.

        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            if not hasattr(self, "last_classification_result"):
                return 0.5  # Default confidence if no classification performed

            # Calculate confidence based on classification probabilities
            if (
                hasattr(self, "last_probabilities")
                and self.last_probabilities is not None
            ):
                # Use the highest probability as confidence
                confidence = (
                    max(self.last_probabilities) if self.last_probabilities else 0.5
                )
            else:
                # Fallback confidence calculation
                confidence = 0.7

            # Ensure confidence is within bounds
            return max(0.0, min(1.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5  # Default confidence on error


# Global market regime agent instance
market_regime_agent = MarketRegimeAgent()


def get_market_regime_agent() -> MarketRegimeAgent:
    """Get the global market regime agent instance."""
    return market_regime_agent
