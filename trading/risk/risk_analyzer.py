"""Risk analyzer agent module with regime detection and LLM integration."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import openai
import pandas as pd

from .risk_metrics import (
    calculate_advanced_metrics,
    calculate_regime_metrics,
    calculate_rolling_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading/risk/logs/risk_analyzer.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Container for risk assessment results."""

    timestamp: datetime
    regime: str
    risk_level: str
    forecast_risk_score: float
    metrics: Dict[str, float]
    explanation: str
    recommendations: List[str]


class RiskAnalyzer:
    """Agent for analyzing and interpreting risk metrics."""

    def __init__(
        self, openai_api_key: Optional[str] = None, memory_path: str = "trading/risk/memory/risk_assessments.json"
    ):
        """Initialize risk analyzer.

        Args:
            openai_api_key: OpenAI API key (optional)
            memory_path: Path to store risk assessments
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.memory_path = memory_path
        self.last_assessment = None

        # Create memory directory if needed
        try:
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory for memory_path: {e}")

        # Initialize OpenAI if key is available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def analyze_risk(self, returns: pd.Series, forecast_confidence: float, historical_error: float) -> RiskAssessment:
        """Analyze risk metrics and generate assessment.

        Args:
            returns: Daily returns series
            forecast_confidence: Model forecast confidence (0-1)
            historical_error: Historical forecast error

        Returns:
            RiskAssessment object
        """
        # Calculate metrics
        rolling_metrics = calculate_rolling_metrics(returns)
        advanced_metrics = calculate_advanced_metrics(returns)
        regime_metrics = calculate_regime_metrics(returns)

        # Calculate forecast risk score
        forecast_risk_score = self._calculate_forecast_risk(forecast_confidence, historical_error, regime_metrics)

        # Determine risk level
        risk_level = self._determine_risk_level(forecast_risk_score, regime_metrics)

        # Generate explanation
        explanation = self._generate_risk_explanation(risk_level, regime_metrics, forecast_risk_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(risk_level, regime_metrics)

        # Create assessment
        assessment = RiskAssessment(
            timestamp=datetime.now(),
            regime=regime_metrics["regime"],
            risk_level=risk_level,
            forecast_risk_score=forecast_risk_score,
            metrics={**regime_metrics, **advanced_metrics},
            explanation=explanation,
            recommendations=recommendations,
        )

        # Store assessment
        self._store_assessment(assessment)
        self.last_assessment = assessment

        return assessment

    def _calculate_forecast_risk(
        self, forecast_confidence: float, historical_error: float, regime_metrics: Dict[str, float]
    ) -> float:
        """Calculate forecast risk score.

        Args:
            forecast_confidence: Model forecast confidence
            historical_error: Historical forecast error
            regime_metrics: Regime-specific metrics

        Returns:
            Forecast risk score (0-1)
        """
        # Weight factors
        confidence_weight = 0.4
        error_weight = 0.3
        regime_weight = 0.3

        # Regime risk factor
        regime_risk = {"bull": 0.2, "neutral": 0.5, "bear": 0.8}[regime_metrics["regime"]]

        # Calculate weighted score
        risk_score = (
            (1 - forecast_confidence) * confidence_weight
            + historical_error * error_weight
            + regime_risk * regime_weight
        )

        return {
            "success": True,
            "result": min(max(risk_score, 0), 1),
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _determine_risk_level(self, forecast_risk_score: float, regime_metrics: Dict[str, float]) -> str:
        """Determine risk level based on metrics.

        Args:
            forecast_risk_score: Calculated forecast risk score
            regime_metrics: Regime-specific metrics

        Returns:
            Risk level string
        """
        if forecast_risk_score > 0.7 or regime_metrics["max_drawdown"] < -0.2:
            return "high"
        elif forecast_risk_score > 0.4 or regime_metrics["volatility"] > 0.25:
            return "moderate"
        else:
            return "low"

    def _generate_risk_explanation(
        self, risk_level: str, regime_metrics: Dict[str, float], forecast_risk_score: float
    ) -> str:
        """Generate risk explanation using LLM.

        Args:
            risk_level: Determined risk level
            regime_metrics: Regime-specific metrics
            forecast_risk_score: Calculated forecast risk score

        Returns:
            Risk explanation string
        """
        if not self.openai_api_key:
            return self._generate_basic_explanation(risk_level, regime_metrics, forecast_risk_score)

        try:
            prompt = f"""
            Analyze the following risk metrics and provide a concise explanation:

            Risk Level: {risk_level}
            Market Regime: {regime_metrics['regime']}
            Sharpe Ratio: {regime_metrics['sharpe_ratio']:.2f}
            Volatility: {regime_metrics['volatility']:.2f}
            Max Drawdown: {regime_metrics['max_drawdown']:.2f}
            Forecast Risk Score: {forecast_risk_score:.2f}

            Provide a clear, professional explanation of the current risk environment
            and its implications for trading strategy.
            """

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional risk analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating LLM explanation: {e}")
            return self._generate_basic_explanation(risk_level, regime_metrics, forecast_risk_score)

    def _generate_basic_explanation(
        self, risk_level: str, regime_metrics: Dict[str, float], forecast_risk_score: float
    ) -> str:
        """Generate basic risk explanation without LLM.

        Args:
            risk_level: Determined risk level
            regime_metrics: Regime-specific metrics
            forecast_risk_score: Calculated forecast risk score

        Returns:
            Basic risk explanation string
        """
        return (
            f"Current risk level is {risk_level} based on {regime_metrics['regime']} "
            f"market regime. Sharpe ratio is {regime_metrics['sharpe_ratio']:.2f} "
            f"with volatility of {regime_metrics['volatility']:.2f}. "
            f"Forecast risk score is {forecast_risk_score:.2f}."
        )

    def _generate_recommendations(self, risk_level: str, regime_metrics: Dict[str, float]) -> List[str]:
        """Generate risk-based recommendations.

        Args:
            risk_level: Determined risk level
            regime_metrics: Regime-specific metrics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if risk_level == "high":
            recommendations.extend(
                [
                    "Reduce position sizes",
                    "Increase stop-loss thresholds",
                    "Consider hedging positions",
                    "Focus on capital preservation",
                ]
            )
        elif risk_level == "moderate":
            recommendations.extend(
                [
                    "Maintain normal position sizes",
                    "Monitor volatility closely",
                    "Consider partial profit taking",
                    "Review risk management rules",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Consider increasing position sizes",
                    "Look for new opportunities",
                    "Maintain current risk management",
                    "Monitor for regime changes",
                ]
            )

        return recommendations

    def _store_assessment(self, assessment: RiskAssessment):
        """Store risk assessment in memory.

        Args:
            assessment: RiskAssessment object to store
        """
        try:
            # Load existing assessments
            if os.path.exists(self.memory_path):
                with open(self.memory_path, "r") as f:
                    assessments = json.load(f)
            else:
                assessments = []

            # Convert assessment to dict
            assessment_dict = {
                "timestamp": assessment.timestamp.isoformat(),
                "regime": assessment.regime,
                "risk_level": assessment.risk_level,
                "forecast_risk_score": assessment.forecast_risk_score,
                "metrics": assessment.metrics,
                "explanation": assessment.explanation,
                "recommendations": assessment.recommendations,
            }

            # Add new assessment
            assessments.append(assessment_dict)

            # Keep only last 100 assessments
            if len(assessments) > 100:
                assessments = assessments[-100:]

            # Save updated assessments
            with open(self.memory_path, "w") as f:
                json.dump(assessments, f, indent=2)

        except Exception as e:
            logger.error(f"Error storing risk assessment: {e}")

    def get_last_assessment(self) -> Optional[RiskAssessment]:
        """Get the last risk assessment.

        Returns:
            Last RiskAssessment object or None
        """
        return self.last_assessment
