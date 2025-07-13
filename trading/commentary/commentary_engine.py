"""
Unified Commentary Engine

This engine provides comprehensive LLM-based commentary for trading decisions,
performance analysis, and market insights. It consolidates and enhances existing
commentary capabilities with advanced features.

Features:
- Trade decision explanations and rationales
- Performance analysis and attribution
- Market regime commentary
- Risk assessment and warnings
- Strategy recommendations
- Counterfactual analysis
- Real-time commentary generation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trading.llm.llm_interface import LLMInterface
from trading.market.market_analyzer import MarketAnalyzer
from trading.memory.agent_memory import AgentMemory
from trading.utils.reasoning_logger import (
    ConfidenceLevel,
    DecisionType,
    ReasoningLogger,
)

logger = logging.getLogger(__name__)


class CommentaryType(Enum):
    """Types of commentary."""

    TRADE_EXPLANATION = "trade_explanation"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    MARKET_REGIME = "market_regime"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_RECOMMENDATION = "strategy_recommendation"
    COUNTERFACTUAL_ANALYSIS = "counterfactual_analysis"
    DAILY_SUMMARY = "daily_summary"
    PORTFOLIO_OVERVIEW = "portfolio_overview"


@dataclass
class CommentaryRequest:
    """Request for commentary generation."""

    commentary_type: CommentaryType
    symbol: str
    timestamp: datetime
    trade_data: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None
    market_data: Optional[pd.DataFrame] = None
    portfolio_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class CommentaryResponse:
    """Response from commentary engine."""

    request_id: str
    commentary_type: CommentaryType
    timestamp: datetime
    title: str
    summary: str
    detailed_analysis: str
    key_insights: List[str]
    recommendations: List[str]
    risk_warnings: List[str]
    confidence_score: float
    metadata: Dict[str, Any]


class CommentaryEngine:
    """
    Unified commentary engine for generating comprehensive trading insights.

    This engine provides LLM-based explanations for trading decisions,
    performance analysis, and market insights with advanced features
    including risk assessment, counterfactual analysis, and strategy recommendations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the commentary engine."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.llm_interface = LLMInterface()
        self.market_analyzer = MarketAnalyzer()
        self.memory = AgentMemory()
        self.reasoning_logger = ReasoningLogger()

        # Configuration
        self.max_context_length = self.config.get("max_context_length", 4000)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_counterfactual = self.config.get("enable_counterfactual", True)
        self.enable_risk_assessment = self.config.get("enable_risk_assessment", True)

        # Storage
        self.commentary_history: List[CommentaryResponse] = []
        self.performance_cache: Dict[str, Dict[str, Any]] = {}

        # Load templates
        self._load_commentary_templates()

        self.logger.info("Commentary Engine initialized successfully")

    def _load_commentary_templates(self):
        """Load commentary templates."""
        self.templates = {
            CommentaryType.TRADE_EXPLANATION: {
                "title": "Trade Explanation: {symbol}",
                "system_prompt": "You are a quantitative trading analyst explaining trade decisions.",
                "user_template": self._get_trade_explanation_template(),
            },
            CommentaryType.PERFORMANCE_ANALYSIS: {
                "title": "Performance Analysis: {symbol}",
                "system_prompt": "You are a performance analyst providing insights on trading results.",
                "user_template": self._get_performance_analysis_template(),
            },
            CommentaryType.MARKET_REGIME: {
                "title": "Market Regime Analysis: {symbol}",
                "system_prompt": "You are a market analyst explaining current market conditions.",
                "user_template": self._get_market_regime_template(),
            },
            CommentaryType.RISK_ASSESSMENT: {
                "title": "Risk Assessment: {symbol}",
                "system_prompt": "You are a risk analyst assessing trading risks.",
                "user_template": self._get_risk_assessment_template(),
            },
        }

    async def generate_commentary(self, request: CommentaryRequest) -> CommentaryResponse:
        """
        Generate comprehensive commentary based on request type.

        Args:
            request: Commentary request with data and context

        Returns:
            Commentary response with analysis and insights
        """
        try:
            self.logger.info(f"Generating {request.commentary_type.value} commentary for {request.symbol}")

            # Generate request ID
            request_id = f"commentary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.symbol}"

            # Route to appropriate handler
            if request.commentary_type == CommentaryType.TRADE_EXPLANATION:
                response = await self._generate_trade_explanation(request, request_id)
            elif request.commentary_type == CommentaryType.PERFORMANCE_ANALYSIS:
                response = await self._generate_performance_analysis(request, request_id)
            elif request.commentary_type == CommentaryType.MARKET_REGIME:
                response = await self._generate_market_regime_commentary(request, request_id)
            elif request.commentary_type == CommentaryType.RISK_ASSESSMENT:
                response = await self._generate_risk_assessment(request, request_id)
            elif request.commentary_type == CommentaryType.STRATEGY_RECOMMENDATION:
                response = await self._generate_strategy_recommendation(request, request_id)
            elif request.commentary_type == CommentaryType.COUNTERFACTUAL_ANALYSIS:
                response = await self._generate_counterfactual_analysis(request, request_id)
            elif request.commentary_type == CommentaryType.DAILY_SUMMARY:
                response = await self._generate_daily_summary(request, request_id)
            elif request.commentary_type == CommentaryType.PORTFOLIO_OVERVIEW:
                response = await self._generate_portfolio_overview(request, request_id)
            else:
                response = await self._generate_general_commentary(request, request_id)

            # Store commentary
            self.commentary_history.append(response)

            # Log decision
            self._log_commentary_decision(response)

            return response

        except Exception as e:
            self.logger.error(f"Error generating commentary: {str(e)}")
            return self._create_error_response(request, str(e))

    async def _generate_trade_explanation(self, request: CommentaryRequest, request_id: str) -> CommentaryResponse:
        """Generate trade explanation commentary."""
        try:
            trade_data = request.trade_data
            if not trade_data:
                raise ValueError("Trade data required for trade explanation")

            # Analyze trade context
            trade_context = self._analyze_trade_context(trade_data, request.market_data)

            # Create prompt
            prompt = self._create_trade_explanation_prompt(trade_data, trade_context)

            # Get LLM response
            llm_response = await self.llm_interface.generate_response(prompt)

            # Parse and structure response
            analysis = self._parse_llm_response(llm_response)

            # Extract insights and recommendations
            key_insights = self._extract_trade_insights(trade_data, trade_context)
            recommendations = self._generate_trade_recommendations(trade_data, trade_context)
            risk_warnings = self._extract_trade_risks(trade_data, trade_context)

            # Calculate confidence
            confidence = self._calculate_trade_confidence(trade_data, trade_context)

            return CommentaryResponse(
                request_id=request_id,
                commentary_type=CommentaryType.TRADE_EXPLANATION,
                timestamp=datetime.now(),
                title=f"Trade Explanation: {request.symbol}",
                summary=analysis.get("summary", "Trade explanation generated"),
                detailed_analysis=analysis.get("analysis", ""),
                key_insights=key_insights,
                recommendations=recommendations,
                risk_warnings=risk_warnings,
                confidence_score=confidence,
                metadata={"trade_data": trade_data, "context": trade_context},
            )

        except Exception as e:
            self.logger.error(f"Error generating trade explanation: {str(e)}")
            raise

    async def _generate_performance_analysis(self, request: CommentaryRequest, request_id: str) -> CommentaryResponse:
        """Generate performance analysis commentary."""
        try:
            performance_data = request.performance_data
            if not performance_data:
                raise ValueError("Performance data required for performance analysis")

            # Analyze performance metrics
            performance_analysis = self._analyze_performance_metrics(performance_data)

            # Create prompt
            prompt = self._create_performance_analysis_prompt(performance_data, performance_analysis)

            # Get LLM response
            llm_response = await self.llm_interface.generate_response(prompt)

            # Parse response
            analysis = self._parse_llm_response(llm_response)

            # Extract insights and recommendations
            key_insights = self._extract_performance_insights(performance_data, performance_analysis)
            recommendations = self._generate_performance_recommendations(performance_data, performance_analysis)
            risk_warnings = self._extract_performance_risks(performance_data)

            # Calculate confidence
            confidence = self._calculate_performance_confidence(performance_data)

            return CommentaryResponse(
                request_id=request_id,
                commentary_type=CommentaryType.PERFORMANCE_ANALYSIS,
                timestamp=datetime.now(),
                title=f"Performance Analysis: {request.symbol}",
                summary=analysis.get("summary", "Performance analysis completed"),
                detailed_analysis=analysis.get("analysis", ""),
                key_insights=key_insights,
                recommendations=recommendations,
                risk_warnings=risk_warnings,
                confidence_score=confidence,
                metadata={"performance_data": performance_data, "analysis": performance_analysis},
            )

        except Exception as e:
            self.logger.error(f"Error generating performance analysis: {str(e)}")
            raise

    def _analyze_trade_context(self, trade_data: Dict[str, Any], market_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Analyze trade context and market conditions."""
        context = {
            "trade_timing": trade_data.get("timestamp", datetime.now()),
            "signal_strength": trade_data.get("signal_strength", 0),
            "model_confidence": trade_data.get("model_confidence", 0),
            "position_size": trade_data.get("quantity", 0),
            "entry_price": trade_data.get("price", 0),
        }

        if market_data is not None and not market_data.empty:
            # Calculate market metrics
            returns = market_data["close"].pct_change().dropna()
            context.update(
                {
                    "volatility": returns.std() * np.sqrt(252),
                    "trend": "up" if market_data["close"].iloc[-1] > market_data["close"].iloc[-20] else "down",
                    "volume_trend": market_data["volume"].iloc[-5:].mean() / market_data["volume"].iloc[-20:].mean()
                    if "volume" in market_data.columns
                    else 1.0,
                }
            )

        return context

    def _create_trade_explanation_prompt(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create prompt for trade explanation."""
        return f"""
        Analyze the following trade and provide a comprehensive explanation:

        Trade Details:
        - Symbol: {trade_data.get('symbol', 'Unknown')}
        - Side: {trade_data.get('side', 'Unknown')}
        - Quantity: {trade_data.get('quantity', 0)}
        - Price: {trade_data.get('price', 0)}
        - Signal Strength: {trade_data.get('signal_strength', 0)}
        - Model Confidence: {trade_data.get('model_confidence', 0)}

        Market Context:
        - Volatility: {context.get('volatility', 0):.4f}
        - Trend: {context.get('trend', 'Unknown')}
        - Volume Trend: {context.get('volume_trend', 0):.2f}

        Please provide:
        1. Clear rationale for the trade
        2. Key factors influencing the decision
        3. Risk considerations
        4. Expected outcome
        5. Potential concerns or warnings

        Format as a structured analysis with clear sections.
        """

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response into structured format."""
        try:
            # Simple parsing - in practice, you might use more sophisticated parsing
            lines = response.split("\n")
            summary = ""
            analysis = ""

            for line in lines:
                if line.strip():
                    if not summary and len(line) < 200:
                        summary = line.strip()
                    else:
                        analysis += line + "\n"

            return {"summary": summary or "Analysis completed", "analysis": analysis.strip() or response}
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return {"summary": "Analysis completed", "analysis": response}

    def _extract_trade_insights(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract key insights from trade data."""
        insights = []

        # Signal strength insights
        signal_strength = trade_data.get("signal_strength", 0)
        if signal_strength > 0.8:
            insights.append("Strong signal indicates high conviction trade")
        elif signal_strength < 0.5:
            insights.append("Weak signal suggests cautious approach")

        # Model confidence insights
        model_confidence = trade_data.get("model_confidence", 0)
        if model_confidence > 0.9:
            insights.append("High model confidence supports trade decision")
        elif model_confidence < 0.7:
            insights.append("Low model confidence requires careful monitoring")

        # Market context insights
        volatility = context.get("volatility", 0)
        if volatility > 0.3:
            insights.append("High volatility environment - increased risk")
        elif volatility < 0.1:
            insights.append("Low volatility environment - reduced risk")

        return insights

    def _generate_trade_recommendations(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trade analysis."""
        recommendations = []

        # Risk management recommendations
        recommendations.append("Set appropriate stop-loss levels based on volatility")
        recommendations.append("Monitor position size relative to portfolio")

        # Market-specific recommendations
        volatility = context.get("volatility", 0)
        if volatility > 0.3:
            recommendations.append("Consider reducing position size in high volatility")
        elif context.get("trend") == "up":
            recommendations.append("Look for additional entry opportunities in uptrend")

        return recommendations

    def _extract_trade_risks(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract risk warnings from trade data."""
        warnings = []

        # Position size warnings
        quantity = trade_data.get("quantity", 0)
        if quantity > 1000:
            warnings.append("Large position size - ensure adequate risk management")

        # Signal strength warnings
        signal_strength = trade_data.get("signal_strength", 0)
        if signal_strength < 0.5:
            warnings.append("Low signal strength - consider waiting for stronger signals")

        # Volatility warnings
        volatility = context.get("volatility", 0)
        if volatility > 0.4:
            warnings.append("Extreme volatility - consider reducing exposure")

        return warnings

    def _calculate_trade_confidence(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence score for trade explanation."""
        confidence = 0.5  # Base confidence

        # Signal strength contribution
        signal_strength = trade_data.get("signal_strength", 0)
        confidence += signal_strength * 0.3

        # Model confidence contribution
        model_confidence = trade_data.get("model_confidence", 0)
        confidence += model_confidence * 0.2

        # Market regime contribution
        volatility = context.get("volatility", 0)
        if volatility < 0.2:
            confidence += 0.1
        elif volatility > 0.4:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _log_commentary_decision(self, response: CommentaryResponse):
        """Log commentary decision for analysis."""
        self.reasoning_logger.log_decision(
            agent_name="CommentaryEngine",
            decision_type=DecisionType.COMMENTARY_GENERATION,
            action_taken=f"Generated {response.commentary_type.value} commentary",
            context={
                "commentary_type": response.commentary_type.value,
                "symbol": response.metadata.get("symbol", "Unknown"),
                "confidence": response.confidence_score,
                "insights_count": len(response.key_insights),
                "recommendations_count": len(response.recommendations),
            },
            reasoning={
                "primary_reason": f"Generated {response.commentary_type.value} commentary",
                "supporting_factors": response.key_insights,
                "recommendations": response.recommendations,
                "risk_warnings": response.risk_warnings,
            },
            confidence_level=ConfidenceLevel.HIGH if response.confidence_score > 0.8 else ConfidenceLevel.MEDIUM,
            metadata=response.metadata,
        )

    def _create_error_response(self, request: CommentaryRequest, error_message: str) -> CommentaryResponse:
        """Create error response when commentary generation fails."""
        return CommentaryResponse(
            request_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            commentary_type=request.commentary_type,
            timestamp=datetime.now(),
            title=f"Error: {request.commentary_type.value}",
            summary="Commentary generation failed",
            detailed_analysis=f"Error: {error_message}",
            key_insights=[],
            recommendations=["Review system logs for details"],
            risk_warnings=["System error detected"],
            confidence_score=0.0,
            metadata={"error": error_message, "request": request.__dict__},
        )

    def get_commentary_statistics(self) -> Dict[str, Any]:
        """Get commentary generation statistics."""
        if not self.commentary_history:
            return {"total_commentaries": 0}

        # Count by type
        type_counts = {}
        for response in self.commentary_history:
            type_name = response.commentary_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Calculate average confidence
        avg_confidence = np.mean([r.confidence_score for r in self.commentary_history])

        return {
            "total_commentaries": len(self.commentary_history),
            "commentary_types": type_counts,
            "average_confidence": avg_confidence,
            "recent_commentaries": [
                {
                    "type": r.commentary_type.value,
                    "title": r.title,
                    "timestamp": r.timestamp.isoformat(),
                    "confidence": r.confidence_score,
                }
                for r in self.commentary_history[-5:]  # Last 5 commentaries
            ],
        }


# Convenience function for creating commentary engine


def create_commentary_engine(config: Optional[Dict[str, Any]] = None) -> CommentaryEngine:
    """Create a configured commentary engine."""
    return CommentaryEngine(config)
