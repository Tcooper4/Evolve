# -*- coding: utf-8 -*-
"""
Enhanced QuantGPT Commentary Agent with advanced analysis and explainability.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.llm.llm_interface import LLMInterface
from trading.market.market_analyzer import MarketAnalyzer
from trading.memory.agent_memory import AgentMemory
from trading.utils.performance_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)


class CommentaryType(str, Enum):
    """Types of commentary."""
    TRADE_EXPLANATION = "trade_explanation"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    OVERFITTING_DETECTION = "overfitting_detection"
    REGIME_ANALYSIS = "regime_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COUNTERFACTUAL_ANALYSIS = "counterfactual_analysis"
    STRATEGY_RECOMMENDATION = "strategy_recommendation"


@dataclass
class CommentaryRequest:
    """Commentary request data."""
    request_type: CommentaryType
    symbol: str
    timestamp: datetime
    trade_data: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None
    market_data: Optional[pd.DataFrame] = None
    model_data: Optional[Dict[str, Any]] = None
    strategy_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class CommentaryResponse:
    """Commentary response data."""
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


class QuantGPTCommentaryAgent:
    """
    Enhanced QuantGPT Commentary Agent with:
    - Trade explanation and justification
    - Overfitting detection and warnings
    - Regime-aware analysis
    - Counterfactual analysis
    - Risk assessment and warnings
    - Performance attribution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the QuantGPT Commentary Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.llm_interface = LLMInterface()
        self.market_analyzer = MarketAnalyzer()
        self.memory = AgentMemory()

        # Configuration
        self.max_context_length = self.config.get('max_context_length', 4000)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.overfitting_threshold = self.config.get('overfitting_threshold', 0.8)
        self.regime_detection_window = self.config.get('regime_detection_window', 60)

        # Storage
        self.commentary_history: List[CommentaryResponse] = []
        self.overfitting_alerts: List[Dict[str, Any]] = []
        self.regime_analysis_cache: Dict[str, Dict[str, Any]] = {}

        # Load templates
        self._load_commentary_templates()

        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}

    async def generate_commentary(self, request: CommentaryRequest) -> CommentaryResponse:
        """
        Generate comprehensive commentary based on request type.

        Args:
            request: Commentary request with data and context

        Returns:
            Commentary response with analysis and insights
        """
        try:
            self.logger.info(f"Generating {request.request_type.value} commentary for {request.symbol}")

            # Generate request ID
            request_id = f"commentary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.symbol}"

            if request.request_type == CommentaryType.TRADE_EXPLANATION:
                response = await self._explain_trade(request, request_id)
            elif request.request_type == CommentaryType.PERFORMANCE_ANALYSIS:
                response = await self._analyze_performance(request, request_id)
            elif request.request_type == CommentaryType.OVERFITTING_DETECTION:
                response = await self._detect_overfitting(request, request_id)
            elif request.request_type == CommentaryType.REGIME_ANALYSIS:
                response = await self._analyze_market_regime(request, request_id)
            elif request.request_type == CommentaryType.RISK_ASSESSMENT:
                response = await self._assess_risk(request, request_id)
            elif request.request_type == CommentaryType.COUNTERFACTUAL_ANALYSIS:
                response = await self._perform_counterfactual_analysis(request, request_id)
            elif request.request_type == CommentaryType.STRATEGY_RECOMMENDATION:
                response = await self._recommend_strategy(request, request_id)
            else:
                response = await self._generate_general_commentary(request, request_id)

            # Store commentary
            self.commentary_history.append(response)

            # Store in memory
            self._store_commentary(response)

            return response

        except Exception as e:
            self.logger.error(f"Error generating commentary: {str(e)}")
            return self._create_error_response(request, str(e))

    async def _explain_trade(self, request: CommentaryRequest, request_id: str) -> CommentaryResponse:
        """Explain why a trade was made."""
        try:
            trade_data = request.trade_data
            if not trade_data:
                raise ValueError("Trade data required for trade explanation")

            # Analyze trade context
            trade_context = self._analyze_trade_context(trade_data, request.market_data)

            # Generate explanation prompt
            prompt = self._create_trade_explanation_prompt(trade_data, trade_context)

            # Get LLM response
            llm_response = await self.llm_interface.generate_response(prompt)

            # Parse response
            analysis = self._parse_llm_response(llm_response)

            # Extract key insights
            key_insights = self._extract_trade_insights(trade_data, trade_context)

            # Generate recommendations
            recommendations = self._generate_trade_recommendations(trade_data, trade_context)

            # Calculate confidence
            confidence = self._calculate_trade_confidence(trade_data, trade_context)

            return CommentaryResponse(
                request_id=request_id,
                commentary_type=CommentaryType.TRADE_EXPLANATION,
                timestamp=datetime.now(),
                title=f"Trade Explanation: {request.symbol}",
                summary=analysis.get('summary', 'Trade explanation generated'),
                detailed_analysis=analysis.get('analysis', ''),
                key_insights=key_insights,
                recommendations=recommendations,
                risk_warnings=self._extract_risk_warnings(trade_data),
                confidence_score=confidence,
                metadata={'trade_data': trade_data, 'context': trade_context}
            )

        except Exception as e:
            self.logger.error(f"Error explaining trade: {str(e)}")
            raise

    def _analyze_trade_context(self, trade_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade context and market conditions."""
        try:
            context = {}

            if market_data is not None and not market_data.empty:
                # Market regime
                context['market_regime'] = self._detect_market_regime(market_data)

                # Volatility
                returns = market_data['close'].pct_change().dropna()
                context['volatility'] = returns.std()

                # Trend
                sma_short = market_data['close'].rolling(window=10).mean()
                sma_long = market_data['close'].rolling(window=50).mean()
                context['trend'] = 'up' if sma_short.iloc[-1] > sma_long.iloc[-1] else 'down'

                # Volume analysis
                if 'volume' in market_data.columns:
                    context['volume_trend'] = market_data['volume'].tail(10).mean() / market_data['volume'].tail(50).mean()

                # Support/resistance levels
                context['support_resistance'] = self._calculate_support_resistance(market_data)

            # Trade-specific context
            context['trade_size'] = trade_data.get('quantity', 0)
            context['trade_value'] = trade_data.get('value', 0)
            context['signal_strength'] = trade_data.get('signal_strength', 0)
            context['model_confidence'] = trade_data.get('model_confidence', 0)

            return context

        except Exception as e:
            self.logger.error(f"Error analyzing trade context: {str(e)}")
            return {}

    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime."""
        try:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]

            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=50).mean()
            trend = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]

            if volatility > 0.03:
                return 'high_volatility'
            elif volatility < 0.01:
                return 'low_volatility'
            elif trend > 0.02:
                return 'trending_up'
            elif trend < -0.02:
                return 'trending_down'
            else:
                return 'sideways'

        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'

    def _calculate_support_resistance(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        try:
            high = market_data['high'].max()
            low = market_data['low'].min()
            current = market_data['close'].iloc[-1]

            return {
                'resistance': high,
                'support': low,
                'current_price': current,
                'distance_to_resistance': (high - current) / current,
                'distance_to_support': (current - low) / current
            }

        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {str(e)}")
            return {}

    def _create_trade_explanation_prompt(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create prompt for trade explanation."""
        try:
            prompt = f"""
            Analyze the following trade and provide a comprehensive explanation:

            Trade Details:
            - Symbol: {trade_data.get('symbol', 'Unknown')}
            - Side: {trade_data.get('side', 'Unknown')}
            - Quantity: {trade_data.get('quantity', 0)}
            - Price: {trade_data.get('price', 0)}
            - Signal Strength: {trade_data.get('signal_strength', 0)}
            - Model Confidence: {trade_data.get('model_confidence', 0)}

            Market Context:
            - Market Regime: {context.get('market_regime', 'Unknown')}
            - Volatility: {context.get('volatility', 0):.4f}
            - Trend: {context.get('trend', 'Unknown')}
            - Volume Trend: {context.get('volume_trend', 0):.2f}

            Please provide:
            1. Why this trade was executed
            2. Key factors influencing the decision
            3. Risk considerations
            4. Expected outcome
            5. Potential concerns or warnings
            """

            return prompt

        except Exception as e:
            self.logger.error(f"Error creating trade explanation prompt: {str(e)}")
            return "Please explain this trade."

    async def _analyze_performance(self, request: CommentaryRequest, request_id: str) -> CommentaryResponse:
        """Analyze trading performance."""
        try:
            performance_data = request.performance_data
            if not performance_data:
                raise ValueError("Performance data required for performance analysis")

            # Analyze performance metrics
            performance_analysis = self._analyze_performance_metrics(performance_data)

            # Generate performance prompt
            prompt = self._create_performance_analysis_prompt(performance_data, performance_analysis)

            # Get LLM response
            llm_response = await self.llm_interface.generate_response(prompt)

            # Parse response
            analysis = self._parse_llm_response(llm_response)

            # Extract insights
            key_insights = self._extract_performance_insights(performance_data, performance_analysis)

            # Generate recommendations
            recommendations = self._generate_performance_recommendations(performance_data, performance_analysis)

            # Calculate confidence
            confidence = self._calculate_performance_confidence(performance_data)

            return CommentaryResponse(
                request_id=request_id,
                commentary_type=CommentaryType.PERFORMANCE_ANALYSIS,
                timestamp=datetime.now(),
                title=f"Performance Analysis: {request.symbol}",
                summary=analysis.get('summary', 'Performance analysis completed'),
                detailed_analysis=analysis.get('analysis', ''),
                key_insights=key_insights,
                recommendations=recommendations,
                risk_warnings=self._extract_performance_risks(performance_data),
                confidence_score=confidence,
                metadata={'performance_data': performance_data, 'analysis': performance_analysis}
            )

        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            raise

    def _analyze_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics and identify patterns."""
        try:
            analysis = {}

            # Calculate key metrics
            returns = performance_data.get('returns', [])
            if returns:
                analysis['total_return'] = sum(returns)
                analysis['avg_return'] = np.mean(returns)
                analysis['volatility'] = np.std(returns)
                analysis['sharpe_ratio'] = calculate_sharpe_ratio(pd.Series(returns))
                analysis['max_drawdown'] = calculate_max_drawdown(pd.Series(returns))
                analysis['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)

            # Identify patterns
            analysis['performance_trend'] = self._identify_performance_trend(returns)
            analysis['risk_adjusted_performance'] = self._assess_risk_adjusted_performance(analysis)
            analysis['consistency'] = self._assess_performance_consistency(returns)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing performance metrics: {str(e)}")
            return {}

    def _identify_performance_trend(self, returns: List[float]) -> str:
        """Identify performance trend."""
        try:
            if len(returns) < 10:
                return 'insufficient_data'

            # Split returns into recent and older periods
            mid_point = len(returns) // 2
            recent_returns = returns[mid_point:]
            older_returns = returns[:mid_point]

            recent_avg = np.mean(recent_returns)
            older_avg = np.mean(older_returns)

            if recent_avg > older_avg * 1.1:
                return 'improving'
            elif recent_avg < older_avg * 0.9:
                return 'declining'
            else:
                return 'stable'

        except Exception as e:
            self.logger.error(f"Error identifying performance trend: {str(e)}")
            return 'unknown'

    def _assess_risk_adjusted_performance(self, analysis: Dict[str, Any]) -> str:
        """Assess risk-adjusted performance."""
        try:
            sharpe = analysis.get('sharpe_ratio', 0)
            max_dd = analysis.get('max_drawdown', 0)

            if sharpe > 1.0 and max_dd < 0.1:
                return 'excellent'
            elif sharpe > 0.5 and max_dd < 0.2:
                return 'good'
            elif sharpe > 0.0 and max_dd < 0.3:
                return 'fair'
            else:
                return 'poor'

        except Exception as e:
            self.logger.error(f"Error assessing risk-adjusted performance: {str(e)}")
            return 'unknown'

    def _assess_performance_consistency(self, returns: List[float]) -> str:
        """Assess performance consistency."""
        try:
            if len(returns) < 10:
                return 'insufficient_data'

            # Calculate coefficient of variation
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if mean_return == 0:
                return 'inconsistent'

            cv = std_return / abs(mean_return)

            if cv < 0.5:
                return 'very_consistent'
            elif cv < 1.0:
                return 'consistent'
            elif cv < 2.0:
                return 'moderate'
            else:
                return 'inconsistent'

        except Exception as e:
            self.logger.error(f"Error assessing performance consistency: {str(e)}")
            return 'unknown'

    async def _detect_overfitting(self, request: CommentaryRequest, request_id: str) -> CommentaryResponse:
        """Detect overfitting in models and strategies."""
        try:
            model_data = request.model_data
            if not model_data:
                raise ValueError("Model data required for overfitting detection")

            # Analyze model performance patterns
            overfitting_analysis = self._analyze_overfitting_patterns(model_data)

            # Generate overfitting prompt
            prompt = self._create_overfitting_detection_prompt(model_data, overfitting_analysis)

            # Get LLM response
            llm_response = await self.llm_interface.generate_response(prompt)

            # Parse response
            analysis = self._parse_llm_response(llm_response)

            # Extract overfitting indicators
            overfitting_indicators = self._extract_overfitting_indicators(model_data, overfitting_analysis)

            # Generate warnings
            warnings = self._generate_overfitting_warnings(overfitting_analysis)

            # Calculate overfitting score
            overfitting_score = self._calculate_overfitting_score(overfitting_analysis)

            # Store overfitting alert if significant
            if overfitting_score > self.overfitting_threshold:
                self._store_overfitting_alert(request.symbol, overfitting_analysis, overfitting_score)

            return CommentaryResponse(
                request_id=request_id,
                commentary_type=CommentaryType.OVERFITTING_DETECTION,
                timestamp=datetime.now(),
                title=f"Overfitting Detection: {request.symbol}",
                summary=analysis.get('summary', 'Overfitting analysis completed'),
                detailed_analysis=analysis.get('analysis', ''),
                key_insights=overfitting_indicators,
                recommendations=self._generate_overfitting_recommendations(overfitting_analysis),
                risk_warnings=warnings,
                confidence_score=1.0 - overfitting_score,  # Higher overfitting = lower confidence
                metadata={'model_data': model_data, 'overfitting_analysis': overfitting_analysis}
            )

        except Exception as e:
            self.logger.error(f"Error detecting overfitting: {str(e)}")
            raise

    def _analyze_overfitting_patterns(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns that indicate overfitting."""
        try:
            analysis = {}

            # Training vs validation performance
            train_performance = model_data.get('training_performance', {})
            val_performance = model_data.get('validation_performance', {})

            if train_performance and val_performance:
                train_metric = train_performance.get('metric', 0)
                val_metric = val_performance.get('metric', 0)

                # Performance gap
                performance_gap = train_metric - val_metric
                analysis['performance_gap'] = performance_gap
                analysis['performance_gap_ratio'] = performance_gap / max(val_metric, 1e-6)

            # Parameter complexity
            num_parameters = model_data.get('num_parameters', 0)
            num_samples = model_data.get('num_samples', 0)

            if num_parameters > 0 and num_samples > 0:
                analysis['parameter_ratio'] = num_parameters / num_samples
                analysis['complexity_risk'] = 'high' if num_parameters / num_samples > 0.1 else 'low'

            # Cross-validation results
            cv_results = model_data.get('cross_validation_results', [])
            if cv_results:
                cv_mean = np.mean(cv_results)
                cv_std = np.std(cv_results)
                analysis['cv_stability'] = cv_std / max(cv_mean, 1e-6)

            # Out-of-sample performance
            oos_performance = model_data.get('out_of_sample_performance', {})
            if oos_performance:
                analysis['oos_degradation'] = self._calculate_oos_degradation(
                    train_performance, oos_performance
                )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing overfitting patterns: {str(e)}")
            return {}

    def _calculate_oos_degradation(self, train_perf: Dict[str, Any], oos_perf: Dict[str, Any]) -> float:
        """Calculate out-of-sample performance degradation."""
        try:
            train_metric = train_perf.get('metric', 0)
            oos_metric = oos_perf.get('metric', 0)

            if train_metric == 0:
                return 0.0

            degradation = (train_metric - oos_metric) / train_metric
            return max(0.0, degradation)

        except Exception as e:
            self.logger.error(f"Error calculating OOS degradation: {str(e)}")
            return 0.0

    async def _analyze_market_regime(self, request: CommentaryRequest, request_id: str) -> CommentaryResponse:
        """Analyze current market regime and implications."""
        try:
            market_data = request.market_data
            if market_data is None or market_data.empty:
                raise ValueError("Market data required for regime analysis")

            # Analyze market regime
            regime_analysis = self._perform_regime_analysis(market_data)

            # Generate regime prompt
            prompt = self._create_regime_analysis_prompt(regime_analysis)

            # Get LLM response
            llm_response = await self.llm_interface.generate_response(prompt)

            # Parse response
            analysis = self._parse_llm_response(llm_response)

            # Extract regime insights
            key_insights = self._extract_regime_insights(regime_analysis)

            # Generate regime-specific recommendations
            recommendations = self._generate_regime_recommendations(regime_analysis)

            # Calculate confidence
            confidence = self._calculate_regime_confidence(regime_analysis)

            # Cache regime analysis
            self.regime_analysis_cache[request.symbol] = regime_analysis

            return CommentaryResponse(
                request_id=request_id,
                commentary_type=CommentaryType.REGIME_ANALYSIS,
                timestamp=datetime.now(),
                title=f"Market Regime Analysis: {request.symbol}",
                summary=analysis.get('summary', 'Regime analysis completed'),
                detailed_analysis=analysis.get('analysis', ''),
                key_insights=key_insights,
                recommendations=recommendations,
                risk_warnings=self._extract_regime_risks(regime_analysis),
                confidence_score=confidence,
                metadata={'regime_analysis': regime_analysis}
            )

        except Exception as e:
            self.logger.error(f"Error analyzing market regime: {str(e)}")
            raise

    def _perform_regime_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive market regime analysis."""
        try:
            analysis = {}

            # Basic regime detection
            analysis['regime'] = self._detect_market_regime(market_data)

            # Volatility analysis
            returns = market_data['close'].pct_change().dropna()
            analysis['volatility'] = {
                'current': returns.tail(20).std(),
                'historical': returns.std(),
                'trend': 'increasing' if returns.tail(10).std() > returns.tail(50).std() else 'decreasing'
            }

            # Trend analysis
            sma_short = market_data['close'].rolling(window=10).mean()
            sma_long = market_data['close'].rolling(window=50).mean()
            analysis['trend'] = {
                'direction': 'up' if sma_short.iloc[-1] > sma_long.iloc[-1] else 'down',
                'strength': abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            }

            # Volume analysis
            if 'volume' in market_data.columns:
                analysis['volume'] = {
                    'trend': market_data['volume'].tail(10).mean() / market_data['volume'].tail(50).mean(),
                    'volatility': market_data['volume'].pct_change().std()
                }

            # Regime stability
            analysis['stability'] = self._assess_regime_stability(market_data)

            # Regime implications
            analysis['implications'] = self._analyze_regime_implications(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error performing regime analysis: {str(e)}")
            return {}

    def _assess_regime_stability(self, market_data: pd.DataFrame) -> str:
        """Assess the stability of the current market regime."""
        try:
            # Calculate regime consistency over time
            returns = market_data['close'].pct_change().dropna()

            # Split data into periods and check regime consistency
            periods = 4
            period_length = len(returns) // periods

            regimes = []
            for i in range(periods):
                start_idx = i * period_length
                end_idx = start_idx + period_length
                period_returns = returns.iloc[start_idx:end_idx]

                volatility = period_returns.std()
                if volatility > 0.03:
                    regimes.append('high_volatility')
                elif volatility < 0.01:
                    regimes.append('low_volatility')
                else:
                    regimes.append('moderate_volatility')

            # Check consistency
            unique_regimes = len(set(regimes))

            if unique_regimes == 1:
                return 'very_stable'
            elif unique_regimes == 2:
                return 'stable'
            elif unique_regimes == 3:
                return 'moderate'
            else:
                return 'unstable'

        except Exception as e:
            self.logger.error(f"Error assessing regime stability: {str(e)}")
            return 'unknown'

    def _analyze_regime_implications(self, regime_analysis: Dict[str, Any]) -> List[str]:
        """Analyze implications of the current market regime."""
        try:
            implications = []
            regime = regime_analysis.get('regime', 'unknown')
            volatility = regime_analysis.get('volatility', {})
            trend = regime_analysis.get('trend', {})

            if regime == 'high_volatility':
                implications.append("High volatility suggests increased risk and potential for large price swings")
                implications.append("Consider reducing position sizes and implementing tighter stop-losses")
                implications.append("Volatility-based strategies may perform better in this environment")

            elif regime == 'low_volatility':
                implications.append("Low volatility suggests reduced risk but also limited profit potential")
                implications.append("Trend-following strategies may struggle in low volatility environments")
                implications.append("Consider mean-reversion strategies or options strategies")

            elif regime == 'trending_up':
                implications.append("Upward trend suggests momentum strategies may be effective")
                implications.append("Consider trend-following indicators and breakout strategies")
                implications.append("Be cautious of potential trend reversals")

            elif regime == 'trending_down':
                implications.append("Downward trend suggests defensive positioning may be appropriate")
                implications.append("Consider short-selling strategies or hedging positions")
                implications.append("Look for oversold conditions and potential reversal signals")

            return implications

        except Exception as e:
            self.logger.error(f"Error analyzing regime implications: {str(e)}")
            return ["Unable to analyze regime implications"]

    async def _perform_counterfactual_analysis(self, request: CommentaryRequest, request_id: str) -> CommentaryResponse:
        """Perform counterfactual analysis of trading decisions."""
        try:
            trade_data = request.trade_data
            if not trade_data:
                raise ValueError("Trade data required for counterfactual analysis")

            # Perform counterfactual analysis
            counterfactual_results = self._analyze_counterfactual_scenarios(trade_data, request.market_data)

            # Generate counterfactual prompt
            prompt = self._create_counterfactual_analysis_prompt(trade_data, counterfactual_results)

            # Get LLM response
            llm_response = await self.llm_interface.generate_response(prompt)

            # Parse response
            analysis = self._parse_llm_response(llm_response)

            # Extract insights
            key_insights = self._extract_counterfactual_insights(counterfactual_results)

            # Generate recommendations
            recommendations = self._generate_counterfactual_recommendations(counterfactual_results)

            return CommentaryResponse(
                request_id=request_id,
                commentary_type=CommentaryType.COUNTERFACTUAL_ANALYSIS,
                timestamp=datetime.now(),
                title=f"Counterfactual Analysis: {request.symbol}",
                summary=analysis.get('summary', 'Counterfactual analysis completed'),
                detailed_analysis=analysis.get('analysis', ''),
                key_insights=key_insights,
                recommendations=recommendations,
                risk_warnings=self._extract_counterfactual_risks(counterfactual_results),
                confidence_score=0.8,  # Counterfactual analysis is inherently uncertain
                metadata={'trade_data': trade_data, 'counterfactual_results': counterfactual_results}
            )

        except Exception as e:
            self.logger.error(f"Error performing counterfactual analysis: {str(e)}")
            raise

    def _analyze_counterfactual_scenarios(self, trade_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different scenarios that could have occurred."""
        try:
            scenarios = {}

            # Scenario 1: Different entry timing
            scenarios['timing_analysis'] = self._analyze_timing_scenarios(trade_data, market_data)

            # Scenario 2: Different position sizing
            scenarios['sizing_analysis'] = self._analyze_sizing_scenarios(trade_data)

            # Scenario 3: Different exit strategies
            scenarios['exit_analysis'] = self._analyze_exit_scenarios(trade_data, market_data)

            # Scenario 4: Alternative strategies
            scenarios['strategy_alternatives'] = self._analyze_strategy_alternatives(trade_data, market_data)

            return scenarios

        except Exception as e:
            self.logger.error(f"Error analyzing counterfactual scenarios: {str(e)}")
            return {}

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response into structured format."""
        try:
            # Simple parsing - in practice, you might use more sophisticated parsing
            return {
                'summary': response[:200] + "..." if len(response) > 200 else response,
                'analysis': response
            }
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return {'success': True, 'result': {'summary': 'Analysis completed', 'analysis': response}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def _extract_trade_insights(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Extract key insights from trade data."""
        try:
            insights = []

            # Signal strength insights
            signal_strength = trade_data.get('signal_strength', 0)
            if signal_strength > 0.8:
                insights.append("Strong signal strength indicates high confidence in the trade")
            elif signal_strength < 0.3:
                insights.append("Weak signal strength suggests cautious approach")

            # Market regime insights
            regime = context.get('market_regime', 'unknown')
            insights.append(f"Trade executed in {regime} market regime")

            # Volatility insights
            volatility = context.get('volatility', 0)
            if volatility > 0.03:
                insights.append("High volatility environment - consider position sizing")

            return insights

        except Exception as e:
            self.logger.error(f"Error extracting trade insights: {str(e)}")
            return ["Unable to extract insights"]

    def _generate_trade_recommendations(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trade analysis."""
        try:
            recommendations = []

            # Risk management recommendations
            recommendations.append("Set appropriate stop-loss levels based on volatility")
            recommendations.append("Monitor position size relative to portfolio")

            # Market-specific recommendations
            regime = context.get('market_regime', 'unknown')
            if regime == 'high_volatility':
                recommendations.append("Consider reducing position size in high volatility")
            elif regime == 'trending_up':
                recommendations.append("Look for additional entry opportunities in trend")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating trade recommendations: {str(e)}")
            return ["Review trade parameters carefully"]

    def _extract_risk_warnings(self, trade_data: Dict[str, Any]) -> List[str]:
        """Extract risk warnings from trade data."""
        try:
            warnings = []

            # Position size warnings
            quantity = trade_data.get('quantity', 0)
            if quantity > 1000:
                warnings.append("Large position size - ensure adequate risk management")

            # Signal strength warnings
            signal_strength = trade_data.get('signal_strength', 0)
            if signal_strength < 0.5:
                warnings.append("Low signal strength - consider waiting for stronger signals")

            return warnings

        except Exception as e:
            self.logger.error(f"Error extracting risk warnings: {str(e)}")
            return ["Review trade for potential risks"]

    def _calculate_trade_confidence(self, trade_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence score for trade explanation."""
        try:
            confidence = 0.5  # Base confidence

            # Signal strength contribution
            signal_strength = trade_data.get('signal_strength', 0)
            confidence += signal_strength * 0.3

            # Model confidence contribution
            model_confidence = trade_data.get('model_confidence', 0)
            confidence += model_confidence * 0.2

            # Market regime contribution
            regime = context.get('market_regime', 'unknown')
            if regime in ['trending_up', 'trending_down']:
                confidence += 0.1
            elif regime == 'high_volatility':
                confidence -= 0.1

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating trade confidence: {str(e)}")
            return 0.5

    def _store_commentary(self, response: CommentaryResponse):
        """Store commentary in memory."""
        try:
            self.memory.store('commentary_history', {
                'response': response.__dict__,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error storing commentary: {str(e)}")

    def _store_overfitting_alert(self, symbol: str, analysis: Dict[str, Any], score: float):
        """Store overfitting alert."""
        try:
            alert = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'overfitting_score': score,
                'analysis': analysis
            }
            self.overfitting_alerts.append(alert)
        except Exception as e:
            self.logger.error(f"Error storing overfitting alert: {str(e)}")

    def _load_commentary_templates(self):
        """Load commentary templates."""
        try:
            # In practice, load from file or database
            self.templates = {
                'trade_explanation': "Analyze trade {symbol} with context {context}",
                'performance_analysis': "Analyze performance for {symbol}",
                'overfitting_detection': "Detect overfitting in {symbol}",
                'regime_analysis': "Analyze market regime for {symbol}"
            }
        except Exception as e:
            self.logger.error(f"Error loading commentary templates: {str(e)}")
            self.templates = {}

    def _create_error_response(self, request: CommentaryRequest, error_message: str) -> CommentaryResponse:
        """Create error response when commentary generation fails."""
        return CommentaryResponse(
            request_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            commentary_type=request.request_type,
            timestamp=datetime.now(),
            title=f"Error: {request.request_type.value}",
            summary=f"Failed to generate commentary: {error_message}",
            detailed_analysis="",
            key_insights=[],
            recommendations=[],
            risk_warnings=[],
            confidence_score=0.0,
            metadata={'error': error_message}
        )

    def get_commentary_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of commentary history."""
        try:
            if symbol:
                commentaries = [c for c in self.commentary_history if symbol in c.title]
            else:
                commentaries = self.commentary_history

            if not commentaries:
                return {}

            return {
                'total_commentaries': len(commentaries),
                'commentary_types': list(set([c.commentary_type.value for c in commentaries])),
                'avg_confidence': np.mean([c.confidence_score for c in commentaries]),
                'recent_commentaries': [
                    {
                        'type': c.commentary_type.value,
                        'title': c.title,
                        'timestamp': c.timestamp.isoformat(),
                        'confidence': c.confidence_score
                    }
                    for c in commentaries[-5:]  # Last 5 commentaries
                ]
            }

        except Exception as e:
            self.logger.error(f"Error getting commentary summary: {str(e)}")
            return {}
