"""
Explainer Agent

This module provides comprehensive explainability for trading decisions:
- Generates detailed explanations for forecasts and trades
- Explains model selection and reasoning
- Identifies key features that triggered decisions
- Provides optional LLM-powered explanations
- Creates human-readable summaries of complex decisions
- Supports multiple explanation formats and detail levels
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

# Local imports
from utils.common_helpers import safe_json_save, load_config
from utils.cache_utils import cache_result


class ExplanationType(Enum):
    """Types of explanations"""
    MODEL_SELECTION = "model_selection"
    FEATURE_IMPORTANCE = "feature_importance"
    FORECAST_REASONING = "forecast_reasoning"
    TRADE_DECISION = "trade_decision"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    STRATEGY_RATIONALE = "strategy_rationale"


class ExplanationLevel(Enum):
    """Explanation detail levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    EXPERT = "expert"


@dataclass
class Explanation:
    """Explanation structure"""
    explanation_id: str
    explanation_type: ExplanationType
    timestamp: str
    ticker: Optional[str] = None
    title: str = ""
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    confidence_score: Optional[float] = None
    key_points: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_explanation: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ModelExplanation(Explanation):
    """Model selection and usage explanation"""
    model_name: str = ""
    model_type: str = ""
    model_version: str = ""
    selection_criteria: List[str] = field(default_factory=list)
    model_performance: Dict[str, float] = field(default_factory=dict)
    alternative_models: List[str] = field(default_factory=list)
    model_limitations: List[str] = field(default_factory=list)


@dataclass
class FeatureExplanation(Explanation):
    """Feature importance and contribution explanation"""
    features_analyzed: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    feature_correlations: Dict[str, float] = field(default_factory=dict)
    feature_trends: Dict[str, str] = field(default_factory=dict)
    key_drivers: List[str] = field(default_factory=list)


@dataclass
class ForecastExplanation(Explanation):
    """Forecast reasoning explanation"""
    forecast_value: float = 0.0
    forecast_horizon: int = 0
    forecast_confidence: float = 0.0
    forecast_interval: Optional[tuple] = None
    model_used: str = ""
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    scenario_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeExplanation(Explanation):
    """Trade decision explanation"""
    trade_type: str = ""
    trade_reason: str = ""
    expected_return: float = 0.0
    expected_risk: float = 0.0
    position_size: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: float = 0.0
    market_timing: str = ""
    technical_signals: List[str] = field(default_factory=list)
    fundamental_factors: List[str] = field(default_factory=list)


class ExplainerAgent:
    """
    Comprehensive explainer agent for trading decisions
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.explainer_config = self.config.get('explainer', {})
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create explanations directory
        self.explanations_dir = Path("logs/explanations")
        self.explanations_dir.mkdir(parents=True, exist_ok=True)
        
        # Explanation storage
        self.explanations: List[Explanation] = []
        self.explanation_index: Dict[str, Explanation] = {}
        
        # LLM configuration
        self.llm_enabled = self.explainer_config.get('llm_enabled', False)
        self.llm_model = self.explainer_config.get('llm_model', 'gpt-3.5-turbo')
        self.llm_max_tokens = self.explainer_config.get('llm_max_tokens', 500)
        self.llm_temperature = self.explainer_config.get('llm_temperature', 0.7)
        
        # Explanation templates
        self.templates = self._load_explanation_templates()
        
        # Performance tracking
        self.explanation_metrics = defaultdict(int)
        self.llm_usage_metrics = defaultdict(int)
        
        # Initialize LLM if enabled
        self.llm_client = None
        if self.llm_enabled:
            self._initialize_llm()
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates"""
        templates = {
            'model_selection': """
Model Selection Explanation for {ticker}:

Selected Model: {model_name} ({model_type})
Version: {model_version}

Selection Criteria:
{selection_criteria}

Performance Metrics:
{performance_metrics}

Key Strengths:
{strengths}

Limitations:
{limitations}

Alternative Models Considered:
{alternatives}
""",
            'feature_importance': """
Feature Analysis for {ticker}:

Key Features Analyzed:
{features_analyzed}

Feature Importance Rankings:
{feature_importance}

Key Drivers:
{key_drivers}

Feature Contributions:
{feature_contributions}

Market Context:
{market_context}
""",
            'forecast_reasoning': """
Forecast Explanation for {ticker}:

Forecast Value: {forecast_value}
Horizon: {forecast_horizon} days
Confidence: {forecast_confidence}

Model Used: {model_used}

Key Factors:
{key_factors}

Market Conditions:
{market_conditions}

Risk Factors:
{risk_factors}

Assumptions:
{assumptions}
""",
            'trade_decision': """
Trade Decision for {ticker}:

Decision: {trade_type}
Reason: {trade_reason}

Expected Return: {expected_return}
Expected Risk: {expected_risk}
Risk/Reward Ratio: {risk_reward_ratio}

Position Size: {position_size}
Entry Price: {entry_price}
Stop Loss: {stop_loss}
Take Profit: {take_profit}

Technical Signals:
{technical_signals}

Fundamental Factors:
{fundamental_factors}

Market Timing: {market_timing}
"""
        }
        
        return templates
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        try:
            # Import OpenAI client
            import openai
            api_key = self.explainer_config.get('openai_api_key')
            if api_key:
                self.llm_client = openai.OpenAI(api_key=api_key)
                self.logger.info("LLM client initialized successfully")
            else:
                self.logger.warning("OpenAI API key not found, LLM explanations disabled")
                self.llm_enabled = False
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_enabled = False
    
    def explain_model_selection(self,
                              ticker: str,
                              model_name: str,
                              model_type: str,
                              model_version: str,
                              selection_criteria: List[str],
                              model_performance: Dict[str, float],
                              alternative_models: List[str],
                              model_limitations: List[str],
                              confidence_score: Optional[float] = None) -> str:
        """Generate explanation for model selection"""
        explanation = ModelExplanation(
            explanation_id=f"model_{ticker}_{int(time.time())}",
            explanation_type=ExplanationType.MODEL_SELECTION,
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            title=f"Model Selection for {ticker}",
            summary=f"Selected {model_name} ({model_type}) for {ticker} forecasting",
            model_name=model_name,
            model_type=model_type,
            model_version=model_version,
            selection_criteria=selection_criteria,
            model_performance=model_performance,
            alternative_models=alternative_models,
            model_limitations=model_limitations,
            confidence_score=confidence_score,
            key_points=[
                f"Selected {model_name} based on {len(selection_criteria)} criteria",
                f"Model performance: {self._format_performance(model_performance)}",
                f"Considered {len(alternative_models)} alternative models"
            ],
            recommendations=[
                "Monitor model performance regularly",
                "Consider ensemble approaches for improved accuracy",
                "Validate model assumptions periodically"
            ],
            tags=['model', 'selection', ticker.lower()]
        )
        
        # Generate detailed explanation
        explanation.details = self._generate_model_explanation_details(explanation)
        
        # Add LLM explanation if enabled
        if self.llm_enabled:
            explanation.llm_explanation = self._generate_llm_explanation(explanation)
        
        return self._add_explanation(explanation)
    
    def explain_feature_importance(self,
                                 ticker: str,
                                 features_analyzed: List[str],
                                 feature_importance: Dict[str, float],
                                 feature_contributions: Dict[str, float],
                                 feature_correlations: Dict[str, float],
                                 feature_trends: Dict[str, str],
                                 market_context: Dict[str, Any]) -> str:
        """Generate explanation for feature importance"""
        # Identify key drivers
        key_drivers = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        key_drivers = [feature for feature, importance in key_drivers]
        
        explanation = FeatureExplanation(
            explanation_id=f"features_{ticker}_{int(time.time())}",
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            title=f"Feature Analysis for {ticker}",
            summary=f"Analyzed {len(features_analyzed)} features for {ticker}",
            features_analyzed=features_analyzed,
            feature_importance=feature_importance,
            feature_contributions=feature_contributions,
            feature_correlations=feature_correlations,
            feature_trends=feature_trends,
            key_drivers=key_drivers,
            key_points=[
                f"Top driver: {key_drivers[0]} ({feature_importance[key_drivers[0]]:.2%})",
                f"Analyzed {len(features_analyzed)} features total",
                f"Strongest correlation: {max(feature_correlations.items(), key=lambda x: abs(x[1]))[0]}"
            ],
            recommendations=[
                "Focus on top 3-5 most important features",
                "Monitor feature stability over time",
                "Consider feature interactions"
            ],
            metadata={'market_context': market_context},
            tags=['features', 'analysis', ticker.lower()]
        )
        
        # Generate detailed explanation
        explanation.details = self._generate_feature_explanation_details(explanation)
        
        # Add LLM explanation if enabled
        if self.llm_enabled:
            explanation.llm_explanation = self._generate_llm_explanation(explanation)
        
        return self._add_explanation(explanation)
    
    def explain_forecast(self,
                        ticker: str,
                        forecast_value: float,
                        forecast_horizon: int,
                        forecast_confidence: float,
                        model_used: str,
                        key_factors: List[str],
                        market_conditions: Dict[str, Any],
                        risk_factors: List[str],
                        assumptions: List[str],
                        scenario_analysis: Dict[str, float],
                        forecast_interval: Optional[tuple] = None) -> str:
        """Generate explanation for forecast"""
        explanation = ForecastExplanation(
            explanation_id=f"forecast_{ticker}_{int(time.time())}",
            explanation_type=ExplanationType.FORECAST_REASONING,
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            title=f"Forecast Explanation for {ticker}",
            summary=f"Forecast: {forecast_value:.2f} ({forecast_horizon}d horizon, {forecast_confidence:.1%} confidence)",
            forecast_value=forecast_value,
            forecast_horizon=forecast_horizon,
            forecast_confidence=forecast_confidence,
            forecast_interval=forecast_interval,
            model_used=model_used,
            market_conditions=market_conditions,
            risk_factors=risk_factors,
            assumptions=assumptions,
            scenario_analysis=scenario_analysis,
            key_points=[
                f"Forecast: {forecast_value:.2f} ({forecast_confidence:.1%} confidence)",
                f"Horizon: {forecast_horizon} days",
                f"Key factors: {len(key_factors)} identified",
                f"Risk factors: {len(risk_factors)} identified"
            ],
            recommendations=[
                "Monitor key factors for changes",
                "Update forecast if assumptions change",
                "Consider scenario analysis for risk management"
            ],
            tags=['forecast', 'prediction', ticker.lower()]
        )
        
        # Generate detailed explanation
        explanation.details = self._generate_forecast_explanation_details(explanation)
        
        # Add LLM explanation if enabled
        if self.llm_enabled:
            explanation.llm_explanation = self._generate_llm_explanation(explanation)
        
        return self._add_explanation(explanation)
    
    def explain_trade_decision(self,
                             ticker: str,
                             trade_type: str,
                             trade_reason: str,
                             expected_return: float,
                             expected_risk: float,
                             position_size: float,
                             entry_price: Optional[float],
                             stop_loss: Optional[float],
                             take_profit: Optional[float],
                             technical_signals: List[str],
                             fundamental_factors: List[str],
                             market_timing: str = "",
                             risk_reward_ratio: Optional[float] = None) -> str:
        """Generate explanation for trade decision"""
        if risk_reward_ratio is None:
            risk_reward_ratio = expected_return / expected_risk if expected_risk > 0 else 0
        
        explanation = TradeExplanation(
            explanation_id=f"trade_{ticker}_{int(time.time())}",
            explanation_type=ExplanationType.TRADE_DECISION,
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            title=f"Trade Decision for {ticker}",
            summary=f"{trade_type.title()} {ticker}: {trade_reason}",
            trade_type=trade_type,
            trade_reason=trade_reason,
            expected_return=expected_return,
            expected_risk=expected_risk,
            position_size=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            market_timing=market_timing,
            technical_signals=technical_signals,
            fundamental_factors=fundamental_factors,
            key_points=[
                f"Decision: {trade_type.title()} {ticker}",
                f"Expected return: {expected_return:.2%}",
                f"Risk/reward ratio: {risk_reward_ratio:.2f}",
                f"Position size: {position_size}"
            ],
            recommendations=[
                "Set appropriate stop loss and take profit levels",
                "Monitor position and adjust if needed",
                "Review decision if market conditions change"
            ],
            tags=['trade', 'decision', trade_type.lower(), ticker.lower()]
        )
        
        # Generate detailed explanation
        explanation.details = self._generate_trade_explanation_details(explanation)
        
        # Add LLM explanation if enabled
        if self.llm_enabled:
            explanation.llm_explanation = self._generate_llm_explanation(explanation)
        
        return self._add_explanation(explanation)
    
    def _generate_model_explanation_details(self, explanation: ModelExplanation) -> Dict[str, Any]:
        """Generate detailed model explanation"""
        details = {
            'model_info': {
                'name': explanation.model_name,
                'type': explanation.model_type,
                'version': explanation.model_version
            },
            'selection_process': {
                'criteria_count': len(explanation.selection_criteria),
                'criteria': explanation.selection_criteria,
                'alternatives_considered': len(explanation.alternative_models)
            },
            'performance_analysis': {
                'metrics': explanation.model_performance,
                'strengths': self._identify_model_strengths(explanation),
                'limitations': explanation.model_limitations
            },
            'recommendations': {
                'monitoring': "Monitor model performance weekly",
                'validation': "Validate assumptions monthly",
                'improvement': "Consider ensemble approaches"
            }
        }
        
        return details
    
    def _generate_feature_explanation_details(self, explanation: FeatureExplanation) -> Dict[str, Any]:
        """Generate detailed feature explanation"""
        details = {
            'feature_analysis': {
                'total_features': len(explanation.features_analyzed),
                'key_drivers': explanation.key_drivers,
                'importance_distribution': self._analyze_importance_distribution(explanation.feature_importance)
            },
            'contributions': {
                'top_contributors': sorted(explanation.feature_contributions.items(), 
                                         key=lambda x: abs(x[1]), reverse=True)[:5],
                'correlation_insights': self._analyze_correlations(explanation.feature_correlations)
            },
            'trends': {
                'feature_trends': explanation.feature_trends,
                'trend_analysis': self._analyze_feature_trends(explanation.feature_trends)
            },
            'market_context': explanation.metadata.get('market_context', {})
        }
        
        return details
    
    def _generate_forecast_explanation_details(self, explanation: ForecastExplanation) -> Dict[str, Any]:
        """Generate detailed forecast explanation"""
        details = {
            'forecast_summary': {
                'value': explanation.forecast_value,
                'horizon': explanation.forecast_horizon,
                'confidence': explanation.forecast_confidence,
                'interval': explanation.forecast_interval
            },
            'model_info': {
                'model_used': explanation.model_used,
                'assumptions': explanation.assumptions
            },
            'factor_analysis': {
                'key_factors': self._categorize_factors(explanation.details.get('key_factors', [])),
                'risk_factors': explanation.risk_factors,
                'market_conditions': explanation.market_conditions
            },
            'scenario_analysis': {
                'scenarios': explanation.scenario_analysis,
                'risk_assessment': self._assess_forecast_risk(explanation)
            }
        }
        
        return details
    
    def _generate_trade_explanation_details(self, explanation: TradeExplanation) -> Dict[str, Any]:
        """Generate detailed trade explanation"""
        details = {
            'trade_summary': {
                'type': explanation.trade_type,
                'reason': explanation.trade_reason,
                'position_size': explanation.position_size,
                'entry_price': explanation.entry_price
            },
            'risk_management': {
                'expected_return': explanation.expected_return,
                'expected_risk': explanation.expected_risk,
                'risk_reward_ratio': explanation.risk_reward_ratio,
                'stop_loss': explanation.stop_loss,
                'take_profit': explanation.take_profit
            },
            'analysis': {
                'technical_signals': explanation.technical_signals,
                'fundamental_factors': explanation.fundamental_factors,
                'market_timing': explanation.market_timing
            },
            'execution_plan': {
                'entry_strategy': self._generate_entry_strategy(explanation),
                'exit_strategy': self._generate_exit_strategy(explanation),
                'risk_controls': self._generate_risk_controls(explanation)
            }
        }
        
        return details
    
    def _generate_llm_explanation(self, explanation: Explanation) -> Optional[str]:
        """Generate LLM-powered explanation"""
        if not self.llm_client:
            return None
        
        try:
            # Create prompt based on explanation type
            prompt = self._create_llm_prompt(explanation)
            
            # Generate explanation
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert trading analyst explaining complex trading decisions in clear, professional language."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature
            )
            
            llm_explanation = response.choices[0].message.content
            
            # Update metrics
            self.llm_usage_metrics['total_requests'] += 1
            self.llm_usage_metrics['tokens_used'] += response.usage.total_tokens
            
            return llm_explanation
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM explanation: {e}")
            return None
    
    def _create_llm_prompt(self, explanation: Explanation) -> str:
        """Create LLM prompt for explanation"""
        base_prompt = f"""
Please provide a clear, professional explanation for the following trading decision:

Ticker: {explanation.ticker}
Type: {explanation.explanation_type.value}
Summary: {explanation.summary}

Key Points:
{chr(10).join(f"- {point}" for point in explanation.key_points)}

Details:
{json.dumps(explanation.details, indent=2)}

Please explain:
1. What factors led to this decision
2. Why this decision makes sense given the current market conditions
3. What risks should be considered
4. What monitoring should be done going forward

Provide a clear, concise explanation suitable for both technical and non-technical audiences.
"""
        
        return base_prompt
    
    def _add_explanation(self, explanation: Explanation) -> str:
        """Add explanation to storage"""
        self.explanations.append(explanation)
        self.explanation_index[explanation.explanation_id] = explanation
        
        # Update metrics
        self.explanation_metrics[explanation.explanation_type.value] += 1
        
        # Log explanation
        self.logger.info(f"Generated explanation: {explanation.explanation_id} - {explanation.title}")
        
        return explanation.explanation_id
    
    def get_explanation(self, explanation_id: str) -> Optional[Explanation]:
        """Get explanation by ID"""
        return self.explanation_index.get(explanation_id)
    
    def get_explanations(self,
                        explanation_types: Optional[List[ExplanationType]] = None,
                        ticker: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        tags: Optional[List[str]] = None) -> List[Explanation]:
        """Get filtered explanations"""
        filtered_explanations = self.explanations
        
        # Filter by type
        if explanation_types:
            filtered_explanations = [e for e in filtered_explanations if e.explanation_type in explanation_types]
        
        # Filter by ticker
        if ticker:
            filtered_explanations = [e for e in filtered_explanations if e.ticker == ticker]
        
        # Filter by time range
        if start_time:
            filtered_explanations = [e for e in filtered_explanations if datetime.fromisoformat(e.timestamp) >= start_time]
        
        if end_time:
            filtered_explanations = [e for e in filtered_explanations if datetime.fromisoformat(e.timestamp) <= end_time]
        
        # Filter by tags
        if tags:
            filtered_explanations = [e for e in filtered_explanations if any(tag in e.tags for tag in tags)]
        
        return filtered_explanations
    
    def generate_summary_report(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Generate summary report of explanations"""
        explanations = self.get_explanations(ticker=ticker) if ticker else self.explanations
        
        report = {
            'summary': {
                'total_explanations': len(explanations),
                'tickers_covered': list(set(e.ticker for e in explanations if e.ticker)),
                'explanation_types': defaultdict(int),
                'time_range': {
                    'start': min(e.timestamp for e in explanations) if explanations else None,
                    'end': max(e.timestamp for e in explanations) if explanations else None
                }
            },
            'metrics': {
                'explanations_by_type': dict(self.explanation_metrics),
                'llm_usage': dict(self.llm_usage_metrics)
            },
            'recent_explanations': [
                {
                    'id': e.explanation_id,
                    'type': e.explanation_type.value,
                    'ticker': e.ticker,
                    'title': e.title,
                    'timestamp': e.timestamp
                }
                for e in sorted(explanations, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
        
        # Count by type
        for explanation in explanations:
            report['summary']['explanation_types'][explanation.explanation_type.value] += 1
        
        return report
    
    def export_explanations(self, 
                          output_path: Optional[str] = None,
                          format: str = "json",
                          ticker: Optional[str] = None) -> str:
        """Export explanations to file"""
        explanations = self.get_explanations(ticker=ticker) if ticker else self.explanations
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ticker_suffix = f"_{ticker}" if ticker else ""
            output_path = self.explanations_dir / f"explanations_{timestamp}{ticker_suffix}.{format}"
        
        if format.lower() == "json":
            data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_explanations': len(explanations),
                    'ticker': ticker
                },
                'explanations': [asdict(e) for e in explanations]
            }
            safe_json_save(str(output_path), data)
        
        elif format.lower() == "csv":
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'explanation_id', 'explanation_type', 'timestamp', 'ticker',
                    'title', 'summary', 'confidence_score', 'tags'
                ])
                
                for explanation in explanations:
                    writer.writerow([
                        explanation.explanation_id,
                        explanation.explanation_type.value,
                        explanation.timestamp,
                        explanation.ticker,
                        explanation.title,
                        explanation.summary,
                        explanation.confidence_score,
                        ','.join(explanation.tags)
                    ])
        
        return str(output_path)
    
    # Helper methods for detailed analysis
    def _format_performance(self, performance: Dict[str, float]) -> str:
        """Format performance metrics"""
        if not performance:
            return "No performance data"
        
        formatted = []
        for metric, value in performance.items():
            if isinstance(value, float):
                formatted.append(f"{metric}: {value:.4f}")
            else:
                formatted.append(f"{metric}: {value}")
        
        return "; ".join(formatted)
    
    def _identify_model_strengths(self, explanation: ModelExplanation) -> List[str]:
        """Identify model strengths based on performance"""
        strengths = []
        
        if 'accuracy' in explanation.model_performance:
            acc = explanation.model_performance['accuracy']
            if acc > 0.8:
                strengths.append("High accuracy")
            elif acc > 0.6:
                strengths.append("Good accuracy")
        
        if 'sharpe_ratio' in explanation.model_performance:
            sr = explanation.model_performance['sharpe_ratio']
            if sr > 1.0:
                strengths.append("Strong risk-adjusted returns")
            elif sr > 0.5:
                strengths.append("Positive risk-adjusted returns")
        
        if 'stability' in explanation.model_performance:
            stability = explanation.model_performance['stability']
            if stability > 0.8:
                strengths.append("High stability")
        
        return strengths
    
    def _analyze_importance_distribution(self, importance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze feature importance distribution"""
        if not importance:
            return {}
        
        values = list(importance.values())
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'max': max(values),
            'min': min(values),
            'top_quartile': np.percentile(values, 75)
        }
    
    def _analyze_correlations(self, correlations: Dict[str, float]) -> Dict[str, Any]:
        """Analyze feature correlations"""
        if not correlations:
            return {}
        
        values = list(correlations.values())
        strong_correlations = [k for k, v in correlations.items() if abs(v) > 0.7]
        
        return {
            'mean_correlation': np.mean(values),
            'strong_correlations': strong_correlations,
            'correlation_range': (min(values), max(values))
        }
    
    def _analyze_feature_trends(self, trends: Dict[str, str]) -> Dict[str, int]:
        """Analyze feature trends"""
        trend_counts = defaultdict(int)
        for trend in trends.values():
            trend_counts[trend] += 1
        
        return dict(trend_counts)
    
    def _categorize_factors(self, factors: List[str]) -> Dict[str, List[str]]:
        """Categorize factors by type"""
        categories = {
            'technical': [],
            'fundamental': [],
            'sentiment': [],
            'macro': [],
            'other': []
        }
        
        for factor in factors:
            factor_lower = factor.lower()
            if any(tech in factor_lower for tech in ['rsi', 'macd', 'bollinger', 'moving_average']):
                categories['technical'].append(factor)
            elif any(fund in factor_lower for fund in ['earnings', 'revenue', 'pe_ratio', 'book_value']):
                categories['fundamental'].append(factor)
            elif any(sent in factor_lower for sent in ['sentiment', 'news', 'social']):
                categories['sentiment'].append(factor)
            elif any(macro in factor_lower for macro in ['gdp', 'inflation', 'interest_rate', 'fed']):
                categories['macro'].append(factor)
            else:
                categories['other'].append(factor)
        
        return {k: v for k, v in categories.items() if v}
    
    def _assess_forecast_risk(self, explanation: ForecastExplanation) -> Dict[str, Any]:
        """Assess forecast risk"""
        risk_level = "low"
        if explanation.forecast_confidence < 0.6:
            risk_level = "high"
        elif explanation.forecast_confidence < 0.8:
            risk_level = "medium"
        
        return {
            'risk_level': risk_level,
            'confidence_score': explanation.forecast_confidence,
            'risk_factors_count': len(explanation.risk_factors),
            'scenario_variance': np.var(list(explanation.scenario_analysis.values())) if explanation.scenario_analysis else 0
        }
    
    def _generate_entry_strategy(self, explanation: TradeExplanation) -> Dict[str, Any]:
        """Generate entry strategy"""
        return {
            'entry_type': 'market' if explanation.entry_price is None else 'limit',
            'entry_price': explanation.entry_price,
            'timing': explanation.market_timing,
            'size': explanation.position_size
        }
    
    def _generate_exit_strategy(self, explanation: TradeExplanation) -> Dict[str, Any]:
        """Generate exit strategy"""
        return {
            'stop_loss': explanation.stop_loss,
            'take_profit': explanation.take_profit,
            'risk_reward_ratio': explanation.risk_reward_ratio
        }
    
    def _generate_risk_controls(self, explanation: TradeExplanation) -> List[str]:
        """Generate risk controls"""
        controls = []
        
        if explanation.stop_loss:
            controls.append(f"Stop loss at {explanation.stop_loss}")
        
        if explanation.take_profit:
            controls.append(f"Take profit at {explanation.take_profit}")
        
        if explanation.expected_risk > 0.05:
            controls.append("Monitor position closely due to high risk")
        
        controls.append(f"Position size: {explanation.position_size}")
        
        return controls


# Convenience functions
def create_explainer_agent(config_path: str = "config/app_config.yaml") -> ExplainerAgent:
    """Create an explainer agent instance"""
    return ExplainerAgent(config_path)


def explain_trading_decision(explainer_agent: ExplainerAgent,
                           decision_type: str,
                           ticker: str,
                           decision_data: Dict[str, Any]) -> str:
    """Convenience function to explain trading decisions"""
    if decision_type == "model_selection":
        return explainer_agent.explain_model_selection(
            ticker=ticker,
            model_name=decision_data['model_name'],
            model_type=decision_data['model_type'],
            model_version=decision_data['model_version'],
            selection_criteria=decision_data['selection_criteria'],
            model_performance=decision_data['model_performance'],
            alternative_models=decision_data['alternative_models'],
            model_limitations=decision_data['model_limitations'],
            confidence_score=decision_data.get('confidence_score')
        )
    elif decision_type == "forecast":
        return explainer_agent.explain_forecast(
            ticker=ticker,
            forecast_value=decision_data['forecast_value'],
            forecast_horizon=decision_data['forecast_horizon'],
            forecast_confidence=decision_data['forecast_confidence'],
            model_used=decision_data['model_used'],
            key_factors=decision_data['key_factors'],
            market_conditions=decision_data['market_conditions'],
            risk_factors=decision_data['risk_factors'],
            assumptions=decision_data['assumptions'],
            scenario_analysis=decision_data.get('scenario_analysis', {}),
            forecast_interval=decision_data.get('forecast_interval')
        )
    elif decision_type == "trade":
        return explainer_agent.explain_trade_decision(
            ticker=ticker,
            trade_type=decision_data['trade_type'],
            trade_reason=decision_data['trade_reason'],
            expected_return=decision_data['expected_return'],
            expected_risk=decision_data['expected_risk'],
            position_size=decision_data['position_size'],
            entry_price=decision_data.get('entry_price'),
            stop_loss=decision_data.get('stop_loss'),
            take_profit=decision_data.get('take_profit'),
            technical_signals=decision_data.get('technical_signals', []),
            fundamental_factors=decision_data.get('fundamental_factors', []),
            market_timing=decision_data.get('market_timing', ''),
            risk_reward_ratio=decision_data.get('risk_reward_ratio')
        )
    else:
        raise ValueError(f"Unknown decision type: {decision_type}")


if __name__ == "__main__":
    # Example usage
    explainer = create_explainer_agent()
    
    # Explain model selection
    model_explanation_id = explainer.explain_model_selection(
        ticker="AAPL",
        model_name="LSTM_Ensemble",
        model_type="Neural Network",
        model_version="v2.1",
        selection_criteria=["accuracy", "stability", "interpretability"],
        model_performance={"accuracy": 0.85, "sharpe_ratio": 1.2, "stability": 0.9},
        alternative_models=["Random Forest", "XGBoost", "Prophet"],
        model_limitations=["Requires large dataset", "Black box model"]
    )
    
    # Explain forecast
    forecast_explanation_id = explainer.explain_forecast(
        ticker="AAPL",
        forecast_value=155.0,
        forecast_horizon=5,
        forecast_confidence=0.8,
        model_used="LSTM_Ensemble",
        key_factors=["Technical momentum", "Earnings growth", "Market sentiment"],
        market_conditions={"volatility": 0.02, "trend": "bullish"},
        risk_factors=["Market volatility", "Earnings uncertainty"],
        assumptions=["Stable market conditions", "No major news events"]
    )
    
    # Explain trade decision
    trade_explanation_id = explainer.explain_trade_decision(
        ticker="AAPL",
        trade_type="buy",
        trade_reason="Strong technical signals with positive forecast",
        expected_return=0.05,
        expected_risk=0.02,
        position_size=1000.0,
        entry_price=150.0,
        stop_loss=145.0,
        take_profit=160.0,
        technical_signals=["RSI oversold", "MACD crossover", "Volume spike"],
        fundamental_factors=["Strong earnings", "Market leadership"]
    )
    
    # Generate summary report
    report = explainer.generate_summary_report()
    print("Explainer Summary Report:")
    print(json.dumps(report, indent=2)) 