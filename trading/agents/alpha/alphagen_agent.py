
Agent - Hypothesis Generation Agent

This agent generates trading hypotheses using LLMs and prompt templates.
It analyzes market data, news, and patterns to propose new alpha strategies.import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from openai import OpenAI

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult, AgentState
from trading.utils.error_handling import log_errors, retry_on_error
from trading.exceptions import StrategyError, ModelError

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A trading hypothesis with metadata.""
    
    id: str
    title: str
    description: str
    strategy_type: str  # momentum, mean_reversion, arbitrage, etc.
    asset_class: str    # equity, crypto, forex, etc.
    timeframe: str      # 1m,5, 1h, 1d, etc.
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_parameters: Dict[str, Any]
    confidence_score: float
    reasoning: str
    data_sources: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return[object Object]
         idlf.id,
       title": self.title,
        description": self.description,
          strategy_type": self.strategy_type,
        asset_class": self.asset_class,
           timeframe": self.timeframe,
           entry_conditions": self.entry_conditions,
            exit_conditions": self.exit_conditions,
            risk_parameters": self.risk_parameters,
           confidence_score: self.confidence_score,
           reasoning": self.reasoning,
         data_sources": self.data_sources,
            created_at:self.created_at.isoformat(),
           tags": self.tags,
            metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hypothesis":
    te from dictionary."""
        if isinstance(data[created_at"], str):
            data["created_at]= datetime.fromisoformat(data["created_at])        return cls(**data)


class AlphaGenAgent(BaseAgent):ent that generates trading hypotheses using LLMs and market analysis."""
    
    __version__ = "100    __author__ = "AlphaGen Team"
    __description__ = Generatestrading hypotheses using LLMs and market analysis"
    __tags__ = ["alpha, hypothesis", "llm,research"]
    __capabilities__ = ["hypothesis_generation", market_analysis", "llm_integration"]
    __dependencies__ = ["openai", "pandas",numpy]   
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.llm_client = None
        self.market_data_cache =[object Object]}
        self.hypothesis_templates = self._load_hypothesis_templates()
        self.generated_hypotheses = []
        
    def _setup(self) -> None:
        tup the agent."""
        try:
            # Initialize LLM client
            api_key = self.config.custom_config.get(openai_api_key")
            if not api_key:
                raise ValueError("OpenAI API key not found in config")
            
            self.llm_client = OpenAI(api_key=api_key)
            logger.info("AlphaGen agent setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup AlphaGen agent: {e}")
            raise
    
    @log_errors()
    async def execute(self, **kwargs) -> AgentResult:
    ute the hypothesis generation process."""
        try:
            self.status.state = AgentState.RUNNING
            start_time = datetime.now()
            
            # Get market context
            market_context = await self._analyze_market_context()
            
            # Generate hypotheses
            hypotheses = await self._generate_hypotheses(market_context)
            
            # Filter and rank hypotheses
            ranked_hypotheses = self._rank_hypotheses(hypotheses)
            
            # Store results
            self.generated_hypotheses.extend(ranked_hypotheses)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.status.state = AgentState.SUCCESS
            
            return AgentResult(
                success=True,
                data={
                  hypotheses": [h.to_dict() for h in ranked_hypotheses],
                   market_context": market_context,
              generation_metadata": {
                        total_generated": len(hypotheses),
                     ranked_count": len(ranked_hypotheses),
                       execution_time: execution_time                   }
                },
                execution_time=execution_time,
                metadata={agent": "alphagen"}
            )
            
        except Exception as e:
            self.status.state = AgentState.ERROR
            return self.handle_error(e)
    
    async def _analyze_market_context(self) -> Dict[str, Any]:
        yze current market context for hypothesis generation."""
        try:
            # Get market data for analysis
            market_data = await self._get_market_data()
            
            # Analyze market regimes
            regime_analysis = self._analyze_market_regime(market_data)
            
            # Identify market anomalies
            anomalies = self._detect_anomalies(market_data)
            
            # Get sentiment data
            sentiment = await self._get_sentiment_data()
            
            return[object Object]
              market_regime: regime_analysis,
          anomalies": anomalies,
          sentiment": sentiment,
                volatility: self._calculate_volatility(market_data),
          correlation_matrix: self._calculate_correlations(market_data),
          timestamp:datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(fFailed to analyze market context: {e}")
            return {"error": str(e)}
    
    @retry_on_error(max_retries=3, delay=1sync def _generate_hypotheses(self, market_context: Dict[str, Any]) -> List[Hypothesis]:
 enerate hypotheses using LLM and templates."""
        try:
            hypotheses = []
            
            # Use LLM to generate hypotheses
            llm_hypotheses = await self._generate_llm_hypotheses(market_context)
            hypotheses.extend(llm_hypotheses)
            
            # Use template-based generation
            template_hypotheses = self._generate_template_hypotheses(market_context)
            hypotheses.extend(template_hypotheses)
            
            # Use pattern-based generation
            pattern_hypotheses = self._generate_pattern_hypotheses(market_context)
            hypotheses.extend(pattern_hypotheses)
            
            logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Failed to generate hypotheses: {e}")
            return    
    async def _generate_llm_hypotheses(self, market_context: Dict[str, Any]) -> List[Hypothesis]:
 enerate hypotheses using LLM."""
        try:
            prompt = self._build_llm_prompt(market_context)
            
            response = await self._call_llm(prompt)
            
            # Parse LLM response
            hypotheses = self._parse_llm_response(response)
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"LLM hypothesis generation failed: {e}")
            return   
    def _build_llm_prompt(self, market_context: Dict[str, Any]) -> str:
     uild prompt for LLM hypothesis generation.      return f"""
You are an expert quantitative trader and alpha researcher. Analyze the following market context and generate 3 specific, testable trading hypotheses.

Market Context:
{json.dumps(market_context, indent=2)}

Generate hypotheses that are:
1. Specific and testable
2n market anomalies or inefficiencies
3. Include clear entry/exit conditions
4ve reasonable risk parameters
5. Include confidence scores (0-1)

For each hypothesis, provide:
- Title
- Description
- Strategy type (momentum, mean_reversion, arbitrage, etc.)
- Asset class
- Timeframe
- Entry conditions (list)
- Exit conditions (list)
- Risk parameters (dict)
- Confidence score (0- Reasoning
- Data sources needed

Format your response as a JSON array of hypothesis objects.
    
    async def _call_llm(self, prompt: str) -> str:
     l the LLM API."""
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4,          messages=[
                    {"role": system, content": You are an expert quantitative trader."},
                    {"role": usercontent": prompt}
                ],
                temperature=0.7               max_tokens=20   )
            
            return response.choices[0tent
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> List[Hypothesis]:
  Parse LLM response into Hypothesis objects."""
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1or json_end == 0            logger.warning("No JSON array found in LLM response)            return []
            
            json_str = response[json_start:json_end]
            hypothesis_data = json.loads(json_str)
            
            hypotheses = 
            for i, data in enumerate(hypothesis_data):
                try:
                    hypothesis = Hypothesis(
                        id=fllm_hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                        title=data.get("title", fLLM Hypothesis {i}"),
                        description=data.get("description", ""),
                        strategy_type=data.get(strategy_type", "unknown"),
                        asset_class=data.get("asset_class", "equity"),
                        timeframe=data.get("timeframe", "1d"),
                        entry_conditions=data.get("entry_conditions", []),
                        exit_conditions=data.get(exit_conditions", []),
                        risk_parameters=data.get(risk_parameters", {}),
                        confidence_score=float(data.get("confidence_score", 0.5)),
                        reasoning=data.get(reasoning                   data_sources=data.get(data_sources", []),
                        tags=data.get(tags", []) + [llm_generated                   )
                    hypotheses.append(hypothesis)
                    
                except Exception as e:
                    logger.error(f"Failed to parse hypothesis {i}: {e}")
                    continue
            
            return hypotheses
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return
    def _generate_template_hypotheses(self, market_context: Dict[str, Any]) -> List[Hypothesis]:
 enerate hypotheses using predefined templates."""
        hypotheses = []
        
        # Momentum template
        if market_context.get("market_regime,[object Object]).get("type) == ding":
            hypothesis = Hypothesis(
                id=ftemplate_momentum_{datetime.now().strftime('%Y%m%d_%H%M%S')},             title="Momentum Continuation Strategy,       description="Trade in the direction of established trends with momentum confirmation,          strategy_type="momentum,             asset_class="equity,         timeframe="1h,             entry_conditions=[
                 Price above 20-period moving average",
                    RSI between 40-80",
               Volume above average",
                    Trend strength > 0.7               ],
                exit_conditions=[
                Price crosses below 20-period moving average",
              RSI > 80              Volume drops below average"
                ],
                risk_parameters={
                   stop_loss": 0.02,
                   take_profit": 0.06,
                    position_size": 00.5               },
                confidence_score=0.75         reasoning="Market is in trending regime, momentum strategies typically perform well,              data_sources=["price_data", "volume_data", "technical_indicators"],
                tags=["template", momentum", "trending"]
            )
            hypotheses.append(hypothesis)
        
        # Mean reversion template
        if market_context.get(volatility", [object Object]}).get(current", 0) > 0.3:
            hypothesis = Hypothesis(
                id=f"template_mean_reversion_{datetime.now().strftime('%Y%m%d_%H%M%S')},             title="Mean Reversion Strategy,       description="Trade against extreme moves expecting reversion to mean,          strategy_type=mean_reversion,             asset_class="equity,         timeframe="4h,             entry_conditions=[
              RSI > 80                 Price deviation from mean > 2 standard deviations",
                 Bollinger Band touch"
                ],
                exit_conditions=[
                    RSI returns to 40-60 range",
                   Price returns to mean",
                 Bollinger Band exit"
                ],
                risk_parameters={
                   stop_loss": 0.03,
                   take_profit": 0.04,
                    position_size": 00.3               },
                confidence_score=0.65         reasoning="High volatility suggests mean reversion opportunities,              data_sources=["price_data", "technical_indicators"],
                tags=["template,mean_reversion", "high_volatility"]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_pattern_hypotheses(self, market_context: Dict[str, Any]) -> List[Hypothesis]:
 enerate hypotheses based on detected patterns."""
        hypotheses = []
        
        # Analyze anomalies for pattern-based hypotheses
        anomalies = market_context.get("anomalies", [])
        
        for anomaly in anomalies:
            if anomaly.get(type") == "volume_spike:        hypothesis = Hypothesis(
                    id=f"pattern_volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    title="Volume Spike Strategy",
                    description="Trade on unusual volume activity",
                    strategy_type="event_driven",
                    asset_class="equity",
                    timeframe="15m",
                    entry_conditions=[
                  Volume >3 average volume",
                        Price movement > 2%",
                     News sentiment change"
                    ],
                    exit_conditions=[
                        Volume returns to normal",
                       Price stabilizes",
                        24 hour timeout                   ],
                    risk_parameters={
                       stop_loss": 0.05,
                       take_profit": 0.08,
                        position_size": 0.02
                    },
                    confidence_score=0.6,
                    reasoning=f"Detected volume anomaly: {anomaly.get(description', '')}",
                    data_sources=["price_data", "volume_data", "news_data"],
                    tags=[pattern", volume", "event_driven]                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _rank_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        Rank hypotheses by confidence and feasibility."""
        try:
            # Score each hypothesis
            scored_hypotheses =     for hypothesis in hypotheses:
                score = self._calculate_hypothesis_score(hypothesis)
                hypothesis.metadatascorere
                scored_hypotheses.append(hypothesis)
            
            # Sort by score (descending)
            ranked_hypotheses = sorted(
                scored_hypotheses,
                key=lambda h: h.metadata.get("score", 0),
                reverse=True
            )
            
            # Return top hypotheses
            return ranked_hypotheses[:10]  # Top 10 hypotheses
            
        except Exception as e:
            logger.error(f"Failed to rank hypotheses: {e}")
            return hypotheses
    
    def _calculate_hypothesis_score(self, hypothesis: Hypothesis) -> float:
    Calculate a score for hypothesis ranking."""
        try:
            score =0      
            # Base confidence score (40% weight)
            score += hypothesis.confidence_score *0.4      
            # Strategy type bonus (20% weight)
            strategy_bonus =[object Object]
               momentum8
                mean_reversion": 0.7
                arbitrage9
                event_driven": 0.6
               statistical": 00.75     }
            score += strategy_bonus.get(hypothesis.strategy_type, 0.5) *0.2      
            # Timeframe bonus (10% weight)
            timeframe_bonus =[object Object]
         1m3
         5m5
          15m7
         1h8
         4h9
        1d:0.8     }
            score += timeframe_bonus.get(hypothesis.timeframe, 0.5) *0.1      
            # Data availability bonus (15% weight)
            data_score = len(hypothesis.data_sources) / 5.0  # Normalize to 0-1
            score += data_score * 0.15      
            # Risk parameter completeness (15% weight)
            risk_params = hypothesis.risk_parameters
            risk_completeness = sum(
                1ram in risk_params else0               for param in [stop_loss", "take_profit",position_size]
            ) / 3.0
            score += risk_completeness * 0.15      
            return min(score, 1          
        except Exception as e:
            logger.error(f"Failed to calculate hypothesis score: {e}")
            return hypothesis.confidence_score
    
    async def _get_market_data(self) -> Dict[str, Any]:
        et market data for analysis."     # Placeholder - implement actual data fetching
        return [object Object]           spy": {"price:450, volume": 100change02},
           qqq": {"price:380volume: 8000change015,
           vix: {rice":15, volume": 5000change":05        }
    
    def _analyze_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        yze current market regime."     # Placeholder - implement actual regime detection
        return {
           type": "trending",
           strength: 0.7,
         volatility": "medium",
          correlation":high"
        }
    
    def _detect_anomalies(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
      t market anomalies."     # Placeholder - implement actual anomaly detection
        return [
           [object Object]
                type": "volume_spike,
              asset,
                description: al volume increase,
           severity": "medium"
            }
        ]
    
    async def _get_sentiment_data(self) -> Dict[str, Any]:
      Get sentiment data."     # Placeholder - implement actual sentiment fetching
        return {
            overall_sentiment": 0.6
            news_sentiment":00.55          social_sentiment":0.65
           fear_greed_index": 65        }
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
   market volatility."     # Placeholder - implement actual volatility calculation
        return {
           current00.25      historical": 0.2,
           regime": "normal"
        }
    
    def _calculate_correlations(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        asset correlations."     # Placeholder - implement actual correlation calculation
        return [object Object]
           spy_qqq00.85,
            spy_vix00.70,
           qqq_vix": -0.65        }
    
    def _load_hypothesis_templates(self) -> Dict[str, Any]:
        othesis generation templates.
        return {
        momentum[object Object]
               description: m-based strategies,
              conditions":trending_market", "volume]    },
           mean_reversion":[object Object]
              description: version strategies,
              conditions":high_volatility",extreme_moves]    },
         arbitrage[object Object]
                description": "Arbitrage opportunities,
              conditions: [rice_discrepancies", "low_risk"]
            }
        }
    
    def validate_input(self, **kwargs) -> bool:
       e input parameters.""   required_params = ["market_data,sentiment_data"]
        return all(param in kwargs for param in required_params)
    
    def validate_config(self) -> bool:
       gent configuration.""   required_config = [openai_api_key"]
        custom_config = self.config.custom_config or[object Object]        return all(key in custom_config for key in required_config)
    
    def handle_error(self, error: Exception) -> AgentResult:
      ndle errors during execution."""
        self.status.state = AgentState.ERROR
        self.status.current_error = str(error)
        
        return AgentResult(
            success=false
            error_message=str(error),
            error_type=type(error).__name__,
            metadata={agent": "alphagen}        )
    
    def get_capabilities(self) -> List[str]:
  agent capabilities.       return self.__capabilities__
    
    def get_requirements(self) -> Dict[str, Any]:
  agent requirements.
        return {
         dependencies": self.__dependencies__,
            api_keys": ["openai_api_key"],
            data_sources: ["market_data,sentiment_data", "news_data]        }
    
    def get_generated_hypotheses(self) -> List[Hypothesis]:
Get all generated hypotheses.       return self.generated_hypotheses.copy()
    
    def clear_hypotheses(self) -> None:
      stored hypotheses.    self.generated_hypotheses.clear() 