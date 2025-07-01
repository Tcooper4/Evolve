"""
Reasoning Logger

Records and formats agent decisions in plain language for transparency.
Provides chat-style explanations of why agents made specific decisions.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from jinja2 import Template
import openai

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions agents can make."""
    FORECAST = "forecast"
    STRATEGY = "strategy"
    MODEL_SELECTION = "model_selection"
    PARAMETER_TUNING = "parameter_tuning"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    SIGNAL_GENERATION = "signal_generation"
    DATA_SELECTION = "data_selection"
    FEATURE_ENGINEERING = "feature_engineering"
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    ALERT = "alert"

class ConfidenceLevel(Enum):
    """Confidence levels for decisions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class DecisionContext:
    """Context information for a decision."""
    symbol: str
    timeframe: str
    timestamp: str
    market_conditions: Dict[str, Any]
    available_data: List[str]
    constraints: Dict[str, Any]
    user_preferences: Dict[str, Any]

@dataclass
class DecisionReasoning:
    """Detailed reasoning for a decision."""
    primary_reason: str
    supporting_factors: List[str]
    alternatives_considered: List[str]
    risks_assessed: List[str]
    confidence_explanation: str
    expected_outcome: str

@dataclass
class AgentDecision:
    """Complete decision record."""
    decision_id: str
    agent_name: str
    decision_type: DecisionType
    action_taken: str
    context: DecisionContext
    reasoning: DecisionReasoning
    confidence_level: ConfidenceLevel
    timestamp: str
    metadata: Dict[str, Any]

class ReasoningLogger:
    """
    Records and formats agent decisions in plain language for transparency.
    
    Provides chat-style explanations and maintains a searchable log of all decisions.
    """
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 openai_api_key: str = None,
                 log_dir: str = "logs/reasoning",
                 enable_gpt_explanations: bool = True):
        """
        Initialize the ReasoningLogger.
        
        Args:
            redis_host: Redis host for distributed logging
            redis_port: Redis port
            redis_db: Redis database
            openai_api_key: OpenAI API key for enhanced explanations
            log_dir: Directory to store log files
            enable_gpt_explanations: Whether to use GPT for enhanced explanations
        """
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.enable_gpt_explanations = enable_gpt_explanations
        
        # Initialize OpenAI if available
        if self.openai_api_key and self.enable_gpt_explanations:
            openai.api_key = self.openai_api_key
        
        # Setup logging directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "decisions").mkdir(exist_ok=True)
        (self.log_dir / "summaries").mkdir(exist_ok=True)
        (self.log_dir / "explanations").mkdir(exist_ok=True)
        
        # Decision counter
        self.decision_counter = 0
        
        logger.info("ReasoningLogger initialized")def log_decision(self,
                    agent_name: str,
                    decision_type: DecisionType,
                    action_taken: str,
                    context: Dict[str, Any],
                    reasoning: Dict[str, Any],
                    confidence_level: ConfidenceLevel,
                    metadata: Dict[str, Any] = None) -> str:
        """
        Log a decision made by an agent.
        
        Args:
            agent_name: Name of the agent making the decision
            decision_type: Type of decision
            action_taken: What action was taken
            context: Context information
            reasoning: Reasoning details
            confidence_level: Confidence in the decision
            metadata: Additional metadata
            
        Returns:
            Decision ID
        """
        try:
            # Generate decision ID
            decision_id = f"{agent_name}_{decision_type.value}_{int(time.time())}"
            
            # Create decision context
            decision_context = DecisionContext(
                symbol=context.get('symbol', 'Unknown'),
                timeframe=context.get('timeframe', 'Unknown'),
                timestamp=datetime.now().isoformat(),
                market_conditions=context.get('market_conditions', {}),
                available_data=context.get('available_data', []),
                constraints=context.get('constraints', {}),
                user_preferences=context.get('user_preferences', {})
            )
            
            # Create reasoning
            decision_reasoning = DecisionReasoning(
                primary_reason=reasoning.get('primary_reason', ''),
                supporting_factors=reasoning.get('supporting_factors', []),
                alternatives_considered=reasoning.get('alternatives_considered', []),
                risks_assessed=reasoning.get('risks_assessed', []),
                confidence_explanation=reasoning.get('confidence_explanation', ''),
                expected_outcome=reasoning.get('expected_outcome', '')
            )
            
            # Create decision record
            decision = AgentDecision(
                decision_id=decision_id,
                agent_name=agent_name,
                decision_type=decision_type,
                action_taken=action_taken,
                context=decision_context,
                reasoning=decision_reasoning,
                confidence_level=confidence_level,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            # Store decision
            self._store_decision(decision)
            
            # Generate explanations
            self._generate_explanations(decision)
            
            # Publish to Redis for real-time updates
            self._publish_decision(decision)
            
            logger.info(f"Logged decision: {decision_id}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
            raise
    
    def _store_decision(self, decision: AgentDecision):
        """Store decision in Redis and file system."""
        try:
            # Store in Redis
            decision_key = f"decision:{decision.decision_id}"
            self.redis_client.setex(
                decision_key,
                86400 * 30,  # 30 days TTL
                json.dumps(asdict(decision), default=str)
            )
            
            # Store in file system
            decision_file = self.log_dir / "decisions" / f"{decision.decision_id}.json"
            with open(decision_file, 'w') as f:
                json.dump(asdict(decision), f, indent=2, default=str)
            
            # Add to decision list
            decision_list_key = f"decisions:{decision.agent_name}"
            self.redis_client.lpush(decision_list_key, decision.decision_id)
            self.redis_client.ltrim(decision_list_key, 0, 999)  # Keep last 1000
            
        except Exception as e:
            logger.error(f"Error storing decision: {e}")

    def _generate_explanations(self, decision: AgentDecision):
        """Generate plain language explanations."""
        try:
            # Generate summary
            summary = self._generate_summary(decision)
            
            # Generate chat-style explanation
            chat_explanation = self._generate_chat_explanation(decision)
            
            # Store explanations
            summary_file = self.log_dir / "summaries" / f"{decision.decision_id}.txt"
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            explanation_file = self.log_dir / "explanations" / f"{decision.decision_id}.txt"
            with open(explanation_file, 'w') as f:
                f.write(chat_explanation)
            
            # Store in Redis
            self.redis_client.setex(
                f"summary:{decision.decision_id}",
                86400 * 30,
                summary
            )
            
            self.redis_client.setex(
                f"explanation:{decision.decision_id}",
                86400 * 30,
                chat_explanation
            )
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")

    def _generate_summary(self, decision: AgentDecision) -> str:
        """Generate a plain language summary of the decision."""
        template = Template("""
# Decision Summary

**Agent:** {{ decision.agent_name }}  
**Type:** {{ decision.decision_type.value.replace('_', ' ').title() }}  
**Time:** {{ decision.timestamp }}  
**Symbol:** {{ decision.context.symbol }}  
**Timeframe:** {{ decision.context.timeframe }}

## Action Taken
{{ decision.action_taken }}

## Primary Reason
{{ decision.reasoning.primary_reason }}

## Confidence Level
{{ decision.confidence_level.value.replace('_', ' ').title() }}

## Supporting Factors
{% for factor in decision.reasoning.supporting_factors %}
- {{ factor }}
{% endfor %}

## Expected Outcome
{{ decision.reasoning.expected_outcome }}

## Market Conditions
{% for key, value in decision.context.market_conditions.items() %}
- {{ key }}: {{ value }}
{% endfor %}
""")
        
        return template.render(decision=decision)
    
    def _generate_chat_explanation(self, decision: AgentDecision) -> str:
        """Generate a chat-style explanation of the decision."""
        if self.enable_gpt_explanations and self.openai_api_key:
            return self._generate_gpt_explanation(decision)
        else:
            return self._generate_fallback_explanation(decision)
    
    def _generate_gpt_explanation(self, decision: AgentDecision) -> str:
        """Generate GPT-enhanced explanation."""
        try:
            prompt = f"""
            You are an AI trading agent explaining a decision you just made. 
            Write a conversational explanation as if you're talking to a human trader.
            
            Decision Details:
            - Agent: {decision.agent_name}
            - Type: {decision.decision_type.value}
            - Action: {decision.action_taken}
            - Symbol: {decision.context.symbol}
            - Timeframe: {decision.context.timeframe}
            - Primary Reason: {decision.reasoning.primary_reason}
            - Supporting Factors: {', '.join(decision.reasoning.supporting_factors)}
            - Confidence: {decision.confidence_level.value}
            - Expected Outcome: {decision.reasoning.expected_outcome}
            - Market Conditions: {decision.context.market_conditions}
            
            Write a natural, conversational explanation that includes:
            1. What you decided to do
            2. Why you made this decision
            3. What factors influenced your choice
            4. Your confidence level and why
            5. What you expect to happen
            
            Make it sound natural and conversational, like you're explaining to a colleague.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI trading assistant explaining decisions in a conversational way."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating GPT explanation: {e}")
            return self._generate_fallback_explanation(decision)
    
    def _generate_fallback_explanation(self, decision: AgentDecision) -> str:
        """Generate fallback explanation without GPT."""
        template = Template("""
🤖 **{{ decision.agent_name }}** - {{ decision.timestamp }}

Hey there! I just made a {{ decision.decision_type.value.replace('_', ' ') }} decision for {{ decision.context.symbol }}.

**What I did:** {{ decision.action_taken }}

**Why I did it:** {{ decision.reasoning.primary_reason }}

**Key factors that influenced me:**
{% for factor in decision.reasoning.supporting_factors %}
• {{ factor }}
{% endfor %}

**My confidence level:** {{ decision.confidence_level.value.replace('_', ' ').title() }}

**What I expect to happen:** {{ decision.reasoning.expected_outcome }}

**Current market conditions:**
{% for key, value in decision.context.market_conditions.items() %}
• {{ key }}: {{ value }}
{% endfor %}

{% if decision.reasoning.alternatives_considered %}
**Other options I considered:**
{% for alt in decision.reasoning.alternatives_considered %}
• {{ alt }}
{% endfor %}
{% endif %}

{% if decision.reasoning.risks_assessed %}
**Risks I'm aware of:**
{% for risk in decision.reasoning.risks_assessed %}
• {{ risk }}
{% endfor %}
{% endif %}
""")
        
        return template.render(decision=decision)
    
    def _publish_decision(self, decision: AgentDecision):
        """Publish decision to Redis for real-time updates."""
        try:
            event_data = {
                'decision_id': decision.decision_id,
                'agent_name': decision.agent_name,
                'decision_type': decision.decision_type.value,
                'action_taken': decision.action_taken,
                'symbol': decision.context.symbol,
                'timestamp': decision.timestamp,
                'confidence_level': decision.confidence_level.value
            }
            
            self.redis_client.publish('agent_decisions', json.dumps(event_data))
            
        except Exception as e:
            logger.error(f"Error publishing decision: {e}")

    def get_decision(self, decision_id: str) -> Optional[AgentDecision]:
        """Retrieve a specific decision."""
        try:
            # Try Redis first
            decision_data = self.redis_client.get(f"decision:{decision_id}")
            if decision_data:
                data = json.loads(decision_data)
                return self._dict_to_decision(data)
            
            # Try file system
            decision_file = self.log_dir / "decisions" / f"{decision_id}.json"
            if decision_file.exists():
                with open(decision_file, 'r') as f:
                    data = json.load(f)
                return self._dict_to_decision(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving decision: {e}")

    def get_agent_decisions(self, agent_name: str, limit: int = 50) -> List[AgentDecision]:
        """Get recent decisions for a specific agent."""
        try:
            decision_ids = self.redis_client.lrange(f"decisions:{agent_name}", 0, limit - 1)
            decisions = []
            
            for decision_id in decision_ids:
                decision = self.get_decision(decision_id)
                if decision:
                    decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error retrieving agent decisions: {e}")
            return []
    
    def get_decisions_by_type(self, decision_type: DecisionType, limit: int = 50) -> List[AgentDecision]:
        """Get recent decisions of a specific type."""
        try:
            # This would require additional indexing in a production system
            # For now, we'll scan recent decisions
            all_decisions = []
            
            # Get decisions from all agents
            agents = self.redis_client.keys("decisions:*")
            for agent_key in agents:
                agent_name = agent_key.split(":", 1)[1]
                agent_decisions = self.get_agent_decisions(agent_name, limit=limit)
                all_decisions.extend(agent_decisions)
            
            # Filter by type
            filtered_decisions = [
                d for d in all_decisions 
                if d.decision_type == decision_type
            ]
            
            # Sort by timestamp and limit
            filtered_decisions.sort(key=lambda x: x.timestamp, reverse=True)
            return filtered_decisions[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving decisions by type: {e}")
            return []
    
    def get_summary(self, decision_id: str) -> Optional[str]:
        """Get the plain language summary for a decision."""
        try:
            # Try Redis first
            summary = self.redis_client.get(f"summary:{decision_id}")
            if summary:
                return summary
            
            # Try file system
            summary_file = self.log_dir / "summaries" / f"{decision_id}.txt"
            if summary_file.exists():
                return summary_file.read_text()
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving summary: {e}")

    def get_explanation(self, decision_id: str) -> Optional[str]:
        """Get the chat-style explanation for a decision."""
        try:
            # Try Redis first
            explanation = self.redis_client.get(f"explanation:{decision_id}")
            if explanation:
                return explanation
            
            # Try file system
            explanation_file = self.log_dir / "explanations" / f"{decision_id}.txt"
            if explanation_file.exists():
                return explanation_file.read_text()
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving explanation: {e}")

    def _dict_to_decision(self, data: Dict[str, Any]) -> AgentDecision:
        """Convert dictionary to AgentDecision object."""
        try:
            # Convert enums
            data['decision_type'] = DecisionType(data['decision_type'])
            data['confidence_level'] = ConfidenceLevel(data['confidence_level'])
            
            # Convert context and reasoning
            data['context'] = DecisionContext(**data['context'])
            data['reasoning'] = DecisionReasoning(**data['reasoning'])
            
            return AgentDecision(**data)
            
        except Exception as e:
            logger.error(f"Error converting dict to decision: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged decisions."""
        try:
            stats = {
                'total_decisions': 0,
                'decisions_by_agent': {},
                'decisions_by_type': {},
                'confidence_distribution': {},
                'recent_activity': []
            }
            
            # Get all agent keys
            agent_keys = self.redis_client.keys("decisions:*")
            
            for agent_key in agent_keys:
                agent_name = agent_key.split(":", 1)[1]
                decision_ids = self.redis_client.lrange(agent_key, 0, -1)
                
                stats['decisions_by_agent'][agent_name] = len(decision_ids)
                stats['total_decisions'] += len(decision_ids)
                
                # Get recent decisions for this agent
                recent_decisions = self.get_agent_decisions(agent_name, limit=10)
                for decision in recent_decisions:
                    stats['recent_activity'].append({
                        'decision_id': decision.decision_id,
                        'agent_name': decision.agent_name,
                        'decision_type': decision.decision_type.value,
                        'symbol': decision.context.symbol,
                        'timestamp': decision.timestamp
                    })
                    
                    # Count by type
                    decision_type = decision.decision_type.value
                    stats['decisions_by_type'][decision_type] = stats['decisions_by_type'].get(decision_type, 0) + 1
                    
                    # Count by confidence
                    confidence = decision.confidence_level.value
                    stats['confidence_distribution'][confidence] = stats['confidence_distribution'].get(confidence, 0) + 1
            
            # Sort recent activity by timestamp
            stats['recent_activity'].sort(key=lambda x: x['timestamp'], reverse=True)
            stats['recent_activity'] = stats['recent_activity'][:20]  # Keep last 20
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def clear_old_decisions(self, days: int = 30):
        """Clear decisions older than specified days."""
        try:
            cutoff_time = datetime.now().timestamp() - (days * 86400)
            
            # Get all agent keys
            agent_keys = self.redis_client.keys("decisions:*")
            
            for agent_key in agent_keys:
                agent_name = agent_key.split(":", 1)[1]
                decision_ids = self.redis_client.lrange(agent_key, 0, -1)
                
                for decision_id in decision_ids:
                    decision = self.get_decision(decision_id)
                    if decision:
                        decision_time = datetime.fromisoformat(decision.timestamp).timestamp()
                        if decision_time < cutoff_time:
                            # Remove from Redis
                            self.redis_client.delete(f"decision:{decision_id}")
                            self.redis_client.delete(f"summary:{decision_id}")
                            self.redis_client.delete(f"explanation:{decision_id}")
                            
                            # Remove from agent list
                            self.redis_client.lrem(agent_key, 0, decision_id)
                            
                            # Remove files
                            for file_type in ['decisions', 'summaries', 'explanations']:
                                file_path = self.log_dir / file_type / f"{decision_id}.json"
                                if file_path.exists():
                                    file_path.unlink()
            
            logger.info(f"Cleared decisions older than {days} days")
            
        except Exception as e:
            logger.error(f"Error clearing old decisions: {e}")

# Convenience functions
def log_forecast_decision(agent_name: str, symbol: str, timeframe: str, 
                         forecast_value: float, confidence: float, 
                         reasoning: Dict[str, Any], **kwargs) -> str:
    """Log a forecast decision."""
    logger = ReasoningLogger(**kwargs)
    
    return logger.log_decision(
        agent_name=agent_name,
        decision_type=DecisionType.FORECAST,
        action_taken=f"Predicted {symbol} will be {forecast_value:.2f}",
        context={
            'symbol': symbol,
            'timeframe': timeframe,
            'market_conditions': reasoning.get('market_conditions', {}),
            'available_data': reasoning.get('available_data', []),
            'constraints': reasoning.get('constraints', {}),
            'user_preferences': reasoning.get('user_preferences', {})
        },
        reasoning=reasoning,
        confidence_level=ConfidenceLevel.HIGH if confidence > 0.7 else ConfidenceLevel.MEDIUM,
        metadata={'forecast_value': forecast_value, 'confidence_score': confidence}
    )

def log_strategy_decision(agent_name: str, symbol: str, action: str, 
                         strategy_name: str, reasoning: Dict[str, Any], **kwargs) -> str:
    """Log a strategy decision."""
    logger = ReasoningLogger(**kwargs)
    
    return logger.log_decision(
        agent_name=agent_name,
        decision_type=DecisionType.STRATEGY,
        action_taken=f"Executed {strategy_name} strategy: {action}",
        context={
            'symbol': symbol,
            'timeframe': reasoning.get('timeframe', 'Unknown'),
            'market_conditions': reasoning.get('market_conditions', {}),
            'available_data': reasoning.get('available_data', []),
            'constraints': reasoning.get('constraints', {}),
            'user_preferences': reasoning.get('user_preferences', {})
        },
        reasoning=reasoning,
        confidence_level=reasoning.get('confidence_level', ConfidenceLevel.MEDIUM),
        metadata={'strategy_name': strategy_name, 'action': action}
    ) 