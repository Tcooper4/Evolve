"""
Prompt Router Agent

This agent intelligently routes user requests to appropriate agents based on:
- Request type and content analysis
- Agent capabilities and availability
- Historical performance and success rates
- Current system load and priorities

Features:
- Natural language request classification
- Agent capability matching
- Load balancing and priority management
- Fallback handling and error recovery
- Performance tracking and optimization
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from .base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.memory.agent_memory import AgentMemory
from trading.utils.reasoning_logger import ReasoningLogger, DecisionType, ConfidenceLevel

logger = logging.getLogger(__name__)

class RequestType(Enum):
    """Types of user requests."""
    FORECAST = "forecast"
    STRATEGY = "strategy"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    GENERAL = "general"
    UNKNOWN = "unknown"

class AgentCapability(Enum):
    """Agent capabilities."""
    FORECASTING = "forecasting"
    STRATEGY_GENERATION = "strategy_generation"
    MARKET_ANALYSIS = "market_analysis"
    OPTIMIZATION = "optimization"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    SYSTEM_MONITORING = "system_monitoring"
    GENERAL_QUERY = "general_query"

@dataclass
class AgentInfo:
    """Information about an available agent."""
    name: str
    capabilities: List[AgentCapability]
    priority: int
    max_concurrent: int
    current_load: int
    success_rate: float
    avg_response_time: float
    last_used: datetime
    is_available: bool = True
    custom_config: Optional[Dict[str, Any]] = None

@dataclass
class RoutingDecision:
    """Routing decision for a user request."""
    request_id: str
    request_type: RequestType
    primary_agent: str
    fallback_agents: List[str]
    confidence: float
    reasoning: str
    expected_response_time: float
    priority: int
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ParsedIntent:
    """Structured parsed intent result."""
    intent: str
    confidence: float
    args: Dict[str, Any]
    provider: str  # 'openai', 'huggingface', 'regex'
    raw_response: str
    error: Optional[str] = None
    json_spec: Optional[Dict[str, Any]] = None  # Full JSON spec for debugging

class PromptRouterAgent(BaseAgent):
    """
    Intelligent prompt router that directs user requests to appropriate agents.
    
    This agent analyzes incoming requests and determines the best agent(s) to handle
    them based on capabilities, availability, performance history, and current load.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the prompt router agent."""
        if config is None:
            config = AgentConfig(
                name="PromptRouterAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=10,
                timeout_seconds=30,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.reasoning_logger = ReasoningLogger()
        
        # Request classification patterns
        self.classification_patterns = {
            RequestType.FORECAST: [
                r'\b(forecast|predict|future|next|upcoming|tomorrow|next week|next month)\b',
                r'\b(price|stock|market|trend|movement|direction)\b',
                r'\b(how much|what will|when will|where will)\b'
            ],
            RequestType.STRATEGY: [
                r'\b(strategy|strategy|trading|signal|entry|exit|position)\b',
                r'\b(buy|sell|hold|long|short|trade)\b',
                r'\b(rsi|macd|bollinger|moving average|indicator)\b'
            ],
            RequestType.ANALYSIS: [
                r'\b(analyze|analysis|examine|study|review|assess|evaluate)\b',
                r'\b(performance|metrics|statistics|data|chart|graph)\b',
                r'\b(why|what caused|what happened|explain)\b'
            ],
            RequestType.OPTIMIZATION: [
                r'\b(optimize|tune|improve|enhance|better|best|optimal)\b',
                r'\b(parameters|settings|configuration|hyperparameters)\b',
                r'\b(performance|efficiency|accuracy|speed)\b'
            ],
            RequestType.PORTFOLIO: [
                r'\b(portfolio|allocation|diversification|risk|balance)\b',
                r'\b(asset|investment|holdings|positions|weights)\b',
                r'\b(rebalance|adjust|change|modify)\b'
            ],
            RequestType.SYSTEM: [
                r'\b(system|status|health|monitor|check|diagnose)\b',
                r'\b(error|problem|issue|bug|fix|repair)\b',
                r'\b(restart|stop|start|configure|setup)\b'
            ]
        }
        
        # Available agents and their capabilities
        self.available_agents: Dict[str, AgentInfo] = {}
        
        # Routing history for learning
        self.routing_history: List[RoutingDecision] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_routes': 0,
            'avg_response_time': 0.0,
            'agent_usage': {}
        }
        
        # Initialize agent registry
        self._initialize_agent_registry()
        
        self.logger.info("PromptRouterAgent initialized successfully")
    
    def _initialize_agent_registry(self):
        """Initialize the registry of available agents."""
        # Define available agents and their capabilities
        agent_definitions = {
            'ModelSelectorAgent': {
                'capabilities': [AgentCapability.FORECASTING],
                'priority': 1,
                'max_concurrent': 3,
                'success_rate': 0.85,
                'avg_response_time': 2.5
            },
            'StrategySelectorAgent': {
                'capabilities': [AgentCapability.STRATEGY_GENERATION],
                'priority': 1,
                'max_concurrent': 3,
                'success_rate': 0.80,
                'avg_response_time': 3.0
            },
            'MarketAnalyzerAgent': {
                'capabilities': [AgentCapability.MARKET_ANALYSIS],
                'priority': 2,
                'max_concurrent': 2,
                'success_rate': 0.90,
                'avg_response_time': 4.0
            },
            'MetaTunerAgent': {
                'capabilities': [AgentCapability.OPTIMIZATION],
                'priority': 2,
                'max_concurrent': 1,
                'success_rate': 0.75,
                'avg_response_time': 15.0
            },
            'PortfolioManagerAgent': {
                'capabilities': [AgentCapability.PORTFOLIO_MANAGEMENT],
                'priority': 1,
                'max_concurrent': 2,
                'success_rate': 0.88,
                'avg_response_time': 5.0
            },
            'SystemMonitorAgent': {
                'capabilities': [AgentCapability.SYSTEM_MONITORING],
                'priority': 3,
                'max_concurrent': 5,
                'success_rate': 0.95,
                'avg_response_time': 1.0
            },
            'QuantGPTAgent': {
                'capabilities': [AgentCapability.GENERAL_QUERY],
                'priority': 2,
                'max_concurrent': 5,
                'success_rate': 0.82,
                'avg_response_time': 3.5
            }
        }
        
        for agent_name, info in agent_definitions.items():
            self.available_agents[agent_name] = AgentInfo(
                name=agent_name,
                capabilities=info['capabilities'],
                priority=info['priority'],
                max_concurrent=info['max_concurrent'],
                current_load=0,
                success_rate=info['success_rate'],
                avg_response_time=info['avg_response_time'],
                last_used=datetime.now() - timedelta(hours=1),
                is_available=True
            )
        
        self.logger.info(f"Initialized {len(self.available_agents)} agents in registry")
    
    def route_request(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Route a user request to the most appropriate agent(s).
        
        Args:
            user_request: The user's request text
            context: Additional context information
            
        Returns:
            Routing decision with agent recommendations
        """
        request_id = f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        try:
            # Classify the request type
            request_type = self._classify_request(user_request)
            
            # Find suitable agents
            suitable_agents = self._find_suitable_agents(request_type, user_request)
            
            # Select primary and fallback agents
            primary_agent, fallback_agents = self._select_agents(suitable_agents, context)
            
            # Calculate confidence and expected response time
            confidence = self._calculate_routing_confidence(request_type, primary_agent, suitable_agents)
            expected_response_time = self._estimate_response_time(primary_agent, request_type)
            
            # Determine priority
            priority = self._determine_priority(request_type, context)
            
            # Create routing decision
            decision = RoutingDecision(
                request_id=request_id,
                request_type=request_type,
                primary_agent=primary_agent,
                fallback_agents=fallback_agents,
                confidence=confidence,
                reasoning=self._generate_routing_reasoning(request_type, primary_agent, suitable_agents),
                expected_response_time=expected_response_time,
                priority=priority,
                timestamp=datetime.now(),
                metadata={
                    'user_request': user_request,
                    'context': context,
                    'suitable_agents': [agent.name for agent in suitable_agents]
                }
            )
            
            # Log the routing decision
            self._log_routing_decision(decision)
            
            # Update performance metrics
            self._update_performance_metrics(decision)
            
            self.logger.info(f"Routed request to {primary_agent} with confidence {confidence:.2f}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error routing request: {e}")
            # Return fallback decision
            return self._create_fallback_decision(request_id, user_request)
    
    def _classify_request(self, user_request: str) -> RequestType:
        """Classify the type of user request."""
        user_request_lower = user_request.lower()
        
        # Calculate scores for each request type
        type_scores = {}
        for request_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, user_request_lower)
                score += len(matches)
            type_scores[request_type] = score
        
        # Find the type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return RequestType.GENERAL
    
    def _find_suitable_agents(self, request_type: RequestType, user_request: str) -> List[AgentInfo]:
        """Find agents suitable for handling the request."""
        suitable_agents = []
        
        # Map request types to capabilities
        capability_mapping = {
            RequestType.FORECAST: [AgentCapability.FORECASTING],
            RequestType.STRATEGY: [AgentCapability.STRATEGY_GENERATION],
            RequestType.ANALYSIS: [AgentCapability.MARKET_ANALYSIS],
            RequestType.OPTIMIZATION: [AgentCapability.OPTIMIZATION],
            RequestType.PORTFOLIO: [AgentCapability.PORTFOLIO_MANAGEMENT],
            RequestType.SYSTEM: [AgentCapability.SYSTEM_MONITORING],
            RequestType.GENERAL: [AgentCapability.GENERAL_QUERY],
            RequestType.UNKNOWN: [AgentCapability.GENERAL_QUERY]
        }
        
        required_capabilities = capability_mapping.get(request_type, [AgentCapability.GENERAL_QUERY])
        
        for agent in self.available_agents.values():
            if not agent.is_available:
                continue
                
            # Check if agent has required capabilities
            if any(cap in agent.capabilities for cap in required_capabilities):
                suitable_agents.append(agent)
        
        # If no specific agents found, include general query agents
        if not suitable_agents:
            for agent in self.available_agents.values():
                if AgentCapability.GENERAL_QUERY in agent.capabilities and agent.is_available:
                    suitable_agents.append(agent)
        
        return suitable_agents
    
    def _select_agents(self, suitable_agents: List[AgentInfo], context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
        """Select primary and fallback agents from suitable agents."""
        if not suitable_agents:
            return "QuantGPTAgent", []
        
        # Score agents based on multiple factors
        agent_scores = []
        for agent in suitable_agents:
            score = self._calculate_agent_score(agent, context)
            agent_scores.append((agent, score))
        
        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select primary agent
        primary_agent = agent_scores[0][0].name
        
        # Select fallback agents (next 2 best)
        fallback_agents = [agent.name for agent, _ in agent_scores[1:3]]
        
        return primary_agent, fallback_agents
    
    def _calculate_agent_score(self, agent: AgentInfo, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate a score for an agent based on multiple factors."""
        score = 0.0
        
        # Success rate (40% weight)
        score += agent.success_rate * 0.4
        
        # Availability (20% weight)
        availability_score = 1.0 - (agent.current_load / agent.max_concurrent)
        score += availability_score * 0.2
        
        # Response time (20% weight)
        response_score = max(0, 1.0 - (agent.avg_response_time / 30.0))  # Normalize to 30s max
        score += response_score * 0.2
        
        # Priority (10% weight)
        priority_score = 1.0 - (agent.priority - 1) / 3.0  # Normalize priority 1-3
        score += priority_score * 0.1
        
        # Recency (10% weight)
        hours_since_use = (datetime.now() - agent.last_used).total_seconds() / 3600
        recency_score = max(0, 1.0 - (hours_since_use / 24.0))  # Normalize to 24h
        score += recency_score * 0.1
        
        return score
    
    def _calculate_routing_confidence(self, request_type: RequestType, primary_agent: str, suitable_agents: List[AgentInfo]) -> float:
        """Calculate confidence in the routing decision."""
        if not suitable_agents:
            return 0.3  # Low confidence if no suitable agents
        
        # Base confidence from primary agent success rate
        primary_agent_info = self.available_agents.get(primary_agent)
        if not primary_agent_info:
            return 0.5
        
        base_confidence = primary_agent_info.success_rate
        
        # Adjust based on number of suitable agents
        if len(suitable_agents) == 1:
            # Only one option, lower confidence
            base_confidence *= 0.8
        elif len(suitable_agents) >= 3:
            # Many options, higher confidence in selection
            base_confidence *= 1.1
        
        # Adjust based on request type clarity
        if request_type == RequestType.UNKNOWN:
            base_confidence *= 0.7
        
        return min(1.0, max(0.0, base_confidence))
    
    def _estimate_response_time(self, primary_agent: str, request_type: RequestType) -> float:
        """Estimate expected response time."""
        agent_info = self.available_agents.get(primary_agent)
        if not agent_info:
            return 5.0  # Default estimate
        
        base_time = agent_info.avg_response_time
        
        # Adjust based on request type complexity
        complexity_multipliers = {
            RequestType.FORECAST: 1.0,
            RequestType.STRATEGY: 1.2,
            RequestType.ANALYSIS: 1.5,
            RequestType.OPTIMIZATION: 3.0,
            RequestType.PORTFOLIO: 2.0,
            RequestType.SYSTEM: 0.8,
            RequestType.GENERAL: 1.0,
            RequestType.UNKNOWN: 1.5
        }
        
        multiplier = complexity_multipliers.get(request_type, 1.0)
        return base_time * multiplier
    
    def _determine_priority(self, request_type: RequestType, context: Optional[Dict[str, Any]] = None) -> int:
        """Determine the priority of the request."""
        # Base priority from request type
        base_priorities = {
            RequestType.SYSTEM: 1,  # Highest priority
            RequestType.FORECAST: 2,
            RequestType.STRATEGY: 2,
            RequestType.ANALYSIS: 3,
            RequestType.OPTIMIZATION: 3,
            RequestType.PORTFOLIO: 2,
            RequestType.GENERAL: 4,
            RequestType.UNKNOWN: 4
        }
        
        priority = base_priorities.get(request_type, 4)
        
        # Adjust based on context
        if context:
            if context.get('urgent', False):
                priority = max(1, priority - 1)
            if context.get('user_type') == 'premium':
                priority = max(1, priority - 1)
        
        return priority
    
    def _generate_routing_reasoning(self, request_type: RequestType, primary_agent: str, suitable_agents: List[AgentInfo]) -> str:
        """Generate reasoning for the routing decision."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Request classified as {request_type.value}")
        reasoning_parts.append(f"Selected {primary_agent} as primary agent")
        
        if suitable_agents:
            reasoning_parts.append(f"Found {len(suitable_agents)} suitable agents")
            
            # Add agent-specific reasoning
            primary_agent_info = self.available_agents.get(primary_agent)
            if primary_agent_info:
                reasoning_parts.append(f"{primary_agent} has {primary_agent_info.success_rate:.1%} success rate")
                reasoning_parts.append(f"Expected response time: {primary_agent_info.avg_response_time:.1f}s")
        
        return "; ".join(reasoning_parts)
    
    def _log_routing_decision(self, decision: RoutingDecision):
        """Log the routing decision for analysis."""
        # Log to reasoning logger
        self.reasoning_logger.log_decision(
            agent_name='PromptRouterAgent',
            decision_type=DecisionType.ROUTING,
            action_taken=f"Routed to {decision.primary_agent}",
            context={
                'request_type': decision.request_type.value,
                'confidence': decision.confidence,
                'priority': decision.priority,
                'expected_response_time': decision.expected_response_time
            },
            reasoning={
                'primary_reason': decision.reasoning,
                'supporting_factors': [
                    f"Request type: {decision.request_type.value}",
                    f"Primary agent: {decision.primary_agent}",
                    f"Fallback agents: {', '.join(decision.fallback_agents)}"
                ],
                'alternatives_considered': [f"Could have used: {', '.join(decision.fallback_agents)}"],
                'confidence_explanation': f"Confidence {decision.confidence:.1%} based on agent capabilities and performance"
            },
            confidence_level=ConfidenceLevel.HIGH if decision.confidence > 0.8 else ConfidenceLevel.MEDIUM,
            metadata=decision.metadata
        )
        
        # Store in routing history
        self.routing_history.append(decision)
        
        # Keep only last 1000 decisions
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def _update_performance_metrics(self, decision: RoutingDecision):
        """Update performance tracking metrics."""
        self.performance_metrics['total_requests'] += 1
        
        # Update agent usage
        primary_agent = decision.primary_agent
        if primary_agent not in self.performance_metrics['agent_usage']:
            self.performance_metrics['agent_usage'][primary_agent] = 0
        self.performance_metrics['agent_usage'][primary_agent] += 1
    
    def _create_fallback_decision(self, request_id: str, user_request: str) -> RoutingDecision:
        """Create a fallback routing decision when normal routing fails."""
        return RoutingDecision(
            request_id=request_id,
            request_type=RequestType.UNKNOWN,
            primary_agent="QuantGPTAgent",
            fallback_agents=[],
            confidence=0.3,
            reasoning="Fallback routing due to error in request classification",
            expected_response_time=5.0,
            priority=4,
            timestamp=datetime.now(),
            metadata={'user_request': user_request, 'error': 'Routing failed'}
        )
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        return {
            'total_requests': self.performance_metrics['total_requests'],
            'successful_routes': self.performance_metrics['successful_routes'],
            'avg_response_time': self.performance_metrics['avg_response_time'],
            'agent_usage': self.performance_metrics['agent_usage'],
            'recent_decisions': len(self.routing_history),
            'available_agents': len([a for a in self.available_agents.values() if a.is_available])
        }
    
    def update_agent_status(self, agent_name: str, is_available: bool, current_load: int = 0):
        """Update the status of an agent."""
        if agent_name in self.available_agents:
            self.available_agents[agent_name].is_available = is_available
            self.available_agents[agent_name].current_load = current_load
            self.logger.info(f"Updated {agent_name} status: available={is_available}, load={current_load}")
    
    def record_agent_performance(self, agent_name: str, success: bool, response_time: float):
        """Record performance metrics for an agent."""
        if agent_name in self.available_agents:
            agent = self.available_agents[agent_name]
            
            # Update success rate with exponential moving average
            alpha = 0.1
            agent.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * agent.success_rate
            
            # Update response time with exponential moving average
            agent.avg_response_time = alpha * response_time + (1 - alpha) * agent.avg_response_time
            
            # Update last used time
            agent.last_used = datetime.now()
            
            self.logger.debug(f"Updated {agent_name} performance: success_rate={agent.success_rate:.3f}, avg_response_time={agent.avg_response_time:.2f}s")

# Convenience function for creating router agent
def create_prompt_router() -> PromptRouterAgent:
    """Create a configured prompt router agent."""
    return PromptRouterAgent() 