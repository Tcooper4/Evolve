"""
Refactored Prompt Router Module

This module provides a modular, memory-aware prompt routing system with:
- Memory of past prompts and performance
- Automatic detection of when to update weights or search for better models
- Modular agent classes for each prompt route
- Performance tracking and optimization
"""

import logging
import re
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
import numpy as np

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
    INVESTMENT = "investment"
    UNKNOWN = "unknown"


@dataclass
class PromptContext:
    """Enhanced context information for prompt processing."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    previous_requests: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessedPrompt:
    """Processed prompt information with enhanced metadata."""
    original_prompt: str
    request_type: RequestType
    confidence: float
    extracted_parameters: Dict[str, Any]
    context: PromptContext
    routing_suggestions: List[str]
    processing_time: float
    memory_hits: int = 0
    similar_prompts: List[str] = field(default_factory=list)


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    agent_name: str
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    user_satisfaction: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    weight: float = 1.0  # Dynamic weight for agent selection


@dataclass
class PromptMemory:
    """Memory entry for a processed prompt."""
    prompt_hash: str
    original_prompt: str
    request_type: RequestType
    agent_used: str
    success: bool
    response_time: float
    user_feedback: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)


class BasePromptAgent(ABC):
    """Base class for all prompt routing agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.performance = AgentPerformance(agent_name=name)
        self.memory = deque(maxlen=1000)  # Keep last 1000 interactions
        
    @abstractmethod
    def can_handle(self, prompt: str, processed: ProcessedPrompt) -> bool:
        """Check if this agent can handle the given prompt."""
        pass
    
    @abstractmethod
    def handle(self, prompt: str, processed: ProcessedPrompt) -> Dict[str, Any]:
        """Handle the prompt and return a response."""
        pass
    
    def update_performance(self, success: bool, response_time: float, user_feedback: Optional[float] = None):
        """Update agent performance metrics."""
        self.performance.total_requests += 1
        if success:
            self.performance.successful_requests += 1
        
        # Update success rate
        self.performance.success_rate = self.performance.successful_requests / self.performance.total_requests
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        self.performance.avg_response_time = (
            alpha * response_time + (1 - alpha) * self.performance.avg_response_time
        )
        
        # Update user satisfaction if feedback provided
        if user_feedback is not None:
            if self.performance.user_satisfaction == 0.0:
                self.performance.user_satisfaction = user_feedback
            else:
                self.performance.user_satisfaction = (
                    alpha * user_feedback + (1 - alpha) * self.performance.user_satisfaction
                )
        
        self.performance.last_updated = datetime.now()
        
        # Store in memory
        self.memory.append({
            'timestamp': datetime.now(),
            'success': success,
            'response_time': response_time,
            'user_feedback': user_feedback
        })
    
    def should_update_weights(self) -> bool:
        """Check if agent weights should be updated based on performance."""
        if self.performance.total_requests < 10:
            return False
        
        # Update if success rate is low or response time is high
        return (self.performance.success_rate < 0.7 or 
                self.performance.avg_response_time > 5.0)
    
    def get_weight(self) -> float:
        """Get current agent weight based on performance."""
        # Base weight on success rate and user satisfaction
        weight = (self.performance.success_rate * 0.6 + 
                 self.performance.user_satisfaction * 0.4)
        
        # Penalize slow response times
        if self.performance.avg_response_time > 2.0:
            weight *= 0.8
        
        return max(0.1, weight)  # Minimum weight of 0.1


class InvestmentAgent(BasePromptAgent):
    """Agent for handling investment-related queries."""
    
    def __init__(self):
        super().__init__("InvestmentAgent")
        self.investment_keywords = [
            "invest", "investment", "buy", "purchase", "acquire",
            "top stocks", "best stocks", "recommended", "suggest",
            "what should", "which stocks", "what to buy", "where to invest",
            "opportunity", "potential", "growth", "returns"
        ]
    
    def can_handle(self, prompt: str, processed: ProcessedPrompt) -> bool:
        """Check if this is an investment query."""
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.investment_keywords)
    
    def handle(self, prompt: str, processed: ProcessedPrompt) -> Dict[str, Any]:
        """Handle investment query."""
        start_time = datetime.now()
        
        try:
            # Try to route to TopRankedForecastAgent
            from agents.top_ranked_forecast_agent import TopRankedForecastAgent
            
            agent = TopRankedForecastAgent()
            result = agent.run(prompt)
            
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_performance(True, response_time)
            
            return {
                "success": True,
                "message": f"Investment analysis completed. {result.get('message', '')}",
                "request_type": "investment",
                "confidence": processed.confidence,
                "agent_used": self.name,
                "result": result,
                "response_time": response_time
            }
            
        except ImportError:
            self.logger.warning("TopRankedForecastAgent not available, using fallback")
            return self._fallback_response(prompt, processed, start_time)
        except Exception as e:
            self.logger.error(f"Error in TopRankedForecastAgent: {e}")
            return self._fallback_response(prompt, processed, start_time)
    
    def _fallback_response(self, prompt: str, processed: ProcessedPrompt, start_time: datetime) -> Dict[str, Any]:
        """Fallback response for investment queries."""
        response_time = (datetime.now() - start_time).total_seconds()
        self.update_performance(True, response_time)
        
        return {
            "success": True,
            "message": "I can help you with investment decisions. Please try asking about specific stocks or use the forecast feature for detailed analysis.",
            "request_type": "investment",
            "confidence": processed.confidence,
            "agent_used": self.name,
            "fallback_used": True,
            "suggestions": [
                "Try: 'Forecast AAPL for next week'",
                "Try: 'Analyze TSLA performance'",
                "Try: 'What's the best strategy for tech stocks?'",
            ],
            "response_time": response_time
        }


class ForecastAgent(BasePromptAgent):
    """Agent for handling forecast-related queries."""
    
    def __init__(self):
        super().__init__("ForecastAgent")
        self.forecast_keywords = [
            "forecast", "predict", "future", "next", "upcoming",
            "tomorrow", "next week", "next month", "price", "stock",
            "market", "trend", "movement", "direction"
        ]
    
    def can_handle(self, prompt: str, processed: ProcessedPrompt) -> bool:
        """Check if this is a forecast query."""
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.forecast_keywords)
    
    def handle(self, prompt: str, processed: ProcessedPrompt) -> Dict[str, Any]:
        """Handle forecast query."""
        start_time = datetime.now()
        
        try:
            # Extract forecast parameters
            symbol = processed.extracted_parameters.get('symbol')
            timeframe = processed.extracted_parameters.get('timeframe', '1d')
            days = processed.extracted_parameters.get('days', 7)
            
            # Route to appropriate forecast engine
            if symbol:
                # Try to use specific model if specified
                model = processed.extracted_parameters.get('model', 'ensemble')
                result = self._run_forecast(symbol, timeframe, days, model)
            else:
                result = {"message": "Please specify a stock symbol for forecasting"}
            
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_performance(True, response_time)
            
            return {
                "success": True,
                "message": f"Forecast request processed: {result.get('message', '')}",
                "request_type": "forecast",
                "confidence": processed.confidence,
                "agent_used": self.name,
                "parameters": processed.extracted_parameters,
                "result": result,
                "response_time": response_time
            }
            
        except Exception as e:
            self.logger.error(f"Error in forecast handling: {e}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_performance(False, response_time)
            
            return {
                "success": False,
                "message": f"Error processing forecast: {str(e)}",
                "agent_used": self.name,
                "response_time": response_time
            }
    
    def _run_forecast(self, symbol: str, timeframe: str, days: int, model: str) -> Dict[str, Any]:
        """Run forecast using specified model."""
        # This would integrate with your actual forecast engines
        return {
            "message": f"Forecast for {symbol} using {model} model for {days} days",
            "symbol": symbol,
            "timeframe": timeframe,
            "model": model,
            "days": days
        }


class StrategyAgent(BasePromptAgent):
    """Agent for handling strategy-related queries."""
    
    def __init__(self):
        super().__init__("StrategyAgent")
        self.strategy_keywords = [
            "strategy", "trading", "signal", "entry", "exit", "position",
            "buy", "sell", "hold", "long", "short", "trade",
            "rsi", "macd", "bollinger", "moving average", "indicator"
        ]
    
    def can_handle(self, prompt: str, processed: ProcessedPrompt) -> bool:
        """Check if this is a strategy query."""
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.strategy_keywords)
    
    def handle(self, prompt: str, processed: ProcessedPrompt) -> Dict[str, Any]:
        """Handle strategy query."""
        start_time = datetime.now()
        
        try:
            strategy_type = processed.extracted_parameters.get('strategy', 'custom')
            symbol = processed.extracted_parameters.get('symbol')
            
            result = self._run_strategy_analysis(symbol, strategy_type)
            
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_performance(True, response_time)
            
            return {
                "success": True,
                "message": f"Strategy analysis completed: {result.get('message', '')}",
                "request_type": "strategy",
                "confidence": processed.confidence,
                "agent_used": self.name,
                "parameters": processed.extracted_parameters,
                "result": result,
                "response_time": response_time
            }
            
        except Exception as e:
            self.logger.error(f"Error in strategy handling: {e}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_performance(False, response_time)
            
            return {
                "success": False,
                "message": f"Error processing strategy: {str(e)}",
                "agent_used": self.name,
                "response_time": response_time
            }
    
    def _run_strategy_analysis(self, symbol: Optional[str], strategy_type: str) -> Dict[str, Any]:
        """Run strategy analysis."""
        return {
            "message": f"Strategy analysis for {symbol or 'general market'} using {strategy_type}",
            "strategy_type": strategy_type,
            "symbol": symbol
        }


class GeneralAgent(BasePromptAgent):
    """Agent for handling general queries."""
    
    def __init__(self):
        super().__init__("GeneralAgent")
    
    def can_handle(self, prompt: str, processed: ProcessedPrompt) -> bool:
        """General agent can handle any query as fallback."""
        return True
    
    def handle(self, prompt: str, processed: ProcessedPrompt) -> Dict[str, Any]:
        """Handle general query."""
        start_time = datetime.now()
        
        response_time = (datetime.now() - start_time).total_seconds()
        self.update_performance(True, response_time)
        
        return {
            "success": True,
            "message": f"General request processed. Routing to: {', '.join(processed.routing_suggestions)}",
            "request_type": processed.request_type.value,
            "confidence": processed.confidence,
            "agent_used": self.name,
            "routing_suggestions": processed.routing_suggestions,
            "parameters": processed.extracted_parameters,
            "response_time": response_time
        }


class PromptMemoryManager:
    """Manages memory of past prompts and performance."""
    
    def __init__(self, memory_file: str = "data/prompt_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memories: Dict[str, PromptMemory] = {}
        self.prompt_similarity_cache: Dict[str, List[str]] = {}
        self.load_memory()
    
    def add_memory(self, memory: PromptMemory):
        """Add a new memory entry."""
        self.memories[memory.prompt_hash] = memory
        self._update_similarity_cache(memory)
        self.save_memory()
    
    def find_similar_prompts(self, prompt: str, threshold: float = 0.8) -> List[PromptMemory]:
        """Find similar prompts in memory."""
        prompt_hash = self._hash_prompt(prompt)
        
        if prompt_hash in self.prompt_similarity_cache:
            similar_hashes = self.prompt_similarity_cache[prompt_hash]
            return [self.memories[h] for h in similar_hashes if h in self.memories]
        
        similar_prompts = []
        for memory in self.memories.values():
            similarity = self._calculate_similarity(prompt, memory.original_prompt)
            if similarity >= threshold:
                similar_prompts.append(memory)
        
        return similar_prompts
    
    def get_agent_performance(self, agent_name: str, days: int = 30) -> AgentPerformance:
        """Get performance metrics for an agent over the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        agent_memories = [
            m for m in self.memories.values()
            if m.agent_used == agent_name and m.timestamp >= cutoff_date
        ]
        
        if not agent_memories:
            return AgentPerformance(agent_name=agent_name)
        
        total_requests = len(agent_memories)
        successful_requests = sum(1 for m in agent_memories if m.success)
        avg_response_time = np.mean([m.response_time for m in agent_memories])
        user_satisfaction = np.mean([m.user_feedback for m in agent_memories if m.user_feedback is not None])
        
        return AgentPerformance(
            agent_name=agent_name,
            success_rate=successful_requests / total_requests,
            avg_response_time=avg_response_time,
            user_satisfaction=user_satisfaction if not np.isnan(user_satisfaction) else 0.0,
            total_requests=total_requests,
            successful_requests=successful_requests
        )
    
    def should_search_better_models(self, agent_name: str) -> bool:
        """Check if we should search for better models based on performance."""
        performance = self.get_agent_performance(agent_name)
        
        # Search for better models if:
        # 1. Success rate is low
        # 2. Response time is high
        # 3. User satisfaction is low
        return (performance.success_rate < 0.6 or 
                performance.avg_response_time > 3.0 or 
                performance.user_satisfaction < 0.5)
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash for the prompt."""
        import hashlib
        return hashlib.md5(prompt.lower().encode()).hexdigest()
    
    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts."""
        return SequenceMatcher(None, prompt1.lower(), prompt2.lower()).ratio()
    
    def _update_similarity_cache(self, memory: PromptMemory):
        """Update similarity cache for new memory."""
        for existing_hash, existing_memory in self.memories.items():
            if existing_hash != memory.prompt_hash:
                similarity = self._calculate_similarity(
                    memory.original_prompt, existing_memory.original_prompt
                )
                if similarity >= 0.8:
                    # Add to both caches
                    if memory.prompt_hash not in self.prompt_similarity_cache:
                        self.prompt_similarity_cache[memory.prompt_hash] = []
                    if existing_hash not in self.prompt_similarity_cache:
                        self.prompt_similarity_cache[existing_hash] = []
                    
                    self.prompt_similarity_cache[memory.prompt_hash].append(existing_hash)
                    self.prompt_similarity_cache[existing_hash].append(memory.prompt_hash)
    
    def load_memory(self):
        """Load memory from file."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    for memory_data in data.get('memories', []):
                        memory = PromptMemory(**memory_data)
                        memory.timestamp = datetime.fromisoformat(memory_data['timestamp'])
                        self.memories[memory.prompt_hash] = memory
                        self._update_similarity_cache(memory)
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
    
    def save_memory(self):
        """Save memory to file."""
        try:
            data = {
                'memories': [
                    {
                        **memory.__dict__,
                        'timestamp': memory.timestamp.isoformat()
                    }
                    for memory in self.memories.values()
                ]
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory: {e}")


class EnhancedPromptProcessor:
    """Enhanced prompt processor with memory and learning capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = PromptMemoryManager()
        
        # Classification patterns (same as original)
        self.classification_patterns = {
            RequestType.FORECAST: [
                r"\b(forecast|predict|future|next|upcoming|tomorrow|next week|next month)\b",
                r"\b(price|stock|market|trend|movement|direction)\b",
                r"\b(how much|what will|when will|where will)\b",
            ],
            RequestType.STRATEGY: [
                r"\b(strategy|trading|signal|entry|exit|position)\b",
                r"\b(buy|sell|hold|long|short|trade)\b",
                r"\b(rsi|macd|bollinger|moving average|indicator)\b",
            ],
            RequestType.ANALYSIS: [
                r"\b(analyze|analysis|examine|study|review|assess|evaluate)\b",
                r"\b(performance|metrics|statistics|data|chart|graph)\b",
                r"\b(why|what caused|what happened|explain)\b",
            ],
            RequestType.OPTIMIZATION: [
                r"\b(optimize|tune|improve|enhance|better|best|optimal)\b",
                r"\b(parameters|settings|configuration|hyperparameters)\b",
                r"\b(performance|efficiency|accuracy|speed)\b",
            ],
            RequestType.PORTFOLIO: [
                r"\b(portfolio|allocation|diversification|risk|balance)\b",
                r"\b(asset|investment|holdings|positions|weights)\b",
                r"\b(rebalance|adjust|change|modify)\b",
            ],
            RequestType.SYSTEM: [
                r"\b(system|status|health|monitor|check|diagnose)\b",
                r"\b(error|problem|issue|bug|fix|repair)\b",
                r"\b(restart|stop|start|configure|setup)\b",
            ],
            RequestType.INVESTMENT: [
                r"\b(invest|investment|buy|purchase|acquire)\b",
                r"\b(top stocks|best stocks|recommended|suggest)\b",
                r"\b(what should|which stocks|what to buy|where to invest)\b",
                r"\b(opportunity|potential|growth|returns)\b",
                r"\b(today|now|current|market)\b",
            ],
        }
        
        # Parameter extraction patterns
        self.parameter_patterns = {
            "symbol": r"\b([A-Z]{1,5})\b",
            "timeframe": r"\b(1m|5m|15m|30m|1h|4h|1d|1w|1M)\b",
            "days": r"\b(\d+)\s*(days?|d)\b",
            "model": r"\b(lstm|arima|xgboost|prophet|ensemble|transformer)\b",
            "strategy": r"\b(rsi|macd|bollinger|sma|ema|custom)\b",
            "risk_level": r"\b(low|medium|high|conservative|aggressive)\b",
        }
    
    def process_prompt(self, prompt: str, context: Optional[PromptContext] = None) -> ProcessedPrompt:
        """Process a user prompt with memory enhancement."""
        start_time = datetime.now()
        
        if context is None:
            context = PromptContext()
        
        # Normalize prompt
        normalized_prompt = self._normalize_prompt(prompt)
        
        # Check memory for similar prompts
        similar_prompts = self.memory_manager.find_similar_prompts(prompt)
        memory_hits = len(similar_prompts)
        
        # Classify request type
        request_type = self._classify_request(normalized_prompt)
        confidence = self._calculate_confidence(normalized_prompt, request_type)
        
        # Extract parameters
        extracted_parameters = self._extract_parameters(normalized_prompt)
        
        # Generate routing suggestions
        routing_suggestions = self._generate_routing_suggestions(request_type, extracted_parameters)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessedPrompt(
            original_prompt=prompt,
            request_type=request_type,
            confidence=confidence,
            extracted_parameters=extracted_parameters,
            context=context,
            routing_suggestions=routing_suggestions,
            processing_time=processing_time,
            memory_hits=memory_hits,
            similar_prompts=[m.original_prompt for m in similar_prompts[:3]]  # Top 3 similar
        )
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for consistent processing."""
        normalized = prompt.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized
    
    def _classify_request(self, prompt: str) -> RequestType:
        """Classify the type of request."""
        prompt_lower = prompt.lower()
        scores = {}
        
        for request_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower)
                score += len(matches)
            scores[request_type] = score
        
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return RequestType.UNKNOWN
    
    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract parameters from the prompt."""
        parameters = {}
        prompt_lower = prompt.lower()
        
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, prompt_lower)
            if matches:
                if param_name == "days":
                    numbers = re.findall(r"\d+", str(matches[0]))
                    if numbers:
                        parameters[param_name] = int(numbers[0])
                else:
                    parameters[param_name] = matches[0] if len(matches) == 1 else matches
        
        return parameters
    
    def _calculate_confidence(self, prompt: str, request_type: RequestType) -> float:
        """Calculate confidence in the classification."""
        if request_type == RequestType.UNKNOWN:
            return 0.0
        
        prompt_lower = prompt.lower()
        patterns = self.classification_patterns[request_type]
        
        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, prompt_lower)
            total_matches += len(matches)
        
        confidence = min(1.0, total_matches / max(1, len(prompt.split())))
        return confidence
    
    def _generate_routing_suggestions(self, request_type: RequestType, parameters: Dict[str, Any]) -> List[str]:
        """Generate routing suggestions based on request type and parameters."""
        suggestions = []
        
        if request_type == RequestType.FORECAST:
            suggestions.extend(["ForecastAgent", "ModelSelectorAgent", "ForecastEngine"])
            if "model" in parameters:
                suggestions.append(f"{parameters['model'].title()}Model")
        
        elif request_type == RequestType.STRATEGY:
            suggestions.extend(["StrategyAgent", "StrategySelectorAgent", "StrategyEngine"])
            if "strategy" in parameters:
                suggestions.append(f"{parameters['strategy'].title()}Strategy")
        
        elif request_type == RequestType.INVESTMENT:
            suggestions.extend(["InvestmentAgent", "TopRankedForecastAgent", "PortfolioManager"])
        
        elif request_type == RequestType.ANALYSIS:
            suggestions.extend(["AnalysisEngine", "DataAnalyzer"])
        
        elif request_type == RequestType.OPTIMIZATION:
            suggestions.extend(["OptimizationEngine", "HyperparameterTuner"])
        
        elif request_type == RequestType.PORTFOLIO:
            suggestions.extend(["PortfolioManager", "AssetAllocator"])
        
        elif request_type == RequestType.SYSTEM:
            suggestions.extend(["SystemMonitor", "HealthChecker"])
        
        else:
            suggestions.extend(["GeneralAgent", "HelpSystem"])
        
        return suggestions


class RefactoredPromptRouter:
    """Refactored prompt router with modular agents and memory management."""
    
    def __init__(self):
        self.processor = EnhancedPromptProcessor()
        self.memory_manager = self.processor.memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.agents = [
            InvestmentAgent(),
            ForecastAgent(),
            StrategyAgent(),
            GeneralAgent()  # Fallback agent
        ]
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.last_weight_update = datetime.now()
        self.weight_update_interval = timedelta(hours=1)
    
    def handle_prompt(self, prompt: str, context: Optional[PromptContext] = None) -> Dict[str, Any]:
        """Handle a user prompt with intelligent routing."""
        start_time = datetime.now()
        
        try:
            # Process the prompt
            processed = self.processor.process_prompt(prompt, context)
            
            # Check if we should update weights
            if self._should_update_weights():
                self._update_agent_weights()
            
            # Find the best agent to handle the request
            best_agent = self._select_best_agent(prompt, processed)
            
            # Handle the prompt
            result = best_agent.handle(prompt, processed)
            
            # Update memory
            self._update_memory(prompt, processed, best_agent, result, start_time)
            
            # Check if we should search for better models
            if self.memory_manager.should_search_better_models(best_agent.name):
                result["model_search_recommended"] = True
                result["search_reason"] = "Low performance detected"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling prompt: {e}")
            return {
                "success": False,
                "message": f"Error processing prompt: {str(e)}",
                "agent_used": "ErrorHandler",
                "fallback_used": True,
            }
    
    def _select_best_agent(self, prompt: str, processed: ProcessedPrompt) -> BasePromptAgent:
        """Select the best agent based on prompt and performance."""
        # First, find agents that can handle this prompt
        capable_agents = [agent for agent in self.agents if agent.can_handle(prompt, processed)]
        
        if not capable_agents:
            # Fallback to general agent
            return next(agent for agent in self.agents if isinstance(agent, GeneralAgent))
        
        # Select agent based on performance weights
        agent_scores = []
        for agent in capable_agents:
            # Get current performance
            performance = self.memory_manager.get_agent_performance(agent.name)
            agent.performance = performance
            
            # Calculate score based on performance and confidence
            score = (performance.success_rate * 0.4 + 
                    performance.user_satisfaction * 0.3 + 
                    processed.confidence * 0.3)
            
            # Penalize slow response times
            if performance.avg_response_time > 2.0:
                score *= 0.8
            
            agent_scores.append((agent, score))
        
        # Return the agent with the highest score
        return max(agent_scores, key=lambda x: x[1])[0]
    
    def _should_update_weights(self) -> bool:
        """Check if agent weights should be updated."""
        return datetime.now() - self.last_weight_update > self.weight_update_interval
    
    def _update_agent_weights(self):
        """Update agent weights based on recent performance."""
        self.logger.info("Updating agent weights based on performance")
        
        for agent in self.agents:
            performance = self.memory_manager.get_agent_performance(agent.name)
            agent.performance = performance
            
            # Update weight based on performance
            new_weight = agent.get_weight()
            agent.performance.weight = new_weight
            
            self.logger.info(f"Agent {agent.name}: weight={new_weight:.3f}, "
                           f"success_rate={performance.success_rate:.3f}")
        
        self.last_weight_update = datetime.now()
    
    def _update_memory(self, prompt: str, processed: ProcessedPrompt, 
                      agent: BasePromptAgent, result: Dict[str, Any], start_time: datetime):
        """Update memory with the interaction."""
        import hashlib
        
        prompt_hash = hashlib.md5(prompt.lower().encode()).hexdigest()
        response_time = (datetime.now() - start_time).total_seconds()
        
        memory = PromptMemory(
            prompt_hash=prompt_hash,
            original_prompt=prompt,
            request_type=processed.request_type,
            agent_used=agent.name,
            success=result.get("success", False),
            response_time=response_time,
            user_feedback=None,  # Would be updated if user provides feedback
            parameters=processed.extracted_parameters
        )
        
        self.memory_manager.add_memory(memory)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a performance report for all agents."""
        report = {
            "agents": {},
            "overall_stats": {
                "total_requests": 0,
                "avg_success_rate": 0.0,
                "avg_response_time": 0.0
            }
        }
        
        total_requests = 0
        total_success_rate = 0.0
        total_response_time = 0.0
        agent_count = 0
        
        for agent in self.agents:
            performance = self.memory_manager.get_agent_performance(agent.name)
            
            report["agents"][agent.name] = {
                "success_rate": performance.success_rate,
                "avg_response_time": performance.avg_response_time,
                "user_satisfaction": performance.user_satisfaction,
                "total_requests": performance.total_requests,
                "weight": performance.weight,
                "last_updated": performance.last_updated.isoformat()
            }
            
            if performance.total_requests > 0:
                total_requests += performance.total_requests
                total_success_rate += performance.success_rate
                total_response_time += performance.avg_response_time
                agent_count += 1
        
        if agent_count > 0:
            report["overall_stats"]["total_requests"] = total_requests
            report["overall_stats"]["avg_success_rate"] = total_success_rate / agent_count
            report["overall_stats"]["avg_response_time"] = total_response_time / agent_count
        
        return report
    
    def run(self, prompt: str, *args, **kwargs) -> Dict[str, Any]:
        """Main run method for the router."""
        return self.handle_prompt(prompt)


# Convenience function to get the router instance
def get_prompt_router() -> RefactoredPromptRouter:
    """Get a singleton instance of the refactored prompt router."""
    if not hasattr(get_prompt_router, "_instance"):
        get_prompt_router._instance = RefactoredPromptRouter()
    return get_prompt_router._instance 