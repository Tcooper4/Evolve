"""
Prompt Router Module

This module handles prompt processing and routing for the Evolve Trading Platform:
- Natural language prompt analysis
- Request classification and routing
- Prompt validation and preprocessing
- Context management and enhancement
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

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

@dataclass
class PromptContext:
    """Context information for prompt processing."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    previous_requests: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedPrompt:
    """Processed prompt information."""
    original_prompt: str
    request_type: RequestType
    confidence: float
    extracted_parameters: Dict[str, Any]
    context: PromptContext
    routing_suggestions: List[str]
    processing_time: float

class PromptProcessor:
    """Processes and analyzes user prompts."""
    
    def __init__(self):
        """Initialize the prompt processor."""
        self.logger = logging.getLogger(__name__)
        
        # Classification patterns
        self.classification_patterns = {
            RequestType.FORECAST: [
                r'\b(forecast|predict|future|next|upcoming|tomorrow|next week|next month)\b',
                r'\b(price|stock|market|trend|movement|direction)\b',
                r'\b(how much|what will|when will|where will)\b'
            ],
            RequestType.STRATEGY: [
                r'\b(strategy|trading|signal|entry|exit|position)\b',
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
        
        # Parameter extraction patterns
        self.parameter_patterns = {
            'symbol': r'\b([A-Z]{1,5})\b',
            'timeframe': r'\b(1m|5m|15m|30m|1h|4h|1d|1w|1M)\b',
            'days': r'\b(\d+)\s*(days?|d)\b',
            'model': r'\b(lstm|arima|xgboost|prophet|ensemble|transformer)\b',
            'strategy': r'\b(rsi|macd|bollinger|sma|ema|custom)\b',
            'risk_level': r'\b(low|medium|high|conservative|aggressive)\b'
        }
    
    def process_prompt(self, prompt: str, context: Optional[PromptContext] = None) -> ProcessedPrompt:
        """
        Process a user prompt and extract information.
        
        Args:
            prompt: User's input prompt
            context: Optional context information
            
        Returns:
            ProcessedPrompt: Processed prompt information
        """
        start_time = datetime.now()
        
        if context is None:
            context = PromptContext()
        
        # Classify request type
        request_type = self._classify_request(prompt)
        
        # Extract parameters
        extracted_parameters = self._extract_parameters(prompt)
        
        # Calculate confidence
        confidence = self._calculate_confidence(prompt, request_type)
        
        # Generate routing suggestions
        routing_suggestions = self._generate_routing_suggestions(request_type, extracted_parameters)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        processed_prompt = ProcessedPrompt(
            original_prompt=prompt,
            request_type=request_type,
            confidence=confidence,
            extracted_parameters=extracted_parameters,
            context=context,
            routing_suggestions=routing_suggestions,
            processing_time=processing_time
        )
        
        self.logger.info(f"Processed prompt: {request_type.value} (confidence: {confidence:.2f})")
        return processed_prompt
    
    def _classify_request(self, prompt: str) -> RequestType:
        """
        Classify the type of request.
        
        Args:
            prompt: User prompt
            
        Returns:
            RequestType: Classified request type
        """
        prompt_lower = prompt.lower()
        scores = {}
        
        for request_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower)
                score += len(matches)
            scores[request_type] = score
        
        # Find the request type with the highest score
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return RequestType.UNKNOWN
    
    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """
        Extract parameters from the prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dict: Extracted parameters
        """
        parameters = {}
        prompt_lower = prompt.lower()
        
        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, prompt_lower)
            if matches:
                if param_name == 'days':
                    # Extract the number
                    numbers = re.findall(r'\d+', str(matches[0]))
                    if numbers:
                        parameters[param_name] = int(numbers[0])
                else:
                    parameters[param_name] = matches[0] if len(matches) == 1 else matches
        
        return parameters
    
    def _calculate_confidence(self, prompt: str, request_type: RequestType) -> float:
        """
        Calculate confidence in the classification.
        
        Args:
            prompt: User prompt
            request_type: Classified request type
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if request_type == RequestType.UNKNOWN:
            return 0.0
        
        prompt_lower = prompt.lower()
        patterns = self.classification_patterns[request_type]
        
        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, prompt_lower)
            total_matches += len(matches)
        
        # Normalize by prompt length and pattern count
        confidence = min(1.0, total_matches / max(1, len(prompt.split())))
        
        return confidence
    
    def _generate_routing_suggestions(self, request_type: RequestType, 
                                    parameters: Dict[str, Any]) -> List[str]:
        """
        Generate routing suggestions based on request type and parameters.
        
        Args:
            request_type: Classified request type
            parameters: Extracted parameters
            
        Returns:
            List[str]: Suggested routing targets
        """
        suggestions = []
        
        if request_type == RequestType.FORECAST:
            suggestions.extend(['ModelSelectorAgent', 'ForecastEngine'])
            if 'model' in parameters:
                suggestions.append(f"{parameters['model'].title()}Model")
        
        elif request_type == RequestType.STRATEGY:
            suggestions.extend(['StrategySelectorAgent', 'StrategyEngine'])
            if 'strategy' in parameters:
                suggestions.append(f"{parameters['strategy'].title()}Strategy")
        
        elif request_type == RequestType.ANALYSIS:
            suggestions.extend(['MarketAnalyzerAgent', 'AnalysisEngine'])
        
        elif request_type == RequestType.OPTIMIZATION:
            suggestions.extend(['MetaTunerAgent', 'OptimizationEngine'])
        
        elif request_type == RequestType.PORTFOLIO:
            suggestions.extend(['PortfolioManagerAgent', 'PortfolioEngine'])
        
        elif request_type == RequestType.SYSTEM:
            suggestions.extend(['SystemMonitorAgent', 'SystemEngine'])
        
        # Add general suggestions
        suggestions.extend(['PromptRouterAgent', 'AgentHub'])
        
        return list(set(suggestions))  # Remove duplicates
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Validate a prompt for processing.
        
        Args:
            prompt: User prompt
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        if not prompt or not prompt.strip():
            errors.append("Prompt cannot be empty")
        
        if len(prompt) > 1000:
            errors.append("Prompt too long (max 1000 characters)")
        
        if len(prompt.split()) < 2:
            errors.append("Prompt too short (minimum 2 words)")
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(delete|remove|drop|truncate)\b',
            r'\b(password|secret|key)\b',
            r'\b(exec|eval|system)\b'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, prompt.lower()):
                errors.append("Prompt contains potentially harmful content")
                break
        
        return len(errors) == 0, errors
    
    def enhance_prompt(self, prompt: str, context: PromptContext) -> str:
        """
        Enhance a prompt with context information.
        
        Args:
            prompt: Original prompt
            context: Context information
            
        Returns:
            str: Enhanced prompt
        """
        enhanced_prompt = prompt
        
        # Add user preferences if available
        if context.user_preferences:
            preferences_str = ", ".join([f"{k}: {v}" for k, v in context.user_preferences.items()])
            enhanced_prompt += f" [User preferences: {preferences_str}]"
        
        # Add system state if available
        if context.system_state:
            state_str = ", ".join([f"{k}: {v}" for k, v in context.system_state.items()])
            enhanced_prompt += f" [System state: {state_str}]"
        
        return enhanced_prompt

def get_prompt_processor() -> PromptProcessor:
    """
    Get a prompt processor instance.
    
    Returns:
        PromptProcessor: Configured prompt processor
    """
    return PromptProcessor() 