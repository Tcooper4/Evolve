"""
Prompt Clarification Agent

Agent that detects ambiguous prompts and asks for clarification from users.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AmbiguityType(Enum):
    """Types of prompt ambiguity."""
    MULTIPLE_STRATEGIES = "multiple_strategies"
    VAGUE_REQUEST = "vague_request"
    CONTRADICTORY_INSTRUCTIONS = "contradictory_instructions"
    MISSING_CONTEXT = "missing_context"
    UNCLEAR_TIMEFRAME = "unclear_timeframe"
    AMBIGUOUS_SYMBOL = "ambiguous_symbol"


@dataclass
class ClarificationRequest:
    """Clarification request with options."""
    ambiguity_type: AmbiguityType
    original_prompt: str
    clarification_question: str
    options: List[str]
    confidence: float
    reasoning: str


class PromptClarificationAgent:
    """Agent that detects and resolves ambiguous prompts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the prompt clarification agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.ambiguity_threshold = self.config.get("ambiguity_threshold", 0.7)
        self.strategy_keywords = self.config.get("strategy_keywords", {
            "rsi": ["rsi", "relative strength index", "oversold", "overbought"],
            "macd": ["macd", "moving average convergence divergence", "crossover"],
            "bollinger": ["bollinger", "bands", "volatility", "squeeze"],
            "sma": ["sma", "simple moving average", "moving average"],
            "ema": ["ema", "exponential moving average"],
            "momentum": ["momentum", "velocity", "acceleration"]
        })
        self.timeframe_keywords = self.config.get("timeframe_keywords", {
            "short": ["1m", "5m", "15m", "30m", "1h", "short term", "intraday"],
            "medium": ["4h", "1d", "daily", "medium term"],
            "long": ["1w", "1M", "weekly", "monthly", "long term"]
        })
        
    def analyze_prompt(self, prompt: str) -> Optional[ClarificationRequest]:
        """
        Analyze a prompt for ambiguity and generate clarification request if needed.
        
        Args:
            prompt: User prompt to analyze
            
        Returns:
            ClarificationRequest if ambiguity detected, None otherwise
        """
        prompt_lower = prompt.lower()
        
        # Check for multiple strategy mentions
        strategy_ambiguity = self._detect_strategy_ambiguity(prompt_lower)
        if strategy_ambiguity:
            return strategy_ambiguity
            
        # Check for vague requests
        vagueness = self._detect_vague_request(prompt_lower)
        if vagueness:
            return vagueness
            
        # Check for contradictory instructions
        contradiction = self._detect_contradictory_instructions(prompt_lower)
        if contradiction:
            return contradiction
            
        # Check for missing context
        missing_context = self._detect_missing_context(prompt_lower)
        if missing_context:
            return missing_context
            
        # Check for unclear timeframe
        timeframe_ambiguity = self._detect_timeframe_ambiguity(prompt_lower)
        if timeframe_ambiguity:
            return timeframe_ambiguity
            
        # Check for ambiguous symbol references
        symbol_ambiguity = self._detect_symbol_ambiguity(prompt_lower)
        if symbol_ambiguity:
            return symbol_ambiguity
            
        return None
        
    def _detect_strategy_ambiguity(self, prompt: str) -> Optional[ClarificationRequest]:
        """Detect when multiple strategies are mentioned."""
        detected_strategies = []
        
        for strategy_name, keywords in self.strategy_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    detected_strategies.append(strategy_name)
                    break
                    
        if len(detected_strategies) > 1:
            question = f"Did you mean {detected_strategies[0].upper()} strategy or {detected_strategies[1].upper()}?"
            if len(detected_strategies) > 2:
                question = f"Multiple strategies detected: {', '.join(detected_strategies)}. Which one would you like to use?"
                
            return ClarificationRequest(
                ambiguity_type=AmbiguityType.MULTIPLE_STRATEGIES,
                original_prompt=prompt,
                clarification_question=question,
                options=detected_strategies,
                confidence=0.8,
                reasoning=f"Detected {len(detected_strategies)} different strategies: {detected_strategies}"
            )
            
        return None
        
    def _detect_vague_request(self, prompt: str) -> Optional[ClarificationRequest]:
        """Detect vague or unclear requests."""
        vague_patterns = [
            r"\b(analyze|check|look at|examine)\b",
            r"\b(what about|how about)\b",
            r"\b(tell me|show me)\b"
        ]
        
        vague_count = 0
        for pattern in vague_patterns:
            if re.search(pattern, prompt):
                vague_count += 1
                
        if vague_count >= 2:
            return ClarificationRequest(
                ambiguity_type=AmbiguityType.VAGUE_REQUEST,
                original_prompt=prompt,
                clarification_question="Could you be more specific about what you'd like me to analyze?",
                options=["Price analysis", "Technical indicators", "Fundamental analysis", "Risk assessment"],
                confidence=0.6,
                reasoning="Request contains multiple vague terms without specific context"
            )
            
        return None
        
    def _detect_contradictory_instructions(self, prompt: str) -> Optional[ClarificationRequest]:
        """Detect contradictory instructions in the prompt."""
        contradictions = [
            (r"\b(buy|long)\b", r"\b(sell|short)\b"),
            (r"\b(conservative|safe)\b", r"\b(aggressive|risky)\b"),
            (r"\b(short term|quick)\b", r"\b(long term|hold)\b"),
            (r"\b(high frequency|frequent)\b", r"\b(infrequent|rare)\b")
        ]
        
        for pattern1, pattern2 in contradictions:
            if re.search(pattern1, prompt) and re.search(pattern2, prompt):
                return ClarificationRequest(
                    ambiguity_type=AmbiguityType.CONTRADICTORY_INSTRUCTIONS,
                    original_prompt=prompt,
                    clarification_question="I detected conflicting instructions. Could you clarify your preference?",
                    options=["Conservative approach", "Aggressive approach", "Balanced approach"],
                    confidence=0.9,
                    reasoning=f"Contradictory terms detected: {pattern1} vs {pattern2}"
                )
                
        return None
        
    def _detect_missing_context(self, prompt: str) -> Optional[ClarificationRequest]:
        """Detect when important context is missing."""
        missing_context_indicators = [
            (r"\b(forecast|predict)\b", "timeframe"),
            (r"\b(analyze|examine)\b", "analysis type"),
            (r"\b(optimize|improve)\b", "optimization target"),
            (r"\b(risk|volatility)\b", "risk tolerance")
        ]
        
        for pattern, context_type in missing_context_indicators:
            if re.search(pattern, prompt):
                # Check if specific context is provided
                if not self._has_specific_context(prompt, context_type):
                    return ClarificationRequest(
                        ambiguity_type=AmbiguityType.MISSING_CONTEXT,
                        original_prompt=prompt,
                        clarification_question=f"What {context_type} would you like me to consider?",
                        options=self._get_context_options(context_type),
                        confidence=0.7,
                        reasoning=f"Missing {context_type} context for {pattern} request"
                    )
                    
        return None
        
    def _detect_timeframe_ambiguity(self, prompt: str) -> Optional[ClarificationRequest]:
        """Detect unclear or conflicting timeframes."""
        detected_timeframes = []
        
        for timeframe_type, keywords in self.timeframe_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    detected_timeframes.append(timeframe_type)
                    break
                    
        if len(detected_timeframes) > 1:
            return ClarificationRequest(
                ambiguity_type=AmbiguityType.UNCLEAR_TIMEFRAME,
                original_prompt=prompt,
                clarification_question="I detected multiple timeframes. Which timeframe should I focus on?",
                options=detected_timeframes,
                confidence=0.8,
                reasoning=f"Multiple timeframes detected: {detected_timeframes}"
            )
            
        return None
        
    def _detect_symbol_ambiguity(self, prompt: str) -> Optional[ClarificationRequest]:
        """Detect ambiguous symbol references."""
        # Look for potential stock symbols (3-5 letter codes)
        symbols = re.findall(r'\b[A-Z]{3,5}\b', prompt.upper())
        
        if len(symbols) > 1:
            return ClarificationRequest(
                ambiguity_type=AmbiguityType.AMBIGUOUS_SYMBOL,
                original_prompt=prompt,
                clarification_question=f"Multiple symbols detected: {', '.join(symbols)}. Which one should I analyze?",
                options=symbols,
                confidence=0.7,
                reasoning=f"Multiple potential symbols found: {symbols}"
            )
            
        return None
        
    def _has_specific_context(self, prompt: str, context_type: str) -> bool:
        """Check if prompt has specific context for the given type."""
        context_patterns = {
            "timeframe": r"\b(1m|5m|15m|30m|1h|4h|1d|1w|1M|daily|weekly|monthly)\b",
            "analysis type": r"\b(technical|fundamental|sentiment|price|volume)\b",
            "optimization target": r"\b(returns|sharpe|drawdown|volatility|accuracy)\b",
            "risk tolerance": r"\b(low|medium|high|conservative|aggressive)\b"
        }
        
        pattern = context_patterns.get(context_type, "")
        return bool(re.search(pattern, prompt))
        
    def _get_context_options(self, context_type: str) -> List[str]:
        """Get options for missing context."""
        context_options = {
            "timeframe": ["Short term (1h-1d)", "Medium term (1d-1w)", "Long term (1w+)"],
            "analysis type": ["Technical analysis", "Fundamental analysis", "Sentiment analysis"],
            "optimization target": ["Maximize returns", "Minimize risk", "Balance risk/return"],
            "risk tolerance": ["Conservative", "Moderate", "Aggressive"]
        }
        
        return context_options.get(context_type, ["Option 1", "Option 2", "Option 3"])
        
    def generate_clarification_response(self, clarification: ClarificationRequest) -> str:
        """
        Generate a user-friendly clarification response.
        
        Args:
            clarification: Clarification request
            
        Returns:
            Formatted clarification message
        """
        response = f"I need some clarification about your request:\n\n"
        response += f"**{clarification.clarification_question}**\n\n"
        
        if clarification.options:
            response += "**Options:**\n"
            for i, option in enumerate(clarification.options, 1):
                response += f"{i}. {option}\n"
            response += "\n"
            
        response += f"*Confidence: {clarification.confidence:.1%}*\n"
        response += f"*Reasoning: {clarification.reasoning}*"
        
        return response
        
    def handle_user_clarification(self, original_prompt: str, user_response: str) -> str:
        """
        Handle user's clarification response and generate refined prompt.
        
        Args:
            original_prompt: Original ambiguous prompt
            user_response: User's clarification response
            
        Returns:
            Refined prompt with clarification incorporated
        """
        # Simple approach: append clarification to original prompt
        refined_prompt = f"{original_prompt} [Clarification: {user_response}]"
        
        logger.info(f"Refined prompt: {refined_prompt}")
        return refined_prompt
        
    def get_ambiguity_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected ambiguities."""
        return {
            "ambiguity_threshold": self.ambiguity_threshold,
            "strategy_keywords": len(self.strategy_keywords),
            "timeframe_keywords": len(self.timeframe_keywords),
            "supported_ambiguity_types": [t.value for t in AmbiguityType]
        } 