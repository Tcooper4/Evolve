import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from pathlib import Path

@dataclass
class EntityMatch:
    """Data class to hold entity match information."""
    entity_type: str
    value: str
    start: int
    end: int
    confidence: float

@dataclass
class ProcessedPrompt:
    """Data class to hold processed prompt information."""
    original_text: str
    entities: List[EntityMatch]
    intent: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Intent:
    name: str
    confidence: float
    entities: Dict[str, Any]
    context: Dict[str, Any]

class PromptProcessor:
    """Class to process natural language prompts and extract entities and intent."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the prompt processor.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "config"
        
        # Load entity patterns and intent patterns
        self.entity_patterns = self._load_entity_patterns()
        self.intent_patterns = self._load_intent_patterns()
        
        # Initialize context
        self.context = {}
        
    def _load_entity_patterns(self) -> Dict[str, List[str]]:
        """Load entity patterns from JSON file."""
        try:
            with open(self.config_dir / "entity_patterns.json", "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading entity patterns: {e}")
            return {}
            
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent patterns from JSON file."""
        try:
            with open(self.config_dir / "intent_patterns.json", "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading intent patterns: {e}")
            return {}
            
    def process_prompt(self, text: str) -> ProcessedPrompt:
        """Process a natural language prompt.
        
        Args:
            text: The input text to process
            
        Returns:
            ProcessedPrompt object containing extracted entities and intent
        """
        try:
            # Extract entities
            entities = self._extract_entities(text)
            
            # Determine intent
            intent, confidence = self._determine_intent(text, entities)
            
            return ProcessedPrompt(
                original_text=text,
                entities=entities,
                intent=intent,
                confidence=confidence
            )
        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}")
            return ProcessedPrompt(
                original_text=text,
                entities=[],
                intent="unknown",
                confidence=0.0
            )
            
    def _extract_entities(self, text: str) -> List[EntityMatch]:
        """Extract entities from text using regex patterns.
        
        Args:
            text: The input text to process
            
        Returns:
            List of EntityMatch objects
        """
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(EntityMatch(
                        entity_type=entity_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0  # Could be improved with ML-based confidence scoring
                    ))
                    
        # Sort entities by start position
        entities.sort(key=lambda x: x.start)
        
        return entities
        
    def _determine_intent(self, text: str, entities: List[EntityMatch]) -> Tuple[str, float]:
        """Determine the intent of the prompt.
        
        Args:
            text: The input text to process
            entities: List of extracted entities
            
        Returns:
            Tuple of (intent, confidence)
        """
        best_intent = "unknown"
        best_confidence = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Calculate confidence based on pattern match and entity presence
                    confidence = 0.5  # Base confidence for pattern match
                    
                    # Boost confidence if relevant entities are present
                    if intent == "forecast" and any(e.entity_type in ["timeframe", "asset"] for e in entities):
                        confidence += 0.3
                    elif intent == "analyze" and any(e.entity_type in ["asset", "indicator"] for e in entities):
                        confidence += 0.3
                    elif intent == "recommend" and any(e.entity_type in ["asset", "action"] for e in entities):
                        confidence += 0.3
                    elif intent == "explain" and any(e.entity_type in ["topic", "concept"] for e in entities):
                        confidence += 0.3
                        
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence
                        
        return best_intent, best_confidence
        
    def get_entity_values(self, prompt: ProcessedPrompt, entity_type: str) -> List[str]:
        """Get values for a specific entity type from a processed prompt.
        
        Args:
            prompt: ProcessedPrompt object
            entity_type: Type of entity to extract
            
        Returns:
            List of entity values
        """
        return [e.value for e in prompt.entities if e.entity_type == entity_type]
        
    def get_entity_by_type(self, prompt: ProcessedPrompt, entity_type: str) -> Optional[EntityMatch]:
        """Get the first entity of a specific type from a processed prompt.
        
        Args:
            prompt: ProcessedPrompt object
            entity_type: Type of entity to extract
            
        Returns:
            EntityMatch object or None if not found
        """
        for entity in prompt.entities:
            if entity.entity_type == entity_type:
                return entity
        return None
        
    def has_entity(self, prompt: ProcessedPrompt, entity_type: str) -> bool:
        """Check if a processed prompt contains a specific entity type.
        
        Args:
            prompt: ProcessedPrompt object
            entity_type: Type of entity to check for
            
        Returns:
            True if entity type is present, False otherwise
        """
        return any(e.entity_type == entity_type for e in prompt.entities)
        
    def get_required_entities(self, prompt: ProcessedPrompt, intent: str) -> List[str]:
        """Get list of required entities for a specific intent.
        
        Args:
            prompt: ProcessedPrompt object
            intent: Intent to check requirements for
            
        Returns:
            List of required entity types
        """
        required_entities = {
            "forecast": ["timeframe", "asset"],
            "analyze": ["asset"],
            "recommend": ["asset", "action"],
            "explain": ["topic"],
            "compare": ["asset"],
            "optimize": ["strategy"],
            "validate": ["test"],
            "monitor": ["asset"]
        }
        
        return required_entities.get(intent, [])
        
    def validate_prompt(self, prompt: ProcessedPrompt) -> Tuple[bool, List[str]]:
        """Validate if a processed prompt has all required entities for its intent.
        
        Args:
            prompt: ProcessedPrompt object to validate
            
        Returns:
            Tuple of (is_valid, missing_entities)
        """
        required_entities = self.get_required_entities(prompt, prompt.intent)
        missing_entities = []
        
        for entity_type in required_entities:
            if not self.has_entity(prompt, entity_type):
                missing_entities.append(entity_type)
                
        return len(missing_entities) == 0, missing_entities
    
    def _update_context(self, intent: str, entities: Dict[str, List[str]]):
        """Update context with new information."""
        self.context.update({
            'last_intent': intent,
            'last_entities': entities,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self.context
    
    def clear_context(self):
        """Clear current context."""
        self.context = {}
    
    def format_response(self, intent: Intent, response: str) -> str:
        """Format response based on intent and context."""
        try:
            if intent.name == "forecast":
                return self._format_forecast_response(intent, response)
            elif intent.name == "analyze":
                return self._format_analysis_response(intent, response)
            elif intent.name == "recommend":
                return self._format_recommendation_response(intent, response)
            elif intent.name == "explain":
                return self._format_explanation_response(intent, response)
            else:
                return response
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}")
            return response
    
    def _format_forecast_response(self, intent: Intent, response: str) -> str:
        """Format forecast response."""
        timeframe = intent.entities.get('timeframe', ['unknown timeframe'])[0]
        return f"Forecast for {timeframe}:\n{response}"
    
    def _format_analysis_response(self, intent: Intent, response: str) -> str:
        """Format analysis response."""
        metric = intent.entities.get('metric', ['current situation'])[0]
        return f"Analysis of {metric}:\n{response}"
    
    def _format_recommendation_response(self, intent: Intent, response: str) -> str:
        """Format recommendation response."""
        action = intent.entities.get('action', ['action'])[0]
        return f"Recommendation for {action}:\n{response}"
    
    def _format_explanation_response(self, intent: Intent, response: str) -> str:
        """Format explanation response."""
        return f"Explanation:\n{response}" 