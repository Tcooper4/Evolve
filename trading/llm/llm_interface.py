"""Advanced LLM interface for trading system with robust prompt processing and context management."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import deque
import re
import os
import random
from jinja2 import Template
import openai
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache

# Constants
DEFAULT_MODEL = "gpt2"
DEFAULT_MAX_HISTORY = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

@dataclass
class PromptContext:
    """Context management for prompts with history and metadata."""
    history: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_MAX_HISTORY))
    max_history: int = DEFAULT_MAX_HISTORY
    current_intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    last_ticker: Optional[str] = None
    last_timeframe: Optional[str] = None
    last_strategy: Optional[str] = None
    last_analysis: Optional[Dict[str, Any]] = None
    last_signals: Optional[Dict[str, Any]] = None
    last_risk_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            'history': list(self.history),
            'max_history': self.max_history,
            'current_intent': self.current_intent,
            'entities': self.entities,
            'confidence': self.confidence,
            'last_ticker': self.last_ticker,
            'last_timeframe': self.last_timeframe,
            'last_strategy': self.last_strategy,
            'last_analysis': self.last_analysis,
            'last_signals': self.last_signals,
            'last_risk_metrics': self.last_risk_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptContext':
        """Create context from dictionary."""
        history = deque(data.get('history', []), maxlen=data.get('max_history', DEFAULT_MAX_HISTORY))
        return cls(
            history=history,
            max_history=data.get('max_history', DEFAULT_MAX_HISTORY),
            current_intent=data.get('current_intent'),
            entities=data.get('entities', {}),
            confidence=data.get('confidence', 0.0),
            last_ticker=data.get('last_ticker'),
            last_timeframe=data.get('last_timeframe'),
            last_strategy=data.get('last_strategy'),
            last_analysis=data.get('last_analysis'),
            last_signals=data.get('last_signals'),
            last_risk_metrics=data.get('last_risk_metrics')
        )
    
    def add_to_history(self, prompt: str, response: str) -> None:
        """Add prompt and response to history."""
        self.history.append({
            'prompt': prompt,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    def clear_history(self) -> None:
        """Clear the history."""
        self.history.clear()

class PromptProcessor:
    """Advanced prompt processing with intent and entity extraction."""
    
    def __init__(self):
        """Initialize the prompt processor with enhanced patterns."""
        self.intent_patterns = {
            'forecast': r'(forecast|predict|outlook|projection)',
            'analyze': r'(analyze|analysis|examine|study)',
            'recommend': r'(recommend|suggest|advise|propose)',
            'explain': r'(explain|describe|detail|elaborate)',
            'backtest': r'(backtest|backtesting|historical\s+test)',
            'optimize': r'(optimize|optimization|tune|improve)',
            'compare': r'(compare|comparison|versus|vs)',
            'report': r'(report|summary|overview|status)',
            'visualize': r'(visualize|plot|chart|graph)',
            'risk': r'(risk|volatility|exposure|hedge)',
            'portfolio': r'(portfolio|allocation|diversification|rebalance)'
        }
        
        self.entity_patterns = {
            'ticker': r'\b[A-Z]{1,5}\b',  # Stock ticker symbols
            'timeframe': r'(daily|weekly|monthly|quarterly|yearly|\d+\s*(day|week|month|year)s?)',
            'metric': r'(price|volume|return|volatility|trend|sharpe|sortino|alpha|beta)',
            'strategy_name': r'(momentum|mean\s+reversion|ml\s+based|hybrid|adaptive)',
            'date_range': r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})',
            'confidence': r'(high|medium|low)\s+confidence',
            'action': r'(buy|sell|hold|trade|invest|exit|enter)',
            'risk_level': r'(low|medium|high)\s+risk',
            'position_size': r'(\d+%|\d+\s*(shares|contracts))',
            'stop_loss': r'(stop\s+loss|stop\s+limit|trailing\s+stop)',
            'take_profit': r'(take\s+profit|profit\s+target)'
        }
        
        # Compile patterns for better performance
        self.compiled_intent_patterns = {
            intent: re.compile(pattern, re.IGNORECASE)
            for intent, pattern in self.intent_patterns.items()
        }
        
        self.compiled_entity_patterns = {
            entity: re.compile(pattern, re.IGNORECASE)
            for entity, pattern in self.entity_patterns.items()
        }
        
        # Initialize cache for pattern matching
        self._pattern_cache: Dict[str, List[str]] = {}
    
    @lru_cache(maxsize=1000)
    def extract_intent(self, text: str) -> Tuple[str, float]:
        """Extract intent with confidence scoring.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        text = text.lower()
        best_intent = 'explain'  # Default fallback
        best_score = 0.0
        
        for intent, pattern in self.compiled_intent_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Calculate score based on number and position of matches
                score = len(matches) * 0.3  # Base score for matches
                
                # Bonus for matches at start of text
                if text.startswith(matches[0]):
                    score += 0.2
                
                # Bonus for multiple matches
                if len(matches) > 1:
                    score += 0.1
                
                # Bonus for specific intent patterns
                if intent in ['forecast', 'analyze', 'backtest']:
                    score += 0.1
                
                if score > best_score:
                    best_score = score
                    best_intent = intent
        
        return best_intent, min(best_score, 1.0)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities with enhanced pattern matching.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {}
        text = text.lower()
        
        # Check cache first
        if text in self._pattern_cache:
            return self._pattern_cache[text]
        
        for entity_type, pattern in self.compiled_entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches
        
        # Cache the result
        self._pattern_cache[text] = entities
        return entities
    
    def validate_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate and clean extracted entities.
        
        Args:
            entities: Dictionary of extracted entities
            
        Returns:
            Cleaned and validated entities
        """
        validated = {}
        
        for entity_type, values in entities.items():
            if entity_type == 'ticker':
                # Validate ticker symbols
                validated[entity_type] = [v.upper() for v in values if v.isalpha() and len(v) <= 5]
            elif entity_type == 'timeframe':
                # Normalize timeframe values
                validated[entity_type] = [self._normalize_timeframe(v) for v in values]
            elif entity_type == 'date_range':
                # Validate and normalize dates
                validated[entity_type] = [self._normalize_date(v) for v in values if self._is_valid_date(v)]
            else:
                validated[entity_type] = values
        
        return validated
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe string to standard format."""
        timeframe = timeframe.lower()
        if 'day' in timeframe:
            return 'daily'
        elif 'week' in timeframe:
            return 'weekly'
        elif 'month' in timeframe:
            return 'monthly'
        elif 'quarter' in timeframe:
            return 'quarterly'
        elif 'year' in timeframe:
            return 'yearly'
        return timeframe
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Check if a string is a valid date."""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            try:
                datetime.strptime(date_str, '%m/%d/%Y')
                return True
            except ValueError:
                return False
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format."""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            date = datetime.strptime(date_str, '%m/%d/%Y')
        return date.strftime('%Y-%m-%d')

class LLMInterface:
    """Advanced LLM interface with multi-provider support and robust error handling."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        """Initialize the LLM interface with enhanced configuration.
        
        Args:
            api_key: Optional API key for OpenAI
            model_name: Name of the HuggingFace model to use
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize components
        self.prompt_processor = PromptProcessor()
        self.context = PromptContext()
        
        # Load models
        self._load_models(model_name)
        
        # Configure API access
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.is_configured = bool(self.api_key)
        
        if self.is_configured:
            openai.api_key = self.api_key
            self.logger.info("OpenAI API configured successfully")
        
        # Load templates
        self._load_templates()
        
        # Initialize thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("LLMInterface initialized successfully")
    
    def _load_models(self, model_name: str) -> None:
        """Load and initialize language models.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.logger.info(f"Loaded models: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def _load_templates(self) -> None:
        """Load prompt templates from configuration."""
        template_dir = Path(__file__).parent / "templates"
        self.templates = {}
        
        try:
            for template_file in template_dir.glob("*.j2"):
                with open(template_file, "r") as f:
                    self.templates[template_file.stem] = Template(f.read())
        except Exception as e:
            self.logger.error(f"Error loading templates: {e}")
    
    async def process_prompt_async(self, prompt: str) -> Dict[str, Any]:
        """Process a natural language prompt asynchronously.
        
        Args:
            prompt: Input prompt to process
            
        Returns:
            Dictionary containing processed results
        """
        try:
            self.logger.info(f"Processing prompt asynchronously: {prompt}")
            
            # Extract intent and entities
            intent, confidence = self.prompt_processor.extract_intent(prompt)
            entities = self.prompt_processor.extract_entities(prompt)
            entities = self.prompt_processor.validate_entities(entities)
            
            # Update context
            self._update_context(intent, entities, confidence)
            
            # Generate response asynchronously
            if self.is_configured:
                response = await self._generate_openai_response_async(prompt, intent, entities)
            else:
                response = await self._generate_huggingface_response_async(prompt, intent, entities)
            
            # Structure the response
            result = {
                'content': response,
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add metadata
            if hasattr(self, 'context'):
                result['context'] = self.context.to_dict()
            
            # Add to history
            self.context.add_to_history(prompt, response)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'type': 'error',
                'timestamp': datetime.now().isoformat()
            }
    
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a natural language prompt with enhanced context awareness.
        
        Args:
            prompt: Input prompt to process
            
        Returns:
            Dictionary containing processed results
        """
        try:
            self.logger.info(f"Processing prompt: {prompt}")
            
            # Extract intent and entities
            intent, confidence = self.prompt_processor.extract_intent(prompt)
            entities = self.prompt_processor.extract_entities(prompt)
            entities = self.prompt_processor.validate_entities(entities)
            
            # Update context
            self._update_context(intent, entities, confidence)
            
            # Generate response
            if self.is_configured:
                response = self._generate_openai_response(prompt, intent, entities)
            else:
                response = self._generate_huggingface_response(prompt, intent, entities)
            
            # Structure the response
            result = {
                'content': response,
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add metadata
            if hasattr(self, 'context'):
                result['context'] = self.context.to_dict()
            
            # Add to history
            self.context.add_to_history(prompt, response)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'type': 'error',
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_context(self, intent: str, entities: Dict[str, List[str]], confidence: float) -> None:
        """Update context with new information.
        
        Args:
            intent: Detected intent
            entities: Extracted entities
            confidence: Confidence score
        """
        self.context.current_intent = intent
        self.context.entities = entities
        self.context.confidence = confidence
        
        # Update last known values
        if 'ticker' in entities:
            self.context.last_ticker = entities['ticker'][0]
        if 'timeframe' in entities:
            self.context.last_timeframe = entities['timeframe'][0]
        if 'strategy_name' in entities:
            self.context.last_strategy = entities['strategy_name'][0]
    
    async def _generate_openai_response_async(self, prompt: str, intent: str, entities: Dict) -> str:
        """Generate response using OpenAI API asynchronously.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Generated response text
        """
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self._get_system_prompt(intent)},
                {"role": "user", "content": self._prepare_prompt(prompt, intent, entities)}
            ]
            
            # Get response asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.thread_pool,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS
                )
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_openai_response(self, prompt: str, intent: str, entities: Dict) -> str:
        """Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Generated response text
        """
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": self._get_system_prompt(intent)},
                {"role": "user", "content": self._prepare_prompt(prompt, intent, entities)}
            ]
            
            # Get response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_response(prompt)
    
    async def _generate_huggingface_response_async(self, prompt: str, intent: str, entities: Dict) -> str:
        """Generate response using HuggingFace model asynchronously.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Generated response text
        """
        try:
            # Prepare input
            inputs = self.tokenizer(
                self._prepare_prompt(prompt, intent, entities),
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Generate response asynchronously
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                self.thread_pool,
                lambda: self.model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P,
                    do_sample=True
                )
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"HuggingFace model error: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_huggingface_response(self, prompt: str, intent: str, entities: Dict) -> str:
        """Generate response using HuggingFace model.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Generated response text
        """
        try:
            # Prepare input
            inputs = self.tokenizer(
                self._prepare_prompt(prompt, intent, entities),
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                do_sample=True
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            self.logger.error(f"HuggingFace model error: {e}")
            return self._generate_fallback_response(prompt)
    
    def _prepare_prompt(self, prompt: str, intent: str, entities: Dict) -> str:
        """Prepare prompt with context and templates.
        
        Args:
            prompt: Input prompt
            intent: Detected intent
            entities: Extracted entities
            
        Returns:
            Prepared prompt text
        """
        # Get template for intent
        template = self.templates.get(intent, self.templates.get('default'))
        
        if template:
            return template.render(
                prompt=prompt,
                intent=intent,
                entities=entities,
                context=self.context.to_dict()
            )
        return prompt
    
    def _get_system_prompt(self, intent: str) -> str:
        """Get system prompt based on intent.
        
        Args:
            intent: Detected intent
            
        Returns:
            System prompt text
        """
        prompts = {
            'forecast': "You are a market forecasting expert. Provide clear, data-driven predictions.",
            'analyze': "You are a market analysis expert. Provide detailed technical and fundamental analysis.",
            'backtest': "You are a backtesting expert. Provide comprehensive strategy evaluation.",
            'optimize': "You are an optimization expert. Provide portfolio and strategy optimization advice.",
            'explain': "You are a trading education expert. Provide clear, detailed explanations.",
            'risk': "You are a risk management expert. Provide detailed risk analysis and mitigation strategies.",
            'portfolio': "You are a portfolio management expert. Provide comprehensive portfolio analysis and recommendations."
        }
        return prompts.get(intent, "You are a trading assistant. Provide helpful, accurate responses.")
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when primary methods fail.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Fallback response text
        """
        return f"I apologize, but I'm having trouble processing your request: {prompt}. Please try rephrasing or ask for something else."
    
    def set_prompts(self, prompts: Dict[str, str]) -> None:
        """Set custom system prompts.
        
        Args:
            prompts: Dictionary of intent to prompt mapping
        """
        self.custom_prompts = prompts
    
    def get_llm_metrics(self) -> Dict[str, Any]:
        """Get LLM performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            'model_name': self.model.config.model_type if hasattr(self, 'model') else None,
            'is_openai_configured': self.is_configured,
            'context_size': len(self.context.history),
            'last_intent': self.context.current_intent,
            'last_confidence': self.context.confidence,
            'last_ticker': self.context.last_ticker,
            'last_timeframe': self.context.last_timeframe,
            'last_strategy': self.context.last_strategy
        }
    
    def clear_context(self) -> None:
        """Clear the context history."""
        self.context.clear_history()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True) 