"""
Enhanced LLM Agent with full trading pipeline routing.

This module provides a comprehensive LLM agent that can handle various
trading-related prompts and route them through the appropriate components
of the trading pipeline.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

# Import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Prompt examples will be disabled.")

# Import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available. Token counting will be disabled.")

from trading.agents.forecast_router import ForecastRouter
from trading.agents.strategy_gatekeeper import StrategyGatekeeper
from trading.agents.trade_execution import TradeExecutionSimulator
from trading.agents.self_tuning_optimizer import SelfTuningOptimizer
from trading.data.fallback_data_provider import FallbackDataProvider

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for LLM agents."""

    name: str
    role: str
    model_name: str
    max_tokens: int = 500
    temperature: float = 0.7
    memory_enabled: bool = True
    tools_enabled: bool = True


class LLMAgent:
    """LLM Agent for processing prompts with tools and memory."""

    def __init__(
        self,
        config: AgentConfig,
        model_loader=None,
        memory_manager=None,
        tool_registry=None,
    ):
        """Initialize LLM agent."""
        self.config = config
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        self.metrics = {
            "prompts_processed": 0,
            "tokens_used": 0,
            "tool_calls": 0,
            "memory_hits": 0,
        }

    async def process_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process a prompt asynchronously."""
        self.metrics["prompts_processed"] += 1

        # Simple placeholder implementation
        return {
            "content": f"Processed prompt: {prompt}",
            "metadata": {
                "tokens": len(prompt.split()),
                "tool_calls": 0,
                "memory_hits": 0,
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self.metrics = {
            "prompts_processed": 0,
            "tokens_used": 0,
            "tool_calls": 0,
            "memory_hits": 0,
        }


@dataclass
class AgentResponse:
    """Response from the prompt agent."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Any]] = None
    recommendations: Optional[List[str]] = None
    next_actions: Optional[List[str]] = None


class PromptAgent:
    """Enhanced prompt agent with full trading pipeline routing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize prompt agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize prompt examples system
        self.prompt_examples = self._load_prompt_examples()
        self.sentence_transformer = None
        self.example_embeddings = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.prompt_examples:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                self.example_embeddings = self._compute_example_embeddings()
                logger.info("Prompt examples system initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize sentence transformer: {e}")
        else:
            logger.info("Prompt examples system disabled (SentenceTransformers not available)")

        # Initialize token usage tracking
        self.token_usage = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_count": 0,
            "model_costs": {
                "gpt-4": 0.03,  # per 1K tokens
                "gpt-3.5-turbo": 0.002,
                "claude-3": 0.015,
                "default": 0.01
            }
        }

        # Initialize log batching
        self.log_buffer = deque(maxlen=100)
        self.last_log_flush = time.time()
        self.log_flush_interval = 60  # seconds

        # Initialize tiktoken for token counting
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
                logger.info("Token counting initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize tokenizer: {e}")
        else:
            logger.info("Token counting disabled (tiktoken not available)")

        # Initialize components
        self.forecast_router = ForecastRouter()

        # Initialize model creator for dynamic model generation
        try:
            from trading.agents.model_creator_agent import get_model_creator_agent

            self.model_creator = get_model_creator_agent()
            logger.info("Model creator agent initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize model creator: {e}")
            self.model_creator = None

        # Initialize prompt router for intelligent routing
        try:
            from trading.agents.prompt_router_agent import create_prompt_router

            self.prompt_router = create_prompt_router()
            logger.info("Prompt router agent initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize prompt router: {e}")
            self.prompt_router = None

        # Default strategy configurations
        default_strategies = {
            "RSI Mean Reversion": {
                "default_active": True,
                "preferred_regimes": ["neutral", "volatile"],
                "regime_weights": {"neutral": 1.0, "volatile": 0.8},
                "preferred_volatility": "medium",
                "volatility_range": [0.01, 0.03],
                "momentum_requirement": "any",
            },
            "Moving Average Crossover": {
                "default_active": True,
                "preferred_regimes": ["bull", "bear"],
                "regime_weights": {"bull": 1.0, "bear": 0.9},
                "preferred_volatility": "low",
                "volatility_range": [0.005, 0.02],
                "momentum_requirement": "positive",
            },
            "Bollinger Bands": {
                "default_active": True,
                "preferred_regimes": ["neutral", "volatile"],
                "regime_weights": {"neutral": 1.0, "volatile": 0.9},
                "preferred_volatility": "high",
                "volatility_range": [0.02, 0.05],
                "momentum_requirement": "any",
            },
        }
        self.strategy_gatekeeper = StrategyGatekeeper(default_strategies)
        self.trade_executor = TradeExecutionSimulator()
        self.optimizer = SelfTuningOptimizer()
        self.data_provider = FallbackDataProvider()

        # Strategy registry
        self.strategy_registry = {
            "rsi": "RSI Mean Reversion",
            "bollinger": "Bollinger Bands",
            "macd": "MACD Strategy",
            "sma": "Moving Average Crossover",
            "garch": "GARCH Volatility",
            "ridge": "Ridge Regression",
            "informer": "Informer Model",
            "transformer": "Transformer",
            "autoformer": "Autoformer",
            "lstm": "LSTM Strategy",
            "xgboost": "XGBoost Strategy",
            "ensemble": "Ensemble Strategy",
        }

        # Model registry
        self.model_registry = {
            "arima": "ARIMA",
            "lstm": "LSTM",
            "xgboost": "XGBoost",
            "prophet": "Prophet",
            "autoformer": "Autoformer",
            "transformer": "Transformer",
            "informer": "Informer",
            "garch": "GARCH",
            "ridge": "Ridge Regression",
        }

        logger.info("Enhanced Prompt Agent initialized with full pipeline routing")

    def _load_prompt_examples(self) -> Optional[Dict[str, Any]]:
        """Load prompt examples from JSON file.
        
        Returns:
            Dictionary containing prompt examples or None if file not found
        """
        try:
            examples_path = Path(__file__).parent / "prompt_examples.json"
            if examples_path.exists():
                with open(examples_path, 'r') as f:
                    examples = json.load(f)
                logger.info(f"Loaded {len(examples.get('examples', []))} prompt examples")
                return examples
            else:
                logger.warning("Prompt examples file not found")
                return None
        except Exception as e:
            logger.error(f"Error loading prompt examples: {e}")
            return None

    def _compute_example_embeddings(self) -> Optional[np.ndarray]:
        """Compute embeddings for all prompt examples.
        
        Returns:
            Numpy array of embeddings or None if computation fails
        """
        if not self.prompt_examples or not self.sentence_transformer:
            return None
            
        try:
            examples = self.prompt_examples.get('examples', [])
            prompts = [example['prompt'] for example in examples]
            embeddings = self.sentence_transformer.encode(prompts)
            logger.info(f"Computed embeddings for {len(prompts)} examples")
            return embeddings
        except Exception as e:
            logger.error(f"Error computing example embeddings: {e}")
            return None

    def estimate_token_usage(self, prompt: str, model: str = "gpt-4") -> Dict[str, Any]:
        """Estimate token usage and cost for a prompt.
        
        Args:
            prompt: Input prompt
            model: Model name for cost calculation
            
        Returns:
            Dictionary with token count and estimated cost
        """
        try:
            if not self.tokenizer:
                # Fallback estimation: ~4 characters per token
                token_count = len(prompt) // 4
            else:
                token_count = len(self.tokenizer.encode(prompt))
            
            # Get cost per 1K tokens
            cost_per_1k = self.token_usage["model_costs"].get(model, self.token_usage["model_costs"]["default"])
            estimated_cost = (token_count / 1000) * cost_per_1k
            
            return {
                "token_count": token_count,
                "estimated_cost": estimated_cost,
                "model": model,
                "cost_per_1k": cost_per_1k
            }
            
        except Exception as e:
            logger.warning(f"Error estimating token usage: {e}")
            return {
                "token_count": 0,
                "estimated_cost": 0.0,
                "model": model,
                "cost_per_1k": 0.0
            }

    def sanitize_prompt(self, prompt: str, max_length: int = 4000) -> str:
        """Sanitize and trim overly long prompts to prevent injection attacks.
        
        Args:
            prompt: Input prompt
            max_length: Maximum allowed length
            
        Returns:
            Sanitized prompt
        """
        try:
            # Remove potential injection patterns
            injection_patterns = [
                r'<script.*?</script>',
                r'javascript:',
                r'data:text/html',
                r'vbscript:',
                r'on\w+\s*=',
                r'<iframe',
                r'<object',
                r'<embed',
            ]
            
            sanitized = prompt
            for pattern in injection_patterns:
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            # Remove excessive whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            # Truncate if too long
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length] + "..."
                logger.warning(f"Prompt truncated from {len(prompt)} to {len(sanitized)} characters")
            
            # Log if significant changes were made
            if len(sanitized) != len(prompt):
                logger.info(f"Prompt sanitized: {len(prompt)} -> {len(sanitized)} characters")
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing prompt: {e}")
            return prompt[:max_length] if len(prompt) > max_length else prompt

    def batch_log(self, message: str, level: str = "info"):
        """Add log message to buffer for batch processing."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "level": level,
                "message": message
            }
            
            self.log_buffer.append(log_entry)
            
            # Flush buffer if it's full or enough time has passed
            current_time = time.time()
            if (len(self.log_buffer) >= 100 or 
                current_time - self.last_log_flush >= self.log_flush_interval):
                self._flush_log_buffer()
                
        except Exception as e:
            # Fallback to direct logging
            logger.error(f"Error in batch logging: {e}")
            logger.info(message)

    def _flush_log_buffer(self):
        """Flush the log buffer to actual logging."""
        try:
            if not self.log_buffer:
                return
            
            # Group logs by level
            logs_by_level = {}
            for entry in self.log_buffer:
                level = entry["level"]
                if level not in logs_by_level:
                    logs_by_level[level] = []
                logs_by_level[level].append(entry["message"])
            
            # Log grouped messages
            for level, messages in logs_by_level.items():
                if len(messages) == 1:
                    # Single message, log directly
                    getattr(logger, level)(messages[0])
                else:
                    # Multiple messages, log as batch
                    batch_message = f"Batch of {len(messages)} {level} messages: " + "; ".join(messages[:5])
                    if len(messages) > 5:
                        batch_message += f" ... and {len(messages) - 5} more"
                    getattr(logger, level)(batch_message)
            
            # Clear buffer
            self.log_buffer.clear()
            self.last_log_flush = time.time()
            
        except Exception as e:
            logger.error(f"Error flushing log buffer: {e}")

    def update_token_usage(self, tokens_used: int, model: str = "gpt-4"):
        """Update token usage tracking.
        
        Args:
            tokens_used: Number of tokens used
            model: Model name for cost calculation
        """
        try:
            self.token_usage["total_tokens"] += tokens_used
            self.token_usage["requests_count"] += 1
            
            # Calculate cost
            cost_per_1k = self.token_usage["model_costs"].get(model, self.token_usage["model_costs"]["default"])
            cost = (tokens_used / 1000) * cost_per_1k
            self.token_usage["total_cost"] += cost
            
            # Log usage periodically
            if self.token_usage["requests_count"] % 10 == 0:
                self.batch_log(
                    f"Token usage update: {tokens_used} tokens, ${cost:.4f} cost, "
                    f"Total: {self.token_usage['total_tokens']} tokens, ${self.token_usage['total_cost']:.4f}"
                )
                
        except Exception as e:
            logger.error(f"Error updating token usage: {e}")

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get current token usage statistics."""
        return self.token_usage.copy()

    def _find_similar_examples(self, prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar examples using cosine similarity.
        
        Args:
            prompt: Input prompt to find similar examples for
            top_k: Number of top similar examples to return
            
        Returns:
            List of similar examples with their similarity scores
        """
        if not self.sentence_transformer or not self.example_embeddings:
            return []
            
        try:
            # Encode the input prompt
            prompt_embedding = self.sentence_transformer.encode([prompt])
            
            # Compute cosine similarities
            similarities = np.dot(self.example_embeddings, prompt_embedding.T).flatten()
            
            # Get top-k similar examples
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            examples = self.prompt_examples.get('examples', [])
            similar_examples = []
            
            for idx in top_indices:
                if idx < len(examples):
                    example = examples[idx]
                    similar_examples.append({
                        'example': example,
                        'similarity_score': float(similarities[idx]),
                        'prompt': example['prompt'],
                        'parsed_output': example['parsed_output'],
                        'category': example.get('category', 'unknown'),
                        'performance_score': example.get('performance_score', 0.0)
                    })
            
            return similar_examples
            
        except Exception as e:
            logger.error(f"Error finding similar examples: {e}")
            return []

    def _create_few_shot_prompt(self, prompt: str, similar_examples: List[Dict[str, Any]]) -> str:
        """Create a few-shot prompt with similar examples as context.
        
        Args:
            prompt: Original user prompt
            similar_examples: List of similar examples with their outputs
            
        Returns:
            Enhanced prompt with few-shot examples
        """
        if not similar_examples:
            return prompt
            
        # Create few-shot context
        few_shot_context = "Here are some similar examples to help guide your response:\n\n"
        
        for i, example_data in enumerate(similar_examples, 1):
            example = example_data['example']
            few_shot_context += f"Example {i}:\n"
            few_shot_context += f"Input: {example['prompt']}\n"
            few_shot_context += f"Output: {json.dumps(example['parsed_output'], indent=2)}\n"
            few_shot_context += f"Category: {example.get('category', 'unknown')}\n"
            few_shot_context += f"Performance Score: {example.get('performance_score', 0.0):.2f}\n\n"
        
        few_shot_context += f"Now, please process this request:\n{prompt}\n\n"
        few_shot_context += "Based on the examples above, provide a structured response in JSON format."
        
        return few_shot_context

    def _save_successful_example(self, prompt: str, parsed_output: Dict[str, Any], 
                                category: str = "unknown", performance_score: float = 0.0) -> None:
        """Save a successful prompt example to the examples file.
        
        Args:
            prompt: User prompt
            parsed_output: Parsed output from the prompt
            category: Category of the prompt
            performance_score: Performance score of the response
        """
        try:
            if not self.prompt_examples:
                return
                
            new_example = {
                "id": f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "prompt": prompt,
                "category": category,
                "symbols": self._extract_symbols_from_prompt(prompt),
                "timeframe": self._extract_timeframe_from_prompt(prompt),
                "strategy_type": self._extract_strategy_type_from_prompt(prompt),
                "parsed_output": parsed_output,
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "performance_score": performance_score
            }
            
            # Add to examples
            self.prompt_examples['examples'].append(new_example)
            
            # Update metadata
            self.prompt_examples['metadata']['total_examples'] = len(self.prompt_examples['examples'])
            self.prompt_examples['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Save to file
            examples_path = Path(__file__).parent / "prompt_examples.json"
            with open(examples_path, 'w') as f:
                json.dump(self.prompt_examples, f, indent=2)
                
            # Update embeddings if available
            if self.sentence_transformer:
                self.example_embeddings = self._compute_example_embeddings()
                
            logger.info(f"Saved successful prompt example: {new_example['id']}")
            
        except Exception as e:
            logger.error(f"Error saving prompt example: {e}")

    def _extract_symbols_from_prompt(self, prompt: str) -> List[str]:
        """Extract stock symbols from prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            List of extracted symbols
        """
        # Simple regex to find stock symbols (1-5 capital letters)
        symbols = re.findall(r'\b([A-Z]{1,5})\b', prompt.upper())
        return list(set(symbols))  # Remove duplicates

    def _extract_timeframe_from_prompt(self, prompt: str) -> str:
        """Extract timeframe from prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Extracted timeframe
        """
        prompt_lower = prompt.lower()
        
        if "next" in prompt_lower and "day" in prompt_lower:
            match = re.search(r"next (\d+)d?", prompt_lower)
            if match:
                return f"{match.group(1)} days"
        elif "next" in prompt_lower and "week" in prompt_lower:
            match = re.search(r"next (\d+)w?", prompt_lower)
            if match:
                return f"{int(match.group(1)) * 7} days"
        elif "next" in prompt_lower and "month" in prompt_lower:
            match = re.search(r"next (\d+)m?", prompt_lower)
            if match:
                return f"{int(match.group(1)) * 30} days"
        elif "last" in prompt_lower and "month" in prompt_lower:
            match = re.search(r"last (\d+)m?", prompt_lower)
            if match:
                return f"{int(match.group(1)) * 30} days"
                
        return "unknown"

    def _extract_strategy_type_from_prompt(self, prompt: str) -> str:
        """Extract strategy type from prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Extracted strategy type
        """
        prompt_lower = prompt.lower()
        
        strategy_keywords = {
            "rsi": "RSI",
            "macd": "MACD", 
            "bollinger": "Bollinger_Bands",
            "moving average": "Moving_Average",
            "sma": "SMA",
            "ema": "EMA",
            "stochastic": "Stochastic",
            "williams": "Williams_R",
            "cci": "CCI",
            "atr": "ATR",
            "forecast": "Forecasting",
            "predict": "Prediction",
            "backtest": "Backtesting",
            "optimize": "Optimization",
            "analyze": "Analysis"
        }
        
        for keyword, strategy_type in strategy_keywords.items():
            if keyword in prompt_lower:
                return strategy_type
                
        return "unknown"

    def get_prompt_examples_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded prompt examples.
        
        Returns:
            Dictionary with prompt examples statistics
        """
        if not self.prompt_examples:
            return {"error": "No prompt examples loaded"}
            
        examples = self.prompt_examples.get('examples', [])
        metadata = self.prompt_examples.get('metadata', {})
        
        # Count examples by category
        categories = {}
        symbols = set()
        strategy_types = set()
        
        for example in examples:
            category = example.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # Collect symbols
            for symbol in example.get('symbols', []):
                symbols.add(symbol)
                
            # Collect strategy types
            strategy_type = example.get('strategy_type', 'unknown')
            strategy_types.add(strategy_type)
        
        # Calculate average performance score
        performance_scores = [ex.get('performance_score', 0.0) for ex in examples]
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
        
        return {
            "total_examples": len(examples),
            "categories": categories,
            "unique_symbols": list(symbols),
            "unique_strategy_types": list(strategy_types),
            "average_performance_score": avg_performance,
            "embeddings_available": self.example_embeddings is not None,
            "sentence_transformer_available": self.sentence_transformer is not None,
            "metadata": metadata
        }

    def process_prompt(self, prompt: str) -> AgentResponse:
        """Process user prompt and route through trading pipeline.

        Args:
            prompt: User prompt

        Returns:
            Agent response with results and recommendations
        """
        try:
            # Sanitize and validate prompt
            original_prompt = prompt
            prompt = self.sanitize_prompt(prompt)
            
            # Estimate token usage and cost
            token_estimate = self.estimate_token_usage(prompt, model="gpt-4")
            self.batch_log(f"Processing prompt: {len(prompt)} chars, estimated {token_estimate['token_count']} tokens, ${token_estimate['estimated_cost']:.4f}")

            # Find similar examples for few-shot learning
            similar_examples = self._find_similar_examples(prompt, top_k=3)
            
            # Create enhanced prompt with few-shot examples
            enhanced_prompt = self._create_few_shot_prompt(prompt, similar_examples)
            
            if similar_examples:
                self.batch_log(f"Found {len(similar_examples)} similar examples with scores: "
                          f"{[f'{ex['similarity_score']:.3f}' for ex in similar_examples]}")

            # Parse prompt to extract intent and parameters
            intent, params = self._parse_prompt(prompt)

            # Process the request based on intent
            if intent == "forecast":
                response = self._handle_forecast_request(params)
            elif intent == "strategy":
                response = self._handle_strategy_request(params)
            elif intent == "backtest":
                response = self._handle_backtest_request(params)
            elif intent == "trade":
                response = self._handle_trade_request(params)
            elif intent == "optimize":
                response = self._handle_optimization_request(params)
            elif intent == "analyze":
                response = self._handle_analysis_request(params)
            elif intent == "create_model":
                response = self._handle_model_creation_request(params)
            else:
                response = {
                    "success": True,
                    "result": self._handle_general_request(prompt, params),
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                }

            # Save successful example if response was successful
            if response and response.get("success", False):
                try:
                    # Extract category from intent
                    category = intent if intent != "general" else "general_request"
                    
                    # Create parsed output for saving
                    parsed_output = {
                        "action": intent,
                        "parameters": params,
                        "response": response.get("result", {}),
                        "message": response.get("message", ""),
                        "timestamp": response.get("timestamp", datetime.now().isoformat())
                    }
                    
                    # Calculate performance score (simple heuristic)
                    performance_score = 0.8  # Default score
                    if response.get("result"):
                        performance_score = 0.9
                    if similar_examples:
                        # Boost score if we found similar examples
                        avg_similarity = sum(ex['similarity_score'] for ex in similar_examples) / len(similar_examples)
                        performance_score = min(1.0, performance_score + avg_similarity * 0.1)
                    
                    # Save the successful example
                    self._save_successful_example(prompt, parsed_output, category, performance_score)
                    
                except Exception as e:
                    logger.warning(f"Could not save successful example: {e}")

            return response

        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            return AgentResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                recommendations=["Please try rephrasing your request"],
            )

    def _parse_prompt(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Parse prompt to extract intent and parameters.

        Args:
            prompt: User prompt

        Returns:
            Tuple of (intent, parameters)
        """
        prompt_lower = prompt.lower()

        # Extract symbol
        symbol_match = re.search(r"\b([A-Z]{1,5})\b", prompt.upper())
        symbol = symbol_match.group(1) if symbol_match else "AAPL"

        # Extract timeframe
        timeframe = "15d"
        if "next" in prompt_lower and "day" in prompt_lower:
            timeframe_match = re.search(r"next (\d+)d?", prompt_lower)
            if timeframe_match:
                timeframe = f"{timeframe_match.group(1)}d"
        elif "next" in prompt_lower and "week" in prompt_lower:
            timeframe_match = re.search(r"next (\d+)w?", prompt_lower)
            if timeframe_match:
                timeframe = f"{int(timeframe_match.group(1)) * 7}d"
        elif "next" in prompt_lower and "month" in prompt_lower:
            timeframe_match = re.search(r"next (\d+)m?", prompt_lower)
            if timeframe_match:
                timeframe = f"{int(timeframe_match.group(1)) * 30}d"

        # Determine intent
        if any(
            word in prompt_lower
            for word in ["create model", "build model", "new model", "custom model"]
        ):
            intent = "create_model"
        elif any(word in prompt_lower for word in ["forecast", "predict", "price"]):
            intent = "forecast"
        elif any(word in prompt_lower for word in ["strategy", "strategy", "signal"]):
            intent = "strategy"
        elif any(word in prompt_lower for word in ["backtest", "test", "simulate"]):
            intent = "backtest"
        elif any(word in prompt_lower for word in ["trade", "buy", "sell", "execute"]):
            intent = "trade"
        elif any(word in prompt_lower for word in ["optimize", "improve", "tune"]):
            intent = "optimize"
        elif any(word in prompt_lower for word in ["analyze", "analysis", "report"]):
            intent = "analyze"
        else:
            intent = "general"

        # Extract strategy preference
        strategy = None
        for strategy_key, strategy_name in self.strategy_registry.items():
            if strategy_key in prompt_lower:
                strategy = strategy_name
                break

        # Extract model preference
        model = None
        create_new_model = False

        # Check for dynamic model creation requests
        if any(
            phrase in prompt_lower
            for phrase in ["create model", "build model", "new model", "custom model"]
        ):
            create_new_model = True
            model = "dynamic"
        else:
            for model_key, model_name in self.model_registry.items():
                if model_key in prompt_lower:
                    model = model_name
                    break

        # Auto-select best strategy if none specified
        if not strategy:
            strategy = self._select_best_strategy(symbol)

        # Auto-select best model if none specified
        if not model:
            model = self._select_best_model(symbol, timeframe)

        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "model": model,
            "create_new_model": create_new_model,
            "prompt": prompt,
        }

        logger.info(f"Parsed prompt - Intent: {intent}, Params: {params}")

        return intent, params

    def _select_best_strategy(self, symbol: str) -> str:
        """Select best strategy based on symbol characteristics.

        Args:
            symbol: Trading symbol

        Returns:
            Best strategy name
        """
        # Simple heuristic based on symbol type
        if symbol in ["BTC", "ETH", "ADA", "DOT"]:
            return "RSI Mean Reversion"  # Good for crypto
        elif symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
            return "Moving Average Crossover"  # Good for tech stocks
        elif symbol in ["SPY", "QQQ", "IWM"]:
            return "Bollinger Bands"  # Good for ETFs
        else:
            return "Ensemble Strategy"  # Default to ensemble

    def _select_best_model(self, symbol: str, timeframe: str) -> str:
        """Select best model based on symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Forecast timeframe

        Returns:
            Best model name
        """
        # Parse timeframe
        days = int(timeframe.replace("d", ""))

        if days <= 7:
            return "ARIMA"  # Good for short-term
        elif days <= 30:
            return "LSTM"  # Good for medium-term
        else:
            return "Transformer"  # Good for long-term

    def _handle_forecast_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle forecast request.

        Args:
            params: Request parameters

        Returns:
            Agent response
        """
        try:
            symbol = params["symbol"]
            timeframe = params["timeframe"]
            model = params["model"]

            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data

            data = self.data_provider.get_historical_data(
                symbol, start_date, end_date, "1d"
            )

            if data is None or data.empty:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get data for {symbol}",
                    recommendations=[
                        "Try a different symbol or check data availability"
                    ],
                )

            # Generate forecast
            horizon = int(timeframe.replace("d", ""))
            forecast_result = self.forecast_router.get_forecast(
                data=data, horizon=horizon, model_type=model.lower()
            )

            # Create response
            message = f"Forecast for {symbol} using {model} model:\n"
            message += f"Horizon: {timeframe}\n"
            message += f"Confidence: {forecast_result['confidence']:.2%}\n"

            if "warnings" in forecast_result:
                message += f"Warnings: {', '.join(forecast_result['warnings'])}\n"

            recommendations = [
                f"Consider using {forecast_result['model']} for future forecasts",
                "Monitor forecast accuracy and adjust model selection",
                "Use multiple models for ensemble forecasting",
            ]

            next_actions = [
                f"Run backtest with {params['strategy']} strategy",
                "Generate trading signals",
                "Execute paper trade",
            ]

            return AgentResponse(
                success=True,
                message=message,
                data=forecast_result,
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            logger.error(f"Error in forecast request: {e}")
            return AgentResponse(
                success=False,
                message=f"Forecast failed: {str(e)}",
                recommendations=["Try a different model or timeframe"],
            )

    def _handle_strategy_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle strategy request.

        Args:
            params: Request parameters

        Returns:
            Agent response
        """
        try:
            symbol = params["symbol"]
            strategy = params["strategy"]

            # Get strategy analysis
            strategy_analysis = self.strategy_gatekeeper.analyze_strategy(
                strategy, symbol
            )

            message = f"Strategy Analysis for {strategy} on {symbol}:\n"
            message += f"Health Score: {strategy_analysis.get('health_score', 'N/A')}\n"
            message += f"Risk Level: {strategy_analysis.get('risk_level', 'N/A')}\n"

            recommendations = strategy_analysis.get("recommendations", [])

            next_actions = [
                "Run backtest to validate strategy",
                "Optimize strategy parameters",
                "Execute strategy in paper trading",
            ]

            return AgentResponse(
                success=True,
                message=message,
                data=strategy_analysis,
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            logger.error(f"Error in strategy request: {e}")
            return AgentResponse(
                success=False,
                message=f"Strategy analysis failed: {str(e)}",
                recommendations=[
                    "Try a different strategy or check symbol availability"
                ],
            )

    def _handle_backtest_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle backtest request.

        Args:
            params: Request parameters

        Returns:
            Agent response
        """
        try:
            symbol = params["symbol"]
            strategy = params["strategy"]

            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            data = self.data_provider.get_historical_data(
                symbol, start_date, end_date, "1d"
            )

            if data is None or data.empty:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get data for {symbol}",
                    recommendations=[
                        "Try a different symbol or check data availability"
                    ],
                )

            # Run backtest
            backtester = Backtester(data)
            equity_curve, trade_log, metrics = backtester.run_backtest([strategy])

            # Analyze results
            sharpe = metrics.get("sharpe_ratio", 0)
            total_return = metrics.get("total_return", 0)
            max_dd = metrics.get("max_drawdown", 0)

            message = f"Backtest Results for {strategy} on {symbol}:\n"
            message += f"Sharpe Ratio: {sharpe:.2f}\n"
            message += f"Total Return: {total_return:.2%}\n"
            message += f"Max Drawdown: {max_dd:.2%}\n"
            message += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"

            recommendations = []
            if sharpe < 1.0:
                recommendations.append(
                    "Sharpe ratio below 1.0 - consider strategy optimization"
                )
            if max_dd > 0.2:
                recommendations.append(
                    "High drawdown - implement better risk management"
                )
            if total_return < 0.05:
                recommendations.append("Low returns - consider alternative strategies")

            if not recommendations:
                recommendations.append(
                    "Strategy performing well - consider live trading"
                )

            next_actions = []
            if sharpe >= 1.0 and total_return > 0.1:
                next_actions.append("Execute paper trade")
                next_actions.append("Optimize strategy parameters")
            else:
                next_actions.append("Try different strategy")
                next_actions.append("Adjust risk parameters")

            return AgentResponse(
                success=True,
                message=message,
                data={
                    "metrics": metrics,
                    "equity_curve": equity_curve,
                    "trade_log": trade_log,
                },
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            logger.error(f"Error in backtest request: {e}")
            return AgentResponse(
                success=False,
                message=f"Backtest failed: {str(e)}",
                recommendations=["Check data availability and strategy parameters"],
            )

    def _handle_trade_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle trade request.

        Args:
            params: Request parameters

        Returns:
            Agent response
        """
        try:
            symbol = params["symbol"]

            # Get current market price
            current_price = self.data_provider.get_live_price(symbol)

            if current_price is None:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get current price for {symbol}",
                    recommendations=["Check symbol validity and market hours"],
                )

            # Simulate trade execution
            quantity = 100  # Default quantity
            side = "buy"  # Default side

            execution_result = self.trade_executor.simulate_trade(
                symbol=symbol, side=side, quantity=quantity, market_price=current_price
            )

            if execution_result.success:
                message = f"Trade Simulation for {symbol}:\n"
                message += f"Side: {side.upper()}\n"
                message += f"Quantity: {quantity}\n"
                message += f"Execution Price: ${execution_result.execution_price:.2f}\n"
                message += f"Commission: ${execution_result.commission:.2f}\n"
                message += f"Slippage: {execution_result.slippage:.4f}\n"
                message += f"Total Cost: ${execution_result.total_cost:.2f}\n"

                recommendations = [
                    "Monitor trade performance",
                    "Set stop-loss and take-profit levels",
                    "Review execution quality",
                ]

                next_actions = [
                    "Monitor position",
                    "Set up alerts",
                    "Plan exit strategy",
                ]
            else:
                message = f"Trade execution failed: {execution_result.error_message}"
                recommendations = ["Check market conditions", "Verify order parameters"]
                next_actions = ["Retry trade", "Adjust parameters"]

            return AgentResponse(
                success=execution_result.success,
                message=message,
                data={
                    "execution_result": execution_result,
                    "current_price": current_price,
                },
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            logger.error(f"Error in trade request: {e}")
            return AgentResponse(
                success=False,
                message=f"Trade execution failed: {str(e)}",
                recommendations=["Check market hours and symbol validity"],
            )

    def _handle_optimization_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle optimization request.

        Args:
            params: Request parameters

        Returns:
            Agent response
        """
        try:
            strategy = params["strategy"]

            # Get current performance metrics
            current_metrics = {
                "sharpe_ratio": 0.8,  # Mock metrics
                "total_return": 0.15,
                "max_drawdown": 0.12,
                "win_rate": 0.55,
            }

            # Run optimization
            optimization_result = self.optimizer.optimize_strategy(
                strategy=strategy,
                current_parameters={"period": 14, "threshold": 0.05},
                current_metrics=current_metrics,
            )

            if optimization_result:
                message = f"Optimization Results for {strategy}:\n"
                message += f"Confidence: {optimization_result.confidence:.2%}\n"
                message += f"Improvements: {optimization_result.improvement}\n"

                recommendations = optimization_result.recommendations
                next_actions = [
                    "Apply new parameters",
                    "Run backtest with optimized parameters",
                    "Monitor performance improvement",
                ]
            else:
                message = f"No optimization needed for {strategy}"
                recommendations = ["Continue monitoring performance"]
                next_actions = ["Run periodic optimization checks"]

            return AgentResponse(
                success=True,
                message=message,
                data={"optimization_result": optimization_result},
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            logger.error(f"Error in optimization request: {e}")
            return AgentResponse(
                success=False,
                message=f"Optimization failed: {str(e)}",
                recommendations=["Check strategy configuration and historical data"],
            )

    def _handle_analysis_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle analysis request.

        Args:
            params: Request parameters

        Returns:
            Agent response
        """
        try:
            symbol = params["symbol"]

            # Get market data
            market_data = self.data_provider.get_market_data([symbol])

            if symbol not in market_data:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get market data for {symbol}",
                    recommendations=["Check symbol validity and market hours"],
                )

            data = market_data[symbol]

            message = f"Market Analysis for {symbol}:\n"
            message += f"Current Price: ${data['price']:.2f}\n"
            message += f"Change: {data['change']:+.2f} ({data['change_pct']:+.2f}%)\n"
            message += f"Volume: {data['volume']:,.0f}\n"

            recommendations = [
                "Monitor key support/resistance levels",
                "Check for news catalysts",
                "Review technical indicators",
            ]

            next_actions = [
                "Generate price forecast",
                "Run technical analysis",
                "Check fundamental data",
            ]

            return AgentResponse(
                success=True,
                message=message,
                data=market_data,
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            logger.error(f"Error in analysis request: {e}")
            return AgentResponse(
                success=False,
                message=f"Analysis failed: {str(e)}",
                recommendations=["Check data availability and symbol validity"],
            )

    def _handle_model_creation_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle dynamic model creation request.

        Args:
            params: Request parameters

        Returns:
            Agent response
        """
        try:
            if not self.model_creator:
                return AgentResponse(
                    success=False,
                    message="Model creator not available",
                    recommendations=["Please try using an existing model"],
                )

            # Extract model requirements from prompt
            prompt = params["prompt"]
            symbol = params["symbol"]

            # Generate model requirements based on prompt
            requirements = self._generate_model_requirements(prompt, symbol)

            # Create model name
            model_name = f"dynamic_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create and validate model
            model_spec, success, errors = self.model_creator.create_and_validate_model(
                requirements, model_name
            )

            if success:
                # Run full evaluation
                evaluation = self.model_creator.run_full_evaluation(model_name)

                message = f"Successfully created model '{model_spec.name}':\n"
                message += f"Framework: {model_spec.framework}\n"
                message += f"Type: {model_spec.model_type}\n"
                message += f"Performance Grade: {evaluation.performance_grade}\n"
                message += f"RMSE: {evaluation.metrics.get('rmse', 'N/A'):.4f}\n"
                message += (
                    f"Sharpe: {evaluation.metrics.get('sharpe_ratio', 'N/A'):.4f}\n"
                )

                recommendations = evaluation.recommendations

                next_actions = [
                    f"Test model on {symbol} data",
                    "Compare with existing models",
                    "Add to ensemble if performance is good",
                ]

                return AgentResponse(
                    success=True,
                    message=message,
                    data={
                        "model_spec": asdict(model_spec),
                        "evaluation": asdict(evaluation),
                    },
                    recommendations=recommendations,
                    next_actions=next_actions,
                )
            else:
                return AgentResponse(
                    success=False,
                    message=f"Model creation failed: {', '.join(errors)}",
                    recommendations=[
                        "Try a different model description",
                        "Use an existing model instead",
                    ],
                )

        except Exception as e:
            logger.error(f"Error in model creation request: {e}")
            return AgentResponse(
                success=False,
                message=f"Model creation failed: {str(e)}",
                recommendations=[
                    "Try a simpler model description",
                    "Use an existing model",
                ],
            )

    def _generate_model_requirements(self, prompt: str, symbol: str) -> str:
        """Generate model requirements from prompt.

        Args:
            prompt: User prompt
            symbol: Trading symbol

        Returns:
            Model requirements string
        """
        prompt_lower = prompt.lower()

        # Extract model type preferences
        if "lstm" in prompt_lower or "neural" in prompt_lower or "deep" in prompt_lower:
            model_type = "LSTM neural network"
        elif "transformer" in prompt_lower or "attention" in prompt_lower:
            model_type = "Transformer model"
        elif "xgboost" in prompt_lower or "gradient" in prompt_lower:
            model_type = "XGBoost gradient boosting"
        elif "random" in prompt_lower or "forest" in prompt_lower:
            model_type = "Random Forest"
        elif "linear" in prompt_lower or "regression" in prompt_lower:
            model_type = "Linear regression"
        else:
            model_type = "machine learning model"

        # Extract complexity preferences
        if "simple" in prompt_lower or "basic" in prompt_lower:
            complexity = "simple"
        elif "complex" in prompt_lower or "advanced" in prompt_lower:
            complexity = "complex"
        else:
            complexity = "moderate"

        # Generate requirements
        requirements = (
            f"Create a {complexity} {model_type} for forecasting {symbol} stock prices"
        )

        # Add specific requirements based on prompt
        if "accurate" in prompt_lower or "precise" in prompt_lower:
            requirements += " with high accuracy"
        if "fast" in prompt_lower or "quick" in prompt_lower:
            requirements += " optimized for speed"
        if "robust" in prompt_lower or "stable" in prompt_lower:
            requirements += " with robust performance"

        return requirements

    def _handle_general_request(
        self, prompt: str, params: Dict[str, Any]
    ) -> AgentResponse:
        """Handle general request with full pipeline.

        Args:
            prompt: Original prompt
            params: Parsed parameters

        Returns:
            Agent response
        """
        try:
            # Run full pipeline: Forecast → Strategy → Backtest → Report
            results = {}

            # 1. Generate forecast
            forecast_response = self._handle_forecast_request(params)
            if forecast_response.success:
                results["forecast"] = forecast_response.data

            # 2. Analyze strategy
            strategy_response = self._handle_strategy_request(params)
            if strategy_response.success:
                results["strategy"] = strategy_response.data

            # 3. Run backtest
            backtest_response = self._handle_backtest_request(params)
            if backtest_response.success:
                results["backtest"] = backtest_response.data

            # 4. Generate comprehensive report
            message = f"Complete Analysis for {params['symbol']}:\n\n"

            if "forecast" in results:
                message += "📈 FORECAST:\n"
                message += f"Model: {results['forecast']['model']}\n"
                message += f"Confidence: {results['forecast']['confidence']:.2%}\n\n"

            if "strategy" in results:
                message += "🎯 STRATEGY:\n"
                message += (
                    f"Health Score: {results['strategy'].get('health_score', 'N/A')}\n"
                )
                message += (
                    f"Risk Level: {results['strategy'].get('risk_level', 'N/A')}\n\n"
                )

            if "backtest" in results:
                metrics = results["backtest"]["metrics"]
                message += "📊 BACKTEST RESULTS:\n"
                message += f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
                message += f"Total Return: {metrics.get('total_return', 0):.2%}\n"
                message += f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
                message += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n\n"

            # Generate recommendations
            recommendations = []
            if "backtest" in results:
                metrics = results["backtest"]["metrics"]
                sharpe = metrics.get("sharpe_ratio", 0)

                if sharpe >= 1.0:
                    recommendations.append(
                        "✅ Strategy performing well - consider live trading"
                    )
                elif sharpe >= 0.5:
                    recommendations.append(
                        "⚠️ Strategy needs optimization - run parameter tuning"
                    )
                else:
                    recommendations.append(
                        "❌ Strategy underperforming - try alternative approach"
                    )

            recommendations.extend(
                [
                    "Monitor performance regularly",
                    "Set up automated alerts",
                    "Review and adjust parameters monthly",
                ]
            )

            # Suggest next actions
            next_actions = [
                "Execute paper trade",
                "Set up performance monitoring",
                "Schedule regular reviews",
            ]

            return AgentResponse(
                success=True,
                message=message,
                data=results,
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            logger.error(f"Error in general request: {e}")
            return AgentResponse(
                success=False,
                message=f"Analysis failed: {str(e)}",
                recommendations=["Try breaking down the request into smaller parts"],
            )


# Global prompt agent instance
prompt_agent = PromptAgent()


def get_prompt_agent() -> PromptAgent:
    """Get the global prompt agent instance."""
    return prompt_agent
