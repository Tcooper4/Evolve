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
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import sentence transformers for semantic similarity (optional; guarded for version conflicts)
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.getLogger(__name__).warning("Semantic matching disabled: %s", e)

# Import tiktoken for token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from trading.data.providers.fallback_provider import FallbackDataProvider
from trading.data.earnings_calendar import get_upcoming_earnings
from trading.data.insider_flow import get_insider_flow
from trading.data.news_aggregator import get_news
from trading.data.short_interest import get_short_interest
from trading.utils.data_manager import disk_cache_get, disk_cache_set
from trading.execution.trade_execution_simulator import TradeExecutionSimulator
from trading.models.forecast_router import ForecastRouter
from trading.optimization.self_tuning_optimizer import SelfTuningOptimizer
from trading.strategies.gatekeeper import StrategyGatekeeper

logger = logging.getLogger(__name__)

# Log warnings after logger is defined
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    logger.warning(
        "Semantic prompt matching disabled (pip install sentence-transformers to enable)"
    )

if not TIKTOKEN_AVAILABLE:
    logger.warning("Token counting disabled (pip install tiktoken to enable)")


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
        # Removed return statement - __init__ should not return values

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
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Initialize prompt examples system
        self.prompt_examples = self._load_prompt_examples()
        self.sentence_transformer = None
        self.example_embeddings = None

        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            try:
                self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
                if self.prompt_examples:
                    self.example_embeddings = self._compute_example_embeddings()
                    self.logger.info("Prompt examples system initialized successfully")
                else:
                    self.logger.info("SentenceTransformers available but no prompt examples file found - system will work without examples")
            except Exception as e:
                self.logger.warning(f"Could not initialize sentence transformer: {e}")
        else:
            self.logger.info(
                "Prompt examples system disabled (SentenceTransformers not available)"
            )

        # Initialize token usage tracking
        self.token_usage = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_count": 0,
            "model_costs": {
                "gpt-4": 0.03,  # per 1K tokens
                "gpt-3.5-turbo": 0.002,
                "claude-3": 0.015,
                "default": 0.01,
            },
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
                self.logger.info("Token counting initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize tokenizer: {e}")
        else:
            # Already logged once at import time; avoid duplicate startup log lines.
            pass

        # Initialize components
        self.forecast_router = ForecastRouter()

        # Initialize model creator for dynamic model generation
        try:
            from trading.agents.model_builder_agent import ModelBuilderAgent, ModelBuildRequest

            self.model_creator = ModelBuilderAgent()
            self.logger.info("Model builder agent initialized successfully")
        except Exception as e:
            self.logger.debug(f"Could not initialize model builder: {e}")
            self.model_creator = None

        # Initialize prompt router for intelligent routing
        try:
            from trading.agents.enhanced_prompt_router import EnhancedPromptRouterAgent

            self.prompt_router = EnhancedPromptRouterAgent()
            self.logger.info("Prompt router agent initialized successfully")
        except Exception as e:
            self.logger.debug(f"Could not initialize prompt router: {e}")
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

        self.logger.info("Enhanced Prompt Agent initialized with full pipeline routing")

    def _load_prompt_examples(self) -> Optional[Dict[str, Any]]:
        """Load prompt examples from JSON file.

        Returns:
            Dictionary containing prompt examples or None if file not found
        """
        try:
            examples_path = Path(__file__).parent / "prompt_examples.json"
            if examples_path.exists():
                with open(examples_path, "r") as f:
                    examples = json.load(f)
                self.logger.info(
                    f"Loaded {len(examples.get('examples', []))} prompt examples"
                )
                return examples
            else:
                self.logger.warning("Prompt examples file not found")
                return None
        except Exception as e:
            self.logger.error(f"Error loading prompt examples: {e}")
            return None

    def _compute_example_embeddings(self) -> Optional[np.ndarray]:
        """Compute embeddings for all prompt examples.

        Returns:
            Numpy array of embeddings or None if computation fails
        """
        if not self.prompt_examples or not self.sentence_transformer:
            return None

        try:
            examples = self.prompt_examples.get("examples", [])
            prompts = [example["prompt"] for example in examples]
            embeddings = self.sentence_transformer.encode(prompts)
            self.logger.info(f"Computed embeddings for {len(prompts)} examples")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error computing example embeddings: {e}")
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
            cost_per_1k = self.token_usage["model_costs"].get(
                model, self.token_usage["model_costs"]["default"]
            )
            estimated_cost = (token_count / 1000) * cost_per_1k

            return {
                "token_count": token_count,
                "estimated_cost": estimated_cost,
                "model": model,
                "cost_per_1k": cost_per_1k,
            }

        except Exception as e:
            self.logger.warning(f"Error estimating token usage: {e}")
            return {
                "token_count": 0,
                "estimated_cost": 0.0,
                "model": model,
                "cost_per_1k": 0.0,
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
                r"<script.*?</script>",
                r"javascript:",
                r"data:text/html",
                r"vbscript:",
                r"on\w+\s*=",
                r"<iframe",
                r"<object",
                r"<embed",
            ]

            sanitized = prompt
            for pattern in injection_patterns:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

            # Remove excessive whitespace
            sanitized = re.sub(r"\s+", " ", sanitized).strip()

            # Truncate if too long
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length] + "..."
                self.logger.warning(
                    f"Prompt truncated from {len(prompt)} to {len(sanitized)} characters"
                )

            # Log if significant changes were made
            if len(sanitized) != len(prompt):
                self.logger.info(
                    f"Prompt sanitized: {len(prompt)} -> {len(sanitized)} characters"
                )

            return sanitized

        except Exception as e:
            self.logger.error(f"Error sanitizing prompt: {e}")
            return prompt[:max_length] if len(prompt) > max_length else prompt

    def batch_log(self, message: str, level: str = "info"):
        """Add log message to buffer for batch processing."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {"timestamp": timestamp, "level": level, "message": message}

            self.log_buffer.append(log_entry)

            # Flush buffer if it's full or enough time has passed
            current_time = time.time()
            if (
                len(self.log_buffer) >= 100
                or current_time - self.last_log_flush >= self.log_flush_interval
            ):
                self._flush_log_buffer()

        except Exception as e:
            # Fallback to direct logging
            self.logger.error(f"Error in batch logging: {e}")
            self.logger.info(message)

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
                    getattr(self.logger, level)(messages[0])
                else:
                    # Multiple messages, log as batch
                    batch_message = (
                        f"Batch of {len(messages)} {level} messages: "
                        + "; ".join(messages[:5])
                    )
                    if len(messages) > 5:
                        batch_message += f" ... and {len(messages) - 5} more"
                    getattr(self.logger, level)(batch_message)

            # Clear buffer
            self.log_buffer.clear()
            self.last_log_flush = time.time()

        except Exception as e:
            self.logger.error(f"Error flushing log buffer: {e}")

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
            cost_per_1k = self.token_usage["model_costs"].get(
                model, self.token_usage["model_costs"]["default"]
            )
            cost = (tokens_used / 1000) * cost_per_1k
            self.token_usage["total_cost"] += cost

            # Log usage periodically
            if self.token_usage["requests_count"] % 10 == 0:
                self.batch_log(
                    f"Token usage update: {tokens_used} tokens, ${cost:.4f} cost, "
                    f"Total: {self.token_usage['total_tokens']} tokens, ${self.token_usage['total_cost']:.4f}"
                )

        except Exception as e:
            self.logger.error(f"Error updating token usage: {e}")

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get current token usage statistics."""
        return self.token_usage.copy()

    def _find_similar_examples(
        self, prompt: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
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

            examples = self.prompt_examples.get("examples", [])
            similar_examples = []

            for idx in top_indices:
                if idx < len(examples):
                    example = examples[idx]
                    similar_examples.append(
                        {
                            "example": example,
                            "similarity_score": float(similarities[idx]),
                            "prompt": example["prompt"],
                            "parsed_output": example["parsed_output"],
                            "category": example.get("category", "unknown"),
                            "performance_score": example.get("performance_score", 0.0),
                        }
                    )

            return similar_examples

        except Exception as e:
            self.logger.error(f"Error finding similar examples: {e}")
            return []

    def _create_few_shot_prompt(
        self, prompt: str, similar_examples: List[Dict[str, Any]]
    ) -> str:
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
        few_shot_context = (
            "Here are some similar examples to help guide your response:\n\n"
        )

        for i, example_data in enumerate(similar_examples, 1):
            example = example_data["example"]
            few_shot_context += f"Example {i}:\n"
            few_shot_context += f"Input: {example['prompt']}\n"
            few_shot_context += (
                f"Output: {json.dumps(example['parsed_output'], indent=2)}\n"
            )
            few_shot_context += f"Category: {example.get('category', 'unknown')}\n"
            few_shot_context += (
                f"Performance Score: {example.get('performance_score', 0.0):.2f}\n\n"
            )

        few_shot_context += f"Now, please process this request:\n{prompt}\n\n"
        few_shot_context += (
            "Based on the examples above, provide a structured response in JSON format."
        )

        return few_shot_context

    def _save_successful_example(
        self,
        prompt: str,
        parsed_output: Dict[str, Any],
        category: str = "unknown",
        performance_score: float = 0.0,
    ) -> None:
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
                "performance_score": performance_score,
            }

            # Add to examples
            self.prompt_examples["examples"].append(new_example)

            # Update metadata
            self.prompt_examples["metadata"]["total_examples"] = len(
                self.prompt_examples["examples"]
            )
            self.prompt_examples["metadata"][
                "last_updated"
            ] = datetime.now().isoformat()

            # Save to file
            examples_path = Path(__file__).parent / "prompt_examples.json"
            with open(examples_path, "w") as f:
                json.dump(self.prompt_examples, f, indent=2)

            # Update embeddings if available
            if self.sentence_transformer:
                self.example_embeddings = self._compute_example_embeddings()

            self.logger.info(f"Saved successful prompt example: {new_example['id']}")

        except Exception as e:
            self.logger.error(f"Error saving prompt example: {e}")

    # Common English words that must not be treated as tickers (1-5 letters)
    _TICKER_STOPWORDS = frozenset({
        "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
        "this", "that", "these", "those", "them", "they", "the", "and", "are",
        "is", "it", "its", "for", "from", "with", "was", "were", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "about", "into", "than",
        "then", "some", "more", "most", "other", "only", "just", "also",
        "very", "your", "our", "all", "any", "each", "both", "such", "here",
        "there", "out", "not", "but", "yes", "no", "see", "saw", "get", "got",
        "give", "me", "you",
    })

    # Company / index / commodity name (lowercase) -> ticker
    _COMPANY_TO_TICKER = {
        "apple": "AAPL",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "netflix": "NFLX",
        "alphabet": "GOOGL",
        "amd": "AMD",
        "intel": "INTC",
        "ibm": "IBM",
        "salesforce": "CRM",
        "oracle": "ORCL",
        "adobe": "ADBE",
        "cisco": "CSCO",
        # Common commodities and indices
        "oil": "CL=F",
        "crude": "CL=F",
        "crude oil": "CL=F",
        "wti": "CL=F",
        "gold": "GC=F",
        "silver": "SI=F",
        "copper": "HG=F",
        "natural gas": "NG=F",
        "gas": "NG=F",
        "wheat": "ZW=F",
        "corn": "ZC=F",
        "bitcoin": "BTC-USD",
        "btc": "BTC-USD",
        "ethereum": "ETH-USD",
        "eth": "ETH-USD",
        "s&p": "SPY",
        "s&p 500": "SPY",
        "sp500": "SPY",
        "nasdaq": "QQQ",
        "dow": "DIA",
        "russell": "IWM",
    }

    def _extract_symbols_from_prompt(self, prompt: str) -> List[str]:
        """Extract stock symbols from prompt.

        Args:
            prompt: User prompt

        Returns:
            List of extracted symbols (valid tickers only; excludes stopwords and maps company names).
        """
        prompt_upper = prompt.upper()
        prompt_lower = prompt.lower()
        # 1) Map company names to tickers (whole-word match)
        for company, ticker in self._COMPANY_TO_TICKER.items():
            if re.search(r"\b" + re.escape(company) + r"\b", prompt_lower):
                return [ticker]
        # 2) Find 1-5 uppercase letter tokens and filter out stopwords
        candidates = re.findall(r"\b([A-Z]{1,5})\b", prompt_upper)
        symbols = [
            s for s in set(candidates)
            if s and s.lower() not in self._TICKER_STOPWORDS
        ]
        return list(symbols)

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
            "analyze": "Analysis",
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

        examples = self.prompt_examples.get("examples", [])
        metadata = self.prompt_examples.get("metadata", {})

        # Count examples by category
        categories = {}
        symbols = set()
        strategy_types = set()

        for example in examples:
            category = example.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1

            # Collect symbols
            for symbol in example.get("symbols", []):
                symbols.add(symbol)

            # Collect strategy types
            strategy_type = example.get("strategy_type", "unknown")
            strategy_types.add(strategy_type)

        # Calculate average performance score
        performance_scores = [ex.get("performance_score", 0.0) for ex in examples]
        avg_performance = (
            sum(performance_scores) / len(performance_scores)
            if performance_scores
            else 0.0
        )

        return {
            "total_examples": len(examples),
            "categories": categories,
            "unique_symbols": list(symbols),
            "unique_strategy_types": list(strategy_types),
            "average_performance_score": avg_performance,
            "embeddings_available": self.example_embeddings is not None,
            "sentence_transformer_available": self.sentence_transformer is not None,
            "metadata": metadata,
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
            prompt = self.sanitize_prompt(prompt)

            # AGENT_MEMORY_LAYER: Capture user-stated preferences (risk tolerance, style, etc.)
            try:
                from trading.memory import get_memory_store

                get_memory_store().ingest_preference_text(prompt, source="PromptAgent")
            except Exception:
                pass

            # Estimate token usage and cost
            token_estimate = self.estimate_token_usage(prompt, model="gpt-4")
            self.batch_log(
                f"Processing prompt: {len(prompt)} chars, "
                f"estimated {token_estimate['token_count']} tokens, "
                f"${token_estimate['estimated_cost']:.4f}"
            )

            # Find similar examples for few-shot learning
            similar_examples = self._find_similar_examples(prompt, top_k=3)

            # Create enhanced prompt with few-shot examples
            self._create_few_shot_prompt(prompt, similar_examples)

            if similar_examples:
                scores = [f"{ex['similarity_score']:.3f}" for ex in similar_examples]
                self.batch_log(
                    f"Found {len(similar_examples)} similar examples with scores: {scores}"
                )

            # Parse prompt to extract intent and parameters
            intent, params = self._parse_prompt(prompt)

            # No valid ticker: respond with general market analysis via LLM only (no data fetch)
            if params.get("symbol") is None:
                return self._handle_general_request_llm_only(prompt, params)

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
            elif intent == "critique_backtest":
                response = self._handle_critique_backtest_request(params)
            elif intent == "recommend_model":
                response = self._handle_recommend_model_request(params)
            elif intent == "general":
                # Price/market questions: use LLM-only path with live data so reply includes real price
                response = self._handle_general_request_llm_only(prompt, params)
            else:
                response = {
                    "success": True,
                    "result": self._handle_general_request(prompt, params),
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                }

            # Normalize AgentResponse to dict so downstream .get() and save block work
            if hasattr(response, "success") and not isinstance(response, dict):
                response = {
                    "success": response.success,
                    "message": getattr(response, "message", ""),
                    "data": getattr(response, "data", None),
                    "recommendations": getattr(response, "recommendations", None) or [],
                    "next_actions": getattr(response, "next_actions", None) or [],
                    "result": response,
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
                        "timestamp": response.get(
                            "timestamp", datetime.now().isoformat()
                        ),
                    }

                    # Calculate performance score (simple heuristic)
                    performance_score = 0.8  # Default score
                    if response.get("result"):
                        performance_score = 0.9
                    if similar_examples:
                        # Boost score if we found similar examples
                        avg_similarity = sum(
                            ex["similarity_score"] for ex in similar_examples
                        ) / len(similar_examples)
                        performance_score = min(
                            1.0, performance_score + avg_similarity * 0.1
                        )

                    # Save the successful example
                    self._save_successful_example(
                        prompt, parsed_output, category, performance_score
                    )

                except Exception as e:
                    self.logger.warning(f"Could not save successful example: {e}")

            return response

        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}")
            return AgentResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                recommendations=["Please try rephrasing your request"],
            )

    def _parse_prompt(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Parse prompt to extract intent and parameters using Claude API.

        Uses Claude for semantic intent detection; falls back to regex/keywords
        if Claude is unavailable or returns invalid output. AGENT_UPGRADE.
        """
        try:
            from config.llm_config import get_llm_config, CLAUDE_PRIMARY_MODEL
            llm = get_llm_config()
            if getattr(llm, "anthropic_api_key", None):
                import anthropic
                client = anthropic.Anthropic(api_key=llm.anthropic_api_key)
                model = getattr(llm, "primary_model", CLAUDE_PRIMARY_MODEL)
                system = (
                    "You are an intent classifier for the Evolve trading system. "
                    "Given a user message, respond with ONLY a single JSON object (no markdown, no explanation) with two keys: "
                    '"intent" and "params". '
                    "intent must be exactly one of: forecast, strategy, backtest, trade, optimize, analyze, create_model, critique_backtest, recommend_model, general. "
                    "params must be an object with: symbol (ticker, default AAPL if missing), timeframe (e.g. 15d, 30d), "
                    "strategy (name from: RSI Mean Reversion, Bollinger Bands, Moving Average Crossover, etc.), "
                    "model (e.g. ARIMA, LSTM, XGBoost), create_new_model (boolean), prompt (original user message). "
                    "Extract ticker symbols (1-5 uppercase letters), timeframes (e.g. next 5 days -> 5d), and strategy/model keywords from the message."
                )
                resp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    temperature=0.0,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    text = "\n".join(lines)
                data = json.loads(text)
                intent = (data.get("intent") or "general").lower()
                valid_intents = {
                    "forecast", "strategy", "backtest", "trade", "optimize",
                    "analyze", "create_model", "critique_backtest", "recommend_model", "general",
                }
                if intent not in valid_intents:
                    intent = "general"
                params = data.get("params") or {}
                # Resolve symbol from prompt: valid 1-5 letter ticker, exclude stopwords, map company names
                symbol = self._resolve_symbol_from_prompt(prompt)
                params["symbol"] = symbol
                params.setdefault("timeframe", "15d")
                params.setdefault("prompt", prompt)
                if not params.get("strategy"):
                    params["strategy"] = self._select_best_strategy(params["symbol"])
                if not params.get("model"):
                    params["model"] = self._select_best_model(
                        params["symbol"], params["timeframe"]
                    )
                params["create_new_model"] = bool(params.get("create_new_model", False))
                self.logger.info(f"Parsed prompt (Claude) - Intent: {intent}, Params: {params}")
                return intent, params
        except Exception as e:
            self.logger.debug(f"Claude intent parsing failed, using regex fallback: {e}")
        return self._parse_prompt_regex_fallback(prompt)

    def _parse_prompt_regex_fallback(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Regex/keyword fallback for intent and parameter extraction (original logic)."""
        prompt_lower = prompt.lower()

        # Extract symbol: valid ticker 1-5 uppercase letters, exclude common words, map company names
        symbol = self._resolve_symbol_from_prompt(prompt)

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
        elif any(word in prompt_lower for word in ["critique", "critic", "review"]) and ("backtest" in prompt_lower or "last" in prompt_lower or "result" in prompt_lower):
            intent = "critique_backtest"
        elif any(phrase in prompt_lower for phrase in ["what model", "which model", "model should i use", "model for", "best model"]):
            intent = "recommend_model"
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

        # Auto-select best strategy if none specified (need a symbol for selection)
        if not strategy:
            strategy = self._select_best_strategy(symbol or "AAPL")

        # Auto-select best model if none specified
        if not model:
            model = self._select_best_model(symbol or "AAPL", timeframe)

        params = {
            "symbol": symbol if symbol else None,
            "timeframe": timeframe,
            "strategy": strategy,
            "model": model,
            "create_new_model": create_new_model,
            "prompt": prompt,
        }

        self.logger.info(f"Parsed prompt - Intent: {intent}, Params: {params}")

        return intent, params

    def _resolve_symbol_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Resolve a single ticker from the prompt. Returns None if no valid ticker.
        Valid ticker = 1-5 uppercase letters; excludes common English stopwords;
        maps company names (apple, tesla, etc.) to their tickers.
        """
        prompt_lower = prompt.lower()
        for company, ticker in self._COMPANY_TO_TICKER.items():
            if re.search(r"\b" + re.escape(company) + r"\b", prompt_lower):
                return ticker

        # Prefer longer ticker-like tokens first (e.g., AAPL over A)
        raw_candidates = re.findall(r"\b([A-Z]{1,5})\b", prompt.upper())
        candidates = sorted(set(raw_candidates), key=len, reverse=True)
        for c in candidates:
            if c and c.lower() not in self._TICKER_STOPWORDS:
                return c
        return None

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

            # Create response with explicit price quotes
            message = f"Forecast for {symbol} using {model} model:\n"
            message += f"Horizon: {timeframe}\n"

            fc = np.asarray(forecast_result.get("forecast", []), dtype="float64").ravel()
            last_price = None
            try:
                if not getattr(data, "empty", True) and "close" in data.columns:
                    last_price = float(data["close"].iloc[-1])
                elif not getattr(data, "empty", True) and "Close" in data.columns:
                    last_price = float(data["Close"].iloc[-1])
                else:
                    last_price = None
            except Exception:
                last_price = None

            if last_price is not None:
                message += f"Last close: ${last_price:.2f}\n"

            if fc.size:
                # Quote a few concrete forecast points
                first = float(fc[0])
                mid = float(fc[min(2, fc.size - 1)])
                last = float(fc[-1])
                message += (
                    "Sample forecast prices: "
                    f"Day 1 ${first:.2f}, "
                    f"Day {min(3, fc.size)} ${mid:.2f}, "
                    f"Day {fc.size} ${last:.2f}\n"
                )

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
            self.logger.error(f"Error in forecast request: {e}")
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
            strategy_name = params["strategy"]

            # Get historical data and strategy signals, then evaluate via gatekeeper
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            data = self.data_provider.get_historical_data(
                symbol, start_date, end_date, "1d"
            )
            if data is None or data.empty:
                return AgentResponse(
                    success=False,
                    message=f"Unable to get data for {symbol}",
                    recommendations=["Try a different symbol or check data availability"],
                )
            data = data.copy()
            try:
                if hasattr(data.columns, "levels"):
                    data = data.droplevel(axis=1, level=0) if data.columns.nlevels > 1 else data
                data.columns = [str(c).lower().strip() for c in data.columns]
            except Exception:
                pass
            for col, default in [("close", "close"), ("open", "close"), ("high", "close"), ("low", "close"), ("volume", 1.0)]:
                if col not in data.columns and col != "close":
                    data[col] = data["close"] if "close" in data.columns else data.iloc[:, 0]
                elif col == "volume" and col not in data.columns:
                    data[col] = 1.0
            if "close" not in data.columns:
                data["close"] = data.iloc[:, 0]
            # Gatekeeper expects Close, High, Low, Volume
            data_ck = data.rename(columns={"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
            for c in ["Close", "High", "Low", "Volume"]:
                if c not in data_ck.columns and c.lower() in data.columns:
                    data_ck[c] = data[c.lower()]
            strategy_instance, _ = self._get_backtest_strategy(strategy_name or "RSI")
            if strategy_instance is None:
                return AgentResponse(
                    success=False,
                    message=f"Unknown strategy: {strategy_name}",
                    recommendations=["Use one of: RSI, Bollinger Bands, MACD, SMA Crossover"],
                )
            signals_df = strategy_instance.generate_signals(data)
            if signals_df is None or signals_df.empty:
                return AgentResponse(
                    success=False,
                    message=f"Strategy produced no signals for {symbol}",
                    recommendations=["Try a longer date range or different symbol"],
                )
            signal_col = "signal" if "signal" in signals_df.columns else (signals_df.columns[0] if len(signals_df.columns) > 0 else None)
            if signal_col is None:
                return AgentResponse(
                    success=False,
                    message="No signal column from strategy",
                    recommendations=["Check strategy implementation"],
                )
            decision = self.strategy_gatekeeper.evaluate_strategy(
                strategy_name or "RSI Mean Reversion", data_ck, signals_df
            )
            risk_level = "low" if decision.risk_score < 0.3 else "medium" if decision.risk_score < 0.6 else "high"
            strategy_analysis = {
                "health_score": decision.confidence,
                "risk_level": risk_level,
                "recommendations": decision.reasoning or [],
                "decision": getattr(decision.decision, "value", str(decision.decision)),
            }

            message = f"Strategy Analysis for {strategy_name} on {symbol}:\n"
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
            self.logger.error(f"Error in strategy request: {e}")
            return AgentResponse(
                success=False,
                message=f"Strategy analysis failed: {str(e)}",
                recommendations=[
                    "Try a different strategy or check symbol availability"
                ],
            )

    def _handle_backtest_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle backtest request.

        Uses strategy registry + generate_signals + metrics (same pattern as
        pages/2_Strategy_Testing). Backtester class has no instance run_backtest;
        module-level run_backtest expects a .run() that does not exist. See BACKTEST_FIX.md.
        """
        try:
            import pandas as pd

            symbol = params["symbol"]
            strategy_name = params.get("strategy") or "RSI"

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

            # Normalize to lowercase columns and ensure OHLCV (yfinance can return MultiIndex)
            data = data.copy()
            try:
                if hasattr(data.columns, "levels"):
                    data = data.droplevel(axis=1, level=0) if data.columns.nlevels > 1 else data
                data.columns = [str(c).lower().strip() for c in data.columns]
            except Exception:
                pass
            if "close" not in data.columns:
                for c in data.columns:
                    if "close" in str(c).lower():
                        data["close"] = data[c]
                        break
            if "close" not in data.columns:
                return AgentResponse(
                    success=False,
                    message=f"Data for {symbol} has no 'close' column",
                    recommendations=["Check data provider output format"],
                )
            for col, default in [("open", "close"), ("high", "close"), ("low", "close"), ("volume", 1.0)]:
                if col not in data.columns:
                    data[col] = data[default] if default != 1.0 else 1.0

            # Resolve strategy and run backtest (strategy -> signals -> metrics)
            strategy_instance, resolved_name = self._get_backtest_strategy(strategy_name)
            if strategy_instance is None:
                return AgentResponse(
                    success=False,
                    message=f"Unknown strategy: {strategy_name}",
                    recommendations=["Use one of: RSI, Bollinger Bands, MACD, SMA Crossover"],
                )

            signals_df = strategy_instance.generate_signals(data)
            if signals_df is None or signals_df.empty:
                return AgentResponse(
                    success=False,
                    message=f"Strategy {resolved_name} produced no signals for {symbol}",
                    recommendations=["Try a longer date range or different symbol"],
                )

            signal_col = "signal" if "signal" in signals_df.columns else (signals_df.columns[0] if len(signals_df.columns) > 0 else None)
            if signal_col is None:
                return AgentResponse(
                    success=False,
                    message=f"No signal column from {resolved_name}",
                    recommendations=["Check strategy implementation"],
                )

            # Metrics and equity curve (mirror pages/2_Strategy_Testing)
            initial_capital = 100000.0
            data = data.reindex(signals_df.index).ffill().bfill()
            data["returns"] = data["close"].pct_change()
            data["strategy_returns"] = signals_df[signal_col].shift(1) * data["returns"]
            data["strategy_returns"] = data["strategy_returns"].fillna(0)
            equity_curve = initial_capital * (1 + data["strategy_returns"]).cumprod()
            equity_curve = equity_curve.fillna(initial_capital)

            total_return = float((equity_curve.iloc[-1] / initial_capital - 1)) if len(equity_curve) > 0 else 0.0
            returns_series = data["strategy_returns"].replace(0, np.nan).dropna()
            if len(returns_series) > 0 and returns_series.std() > 0:
                sharpe = float((returns_series.mean() / returns_series.std()) * np.sqrt(252))
            else:
                sharpe = 0.0
            cumulative = (1 + data["strategy_returns"]).cumprod()
            running_max = cumulative.cummax()
            drawdown = cumulative / running_max - 1
            max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
            trades = returns_series[returns_series != 0]
            win_rate = float((trades > 0).sum() / len(trades)) if len(trades) > 0 else 0.0

            metrics = {
                "sharpe_ratio": sharpe,
                "total_return": total_return,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
            }

            equity_curve_df = pd.DataFrame({"equity_curve": equity_curve}, index=equity_curve.index)
            trade_log = signals_df[signals_df[signal_col] != 0].to_dict("records") if signal_col in signals_df.columns else []

            message = f"Backtest Results for {resolved_name} on {symbol}:\n"
            message += f"Sharpe Ratio: {sharpe:.2f}\n"
            message += f"Total Return: {total_return:.2%}\n"
            message += f"Max Drawdown: {max_dd:.2%}\n"
            message += f"Win Rate: {win_rate:.2%}\n"

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
                    "equity_curve": equity_curve_df,
                    "trade_log": trade_log,
                },
                recommendations=recommendations,
                next_actions=next_actions,
            )

        except Exception as e:
            self.logger.error(f"Error in backtest request: {e}")
            return AgentResponse(
                success=False,
                message=f"Backtest failed: {str(e)}",
                recommendations=["Check data availability and strategy parameters"],
            )

    def _handle_critique_backtest_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle 'critique my last backtest' — use agent_tools.critique_backtest."""
        try:
            from trading.services.agent_tools import critique_backtest

            metrics = params.get("metrics") or {}
            out = critique_backtest(metrics)
            if out.get("success"):
                msg = out.get("critique", "") or "No specific issues found."
                if out.get("suggestions"):
                    msg += "\n\nSuggestions:\n" + "\n".join(f"• {s}" for s in out["suggestions"])
                return AgentResponse(
                    success=True,
                    message=msg,
                    data=out,
                    recommendations=out.get("suggestions", []),
                    next_actions=["Run walk-forward validation", "Adjust parameters"],
                )
            return AgentResponse(
                success=False,
                message=out.get("error", "Critique unavailable.") or "Run a backtest first, then ask to critique it.",
                recommendations=out.get("suggestions", ["Run a backtest in Strategy Testing"]),
            )
        except Exception as e:
            self.logger.exception("Critique backtest failed: %s", e)
            return AgentResponse(
                success=False,
                message=f"Critique failed: {e}",
                recommendations=["Run a backtest in Strategy Testing, then ask again."],
            )

    def _handle_recommend_model_request(self, params: Dict[str, Any]) -> AgentResponse:
        """Handle 'what model should I use for X' — use agent_tools.recommend_model."""
        try:
            from trading.services.agent_tools import recommend_model

            symbol = params.get("symbol", "AAPL")
            timeframe = params.get("timeframe", "15d")
            days = 30
            try:
                days = int(str(timeframe).replace("d", "").replace("w", "").replace("m", ""))
                if "w" in str(timeframe).lower():
                    days = days * 7
                if "m" in str(timeframe).lower():
                    days = days * 30
            except Exception:
                pass
            horizon = "short_term" if days <= 14 else "medium_term" if days <= 60 else "long_term"
            out = recommend_model(horizon=horizon, n_data_points=min(500, max(100, days * 2)))
            if out.get("success") and out.get("model"):
                msg = f"For {symbol} (horizon ~{days}d), recommended model: **{out['model']}**"
                if out.get("confidence") is not None:
                    msg += f" (confidence: {out['confidence']:.2f})"
                msg += "."
                return AgentResponse(
                    success=True,
                    message=msg,
                    data=out,
                    recommendations=["Use Forecasting page to train and compare models"],
                    next_actions=["Run forecast with recommended model", "Compare with other models"],
                )
            fallback = out.get("fallback_models", ["LSTM", "ARIMA", "Prophet"])
            msg = f"Model suggestion for {symbol}: try {', '.join(fallback)}. " + (out.get("error") or "")
            return AgentResponse(
                success=True,
                message=msg,
                data=out,
                recommendations=["Use Model Lab to train and compare"],
                next_actions=["Open Forecasting or Model Lab"],
            )
        except Exception as e:
            self.logger.exception("Recommend model failed: %s", e)
            return AgentResponse(
                success=False,
                message=f"Model recommendation failed: {e}",
                recommendations=["Try: LSTM for short-term, Prophet for longer horizons"],
            )

    def _get_backtest_strategy(self, strategy_name: str) -> Tuple[Optional[Any], str]:
        """Resolve strategy name to (instance, resolved_name). Uses same strategies as 2_Strategy_Testing."""
        if not strategy_name or not isinstance(strategy_name, str):
            strategy_name = "RSI"
        name = strategy_name.strip()
        name_lower = name.lower()

        try:
            from trading.strategies.bollinger_strategy import BollingerStrategy, BollingerConfig
            from trading.strategies.macd_strategy import MACDStrategy, MACDConfig
            from trading.strategies.rsi_strategy import RSIStrategy
            from trading.strategies.sma_strategy import SMAStrategy, SMAConfig
        except ImportError as e:
            self.logger.warning(f"Strategy imports failed: {e}")
            return None, name

        # Map display names and aliases to (class, config_class, default_params)
        if "bollinger" in name_lower or "bollinger bands" in name_lower:
            return BollingerStrategy(BollingerConfig(window=20, num_std=2.0)), "Bollinger Bands"
        if "macd" in name_lower:
            return MACDStrategy(MACDConfig(fast_period=12, slow_period=26, signal_period=9)), "MACD"
        if "sma" in name_lower or "moving average" in name_lower or "crossover" in name_lower:
            return SMAStrategy(SMAConfig(short_window=20, long_window=50)), "SMA Crossover"
        if "rsi" in name_lower or "mean reversion" in name_lower:
            return RSIStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70), "RSI Mean Reversion"

        # Default
        return RSIStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70), "RSI Mean Reversion"

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
            self.logger.error(f"Error in trade request: {e}")
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
            self.logger.error(f"Error in optimization request: {e}")
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
            self.logger.error(f"Error in analysis request: {e}")
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
                    message="Model builder not available",
                    recommendations=["Please try using an existing model"],
                )

            # Extract model requirements from prompt
            prompt = params["prompt"]
            symbol = params["symbol"]

            # Infer model type from prompt
            prompt_lower = prompt.lower()
            if "xgboost" in prompt_lower or "gradient" in prompt_lower:
                model_type = "xgboost"
            elif "ensemble" in prompt_lower:
                model_type = "ensemble"
            else:
                model_type = "lstm"

            # Resolve data path for symbol (ModelBuilderAgent requires an existing file)
            possible_paths = [
                Path(f"data/{symbol}.csv"),
                Path("data") / f"{symbol}_prices.csv",
                Path("trading/data") / f"{symbol}.csv",
            ]
            data_path = None
            for p in possible_paths:
                if p.exists():
                    data_path = str(p)
                    break
            if not data_path:
                return AgentResponse(
                    success=False,
                    message=f"No price data file found for {symbol}. Add a CSV (e.g. data/{symbol}.csv) to use model creation.",
                    recommendations=[
                        "Export price data for the symbol to data/",
                        "Use an existing model instead",
                    ],
                )

            model_name = f"dynamic_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            request = ModelBuildRequest(
                model_type=model_type,
                data_path=data_path,
                target_column="Close",
                request_id=model_name,
            )
            result = self.model_creator.build_model(request)

            if result.build_status == "success":
                metrics = result.training_metrics or {}
                message = f"Successfully built model '{result.model_id}':\n"
                message += f"Framework: {result.framework}\n"
                message += f"Type: {result.model_type}\n"
                message += f"Path: {result.model_path}\n"
                if metrics:
                    message += f"Training metrics: {metrics}\n"

                return AgentResponse(
                    success=True,
                    message=message,
                    data={
                        "model_id": result.model_id,
                        "model_path": result.model_path,
                        "training_metrics": metrics,
                    },
                    recommendations=[],
                    next_actions=[
                        f"Test model on {symbol} data",
                        "Compare with existing models",
                    ],
                )
            else:
                return AgentResponse(
                    success=False,
                    message=f"Model build failed: {result.error_message or 'Unknown error'}",
                    recommendations=[
                        "Try a different model type or data",
                        "Use an existing model instead",
                    ],
                )

        except Exception as e:
            self.logger.error(f"Error in model creation request: {e}")
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

    def _get_rich_context(self, symbol: str) -> str:
        """Assemble earnings, short interest, insider flow, news, RSI/MACD, returns, 52wk range, P/E, 50/200 MA for ticker-specific answers."""
        parts: List[str] = []
        try:
            end_date = datetime.now()
            start_1y = end_date - timedelta(days=365)
            start_20d = end_date - timedelta(days=30)
            data = self.data_provider.get_historical_data(
                symbol, start_1y, end_date, "1d"
            )
            if data is not None and not data.empty:
                df = data.copy()
                if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
                    df = df.droplevel(axis=1, level=0)
                df.columns = [str(c).lower().strip() for c in df.columns]
                close_col = "close" if "close" in df.columns else next((c for c in df.columns if "close" in c.lower()), None)
                if close_col and len(df) >= 5:
                    closes = df[close_col].dropna()
                    if len(closes) >= 5:
                        ret_5d = (float(closes.iloc[-1]) / float(closes.iloc[-5]) - 1.0) * 100 if len(closes) >= 5 else None
                        ret_20d = (float(closes.iloc[-1]) / float(closes.iloc[-min(20, len(closes))]) - 1.0) * 100 if len(closes) >= 20 else None
                        parts.append(f"[Returns] 5-day: {ret_5d:+.2f}%" if ret_5d is not None else "")
                        parts.append(f"20-day: {ret_20d:+.2f}%" if ret_20d is not None else "")
                    if len(closes) >= 252:
                        high_52 = float(closes.tail(252).max())
                        low_52 = float(closes.tail(252).min())
                        parts.append(f"[52-week] High: ${high_52:.2f}, Low: ${low_52:.2f}")
                    if len(closes) >= 200:
                        ma50 = float(closes.tail(50).mean())
                        ma200 = float(closes.tail(200).mean())
                        last_p = float(closes.iloc[-1])
                        parts.append(f"[MAs] Price vs 50-day MA: {'above' if last_p >= ma50 else 'below'}; vs 200-day: {'above' if last_p >= ma200 else 'below'}")
                    try:
                        from trading.utils.safe_math import safe_rsi
                        rsi = safe_rsi(closes, period=14)
                        if rsi is not None and len(rsi) and not np.isnan(rsi.iloc[-1]):
                            parts.append(f"[RSI(14)] {float(rsi.iloc[-1]):.1f}")
                    except Exception:
                        pass
                    try:
                        ema12 = closes.ewm(span=12).mean()
                        ema26 = closes.ewm(span=26).mean()
                        macd = ema12 - ema26
                        if len(macd) and not np.isnan(macd.iloc[-1]):
                            parts.append(f"[MACD] {float(macd.iloc[-1]):.4f}")
                    except Exception:
                        pass
            # Earnings context (upcoming earnings and last EPS surprise)
            try:
                earnings = get_upcoming_earnings(symbol)
                if earnings.get("next_earnings_date"):
                    line = f"Next Earnings: {earnings['next_earnings_date']} ({earnings.get('days_until', '?')}d away)"
                    if earnings.get("eps_estimate") is not None:
                        try:
                            line += f", EPS est: ${earnings['eps_estimate']:.2f}"
                        except Exception:
                            pass
                    if earnings.get("last_eps_surprise_pct") is not None:
                        line += f", last surprise: {earnings['last_eps_surprise_pct']:+.1f}%"
                    parts.append(line)
            except Exception:
                pass

            # Short interest and squeeze risk
            try:
                si = get_short_interest(symbol)
                if si.get("short_pct_float"):
                    line = f"Short Interest: {si['short_pct_float']:.1f}% of float"
                    if si.get("short_ratio"):
                        line += f", {si['short_ratio']:.1f}d to cover"
                    if si.get("signal") == "HIGH_SHORT":
                        line += " [HIGH SHORT — squeeze potential]"
                    parts.append(line)
            except Exception:
                pass

            # Insider activity summary
            try:
                insider = get_insider_flow(symbol)
                buys = insider.get("buy_count", 0)
                sells = insider.get("sell_count", 0)
                sig = insider.get("signal")
                if buys + sells > 0 and sig not in ("NO_ACTIVITY", None):
                    line = f"Insider activity (90d): {buys} buys, {sells} sells → {sig}"
                    parts.append(line)
            except Exception:
                pass
            # AI Score
            try:
                from trading.analysis.ai_score import compute_ai_score
                _hist = None
                if data is not None and not data.empty and close_col and len(df) >= 20:
                    _hist = df.rename(columns={close_col: "Close"}).copy()
                    if "volume" in _hist.columns:
                        _hist = _hist.rename(columns={"volume": "Volume"})
                _ai = compute_ai_score(symbol, _hist)
                if _ai.get("error") is None:
                    parts.append(
                        f"AI Score: {_ai['overall_score']}/10 ({_ai['grade']}) — "
                        f"Technical {_ai['technical_score']}, Momentum {_ai['momentum_score']}, "
                        f"Sentiment {_ai['sentiment_score']}, Fundamental {_ai['fundamental_score']}"
                    )
            except Exception:
                pass
            # Recent news headlines (multi-source aggregator)
            try:
                news_items = get_news(symbol, max_items=5)
                if news_items:
                    parts.append("Recent News Headlines:")
                    for item in news_items[:5]:
                        age = ""
                        try:
                            pub_raw = item.get("published", "") or ""
                            if pub_raw:
                                s = pub_raw.replace("Z", "+00:00")
                                pub_dt = datetime.fromisoformat(s.split("+")[0])
                                hrs = int(
                                    (datetime.utcnow() - pub_dt).total_seconds() / 3600
                                )
                                if hrs < 48:
                                    age = f" ({hrs}h ago)"
                        except Exception:
                            pass
                        parts.append(
                            f"- [{item.get('source','?')}]"
                            f"{age}: {item.get('title','')}"
                        )
            except Exception:
                pass
            # Consensus forecast snapshot (cached; avoid heavy recompute every call)
            try:
                cache_key = f"consensus:{symbol}:7d"
                consensus = disk_cache_get(cache_key)
                if consensus is None:
                    from trading.models.forecast_router import ForecastRouter
                    import yfinance as yf

                    hist = yf.Ticker(symbol).history(period="6mo")
                    if hist is not None and not hist.empty:
                        router = ForecastRouter()
                        consensus = router.get_consensus_forecast(hist, horizon=7)
                        # Only cache successful results
                        if consensus and not consensus.get("error"):
                            disk_cache_set(cache_key, consensus, ttl=300)
                if consensus and not consensus.get("error"):
                    used = consensus.get("models_used") or []
                    direction = consensus.get("direction", "NEUTRAL")
                    conviction = consensus.get("conviction", "INSUFFICIENT")
                    price = consensus.get("consensus_price")
                    if price is None:
                        series = consensus.get("consensus_forecast") or []
                        price = series[-1] if series else None
                    if price is not None:
                        try:
                            parts.append(
                                f"Model consensus ({len(used)} models): {direction} at ${float(price):.2f} "
                                f"(conviction: {conviction})"
                            )
                        except Exception:
                            parts.append(
                                f"Model consensus ({len(used)} models): {direction} (conviction: {conviction})"
                            )
            except Exception:
                pass
            try:
                info = getattr(self.data_provider, "get_ticker_info", None)
                if callable(info):
                    inf = info(symbol)
                    if isinstance(inf, dict) and inf.get("trailingPE"):
                        parts.append(f"[P/E (trailing)] {inf['trailingPE']}")
            except Exception:
                pass
        except Exception as e:
            self.logger.debug("_get_rich_context failed for %s: %s", symbol, e)
        return "\n".join(p for p in parts if p).strip()

    def _handle_general_request_llm_only(
        self, prompt: str, params: Dict[str, Any]
    ) -> AgentResponse:
        """
        Respond with general market analysis using the active LLM.
        When a symbol is detected in the prompt, fetch real price data first and pass it to the LLM
        so the answer is grounded in actual numbers (e.g. 'why is apple down this week').
        """
        # Diagnostic tracing to confirm this path is hit from the chat pipeline
        try:
            symbol = self._resolve_symbol_from_prompt(prompt)
        except Exception:
            symbol = None
        logger.info("[CHAT DIAG 1] method called, symbol=%s", symbol)
        logger.info("[CHAT DIAG 1b] prompt preview=%s", (prompt or "")[:80])
        try:
            from agents.llm.active_llm_calls import call_active_llm_simple

            system = (
                "You are the Evolve trading platform assistant. "
                "You have access to real-time market data fetched live from yfinance. "
                "The data context below contains current price information fetched right now. "
                "Use these exact numbers in your response. "
                "Never say you lack access to current market data — you have it. "
                "Always cite the data source and note its freshness. "
                "You also have access to forecasting models (LSTM, XGBoost, ARIMA, Prophet, etc.) "
                "and recent forecast/backtest results stored in memory when available."
            )

            data_context = ""

            # For "forecast X" type questions, route to ForecastRouter and return actual model output
            prompt_lower = (prompt or "").lower()
            if symbol and ("forecast" in prompt_lower or "predict" in prompt_lower or "outlook" in prompt_lower):
                try:
                    router = ForecastRouter()
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)
                    data = self.data_provider.get_historical_data(symbol, start_date, end_date, "1d")
                    if data is not None and not getattr(data, "empty", True):
                        if hasattr(data.columns, "levels") and data.columns.nlevels > 1:
                            data = data.droplevel(axis=1, level=0)
                        result = router.get_forecast(data, horizon=7, model_type="auto")
                        fc = result.get("forecast")
                        if fc is not None and len(fc) > 0:
                            last_p = getattr(router, "_last_price_used", None) or (float(data["close"].iloc[-1]) if "close" in data.columns and len(data) > 0 else None)
                            summary = f"7-day forecast for {symbol}: last price ${last_p:.2f}; forecast (next 7 days): " + ", ".join(f"${x:.2f}" for x in fc[:7])
                            return AgentResponse(success=True, message=summary, data=result, recommendations=[], next_actions=[])
                except Exception as e:
                    self.logger.debug("ForecastRouter path failed for chat: %s", e)

            # Rich context (news, RSI, MACD, 5d/20d returns, 52wk, P/E, 50/200 MA) for ticker questions
            if symbol:
                rich = self._get_rich_context(symbol)
                if rich:
                    data_context += "\n\n[Rich context for " + symbol + "]\n" + rich

            # Check if multi-agent orchestration mode is enabled (Streamlit UI toggle)
            use_orchestration = False
            try:
                import streamlit as st  # type: ignore

                use_orchestration = bool(
                    st.session_state.get("agent_orchestration_mode", False)
                )
            except Exception:
                use_orchestration = False

            if use_orchestration and symbol:
                try:
                    from agents.orchestrator import AgentOrchestrator

                    orch = AgentOrchestrator(symbol, prompt, self.data_provider)
                    orch_result = orch.run()
                    data_context = orch_result.get("context", "") or ""
                    agents_used = orch_result.get("agents_used") or []
                    errors = orch_result.get("errors") or {}
                    if agents_used:
                        data_context += (
                            f"\n\n[Agents used: {', '.join(agents_used)}]"
                        )
                    if errors:
                        data_context += f"\n[Agent errors: {errors}]"
                except Exception as e:
                    self.logger.error(
                        "Orchestrator failed for symbol %s: %s", symbol, e
                    )
                    # Fall back to solo mode data fetch if orchestrator fails
                    use_orchestration = False

            if not use_orchestration and symbol:
                try:
                    # 1) Always fetch live price data first (freshest source)
                    live_price = self.data_provider.get_live_price(symbol)
                    live_price_block = ""
                    if live_price is not None:
                        try:
                            live_price_f = float(live_price)
                            live_price_block = (
                                f"\n\n[Live market data for {symbol} — fetched now]\n"
                                f"Current live price: ${live_price_f:.2f}\n"
                            )
                            self.logger.info(
                                "Live price fetch for %s: current_price=%.2f",
                                symbol,
                                live_price_f,
                            )
                        except Exception:
                            # Fall back to string formatting if casting fails
                            live_price_block = (
                                f"\n\n[Live market data for {symbol} — fetched now]\n"
                                f"Current live price: {live_price}\n"
                            )
                            self.logger.info(
                                "Live price fetch for %s (non-float): %s",
                                symbol,
                                str(live_price),
                            )
                    else:
                        self.logger.warning(
                            "Live price fetch for %s returned None", symbol
                        )

                    # 2) Recent historical window for context (trend/high/low)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=14)
                    data = self.data_provider.get_historical_data(
                        symbol, start_date, end_date, "1d"
                    )
                    history_block = ""
                    if data is not None and not data.empty:
                        df = data.copy()
                        if hasattr(df.columns, "levels"):
                            df = (
                                df.droplevel(axis=1, level=0)
                                if df.columns.nlevels > 1
                                else df
                            )
                        df.columns = [str(c).lower().strip() for c in df.columns]
                        close_col = (
                            "close"
                            if "close" in df.columns
                            else next(
                                (c for c in df.columns if "close" in c.lower()), None
                            )
                        )
                        if not close_col:
                            self.logger.debug(
                                "Chat data context: no close column for symbol %s; columns=%s",
                                symbol,
                                list(df.columns),
                            )
                        elif len(df) < 2:
                            self.logger.debug(
                                "Chat data context: not enough rows for symbol %s (rows=%d)",
                                symbol,
                                len(df),
                            )
                        else:
                            closes = df[close_col].dropna()
                            if len(closes) >= 2:
                                current_price_hist = float(closes.iloc[-1])
                                price_start = float(closes.iloc[0])
                                pct_change = (
                                    (current_price_hist - price_start)
                                    / price_start
                                    * 100
                                    if price_start
                                    else None
                                )
                                high_col = (
                                    "high"
                                    if "high" in df.columns
                                    else close_col
                                )
                                low_col = (
                                    "low" if "low" in df.columns else close_col
                                )
                                period_high = (
                                    float(df[high_col].max())
                                    if high_col in df.columns
                                    else current_price_hist
                                )
                                period_low = (
                                    float(df[low_col].min())
                                    if low_col in df.columns
                                    else current_price_hist
                                )
                                history_block = (
                                    f"[Historical window for {symbol} — last ~2 weeks]\n"
                                    f"Price at start of period: ${price_start:.2f}\n"
                                )
                                if pct_change is not None:
                                    history_block += (
                                        f"Period change: {pct_change:+.2f}%\n"
                                    )
                                history_block += (
                                    f"Period high: ${period_high:.2f}, "
                                    f"Period low: ${period_low:.2f}\n"
                                )
                            else:
                                self.logger.debug(
                                    "Chat data context: insufficient non-NaN closes for symbol %s (len=%d)",
                                    symbol,
                                    len(closes),
                                )
                    else:
                        self.logger.info(
                            "Chat data context: empty historical data for symbol %s from provider",
                            symbol,
                        )

                    data_context += live_price_block + ("\n" + history_block if history_block else "")

                    # 3) Recent news (always labeled as fetched now)
                    try:
                        from trading.data.news_fetcher import (
                            fetch_recent_news,
                            format_news_for_context,
                        )

                        headlines = fetch_recent_news(symbol, max_items=5)
                        if headlines:
                            data_context += (
                                f"\n[Recent news headlines — fetched now]\n"
                            )
                            data_context += format_news_for_context(headlines)
                    except Exception as ne:
                        self.logger.debug(
                            "Could not fetch news for LLM context: %s", ne
                        )

                    # 4) Enrich with recent MemoryStore context (forecasts/regime) if fresh
                    try:
                        from trading.memory import get_memory_store

                        store = get_memory_store()
                        now_utc = datetime.utcnow()

                        # Forecast freshness
                        recent_forecast = store.get_preference(f"forecast_{symbol}")
                        forecast_added = False
                        if isinstance(recent_forecast, dict):
                            ts_str = recent_forecast.get("timestamp")
                            try:
                                if ts_str:
                                    s = (
                                        str(ts_str)
                                        .replace("Z", "")
                                        .replace("+00:00", "")[:26]
                                    )
                                    ts_dt = datetime.fromisoformat(s)
                                    if getattr(ts_dt, "tzinfo", None):
                                        ts_dt = ts_dt.replace(tzinfo=None)
                                    minutes = (
                                        now_utc - ts_dt
                                    ).total_seconds() / 60.0
                                    if minutes <= 60:
                                        data_context += (
                                            f"\n[Forecast from this session ({int(minutes)} min ago)]\n"
                                            f"{recent_forecast.get('forecast') or recent_forecast}\n"
                                        )
                                        forecast_added = True
                            except Exception:
                                # Malformed timestamp; treat as stale
                                pass

                        if not forecast_added:
                            data_context += (
                                f"\n[Note: No recent forecast available for {symbol} — "
                                "run the Forecasting page for model predictions]\n"
                            )

                        # Market regime freshness
                        recent_regime = store.get_preference(
                            f"market_regime_{symbol}"
                        )
                        regime_added = False
                        if isinstance(recent_regime, dict):
                            ts_str = recent_regime.get("timestamp")
                            try:
                                if ts_str:
                                    s = (
                                        str(ts_str)
                                        .replace("Z", "")
                                        .replace("+00:00", "")[:26]
                                    )
                                    ts_dt = datetime.fromisoformat(s)
                                    if getattr(ts_dt, "tzinfo", None):
                                        ts_dt = ts_dt.replace(tzinfo=None)
                                    minutes = (
                                        now_utc - ts_dt
                                    ).total_seconds() / 60.0
                                    if minutes <= 60:
                                        data_context += (
                                            f"\n[Market regime analysis from this session ({int(minutes)} min ago)]\n"
                                            f"{recent_regime.get('regime') or recent_regime}\n"
                                        )
                                        regime_added = True
                            except Exception:
                                pass

                        if not regime_added:
                            data_context += (
                                f"\n[Note: No recent market regime analysis available for {symbol} — "
                                "run the Market Analysis or Strategy pages for regime diagnostics]\n"
                            )

                        # If neither forecast nor regime exists at all, add session-history note
                        if not recent_forecast and not recent_regime:
                            data_context += (
                                f"\n[Note: {symbol} has not been analyzed in this session. "
                                "Live price data shown above. For forecasts and strategy signals, "
                                "navigate to the Forecasting or Strategy pages.]\n"
                            )
                    except Exception as me:
                        self.logger.warning(
                            "Could not read MemoryStore for chat context: %s", me
                        )
                except Exception as e:
                    self.logger.debug(
                        "Could not build data context for symbol %s: %s", symbol, e
                    )

            # Log final context size so we can debug empty-context cases
            logger.info("[CHAT DIAG 2] data_context length=%d", len(data_context))
            self.logger.info(
                "Chat data context length: %d, symbol: %s",
                len(data_context),
                symbol or "None",
            )
            # For "why did X move" type: instruct to use ONLY the provided news and data
            if symbol and ("why" in prompt_lower and ("move" in prompt_lower or "recent" in prompt_lower or "change" in prompt_lower)):
                system += " When explaining price movement, use ONLY the specific news and data provided above. Do not speculate about general market factors."

            system_prompt = system + (data_context or "")
            logger.info("[CHAT DIAG 3] system_prompt preview=%s", (system_prompt[:200] if system_prompt else ""))
            logger.info("[CHAT DIAG 4] about to call LLM with provider=%s", os.getenv("LLM_PROVIDER"))

            full_prompt = f"{system}{data_context}\n\nUser: {prompt}\n\nAssistant:"
            text = call_active_llm_simple(full_prompt, max_tokens=1024)
            return AgentResponse(
                success=True,
                message=text or "I couldn't generate a response. Please try rephrasing.",
                data=None,
                recommendations=[],
                next_actions=[],
            )
        except Exception as e:
            self.logger.error(f"General LLM response failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Could not generate response: {str(e)}",
                recommendations=["Try asking with a specific ticker (e.g. AAPL) or rephrasing."],
            )

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
            # Run full pipeline: Forecast â†’ Strategy â†’ Backtest â†’ Report
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
                message += "ðŸ“ˆ FORECAST:\n"
                message += f"Model: {results['forecast']['model']}\n"
                message += f"Confidence: {results['forecast']['confidence']:.2%}\n\n"

            if "strategy" in results:
                message += "ðŸŽ¯ STRATEGY:\n"
                message += (
                    f"Health Score: {results['strategy'].get('health_score', 'N/A')}\n"
                )
                message += (
                    f"Risk Level: {results['strategy'].get('risk_level', 'N/A')}\n\n"
                )

            if "backtest" in results:
                metrics = results["backtest"]["metrics"]
                message += "ðŸ“Š BACKTEST RESULTS:\n"
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
                        "âœ… Strategy performing well - consider live trading"
                    )
                elif sharpe >= 0.5:
                    recommendations.append(
                        "âš ï¸Œ Strategy needs optimization - run parameter tuning"
                    )
                else:
                    recommendations.append(
                        "âŒ Strategy underperforming - try alternative approach"
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
            self.logger.error(f"Error in general request: {e}")
            return AgentResponse(
                success=False,
                message=f"Analysis failed: {str(e)}",
                recommendations=["Try breaking down the request into smaller parts"],
            )


# Global prompt agent instance (lazy; avoid module-level init that pulls in ForecastRouter/ModelRegistry and can trigger Windows Unicode errors)
prompt_agent = None
try:
    prompt_agent = PromptAgent()
except Exception as e:
    import logging
    _log = logging.getLogger(__name__)
    _log.warning("PromptAgent init failed: %s", e)


def get_prompt_agent():
    """Get the global prompt agent instance. May be None if init failed."""
    return prompt_agent
