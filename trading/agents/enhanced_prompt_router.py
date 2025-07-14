"""
Enhanced PromptRouterAgent: Smart prompt router with comprehensive fallback logic.
- Detects user intent (forecasting, backtesting, tuning, research)
- Parses arguments using OpenAI, HuggingFace, or regex fallback
- Routes to the correct agent automatically
- Always returns a usable parsed intent
- Enhanced with named intent templates and JSON spec debugging
"""

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import openai
except ImportError:
    openai = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from prompt_templates import format_template

logger = logging.getLogger(__name__)


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


class EnhancedPromptRouterAgent:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        huggingface_model: str = "gpt2",
        huggingface_api_key: Optional[str] = None,
        enable_debug_mode: bool = False,
    ):
        """
        Initialize the enhanced prompt router agent.

        Args:
            openai_api_key: OpenAI API key
            huggingface_model: HuggingFace model name
            huggingface_api_key: HuggingFace API key
            enable_debug_mode: Enable JSON spec return for debugging
        """
        self.openai_api_key = openai_api_key
        self.huggingface_model = huggingface_model
        self.huggingface_api_key = huggingface_api_key
        self.hf_pipeline = None
        self.enable_debug_mode = enable_debug_mode

        # Initialize OpenAI if available
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("✅ OpenAI initialized for prompt routing")

        # Initialize HuggingFace if available
        if HUGGINGFACE_AVAILABLE:
            try:
                self._init_huggingface()
                logger.info("✅ HuggingFace initialized for prompt routing")
            except Exception as e:
                logger.warning(f"⚠️ HuggingFace initialization failed: {e}")

        # Enhanced intent keywords for regex fallback with ambiguous prompt handling
        self.intent_keywords = {
            "forecasting": [
                "forecast",
                "predict",
                "projection",
                "future",
                "price",
                "trend",
                "outlook",
            ],
            "backtesting": [
                "backtest",
                "historical",
                "simulate",
                "performance",
                "past",
                "test",
                "simulation",
            ],
            "tuning": [
                "tune",
                "optimize",
                "hyperparameter",
                "search",
                "bayesian",
                "parameter",
                "improve",
            ],
            "research": [
                "research",
                "find",
                "paper",
                "github",
                "arxiv",
                "summarize",
                "analyze",
                "study",
            ],
            "portfolio": [
                "portfolio",
                "position",
                "holdings",
                "allocation",
                "balance",
                "asset",
            ],
            "risk": [
                "risk",
                "volatility",
                "drawdown",
                "var",
                "sharpe",
                "danger",
                "exposure",
            ],
            "sentiment": [
                "sentiment",
                "news",
                "social",
                "twitter",
                "reddit",
                "emotion",
                "mood",
            ],
            "compare_strategies": [
                "compare",
                "comparison",
                "versus",
                "vs",
                "against",
                "different",
                "strategy",
            ],
            "optimize_model": [
                "optimize",
                "improve",
                "enhance",
                "tune",
                "model",
                "performance",
            ],
            "debug_forecast": [
                "debug",
                "fix",
                "error",
                "issue",
                "problem",
                "forecast",
                "prediction",
            ],
        }

        # Named intent templates for specific actions
        self.named_intent_templates = {
            "compare_strategies": {
                "keywords": [
                    "compare",
                    "comparison",
                    "versus",
                    "vs",
                    "against",
                    "different",
                ],
                "required_args": ["strategies"],
                "optional_args": ["timeframe", "metrics", "period"],
            },
            "optimize_model": {
                "keywords": ["optimize", "improve", "enhance", "tune", "model"],
                "required_args": ["model"],
                "optional_args": ["parameters", "objective", "constraints"],
            },
            "debug_forecast": {
                "keywords": ["debug", "fix", "error", "issue", "problem", "forecast"],
                "required_args": ["forecast"],
                "optional_args": ["error_type", "timeframe", "model"],
            },
        }

        # Enhanced argument extraction patterns for ambiguous prompts
        self.arg_patterns = {
            "symbol": r"\b([A-Z]{1,5})\b",
            "date": r"\b(\d{4}-\d{2}-\d{2})\b",
            "number": r"\b(\d+(?:\.\d+)?)\b",
            "percentage": r"\b(\d+(?:\.\d+)?%)\b",
            "timeframe": r"\b(daily|weekly|monthly|yearly|1d|1w|1m|1y|this week|this month|this year)\b",
            "model": r"\b(lstm|transformer|xgboost|prophet|arima|ensemble|gpt|bert)\b",
            "strategy": r"\b(momentum|mean_reversion|bollinger|macd|rsi|sma|ema|bollinger bands)\b",
            "action": r"\b(buy|sell|hold|long|short|exit|enter)\b",
            "comparison": r"\b(compare|versus|vs|against|better|worse|best|worst)\b",
            "optimization": r"\b(optimize|improve|enhance|tune|best|optimal)\b",
            "debug": r"\b(debug|fix|error|issue|problem|wrong|broken)\b",
        }

        # Cache for HuggingFace model responses
        self.hf_cache = {}
        self.cache_file = "trading/agents/hf_cache.pkl"
        self._load_cache()

    def _init_huggingface(self):
        """Initialize HuggingFace pipeline with caching."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model)
            model = AutoModelForCausalLM.from_pretrained(self.huggingface_model)

            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.hf_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {e}")
            self.hf_pipeline = None

    def _load_cache(self):
        """Load HuggingFace cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    self.hf_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.hf_cache)} cached responses")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.hf_cache = {}

    def _save_cache(self):
        """Save HuggingFace cache to file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.hf_cache, f)
            logger.info(f"Saved {len(self.hf_cache)} cached responses")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def parse_intent_openai(self, prompt: str) -> Optional[ParsedIntent]:
        """Parse intent using OpenAI GPT-4 with enhanced JSON spec."""
        if not openai or not self.openai_api_key:
            return None

        try:
            # Use enhanced template for intent classification
            system_prompt = format_template(
                "enhanced_intent_classification", query=prompt
            )

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )

            content = response.choices[0].message["content"].strip()

            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))

                # Create JSON spec for debugging if enabled
                json_spec = None
                if self.enable_debug_mode:
                    json_spec = {
                        "raw_response": content,
                        "parsed_json": parsed,
                        "provider": "openai",
                        "model": "gpt-4",
                        "timestamp": datetime.now().isoformat(),
                        "prompt_length": len(prompt),
                        "response_length": len(content),
                    }

                return ParsedIntent(
                    intent=parsed.get("intent", "unknown"),
                    confidence=parsed.get("confidence", 0.8),
                    args=parsed.get("args", {}),
                    provider="openai",
                    raw_response=content,
                    json_spec=json_spec,
                )

            # Fallback: extract intent from text
            intent = self._extract_intent_from_text(content)
            return ParsedIntent(
                intent=intent,
                confidence=0.7,
                args=self._extract_args_regex(prompt),
                provider="openai",
                raw_response=content,
            )

        except Exception as e:
            logger.warning(f"OpenAI parsing failed: {e}")
            return None

    def parse_intent_huggingface(self, prompt: str) -> Optional[ParsedIntent]:
        """Parse intent using HuggingFace model with caching."""
        if not self.hf_pipeline:
            return None

        # Check cache first
        cache_key = hash(prompt)
        if cache_key in self.hf_cache:
            logger.info("Using cached HuggingFace response")
            return self.hf_cache[cache_key]

        try:
            # Use enhanced template for intent extraction
            structured_prompt = format_template(
                "enhanced_intent_extraction", user_input=prompt
            )

            response = self.hf_pipeline(
                structured_prompt,
                max_length=150,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
            )

            generated_text = response[0]["generated_text"]

            # Extract intent from generated text
            intent = self._extract_intent_from_text(generated_text)
            args = self._extract_args_regex(prompt)

            result = ParsedIntent(
                intent=intent,
                confidence=0.6,  # Lower confidence for HF
                args=args,
                provider="huggingface",
                raw_response=generated_text,
            )

            # Cache the result
            self.hf_cache[cache_key] = result
            self._save_cache()

            return result

        except Exception as e:
            logger.warning(f"HuggingFace parsing failed: {e}")
            return None

    def parse_intent_regex(self, prompt: str) -> ParsedIntent:
        """Parse intent using enhanced regex and keyword matching for ambiguous prompts."""
        prompt_lower = prompt.lower()

        # Enhanced intent detection with named templates
        best_intent = "unknown"
        best_score = 0
        matched_template = None

        # Check named intent templates first
        for intent_name, template in self.named_intent_templates.items():
            keyword_matches = sum(
                1 for keyword in template["keywords"] if keyword in prompt_lower
            )
            if keyword_matches > best_score:
                best_score = keyword_matches
                best_intent = intent_name
                matched_template = template

        # If no named template matched, check general intents
        if best_score == 0:
            for intent, keywords in self.intent_keywords.items():
                if intent not in self.named_intent_templates:  # Skip named templates
                    score = sum(1 for keyword in keywords if keyword in prompt_lower)
                    if score > best_score:
                        best_score = score
                        best_intent = intent

        # Extract arguments using enhanced regex patterns
        args = self._extract_args_regex(prompt)

        # Calculate confidence based on keyword matches and template validation
        confidence = min(0.5 + (best_score * 0.1), 0.9)

        # Validate against template requirements if matched
        if matched_template:
            required_args = matched_template["required_args"]
            missing_args = [arg for arg in required_args if arg not in args]
            if missing_args:
                confidence *= 0.8  # Reduce confidence for missing required args

        # Create JSON spec for debugging if enabled
        json_spec = None
        if self.enable_debug_mode:
            json_spec = {
                "matched_template": matched_template,
                "keyword_matches": best_score,
                "extracted_args": args,
                "provider": "regex",
                "timestamp": datetime.now().isoformat(),
                "prompt_length": len(prompt),
                "confidence_factors": {
                    "keyword_matches": best_score,
                    "template_validation": matched_template is not None,
                    "missing_required_args": missing_args if matched_template else [],
                },
            }

        return ParsedIntent(
            intent=best_intent,
            confidence=confidence,
            args=args,
            provider="regex",
            raw_response=f"Regex matched intent: {best_intent} with {best_score} keyword matches",
            json_spec=json_spec,
        )

    def _extract_intent_from_text(self, text: str) -> str:
        """Extract intent from text using keyword matching."""
        text_lower = text.lower()

        for intent, keywords in self.intent_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent

        return "unknown"

    def _extract_args_regex(self, prompt: str) -> Dict[str, Any]:
        """Extract arguments using enhanced regex patterns for ambiguous prompts."""
        args = {}

        for arg_name, pattern in self.arg_patterns.items():
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                if len(matches) == 1:
                    args[arg_name] = matches[0]
                else:
                    args[arg_name] = matches

        # Enhanced context extraction for ambiguous prompts
        if "forecast" in prompt.lower() or "predict" in prompt.lower():
            if "price" in prompt.lower():
                args["target"] = "price"
            elif "return" in prompt.lower():
                args["target"] = "return"

        # Extract time periods with more flexible patterns
        period_patterns = [
            r"(\d+)\s*(days?|weeks?|months?|years?)",
            r"(this|next|last)\s+(week|month|year)",
            r"(today|tomorrow|yesterday)",
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        ]

        for pattern in period_patterns:
            matches = re.findall(pattern, prompt.lower())
            if matches:
                args["period"] = (
                    matches[0] if isinstance(matches[0], str) else " ".join(matches[0])
                )
                break

        # Extract comparison context
        if any(
            word in prompt.lower() for word in ["compare", "versus", "vs", "against"]
        ):
            args["comparison_mode"] = True
            # Try to extract what's being compared
            comparison_pattern = (
                r"(compare|versus|vs|against)\s+([^,]+?)\s+(?:with|to|and)\s+([^,]+)"
            )
            comp_match = re.search(comparison_pattern, prompt.lower())
            if comp_match:
                args["compare_items"] = [
                    comp_match.group(2).strip(),
                    comp_match.group(3).strip(),
                ]

        return args

    def parse_intent(self, prompt: str) -> ParsedIntent:
        """
        Parse intent with enhanced fallback logic and debugging support.

        Args:
            prompt: User prompt to parse

        Returns:
            ParsedIntent object with intent, confidence, arguments, and optional JSON spec
        """
        logger.info(f"Parsing intent for prompt: {prompt[:100]}...")

        # Try OpenAI first
        if openai and self.openai_api_key:
            result = self.parse_intent_openai(prompt)
            if result and result.intent != "unknown":
                logger.info(
                    f"✅ OpenAI parsed intent: {result.intent} (confidence: {result.confidence})"
                )
                return result

        # Try HuggingFace second
        if self.hf_pipeline:
            result = self.parse_intent_huggingface(prompt)
            if result and result.intent != "unknown":
                logger.info(
                    f"✅ HuggingFace parsed intent: {result.intent} (confidence: {result.confidence})"
                )
                return result

        # Fallback to enhanced regex
        result = self.parse_intent_regex(prompt)
        logger.info(
            f"✅ Regex parsed intent: {result.intent} (confidence: {result.confidence})"
        )
        return result

    def route(self, prompt: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route the prompt to the correct agent based on intent and arguments.

        Args:
            prompt: User prompt
            agents: Dictionary of available agents

        Returns:
            Dictionary with routing result and metadata
        """
        parsed_intent = self.parse_intent(prompt)

        logger.info(
            f"Routing intent: {parsed_intent.intent}, args: {parsed_intent.args}"
        )

        # Route to appropriate agent
        result = {
            "intent": parsed_intent.intent,
            "confidence": parsed_intent.confidence,
            "args": parsed_intent.args,
            "provider": parsed_intent.provider,
            "raw_response": parsed_intent.raw_response,
            "timestamp": str(datetime.now()),
            "success": False,
            "error": None,
            "result": None,
        }

        # Add JSON spec for debugging if enabled
        if self.enable_debug_mode and parsed_intent.json_spec:
            result["debug_spec"] = parsed_intent.json_spec

        # Route based on intent
        if parsed_intent.intent == "compare_strategies":
            result["routed_agent"] = "strategy_comparison_agent"
            result["routing_reason"] = "Strategy comparison requested"
        elif parsed_intent.intent == "optimize_model":
            result["routed_agent"] = "model_optimization_agent"
            result["routing_reason"] = "Model optimization requested"
        elif parsed_intent.intent == "debug_forecast":
            result["routed_agent"] = "forecast_debug_agent"
            result["routing_reason"] = "Forecast debugging requested"
        elif parsed_intent.intent in agents:
            result["routed_agent"] = parsed_intent.intent
            result["routing_reason"] = f"Direct intent match: {parsed_intent.intent}"
        else:
            result["routed_agent"] = "general_agent"
            result["routing_reason"] = "No specific agent found, using general agent"

        return result

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information about the router."""
        return {
            "cache_size": len(self.hf_cache),
            "available_providers": {
                "openai": openai is not None and self.openai_api_key is not None,
                "huggingface": self.hf_pipeline is not None,
                "regex": True,
            },
            "named_templates": list(self.named_intent_templates.keys()),
            "intent_keywords": {k: len(v) for k, v in self.intent_keywords.items()},
            "arg_patterns": list(self.arg_patterns.keys()),
            "debug_mode": self.enable_debug_mode,
        }
