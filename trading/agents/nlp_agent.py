"""
NLP Agent with Transformers and spaCy

Advanced NLP agent using transformers and spaCy for prompt parsing and model routing.
Provides intelligent routing to appropriate trading models and strategies.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import spaCy
try:
    import spacy

    SPACY_AVAILABLE = True
    # Load English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print(
            "⚠️ spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
        )
        SPACY_AVAILABLE = False
except ImportError as e:
    print("⚠️ spaCy not available. Disabling spaCy-based NLP features.")
    print(f"   Missing: {e}")
    spacy = None
    SPACY_AVAILABLE = False

# Try to import transformers
try:
    from transformers import AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print("⚠️ transformers not available. Disabling transformer-based NLP features.")
    print(f"   Missing: {e}")
    AutoTokenizer = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

# Try to import TextBlob
try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError as e:
    print("⚠️ TextBlob not available. Disabling TextBlob-based features.")
    print(f"   Missing: {e}")
    TextBlob = None
    TEXTBLOB_AVAILABLE = False

from trading.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class NLPRequest:
    """Request for NLP processing."""

    prompt: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NLPResult:
    """Result of NLP processing."""

    original_prompt: str
    timestamp: datetime
    intent: Dict[str, Any]
    sentiment: Dict[str, Any]
    entities: Dict[str, List[str]]
    strategy_suggestions: List[Dict[str, Any]]
    market_regime: Dict[str, Any]
    timeframe: Dict[str, Any]
    tickers: List[str]
    confidence: float
    routing: Dict[str, Any]
    error: Optional[str] = None


class NLPAgent:
    """Advanced NLP agent for trading prompt parsing and model routing."""

    def __init__(self):
        """Initialize the NLP agent."""
        self.cache_dir = Path("cache/nlp_agent")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize NLP components
        self.nlp = None
        self.sentiment_analyzer = None
        self.intent_classifier = None
        self.entity_extractor = None

        self._initialize_nlp_components()

        # Define trading intents and their patterns
        self.intent_patterns = {
            "forecast": [
                r"forecast",
                r"predict",
                r"future",
                r"outlook",
                r"trend",
                r"what will",
                r"going to",
                r"expected",
                r"projection",
            ],
            "analyze": [
                r"analyze",
                r"analysis",
                r"study",
                r"examine",
                r"review",
                r"look at",
                r"check",
                r"investigate",
                r"assess",
            ],
            "trade": [
                r"trade",
                r"buy",
                r"sell",
                r"position",
                r"entry",
                r"exit",
                r"long",
                r"short",
                r"order",
                r"execute",
            ],
            "risk": [
                r"risk",
                r"volatility",
                r"drawdown",
                r"loss",
                r"exposure",
                r"hedge",
                r"protect",
                r"safe",
                r"conservative",
            ],
            "portfolio": [
                r"portfolio",
                r"allocation",
                r"weight",
                r"diversify",
                r"balance",
                r"rebalance",
                r"optimize",
            ],
        }

        # Define strategy keywords
        self.strategy_keywords = {
            "momentum": ["momentum", "trend", "breakout", "moving average", "rsi"],
            "mean_reversion": ["mean reversion", "oversold", "overbought", "bollinger"],
            "arbitrage": ["arbitrage", "pairs", "spread", "correlation"],
            "fundamental": ["fundamental", "earnings", "pe ratio", "dividend"],
            "sentiment": ["sentiment", "news", "social", "reddit", "twitter"],
        }

        # Define market regime keywords
        self.market_regime_keywords = {
            "bullish": ["bullish", "uptrend", "rally", "bull market", "positive"],
            "bearish": ["bearish", "downtrend", "decline", "bear market", "negative"],
            "sideways": ["sideways", "range", "consolidation", "flat"],
            "volatile": ["volatile", "choppy", "unstable", "wild"],
        }

        logger.info("NLP Agent initialized")

    def _initialize_nlp_components(self):
        """Initialize NLP components with fallback handling."""
        try:
            # Initialize spaCy
            if SPACY_AVAILABLE:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")

            # Initialize sentiment analyzer
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        return_all_scores=True,
                    )
                    logger.info("Transformers sentiment analyzer loaded")
                except Exception as e:
                    logger.warning(
                        f"Failed to load transformers sentiment analyzer: {e}"
                    )
                    self.sentiment_analyzer = None

            # Initialize intent classifier
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model="facebook/bart-large-mnli",
                        return_all_scores=True,
                    )
                    logger.info("Transformers intent classifier loaded")
                except Exception as e:
                    logger.warning(
                        f"Failed to load transformers intent classifier: {e}"
                    )
                    self.intent_classifier = None

        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")

    def parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse a trading prompt and extract structured information.

        Args:
            prompt: User prompt text

        Returns:
            Dictionary with parsed information
        """
        try:
            if not prompt or not prompt.strip():
                return {"error": "Empty prompt provided"}

            # Log token count to monitor cost/performance tradeoffs
            if TRANSFORMERS_AVAILABLE:
                try:
                    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                    token_count = len(tokenizer.tokenize(prompt))
                    logger.info(f"Token count for input: {token_count}")
                except Exception as e:
                    logger.warning(f"Could not count tokens: {e}")

            prompt = prompt.strip().lower()

            # Detect if prompt contains financial jargon
            jargon_detection = self._detect_financial_jargon(prompt)

            # Extract basic information
            result = {
                "original_prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "jargon_detection": jargon_detection,
                "intent": self._classify_intent(prompt),
                "sentiment": self._analyze_sentiment(prompt),
                "entities": self._extract_entities(prompt),
                "strategy_suggestions": self._suggest_strategies(prompt),
                "market_regime": self._classify_market_regime(prompt),
                "timeframe": self._extract_timeframe(prompt),
                "tickers": self._extract_tickers(prompt),
                "confidence": self._calculate_confidence(prompt),
                "routing": self._generate_routing_recommendations(
                    {"prompt": prompt, "jargon_detection": jargon_detection}
                ),
            }

            # Cache the result
            self._cache_parsing_result(prompt, result)

            return result

        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            return {"error": str(e)}

    def _classify_intent(self, prompt: str) -> Dict[str, Any]:
        """Classify the intent of the prompt."""
        try:
            intent_scores = {}

            # Rule-based classification
            for intent, patterns in self.intent_patterns.items():
                score = 0
                for pattern in patterns:
                    if re.search(pattern, prompt):
                        score += 1
                intent_scores[intent] = score / len(patterns)

            # Transformers-based classification (if available)
            if self.intent_classifier:
                try:
                    transformer_result = self.intent_classifier(prompt)
                    # Map transformer labels to our intents
                    for result in transformer_result:
                        label = result["label"].lower()
                        score = result["score"]
                        if "forecast" in label or "predict" in label:
                            intent_scores["forecast"] = max(
                                intent_scores.get("forecast", 0), score
                            )
                        elif "analyze" in label or "study" in label:
                            intent_scores["analyze"] = max(
                                intent_scores.get("analyze", 0), score
                            )
                        elif "trade" in label or "buy" in label or "sell" in label:
                            intent_scores["trade"] = max(
                                intent_scores.get("trade", 0), score
                            )
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Transformers intent classification failed: {e}")

            # Get primary intent
            primary_intent = (
                max(intent_scores.items(), key=lambda x: x[1])
                if intent_scores
                else ("unknown", 0)
            )

            return {
                "primary": primary_intent[0],
                "confidence": primary_intent[1],
                "all_scores": intent_scores,
            }

        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {"primary": "unknown", "confidence": 0, "all_scores": {}}

    def _analyze_sentiment(self, prompt: str) -> Dict[str, Any]:
        """Analyze sentiment of the prompt."""
        try:
            sentiment_result = {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "neutral",
                "confidence": 0.0,
            }

            # TextBlob sentiment analysis
            if TEXTBLOB_AVAILABLE:
                try:
                    blob = TextBlob(prompt)
                    sentiment_result["polarity"] = blob.sentiment.polarity
                    sentiment_result["subjectivity"] = blob.sentiment.subjectivity

                    # Map polarity to label
                    if sentiment_result["polarity"] > 0.1:
                        sentiment_result["label"] = "positive"
                    elif sentiment_result["polarity"] < -0.1:
                        sentiment_result["label"] = "negative"
                    else:
                        sentiment_result["label"] = "neutral"

                    sentiment_result["confidence"] = abs(sentiment_result["polarity"])
                except Exception as e:
                    logger.warning(f"TextBlob sentiment analysis failed: {e}")

            # Transformers sentiment analysis
            if self.sentiment_analyzer:
                try:
                    transformer_result = self.sentiment_analyzer(prompt)
                    # Get the highest scoring sentiment
                    best_sentiment = max(
                        transformer_result[0], key=lambda x: x["score"]
                    )
                    sentiment_result["transformer_label"] = best_sentiment["label"]
                    sentiment_result["transformer_confidence"] = best_sentiment["score"]
                except Exception as e:
                    logger.warning(f"Transformers sentiment analysis failed: {e}")

            return sentiment_result

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "neutral",
                "confidence": 0.0,
            }

    def _extract_entities(self, prompt: str) -> Dict[str, List[str]]:
        """Extract named entities from the prompt."""
        try:
            entities = {
                "tickers": [],
                "companies": [],
                "dates": [],
                "numbers": [],
                "currencies": [],
            }

            # spaCy entity extraction
            if self.nlp:
                try:
                    doc = self.nlp(prompt)
                    for ent in doc.ents:
                        if ent.label_ == "ORG":
                            entities["companies"].append(ent.text)
                        elif ent.label_ == "DATE":
                            entities["dates"].append(ent.text)
                        elif ent.label_ == "MONEY":
                            entities["currencies"].append(ent.text)
                        elif ent.label_ == "CARDINAL":
                            entities["numbers"].append(ent.text)
                except Exception as e:
                    logger.warning(f"spaCy entity extraction failed: {e}")

            # Regex-based ticker extraction
            ticker_pattern = r"\b[A-Z]{1,5}\b"
            potential_tickers = re.findall(ticker_pattern, prompt.upper())
            entities["tickers"] = [
                ticker for ticker in potential_tickers if len(ticker) >= 2
            ]

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {
                "tickers": [],
                "companies": [],
                "dates": [],
                "numbers": [],
                "currencies": [],
            }

    def _suggest_strategies(self, prompt: str) -> List[Dict[str, Any]]:
        """Suggest trading strategies based on prompt content."""
        try:
            suggestions = []

            for strategy, keywords in self.strategy_keywords.items():
                score = 0
                matched_keywords = []

                for keyword in keywords:
                    if keyword in prompt:
                        score += 1
                        matched_keywords.append(keyword)

                if score > 0:
                    suggestions.append(
                        {
                            "strategy": strategy,
                            "relevance_score": score / len(keywords),
                            "matched_keywords": matched_keywords,
                            "description": self._get_strategy_description(strategy),
                        }
                    )

            # Sort by relevance score
            suggestions.sort(key=lambda x: x["relevance_score"], reverse=True)

            return suggestions

        except Exception as e:
            logger.error(f"Error suggesting strategies: {e}")
            return []

    def _classify_market_regime(self, prompt: str) -> Dict[str, Any]:
        """Classify market regime mentioned in the prompt."""
        try:
            regime_scores = {}

            for regime, keywords in self.market_regime_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in prompt:
                        score += 1
                regime_scores[regime] = score / len(keywords)

            primary_regime = (
                max(regime_scores.items(), key=lambda x: x[1])
                if regime_scores
                else ("unknown", 0)
            )

            return {
                "primary": primary_regime[0],
                "confidence": primary_regime[1],
                "all_scores": regime_scores,
            }

        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return {"primary": "unknown", "confidence": 0, "all_scores": {}}

    def _extract_timeframe(self, prompt: str) -> Dict[str, Any]:
        """Extract timeframe information from the prompt."""
        try:
            timeframe_patterns = {
                "short_term": [
                    r"\b(day|daily|intraday|hour|minute)\b",
                    r"\b(1d|1h|1m|5m|15m)\b",
                ],
                "medium_term": [
                    r"\b(week|weekly|month|monthly)\b",
                    r"\b(1w|1mo|3mo)\b",
                ],
                "long_term": [
                    r"\b(year|yearly|annual|long term)\b",
                    r"\b(1y|5y|10y)\b",
                ],
            }

            timeframe_scores = {}
            for timeframe, patterns in timeframe_patterns.items():
                score = 0
                for pattern in patterns:
                    if re.search(pattern, prompt):
                        score += 1
                timeframe_scores[timeframe] = score / len(patterns)

            primary_timeframe = (
                max(timeframe_scores.items(), key=lambda x: x[1])
                if timeframe_scores
                else ("unknown", 0)
            )

            return {
                "primary": primary_timeframe[0],
                "confidence": primary_timeframe[1],
                "all_scores": timeframe_scores,
            }

        except Exception as e:
            logger.error(f"Error extracting timeframe: {e}")
            return {"primary": "unknown", "confidence": 0, "all_scores": {}}

    def _extract_tickers(self, prompt: str) -> List[str]:
        """Extract stock tickers from the prompt."""
        try:
            # Enhanced ticker extraction
            ticker_patterns = [
                r"\b[A-Z]{1,5}\b",  # Basic ticker pattern
                r"\$([A-Z]{1,5})\b",  # $TICKER format
                r"\b([A-Z]{1,5})\s+stock\b",  # TICKER stock format
            ]

            tickers = set()
            for pattern in ticker_patterns:
                matches = re.findall(pattern, prompt.upper())
                tickers.update(matches)

            # Filter out common words that might be mistaken for tickers
            common_words = {
                "THE",
                "AND",
                "FOR",
                "ARE",
                "BUT",
                "NOT",
                "YOU",
                "ALL",
                "CAN",
                "HAD",
                "HER",
                "WAS",
                "ONE",
                "OUR",
                "OUT",
                "DAY",
                "HAS",
                "HIM",
                "HIS",
                "HOW",
                "MAN",
                "NEW",
                "NOW",
                "OLD",
                "SEE",
                "TWO",
                "WAY",
                "WHO",
                "BOY",
                "DID",
                "ITS",
                "LET",
                "PUT",
                "SAY",
                "SHE",
                "TOO",
                "USE",
            }
            tickers = [
                ticker
                for ticker in tickers
                if ticker not in common_words and len(ticker) >= 2
            ]

            return list(tickers)

        except Exception as e:
            logger.error(f"Error extracting tickers: {e}")
            return []

    def _calculate_confidence(self, prompt: str) -> float:
        """Calculate overall confidence in the parsing."""
        try:
            confidence_factors = []

            # Intent confidence
            intent_result = self._classify_intent(prompt)
            confidence_factors.append(intent_result.get("confidence", 0))

            # Entity extraction confidence
            entities = self._extract_entities(prompt)
            entity_confidence = min(
                1.0, len(entities["tickers"]) * 0.2 + len(entities["companies"]) * 0.1
            )
            confidence_factors.append(entity_confidence)

            # Strategy suggestion confidence
            strategies = self._suggest_strategies(prompt)
            strategy_confidence = (
                max([s["relevance_score"] for s in strategies]) if strategies else 0
            )
            confidence_factors.append(strategy_confidence)

            # Average confidence
            return (
                sum(confidence_factors) / len(confidence_factors)
                if confidence_factors
                else 0
            )

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def _generate_routing_recommendations(
        self, parsed_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate routing recommendations based on parsed information."""
        try:
            routing = {
                "primary_model": None,
                "secondary_models": [],
                "strategy_engine": None,
                "data_providers": [],
                "priority": "normal",
            }

            intent = parsed_result.get("intent", {}).get("primary", "unknown")
            sentiment = parsed_result.get("sentiment", {}).get("label", "neutral")
            strategies = parsed_result.get("strategy_suggestions", [])

            # Route based on intent
            if intent == "forecast":
                routing["primary_model"] = "lstm_model"
                routing["secondary_models"] = ["prophet_model", "xgboost_model"]
            elif intent == "analyze":
                routing["primary_model"] = "market_analyzer"
                routing["secondary_models"] = [
                    "technical_indicators",
                    "fundamental_analyzer",
                ]
            elif intent == "trade":
                routing["primary_model"] = "strategy_engine"
                routing["strategy_engine"] = "enhanced_strategy_engine"
            elif intent == "risk":
                routing["primary_model"] = "risk_manager"
                routing["secondary_models"] = ["portfolio_optimizer"]
            elif intent == "portfolio":
                routing["primary_model"] = "portfolio_optimizer"
                routing["secondary_models"] = ["risk_manager"]

            # Add strategy-specific routing
            if strategies:
                top_strategy = strategies[0]["strategy"]
                if top_strategy == "sentiment":
                    routing["data_providers"].append("news_api")
                    routing["data_providers"].append("reddit_sentiment")
                elif top_strategy == "fundamental":
                    routing["data_providers"].append("alpha_vantage")
                    routing["data_providers"].append("yahoo_finance")

            # Set priority based on sentiment and confidence
            confidence = parsed_result.get("confidence", 0)
            if confidence > 0.8 and sentiment == "positive":
                routing["priority"] = "high"
            elif confidence < 0.3:
                routing["priority"] = "low"

            return routing

        except Exception as e:
            logger.error(f"Error generating routing recommendations: {e}")
            return {
                "primary_model": None,
                "secondary_models": [],
                "strategy_engine": None,
                "data_providers": [],
                "priority": "normal",
            }

    def _get_strategy_description(self, strategy: str) -> str:
        """Get description for a trading strategy."""
        descriptions = {
            "momentum": "Momentum-based strategy using trend following and breakout detection",
            "mean_reversion": "Mean reversion strategy using oversold/overbought indicators",
            "arbitrage": "Arbitrage strategy using pairs trading and correlation analysis",
            "fundamental": "Fundamental analysis using earnings, ratios, and company metrics",
            "sentiment": "Sentiment-based strategy using news and social media analysis",
        }
        return descriptions.get(strategy, "Strategy description not available")

    def _cache_parsing_result(self, prompt: str, result: Dict[str, Any]):
        """Cache parsing result for future reference."""
        try:
            # Create a hash of the prompt for caching
            import hashlib

            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

            cache_file = self.cache_dir / f"parsing_cache_{prompt_hash}.json"

            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to cache parsing result: {e}")

    def get_cached_result(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get cached parsing result if available."""
        try:
            import hashlib

            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

            cache_file = self.cache_dir / f"parsing_cache_{prompt_hash}.json"

            if cache_file.exists():
                with open(cache_file, "r") as f:
                    return json.load(f)

        except Exception as e:
            logger.warning(f"Failed to load cached result: {e}")

        return None

    def _detect_financial_jargon(self, prompt: str) -> Dict[str, Any]:
        """
        Detect financial jargon vs. general NLP to route appropriately.

        Args:
            prompt: Input prompt text

        Returns:
            Dictionary with jargon detection results
        """
        try:
            # Define financial jargon patterns
            financial_jargon_patterns = {
                "technical_analysis": [
                    r"\b(ma|sma|ema|rsi|macd|bollinger|stochastic|fibonacci|support|resistance|trendline|breakout|breakdown)\b",
                    r"\b(oversold|overbought|divergence|convergence|momentum|volume|volatility)\b",
                    r"\b(candlestick|doji|hammer|shooting star|engulfing|pivot|swing)\b",
                ],
                "fundamental_analysis": [
                    r"\b(pe ratio|pb ratio|eps|dividend yield|payout ratio|roe|roa|debt to equity)\b",
                    r"\b(earnings|revenue|profit margin|gross margin|operating margin|net margin)\b",
                    r"\b(cash flow|free cash flow|ebitda|ebit|net income|balance sheet|income statement)\b",
                ],
                "trading_terms": [
                    r"\b(long|short|position|entry|exit|stop loss|take profit|risk reward)\b",
                    r"\b(leverage|margin|options|futures|derivatives|hedge|arbitrage)\b",
                    r"\b(portfolio|allocation|diversification|rebalancing|alpha|beta|sharpe ratio)\b",
                ],
                "market_structure": [
                    r"\b(bull market|bear market|correction|rally|crash|bubble|recession)\b",
                    r"\b(secular|cyclical|sector rotation|market breadth|advance decline)\b",
                    r"\b(liquidity|bid ask|spread|volume|market cap|float|short interest)\b",
                ],
                "economic_indicators": [
                    r"\b(gdp|inflation|cpi|ppi|unemployment|fed funds rate|yield curve)\b",
                    r"\b(consumer confidence|pmi|ism|housing starts|retail sales)\b",
                    r"\b(monetary policy|fiscal policy|quantitative easing|tapering)\b",
                ],
            }

            # Define general NLP patterns (non-financial)
            general_nlp_patterns = {
                "general_questions": [
                    r"\b(what is|how does|why|when|where|who)\b",
                    r"\b(explain|describe|tell me about|help me understand)\b",
                ],
                "conversational": [
                    r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b",
                    r"\b(thank you|thanks|please|sorry|excuse me)\b",
                ],
                "general_analysis": [
                    r"\b(analyze|study|research|investigate|examine|review)\b",
                    r"\b(compare|contrast|similar|different|better|worse)\b",
                ],
            }

            # Count matches for each category
            jargon_scores = {}
            general_scores = {}

            # Check financial jargon patterns
            for category, patterns in financial_jargon_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = re.findall(pattern, prompt, re.IGNORECASE)
                    score += len(matches)
                jargon_scores[category] = score

            # Check general NLP patterns
            for category, patterns in general_nlp_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = re.findall(pattern, prompt, re.IGNORECASE)
                    score += len(matches)
                general_scores[category] = score

            # Calculate overall scores
            total_jargon_score = sum(jargon_scores.values())
            total_general_score = sum(general_scores.values())

            # Determine primary classification
            if total_jargon_score > total_general_score:
                primary_classification = "financial"
                confidence = min(
                    total_jargon_score / (total_jargon_score + total_general_score + 1),
                    1.0,
                )
            elif total_general_score > total_jargon_score:
                primary_classification = "general"
                confidence = min(
                    total_general_score
                    / (total_jargon_score + total_general_score + 1),
                    1.0,
                )
            else:
                primary_classification = "mixed"
                confidence = 0.5

            # Identify specific jargon categories
            dominant_jargon_categories = [
                category for category, score in jargon_scores.items() if score > 0
            ]

            # Determine routing recommendation
            routing_recommendation = self._get_routing_recommendation(
                primary_classification, dominant_jargon_categories, total_jargon_score
            )

            return {
                "primary_classification": primary_classification,
                "confidence": confidence,
                "jargon_scores": jargon_scores,
                "general_scores": general_scores,
                "total_jargon_score": total_jargon_score,
                "total_general_score": total_general_score,
                "dominant_jargon_categories": dominant_jargon_categories,
                "routing_recommendation": routing_recommendation,
                "requires_specialized_processing": total_jargon_score > 2,
            }

        except Exception as e:
            logger.error(f"Error detecting financial jargon: {e}")
            return {
                "primary_classification": "unknown",
                "confidence": 0.0,
                "error": str(e),
            }

    def _get_routing_recommendation(
        self, classification: str, jargon_categories: List[str], jargon_score: int
    ) -> Dict[str, Any]:
        """
        Get routing recommendation based on jargon detection.

        Args:
            classification: Primary classification (financial/general/mixed)
            jargon_categories: List of dominant jargon categories
            jargon_score: Total jargon score

        Returns:
            Routing recommendation dictionary
        """
        try:
            routing = {
                "primary_agent": None,
                "fallback_agent": None,
                "processing_pipeline": [],
                "confidence_threshold": 0.7,
                "requires_validation": False,
            }

            if classification == "financial":
                if jargon_score >= 5:
                    # High financial jargon - use specialized financial agents
                    routing["primary_agent"] = "financial_analysis_agent"
                    routing["fallback_agent"] = "technical_analysis_agent"
                    routing["processing_pipeline"] = [
                        "financial_entity_extraction",
                        "technical_analysis_parsing",
                        "strategy_matching",
                        "risk_assessment",
                    ]
                    routing["requires_validation"] = True

                elif jargon_score >= 3:
                    # Medium financial jargon - use mixed approach
                    routing["primary_agent"] = "mixed_analysis_agent"
                    routing["fallback_agent"] = "general_nlp_agent"
                    routing["processing_pipeline"] = [
                        "basic_entity_extraction",
                        "financial_keyword_matching",
                        "general_analysis",
                    ]

                else:
                    # Low financial jargon - use general NLP with financial awareness
                    routing["primary_agent"] = "general_nlp_agent"
                    routing["fallback_agent"] = "financial_analysis_agent"
                    routing["processing_pipeline"] = [
                        "general_entity_extraction",
                        "financial_keyword_detection",
                        "context_aware_analysis",
                    ]

            elif classification == "general":
                # General NLP - use standard processing
                routing["primary_agent"] = "general_nlp_agent"
                routing["fallback_agent"] = "conversational_agent"
                routing["processing_pipeline"] = [
                    "general_entity_extraction",
                    "intent_classification",
                    "sentiment_analysis",
                    "response_generation",
                ]

            else:  # mixed
                # Mixed content - use adaptive processing
                routing["primary_agent"] = "adaptive_nlp_agent"
                routing["fallback_agent"] = "general_nlp_agent"
                routing["processing_pipeline"] = [
                    "multi_domain_entity_extraction",
                    "context_classification",
                    "adaptive_analysis",
                    "hybrid_response_generation",
                ]
                routing["requires_validation"] = True

            # Add specific category-based routing
            if "technical_analysis" in jargon_categories:
                routing["technical_agent"] = "technical_analysis_agent"
            if "fundamental_analysis" in jargon_categories:
                routing["fundamental_agent"] = "fundamental_analysis_agent"
            if "trading_terms" in jargon_categories:
                routing["trading_agent"] = "trading_strategy_agent"

            return routing

        except Exception as e:
            logger.error(f"Error generating routing recommendation: {e}")
            return {
                "primary_agent": "general_nlp_agent",
                "fallback_agent": "error_handler_agent",
                "processing_pipeline": ["error_handling"],
                "confidence_threshold": 0.5,
                "requires_validation": True,
            }

    def route_to_appropriate_agent(self, prompt: str) -> Dict[str, Any]:
        """
        Route prompt to appropriate agent based on jargon detection.

        Args:
            prompt: Input prompt

        Returns:
            Routing decision dictionary
        """
        try:
            # Parse prompt and detect jargon
            parsed_result = self.parse_prompt(prompt)

            if "error" in parsed_result:
                return {
                    "success": False,
                    "error": parsed_result["error"],
                    "fallback_agent": "error_handler_agent",
                }

            jargon_detection = parsed_result.get("jargon_detection", {})
            routing_recommendation = jargon_detection.get("routing_recommendation", {})

            # Determine final routing decision
            confidence = jargon_detection.get("confidence", 0.0)
            confidence_threshold = routing_recommendation.get(
                "confidence_threshold", 0.7
            )

            if confidence >= confidence_threshold:
                primary_agent = routing_recommendation.get(
                    "primary_agent", "general_nlp_agent"
                )
                fallback_agent = routing_recommendation.get(
                    "fallback_agent", "error_handler_agent"
                )
            else:
                # Low confidence - use fallback
                primary_agent = routing_recommendation.get(
                    "fallback_agent", "general_nlp_agent"
                )
                fallback_agent = "error_handler_agent"

            return {
                "success": True,
                "primary_agent": primary_agent,
                "fallback_agent": fallback_agent,
                "confidence": confidence,
                "classification": jargon_detection.get(
                    "primary_classification", "unknown"
                ),
                "jargon_categories": jargon_detection.get(
                    "dominant_jargon_categories", []
                ),
                "processing_pipeline": routing_recommendation.get(
                    "processing_pipeline", []
                ),
                "requires_validation": routing_recommendation.get(
                    "requires_validation", False
                ),
                "parsed_result": parsed_result,
            }

        except Exception as e:
            logger.error(f"Error routing to appropriate agent: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_agent": "error_handler_agent",
            }

    def validate_financial_entities(self, entities: List[str]) -> Dict[str, Any]:
        """
        Validate extracted financial entities against known financial terms.

        Args:
            entities: List of extracted entities

        Returns:
            Validation result dictionary
        """
        try:
            # Define known financial terms for validation
            known_financial_terms = {
                "technical_indicators": [
                    "rsi",
                    "macd",
                    "bollinger bands",
                    "moving average",
                    "sma",
                    "ema",
                    "stochastic",
                    "fibonacci",
                    "support",
                    "resistance",
                    "trendline",
                ],
                "fundamental_metrics": [
                    "pe ratio",
                    "pb ratio",
                    "eps",
                    "dividend yield",
                    "roe",
                    "roa",
                    "debt to equity",
                    "cash flow",
                    "ebitda",
                    "revenue",
                    "earnings",
                ],
                "trading_terms": [
                    "long",
                    "short",
                    "position",
                    "entry",
                    "exit",
                    "stop loss",
                    "take profit",
                    "leverage",
                    "margin",
                    "portfolio",
                ],
                "market_terms": [
                    "bull market",
                    "bear market",
                    "volatility",
                    "liquidity",
                    "volume",
                    "market cap",
                    "correlation",
                    "beta",
                    "alpha",
                ],
            }

            validation_results = {
                "valid_entities": [],
                "invalid_entities": [],
                "category_matches": {},
                "confidence": 0.0,
            }

            for entity in entities:
                entity_lower = entity.lower()
                matched = False

                for category, terms in known_financial_terms.items():
                    if entity_lower in [term.lower() for term in terms]:
                        validation_results["valid_entities"].append(entity)
                        if category not in validation_results["category_matches"]:
                            validation_results["category_matches"][category] = []
                        validation_results["category_matches"][category].append(entity)
                        matched = True
                        break

                if not matched:
                    validation_results["invalid_entities"].append(entity)

            # Calculate confidence based on validation results
            total_entities = len(entities)
            if total_entities > 0:
                validation_results["confidence"] = (
                    len(validation_results["valid_entities"]) / total_entities
                )

            return validation_results

        except Exception as e:
            logger.error(f"Error validating financial entities: {e}")
            return {
                "valid_entities": [],
                "invalid_entities": entities,
                "category_matches": {},
                "confidence": 0.0,
                "error": str(e),
            }


# Global NLP agent instance
_nlp_agent = None


def get_nlp_agent() -> NLPAgent:
    """Get the global NLP agent instance."""
    global _nlp_agent
    if _nlp_agent is None:
        _nlp_agent = NLPAgent()
    return _nlp_agent
