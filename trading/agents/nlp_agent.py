"""
NLP Agent with Transformers and spaCy

Advanced NLP agent using transformers and spaCy for prompt parsing and model routing.
Provides intelligent routing to appropriate trading models and strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
from pathlib import Path
from datetime import datetime
import re
from dataclasses import dataclass

# Import NLP libraries with fallback handling
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

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
            'forecast': [
                r'forecast', r'predict', r'future', r'outlook', r'trend',
                r'what will', r'going to', r'expected', r'projection'
            ],
            'analyze': [
                r'analyze', r'analysis', r'study', r'examine', r'review',
                r'look at', r'check', r'investigate', r'assess'
            ],
            'trade': [
                r'trade', r'buy', r'sell', r'position', r'entry', r'exit',
                r'long', r'short', r'order', r'execute'
            ],
            'risk': [
                r'risk', r'volatility', r'drawdown', r'loss', r'exposure',
                r'hedge', r'protect', r'safe', r'conservative'
            ],
            'portfolio': [
                r'portfolio', r'allocation', r'weight', r'diversify',
                r'balance', r'rebalance', r'optimize'
            ]
        }
        
        # Define strategy keywords
        self.strategy_keywords = {
            'momentum': ['momentum', 'trend', 'breakout', 'moving average', 'rsi'],
            'mean_reversion': ['mean reversion', 'oversold', 'overbought', 'bollinger'],
            'arbitrage': ['arbitrage', 'pairs', 'spread', 'correlation'],
            'fundamental': ['fundamental', 'earnings', 'pe ratio', 'dividend'],
            'sentiment': ['sentiment', 'news', 'social', 'reddit', 'twitter']
        }
        
        # Define market regime keywords
        self.market_regime_keywords = {
            'bullish': ['bullish', 'uptrend', 'rally', 'bull market', 'positive'],
            'bearish': ['bearish', 'downtrend', 'decline', 'bear market', 'negative'],
            'sideways': ['sideways', 'range', 'consolidation', 'flat'],
            'volatile': ['volatile', 'choppy', 'unstable', 'wild']
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
                        return_all_scores=True
                    )
                    logger.info("Transformers sentiment analyzer loaded")
                except Exception as e:
                    logger.warning(f"Failed to load transformers sentiment analyzer: {e}")
                    self.sentiment_analyzer = None
            
            # Initialize intent classifier
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model="facebook/bart-large-mnli",
                        return_all_scores=True
                    )
                    logger.info("Transformers intent classifier loaded")
                except Exception as e:
                    logger.warning(f"Failed to load transformers intent classifier: {e}")
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
                return {'error': 'Empty prompt provided'}
            
            prompt = prompt.strip().lower()
            
            # Extract basic information
            result = {
                'original_prompt': prompt,
                'timestamp': datetime.now().isoformat(),
                'intent': self._classify_intent(prompt),
                'sentiment': self._analyze_sentiment(prompt),
                'entities': self._extract_entities(prompt),
                'strategy_suggestions': self._suggest_strategies(prompt),
                'market_regime': self._classify_market_regime(prompt),
                'timeframe': self._extract_timeframe(prompt),
                'tickers': self._extract_tickers(prompt),
                'confidence': self._calculate_confidence(prompt)
            }
            
            # Add routing recommendations
            result['routing'] = self._generate_routing_recommendations(result)
            
            # Cache the result
            self._cache_parsing_result(prompt, result)
            
            logger.info(f"Successfully parsed prompt: {prompt[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing prompt: {e}")
            return {'error': str(e), 'original_prompt': prompt}
    
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
                        label = result['label'].lower()
                        score = result['score']
                        if 'forecast' in label or 'predict' in label:
                            intent_scores['forecast'] = max(intent_scores.get('forecast', 0), score)
                        elif 'analyze' in label or 'study' in label:
                            intent_scores['analyze'] = max(intent_scores.get('analyze', 0), score)
                        elif 'trade' in label or 'buy' in label or 'sell' in label:
                            intent_scores['trade'] = max(intent_scores.get('trade', 0), score)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Transformers intent classification failed: {e}")
            
            # Get primary intent
            primary_intent = max(intent_scores.items(), key=lambda x: x[1]) if intent_scores else ('unknown', 0)
            
            return {
                'primary': primary_intent[0],
                'confidence': primary_intent[1],
                'all_scores': intent_scores
            }
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {'primary': 'unknown', 'confidence': 0, 'all_scores': {}}
    
    def _analyze_sentiment(self, prompt: str) -> Dict[str, Any]:
        """Analyze sentiment of the prompt."""
        try:
            sentiment_result = {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'label': 'neutral',
                'confidence': 0.0
            }
            
            # TextBlob sentiment analysis
            if TEXTBLOB_AVAILABLE:
                try:
                    blob = TextBlob(prompt)
                    sentiment_result['polarity'] = blob.sentiment.polarity
                    sentiment_result['subjectivity'] = blob.sentiment.subjectivity
                    
                    # Map polarity to label
                    if sentiment_result['polarity'] > 0.1:
                        sentiment_result['label'] = 'positive'
                    elif sentiment_result['polarity'] < -0.1:
                        sentiment_result['label'] = 'negative'
                    else:
                        sentiment_result['label'] = 'neutral'
                        
                    sentiment_result['confidence'] = abs(sentiment_result['polarity'])
                except Exception as e:
                    logger.warning(f"TextBlob sentiment analysis failed: {e}")
            
            # Transformers sentiment analysis
            if self.sentiment_analyzer:
                try:
                    transformer_result = self.sentiment_analyzer(prompt)
                    # Get the highest scoring sentiment
                    best_sentiment = max(transformer_result[0], key=lambda x: x['score'])
                    sentiment_result['transformer_label'] = best_sentiment['label']
                    sentiment_result['transformer_confidence'] = best_sentiment['score']
                except Exception as e:
                    logger.warning(f"Transformers sentiment analysis failed: {e}")
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'neutral', 'confidence': 0.0}
    
    def _extract_entities(self, prompt: str) -> Dict[str, List[str]]:
        """Extract named entities from the prompt."""
        try:
            entities = {
                'tickers': [],
                'companies': [],
                'dates': [],
                'numbers': [],
                'currencies': []
            }
            
            # spaCy entity extraction
            if self.nlp:
                try:
                    doc = self.nlp(prompt)
                    for ent in doc.ents:
                        if ent.label_ == 'ORG':
                            entities['companies'].append(ent.text)
                        elif ent.label_ == 'DATE':
                            entities['dates'].append(ent.text)
                        elif ent.label_ == 'MONEY':
                            entities['currencies'].append(ent.text)
                        elif ent.label_ == 'CARDINAL':
                            entities['numbers'].append(ent.text)
                except Exception as e:
                    logger.warning(f"spaCy entity extraction failed: {e}")
            
            # Regex-based ticker extraction
            ticker_pattern = r'\b[A-Z]{1,5}\b'
            potential_tickers = re.findall(ticker_pattern, prompt.upper())
            entities['tickers'] = [ticker for ticker in potential_tickers if len(ticker) >= 2]
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {'tickers': [], 'companies': [], 'dates': [], 'numbers': [], 'currencies': []}
    
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
                    suggestions.append({
                        'strategy': strategy,
                        'relevance_score': score / len(keywords),
                        'matched_keywords': matched_keywords,
                        'description': self._get_strategy_description(strategy)
                    })
            
            # Sort by relevance score
            suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
            
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
            
            primary_regime = max(regime_scores.items(), key=lambda x: x[1]) if regime_scores else ('unknown', 0)
            
            return {
                'primary': primary_regime[0],
                'confidence': primary_regime[1],
                'all_scores': regime_scores
            }
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return {'primary': 'unknown', 'confidence': 0, 'all_scores': {}}
    
    def _extract_timeframe(self, prompt: str) -> Dict[str, Any]:
        """Extract timeframe information from the prompt."""
        try:
            timeframe_patterns = {
                'short_term': [r'\b(day|daily|intraday|hour|minute)\b', r'\b(1d|1h|1m|5m|15m)\b'],
                'medium_term': [r'\b(week|weekly|month|monthly)\b', r'\b(1w|1mo|3mo)\b'],
                'long_term': [r'\b(year|yearly|annual|long term)\b', r'\b(1y|5y|10y)\b']
            }
            
            timeframe_scores = {}
            for timeframe, patterns in timeframe_patterns.items():
                score = 0
                for pattern in patterns:
                    if re.search(pattern, prompt):
                        score += 1
                timeframe_scores[timeframe] = score / len(patterns)
            
            primary_timeframe = max(timeframe_scores.items(), key=lambda x: x[1]) if timeframe_scores else ('unknown', 0)
            
            return {
                'primary': primary_timeframe[0],
                'confidence': primary_timeframe[1],
                'all_scores': timeframe_scores
            }
            
        except Exception as e:
            logger.error(f"Error extracting timeframe: {e}")
            return {'primary': 'unknown', 'confidence': 0, 'all_scores': {}}
    
    def _extract_tickers(self, prompt: str) -> List[str]:
        """Extract stock tickers from the prompt."""
        try:
            # Enhanced ticker extraction
            ticker_patterns = [
                r'\b[A-Z]{1,5}\b',  # Basic ticker pattern
                r'\$([A-Z]{1,5})\b',  # $TICKER format
                r'\b([A-Z]{1,5})\s+stock\b',  # TICKER stock format
            ]
            
            tickers = set()
            for pattern in ticker_patterns:
                matches = re.findall(pattern, prompt.upper())
                tickers.update(matches)
            
            # Filter out common words that might be mistaken for tickers
            common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'HAS', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
            tickers = [ticker for ticker in tickers if ticker not in common_words and len(ticker) >= 2]
            
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
            confidence_factors.append(intent_result.get('confidence', 0))
            
            # Entity extraction confidence
            entities = self._extract_entities(prompt)
            entity_confidence = min(1.0, len(entities['tickers']) * 0.2 + len(entities['companies']) * 0.1)
            confidence_factors.append(entity_confidence)
            
            # Strategy suggestion confidence
            strategies = self._suggest_strategies(prompt)
            strategy_confidence = max([s['relevance_score'] for s in strategies]) if strategies else 0
            confidence_factors.append(strategy_confidence)
            
            # Average confidence
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _generate_routing_recommendations(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate routing recommendations based on parsed information."""
        try:
            routing = {
                'primary_model': None,
                'secondary_models': [],
                'strategy_engine': None,
                'data_providers': [],
                'priority': 'normal'
            }
            
            intent = parsed_result.get('intent', {}).get('primary', 'unknown')
            sentiment = parsed_result.get('sentiment', {}).get('label', 'neutral')
            strategies = parsed_result.get('strategy_suggestions', [])
            
            # Route based on intent
            if intent == 'forecast':
                routing['primary_model'] = 'lstm_model'
                routing['secondary_models'] = ['prophet_model', 'xgboost_model']
            elif intent == 'analyze':
                routing['primary_model'] = 'market_analyzer'
                routing['secondary_models'] = ['technical_indicators', 'fundamental_analyzer']
            elif intent == 'trade':
                routing['primary_model'] = 'strategy_engine'
                routing['strategy_engine'] = 'enhanced_strategy_engine'
            elif intent == 'risk':
                routing['primary_model'] = 'risk_manager'
                routing['secondary_models'] = ['portfolio_optimizer']
            elif intent == 'portfolio':
                routing['primary_model'] = 'portfolio_optimizer'
                routing['secondary_models'] = ['risk_manager']
            
            # Add strategy-specific routing
            if strategies:
                top_strategy = strategies[0]['strategy']
                if top_strategy == 'sentiment':
                    routing['data_providers'].append('news_api')
                    routing['data_providers'].append('reddit_sentiment')
                elif top_strategy == 'fundamental':
                    routing['data_providers'].append('alpha_vantage')
                    routing['data_providers'].append('yahoo_finance')
            
            # Set priority based on sentiment and confidence
            confidence = parsed_result.get('confidence', 0)
            if confidence > 0.8 and sentiment == 'positive':
                routing['priority'] = 'high'
            elif confidence < 0.3:
                routing['priority'] = 'low'
            
            return routing
            
        except Exception as e:
            logger.error(f"Error generating routing recommendations: {e}")
            return {'primary_model': None, 'secondary_models': [], 'strategy_engine': None, 'data_providers': [], 'priority': 'normal'}
    
    def _get_strategy_description(self, strategy: str) -> str:
        """Get description for a trading strategy."""
        descriptions = {
            'momentum': 'Momentum-based strategy using trend following and breakout detection',
            'mean_reversion': 'Mean reversion strategy using oversold/overbought indicators',
            'arbitrage': 'Arbitrage strategy using pairs trading and correlation analysis',
            'fundamental': 'Fundamental analysis using earnings, ratios, and company metrics',
            'sentiment': 'Sentiment-based strategy using news and social media analysis'
        }
        return descriptions.get(strategy, 'Strategy description not available')
    
    def _cache_parsing_result(self, prompt: str, result: Dict[str, Any]):
        """Cache parsing result for future reference."""
        try:
            # Create a hash of the prompt for caching
            import hashlib
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            cache_file = self.cache_dir / f"parsing_cache_{prompt_hash}.json"
            
            with open(cache_file, 'w') as f:
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
                with open(cache_file, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.warning(f"Failed to load cached result: {e}")
        
        return None

# Global NLP agent instance
_nlp_agent = None

def get_nlp_agent() -> NLPAgent:
    """Get the global NLP agent instance."""
    global _nlp_agent
    if _nlp_agent is None:
        _nlp_agent = NLPAgent()
    return _nlp_agent 