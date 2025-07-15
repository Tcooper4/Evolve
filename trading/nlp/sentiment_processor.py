"""
Sentiment Processor with Dynamic Polarity Scaling

This module provides advanced sentiment analysis with dynamic polarity scaling,
supporting news source weighting, tweet impact scoring, and Flesch-Kincaid
readability scoring as a quality filter.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from textstat import textstat

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sentiment data sources."""
    NEWS = "news"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FORUM = "forum"
    BLOG = "blog"
    PRESS_RELEASE = "press_release"
    EARNINGS_CALL = "earnings_call"
    ANALYST_REPORT = "analyst_report"


class SentimentType(Enum):
    """Types of sentiment analysis."""
    POLARITY = "polarity"  # Positive/Negative
    EMOTION = "emotion"    # Joy, Fear, Anger, etc.
    INTENSITY = "intensity"  # Strong/Weak
    SUBJECTIVITY = "subjectivity"  # Objective/Subjective


@dataclass
class NewsSource:
    """News source configuration."""
    name: str
    domain: str
    credibility_score: float  # 0.0 to 1.0
    bias_factor: float  # -1.0 (negative bias) to 1.0 (positive bias)
    impact_multiplier: float  # How much weight to give this source
    category: str  # 'financial', 'general', 'social_media'
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TweetMetrics:
    """Twitter-specific metrics."""
    followers_count: int
    verified: bool
    retweet_count: int
    like_count: int
    reply_count: int
    quote_count: int
    account_age_days: int
    tweet_age_hours: int


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    source: SentimentSource
    sentiment_type: SentimentType
    base_score: float  # Raw sentiment score
    scaled_score: float  # Score after applying dynamic scaling
    confidence: float
    source_weight: float
    impact_score: float
    readability_score: Optional[float] = None
    quality_filter_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentContext:
    """Context for sentiment analysis."""
    ticker: Optional[str] = None
    sector: Optional[str] = None
    market_conditions: Optional[str] = None
    time_period: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class SentimentProcessor:
    """
    Advanced sentiment processor with dynamic polarity scaling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize sentiment processor.
        
        Args:
            config_path: Path to configuration file
        """
        self.news_sources: Dict[str, NewsSource] = {}
        self.sentiment_lexicons: Dict[str, Dict[str, float]] = {}
        self.emotion_lexicons: Dict[str, Dict[str, List[str]]] = {}
        
        # Default configuration
        self.default_source_weight = 1.0
        self.min_readability_score = 30.0  # Flesch-Kincaid minimum
        self.max_readability_score = 100.0  # Flesch-Kincaid maximum
        self.quality_filter_enabled = True
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_config()
        
        logger.info("SentimentProcessor initialized with dynamic polarity scaling")
    
    def _load_default_config(self):
        """Load default configuration."""
        # Default news sources
        default_sources = {
            "reuters": NewsSource(
                name="Reuters",
                domain="reuters.com",
                credibility_score=0.95,
                bias_factor=0.0,
                impact_multiplier=1.2,
                category="financial"
            ),
            "bloomberg": NewsSource(
                name="Bloomberg",
                domain="bloomberg.com",
                credibility_score=0.92,
                bias_factor=0.05,
                impact_multiplier=1.1,
                category="financial"
            ),
            "cnbc": NewsSource(
                name="CNBC",
                domain="cnbc.com",
                credibility_score=0.88,
                bias_factor=0.1,
                impact_multiplier=1.0,
                category="financial"
            ),
            "yahoo_finance": NewsSource(
                name="Yahoo Finance",
                domain="finance.yahoo.com",
                credibility_score=0.85,
                bias_factor=0.05,
                impact_multiplier=0.9,
                category="financial"
            ),
            "marketwatch": NewsSource(
                name="MarketWatch",
                domain="marketwatch.com",
                credibility_score=0.87,
                bias_factor=0.08,
                impact_multiplier=0.95,
                category="financial"
            )
        }
        
        for source in default_sources.values():
            self.news_sources[source.domain] = source
        
        # Load basic sentiment lexicons
        self._load_basic_lexicons()
    
    def _load_basic_lexicons(self):
        """Load basic sentiment lexicons."""
        # Positive words
        positive_words = [
            "bullish", "surge", "rally", "gain", "profit", "growth", "positive",
            "strong", "beat", "exceed", "outperform", "upgrade", "buy", "hold",
            "opportunity", "potential", "promising", "robust", "solid", "stable"
        ]
        
        # Negative words
        negative_words = [
            "bearish", "plunge", "crash", "loss", "decline", "negative", "weak",
            "miss", "underperform", "downgrade", "sell", "risk", "concern",
            "volatile", "uncertain", "pressure", "headwind", "challenge", "worry"
        ]
        
        # Create sentiment lexicon
        self.sentiment_lexicons["basic"] = {}
        for word in positive_words:
            self.sentiment_lexicons["basic"][word.lower()] = 1.0
        for word in negative_words:
            self.sentiment_lexicons["basic"][word.lower()] = -1.0
        
        # Emotion lexicons
        self.emotion_lexicons["fear"] = [
            "panic", "fear", "worry", "concern", "anxiety", "nervous", "scared",
            "terrified", "horrified", "dread", "apprehension"
        ]
        
        self.emotion_lexicons["greed"] = [
            "greed", "euphoria", "mania", "frenzy", "excitement", "optimism",
            "enthusiasm", "confidence", "bullish", "moon", "rocket"
        ]
        
        self.emotion_lexicons["anger"] = [
            "anger", "fury", "rage", "outrage", "furious", "angry", "mad",
            "irritated", "annoyed", "frustrated", "disgusted"
        ]
    
    def load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load news sources
            if "news_sources" in config:
                for source_data in config["news_sources"]:
                    source = NewsSource(**source_data)
                    self.news_sources[source.domain] = source
            
            # Load settings
            if "settings" in config:
                settings = config["settings"]
                self.default_source_weight = settings.get("default_source_weight", 1.0)
                self.min_readability_score = settings.get("min_readability_score", 30.0)
                self.max_readability_score = settings.get("max_readability_score", 100.0)
                self.quality_filter_enabled = settings.get("quality_filter_enabled", True)
            
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._load_default_config()
    
    def add_news_source(self, source: NewsSource):
        """Add a new news source."""
        self.news_sources[source.domain] = source
        logger.info(f"Added news source: {source.name}")
    
    def remove_news_source(self, domain: str):
        """Remove a news source."""
        if domain in self.news_sources:
            del self.news_sources[domain]
            logger.info(f"Removed news source: {domain}")
    
    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid readability score.
        
        Args:
            text: Text to analyze
            
        Returns:
            Readability score (0-100, higher is more readable)
        """
        try:
            # Clean text for analysis
            clean_text = re.sub(r'[^\w\s\.]', '', text)
            if len(clean_text.split()) < 10:  # Too short for reliable scoring
                return 50.0
            
            score = textstat.flesch_reading_ease(clean_text)
            return max(0.0, min(100.0, score))
        except Exception as e:
            logger.warning(f"Error calculating readability score: {e}")
            return 50.0  # Default score
    
    def calculate_tweet_impact_score(self, metrics: TweetMetrics) -> float:
        """
        Calculate tweet impact score based on engagement metrics.
        
        Args:
            metrics: Tweet metrics
            
        Returns:
            Impact score (0.0 to 1.0)
        """
        try:
            # Base score from follower count (logarithmic scale)
            follower_score = min(1.0, np.log10(max(1, metrics.followers_count)) / 6.0)
            
            # Engagement rate
            total_engagement = metrics.retweet_count + metrics.like_count + metrics.reply_count + metrics.quote_count
            engagement_rate = total_engagement / max(1, metrics.followers_count)
            engagement_score = min(1.0, engagement_rate * 100)  # Scale up engagement rate
            
            # Verification bonus
            verified_bonus = 0.2 if metrics.verified else 0.0
            
            # Account age factor (older accounts get slight bonus)
            age_factor = min(1.0, metrics.account_age_days / 365.0)
            
            # Tweet recency factor (newer tweets get slight bonus)
            recency_factor = max(0.5, 1.0 - (metrics.tweet_age_hours / 24.0))
            
            # Calculate weighted impact score
            impact_score = (
                follower_score * 0.3 +
                engagement_score * 0.4 +
                verified_bonus * 0.1 +
                age_factor * 0.1 +
                recency_factor * 0.1
            )
            
            return min(1.0, max(0.0, impact_score))
            
        except Exception as e:
            logger.warning(f"Error calculating tweet impact score: {e}")
            return 0.5  # Default score
    
    def get_source_weight(self, source_url: str, source_type: SentimentSource) -> float:
        """
        Get source weight based on URL and source type.
        
        Args:
            source_url: Source URL
            source_type: Type of sentiment source
            
        Returns:
            Source weight (0.0 to 2.0)
        """
        try:
            # Extract domain from URL
            domain = re.search(r'https?://(?:www\.)?([^/]+)', source_url)
            if not domain:
                return self.default_source_weight
            
            domain = domain.group(1).lower()
            
            # Check if we have configuration for this domain
            if domain in self.news_sources:
                source = self.news_sources[domain]
                return source.impact_multiplier
            
            # Default weights by source type
            default_weights = {
                SentimentSource.NEWS: 1.0,
                SentimentSource.TWITTER: 0.7,
                SentimentSource.REDDIT: 0.6,
                SentimentSource.FORUM: 0.5,
                SentimentSource.BLOG: 0.8,
                SentimentSource.PRESS_RELEASE: 1.2,
                SentimentSource.EARNINGS_CALL: 1.5,
                SentimentSource.ANALYST_REPORT: 1.3
            }
            
            return default_weights.get(source_type, self.default_source_weight)
            
        except Exception as e:
            logger.warning(f"Error getting source weight: {e}")
            return self.default_source_weight
    
    def analyze_sentiment(self, text: str, source_type: SentimentSource = SentimentSource.NEWS,
                         source_url: Optional[str] = None, tweet_metrics: Optional[TweetMetrics] = None,
                         context: Optional[SentimentContext] = None) -> SentimentResult:
        """
        Analyze sentiment with dynamic polarity scaling.
        
        Args:
            text: Text to analyze
            source_type: Type of sentiment source
            source_url: Source URL for weight calculation
            tweet_metrics: Twitter metrics for impact calculation
            context: Additional context
            
        Returns:
            SentimentResult with scaled scores
        """
        try:
            # Calculate base sentiment score
            base_score = self._calculate_base_sentiment(text)
            
            # Calculate readability score
            readability_score = self.calculate_readability_score(text)
            
            # Check quality filter
            quality_filter_passed = True
            if self.quality_filter_enabled:
                quality_filter_passed = (
                    self.min_readability_score <= readability_score <= self.max_readability_score
                )
            
            # Calculate source weight
            source_weight = self.get_source_weight(source_url or "", source_type)
            
            # Calculate impact score
            impact_score = 1.0
            if source_type == SentimentSource.TWITTER and tweet_metrics:
                impact_score = self.calculate_tweet_impact_score(tweet_metrics)
            elif source_type == SentimentSource.NEWS:
                # News impact based on source credibility
                if source_url:
                    domain = re.search(r'https?://(?:www\.)?([^/]+)', source_url)
                    if domain and domain.group(1).lower() in self.news_sources:
                        source = self.news_sources[domain.group(1).lower()]
                        impact_score = source.credibility_score
            
            # Apply dynamic polarity scaling
            scaled_score = self._apply_dynamic_scaling(
                base_score, source_weight, impact_score, context
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                base_score, source_weight, impact_score, readability_score
            )
            
            return SentimentResult(
                text=text,
                source=source_type,
                sentiment_type=SentimentType.POLARITY,
                base_score=base_score,
                scaled_score=scaled_score,
                confidence=confidence,
                source_weight=source_weight,
                impact_score=impact_score,
                readability_score=readability_score,
                quality_filter_passed=quality_filter_passed,
                metadata={
                    "context": context.__dict__ if context else {},
                    "tweet_metrics": tweet_metrics.__dict__ if tweet_metrics else {}
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentResult(
                text=text,
                source=source_type,
                sentiment_type=SentimentType.POLARITY,
                base_score=0.0,
                scaled_score=0.0,
                confidence=0.0,
                source_weight=1.0,
                impact_score=1.0,
                quality_filter_passed=False,
                metadata={"error": str(e)}
            )
    
    def _calculate_base_sentiment(self, text: str) -> float:
        """Calculate base sentiment score using lexicon."""
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            if not words:
                return 0.0
            
            total_score = 0.0
            matched_words = 0
            
            for word in words:
                if word in self.sentiment_lexicons["basic"]:
                    total_score += self.sentiment_lexicons["basic"][word]
                    matched_words += 1
            
            if matched_words == 0:
                return 0.0
            
            # Normalize score to [-1, 1] range
            avg_score = total_score / matched_words
            return max(-1.0, min(1.0, avg_score))
            
        except Exception as e:
            logger.warning(f"Error calculating base sentiment: {e}")
            return 0.0
    
    def _apply_dynamic_scaling(self, base_score: float, source_weight: float,
                              impact_score: float, context: Optional[SentimentContext]) -> float:
        """Apply dynamic polarity scaling based on context and source factors."""
        try:
            # Start with base score
            scaled_score = base_score
            
            # Apply source weight
            scaled_score *= source_weight
            
            # Apply impact score
            scaled_score *= impact_score
            
            # Apply context-based adjustments
            if context:
                # Market conditions adjustment
                if context.market_conditions == "bull_market":
                    scaled_score *= 1.1  # Slightly amplify positive sentiment
                elif context.market_conditions == "bear_market":
                    scaled_score *= 0.9  # Slightly dampen positive sentiment
                
                # Sector-specific adjustments
                if context.sector:
                    sector_adjustments = {
                        "technology": 1.05,
                        "healthcare": 1.02,
                        "finance": 0.98,
                        "energy": 1.03,
                        "consumer": 1.0
                    }
                    adjustment = sector_adjustments.get(context.sector.lower(), 1.0)
                    scaled_score *= adjustment
            
            # Ensure score stays in [-1, 1] range
            return max(-1.0, min(1.0, scaled_score))
            
        except Exception as e:
            logger.warning(f"Error applying dynamic scaling: {e}")
            return base_score
    
    def _calculate_confidence(self, base_score: float, source_weight: float,
                             impact_score: float, readability_score: float) -> float:
        """Calculate confidence in sentiment analysis."""
        try:
            # Base confidence from source weight
            confidence = source_weight * 0.4
            
            # Impact score contribution
            confidence += impact_score * 0.3
            
            # Readability contribution
            readability_confidence = (readability_score - 30) / 70  # Normalize to [0, 1]
            confidence += readability_confidence * 0.2
            
            # Base score magnitude contribution (stronger signals are more confident)
            confidence += abs(base_score) * 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        Analyze emotional content in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of emotion scores
        """
        try:
            words = re.findall(r'\b\w+\b', text.lower())
            emotion_scores = {}
            
            for emotion, emotion_words in self.emotion_lexicons.items():
                matches = sum(1 for word in words if word in emotion_words)
                emotion_scores[emotion] = matches / max(1, len(words))
            
            return emotion_scores
            
        except Exception as e:
            logger.warning(f"Error analyzing emotion: {e}")
            return {}
    
    def batch_analyze(self, texts: List[str], source_type: SentimentSource = SentimentSource.NEWS,
                     source_urls: Optional[List[str]] = None) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            source_type: Type of sentiment source
            source_urls: List of source URLs (optional)
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            source_url = source_urls[i] if source_urls and i < len(source_urls) else None
            result = self.analyze_sentiment(text, source_type, source_url)
            results.append(result)
        
        return results
    
    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """
        Generate summary statistics for sentiment results.
        
        Args:
            results: List of sentiment results
            
        Returns:
            Summary statistics
        """
        try:
            if not results:
                return {}
            
            # Filter results that passed quality filter
            quality_results = [r for r in results if r.quality_filter_passed]
            
            if not quality_results:
                return {"error": "No results passed quality filter"}
            
            # Calculate statistics
            scores = [r.scaled_score for r in quality_results]
            confidences = [r.confidence for r in quality_results]
            
            summary = {
                "total_analyzed": len(results),
                "quality_passed": len(quality_results),
                "avg_sentiment": np.mean(scores),
                "std_sentiment": np.std(scores),
                "min_sentiment": np.min(scores),
                "max_sentiment": np.max(scores),
                "avg_confidence": np.mean(confidences),
                "positive_count": sum(1 for s in scores if s > 0.1),
                "negative_count": sum(1 for s in scores if s < -0.1),
                "neutral_count": sum(1 for s in scores if -0.1 <= s <= 0.1),
                "source_distribution": {},
                "readability_stats": {
                    "avg_readability": np.mean([r.readability_score for r in quality_results if r.readability_score]),
                    "min_readability": np.min([r.readability_score for r in quality_results if r.readability_score]),
                    "max_readability": np.max([r.readability_score for r in quality_results if r.readability_score])
                }
            }
            
            # Source distribution
            for result in quality_results:
                source = result.source.value
                summary["source_distribution"][source] = summary["source_distribution"].get(source, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {e}")
            return {"error": str(e)}


def create_sentiment_processor(config_path: Optional[str] = None) -> SentimentProcessor:
    """
    Create a sentiment processor instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SentimentProcessor instance
    """
    return SentimentProcessor(config_path) 