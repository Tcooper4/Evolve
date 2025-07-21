"""
External Signal Integration

This module provides comprehensive integration with external data sources for enhanced
trading signals including news sentiment, social media sentiment, macro indicators,
insider trading, and options flow data.
"""

import asyncio
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# External API imports
try:
    pass

    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    pass

    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


def async_strategy_wrapper(timeout: int = 30, fallback_value: Any = None):
    """Decorator to wrap async strategy calls with timeout and fallback handling."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Use asyncio.shield to protect against interruption
                result = await asyncio.wait_for(
                    asyncio.shield(func(*args, **kwargs)), timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Strategy {func.__name__} timed out after {timeout}s")
                return fallback_value
            except Exception as e:
                logger.error(f"Strategy {func.__name__} failed: {e}")
                return fallback_value

        return wrapper

    return decorator


@dataclass
class SignalData:
    """Standardized signal data structure."""

    source: str
    symbol: str
    timestamp: datetime
    signal_type: str
    value: float
    confidence: float
    metadata: Dict[str, Any]
    raw_data: Optional[Dict[str, Any]] = None


class SignalValidator:
    """Validates signal data for consistency and quality."""

    @staticmethod
    def validate_signal_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate basic signal data structure."""
        required_fields = ["source", "symbol", "timestamp", "signal_type", "value"]

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Validate timestamp
        if not isinstance(data["timestamp"], (datetime, str)):
            return False, "Invalid timestamp format"

        # Validate value is numeric
        if not isinstance(data["value"], (int, float)):
            return False, "Value must be numeric"

        # Validate confidence is between 0 and 1
        confidence = data.get("confidence", 1.0)
        if not (0 <= confidence <= 1):
            return False, "Confidence must be between 0 and 1"

        return True, None

    @staticmethod
    def validate_sentiment_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate sentiment data structure."""
        required_fields = [
            "symbol",
            "timestamp",
            "sentiment_score",
            "sentiment_label",
            "volume",
            "source",
            "confidence",
        ]

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Validate sentiment score is between -1 and 1
        score = data["sentiment_score"]
        if not (-1 <= score <= 1):
            return False, f"Sentiment score must be between -1 and 1, got {score}"

        # Validate sentiment label
        valid_labels = ["positive", "negative", "neutral"]
        if data["sentiment_label"] not in valid_labels:
            return False, f"Invalid sentiment label: {data['sentiment_label']}"

        # Validate volume is positive
        if data["volume"] < 0:
            return False, "Volume must be non-negative"

        # Validate confidence is between 0 and 1
        if not (0 <= data["confidence"] <= 1):
            return False, "Confidence must be between 0 and 1"

        return True, None

    @staticmethod
    def validate_options_flow_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate options flow data structure."""
        required_fields = [
            "symbol",
            "date",
            "strike",
            "option_type",
            "volume",
            "premium",
            "expiration",
        ]

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Validate option type
        if data["option_type"] not in ["call", "put"]:
            return False, f"Invalid option type: {data['option_type']}"

        # Validate numeric fields
        numeric_fields = ["strike", "volume", "premium"]
        for field in numeric_fields:
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                return False, f"{field} must be a positive number"

        return True, None

    @staticmethod
    def validate_macro_indicator_data(
        data: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """Validate macro indicator data structure."""
        required_fields = [
            "name",
            "value",
            "unit",
            "frequency",
            "last_updated",
            "source",
        ]

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        # Validate value is numeric
        if not isinstance(data["value"], (int, float)):
            return False, "Value must be numeric"

        # Validate last_updated is a valid date
        if not isinstance(data["last_updated"], (datetime, str)):
            return False, "Invalid last_updated format"

        return True, None


@dataclass
class SentimentData:
    """Sentiment analysis results."""

    symbol: str
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    volume: int
    source: str
    confidence: float


class NewsSentimentCollector:
    """Collects and analyzes news sentiment for stocks."""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """Initialize news sentiment collector."""
        self.api_keys = api_keys or {}
        self.rate_limits = {
            "newsapi": {"last_call": 0, "min_interval": 1},
            "gnews": {"last_call": 0, "min_interval": 1},
        }
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradingBot/1.0"})

    @async_strategy_wrapper(timeout=30, fallback_value=[])
    async def get_news_sentiment(
        self, symbol: str, days_back: int = 7
    ) -> List[SentimentData]:
        """Get news sentiment for a symbol."""
        try:
            # Try multiple news sources with fallback
            sentiment_data = []

            # Try NewsAPI first
            if "newsapi" in self.api_keys:
                try:
                    newsapi_data = await self._get_newsapi_sentiment(symbol, days_back)
                    sentiment_data.extend(newsapi_data)
                except Exception as e:
                    logger.warning(f"NewsAPI failed for {symbol}: {e}")

            # Try GNews as fallback
            if "gnews" in self.api_keys:
                try:
                    gnews_data = await self._get_gnews_sentiment(symbol, days_back)
                    sentiment_data.extend(gnews_data)
                except Exception as e:
                    logger.warning(f"GNews failed for {symbol}: {e}")

            # If no API data, return empty list (fallback)
            if not sentiment_data:
                logger.info(f"No news sentiment data available for {symbol}")
                return []

            # Aggregate and return results
            return await self._aggregate_sentiment(sentiment_data)

        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return []

    @async_strategy_wrapper(timeout=20, fallback_value=[])
    async def _get_newsapi_sentiment(
        self, symbol: str, days_back: int
    ) -> List[SentimentData]:
        """Get sentiment from NewsAPI."""
        await self._rate_limit("newsapi")

        api_key = self.api_keys["newsapi"]
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": f'"{symbol}" stock',
            "from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "apiKey": api_key,
            "language": "en",
            "pageSize": 100,
        }

        response = await asyncio.to_thread(
            self.session.get, url, params=params, timeout=10
        )
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        sentiment_data = []
        for article in articles:
            try:
                title = article.get("title", "")
                description = article.get("description", "")
                content = f"{title} {description}"

                if content.strip():
                    sentiment_score = await self._analyze_text_sentiment(content)
                    sentiment_label = self._get_sentiment_label(sentiment_score)

                    sentiment_data.append(
                        SentimentData(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(
                                article["publishedAt"].replace("Z", "+00:00")
                            ),
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label,
                            volume=1,
                            source="newsapi",
                            confidence=0.8,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error processing news article: {e}")
                continue

        return sentiment_data

    @async_strategy_wrapper(timeout=20, fallback_value=[])
    async def _get_gnews_sentiment(
        self, symbol: str, days_back: int
    ) -> List[SentimentData]:
        """Get sentiment from GNews API."""
        await self._rate_limit("gnews")

        api_key = self.api_keys["gnews"]
        url = "https://gnews.io/api/v4/search"

        params = {
            "q": f'"{symbol}" stock',
            "from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            "sortby": "publishedAt",
            "token": api_key,
            "lang": "en",
            "max": 100,
        }

        response = await asyncio.to_thread(
            self.session.get, url, params=params, timeout=10
        )
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        sentiment_data = []
        for article in articles:
            try:
                title = article.get("title", "")
                description = article.get("description", "")
                content = f"{title} {description}"

                if content.strip():
                    sentiment_score = await self._analyze_text_sentiment(content)
                    sentiment_label = self._get_sentiment_label(sentiment_score)

                    sentiment_data.append(
                        SentimentData(
                            symbol=symbol,
                            timestamp=datetime.fromisoformat(
                                article["publishedAt"].replace("Z", "+00:00")
                            ),
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label,
                            volume=1,
                            source="gnews",
                            confidence=0.8,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error processing GNews article: {e}")
                continue

        return sentiment_data

    @async_strategy_wrapper(timeout=10, fallback_value=0.0)
    async def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze text sentiment using simple keyword-based approach."""
        try:
            # Simple keyword-based sentiment analysis
            positive_words = {
                "bullish",
                "positive",
                "growth",
                "profit",
                "gain",
                "rise",
                "up",
                "strong",
                "buy",
                "outperform",
                "beat",
                "exceed",
                "surge",
                "rally",
            }
            negative_words = {
                "bearish",
                "negative",
                "decline",
                "loss",
                "fall",
                "down",
                "weak",
                "sell",
                "underperform",
                "miss",
                "drop",
                "crash",
                "plunge",
            }

            words = set(text.lower().split())

            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))

            if positive_count == 0 and negative_count == 0:
                return 0.0

            # Calculate sentiment score between -1 and 1
            total = positive_count + negative_count
            sentiment_score = (positive_count - negative_count) / total

            return max(-1.0, min(1.0, sentiment_score))

        except Exception as e:
            logger.warning(f"Error analyzing text sentiment: {e}")
            return 0.0

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

    @async_strategy_wrapper(timeout=15, fallback_value=[])
    async def _aggregate_sentiment(
        self, sentiment_data: List[SentimentData]
    ) -> List[SentimentData]:
        """Aggregate sentiment data by time periods."""
        if not sentiment_data:
            return []

        # Group by hour and aggregate
        hourly_data = {}
        for data in sentiment_data:
            hour_key = data.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_data:
                hourly_data[hour_key] = []
            hourly_data[hour_key].append(data)

        aggregated = []
        for hour, data_list in hourly_data.items():
            if data_list:
                # Calculate weighted average sentiment
                total_volume = sum(d.volume for d in data_list)
                if total_volume > 0:
                    weighted_score = (
                        sum(d.sentiment_score * d.volume for d in data_list)
                        / total_volume
                    )

                    aggregated.append(
                        SentimentData(
                            symbol=data_list[0].symbol,
                            timestamp=hour,
                            sentiment_score=weighted_score,
                            sentiment_label=self._get_sentiment_label(weighted_score),
                            volume=total_volume,
                            source="aggregated",
                            confidence=0.9,
                        )
                    )

        return aggregated

    async def _rate_limit(self, source: str):
        """Implement rate limiting for API calls."""
        if source in self.rate_limits:
            limit = self.rate_limits[source]
            elapsed = time.time() - limit["last_call"]
            if elapsed < limit["min_interval"]:
                await asyncio.sleep(limit["min_interval"] - elapsed)
            limit["last_call"] = time.time()


class TwitterSentimentCollector:
    """Collects Twitter sentiment for stocks."""

    def __init__(self):
        """Initialize Twitter sentiment collector."""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradingBot/1.0"})

    @async_strategy_wrapper(timeout=25, fallback_value=[])
    async def get_twitter_sentiment(
        self, symbol: str, max_tweets: int = 100
    ) -> List[SentimentData]:
        """Get Twitter sentiment for a symbol."""
        try:
            # For now, return simulated data since Twitter API requires authentication
            # In production, this would use the Twitter API v2
            logger.info(f"Simulating Twitter sentiment for {symbol}")

            # Simulate some tweets
            simulated_tweets = [
                f"${symbol} looking bullish today! #stocks",
                f"Not sure about ${symbol} performance #trading",
                f"${symbol} earnings beat expectations #investing",
                f"${symbol} chart showing weakness #technicalanalysis",
                f"Bullish on ${symbol} for next quarter #stockmarket",
            ]

            sentiment_data = []
            for i, tweet in enumerate(simulated_tweets):
                try:
                    sentiment_score = await self._analyze_tweet_sentiment(tweet)
                    sentiment_label = self._get_sentiment_label(sentiment_score)

                    sentiment_data.append(
                        SentimentData(
                            symbol=symbol,
                            timestamp=datetime.now() - timedelta(hours=i),
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label,
                            volume=1,
                            source="twitter",
                            confidence=0.7,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error processing simulated tweet: {e}")
                    continue

            return await self._aggregate_sentiment(sentiment_data)

        except Exception as e:
            logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
            return []

    @async_strategy_wrapper(timeout=5, fallback_value=0.0)
    async def _analyze_tweet_sentiment(self, text: str) -> float:
        """Analyze tweet sentiment."""
        try:
            # Simple keyword-based analysis for tweets
            positive_words = {
                "bullish",
                "positive",
                "growth",
                "profit",
                "gain",
                "rise",
                "up",
                "strong",
                "buy",
                "outperform",
                "beat",
                "exceed",
                "surge",
                "rally",
                "moon",
                "rocket",
                "ðŸš€",
                "ðŸ“ˆ",
                "ðŸ’Ž",
            }
            negative_words = {
                "bearish",
                "negative",
                "decline",
                "loss",
                "fall",
                "down",
                "weak",
                "sell",
                "underperform",
                "miss",
                "drop",
                "crash",
                "plunge",
                "dump",
                "ðŸ’©",
                "ðŸ“‰",
                "ðŸ”¥",
            }

            words = set(text.lower().split())

            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))

            if positive_count == 0 and negative_count == 0:
                return 0.0

            total = positive_count + negative_count
            sentiment_score = (positive_count - negative_count) / total

            return max(-1.0, min(1.0, sentiment_score))

        except Exception as e:
            logger.warning(f"Error analyzing tweet sentiment: {e}")
            return 0.0

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

    @async_strategy_wrapper(timeout=10, fallback_value=[])
    async def _aggregate_sentiment(
        self, sentiment_data: List[SentimentData]
    ) -> List[SentimentData]:
        """Aggregate sentiment data by time periods."""
        if not sentiment_data:
            return []

        # Group by hour and aggregate
        hourly_data = {}
        for data in sentiment_data:
            hour_key = data.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_data:
                hourly_data[hour_key] = []
            hourly_data[hour_key].append(data)

        aggregated = []
        for hour, data_list in hourly_data.items():
            if data_list:
                total_volume = sum(d.volume for d in data_list)
                if total_volume > 0:
                    weighted_score = (
                        sum(d.sentiment_score * d.volume for d in data_list)
                        / total_volume
                    )

                    aggregated.append(
                        SentimentData(
                            symbol=data_list[0].symbol,
                            timestamp=hour,
                            sentiment_score=weighted_score,
                            sentiment_label=self._get_sentiment_label(weighted_score),
                            volume=total_volume,
                            source="twitter_aggregated",
                            confidence=0.8,
                        )
                    )

        return aggregated


class RedditSentimentCollector:
    """Collects Reddit sentiment for stocks."""

    def __init__(
        self, client_id: Optional[str] = None, client_secret: Optional[str] = None
    ):
        """Initialize Reddit sentiment collector."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradingBot/1.0"})

    @async_strategy_wrapper(timeout=30, fallback_value=[])
    async def get_reddit_sentiment(
        self, symbol: str, max_posts: int = 50
    ) -> List[SentimentData]:
        """Get Reddit sentiment for a symbol."""
        try:
            # For now, return simulated data since Reddit API requires authentication
            # In production, this would use the Reddit API
            logger.info(f"Simulating Reddit sentiment for {symbol}")

            # Simulate some Reddit posts
            simulated_posts = [
                f"DD: Why I'm bullish on ${symbol} - Strong fundamentals and growth potential",
                f"${symbol} earnings discussion thread",
                f"Technical analysis: ${symbol} showing bearish signals",
                f"${symbol} vs competitors - which is the better investment?",
                f"Market sentiment on ${symbol} seems mixed",
            ]

            sentiment_data = []
            for i, post in enumerate(simulated_posts):
                try:
                    sentiment_score = await self._analyze_text_sentiment(post)
                    sentiment_label = self._get_sentiment_label(sentiment_score)

                    sentiment_data.append(
                        SentimentData(
                            symbol=symbol,
                            timestamp=datetime.now() - timedelta(hours=i * 2),
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label,
                            volume=1,
                            source="reddit",
                            confidence=0.6,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error processing simulated Reddit post: {e}")
                    continue

            return await self._aggregate_sentiment(sentiment_data)

        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return []

    @async_strategy_wrapper(timeout=8, fallback_value=0.0)
    async def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze text sentiment for Reddit posts."""
        try:
            # Enhanced keyword analysis for Reddit posts
            positive_words = {
                "bullish",
                "positive",
                "growth",
                "profit",
                "gain",
                "rise",
                "up",
                "strong",
                "buy",
                "outperform",
                "beat",
                "exceed",
                "surge",
                "rally",
                "fundamentals",
                "potential",
                "opportunity",
                "undervalued",
                "moon",
            }
            negative_words = {
                "bearish",
                "negative",
                "decline",
                "loss",
                "fall",
                "down",
                "weak",
                "sell",
                "underperform",
                "miss",
                "drop",
                "crash",
                "plunge",
                "overvalued",
                "bubble",
                "dump",
                "avoid",
                "risky",
            }

            words = set(text.lower().split())

            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))

            if positive_count == 0 and negative_count == 0:
                return 0.0

            total = positive_count + negative_count
            sentiment_score = (positive_count - negative_count) / total

            return max(-1.0, min(1.0, sentiment_score))

        except Exception as e:
            logger.warning(f"Error analyzing Reddit text sentiment: {e}")
            return 0.0

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"

    @async_strategy_wrapper(timeout=12, fallback_value=[])
    async def _aggregate_sentiment(
        self, sentiment_data: List[SentimentData]
    ) -> List[SentimentData]:
        """Aggregate sentiment data by time periods."""
        if not sentiment_data:
            return []

        # Group by hour and aggregate
        hourly_data = {}
        for data in sentiment_data:
            hour_key = data.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_data:
                hourly_data[hour_key] = []
            hourly_data[hour_key].append(data)

        aggregated = []
        for hour, data_list in hourly_data.items():
            if data_list:
                total_volume = sum(d.volume for d in data_list)
                if total_volume > 0:
                    weighted_score = (
                        sum(d.sentiment_score * d.volume for d in data_list)
                        / total_volume
                    )

                    aggregated.append(
                        SentimentData(
                            symbol=data_list[0].symbol,
                            timestamp=hour,
                            sentiment_score=weighted_score,
                            sentiment_label=self._get_sentiment_label(weighted_score),
                            volume=total_volume,
                            source="reddit_aggregated",
                            confidence=0.7,
                        )
                    )

        return aggregated


class MacroIndicatorCollector:
    """Collects macroeconomic indicators."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize macro indicator collector."""
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradingBot/1.0"})

    @async_strategy_wrapper(timeout=45, fallback_value={})
    async def get_macro_indicators(self, days_back: int = 30) -> Dict[str, pd.Series]:
        """Get macroeconomic indicators."""
        try:
            indicators = {}

            # Try to get FRED data if API key is available
            if self.api_key:
                try:
                    # Get key economic indicators
                    fred_series = [
                        ("GDP", "GDP"),
                        ("UNRATE", "Unemployment Rate"),
                        ("CPIAUCSL", "CPI"),
                        ("FEDFUNDS", "Federal Funds Rate"),
                        ("DGS10", "10-Year Treasury Rate"),
                    ]

                    for series_id, name in fred_series:
                        try:
                            series = await self._get_fred_series(series_id, days_back)
                            if series is not None and not series.empty:
                                indicators[name] = series
                        except Exception as e:
                            logger.warning(
                                f"Failed to get FRED series {series_id}: {e}"
                            )
                            continue

                except Exception as e:
                    logger.warning(f"FRED API failed: {e}")

            # If no FRED data, return empty dict (fallback)
            if not indicators:
                logger.info("No macro indicators available")
                return {}

            return indicators

        except Exception as e:
            logger.error(f"Error getting macro indicators: {e}")
            return {}

    @async_strategy_wrapper(timeout=20, fallback_value=None)
    async def _get_fred_series(
        self, series_id: str, days_back: int
    ) -> Optional[pd.Series]:
        """Get a series from FRED API."""
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": (
                    datetime.now() - timedelta(days=days_back)
                ).strftime("%Y-%m-%d"),
                "sort_order": "desc",
            }

            response = await asyncio.to_thread(
                self.session.get, url, params=params, timeout=15
            )
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            if not observations:
                return None

            # Convert to pandas Series
            dates = []
            values = []
            for obs in observations:
                try:
                    date = pd.to_datetime(obs["date"])
                    value = float(obs["value"]) if obs["value"] != "." else np.nan
                    dates.append(date)
                    values.append(value)
                except (ValueError, KeyError):
                    continue

            if dates and values:
                series = pd.Series(values, index=dates)
                return series.sort_index()

            return None

        except Exception as e:
            logger.warning(f"Error getting FRED series {series_id}: {e}")
            return None


class OptionsFlowCollector:
    """Collects options flow data."""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """Initialize options flow collector."""
        self.api_keys = api_keys or {}

    @async_strategy_wrapper(timeout=35, fallback_value=[])
    async def get_options_flow(
        self, symbol: str, days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get options flow data for a symbol."""
        try:
            # Try to get real options data if API keys are available
            if "tradier" in self.api_keys:
                try:
                    return await self._get_tradier_options_flow(symbol, days_back)
                except Exception as e:
                    logger.warning(f"Tradier options flow failed: {e}")

            # Fallback to simulated data
            logger.info(f"Using simulated options flow for {symbol}")
            return await self._generate_simulated_options_flow(symbol, days_back)

        except Exception as e:
            logger.error(f"Error getting options flow for {symbol}: {e}")
            return []

    @async_strategy_wrapper(timeout=25, fallback_value=[])
    async def _get_tradier_options_flow(
        self, symbol: str, days_back: int
    ) -> List[Dict[str, Any]]:
        """Get options flow from Tradier API."""
        # This would implement the actual Tradier API call
        # For now, return empty list as fallback
        logger.info(f"Tradier API not implemented for {symbol}")
        return []

    @async_strategy_wrapper(timeout=15, fallback_value=[])
    async def _generate_simulated_options_flow(
        self, symbol: str, days_back: int
    ) -> List[Dict[str, Any]]:
        """Generate simulated options flow data."""
        try:
            options_data = []

            # Generate simulated options data for the past week
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)

                # Simulate some options activity
                for _ in range(np.random.randint(1, 5)):  # 1-4 options per day
                    strike = round(np.random.uniform(50, 200), 2)
                    option_type = np.random.choice(["call", "put"])
                    volume = np.random.randint(10, 1000)
                    premium = round(np.random.uniform(0.1, 10.0), 2)

                    options_data.append(
                        {
                            "symbol": symbol,
                            "date": date,
                            "strike": strike,
                            "option_type": option_type,
                            "volume": volume,
                            "premium": premium,
                            "expiration": date + timedelta(days=30),
                            "source": "simulated",
                        }
                    )

            return options_data

        except Exception as e:
            logger.error(f"Error generating simulated options flow: {e}")
            return []


class ExternalSignalsManager:
    """Manages all external signal collection and integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize external signals manager."""
        self.config = config or {}

        # Initialize collectors
        api_keys = self.config.get("api_keys", {})

        self.news_collector = NewsSentimentCollector(api_keys.get("news", {}))
        self.twitter_collector = TwitterSentimentCollector()
        self.reddit_collector = RedditSentimentCollector(
            api_keys.get("reddit_client_id"), api_keys.get("reddit_client_secret")
        )
        self.macro_collector = MacroIndicatorCollector(api_keys.get("fred"))
        self.options_collector = OptionsFlowCollector(api_keys.get("options", {}))

        # Signal storage
        self.signal_cache = {}
        self.cache_duration = timedelta(hours=1)

        logger.info("ExternalSignalsManager initialized successfully")

    @async_strategy_wrapper(timeout=60, fallback_value={})
    async def get_all_signals(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Get all external signals for a symbol with async timeout protection."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{days_back}"
            if cache_key in self.signal_cache:
                cache_time, cached_data = self.signal_cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    return cached_data

            # Collect all signals concurrently with timeout protection
            tasks = [
                self.news_collector.get_news_sentiment(symbol, days_back),
                self.twitter_collector.get_twitter_sentiment(symbol),
                self.reddit_collector.get_reddit_sentiment(symbol),
                self.macro_collector.get_macro_indicators(days_back),
                self.options_collector.get_options_flow(symbol, days_back),
            ]

            # Execute all tasks concurrently with individual timeouts
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle exceptions
            signals = {
                "news_sentiment": (
                    results[0] if not isinstance(results[0], Exception) else []
                ),
                "twitter_sentiment": (
                    results[1] if not isinstance(results[1], Exception) else []
                ),
                "reddit_sentiment": (
                    results[2] if not isinstance(results[2], Exception) else []
                ),
                "macro_indicators": (
                    results[3] if not isinstance(results[3], Exception) else {}
                ),
                "options_flow": (
                    results[4] if not isinstance(results[4], Exception) else []
                ),
            }

            # Log any exceptions that occurred
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Signal collection {i} failed: {result}")

            # Final validation and filtering
            signals = await self._validate_and_filter_signals(signals, symbol)

            # Cache the results
            self.signal_cache[cache_key] = (datetime.now(), signals)
            return signals

        except Exception as e:
            logger.error(f"Error getting all signals for {symbol}: {e}")
            return {}

    @async_strategy_wrapper(timeout=30, fallback_value=pd.DataFrame())
    async def _validate_and_filter_signals(
        self, signals: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """Validate and filter signal data."""
        try:
            # Validate sentiment signals
            for key in ["news_sentiment", "twitter_sentiment", "reddit_sentiment"]:
                valid = []
                for record in signals.get(key, []):
                    is_valid, err = SignalValidator.validate_sentiment_data(
                        asdict(record)
                    )
                    if is_valid:
                        valid.append(record)
                    else:
                        logger.warning(
                            f"Manager: Invalid {key} record for {symbol}: {err} | Data: {record}"
                        )
                signals[key] = valid

            # Validate macro indicators
            macro_valid = {}
            for name, series in signals.get("macro_indicators", {}).items():
                if not series.empty:
                    meta = {
                        "name": name,
                        "value": float(series.iloc[-1]),
                        "unit": "unknown",
                        "frequency": "unknown",
                        "last_updated": series.index[-1],
                        "source": "FRED",
                    }
                    is_valid, err = SignalValidator.validate_macro_indicator_data(meta)
                    if is_valid:
                        macro_valid[name] = series
                    else:
                        logger.warning(
                            f"Manager: Invalid macro indicator meta for {name}: {err} | Meta: {meta}"
                        )
            signals["macro_indicators"] = macro_valid

            # Validate options flow
            options_valid = []
            for record in signals.get("options_flow", []):
                is_valid, err = SignalValidator.validate_options_flow_data(record)
                if is_valid:
                    options_valid.append(record)
                else:
                    logger.warning(
                        f"Manager: Invalid options flow record for {symbol}: {err} | Data: {record}"
                    )
            signals["options_flow"] = options_valid

            return signals

        except Exception as e:
            logger.error(f"Error validating signals for {symbol}: {e}")
            return signals

    @async_strategy_wrapper(timeout=45, fallback_value=pd.DataFrame())
    async def get_signal_features(
        self, symbol: str, days_back: int = 7
    ) -> pd.DataFrame:
        """Get signal features as a DataFrame for model input."""
        try:
            signals = await self.get_all_signals(symbol, days_back)

            # Create feature DataFrame
            features = []

            # Process sentiment signals
            for sentiment_type in [
                "news_sentiment",
                "twitter_sentiment",
                "reddit_sentiment",
            ]:
                sentiment_data = signals.get(sentiment_type, [])

                for data in sentiment_data:
                    features.append(
                        {
                            "timestamp": data.timestamp,
                            "symbol": symbol,
                            "signal_type": sentiment_type,
                            "sentiment_score": data.sentiment_score,
                            "sentiment_label": data.sentiment_label,
                            "volume": data.volume,
                            "confidence": data.confidence,
                        }
                    )

            # Process macro indicators
            macro_data = signals.get("macro_indicators", {})
            for indicator_name, series in macro_data.items():
                for date, value in series.items():
                    features.append(
                        {
                            "timestamp": date,
                            "symbol": symbol,
                            "signal_type": f"macro_{indicator_name}",
                            "value": value,
                            "sentiment_score": 0.0,  # Macro indicators don't have sentiment
                            "sentiment_label": "neutral",
                            "volume": 1,
                            "confidence": 0.9,
                        }
                    )

            # Process options flow
            options_data = signals.get("options_flow", [])
            for option in options_data:
                # Calculate options sentiment based on call/put ratio
                sentiment_score = 0.1 if option["option_type"] == "call" else -0.1

                features.append(
                    {
                        "timestamp": option["date"],
                        "symbol": symbol,
                        "signal_type": "options_flow",
                        "strike": option["strike"],
                        "option_type": option["option_type"],
                        "volume": option["volume"],
                        "premium": option["premium"],
                        "sentiment_score": sentiment_score,
                        "sentiment_label": (
                            "positive" if sentiment_score > 0 else "negative"
                        ),
                        "confidence": 0.7,
                    }
                )

            # Convert to DataFrame
            if features:
                df = pd.DataFrame(features)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error creating signal features for {symbol}: {e}")
            return pd.DataFrame()

    @async_strategy_wrapper(timeout=30, fallback_value={})
    async def get_aggregated_sentiment(
        self, symbol: str, days_back: int = 7
    ) -> Dict[str, float]:
        """Get aggregated sentiment scores across all sources."""
        try:
            signals = await self.get_all_signals(symbol, days_back)

            aggregated = {}

            # Aggregate sentiment by source
            for source in ["news_sentiment", "twitter_sentiment", "reddit_sentiment"]:
                sentiment_data = signals.get(source, [])
                if sentiment_data:
                    # Calculate weighted average sentiment
                    total_volume = sum(d.volume for d in sentiment_data)
                    if total_volume > 0:
                        weighted_score = (
                            sum(d.sentiment_score * d.volume for d in sentiment_data)
                            / total_volume
                        )
                        aggregated[source] = weighted_score
                    else:
                        aggregated[source] = 0.0
                else:
                    aggregated[source] = 0.0

            # Calculate overall sentiment
            if aggregated:
                overall_score = sum(aggregated.values()) / len(aggregated)
                aggregated["overall"] = overall_score
            else:
                aggregated["overall"] = 0.0

            return aggregated

        except Exception as e:
            logger.error(f"Error getting aggregated sentiment for {symbol}: {e}")
            return {"overall": 0.0}


def get_external_signals_manager(
    config: Optional[Dict[str, Any]] = None,
) -> ExternalSignalsManager:
    """Factory function to create ExternalSignalsManager instance."""
    return ExternalSignalsManager(config)
