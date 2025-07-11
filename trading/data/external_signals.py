"""
External Signal Integration

This module provides comprehensive integration with external data sources for enhanced
trading signals including news sentiment, social media sentiment, macro indicators,
insider trading, and options flow data.
"""

import logging
import requests
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# External API imports
try:
    import snscrape.modules.twitter as sntwitter
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

logger = logging.getLogger(__name__)

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
    """Validation layer to ensure incoming signal schema matches expected format."""
    
    @staticmethod
    def validate_signal_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate incoming signal data against expected schema.
        
        Args:
            data: Incoming signal data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Required fields
            required_fields = ['source', 'symbol', 'timestamp', 'signal_type', 'value', 'confidence']
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            # Type validation
            if not isinstance(data['source'], str):
                return False, "source must be a string"
            
            if not isinstance(data['symbol'], str):
                return False, "symbol must be a string"
            
            if not isinstance(data['signal_type'], str):
                return False, "signal_type must be a string"
            
            if not isinstance(data['value'], (int, float)):
                return False, "value must be a number"
            
            if not isinstance(data['confidence'], (int, float)):
                return False, "confidence must be a number"
            
            # Value range validation
            if not (0 <= data['confidence'] <= 1):
                return False, "confidence must be between 0 and 1"
            
            # Timestamp validation
            try:
                if isinstance(data['timestamp'], str):
                    datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                elif not isinstance(data['timestamp'], datetime):
                    return False, "timestamp must be a datetime object or ISO string"
            except ValueError:
                return False, "Invalid timestamp format"
            
            # Optional metadata validation
            if 'metadata' in data and not isinstance(data['metadata'], dict):
                return False, "metadata must be a dictionary"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_sentiment_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate sentiment data against expected schema.
        
        Args:
            data: Incoming sentiment data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Required fields for sentiment data
            required_fields = ['symbol', 'timestamp', 'sentiment_score', 'sentiment_label', 'source']
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            
            # Type validation
            if not isinstance(data['symbol'], str):
                return False, "symbol must be a string"
            
            if not isinstance(data['sentiment_score'], (int, float)):
                return False, "sentiment_score must be a number"
            
            if not isinstance(data['sentiment_label'], str):
                return False, "sentiment_label must be a string"
            
            if not isinstance(data['source'], str):
                return False, "source must be a string"
            
            # Value range validation
            if not (-1 <= data['sentiment_score'] <= 1):
                return False, "sentiment_score must be between -1 and 1"
            
            # Valid sentiment labels
            valid_labels = ['positive', 'negative', 'neutral']
            if data['sentiment_label'] not in valid_labels:
                return False, f"sentiment_label must be one of: {valid_labels}"
            
            # Timestamp validation
            try:
                if isinstance(data['timestamp'], str):
                    datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                elif not isinstance(data['timestamp'], datetime):
                    return False, "timestamp must be a datetime object or ISO string"
            except ValueError:
                return False, "Invalid timestamp format"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def validate_options_flow_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate options flow data against expected schema."""
        try:
            required_fields = ['symbol', 'timestamp', 'option_type', 'strike', 'expiry', 'side', 'volume', 'open_interest', 'source']
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            if not isinstance(data['symbol'], str):
                return False, "symbol must be a string"
            if not isinstance(data['option_type'], str) or data['option_type'] not in ['call', 'put']:
                return False, "option_type must be 'call' or 'put'"
            if not isinstance(data['strike'], (int, float)) or data['strike'] <= 0:
                return False, "strike must be a positive number"
            if not isinstance(data['expiry'], str):
                return False, "expiry must be a string (date)"
            if not isinstance(data['side'], str) or data['side'] not in ['buy', 'sell']:
                return False, "side must be 'buy' or 'sell'"
            if not isinstance(data['volume'], int) or data['volume'] < 0:
                return False, "volume must be a non-negative integer"
            if not isinstance(data['open_interest'], int) or data['open_interest'] < 0:
                return False, "open_interest must be a non-negative integer"
            if not isinstance(data['source'], str):
                return False, "source must be a string"
            # Timestamp validation
            try:
                if isinstance(data['timestamp'], str):
                    datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                elif not isinstance(data['timestamp'], datetime):
                    return False, "timestamp must be a datetime object or ISO string"
            except ValueError:
                return False, "Invalid timestamp format"
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def validate_macro_indicator_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate macro indicator data against expected schema."""
        try:
            required_fields = ['name', 'value', 'unit', 'frequency', 'last_updated', 'source']
            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"
            if not isinstance(data['name'], str):
                return False, "name must be a string"
            if not isinstance(data['value'], (int, float)):
                return False, "value must be a number"
            if not isinstance(data['unit'], str):
                return False, "unit must be a string"
            if not isinstance(data['frequency'], str):
                return False, "frequency must be a string"
            if not isinstance(data['source'], str):
                return False, "source must be a string"
            # Timestamp validation
            try:
                if isinstance(data['last_updated'], str):
                    datetime.fromisoformat(data['last_updated'].replace('Z', '+00:00'))
                elif not isinstance(data['last_updated'], datetime):
                    return False, "last_updated must be a datetime object or ISO string"
            except ValueError:
                return False, "Invalid last_updated format"
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

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
    """Collects news sentiment from various sources."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """Initialize news sentiment collector.
        
        Args:
            api_keys: Dictionary containing API keys for news services
        """
        self.api_keys = api_keys or {}
        self.sources = {
            'newsapi': 'https://newsapi.org/v2/everything',
            'gnews': 'https://gnews.io/api/v4/search'
        }
        
        # Rate limiting
        self.last_request = {}
        self.rate_limit_delay = 1.0  # seconds
    
    def get_news_sentiment(self, symbol: str, days_back: int = 7) -> List[SentimentData]:
        """Get news sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            List of sentiment data
        """
        sentiment_data = []
        
        # Collect from multiple sources
        if 'newsapi' in self.api_keys:
            newsapi_data = self._get_newsapi_sentiment(symbol, days_back)
            sentiment_data.extend(newsapi_data)
        
        if 'gnews' in self.api_keys:
            gnews_data = self._get_gnews_sentiment(symbol, days_back)
            sentiment_data.extend(gnews_data)
        
        # Validate and filter
        validated = []
        for record in sentiment_data:
            is_valid, err = SignalValidator.validate_sentiment_data(asdict(record))
            if is_valid:
                validated.append(record)
            else:
                logger.warning(f"Invalid news sentiment record for {symbol}: {err} | Data: {record}")
        
        # Aggregate and normalize sentiment
        aggregated_data = self._aggregate_sentiment(validated)
        
        return aggregated_data
    
    def _get_newsapi_sentiment(self, symbol: str, days_back: int) -> List[SentimentData]:
        """Get sentiment from NewsAPI."""
        try:
            self._rate_limit('newsapi')
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # API parameters
            params = {
                'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.api_keys['newsapi']
            }
            
            response = requests.get(self.sources['newsapi'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                sentiment_data = []
                for article in articles:
                    # Simple sentiment analysis based on title and description
                    sentiment_score = self._analyze_text_sentiment(
                        article.get('title', '') + ' ' + article.get('description', '')
                    )
                    
                    sentiment_data.append(SentimentData(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                        sentiment_score=sentiment_score,
                        sentiment_label=self._get_sentiment_label(sentiment_score),
                        volume=1,
                        source='newsapi',
                        confidence=0.7
                    ))
                
                return sentiment_data
            else:
                logger.warning(f"NewsAPI request failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting NewsAPI sentiment: {e}")
            return []
    
    def _get_gnews_sentiment(self, symbol: str, days_back: int) -> List[SentimentData]:
        """Get sentiment from GNews."""
        try:
            self._rate_limit('gnews')
            
            # API parameters
            params = {
                'q': f'"{symbol}" stock',
                'lang': 'en',
                'country': 'us',
                'max': 100,
                'apikey': self.api_keys['gnews']
            }
            
            response = requests.get(self.sources['gnews'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                sentiment_data = []
                for article in articles:
                    # Simple sentiment analysis
                    sentiment_score = self._analyze_text_sentiment(
                        article.get('title', '') + ' ' + article.get('description', '')
                    )
                    
                    sentiment_data.append(SentimentData(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(article['publishedAt']['dateTime']),
                        sentiment_score=sentiment_score,
                        sentiment_label=self._get_sentiment_label(sentiment_score),
                        volume=1,
                        source='gnews',
                        confidence=0.7
                    ))
                
                return sentiment_data
            else:
                logger.warning(f"GNews request failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting GNews sentiment: {e}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keyword matching."""
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = [
            'bullish', 'surge', 'rally', 'gain', 'up', 'positive', 'growth',
            'profit', 'earnings', 'beat', 'exceed', 'strong', 'buy', 'upgrade'
        ]
        
        # Negative keywords
        negative_words = [
            'bearish', 'drop', 'fall', 'decline', 'down', 'negative', 'loss',
            'miss', 'weak', 'sell', 'downgrade', 'crash', 'plunge'
        ]
        
        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment_score))
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _aggregate_sentiment(self, sentiment_data: List[SentimentData]) -> List[SentimentData]:
        """Aggregate sentiment data by time periods."""
        if not sentiment_data:
            return []
        
        # Group by hour
        df = pd.DataFrame([asdict(data) for data in sentiment_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Aggregate by hour
        aggregated = df.groupby('hour').agg({
            'sentiment_score': 'mean',
            'volume': 'sum',
            'confidence': 'mean'
        }).reset_index()
        
        # Convert back to SentimentData objects
        aggregated_data = []
        for _, row in aggregated.iterrows():
            aggregated_data.append(SentimentData(
                symbol=sentiment_data[0].symbol,
                timestamp=row['hour'],
                sentiment_score=row['sentiment_score'],
                sentiment_label=self._get_sentiment_label(row['sentiment_score']),
                volume=row['volume'],
                source='aggregated',
                confidence=row['confidence']
            ))
        
        return aggregated_data
    
    def _rate_limit(self, source: str):
        """Implement rate limiting."""
        current_time = time.time()
        if source in self.last_request:
            time_since_last = current_time - self.last_request[source]
            if time_since_last < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request[source] = time.time()

class TwitterSentimentCollector:
    """Collects Twitter/X sentiment using snscrape."""
    
    def __init__(self):
        """Initialize Twitter sentiment collector."""
        self.search_terms = [
            'stock', 'trading', 'investing', 'market', 'bullish', 'bearish'
        ]
    
    def get_twitter_sentiment(self, symbol: str, max_tweets: int = 100) -> List[SentimentData]:
        """Get Twitter sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            max_tweets: Maximum number of tweets to analyze
            
        Returns:
            List of sentiment data
        """
        if not TWITTER_AVAILABLE:
            logger.warning("snscrape not available for Twitter sentiment")
            return []
        
        try:
            sentiment_data = []
            
            # Search for tweets about the symbol
            query = f'${symbol} OR #{symbol} OR "{symbol} stock"'
            
            # Collect tweets
            tweets = []
            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                if len(tweets) >= max_tweets:
                    break
                tweets.append(tweet)
            
            # Analyze sentiment for each tweet
            for tweet in tweets:
                sentiment_score = self._analyze_tweet_sentiment(tweet.rawContent)
                
                sentiment_data.append(SentimentData(
                    symbol=symbol,
                    timestamp=tweet.date,
                    sentiment_score=sentiment_score,
                    sentiment_label=self._get_sentiment_label(sentiment_score),
                    volume=1,
                    source='twitter',
                    confidence=0.6
                ))
            
            # Validate and filter
            validated = []
            for record in sentiment_data:
                is_valid, err = SignalValidator.validate_sentiment_data(asdict(record))
                if is_valid:
                    validated.append(record)
                else:
                    logger.warning(f"Invalid twitter sentiment record for {symbol}: {err} | Data: {record}")
            
            # Aggregate sentiment
            aggregated_data = self._aggregate_sentiment(validated)
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return []
    
    def _analyze_tweet_sentiment(self, text: str) -> float:
        """Analyze sentiment of a tweet."""
        text_lower = text.lower()
        
        # Emoji sentiment
        positive_emojis = ['🚀', '📈', '💎', '🔥', '✅', '👍', '💪', '🎯']
        negative_emojis = ['📉', '💩', '🔥', '💀', '👎', '😱', '😭', '🤡']
        
        # Count emojis
        positive_emoji_count = sum(1 for emoji in positive_emojis if emoji in text)
        negative_emoji_count = sum(1 for emoji in negative_emojis if emoji in text)
        
        # Text sentiment
        positive_words = ['bullish', 'moon', 'rocket', 'buy', 'long', 'hodl', 'diamond']
        negative_words = ['bearish', 'dump', 'sell', 'short', 'paper', 'weak']
        
        positive_word_count = sum(1 for word in positive_words if word in text_lower)
        negative_word_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate combined sentiment
        total_signals = positive_emoji_count + negative_emoji_count + positive_word_count + negative_word_count
        
        if total_signals == 0:
            return 0.0
        
        sentiment_score = (
            (positive_emoji_count + positive_word_count) - 
            (negative_emoji_count + negative_word_count)
        ) / total_signals
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _aggregate_sentiment(self, sentiment_data: List[SentimentData]) -> List[SentimentData]:
        """Aggregate sentiment data by time periods."""
        if not sentiment_data:
            return []
        
        # Group by hour
        df = pd.DataFrame([asdict(data) for data in sentiment_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Aggregate by hour
        aggregated = df.groupby('hour').agg({
            'sentiment_score': 'mean',
            'volume': 'sum',
            'confidence': 'mean'
        }).reset_index()
        
        # Convert back to SentimentData objects
        aggregated_data = []
        for _, row in aggregated.iterrows():
            aggregated_data.append(SentimentData(
                symbol=sentiment_data[0].symbol,
                timestamp=row['hour'],
                sentiment_score=row['sentiment_score'],
                sentiment_label=self._get_sentiment_label(row['sentiment_score']),
                volume=row['volume'],
                source='twitter_aggregated',
                confidence=row['confidence']
            ))
        
        return aggregated_data

class RedditSentimentCollector:
    """Collects Reddit sentiment from WallStreetBets and other subreddits."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """Initialize Reddit sentiment collector."""
        self.reddit = None
        
        if REDDIT_AVAILABLE and client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent='EvolveTradingBot/1.0'
                )
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
        
        self.subreddits = ['wallstreetbets', 'stocks', 'investing', 'options']
    
    def get_reddit_sentiment(self, symbol: str, max_posts: int = 50) -> List[SentimentData]:
        """Get Reddit sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            max_posts: Maximum number of posts to analyze
            
        Returns:
            List of sentiment data
        """
        if not self.reddit:
            logger.warning("Reddit client not available")
            return []
        
        try:
            sentiment_data = []
            
            for subreddit_name in self.subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts about the symbol
                    search_query = f'{symbol}'
                    posts = subreddit.search(search_query, limit=max_posts//len(self.subreddits))
                    
                    for post in posts:
                        # Analyze post title and content
                        title_sentiment = self._analyze_text_sentiment(post.title)
                        content_sentiment = self._analyze_text_sentiment(post.selftext)
                        
                        # Combine sentiment
                        combined_sentiment = (title_sentiment * 0.7 + content_sentiment * 0.3)
                        
                        sentiment_data.append(SentimentData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(post.created_utc),
                            sentiment_score=combined_sentiment,
                            sentiment_label=self._get_sentiment_label(combined_sentiment),
                            volume=post.score,  # Use upvotes as volume
                            source=f'reddit_{subreddit_name}',
                            confidence=0.6
                        ))
                
                except Exception as e:
                    logger.error(f"Error getting posts from r/{subreddit_name}: {e}")
                    continue
            
            # Validate and filter
            validated = []
            for record in sentiment_data:
                is_valid, err = SignalValidator.validate_sentiment_data(asdict(record))
                if is_valid:
                    validated.append(record)
                else:
                    logger.warning(f"Invalid reddit sentiment record for {symbol}: {err} | Data: {record}")
            
            # Aggregate sentiment
            aggregated_data = self._aggregate_sentiment(validated)
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of Reddit text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Reddit-specific sentiment indicators
        positive_words = [
            'bullish', 'moon', 'rocket', 'buy', 'long', 'hodl', 'diamond hands',
            'tendies', 'gains', 'profit', 'win', 'success', 'upvote', 'based'
        ]
        
        negative_words = [
            'bearish', 'dump', 'sell', 'short', 'paper hands', 'loss', 'fail',
            'downvote', 'cringe', 'fud', 'manipulation', 'scam'
        ]
        
        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment_score))
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _aggregate_sentiment(self, sentiment_data: List[SentimentData]) -> List[SentimentData]:
        """Aggregate sentiment data by time periods."""
        if not sentiment_data:
            return []
        
        # Group by hour
        df = pd.DataFrame([asdict(data) for data in sentiment_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.floor('H')
        
        # Aggregate by hour
        aggregated = df.groupby('hour').agg({
            'sentiment_score': 'mean',
            'volume': 'sum',
            'confidence': 'mean'
        }).reset_index()
        
        # Convert back to SentimentData objects
        aggregated_data = []
        for _, row in aggregated.iterrows():
            aggregated_data.append(SentimentData(
                symbol=sentiment_data[0].symbol,
                timestamp=row['hour'],
                sentiment_score=row['sentiment_score'],
                sentiment_label=self._get_sentiment_label(row['sentiment_score']),
                volume=row['volume'],
                source='reddit_aggregated',
                confidence=row['confidence']
            ))
        
        return aggregated_data

class MacroIndicatorCollector:
    """Collects macroeconomic indicators from FRED API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize macro indicator collector."""
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series"
        
        # Common macro indicators
        self.indicators = {
            'CPI': 'CPIAUCSL',  # Consumer Price Index
            'FED_FUNDS_RATE': 'FEDFUNDS',  # Federal Funds Rate
            'UNEMPLOYMENT': 'UNRATE',  # Unemployment Rate
            'GDP': 'GDP',  # Gross Domestic Product
            'VIX': 'VIXCLS',  # VIX Volatility Index
            'TREASURY_10Y': 'GS10',  # 10-Year Treasury Rate
            'TREASURY_2Y': 'GS2',  # 2-Year Treasury Rate
        }
    
    def get_macro_indicators(self, days_back: int = 30) -> Dict[str, pd.Series]:
        """Get macro indicators.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary of indicator data
        """
        if not self.api_key:
            logger.warning("FRED API key not provided")
            return {}
        
        try:
            indicator_data = {}
            
            for indicator_name, series_id in self.indicators.items():
                try:
                    data = self._get_fred_series(series_id, days_back)
                    if data is not None:
                        # Validate macro indicator meta
                        meta = {
                            'name': indicator_name,
                            'value': float(data.iloc[-1]) if not data.empty else None,
                            'unit': 'unknown',
                            'frequency': 'unknown',
                            'last_updated': data.index[-1] if not data.empty else None,
                            'source': 'FRED'
                        }
                        is_valid, err = SignalValidator.validate_macro_indicator_data(meta)
                        if is_valid:
                            indicator_data[indicator_name] = data
                        else:
                            logger.warning(f"Invalid macro indicator meta for {indicator_name}: {err} | Meta: {meta}")
                except Exception as e:
                    logger.error(f"Error getting {indicator_name}: {e}")
                    continue
            
            return indicator_data
            
        except Exception as e:
            logger.error(f"Error getting macro indicators: {e}")
            return {}
    
    def _get_fred_series(self, series_id: str, days_back: int) -> Optional[pd.Series]:
        """Get FRED series data."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # API parameters
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                # Convert to pandas Series
                dates = []
                values = []
                
                for obs in observations:
                    try:
                        date = pd.to_datetime(obs['date'])
                        value = float(obs['value']) if obs['value'] != '.' else np.nan
                        dates.append(date)
                        values.append(value)
                    except (ValueError, KeyError):
                        continue
                
                if dates and values:
                    series = pd.Series(values, index=dates, name=series_id)
                    return series
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting FRED series {series_id}: {e}")
            return None

class OptionsFlowCollector:
    """Collects options flow data from various sources."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """Initialize options flow collector."""
        self.api_keys = api_keys or {}
        self.sources = {
            'tradier': 'https://api.tradier.com/v1/markets/options/expirations',
            'barchart': 'https://www.barchart.com/options/unusual-activity'
        }
    
    def get_options_flow(self, symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get options flow data for a symbol.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            List of options flow data
        """
        options_data = []
        
        # Collect from multiple sources
        if 'tradier' in self.api_keys:
            tradier_data = self._get_tradier_options_flow(symbol, days_back)
            options_data.extend(tradier_data)
        
        # Add simulated data for demonstration
        simulated_data = self._generate_simulated_options_flow(symbol, days_back)
        options_data.extend(simulated_data)
        
        # Validate and filter
        validated = []
        for record in options_data:
            is_valid, err = SignalValidator.validate_options_flow_data(record)
            if is_valid:
                validated.append(record)
            else:
                logger.warning(f"Invalid options flow record for {symbol}: {err} | Data: {record}")
        return validated
    
    def _get_tradier_options_flow(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        """Get options flow from Tradier API."""
        try:
            # This would require actual Tradier API integration
            # For now, return empty list
            logger.info(f"Tradier options flow not implemented for {symbol}")
            return []
            
        except Exception as e:
            logger.error(f"Error getting Tradier options flow: {e}")
            return []
    
    def _generate_simulated_options_flow(self, symbol: str, days_back: int) -> List[Dict[str, Any]]:
        """Generate simulated options flow data for demonstration."""
        try:
            options_data = []
            
            # Generate data for the past days
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)
                
                # Simulate unusual options activity
                for _ in range(np.random.randint(1, 5)):  # 1-4 unusual activities per day
                    strike = np.random.choice([50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
                    option_type = np.random.choice(['call', 'put'])
                    volume = np.random.randint(100, 1000)
                    premium = np.random.uniform(1.0, 10.0)
                    
                    options_data.append({
                        'symbol': symbol,
                        'date': date,
                        'strike': strike,
                        'option_type': option_type,
                        'volume': volume,
                        'premium': premium,
                        'unusual_activity': True,
                        'source': 'simulated'
                    })
            
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
        api_keys = self.config.get('api_keys', {})
        
        self.news_collector = NewsSentimentCollector(api_keys.get('news', {}))
        self.twitter_collector = TwitterSentimentCollector()
        self.reddit_collector = RedditSentimentCollector(
            api_keys.get('reddit_client_id'),
            api_keys.get('reddit_client_secret')
        )
        self.macro_collector = MacroIndicatorCollector(api_keys.get('fred'))
        self.options_collector = OptionsFlowCollector(api_keys.get('options', {}))
        
        # Signal storage
        self.signal_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        logger.info("ExternalSignalsManager initialized successfully")
    
    def get_all_signals(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """Get all external signals for a symbol.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            Dictionary containing all signal data
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{days_back}"
            if cache_key in self.signal_cache:
                cache_time, cached_data = self.signal_cache[cache_key]
                if datetime.now() - cache_time < self.cache_duration:
                    return cached_data
            # Collect all signals
            signals = {
                'news_sentiment': self.news_collector.get_news_sentiment(symbol, days_back),
                'twitter_sentiment': self.twitter_collector.get_twitter_sentiment(symbol),
                'reddit_sentiment': self.reddit_collector.get_reddit_sentiment(symbol),
                'macro_indicators': self.macro_collector.get_macro_indicators(days_back),
                'options_flow': self.options_collector.get_options_flow(symbol, days_back)
            }
            # Final validation and filtering
            for key in ['news_sentiment', 'twitter_sentiment', 'reddit_sentiment']:
                valid = []
                for record in signals.get(key, []):
                    is_valid, err = SignalValidator.validate_sentiment_data(asdict(record))
                    if is_valid:
                        valid.append(record)
                    else:
                        logger.warning(f"Manager: Invalid {key} record for {symbol}: {err} | Data: {record}")
                signals[key] = valid
            # Macro indicators: validate meta only (series already validated in collector)
            macro_valid = {}
            for name, series in signals.get('macro_indicators', {}).items():
                if not series.empty:
                    meta = {
                        'name': name,
                        'value': float(series.iloc[-1]),
                        'unit': 'unknown',
                        'frequency': 'unknown',
                        'last_updated': series.index[-1],
                        'source': 'FRED'
                    }
                    is_valid, err = SignalValidator.validate_macro_indicator_data(meta)
                    if is_valid:
                        macro_valid[name] = series
                    else:
                        logger.warning(f"Manager: Invalid macro indicator meta for {name}: {err} | Meta: {meta}")
            signals['macro_indicators'] = macro_valid
            # Options flow
            options_valid = []
            for record in signals.get('options_flow', []):
                is_valid, err = SignalValidator.validate_options_flow_data(record)
                if is_valid:
                    options_valid.append(record)
                else:
                    logger.warning(f"Manager: Invalid options flow record for {symbol}: {err} | Data: {record}")
            signals['options_flow'] = options_valid
            # Cache the results
            self.signal_cache[cache_key] = (datetime.now(), signals)
            return signals
        except Exception as e:
            logger.error(f"Error getting all signals for {symbol}: {e}")
            return {}
    
    def get_signal_features(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """Get signal features as a DataFrame for model input.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            DataFrame with signal features
        """
        try:
            signals = self.get_all_signals(symbol, days_back)
            
            # Create feature DataFrame
            features = []
            
            # Process sentiment signals
            for sentiment_type in ['news_sentiment', 'twitter_sentiment', 'reddit_sentiment']:
                sentiment_data = signals.get(sentiment_type, [])
                
                for data in sentiment_data:
                    features.append({
                        'timestamp': data.timestamp,
                        'symbol': symbol,
                        'signal_type': sentiment_type,
                        'sentiment_score': data.sentiment_score,
                        'sentiment_label': data.sentiment_label,
                        'volume': data.volume,
                        'confidence': data.confidence
                    })
            
            # Process macro indicators
            macro_data = signals.get('macro_indicators', {})
            for indicator_name, series in macro_data.items():
                for date, value in series.items():
                    features.append({
                        'timestamp': date,
                        'symbol': symbol,
                        'signal_type': f'macro_{indicator_name}',
                        'value': value,
                        'sentiment_score': 0.0,  # Macro indicators don't have sentiment
                        'sentiment_label': 'neutral',
                        'volume': 1,
                        'confidence': 0.9
                    })
            
            # Process options flow
            options_data = signals.get('options_flow', [])
            for option in options_data:
                # Calculate options sentiment based on call/put ratio
                sentiment_score = 0.1 if option['option_type'] == 'call' else -0.1
                
                features.append({
                    'timestamp': option['date'],
                    'symbol': symbol,
                    'signal_type': 'options_flow',
                    'strike': option['strike'],
                    'option_type': option['option_type'],
                    'volume': option['volume'],
                    'premium': option['premium'],
                    'sentiment_score': sentiment_score,
                    'sentiment_label': 'positive' if sentiment_score > 0 else 'negative',
                    'confidence': 0.7
                })
            
            # Convert to DataFrame
            if features:
                df = pd.DataFrame(features)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                return df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error creating signal features for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_aggregated_sentiment(self, symbol: str, days_back: int = 7) -> Dict[str, float]:
        """Get aggregated sentiment scores across all sources.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            Dictionary with aggregated sentiment scores
        """
        try:
            signals = self.get_all_signals(symbol, days_back)
            
            # Aggregate sentiment from all sources
            all_sentiment = []
            
            for sentiment_type in ['news_sentiment', 'twitter_sentiment', 'reddit_sentiment']:
                sentiment_data = signals.get(sentiment_type, [])
                all_sentiment.extend(sentiment_data)
            
            if not all_sentiment:
                return {
                    'overall_sentiment': 0.0,
                    'news_sentiment': 0.0,
                    'social_sentiment': 0.0,
                    'sentiment_volume': 0,
                    'sentiment_confidence': 0.0
                }
            
            # Calculate weighted average sentiment
            total_weight = 0
            weighted_sentiment = 0
            
            for data in all_sentiment:
                weight = data.volume * data.confidence
                weighted_sentiment += data.sentiment_score * weight
                total_weight += weight
            
            overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
            
            # Calculate source-specific sentiment
            news_sentiment = np.mean([d.sentiment_score for d in signals.get('news_sentiment', [])]) if signals.get('news_sentiment') else 0.0
            social_sentiment = np.mean([
                d.sentiment_score for d in signals.get('twitter_sentiment', []) + signals.get('reddit_sentiment', [])
            ]) if (signals.get('twitter_sentiment') or signals.get('reddit_sentiment')) else 0.0
            
            return {
                'overall_sentiment': overall_sentiment,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'sentiment_volume': sum(d.volume for d in all_sentiment),
                'sentiment_confidence': np.mean([d.confidence for d in all_sentiment]) if all_sentiment else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating aggregated sentiment for {symbol}: {e}")
            return {
                'overall_sentiment': 0.0,
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'sentiment_volume': 0,
                'sentiment_confidence': 0.0
            }

def get_external_signals_manager(config: Optional[Dict[str, Any]] = None) -> ExternalSignalsManager:
    """Get the external signals manager instance."""
    return ExternalSignalsManager(config) 