SentimentIngestion - Real-time Sentiment Collection

This module collects real-time sentiment data from Reddit, Twitter, Substack, and other sources.
It creates a centralized sentiment score index for alpha strategy development.


import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
import re
from textblob import TextBlob
import praw
import tweepy
import feedparser
import requests
from bs4port BeautifulSoup

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult, AgentState
from trading.utils.error_handling import log_errors, retry_on_error
from trading.exceptions import StrategyError, ModelError

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
   sentiment data point."  
    source: str  # reddit, twitter, substack, news, etc.
    ticker: str
    text: str
    sentiment_score: float  # -1 to1 confidence: float  # 0 to 1
    timestamp: datetime
    url: Optional[str] = None
    author: Optional[str] = None
    engagement: Optional[int] = None  # likes, upvotes, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     source:self.source,
          ticker": self.ticker,
           text": self.text,
           sentiment_score": self.sentiment_score,
           confidence: self.confidence,
           timestamp": self.timestamp.isoformat(),
           url": self.url,
           author": self.author,
           engagement: self.engagement,
           metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->SentimentData":
    te from dictionary."""
        if isinstance(data["timestamp"], str):
            data["timestamp]= datetime.fromisoformat(data["timestamp])        return cls(**data)


@dataclass
class SentimentIndex:
   centralized sentiment index."  
    ticker: str
    timestamp: datetime
    
    # Aggregated scores
    overall_sentiment: float  # -1 to1  reddit_sentiment: float
    twitter_sentiment: float
    news_sentiment: float
    substack_sentiment: float
    
    # Volume metrics
    total_mentions: int
    reddit_mentions: int
    twitter_mentions: int
    news_mentions: int
    substack_mentions: int
    
    # Confidence metrics
    confidence_score: float
    sentiment_volatility: float
    
    # Trend metrics
    sentiment_change_1h: float
    sentiment_change_24h: float
    sentiment_change_7: float
    
    # Additional metrics
    fear_greed_index: float  # 0 to 100  market_mood: str  # bullish, bearish, neutral
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
   vert to dictionary for serialization.
        return [object Object]     ticker:self.ticker,
           timestamp": self.timestamp.isoformat(),
           overall_sentiment": self.overall_sentiment,
           reddit_sentiment": self.reddit_sentiment,
           twitter_sentiment": self.twitter_sentiment,
           news_sentiment": self.news_sentiment,
           substack_sentiment:self.substack_sentiment,
           total_mentions: self.total_mentions,
           reddit_mentions": self.reddit_mentions,
           twitter_mentions": self.twitter_mentions,
           news_mentions": self.news_mentions,
           substack_mentions": self.substack_mentions,
           confidence_score: self.confidence_score,
           sentiment_volatility:self.sentiment_volatility,
           sentiment_change_1h": self.sentiment_change_1h,
           sentiment_change_24h": self.sentiment_change_24h,
           sentiment_change_7d": self.sentiment_change_7d,
           fear_greed_index: self.fear_greed_index,
           market_mood": self.market_mood,
           created_at:self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SentimentIndex":
    te from dictionary."""
        # Convert datetime strings back to datetime objects
        for field in ["timestamp, _at]:     if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class SentimentIngestion(BaseAgent):ent that collects real-time sentiment data."""
    
    __version__ = 10    __author__ = "SentimentIngestion Team"
    __description__ = "Collects real-time sentiment from multiple sources"
    __tags__ = [sentiment", data_collection", "social_media"]
    __capabilities__ = ["sentiment_collection",data_aggregation", "real_time_processing"]
    __dependencies__ = ["praw, eepy, feedparser, tblob,beautifulsoup4"]   
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.reddit_client = None
        self.twitter_client = None
        self.sentiment_data =
        self.sentiment_index =[object Object]      self.tickers = []
        
    def _setup(self) -> None:
        tup the agent."""
        try:
            # Initialize Reddit client
            reddit_config = self.config.custom_config.get("reddit,[object Object]            if reddit_config:
                self.reddit_client = praw.Reddit(
                    client_id=reddit_config.get("client_id"),
                    client_secret=reddit_config.get(client_secret                   user_agent=reddit_config.get("user_agent,SentimentBot/1.0)                )
            
            # Initialize Twitter client
            twitter_config = self.config.custom_config.get("twitter,[object Object]        if twitter_config:
                self.twitter_client = tweepy.Client(
                    bearer_token=twitter_config.get("bearer_token"),
                    consumer_key=twitter_config.get("consumer_key"),
                    consumer_secret=twitter_config.get("consumer_secret"),
                    access_token=twitter_config.get("access_token"),
                    access_token_secret=twitter_config.get("access_token_secret)                )
            
            # Load tickers to monitor
            self.tickers = self.config.custom_config.get("tickers",
                SPY,QQQ, IWM, EFA, EEM, TLT, GLD, USO", "VNQ",XLE"
            ])
            
            logger.info("SentimentIngestion agent setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup SentimentIngestion agent: {e}")
            raise
    
    @log_errors()
    async def execute(self, **kwargs) -> AgentResult:
     ute sentiment collection."""
        try:
            self.status.state = AgentState.RUNNING
            start_time = datetime.now()
            
            # Collect sentiment from all sources
            sentiment_data = []
            
            # Reddit sentiment
            reddit_sentiment = await self._collect_reddit_sentiment()
            sentiment_data.extend(reddit_sentiment)
            
            # Twitter sentiment
            twitter_sentiment = await self._collect_twitter_sentiment()
            sentiment_data.extend(twitter_sentiment)
            
            # News sentiment
            news_sentiment = await self._collect_news_sentiment()
            sentiment_data.extend(news_sentiment)
            
            # Substack sentiment
            substack_sentiment = await self._collect_substack_sentiment()
            sentiment_data.extend(substack_sentiment)
            
            # Store collected data
            self.sentiment_data.extend(sentiment_data)
            
            # Update sentiment index
            await self._update_sentiment_index(sentiment_data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.status.state = AgentState.SUCCESS
            
            return AgentResult(
                success=True,
                data={
                    sentiment_data": [s.to_dict() for s in sentiment_data],
                    sentiment_index": {ticker: index.to_dict() for ticker, index in self.sentiment_index.items()},
                    collection_metadata": {
                        total_collected": len(sentiment_data),
                        sources": [reddit", twitter",news", "substack"],
                        execution_time: execution_time                   }
                },
                execution_time=execution_time,
                metadata=[object Object]agent":sentiment_ingestion"}
            )
            
        except Exception as e:
            self.status.state = AgentState.ERROR
            return self.handle_error(e)
    
    async def _collect_reddit_sentiment(self) -> List[SentimentData]:
     sentiment from Reddit."""
        try:
            sentiment_data = []
            
            if not self.reddit_client:
                logger.warning(Reddit client not configured)            return sentiment_data
            
            # Subreddits to monitor
            subreddits =
               wallstreetbets", "investing", stocks", "options,
               cryptocurrency", "financialindependence"
            ]
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Get hot posts
                    for post in subreddit.hot(limit=25):
                        # Extract tickers from title and content
                        text = f"[object Object]post.title} {post.selftext}"
                        tickers = self._extract_tickers(text)
                        
                        for ticker in tickers:
                            if ticker in self.tickers:
                                sentiment_score, confidence = self._analyze_sentiment(text)
                                
                                sentiment_data.append(SentimentData(
                                    source="reddit",
                                    ticker=ticker,
                                    text=text[:500],  # Truncate long text
                                    sentiment_score=sentiment_score,
                                    confidence=confidence,
                                    timestamp=datetime.fromtimestamp(post.created_utc),
                                    url=f"https://reddit.com{post.permalink}",
                                    author=post.author.name if post.author else None,
                                    engagement=post.score,
                                    metadata={
                                      subreddit": subreddit_name,
                                        post_id": post.id,
                                       comments_count": post.num_comments
                                    }
                                ))
                
                except Exception as e:
                    logger.error(fFailed to collect from subreddit[object Object]subreddit_name}: {e}")
                    continue
            
            logger.info(f"Collected {len(sentiment_data)} Reddit sentiment data points")
            return sentiment_data
            
        except Exception as e:
            logger.error(fFailed to collect Reddit sentiment: {e}")
            return    
    async def _collect_twitter_sentiment(self) -> List[SentimentData]:
     sentiment from Twitter."""
        try:
            sentiment_data = []
            
            if not self.twitter_client:
                logger.warning("Twitter client not configured)            return sentiment_data
            
            for ticker in self.tickers:
                try:
                    # Search for tweets containing ticker
                    query = f"${ticker} OR #{ticker} -is:retweet"
                    tweets = self.twitter_client.search_recent_tweets(
                        query=query,
                        max_results=100,
                        tweet_fields=["created_at,public_metrics", "author_id"]
                    )
                    
                    if tweets.data:
                        for tweet in tweets.data:
                            sentiment_score, confidence = self._analyze_sentiment(tweet.text)
                            
                            sentiment_data.append(SentimentData(
                                source="twitter",
                                ticker=ticker,
                                text=tweet.text,
                                sentiment_score=sentiment_score,
                                confidence=confidence,
                                timestamp=tweet.created_at,
                                url=f"https://twitter.com/user/status/{tweet.id}",
                                author=tweet.author_id,
                                engagement=tweet.public_metrics.get(like_count                   metadata={
                                   tweet_id": tweet.id,
                                  retweet_count": tweet.public_metrics.get(retweet_count", 0),
                                reply_count": tweet.public_metrics.get(reply_count                   }
                            ))
                
                except Exception as e:
                    logger.error(fFailedto collect Twitter sentiment for {ticker}: {e}")
                    continue
            
            logger.info(f"Collected {len(sentiment_data)} Twitter sentiment data points")
            return sentiment_data
            
        except Exception as e:
            logger.error(fFailedto collect Twitter sentiment: {e}")
            return    
    async def _collect_news_sentiment(self) -> List[SentimentData]:
     sentiment from news sources."""
        try:
            sentiment_data = []
            
            # News sources to monitor
            news_sources =
                https://feeds.finance.yahoo.com/rss/2.0/headline,
               https://www.marketwatch.com/rss/topstories,
               https://feeds.reuters.com/reuters/businessNews"
            ]
            
            for source_url in news_sources:
                try:
                    feed = feedparser.parse(source_url)
                    
                    for entry in feed.entries:20mit to 20 articles
                        # Extract tickers from title and summary
                        text = f"{entry.title} {entry.get('summary', '')}"
                        tickers = self._extract_tickers(text)
                        
                        for ticker in tickers:
                            if ticker in self.tickers:
                                sentiment_score, confidence = self._analyze_sentiment(text)
                                
                                sentiment_data.append(SentimentData(
                                    source="news",
                                    ticker=ticker,
                                    text=text[:500],
                                    sentiment_score=sentiment_score,
                                    confidence=confidence,
                                    timestamp=datetime.now(),  # RSS doesn't always have reliable timestamps
                                    url=entry.get("link"),
                                    author=entry.get("author"),
                                    metadata={
                                   source_url": source_url,
                                        published": entry.get("published")
                                    }
                                ))
                
                except Exception as e:
                    logger.error(fFailed to collect from news source {source_url}: {e}")
                    continue
            
            logger.info(f"Collected {len(sentiment_data)} news sentiment data points")
            return sentiment_data
            
        except Exception as e:
            logger.error(fFailed to collect news sentiment: {e}")
            return    
    async def _collect_substack_sentiment(self) -> List[SentimentData]:
     sentiment from Substack newsletters."""
        try:
            sentiment_data = []
            
            # Substack newsletters to monitor
            substack_sources =
               https://www.zerohedge.com/feed,
             https://www.seekingalpha.com/feed,
             https://www.investing.com/rss/news_301ss"
            ]
            
            for source_url in substack_sources:
                try:
                    feed = feedparser.parse(source_url)
                    
                    for entry in feed.entries:10mit to 10 articles
                        # Extract tickers from title and summary
                        text = f"{entry.title} {entry.get('summary', '')}"
                        tickers = self._extract_tickers(text)
                        
                        for ticker in tickers:
                            if ticker in self.tickers:
                                sentiment_score, confidence = self._analyze_sentiment(text)
                                
                                sentiment_data.append(SentimentData(
                                    source="substack",
                                    ticker=ticker,
                                    text=text[:500],
                                    sentiment_score=sentiment_score,
                                    confidence=confidence,
                                    timestamp=datetime.now(),
                                    url=entry.get("link"),
                                    author=entry.get("author"),
                                    metadata={
                                   source_url": source_url,
                                        published": entry.get("published")
                                    }
                                ))
                
                except Exception as e:
                    logger.error(fFailed to collect from Substack source {source_url}: {e}")
                    continue
            
            logger.info(f"Collected {len(sentiment_data)} Substack sentiment data points")
            return sentiment_data
            
        except Exception as e:
            logger.error(fFailed to collect Substack sentiment: {e}")
            return    
    def _extract_tickers(self, text: str) -> List[str]:
     ticker symbols from text."""
        try:
            # Pattern to match ticker symbols (e.g., $SPY, #AAPL, SPY, AAPL)
            ticker_pattern = r'[\$#]?([A-Z]{1           matches = re.findall(ticker_pattern, text)
            
            # Filter to known tickers
            return [ticker for ticker in matches if ticker in self.tickers]
            
        except Exception as e:
            logger.error(fFailed to extract tickers: {e}")
            return    
    def _analyze_sentiment(self, text: str) -> Tuple[float, float]:
     sentiment score and confidence from text."""
        try:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity  # -1      subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Confidence is inverse of subjectivity (more objective = higher confidence)
            confidence = 1 - subjectivity
            
            return sentiment_score, confidence
            
        except Exception as e:
            logger.error(fFailed to analyze sentiment: {e}")
            return 000.5 async def _update_sentiment_index(self, new_sentiment_data: List[SentimentData]):
     sentiment index with new data."""
        try:
            # Group sentiment data by ticker
            ticker_data = [object Object]          for data in new_sentiment_data:
                if data.ticker not in ticker_data:
                    ticker_data[data.ticker] =               ticker_data[data.ticker].append(data)
            
            # Update index for each ticker
            for ticker, data_list in ticker_data.items():
                sentiment_index = self._calculate_sentiment_index(ticker, data_list)
                if sentiment_index:
                    self.sentiment_index[ticker] = sentiment_index
            
        except Exception as e:
            logger.error(f"Failed to update sentiment index: {e}")
    
    def _calculate_sentiment_index(self, ticker: str, data_list: List[SentimentData]) -> Optional[SentimentIndex]:
     late sentiment index for a ticker."""
        try:
            if not data_list:
                return None
            
            # Group by source
            source_data = [object Object]          for data in data_list:
                if data.source not in source_data:
                    source_data[data.source] =             source_data[data.source].append(data)
            
            # Calculate source-specific sentiment
            reddit_sentiment = np.mean([d.sentiment_score for d in source_data.get("reddit",           twitter_sentiment = np.mean([d.sentiment_score for d in source_data.get("twitter",)
            news_sentiment = np.mean([d.sentiment_score for d in source_data.get("news",
            substack_sentiment = np.mean([d.sentiment_score for d in source_data.get(substack])])
            
            # Calculate overall sentiment (weighted average)
            weights =[object Object]
             reddit3
              twitter3
            news5
               substack: 00.15   }
            
            overall_sentiment = (
                reddit_sentiment * weights["reddit"] +
                twitter_sentiment * weights["twitter"] +
                news_sentiment * weights["news"] +
                substack_sentiment * weights["substack"]
            )
            
            # Calculate volume metrics
            reddit_mentions = len(source_data.get("reddit", []))
            twitter_mentions = len(source_data.get("twitter", []))
            news_mentions = len(source_data.get("news", []))
            substack_mentions = len(source_data.get("substack", []))
            total_mentions = len(data_list)
            
            # Calculate confidence and volatility
            confidence_scores =d.confidence for d in data_list]
            confidence_score = np.mean(confidence_scores) if confidence_scores else0.5      
            sentiment_scores = [d.sentiment_score for d in data_list]
            sentiment_volatility = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
            
            # Calculate trend metrics (simplified - would need historical data)
            sentiment_change_1h = 0.0
            sentiment_change_24h = 0.0
            sentiment_change_7d =0      
            # Calculate fear/greed index
            fear_greed_index = self._calculate_fear_greed_index(
                overall_sentiment, sentiment_volatility, total_mentions
            )
            
            # Determine market mood
            market_mood = self._determine_market_mood(overall_sentiment, fear_greed_index)
            
            return SentimentIndex(
                ticker=ticker,
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                reddit_sentiment=reddit_sentiment,
                twitter_sentiment=twitter_sentiment,
                news_sentiment=news_sentiment,
                substack_sentiment=substack_sentiment,
                total_mentions=total_mentions,
                reddit_mentions=reddit_mentions,
                twitter_mentions=twitter_mentions,
                news_mentions=news_mentions,
                substack_mentions=substack_mentions,
                confidence_score=confidence_score,
                sentiment_volatility=sentiment_volatility,
                sentiment_change_1h=sentiment_change_1h,
                sentiment_change_24h=sentiment_change_24h,
                sentiment_change_7d=sentiment_change_7d,
                fear_greed_index=fear_greed_index,
                market_mood=market_mood
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate sentiment index for {ticker}: {e}")
            return None
    
    def _calculate_fear_greed_index(self, sentiment: float, volatility: float, volume: int) -> float:
     late fear/greed index."""
        try:
            # Convert sentiment from [-1, 100]
            sentiment_score = (sentiment + 1) * 50
            
            # Volatility penalty (higher volatility = more fear)
            volatility_penalty = min(volatility * 100,30      
            # Volume bonus (higher volume = more confidence)
            volume_bonus = min(volume / 100,10      
            fear_greed = sentiment_score - volatility_penalty + volume_bonus
            
            return max(0, min(fear_greed, 10          
        except Exception as e:
            logger.error(f"Failed to calculate fear/greed index: {e}")
            return50  
    def _determine_market_mood(self, sentiment: float, fear_greed: float) -> str:
     ermine market mood based on sentiment and fear/greed."""
        try:
            if sentiment > 02 and fear_greed > 60            return "bullish    elif sentiment < -02 and fear_greed < 40            return "bearish            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Failed to determine market mood: {e}")
            return neutral    
    def get_sentiment_index(self, ticker: str) -> Optional[SentimentIndex]:
     sentiment index for a specific ticker.       return self.sentiment_index.get(ticker)
    
    def get_all_sentiment_indices(self) -> Dict[str, SentimentIndex]:
     all sentiment indices.       return self.sentiment_index.copy()
    
    def get_sentiment_data(self, ticker: str = None, source: str = None) -> List[SentimentData]:
     sentiment data with optional filters.""     filtered_data = self.sentiment_data
        
        if ticker:
            filtered_data = [d for d in filtered_data if d.ticker == ticker]
        
        if source:
            filtered_data = [d for d in filtered_data if d.source == source]
        
        return filtered_data
    
    def validate_input(self, **kwargs) -> bool:
       e input parameters."""
        # No specific input validation needed for this agent
        returntrue 
    def validate_config(self) -> bool:
       gent configuration.""   required_config = [tickers"]
        custom_config = self.config.custom_config or[object Object]        return all(key in custom_config for key in required_config)
    
    def handle_error(self, error: Exception) -> AgentResult:
      ndle errors during execution."""
        self.status.state = AgentState.ERROR
        self.status.current_error = str(error)
        
        return AgentResult(
            success=false
            error_message=str(error),
            error_type=type(error).__name__,
            metadata={"agent": "sentiment_ingestion}        )
    
    def get_capabilities(self) -> List[str]:
  agent capabilities.       return self.__capabilities__
    
    def get_requirements(self) -> Dict[str, Any]:
  agent requirements.
        return {
         dependencies": self.__dependencies__,
           api_keys": [reddit", "twitter"],
            data_sources": [reddit", twitter",news", "substack]        }
    
    def clear_data(self) -> None:
      stored sentiment data.
        self.sentiment_data.clear()
        self.sentiment_index.clear()
