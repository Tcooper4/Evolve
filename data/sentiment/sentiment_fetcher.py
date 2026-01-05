"""
Multi-Source Sentiment Fetcher

This module fetches real-time sentiment data from multiple sources:
- News headlines from NewsAPI and Finnhub
- Reddit threads from r/stocks and r/wallstreetbets via Pushshift API
- Twitter mentions via Twitter API v2

Provides unified interface for sentiment data collection across platforms.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

# Local imports
from utils.cache_utils import cache_model_operation
from utils.common_helpers import load_config, safe_json_save


@dataclass
class SentimentData:
    """Represents sentiment data from any source"""

    source: str  # 'news', 'reddit', 'twitter'
    ticker: str
    text: str
    timestamp: str
    url: Optional[str] = None
    author: Optional[str] = None
    score: Optional[float] = None  # Reddit score, Twitter likes, etc.
    sentiment_score: Optional[float] = None  # Pre-calculated sentiment
    metadata: Dict[str, Any] = None


class SentimentFetcher:
    """
    Multi-source sentiment data fetcher
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.sentiment_config = self.config.get("sentiment", {})

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # API Keys - support both NEWSAPI_KEY and NEWS_API_KEY
        self.newsapi_key = (
            os.getenv("NEWSAPI_KEY") or 
            os.getenv("NEWS_API_KEY") or 
            self.sentiment_config.get("newsapi_key")
        )
        self.finnhub_key = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_KEY") or self.sentiment_config.get(
            "finnhub_key"
        )
        self.twitter_bearer_token = os.getenv(
            "TWITTER_BEARER_TOKEN"
        ) or self.sentiment_config.get("twitter_bearer_token")

        # Initialize HTTP sessions
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Evolve-SentimentFetcher/1.0"})

        # Rate limiting
        self.rate_limits = {
            "newsapi": {"calls": 0, "limit": 100, "reset_time": time.time() + 86400},
            "finnhub": {"calls": 0, "limit": 60, "reset_time": time.time() + 60},
            "twitter": {"calls": 0, "limit": 300, "reset_time": time.time() + 900},
            "reddit": {"calls": 0, "limit": 100, "reset_time": time.time() + 60},
        }

        # Create data directory
        self.data_dir = Path("data/sentiment/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for different sources
        (self.data_dir / "news").mkdir(exist_ok=True)
        (self.data_dir / "reddit").mkdir(exist_ok=True)
        (self.data_dir / "twitter").mkdir(exist_ok=True)

    def _check_rate_limit(self, source: str) -> bool:
        """Check if we're within rate limits for a source"""
        limit_info = self.rate_limits[source]

        # Reset counter if time has passed
        if time.time() > limit_info["reset_time"]:
            limit_info["calls"] = 0
            if source == "newsapi":
                limit_info["reset_time"] = time.time() + 86400  # 24 hours
            elif source == "finnhub":
                limit_info["reset_time"] = time.time() + 60  # 1 minute
            elif source == "twitter":
                limit_info["reset_time"] = time.time() + 900  # 15 minutes
            elif source == "reddit":
                limit_info["reset_time"] = time.time() + 60  # 1 minute

        if limit_info["calls"] >= limit_info["limit"]:
            self.logger.warning(f"Rate limit reached for {source}")
            return False

        limit_info["calls"] += 1
        return True

    @cache_model_operation(ttl=300)  # Cache for 5 minutes
    def fetch_news_headlines(
        self, ticker: str, hours_back: int = 24
    ) -> List[SentimentData]:
        """
        Fetch news headlines from NewsAPI and Finnhub
        """
        headlines = []

        # Fetch from NewsAPI
        if self.newsapi_key and self._check_rate_limit("newsapi"):
            try:
                newsapi_headlines = self._fetch_newsapi_headlines(ticker, hours_back)
                headlines.extend(newsapi_headlines)
            except Exception as e:
                self.logger.error(f"NewsAPI fetch failed: {e}")

        # Fetch from Finnhub
        if self.finnhub_key and self._check_rate_limit("finnhub"):
            try:
                finnhub_headlines = self._fetch_finnhub_headlines(ticker, hours_back)
                headlines.extend(finnhub_headlines)
            except Exception as e:
                self.logger.error(f"Finnhub fetch failed: {e}")

        return headlines

    def _fetch_newsapi_headlines(
        self, ticker: str, hours_back: int
    ) -> List[SentimentData]:
        """Fetch headlines from NewsAPI"""
        url = "https://newsapi.org/v2/everything"

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)

        params = {
            "q": f'"{ticker}" OR "{ticker} stock" OR "{ticker} shares"',
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.newsapi_key,
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        headlines = []

        for article in data.get("articles", []):
            sentiment_data = SentimentData(
                source="news",
                ticker=ticker,
                text=article.get("title", "") + " " + article.get("description", ""),
                timestamp=article.get("publishedAt", ""),
                url=article.get("url"),
                author=article.get("author"),
                metadata={
                    "source": article.get("source", {}).get("name"),
                    "content": article.get("content"),
                    "url_to_image": article.get("urlToImage"),
                },
            )
            headlines.append(sentiment_data)

        return headlines

    def _fetch_finnhub_headlines(
        self, ticker: str, hours_back: int
    ) -> List[SentimentData]:
        """Fetch headlines from Finnhub"""
        url = "https://finnhub.io/api/v1/company-news"

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)

        params = {
            "symbol": ticker,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": self.finnhub_key,
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        headlines = []

        for article in data:
            sentiment_data = SentimentData(
                source="news",
                ticker=ticker,
                text=article.get("headline", "") + " " + article.get("summary", ""),
                timestamp=article.get("datetime", ""),
                url=article.get("url"),
                author=article.get("author"),
                metadata={
                    "category": article.get("category"),
                    "id": article.get("id"),
                    "related": article.get("related"),
                },
            )
            headlines.append(sentiment_data)

        return headlines

    @cache_model_operation(ttl=600)  # Cache for 10 minutes
    def fetch_reddit_sentiment(
        self, ticker: str, hours_back: int = 24
    ) -> List[SentimentData]:
        """
        Fetch Reddit posts from r/stocks and r/wallstreetbets via Pushshift API
        """
        if not self._check_rate_limit("reddit"):
            return []

        reddit_data = []

        # Subreddits to search
        subreddits = ["stocks", "wallstreetbets", "investing", "StockMarket"]

        for subreddit in subreddits:
            try:
                subreddit_data = self._fetch_pushshift_data(
                    ticker, subreddit, hours_back
                )
                reddit_data.extend(subreddit_data)
            except Exception as e:
                self.logger.error(f"Pushshift fetch failed for r/{subreddit}: {e}")

        return reddit_data

    def _fetch_pushshift_data(
        self, ticker: str, subreddit: str, hours_back: int
    ) -> List[SentimentData]:
        """Fetch data from Pushshift API"""
        url = "https://api.pushshift.io/reddit/search/submission"

        # Calculate timestamp
        end_time = int(time.time())
        start_time = end_time - (hours_back * 3600)

        params = {
            "subreddit": subreddit,
            "q": ticker,
            "after": start_time,
            "before": end_time,
            "size": 100,
            "sort": "score",
            "sort_type": "score",
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        reddit_posts = []

        for post in data.get("data", []):
            # Combine title and selftext
            text = post.get("title", "")
            if post.get("selftext"):
                text += " " + post.get("selftext")

            sentiment_data = SentimentData(
                source="reddit",
                ticker=ticker,
                text=text,
                timestamp=datetime.fromtimestamp(
                    post.get("created_utc", 0)
                ).isoformat(),
                url=f"https://reddit.com{post.get('permalink', '')}",
                author=post.get("author"),
                score=post.get("score"),
                metadata={
                    "subreddit": subreddit,
                    "num_comments": post.get("num_comments"),
                    "upvote_ratio": post.get("upvote_ratio"),
                    "is_self": post.get("is_self"),
                    "domain": post.get("domain"),
                },
            )
            reddit_posts.append(sentiment_data)

        return reddit_posts

    @cache_model_operation(ttl=300)  # Cache for 5 minutes
    def fetch_twitter_sentiment(
        self, ticker: str, hours_back: int = 24
    ) -> List[SentimentData]:
        """
        Fetch Twitter mentions via Twitter API v2
        """
        if not self.twitter_bearer_token or not self._check_rate_limit("twitter"):
            return []

        try:
            return self._fetch_twitter_api_v2(ticker, hours_back)
        except Exception as e:
            self.logger.error(f"Twitter API fetch failed: {e}")
            return []

    def _fetch_twitter_api_v2(
        self, ticker: str, hours_back: int
    ) -> List[SentimentData]:
        """Fetch tweets using Twitter API v2"""
        url = "https://api.twitter.com/2/tweets/search/recent"

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)

        # Search query
        query = f'${ticker} OR "{ticker}" OR "{ticker} stock"'

        headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}

        params = {
            "query": query,
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "max_results": 100,
            "tweet.fields": "created_at,author_id,public_metrics,lang",
            "user.fields": "username,name",
            "expansions": "author_id",
        }

        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        tweets = []

        # Create user lookup
        users = {}
        if "includes" in data and "users" in data["includes"]:
            for user in data["includes"]["users"]:
                users[user["id"]] = user

        for tweet in data.get("data", []):
            user = users.get(tweet.get("author_id", ""), {})

            sentiment_data = SentimentData(
                source="twitter",
                ticker=ticker,
                text=tweet.get("text", ""),
                timestamp=tweet.get("created_at", ""),
                url=f"https://twitter.com/{user.get('username', '')}/status/{tweet.get('id', '')}",
                author=user.get("username"),
                score=tweet.get("public_metrics", {}).get("like_count"),
                metadata={
                    "retweet_count": tweet.get("public_metrics", {}).get(
                        "retweet_count"
                    ),
                    "reply_count": tweet.get("public_metrics", {}).get("reply_count"),
                    "quote_count": tweet.get("public_metrics", {}).get("quote_count"),
                    "lang": tweet.get("lang"),
                    "tweet_id": tweet.get("id"),
                },
            )
            tweets.append(sentiment_data)

        return tweets

    def fetch_all_sentiment(
        self, ticker: str, hours_back: int = 24
    ) -> Dict[str, List[SentimentData]]:
        """
        Fetch sentiment data from all sources
        """
        self.logger.info(
            f"Fetching sentiment data for {ticker} (last {hours_back} hours)"
        )

        results = {}

        # Fetch from each source
        try:
            results["news"] = self.fetch_news_headlines(ticker, hours_back)
            self.logger.info(f"Fetched {len(results['news'])} news headlines")
        except Exception as e:
            self.logger.error(f"News fetch failed: {e}")
            results["news"] = []

        try:
            results["reddit"] = self.fetch_reddit_sentiment(ticker, hours_back)
            self.logger.info(f"Fetched {len(results['reddit'])} Reddit posts")
        except Exception as e:
            self.logger.error(f"Reddit fetch failed: {e}")
            results["reddit"] = []

        try:
            results["twitter"] = self.fetch_twitter_sentiment(ticker, hours_back)
            self.logger.info(f"Fetched {len(results['twitter'])} tweets")
        except Exception as e:
            self.logger.error(f"Twitter fetch failed: {e}")
            results["twitter"] = []

        # Save raw data
        self._save_raw_data(ticker, results)

        return results

    def _save_raw_data(self, ticker: str, data: Dict[str, List[SentimentData]]):
        """Save raw sentiment data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for source, sentiment_list in data.items():
            if sentiment_list:
                # Convert to dict format
                data_dict = [asdict(item) for item in sentiment_list]

                # Save to file
                filename = f"{ticker}_{source}_{timestamp}.json"
                filepath = self.data_dir / source / filename

                safe_json_save(str(filepath), data_dict)

    def get_sentiment_summary(
        self, ticker: str, hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get summary statistics for sentiment data
        """
        data = self.fetch_all_sentiment(ticker, hours_back)

        summary = {
            "ticker": ticker,
            "timeframe": f"{hours_back}h",
            "total_items": sum(len(items) for items in data.values()),
            "by_source": {},
            "recent_activity": {},
        }

        for source, items in data.items():
            summary["by_source"][source] = {
                "count": len(items),
                "avg_score": (
                    np.mean([item.score or 0 for item in items]) if items else 0
                ),
                "recent_count": len(
                    [
                        item
                        for item in items
                        if datetime.fromisoformat(item.timestamp)
                        > datetime.now() - timedelta(hours=1)
                    ]
                ),
            }

        # Recent activity (last hour)
        recent_items = []
        for source, items in data.items():
            for item in items:
                if datetime.fromisoformat(item.timestamp) > datetime.now() - timedelta(
                    hours=1
                ):
                    recent_items.append(item)

        summary["recent_activity"] = {
            "count": len(recent_items),
            "sources": list(set(item.source for item in recent_items)),
        }

        return summary

    async def fetch_sentiment_async(
        self, tickers: List[str], hours_back: int = 24
    ) -> Dict[str, Dict[str, List[SentimentData]]]:
        """
        Fetch sentiment data for multiple tickers asynchronously
        """

        async def fetch_ticker(
            ticker: str,
        ) -> Tuple[str, Dict[str, List[SentimentData]]]:
            return ticker, self.fetch_all_sentiment(ticker, hours_back)

        # Run fetches concurrently
        tasks = [fetch_ticker(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        sentiment_data = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Async fetch failed: {result}")
            else:
                ticker, data = result
                sentiment_data[ticker] = data

        return sentiment_data

    def get_trending_tickers(self, hours_back: int = 24) -> List[str]:
        """
        Get trending tickers based on mention volume
        """
        # This would typically query a database or cache of recent mentions
        # For now, return a sample list
        trending = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META"]
        return trending

    def cleanup_old_data(self, days_to_keep: int = 7):
        """
        Clean up old sentiment data files
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)

        for source_dir in self.data_dir.iterdir():
            if source_dir.is_dir():
                for file_path in source_dir.glob("*.json"):
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink()
                            self.logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to delete {file_path}: {e}")


# Convenience functions
def create_sentiment_fetcher(
    config_path: str = "config/app_config.yaml",
) -> SentimentFetcher:
    """Create a sentiment fetcher instance"""
    return SentimentFetcher(config_path)


def fetch_ticker_sentiment(
    ticker: str, hours_back: int = 24
) -> Dict[str, List[SentimentData]]:
    """Quick function to fetch sentiment for a single ticker"""
    fetcher = SentimentFetcher()
    return fetcher.fetch_all_sentiment(ticker, hours_back)


if __name__ == "__main__":
    # Example usage
    fetcher = SentimentFetcher()

    # Fetch sentiment for a ticker
    ticker = "AAPL"
    sentiment_data = fetcher.fetch_all_sentiment(ticker, hours_back=6)

    print(f"Fetched sentiment data for {ticker}:")
    for source, data in sentiment_data.items():
        print(f"  {source}: {len(data)} items")

    # Get summary
    summary = fetcher.get_sentiment_summary(ticker, hours_back=6)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
