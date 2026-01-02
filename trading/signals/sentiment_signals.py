"""
Sentiment Signals Module

Generates trading signals based on sentiment analysis from Reddit (PRAW) and news headlines (NewsAPI).
Uses Vader and TextBlob for polarity scoring and integrates with the trading pipeline.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import sentiment analysis libraries with fallback handling
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError as e:
    print("⚠️ vaderSentiment not available. Disabling VADER sentiment analysis.")
    print(f"   Missing: {e}")
    SentimentIntensityAnalyzer = None
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

# Import data collection libraries
try:
    import praw

    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logging.warning("PRAW not available. Install with: pip install praw")

try:
    from newsapi import NewsApiClient

    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    logging.warning("NewsAPI not available. Install with: pip install newsapi-python")

from trading.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class SentimentSignals:
    """Sentiment-based trading signals generator."""

    def __init__(
        self,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: str = "EvolveTradingBot/1.0",
        newsapi_key: Optional[str] = None,
    ):
        """Initialize the sentiment signals generator.

        Args:
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit user agent string
            newsapi_key: NewsAPI key
        """
        self.reddit_client_id = reddit_client_id or os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = reddit_client_secret or os.getenv(
            "REDDIT_CLIENT_SECRET"
        )
        self.reddit_user_agent = reddit_user_agent
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")

        self.cache_dir = Path("cache/sentiment_signals")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sentiment analyzers
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()

        # Initialize data clients
        self.reddit_client = None
        self.newsapi_client = None

        self._initialize_clients()

        # Define subreddits for financial sentiment
        self.financial_subreddits = [
            "wallstreetbets",
            "investing",
            "stocks",
            "options",
            "stockmarket",
            "financialindependence",
            "personalfinance",
        ]

        # Define news sources
        self.news_sources = [
            "reuters",
            "bloomberg",
            "cnbc",
            "marketwatch",
            "yahoo-finance",
            "seeking-alpha",
            "financial-times",
        ]

        logger.info("Sentiment signals generator initialized")

    def _initialize_clients(self):
        """Initialize Reddit and NewsAPI clients."""
        try:
            # Initialize Reddit client
            if PRAW_AVAILABLE and self.reddit_client_id and self.reddit_client_secret:
                self.reddit_client = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent,
                )
                logger.info("Reddit client initialized successfully")
            else:
                logger.warning(
                    "Reddit client not available - missing credentials or PRAW"
                )

            # Initialize NewsAPI client
            if NEWSAPI_AVAILABLE and self.newsapi_key:
                self.newsapi_client = NewsApiClient(api_key=self.newsapi_key)
                logger.info("NewsAPI client initialized successfully")
            else:
                logger.warning(
                    "NewsAPI client not available - missing API key or NewsAPI"
                )

        except Exception as e:
            logger.error(f"Error initializing clients: {e}")

    def get_reddit_sentiment(
        self, ticker: str, hours: int = 24, limit: int = 100
    ) -> Dict[str, Any]:
        """Get sentiment from Reddit posts and comments.

        Args:
            ticker: Stock ticker symbol
            hours: Hours to look back
            limit: Maximum number of posts to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.reddit_client:
            logger.warning("Reddit client not available")
            return self._get_cached_sentiment("reddit", ticker)

        try:
            all_posts = []
            all_comments = []

            # Search across financial subreddits
            for subreddit_name in self.financial_subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)

                    # Search for posts containing the ticker
                    search_query = f"{ticker}"
                    posts = subreddit.search(
                        search_query,
                        time_filter="day",
                        limit=limit // len(self.financial_subreddits),
                    )

                    for post in posts:
                        # Get post data
                        post_data = {
                            "title": post.title,
                            "body": post.selftext,
                            "score": post.score,
                            "created_utc": post.created_utc,
                            "subreddit": subreddit_name,
                            "type": "post",
                        }
                        all_posts.append(post_data)

                        # Get comments
                        post.comments.replace_more(
                            limit=0
                        )  # Remove MoreComments objects
                        for comment in post.comments.list():
                            comment_data = {
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": comment.created_utc,
                                "subreddit": subreddit_name,
                                "type": "comment",
                            }
                            all_comments.append(comment_data)

                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                    continue

            # Analyze sentiment
            sentiment_results = self._analyze_text_sentiment(all_posts + all_comments)

            # Add metadata
            sentiment_results.update(
                {
                    "ticker": ticker,
                    "source": "reddit",
                    "timestamp": datetime.now().isoformat(),
                    "posts_analyzed": len(all_posts),
                    "comments_analyzed": len(all_comments),
                    "subreddits_searched": self.financial_subreddits,
                }
            )

            # Cache results
            self._cache_sentiment("reddit", ticker, sentiment_results)

            logger.info(
                f"Reddit sentiment analysis completed for {ticker}: {len(all_posts)} posts, {len(all_comments)} comments"
            )
            return sentiment_results

        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {ticker}: {e}")
            return self._get_cached_sentiment("reddit", ticker)

    def get_news_sentiment(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Get sentiment from news headlines and articles.

        Args:
            ticker: Stock ticker symbol
            days: Days to look back

        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.newsapi_client:
            logger.warning("NewsAPI client not available")
            return self._get_cached_sentiment("news", ticker)

        try:
            # Get news articles
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            articles = []

            # Search for articles about the ticker
            try:
                response = self.newsapi_client.get_everything(
                    q=ticker,
                    from_param=from_date,
                    language="en",
                    sort_by="relevancy",
                    page_size=100,
                )

                for article in response.get("articles", []):
                    article_data = {
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "published_at": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "url": article.get("url", ""),
                    }
                    articles.append(article_data)

            except Exception as e:
                logger.warning(f"Error fetching news articles: {e}")

            # Analyze sentiment
            sentiment_results = self._analyze_text_sentiment(articles)

            # Add metadata
            sentiment_results.update(
                {
                    "ticker": ticker,
                    "source": "news",
                    "timestamp": datetime.now().isoformat(),
                    "articles_analyzed": len(articles),
                    "date_range": f"{from_date} to {datetime.now().strftime('%Y-%m-%d')}",
                }
            )

            # Cache results
            self._cache_sentiment("news", ticker, sentiment_results)

            logger.info(
                f"News sentiment analysis completed for {ticker}: {len(articles)} articles"
            )
            return sentiment_results

        except Exception as e:
            logger.error(f"Error getting news sentiment for {ticker}: {e}")
            return self._get_cached_sentiment("news", ticker)

    def _analyze_text_sentiment(self, texts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of text data using multiple methods."""
        try:
            vader_scores = []
            textblob_scores = []

            for item in texts:
                # Combine title and body/content
                text = ""
                if "title" in item:
                    text += item["title"] + " "
                if "body" in item:
                    text += item["body"] + " "
                if "description" in item:
                    text += item["description"] + " "
                if "content" in item:
                    text += item["content"] + " "

                text = text.strip()
                if not text:
                    continue

                # Vader sentiment analysis
                if self.vader_analyzer:
                    vader_score = self.vader_analyzer.polarity_scores(text)
                    vader_scores.append(vader_score)

                # TextBlob sentiment analysis
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    textblob_scores.append(
                        {
                            "polarity": blob.sentiment.polarity,
                            "subjectivity": blob.sentiment.subjectivity,
                        }
                    )

            # Calculate aggregate scores
            results = {
                "total_texts": len(texts),
                "analyzed_texts": (
                    len(vader_scores) if vader_scores else len(textblob_scores)
                ),
            }

            # Vader results
            if vader_scores:
                avg_vader = {
                    "compound": np.mean([s["compound"] for s in vader_scores]),
                    "positive": np.mean([s["pos"] for s in vader_scores]),
                    "negative": np.mean([s["neg"] for s in vader_scores]),
                    "neutral": np.mean([s["neu"] for s in vader_scores]),
                }
                results["vader_sentiment"] = avg_vader

                # Sentiment classification
                if avg_vader["compound"] >= 0.05:
                    results["vader_classification"] = "positive"
                elif avg_vader["compound"] <= -0.05:
                    results["vader_classification"] = "negative"
                else:
                    results["vader_classification"] = "neutral"

            # TextBlob results
            if textblob_scores:
                avg_textblob = {
                    "polarity": np.mean([s["polarity"] for s in textblob_scores]),
                    "subjectivity": np.mean(
                        [s["subjectivity"] for s in textblob_scores]
                    ),
                }
                results["textblob_sentiment"] = avg_textblob

                # Sentiment classification
                if avg_textblob["polarity"] > 0.1:
                    results["textblob_classification"] = "positive"
                elif avg_textblob["polarity"] < -0.1:
                    results["textblob_classification"] = "negative"
                else:
                    results["textblob_classification"] = "neutral"

            # Combined sentiment score
            if "vader_sentiment" in results and "textblob_sentiment" in results:
                combined_score = (
                    results["vader_sentiment"]["compound"]
                    + results["textblob_sentiment"]["polarity"]
                ) / 2
                results["combined_sentiment_score"] = combined_score

                if combined_score > 0.1:
                    results["overall_classification"] = "positive"
                elif combined_score < -0.1:
                    results["overall_classification"] = "negative"
                else:
                    results["overall_classification"] = "neutral"

            return results

        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {"error": str(e)}

    def generate_sentiment_signals(
        self, ticker: str, include_reddit: bool = True, include_news: bool = True
    ) -> Dict[str, Any]:
        """Generate trading signals based on sentiment analysis.

        Args:
            ticker: Stock ticker symbol
            include_reddit: Whether to include Reddit sentiment
            include_news: Whether to include news sentiment

        Returns:
            Dictionary with sentiment signals
        """
        try:
            signals = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "signals": {},
                "confidence": 0.0,
            }

            reddit_sentiment = None
            news_sentiment = None

            # Get Reddit sentiment
            if include_reddit:
                reddit_sentiment = self.get_reddit_sentiment(ticker)
                if "error" not in reddit_sentiment:
                    signals["reddit_sentiment"] = reddit_sentiment

            # Get news sentiment
            if include_news:
                news_sentiment = self.get_news_sentiment(ticker)
                if "error" not in news_sentiment:
                    signals["news_sentiment"] = news_sentiment

            # Generate trading signals
            sentiment_signals = self._create_trading_signals(
                reddit_sentiment, news_sentiment
            )
            signals["signals"] = sentiment_signals

            # Calculate overall confidence
            confidence_factors = []
            if reddit_sentiment and "error" not in reddit_sentiment:
                confidence_factors.append(
                    reddit_sentiment.get("analyzed_texts", 0) / 100
                )  # Normalize
            if news_sentiment and "error" not in news_sentiment:
                confidence_factors.append(
                    news_sentiment.get("analyzed_texts", 0) / 50
                )  # Normalize

            # Safely calculate confidence with division-by-zero protection
            if len(confidence_factors) > 0:
                signals["confidence"] = min(1.0, sum(confidence_factors) / len(confidence_factors))
            else:
                signals["confidence"] = 0.0

            return signals

        except Exception as e:
            logger.error(f"Error generating sentiment signals for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}

    def _create_trading_signals(
        self, reddit_sentiment: Optional[Dict], news_sentiment: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create trading signals from sentiment data."""
        try:
            signals = {
                "buy_signal": False,
                "sell_signal": False,
                "hold_signal": False,
                "signal_strength": 0.0,
                "reasoning": [],
            }

            sentiment_scores = []

            # Process Reddit sentiment
            if reddit_sentiment and "error" not in reddit_sentiment:
                if "combined_sentiment_score" in reddit_sentiment:
                    score = reddit_sentiment["combined_sentiment_score"]
                    sentiment_scores.append(score)
                    signals["reasoning"].append(f"Reddit sentiment: {score:.3f}")

                if "overall_classification" in reddit_sentiment:
                    classification = reddit_sentiment["overall_classification"]
                    signals["reasoning"].append(
                        f"Reddit classification: {classification}"
                    )

            # Process news sentiment
            if news_sentiment and "error" not in news_sentiment:
                if "combined_sentiment_score" in news_sentiment:
                    score = news_sentiment["combined_sentiment_score"]
                    sentiment_scores.append(score)
                    signals["reasoning"].append(f"News sentiment: {score:.3f}")

                if "overall_classification" in news_sentiment:
                    classification = news_sentiment["overall_classification"]
                    signals["reasoning"].append(
                        f"News classification: {classification}"
                    )

            # Generate signals based on average sentiment
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                signals["signal_strength"] = abs(avg_sentiment)

                if avg_sentiment > 0.2:
                    signals["buy_signal"] = True
                    signals["reasoning"].append("Strong positive sentiment")
                elif avg_sentiment < -0.2:
                    signals["sell_signal"] = True
                    signals["reasoning"].append("Strong negative sentiment")
                else:
                    signals["hold_signal"] = True
                    signals["reasoning"].append("Neutral sentiment")

            return signals

        except Exception as e:
            logger.error(f"Error creating trading signals: {e}")
            return {"error": str(e)}

    def _cache_sentiment(self, source: str, ticker: str, data: Dict[str, Any]):
        """Cache sentiment data."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{source}_{ticker}_{timestamp}.json"
            filepath = self.cache_dir / filename

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to cache sentiment data: {e}")

    def _get_cached_sentiment(self, source: str, ticker: str) -> Dict[str, Any]:
        """Get cached sentiment data."""
        try:
            pattern = f"{source}_{ticker}_*.json"
            files = list(self.cache_dir.glob(pattern))

            if files:
                # Get most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)

                # Check if cache is recent (within 1 hour)
                if (
                    datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
                ).seconds < 3600:
                    with open(latest_file, "r") as f:
                        return json.load(f)

        except Exception as e:
            logger.warning(f"Failed to load cached sentiment: {e}")

        return {"error": "No cached data available"}


# Global sentiment signals instance
_sentiment_signals = None


def get_sentiment_signals() -> SentimentSignals:
    """Get the global sentiment signals instance."""
    global _sentiment_signals
    if _sentiment_signals is None:
        _sentiment_signals = SentimentSignals()
    return _sentiment_signals
