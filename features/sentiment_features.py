"""
Sentiment Features Module

This module processes sentiment data from multiple sources and generates
features for use in trading strategies and ensemble weighting.

Features:
- VADER sentiment scoring
- BERT-based sentiment analysis
- Time-window aggregation
- Ticker-specific sentiment features
- Cross-source sentiment correlation
"""

import json
import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.sentiment.sentiment_fetcher import SentimentData, SentimentFetcher
from utils.common_helpers import load_config, safe_json_save

warnings.filterwarnings("ignore")

# Sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ vaderSentiment not available. Disabling VADER sentiment analysis.")
    print(f"   Missing: {e}")
    SentimentIntensityAnalyzer = None
    VADER_AVAILABLE = False

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    BERT_AVAILABLE = True
except ImportError as e:
    print(
        "âš ï¸ HuggingFace libraries not available. Disabling BERT sentiment analysis."
    )
    print(f"   Missing: {e}")
    torch = None
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    BERT_AVAILABLE = False

# Local imports


@dataclass
class SentimentFeatures:
    """Container for sentiment features"""

    ticker: str
    timestamp: str
    vader_compound: float
    vader_positive: float
    vader_negative: float
    vader_neutral: float
    bert_sentiment: Optional[float] = None
    bert_confidence: Optional[float] = None
    source: str = "aggregated"
    volume: int = 0
    weighted_score: float = 0.0


class SentimentAnalyzer:
    """
    Sentiment analysis and feature generation
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.sentiment_config = self.config.get("sentiment", {})

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize sentiment analyzers
        self.vader_analyzer = None
        self.bert_analyzer = None
        self._initialize_analyzers()

        # Source weights for aggregation
        self.source_weights = {"news": 1.0, "reddit": 0.8, "twitter": 0.6}

        # Time window configurations
        self.time_windows = [1, 6, 24]  # hours

        # Create output directory
        self.output_dir = Path("data/sentiment/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_analyzers(self):
        """Initialize sentiment analysis models"""
        # Initialize VADER
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer initialized")
        else:
            self.logger.warning(
                "VADER not available - sentiment scoring will be limited"
            )

        # Initialize BERT
        if BERT_AVAILABLE:
            try:
                # Use a pre-trained sentiment analysis model
                model_name = self.sentiment_config.get(
                    "bert_model", "cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                self.bert_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                )
                self.logger.info(
                    f"BERT sentiment analyzer initialized with {model_name}"
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize BERT: {e}")
                self.bert_analyzer = None
        else:
            self.logger.warning("BERT not available - using VADER only")

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text using available analyzers
        """
        results = {
            "vader_compound": 0.0,
            "vader_positive": 0.0,
            "vader_negative": 0.0,
            "vader_neutral": 0.0,
            "bert_sentiment": None,
            "bert_confidence": None,
        }

        # Clean text
        cleaned_text = self._clean_text(text)

        # VADER analysis
        if self.vader_analyzer and cleaned_text:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
                results["vader_compound"] = vader_scores["compound"]
                results["vader_positive"] = vader_scores["pos"]
                results["vader_negative"] = vader_scores["neg"]
                results["vader_neutral"] = vader_scores["neu"]
            except Exception as e:
                self.logger.warning(f"VADER analysis failed: {e}")

        # BERT analysis
        if self.bert_analyzer and cleaned_text:
            try:
                bert_result = self.bert_analyzer(cleaned_text[:512])[
                    0
                ]  # Limit text length

                # Convert BERT labels to sentiment scores
                label = bert_result["label"].lower()
                confidence = bert_result["score"]

                if "positive" in label:
                    sentiment_score = confidence
                elif "negative" in label:
                    sentiment_score = -confidence
                else:
                    sentiment_score = 0.0

                results["bert_sentiment"] = sentiment_score
                results["bert_confidence"] = confidence
            except Exception as e:
                self.logger.warning(f"BERT analysis failed: {e}")

        return results

    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove ticker symbols (keep the text but remove $ symbols)
        text = re.sub(r"\$[A-Z]+", "", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\(\)]", "", text)

        return text.strip()

    def process_sentiment_data(
        self, sentiment_data: List[SentimentData]
    ) -> List[SentimentFeatures]:
        """
        Process raw sentiment data into features
        """
        features = []

        for data in sentiment_data:
            # Analyze sentiment
            sentiment_scores = self.analyze_text_sentiment(data.text)

            # Calculate weighted score based on source and engagement
            weight = self.source_weights.get(data.source, 0.5)
            engagement_score = self._calculate_engagement_score(data)
            weighted_score = (
                sentiment_scores["vader_compound"] * weight * engagement_score
            )

            # Create feature object
            feature = SentimentFeatures(
                ticker=data.ticker,
                timestamp=data.timestamp,
                vader_compound=sentiment_scores["vader_compound"],
                vader_positive=sentiment_scores["vader_positive"],
                vader_negative=sentiment_scores["vader_negative"],
                vader_neutral=sentiment_scores["vader_neutral"],
                bert_sentiment=sentiment_scores["bert_sentiment"],
                bert_confidence=sentiment_scores["bert_confidence"],
                source=data.source,
                volume=1,
                weighted_score=weighted_score,
            )

            features.append(feature)

        return features

    def _calculate_engagement_score(self, data: SentimentData) -> float:
        """
        Calculate engagement score based on source-specific metrics
        """
        base_score = 1.0

        if data.source == "reddit":
            # Use Reddit score and comment count
            score = data.score or 0
            comments = data.metadata.get("num_comments", 0) if data.metadata else 0
            upvote_ratio = (
                data.metadata.get("upvote_ratio", 0.5) if data.metadata else 0.5
            )

            engagement = (score * upvote_ratio + comments * 0.1) / 100
            return max(0.1, min(2.0, 1.0 + engagement))

        elif data.source == "twitter":
            # Use Twitter engagement metrics
            likes = data.score or 0
            retweets = data.metadata.get("retweet_count", 0) if data.metadata else 0
            replies = data.metadata.get("reply_count", 0) if data.metadata else 0

            engagement = (likes + retweets * 2 + replies * 3) / 1000
            return max(0.1, min(2.0, 1.0 + engagement))

        elif data.source == "news":
            # News articles get base weight
            return base_score

        return base_score

    def aggregate_sentiment_features(
        self, features: List[SentimentFeatures], time_window: int = 24
    ) -> pd.DataFrame:
        """
        Aggregate sentiment features over time windows
        """
        if not features:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([vars(f) for f in features])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Resample to time windows
        resampled = (
            df.resample(f"{time_window}H")
            .agg(
                {
                    "ticker": "first",
                    "vader_compound": ["mean", "std", "count"],
                    "vader_positive": "mean",
                    "vader_negative": "mean",
                    "vader_neutral": "mean",
                    "bert_sentiment": ["mean", "std"],
                    "bert_confidence": "mean",
                    "weighted_score": ["mean", "std", "sum"],
                    "volume": "sum",
                }
            )
            .fillna(0)
        )

        # Flatten column names
        resampled.columns = ["_".join(col).strip() for col in resampled.columns.values]

        # Add additional features
        resampled["sentiment_momentum"] = resampled["vader_compound_mean"].diff()
        resampled["sentiment_volatility"] = resampled["vader_compound_std"]
        resampled["engagement_rate"] = resampled["weighted_score_sum"] / resampled[
            "volume_sum"
        ].replace(0, 1)

        # Add source-specific features
        for source in ["news", "reddit", "twitter"]:
            source_df = df[df["source"] == source]
            if not source_df.empty:
                source_resampled = (
                    source_df.resample(f"{time_window}H")
                    .agg(
                        {
                            "vader_compound": "mean",
                            "weighted_score": "sum",
                            "volume": "sum",
                        }
                    )
                    .fillna(0)
                )

                resampled[f"{source}_sentiment"] = source_resampled["vader_compound"]
                resampled[f"{source}_volume"] = source_resampled["volume"]
                resampled[f"{source}_weighted_score"] = source_resampled[
                    "weighted_score"
                ]

        return resampled

    def generate_sentiment_features(
        self, ticker: str, hours_back: int = 24
    ) -> pd.DataFrame:
        """
        Generate comprehensive sentiment features for a ticker
        """
        self.logger.info(f"Generating sentiment features for {ticker}")

        # Fetch sentiment data
        fetcher = SentimentFetcher()
        sentiment_data = fetcher.fetch_all_sentiment(ticker, hours_back)

        # Flatten data from all sources
        all_data = []
        for source, data_list in sentiment_data.items():
            all_data.extend(data_list)

        if not all_data:
            self.logger.warning(f"No sentiment data found for {ticker}")
            return pd.DataFrame()

        # Process sentiment data
        features = self.process_sentiment_data(all_data)

        # Generate features for different time windows
        all_features = []

        for window in self.time_windows:
            window_features = self.aggregate_sentiment_features(features, window)
            if not window_features.empty:
                # Add window identifier
                window_features["time_window"] = window
                all_features.append(window_features)

        # Combine all time windows
        if all_features:
            combined_features = pd.concat(all_features, axis=0)
            combined_features["ticker"] = ticker

            # Sort by timestamp
            combined_features.sort_index(inplace=True)

            return combined_features
        else:
            return pd.DataFrame()

    def generate_multi_ticker_features(
        self, tickers: List[str], hours_back: int = 24
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate sentiment features for multiple tickers
        """
        results = {}

        for ticker in tickers:
            try:
                features = self.generate_sentiment_features(ticker, hours_back)
                if not features.empty:
                    results[ticker] = features
                    self.logger.info(
                        f"Generated features for {ticker}: {len(features)} rows"
                    )
                else:
                    self.logger.warning(f"No features generated for {ticker}")
            except Exception as e:
                self.logger.error(f"Failed to generate features for {ticker}: {e}")

        return results

    def create_sentiment_signal(
        self, features_df: pd.DataFrame, lookback_periods: int = 3
    ) -> pd.Series:
        """
        Create trading signal based on sentiment features
        """
        if features_df.empty:
            return pd.Series()

        # Calculate sentiment momentum
        sentiment_momentum = (
            features_df["vader_compound_mean"].rolling(lookback_periods).mean()
        )

        # Calculate sentiment strength
        sentiment_strength = (
            features_df["vader_compound_std"].rolling(lookback_periods).mean()
        )

        # Calculate volume-weighted sentiment
        volume_sentiment = (
            (
                features_df["weighted_score_sum"]
                / features_df["volume_sum"].replace(0, 1)
            )
            .rolling(lookback_periods)
            .mean()
        )

        # Combine signals
        signal = (
            sentiment_momentum * 0.4 + sentiment_strength * 0.3 + volume_sentiment * 0.3
        )

        # Normalize to [-1, 1] range
        signal = signal.clip(-1, 1)

        return signal

    def get_sentiment_correlation(
        self, features_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate correlation between sentiment features across tickers
        """
        if not features_dict:
            return pd.DataFrame()

        # Extract sentiment scores for correlation analysis
        sentiment_scores = {}

        for ticker, features in features_dict.items():
            if "vader_compound_mean" in features.columns:
                sentiment_scores[ticker] = features["vader_compound_mean"]

        if len(sentiment_scores) < 2:
            return pd.DataFrame()

        # Create correlation matrix
        correlation_df = pd.DataFrame(sentiment_scores).corr()

        return correlation_df

    def save_features(
        self, features_dict: Dict[str, pd.DataFrame], filename: Optional[str] = None
    ):
        """
        Save sentiment features to disk
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_features_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert DataFrames to JSON-serializable format
        serializable_data = {}
        for ticker, df in features_dict.items():
            serializable_data[ticker] = {
                "data": df.to_dict("records"),
                "columns": df.columns.tolist(),
                "index": df.index.tolist(),
            }

        safe_json_save(str(filepath), serializable_data)
        self.logger.info(f"Saved sentiment features to {filepath}")

    def load_features(self, filepath: str) -> Dict[str, pd.DataFrame]:
        """
        Load sentiment features from disk
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            features_dict = {}
            for ticker, ticker_data in data.items():
                df = pd.DataFrame(ticker_data["data"])
                df["timestamp"] = pd.to_datetime(ticker_data["index"])
                df.set_index("timestamp", inplace=True)
                features_dict[ticker] = df

            return features_dict
        except Exception as e:
            self.logger.error(f"Failed to load features from {filepath}: {e}")
            return {}

    def get_sentiment_summary(
        self, features_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for sentiment features
        """
        summary = {
            "total_tickers": len(features_dict),
            "ticker_summaries": {},
            "overall_stats": {},
        }

        all_sentiments = []

        for ticker, features in features_dict.items():
            if "vader_compound_mean" in features.columns:
                sentiments = features["vader_compound_mean"].dropna()
                all_sentiments.extend(sentiments.tolist())

                summary["ticker_summaries"][ticker] = {
                    "mean_sentiment": sentiments.mean(),
                    "sentiment_std": sentiments.std(),
                    "data_points": len(sentiments),
                    "positive_ratio": (sentiments > 0.1).mean(),
                    "negative_ratio": (sentiments < -0.1).mean(),
                }

        if all_sentiments:
            summary["overall_stats"] = {
                "mean_sentiment": np.mean(all_sentiments),
                "sentiment_std": np.std(all_sentiments),
                "total_data_points": len(all_sentiments),
            }

        return summary


# Convenience functions
def create_sentiment_analyzer(
    config_path: str = "config/app_config.yaml",
) -> SentimentAnalyzer:
    """Create a sentiment analyzer instance"""
    return SentimentAnalyzer(config_path)


def analyze_ticker_sentiment(ticker: str, hours_back: int = 24) -> pd.DataFrame:
    """Quick function to analyze sentiment for a single ticker"""
    analyzer = SentimentAnalyzer()
    return analyzer.generate_sentiment_features(ticker, hours_back)


def get_sentiment_signal(ticker: str, hours_back: int = 24) -> pd.Series:
    """Get sentiment-based trading signal for a ticker"""
    analyzer = SentimentAnalyzer()
    features = analyzer.generate_sentiment_features(ticker, hours_back)
    return analyzer.create_sentiment_signal(features)


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()

    # Analyze sentiment for a ticker
    ticker = "AAPL"
    features = analyzer.generate_sentiment_features(ticker, hours_back=24)

    if not features.empty:
        print(f"Generated {len(features)} sentiment features for {ticker}")
        print(f"Columns: {features.columns.tolist()}")
        print(f"Latest sentiment: {features['vader_compound_mean'].iloc[-1]:.3f}")

        # Create trading signal
        signal = analyzer.create_sentiment_signal(features)
        print(f"Latest signal: {signal.iloc[-1]:.3f}")

        # Save features
        analyzer.save_features({ticker: features})
    else:
        print(f"No sentiment features generated for {ticker}")

    # Multi-ticker analysis
    tickers = ["AAPL", "TSLA", "NVDA"]
    multi_features = analyzer.generate_multi_ticker_features(tickers, hours_back=24)

    print(f"\nMulti-ticker analysis:")
    for ticker, ticker_features in multi_features.items():
        print(f"  {ticker}: {len(ticker_features)} features")

    # Correlation analysis
    if len(multi_features) > 1:
        correlation = analyzer.get_sentiment_correlation(multi_features)
        print(f"\nSentiment correlation matrix:\n{correlation}")

    # Summary
    summary = analyzer.get_sentiment_summary(multi_features)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
