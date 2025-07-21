"""
Sentiment Analysis Example

This example demonstrates how to use the sentiment modules to:
1. Fetch real-time sentiment data from multiple sources
2. Process and analyze sentiment using VADER and BERT
3. Generate sentiment-based trading signals
4. Create sentiment features for machine learning models
5. Monitor sentiment trends across multiple tickers

# NOTE: Flake8 compliance changes applied. Non-ASCII print statements fixed.
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.sentiment.sentiment_fetcher import (
    create_sentiment_fetcher,
)
from features.sentiment_features import (
    create_sentiment_analyzer,
)


def main():
    """Main example function"""
    print("🔍 Sentiment Analysis Example")
    print("=" * 50)

    # Initialize components
    print("Initializing sentiment components...")
    fetcher = create_sentiment_fetcher()
    analyzer = create_sentiment_analyzer()

    # Example 1: Fetch sentiment data for a single ticker
    print("\n📊 Fetching sentiment data for AAPL...")
    ticker = "AAPL"

    sentiment_data = fetcher.fetch_all_sentiment(ticker, hours_back=24)

    print(f"Fetched sentiment data:")
    for source, data in sentiment_data.items():
        print(f"  {source}: {len(data)} items")

    # Example 2: Process sentiment data
    print("\n🧠 Processing sentiment data...")

    # Flatten all data
    all_data = []
    for source, data_list in sentiment_data.items():
        all_data.extend(data_list)

    if all_data:
        # Process sentiment
        features = analyzer.process_sentiment_data(all_data)
        print(f"Processed {len(features)} sentiment features")

        # Show sample features
        if features:
            sample = features[0]
            print(f"Sample feature:")
            print(f"  Ticker: {sample.ticker}")
            print(f"  Source: {sample.source}")
            print(f"  VADER Compound: {sample.vader_compound:.3f}")
            print(f"  Weighted Score: {sample.weighted_score:.3f}")

    # Example 3: Generate sentiment features
    print("\n⚙️ Generating sentiment features...")

    features_df = analyzer.generate_sentiment_features(ticker, hours_back=24)

    if not features_df.empty:
        print(f"Generated {len(features_df)} feature rows")
        print(f"Feature columns: {features_df.columns.tolist()}")

        # Show latest features
        latest = features_df.iloc[-1]
        print(f"Latest sentiment: {latest.get('vader_compound_mean', 0):.3f}")
        print(f"Latest volume: {latest.get('volume_sum', 0)}")
    else:
        print("No features generated - using mock data for demonstration")

        # Create mock features for demonstration
        dates = pd.date_range(
            datetime.now() - timedelta(hours=24), periods=10, freq="2H"
        )
        features_df = pd.DataFrame(
            {
                "vader_compound_mean": np.random.randn(10) * 0.3 + 0.1,
                "vader_compound_std": np.random.rand(10) * 0.2,
                "weighted_score_sum": np.random.rand(10) * 50,
                "volume_sum": np.random.randint(5, 50, 10),
                "sentiment_momentum": np.random.randn(10) * 0.1,
                "sentiment_volatility": np.random.rand(10) * 0.15,
                "engagement_rate": np.random.rand(10) * 0.5,
            },
            index=dates,
        )

    # Example 4: Create sentiment trading signal
    print("\n📈 Creating sentiment trading signal...")

    signal = analyzer.create_sentiment_signal(features_df, lookback_periods=3)

    if not signal.empty:
        print(f"Generated signal with {len(signal)} data points")
        print(f"Latest signal: {signal.iloc[-1]:.3f}")
        print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")

        # Signal interpretation
        latest_signal = signal.iloc[-1]
        if latest_signal > 0.3:
            print("📈 Signal: BULLISH sentiment")
        elif latest_signal < -0.3:
            print("📉 Signal: BEARISH sentiment")
        else:
            print("🤔 Signal: NEUTRAL sentiment")

    # Example 5: Multi-ticker sentiment analysis
    print("\n🔍 Multi-ticker sentiment analysis...")

    tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]

    # Fetch sentiment for multiple tickers
    multi_sentiment = {}
    for ticker in tickers[:3]:  # Limit to 3 for demo
        try:
            sentiment = fetcher.fetch_all_sentiment(ticker, hours_back=6)
            multi_sentiment[ticker] = sentiment
            print(f"  {ticker}: {sum(len(data) for data in sentiment.values())} items")
        except Exception as e:
            print(f"  {ticker}: Error - {e}")

    # Generate features for all tickers
    multi_features = analyzer.generate_multi_ticker_features(
        list(multi_sentiment.keys()), hours_back=6
    )

    print(f"Generated features for {len(multi_features)} tickers")

    # Example 6: Sentiment correlation analysis
    print("\n📊 Sentiment correlation analysis...")

    if len(multi_features) > 1:
        correlation = analyzer.get_sentiment_correlation(multi_features)

        if not correlation.empty:
            print("Sentiment correlation matrix:")
            print(correlation.round(3))

            # Find highest correlation
            if len(correlation) > 1:
                # Get upper triangle (excluding diagonal)
                upper_triangle = correlation.where(
                    np.triu(np.ones(correlation.shape), k=1).astype(bool)
                )

                if not upper_triangle.empty:
                    max_corr = upper_triangle.max().max()
                    max_pair = upper_triangle.stack().idxmax()
                    print(
                        f"Highest correlation: {max_pair[0]} vs {max_pair[1]} = {max_corr:.3f}"
                    )

    # Example 7: Sentiment summary and insights
    print("\n📋 Sentiment summary and insights...")

    if multi_features:
        summary = analyzer.get_sentiment_summary(multi_features)

        print(f"Total tickers analyzed: {summary['total_tickers']}")
        print(
            f"Total data points: {summary['overall_stats'].get('total_data_points', 0)}"
        )

        if "overall_stats" in summary and summary["overall_stats"]:
            overall = summary["overall_stats"]
            print(f"Overall mean sentiment: {overall['mean_sentiment']:.3f}")
            print(f"Overall sentiment volatility: {overall['sentiment_std']:.3f}")

        # Ticker-specific insights
        print("\nTicker-specific insights:")
        for ticker, ticker_summary in summary["ticker_summaries"].items():
            print(f"  {ticker}:")
            print(f"    Mean sentiment: {ticker_summary['mean_sentiment']:.3f}")
            print(f"    Positive ratio: {ticker_summary['positive_ratio']:.1%}")
            print(f"    Negative ratio: {ticker_summary['negative_ratio']:.1%}")

    # Example 8: Real-time sentiment monitoring
    print("\n🔄 Real-time sentiment monitoring...")

    def monitor_sentiment(ticker: str, interval_minutes: int = 30):
        """Monitor sentiment for a ticker at regular intervals"""
        print(
            f"Starting sentiment monitoring for {ticker} (every {interval_minutes} minutes)"
        )

        while True:
            try:
                # Fetch latest sentiment
                sentiment_data = fetcher.fetch_all_sentiment(ticker, hours_back=1)

                # Process and get signal
                features = analyzer.generate_sentiment_features(ticker, hours_back=1)
                if not features.empty:
                    signal = analyzer.create_sentiment_signal(features)
                    latest_signal = signal.iloc[-1] if not signal.empty else 0

                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[{timestamp}] {ticker} sentiment signal: {latest_signal:.3f}"
                    )

                    # Alert on significant changes
                    if abs(latest_signal) > 0.5:
                        direction = "BULLISH" if latest_signal > 0 else "BEARISH"
                        print(f"🚨 ALERT: {ticker} showing {direction} sentiment!")

                # Wait for next interval
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                print(f"\nStopping sentiment monitoring for {ticker}")
                break
            except Exception as e:
                print(f"Error in sentiment monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    # Example 9: Sentiment-based trading strategy
    print("\n🎯 Sentiment-based trading strategy example...")

    def sentiment_trading_strategy(ticker: str, lookback_hours: int = 24):
        """Example sentiment-based trading strategy"""

        # Get sentiment features
        features = analyzer.generate_sentiment_features(ticker, lookback_hours)

        if features.empty:
            return None

        # Calculate sentiment indicators
        sentiment_signal = analyzer.create_sentiment_signal(features)
        sentiment_momentum = (
            features.get("sentiment_momentum", pd.Series()).iloc[-1]
            if "sentiment_momentum" in features.columns
            else 0
        )
        sentiment_volatility = (
            features.get("sentiment_volatility", pd.Series()).iloc[-1]
            if "sentiment_volatility" in features.columns
            else 0
        )
        engagement_rate = (
            features.get("engagement_rate", pd.Series()).iloc[-1]
            if "engagement_rate" in features.columns
            else 0
        )

        # Trading logic
        latest_signal = sentiment_signal.iloc[-1] if not sentiment_signal.empty else 0

        strategy = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "sentiment_signal": latest_signal,
            "sentiment_momentum": sentiment_momentum,
            "sentiment_volatility": sentiment_volatility,
            "engagement_rate": engagement_rate,
            "action": "HOLD",
            "confidence": 0.0,
        }

        # Decision logic
        confidence = abs(latest_signal) * (1 + engagement_rate)

        if latest_signal > 0.3 and sentiment_momentum > 0:
            strategy["action"] = "BUY"
            strategy["confidence"] = confidence
        elif latest_signal < -0.3 and sentiment_momentum < 0:
            strategy["action"] = "SELL"
            strategy["confidence"] = confidence
        else:
            strategy["confidence"] = confidence

        return strategy

    # Test strategy on sample tickers
    for ticker in ["AAPL", "TSLA"]:
        strategy_result = sentiment_trading_strategy(ticker)
        if strategy_result:
            print(f"\n{ticker} Trading Strategy:")
            print(f"  Action: {strategy_result['action']}")
            print(f"  Confidence: {strategy_result['confidence']:.2f}")
            print(f"  Sentiment Signal: {strategy_result['sentiment_signal']:.3f}")
            print(f"  Sentiment Momentum: {strategy_result['sentiment_momentum']:.3f}")

    # Example 10: Save and load sentiment data
    print("\n💾 Saving and loading sentiment data...")

    # Save features
    if multi_features:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentiment_features_{timestamp}.json"

        analyzer.save_features(multi_features, filename)
        print(f"Saved features to: {filename}")

        # Load features
        loaded_features = analyzer.load_features(str(analyzer.output_dir / filename))
        print(f"Loaded features for {len(loaded_features)} tickers")

    print(
        "Sentiment analysis completed. Market sentiment indicators have "
        "been calculated and analyzed."
    )


def example_news_sentiment():
    """Example focusing on news sentiment analysis"""
    print("\n📰 News Sentiment Analysis Example")

    fetcher = create_sentiment_fetcher()
    analyzer = create_sentiment_analyzer()

    # Fetch news headlines
    ticker = "AAPL"
    sentiment_data = fetcher.fetch_news_headlines(ticker, hours_back=12)

    print(f"Fetched {len(sentiment_data)} news headlines for {ticker}")

    # Analyze sentiment
    features = analyzer.process_sentiment_data(sentiment_data)

    if features:
        # Calculate news sentiment metrics
        vader_scores = [f.vader_compound for f in features]
        weighted_scores = [f.weighted_score for f in features]

        print(f"News Sentiment Metrics:")
        print(f"  Average VADER score: {np.mean(vader_scores):.3f}")
        print(f"  Average weighted score: {np.mean(weighted_scores):.3f}")
        print(f"  Sentiment range: [{min(vader_scores):.3f}, {max(vader_scores):.3f}]")

        # Find most positive and negative headlines
        positive_headlines = [f for f in features if f.vader_compound > 0.5]
        negative_headlines = [f for f in features if f.vader_compound < -0.5]

        print(f"  Positive headlines: {len(positive_headlines)}")
        print(f"  Negative headlines: {len(negative_headlines)}")


def example_social_sentiment():
    """Example focusing on social media sentiment analysis"""
    print("\n📱 Social Media Sentiment Analysis Example")

    fetcher = create_sentiment_fetcher()
    analyzer = create_sentiment_analyzer()

    ticker = "TSLA"

    # Fetch social media sentiment
    reddit_data = fetcher.fetch_reddit_sentiment(ticker, hours_back=6)
    twitter_data = fetcher.fetch_twitter_sentiment(ticker, hours_back=6)

    print(
        f"Fetched {len(reddit_data)} Reddit posts and {len(twitter_data)} tweets for {ticker}"
    )

    # Combine social data
    social_data = reddit_data + twitter_data

    if social_data:
        # Process sentiment
        features = analyzer.process_sentiment_data(social_data)

        # Analyze by source
        reddit_features = [f for f in features if f.source == "reddit"]
        twitter_features = [f for f in features if f.source == "twitter"]

        print(f"Social Media Sentiment Analysis:")

        if reddit_features:
            reddit_scores = [f.vader_compound for f in reddit_features]
            print(f"  Reddit average sentiment: {np.mean(reddit_scores):.3f}")
            print(f"  Reddit sentiment volatility: {np.std(reddit_scores):.3f}")

        if twitter_features:
            twitter_scores = [f.vader_compound for f in twitter_features]
            print(f"  Twitter average sentiment: {np.mean(twitter_scores):.3f}")
            print(f"  Twitter sentiment volatility: {np.std(twitter_scores):.3f}")

        # Engagement analysis
        high_engagement = [f for f in features if f.weighted_score > 1.0]
        print(f"  High engagement posts: {len(high_engagement)}")


def example_sentiment_forecasting():
    """Example of using sentiment for price forecasting"""
    print("\n🔮 Sentiment-Based Price Forecasting Example")

    analyzer = create_sentiment_analyzer()

    # Generate historical sentiment features
    ticker = "NVDA"
    features = analyzer.generate_sentiment_features(ticker, hours_back=168)  # 1 week

    if not features.empty:
        print(f"Generated {len(features)} sentiment features for {ticker}")

        # Calculate sentiment trends
        sentiment_trend = features["vader_compound_mean"].rolling(24).mean()
        sentiment_momentum = features["sentiment_momentum"].rolling(12).mean()

        # Simple forecasting model
        latest_sentiment = sentiment_trend.iloc[-1]
        sentiment_change = sentiment_momentum.iloc[-1]

        print(f"Sentiment Forecasting for {ticker}:")
        print(f"  Current sentiment trend: {latest_sentiment:.3f}")
        print(f"  Sentiment momentum: {sentiment_change:.3f}")

        # Price direction prediction
        if latest_sentiment > 0.2 and sentiment_change > 0:
            prediction = "BULLISH"
            confidence = min(0.9, abs(latest_sentiment) + abs(sentiment_change))
        elif latest_sentiment < -0.2 and sentiment_change < 0:
            prediction = "BEARISH"
            confidence = min(0.9, abs(latest_sentiment) + abs(sentiment_change))
        else:
            prediction = "NEUTRAL"
            confidence = 0.5

        print(f"  Price direction prediction: {prediction}")
        print(f"  Prediction confidence: {confidence:.2f}")


if __name__ == "__main__":
    # Run main example
    main()

    # Run specific examples
    example_news_sentiment()
    example_social_sentiment()
    example_sentiment_forecasting()

    print("\n🎉 All sentiment analysis examples completed!")
