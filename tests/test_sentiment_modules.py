"""
Tests for Sentiment Modules

Tests both sentiment_fetcher.py and sentiment_features.py modules.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data.sentiment.sentiment_fetcher import (
    SentimentFetcher, 
    SentimentData,
    fetch_ticker_sentiment
)
from features.sentiment_features import (
    SentimentAnalyzer,
    SentimentFeatures,
    analyze_ticker_sentiment,
    get_sentiment_signal
)


class TestSentimentData:
    """Test SentimentData dataclass"""
    
    def test_sentiment_data_creation(self):
        """Test creating a SentimentData instance"""
        data = SentimentData(
            source="news",
            ticker="AAPL",
            text="Apple stock rises on strong earnings",
            timestamp="2024-01-01T12:00:00",
            url="https://example.com",
            author="John Doe",
            score=100,
            sentiment_score=0.8,
            metadata={"category": "earnings"}
        )
        
        assert data.source == "news"
        assert data.ticker == "AAPL"
        assert data.text == "Apple stock rises on strong earnings"
        assert data.score == 100
        assert data.sentiment_score == 0.8


class TestSentimentFetcher:
    """Test SentimentFetcher functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def fetcher(self, temp_dir):
        """Create SentimentFetcher instance for testing"""
        config = {
            'sentiment': {
                'newsapi_key': 'test_key',
                'finnhub_key': 'test_key',
                'twitter_bearer_token': 'test_token'
            }
        }
        
        with patch('data.sentiment.sentiment_fetcher.load_config', return_value=config):
            fetcher = SentimentFetcher()
            fetcher.data_dir = Path(temp_dir) / "data" / "sentiment" / "raw"
            fetcher.data_dir.mkdir(parents=True, exist_ok=True)
            return fetcher
    
    def test_fetcher_initialization(self, fetcher):
        """Test fetcher initialization"""
        assert fetcher.newsapi_key == 'test_key'
        assert fetcher.finnhub_key == 'test_key'
        assert fetcher.twitter_bearer_token == 'test_token'
        assert hasattr(fetcher, 'rate_limits')
        assert hasattr(fetcher, 'session')
    
    def test_check_rate_limit(self, fetcher):
        """Test rate limiting functionality"""
        # Test initial state
        assert fetcher._check_rate_limit('newsapi') == True
        assert fetcher.rate_limits['newsapi']['calls'] == 1
        
        # Test rate limit exceeded
        fetcher.rate_limits['newsapi']['calls'] = 100
        assert fetcher._check_rate_limit('newsapi') == False
    
    @patch('data.sentiment.sentiment_fetcher.requests.Session')
    def test_fetch_newsapi_headlines(self, mock_session, fetcher):
        """Test NewsAPI headline fetching"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'articles': [
                {
                    'title': 'Apple Stock Rises',
                    'description': 'Strong earnings report',
                    'publishedAt': '2024-01-01T12:00:00Z',
                    'url': 'https://example.com',
                    'author': 'John Doe',
                    'source': {'name': 'Reuters'},
                    'content': 'Full article content'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        headlines = fetcher._fetch_newsapi_headlines('AAPL', 24)
        
        assert len(headlines) == 1
        assert headlines[0].source == 'news'
        assert headlines[0].ticker == 'AAPL'
        assert 'Apple Stock Rises' in headlines[0].text
    
    @patch('data.sentiment.sentiment_fetcher.requests.Session')
    def test_fetch_finnhub_headlines(self, mock_session, fetcher):
        """Test Finnhub headline fetching"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                'headline': 'Tesla Stock Update',
                'summary': 'Tesla reports strong sales',
                'datetime': 1704110400,  # Unix timestamp
                'url': 'https://example.com',
                'author': 'Jane Smith',
                'category': 'earnings',
                'id': 12345
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        headlines = fetcher._fetch_finnhub_headlines('TSLA', 24)
        
        assert len(headlines) == 1
        assert headlines[0].source == 'news'
        assert headlines[0].ticker == 'TSLA'
        assert 'Tesla Stock Update' in headlines[0].text
    
    @patch('data.sentiment.sentiment_fetcher.requests.Session')
    def test_fetch_pushshift_data(self, mock_session, fetcher):
        """Test Pushshift Reddit data fetching"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {
                    'title': 'AAPL Discussion',
                    'selftext': 'What do you think about Apple?',
                    'created_utc': 1704110400,
                    'permalink': '/r/stocks/comments/123',
                    'author': 'reddit_user',
                    'score': 50,
                    'num_comments': 25,
                    'upvote_ratio': 0.95,
                    'is_self': True,
                    'domain': 'self.stocks'
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        reddit_data = fetcher._fetch_pushshift_data('AAPL', 'stocks', 24)
        
        assert len(reddit_data) == 1
        assert reddit_data[0].source == 'reddit'
        assert reddit_data[0].ticker == 'AAPL'
        assert 'AAPL Discussion' in reddit_data[0].text
    
    @patch('data.sentiment.sentiment_fetcher.requests.Session')
    def test_fetch_twitter_api_v2(self, mock_session, fetcher):
        """Test Twitter API v2 fetching"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'data': [
                {
                    'id': '123456789',
                    'text': '$AAPL looking strong today!',
                    'created_at': '2024-01-01T12:00:00.000Z',
                    'author_id': 'user123',
                    'public_metrics': {
                        'like_count': 100,
                        'retweet_count': 50,
                        'reply_count': 25,
                        'quote_count': 10
                    },
                    'lang': 'en'
                }
            ],
            'includes': {
                'users': [
                    {
                        'id': 'user123',
                        'username': 'trader_joe',
                        'name': 'Joe Trader'
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        tweets = fetcher._fetch_twitter_api_v2('AAPL', 24)
        
        assert len(tweets) == 1
        assert tweets[0].source == 'twitter'
        assert tweets[0].ticker == 'AAPL'
        assert '$AAPL' in tweets[0].text
    
    def test_fetch_all_sentiment(self, fetcher):
        """Test fetching sentiment from all sources"""
        # Mock individual fetch methods
        with patch.object(fetcher, 'fetch_news_headlines', return_value=[]):
            with patch.object(fetcher, 'fetch_reddit_sentiment', return_value=[]):
                with patch.object(fetcher, 'fetch_twitter_sentiment', return_value=[]):
                    results = fetcher.fetch_all_sentiment('AAPL', 24)
        
        assert 'news' in results
        assert 'reddit' in results
        assert 'twitter' in results
        assert all(len(data) == 0 for data in results.values())
    
    def test_get_sentiment_summary(self, fetcher):
        """Test sentiment summary generation"""
        # Create test data
        test_data = {
            'news': [
                SentimentData(
                    source='news',
                    ticker='AAPL',
                    text='Test news',
                    timestamp='2024-01-01T12:00:00',
                    score=100
                )
            ],
            'reddit': [],
            'twitter': []
        }
        
        with patch.object(fetcher, 'fetch_all_sentiment', return_value=test_data):
            summary = fetcher.get_sentiment_summary('AAPL', 24)
        
        assert summary['ticker'] == 'AAPL'
        assert summary['timeframe'] == '24h'
        assert summary['total_items'] == 1
        assert summary['by_source']['news']['count'] == 1


class TestSentimentFeatures:
    """Test SentimentFeatures dataclass"""
    
    def test_sentiment_features_creation(self):
        """Test creating a SentimentFeatures instance"""
        features = SentimentFeatures(
            ticker="AAPL",
            timestamp="2024-01-01T12:00:00",
            vader_compound=0.5,
            vader_positive=0.6,
            vader_negative=0.1,
            vader_neutral=0.3,
            bert_sentiment=0.4,
            bert_confidence=0.8,
            source="news",
            volume=10,
            weighted_score=0.45
        )
        
        assert features.ticker == "AAPL"
        assert features.vader_compound == 0.5
        assert features.bert_sentiment == 0.4
        assert features.weighted_score == 0.45


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def analyzer(self, temp_dir):
        """Create SentimentAnalyzer instance for testing"""
        config = {
            'sentiment': {
                'bert_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
            }
        }
        
        with patch('features.sentiment_features.load_config', return_value=config):
            analyzer = SentimentAnalyzer()
            analyzer.output_dir = Path(temp_dir) / "output"
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
            return analyzer
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert hasattr(analyzer, 'vader_analyzer')
        assert hasattr(analyzer, 'bert_analyzer')
        assert hasattr(analyzer, 'source_weights')
        assert hasattr(analyzer, 'time_windows')
    
    def test_clean_text(self, analyzer):
        """Test text cleaning functionality"""
        # Test URL removal
        text = "Check out this link: https://example.com and $AAPL stock"
        cleaned = analyzer._clean_text(text)
        assert "https://example.com" not in cleaned
        assert "$AAPL" not in cleaned
        assert "AAPL" in cleaned
        
        # Test whitespace normalization
        text = "  Multiple    spaces   "
        cleaned = analyzer._clean_text(text)
        assert cleaned == "Multiple spaces"
        
        # Test special character removal
        text = "Special chars: @#$%^&*()"
        cleaned = analyzer._clean_text(text)
        assert "@#$%^&*()" not in cleaned
    
    def test_analyze_text_sentiment_vader_only(self, analyzer):
        """Test sentiment analysis with VADER only"""
        # Mock VADER analyzer
        mock_vader = Mock()
        mock_vader.polarity_scores.return_value = {
            'compound': 0.5,
            'pos': 0.6,
            'neg': 0.1,
            'neu': 0.3
        }
        analyzer.vader_analyzer = mock_vader
        analyzer.bert_analyzer = None
        
        text = "This is a positive text about stocks"
        results = analyzer.analyze_text_sentiment(text)
        
        assert results['vader_compound'] == 0.5
        assert results['vader_positive'] == 0.6
        assert results['vader_negative'] == 0.1
        assert results['vader_neutral'] == 0.3
        assert results['bert_sentiment'] is None
    
    def test_analyze_text_sentiment_bert_only(self, analyzer):
        """Test sentiment analysis with BERT only"""
        # Mock BERT analyzer
        mock_bert = Mock()
        mock_bert.return_value = [{'label': 'POSITIVE', 'score': 0.8}]
        analyzer.bert_analyzer = mock_bert
        analyzer.vader_analyzer = None
        
        text = "This is a positive text about stocks"
        results = analyzer.analyze_text_sentiment(text)
        
        assert results['bert_sentiment'] == 0.8
        assert results['bert_confidence'] == 0.8
        assert results['vader_compound'] == 0.0
    
    def test_calculate_engagement_score(self, analyzer):
        """Test engagement score calculation"""
        # Test Reddit engagement
        reddit_data = SentimentData(
            source='reddit',
            ticker='AAPL',
            text='Test post',
            timestamp='2024-01-01T12:00:00',
            score=100,
            metadata={
                'num_comments': 50,
                'upvote_ratio': 0.9
            }
        )
        
        engagement = analyzer._calculate_engagement_score(reddit_data)
        assert engagement > 1.0  # Should be boosted
        
        # Test Twitter engagement
        twitter_data = SentimentData(
            source='twitter',
            ticker='AAPL',
            text='Test tweet',
            timestamp='2024-01-01T12:00:00',
            score=200,  # likes
            metadata={
                'retweet_count': 100,
                'reply_count': 50
            }
        )
        
        engagement = analyzer._calculate_engagement_score(twitter_data)
        assert engagement > 1.0  # Should be boosted
        
        # Test news engagement
        news_data = SentimentData(
            source='news',
            ticker='AAPL',
            text='Test news',
            timestamp='2024-01-01T12:00:00'
        )
        
        engagement = analyzer._calculate_engagement_score(news_data)
        assert engagement == 1.0  # Base score
    
    def test_process_sentiment_data(self, analyzer):
        """Test sentiment data processing"""
        # Mock sentiment analysis
        with patch.object(analyzer, 'analyze_text_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                'vader_compound': 0.5,
                'vader_positive': 0.6,
                'vader_negative': 0.1,
                'vader_neutral': 0.3,
                'bert_sentiment': 0.4,
                'bert_confidence': 0.8
            }
            
            # Test data
            sentiment_data = [
                SentimentData(
                    source='news',
                    ticker='AAPL',
                    text='Positive news about Apple',
                    timestamp='2024-01-01T12:00:00',
                    score=100
                )
            ]
            
            features = analyzer.process_sentiment_data(sentiment_data)
            
            assert len(features) == 1
            assert features[0].ticker == 'AAPL'
            assert features[0].vader_compound == 0.5
            assert features[0].source == 'news'
    
    def test_aggregate_sentiment_features(self, analyzer):
        """Test sentiment feature aggregation"""
        # Create test features
        features = [
            SentimentFeatures(
                ticker='AAPL',
                timestamp='2024-01-01T12:00:00',
                vader_compound=0.5,
                vader_positive=0.6,
                vader_negative=0.1,
                vader_neutral=0.3,
                bert_sentiment=0.4,
                bert_confidence=0.8,
                source='news',
                volume=1,
                weighted_score=0.45
            ),
            SentimentFeatures(
                ticker='AAPL',
                timestamp='2024-01-01T13:00:00',
                vader_compound=0.7,
                vader_positive=0.8,
                vader_negative=0.05,
                vader_neutral=0.15,
                bert_sentiment=0.6,
                bert_confidence=0.9,
                source='reddit',
                volume=1,
                weighted_score=0.65
            )
        ]
        
        aggregated = analyzer.aggregate_sentiment_features(features, time_window=24)
        
        assert not aggregated.empty
        assert 'vader_compound_mean' in aggregated.columns
        assert 'vader_compound_std' in aggregated.columns
        assert 'volume_sum' in aggregated.columns
        assert aggregated['vader_compound_mean'].iloc[0] == 0.6  # Average of 0.5 and 0.7
    
    def test_create_sentiment_signal(self, analyzer):
        """Test sentiment signal creation"""
        # Create test features DataFrame
        dates = pd.date_range('2024-01-01', periods=10, freq='H')
        features_df = pd.DataFrame({
            'vader_compound_mean': np.random.randn(10) * 0.5,
            'vader_compound_std': np.random.rand(10) * 0.3,
            'weighted_score_sum': np.random.rand(10) * 100,
            'volume_sum': np.random.randint(10, 100, 10)
        }, index=dates)
        
        signal = analyzer.create_sentiment_signal(features_df, lookback_periods=3)
        
        assert len(signal) == len(features_df)
        assert all(-1 <= score <= 1 for score in signal.dropna())
    
    def test_get_sentiment_correlation(self, analyzer):
        """Test sentiment correlation calculation"""
        # Create test features for multiple tickers
        dates = pd.date_range('2024-01-01', periods=5, freq='H')
        
        features_dict = {
            'AAPL': pd.DataFrame({
                'vader_compound_mean': [0.1, 0.2, 0.3, 0.4, 0.5]
            }, index=dates),
            'TSLA': pd.DataFrame({
                'vader_compound_mean': [0.2, 0.4, 0.6, 0.8, 1.0]
            }, index=dates)
        }
        
        correlation = analyzer.get_sentiment_correlation(features_dict)
        
        assert not correlation.empty
        assert 'AAPL' in correlation.index
        assert 'TSLA' in correlation.columns
        assert correlation.loc['AAPL', 'TSLA'] > 0  # Should be positively correlated
    
    def test_get_sentiment_summary(self, analyzer):
        """Test sentiment summary generation"""
        # Create test features
        dates = pd.date_range('2024-01-01', periods=10, freq='H')
        features_dict = {
            'AAPL': pd.DataFrame({
                'vader_compound_mean': np.random.randn(10) * 0.5
            }, index=dates)
        }
        
        summary = analyzer.get_sentiment_summary(features_dict)
        
        assert summary['total_tickers'] == 1
        assert 'AAPL' in summary['ticker_summaries']
        assert 'mean_sentiment' in summary['ticker_summaries']['AAPL']
        assert 'overall_stats' in summary


class TestSentimentIntegration:
    """Integration tests for sentiment modules"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_sentiment_pipeline(self, temp_dir):
        """Test complete sentiment analysis pipeline"""
        # Create test sentiment data
        sentiment_data = [
            SentimentData(
                source='news',
                ticker='AAPL',
                text='Apple stock rises on strong earnings report',
                timestamp='2024-01-01T12:00:00',
                score=100
            ),
            SentimentData(
                source='reddit',
                ticker='AAPL',
                text='AAPL looking bullish today!',
                timestamp='2024-01-01T13:00:00',
                score=50,
                metadata={'num_comments': 25, 'upvote_ratio': 0.9}
            )
        ]
        
        # Initialize analyzer
        config = {'sentiment': {}}
        with patch('features.sentiment_features.load_config', return_value=config):
            analyzer = SentimentAnalyzer()
            analyzer.output_dir = Path(temp_dir) / "output"
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock sentiment analysis
        with patch.object(analyzer, 'analyze_text_sentiment') as mock_analyze:
            mock_analyze.return_value = {
                'vader_compound': 0.5,
                'vader_positive': 0.6,
                'vader_negative': 0.1,
                'vader_neutral': 0.3,
                'bert_sentiment': 0.4,
                'bert_confidence': 0.8
            }
            
            # Process data
            features = analyzer.process_sentiment_data(sentiment_data)
            
            # Aggregate features
            aggregated = analyzer.aggregate_sentiment_features(features, time_window=24)
            
            # Create signal
            signal = analyzer.create_sentiment_signal(aggregated)
            
            # Test results
            assert len(features) == 2
            assert not aggregated.empty
            assert len(signal) == len(aggregated)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test fetch_ticker_sentiment
        with patch('data.sentiment.sentiment_fetcher.SentimentFetcher') as mock_fetcher:
            mock_instance = Mock()
            mock_instance.fetch_all_sentiment.return_value = {'news': [], 'reddit': [], 'twitter': []}
            mock_fetcher.return_value = mock_instance
            
            result = fetch_ticker_sentiment('AAPL', 24)
            assert isinstance(result, dict)
            assert 'news' in result
        
        # Test analyze_ticker_sentiment
        with patch('features.sentiment_features.SentimentAnalyzer') as mock_analyzer:
            mock_instance = Mock()
            mock_instance.generate_sentiment_features.return_value = pd.DataFrame()
            mock_analyzer.return_value = mock_instance
            
            result = analyze_ticker_sentiment('AAPL', 24)
            assert isinstance(result, pd.DataFrame)
        
        # Test get_sentiment_signal
        with patch('features.sentiment_features.SentimentAnalyzer') as mock_analyzer:
            mock_instance = Mock()
            mock_instance.generate_sentiment_features.return_value = pd.DataFrame()
            mock_instance.create_sentiment_signal.return_value = pd.Series()
            mock_analyzer.return_value = mock_instance
            
            result = get_sentiment_signal('AAPL', 24)
            assert isinstance(result, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 