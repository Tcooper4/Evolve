"""
Tests for enhanced signal collector and Kubernetes deployment script

Tests async strategy handling, timeout protection, fallback mechanisms,
and deployment script error handling and configuration.
"""

import pytest
import pytest_asyncio
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import subprocess

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trading.data.external_signals import (
    ExternalSignalsManager,
    NewsSentimentCollector,
    TwitterSentimentCollector,
    RedditSentimentCollector,
    MacroIndicatorCollector,
    OptionsFlowCollector,
    async_strategy_wrapper
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAsyncStrategyWrapper:
    """Test the async strategy wrapper functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_strategy_execution(self):
        """Test successful strategy execution with timeout."""
        @async_strategy_wrapper(timeout=5, fallback_value="fallback")
        async def test_strategy():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await test_strategy()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_strategy_timeout(self):
        """Test strategy timeout handling."""
        @async_strategy_wrapper(timeout=0.1, fallback_value="timeout_fallback")
        async def slow_strategy():
            await asyncio.sleep(1.0)
            return "should_not_reach_here"
        
        result = await slow_strategy()
        assert result == "timeout_fallback"
    
    @pytest.mark.asyncio
    async def test_strategy_exception_fallback(self):
        """Test strategy exception handling with fallback."""
        @async_strategy_wrapper(timeout=5, fallback_value="exception_fallback")
        async def failing_strategy():
            raise ValueError("Test exception")
        
        result = await failing_strategy()
        assert result == "exception_fallback"
    
    @pytest.mark.asyncio
    async def test_asyncio_shield_protection(self):
        """Test that asyncio.shield protects against interruption."""
        @async_strategy_wrapper(timeout=5, fallback_value="shield_fallback")
        async def shielded_strategy():
            await asyncio.sleep(0.1)
            return "shielded_success"
        
        # Create a task that could potentially be cancelled
        task = asyncio.create_task(shielded_strategy())
        result = await task
        
        assert result == "shielded_success"

class TestNewsSentimentCollector:
    """Test enhanced news sentiment collector."""
    
    @pytest_asyncio.fixture
    async def collector(self):
        """Create a news sentiment collector for testing."""
        api_keys = {
            "newsapi": "test_newsapi_key",
            "gnews": "test_gnews_key"
        }
        return NewsSentimentCollector(api_keys)
    
    @pytest.mark.asyncio
    async def test_get_news_sentiment_success(self, collector):
        """Test successful news sentiment collection."""
        with patch.object(collector, '_get_newsapi_sentiment') as mock_newsapi:
            mock_newsapi.return_value = [
                Mock(
                    symbol="AAPL",
                    timestamp=datetime.now(),
                    sentiment_score=0.8,
                    sentiment_label="positive",
                    volume=1,
                    source="newsapi",
                    confidence=0.8
                )
            ]
            
            result = await collector.get_news_sentiment("AAPL", days_back=7)
            
            assert len(result) == 1
            assert result[0].symbol == "AAPL"
            assert result[0].sentiment_score == 0.8
    
    @pytest.mark.asyncio
    async def test_get_news_sentiment_timeout(self, collector):
        """Test news sentiment collection timeout."""
        with patch.object(collector, '_get_newsapi_sentiment') as mock_newsapi:
            async def slow_newsapi(*args, **kwargs):
                await asyncio.sleep(2.0)  # Longer than timeout
                return []
            
            mock_newsapi.side_effect = slow_newsapi
            
            # Should return fallback value (empty list) due to timeout
            result = await collector.get_news_sentiment("AAPL", days_back=7)
            assert result == []
    
    @pytest.mark.asyncio
    async def test_get_news_sentiment_exception(self, collector):
        """Test news sentiment collection exception handling."""
        with patch.object(collector, '_get_newsapi_sentiment') as mock_newsapi:
            mock_newsapi.side_effect = Exception("API Error")
            
            # Should return fallback value due to exception
            result = await collector.get_news_sentiment("AAPL", days_back=7)
            assert result == []
    
    @pytest.mark.asyncio
    async def test_analyze_text_sentiment(self, collector):
        """Test text sentiment analysis."""
        # Test positive sentiment
        positive_text = "AAPL stock is bullish and showing strong growth"
        score = await collector._analyze_text_sentiment(positive_text)
        assert score > 0
        
        # Test negative sentiment
        negative_text = "AAPL stock is bearish and showing decline"
        score = await collector._analyze_text_sentiment(negative_text)
        assert score < 0
        
        # Test neutral sentiment
        neutral_text = "AAPL stock price is stable"
        score = await collector._analyze_text_sentiment(neutral_text)
        assert score == 0.0

class TestTwitterSentimentCollector:
    """Test enhanced Twitter sentiment collector."""
    
    @pytest_asyncio.fixture
    async def collector(self):
        """Create a Twitter sentiment collector for testing."""
        return TwitterSentimentCollector()
    
    @pytest.mark.asyncio
    async def test_get_twitter_sentiment_simulation(self, collector):
        """Test Twitter sentiment simulation."""
        result = await collector.get_twitter_sentiment("AAPL", max_tweets=10)
        
        # Should return simulated data
        assert len(result) > 0
        assert all(hasattr(item, 'symbol') for item in result)
        assert all(hasattr(item, 'sentiment_score') for item in result)
    
    @pytest.mark.asyncio
    async def test_analyze_tweet_sentiment(self, collector):
        """Test tweet sentiment analysis."""
        # Test positive tweet
        positive_tweet = "AAPL looking bullish today! ðŸš€ #stocks"
        score = await collector._analyze_tweet_sentiment(positive_tweet)
        assert score > 0
        
        # Test negative tweet
        negative_tweet = "AAPL showing bearish signals ðŸ“‰"
        score = await collector._analyze_tweet_sentiment(negative_tweet)
        assert score < 0

class TestRedditSentimentCollector:
    """Test enhanced Reddit sentiment collector."""
    
    @pytest_asyncio.fixture
    async def collector(self):
        """Create a Reddit sentiment collector for testing."""
        return RedditSentimentCollector("test_client_id", "test_client_secret")
    
    @pytest.mark.asyncio
    async def test_get_reddit_sentiment_simulation(self, collector):
        """Test Reddit sentiment simulation."""
        result = await collector.get_reddit_sentiment("AAPL", max_posts=10)
        
        # Should return simulated data
        assert len(result) > 0
        assert all(hasattr(item, 'symbol') for item in result)
        assert all(hasattr(item, 'sentiment_score') for item in result)

class TestMacroIndicatorCollector:
    """Test enhanced macro indicator collector."""
    
    @pytest_asyncio.fixture
    async def collector(self):
        """Create a macro indicator collector for testing."""
        return MacroIndicatorCollector("test_fred_key")
    
    @pytest.mark.asyncio
    async def test_get_macro_indicators_no_api_key(self):
        """Test macro indicators without API key."""
        collector = MacroIndicatorCollector()
        result = await collector.get_macro_indicators(days_back=30)
        
        # Should return empty dict as fallback
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_get_macro_indicators_with_api_key(self, collector):
        """Test macro indicators with API key."""
        with patch.object(collector, '_get_fred_series') as mock_fred:
            mock_fred.return_value = pd.Series([100, 101, 102], index=pd.date_range('2024-01-01', periods=3))
            
            result = await collector.get_macro_indicators(days_back=30)
            
            # Should return some indicators
            assert len(result) > 0

class TestOptionsFlowCollector:
    """Test enhanced options flow collector."""
    
    @pytest_asyncio.fixture
    async def collector(self):
        """Create an options flow collector for testing."""
        return OptionsFlowCollector({"tradier": "test_tradier_key"})
    
    @pytest.mark.asyncio
    async def test_get_options_flow_simulation(self, collector):
        """Test options flow simulation."""
        result = await collector.get_options_flow("AAPL", days_back=7)
        
        # Should return simulated data
        assert len(result) > 0
        assert all("symbol" in item for item in result)
        assert all("option_type" in item for item in result)

class TestExternalSignalsManager:
    """Test enhanced external signals manager."""
    
    @pytest_asyncio.fixture
    async def manager(self):
        """Create an external signals manager for testing."""
        config = {
            "api_keys": {
                "news": {"newsapi": "test_key", "gnews": "test_key"},
                "fred": "test_fred_key",
                "options": {"tradier": "test_tradier_key"}
            }
        }
        return ExternalSignalsManager(config)
    
    @pytest.mark.asyncio
    async def test_get_all_signals_concurrent(self, manager):
        """Test concurrent signal collection."""
        with patch.object(manager.news_collector, 'get_news_sentiment') as mock_news:
            with patch.object(manager.twitter_collector, 'get_twitter_sentiment') as mock_twitter:
                with patch.object(manager.reddit_collector, 'get_reddit_sentiment') as mock_reddit:
                    with patch.object(manager.macro_collector, 'get_macro_indicators') as mock_macro:
                        with patch.object(manager.options_collector, 'get_options_flow') as mock_options:
                            
                            # Mock successful responses
                            mock_news.return_value = []
                            mock_twitter.return_value = []
                            mock_reddit.return_value = []
                            mock_macro.return_value = {}
                            mock_options.return_value = []
                            
                            result = await manager.get_all_signals("AAPL", days_back=7)
                            
                            # Should return all signal types
                            assert "news_sentiment" in result
                            assert "twitter_sentiment" in result
                            assert "reddit_sentiment" in result
                            assert "macro_indicators" in result
                            assert "options_flow" in result
    
    @pytest.mark.asyncio
    async def test_get_all_signals_with_exceptions(self, manager):
        """Test signal collection with some failures."""
        with patch.object(manager.news_collector, 'get_news_sentiment') as mock_news:
            with patch.object(manager.twitter_collector, 'get_twitter_sentiment') as mock_twitter:
                with patch.object(manager.reddit_collector, 'get_reddit_sentiment') as mock_reddit:
                    with patch.object(manager.macro_collector, 'get_macro_indicators') as mock_macro:
                        with patch.object(manager.options_collector, 'get_options_flow') as mock_options:
                            
                            # Mock some successful and some failed responses
                            mock_news.return_value = []
                            mock_twitter.side_effect = Exception("Twitter API Error")
                            mock_reddit.return_value = []
                            mock_macro.side_effect = Exception("FRED API Error")
                            mock_options.return_value = []
                            
                            result = await manager.get_all_signals("AAPL", days_back=7)
                            
                            # Should still return results for successful collectors
                            assert "news_sentiment" in result
                            assert "twitter_sentiment" in result  # Should be empty list due to exception
                            assert "reddit_sentiment" in result
                            assert "macro_indicators" in result  # Should be empty dict due to exception
                            assert "options_flow" in result
    
    @pytest.mark.asyncio
    async def test_get_signal_features(self, manager):
        """Test signal features generation."""
        with patch.object(manager, 'get_all_signals') as mock_signals:
            # Mock signal data
            mock_signals.return_value = {
                "news_sentiment": [
                    Mock(
                        timestamp=datetime.now(),
                        sentiment_score=0.8,
                        sentiment_label="positive",
                        volume=1,
                        confidence=0.8
                    )
                ],
                "twitter_sentiment": [],
                "reddit_sentiment": [],
                "macro_indicators": {},
                "options_flow": []
            }
            
            result = await manager.get_signal_features("AAPL", days_back=7)
            
            # Should return a DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_get_aggregated_sentiment(self, manager):
        """Test aggregated sentiment calculation."""
        with patch.object(manager, 'get_all_signals') as mock_signals:
            # Mock sentiment data
            mock_signals.return_value = {
                "news_sentiment": [
                    Mock(sentiment_score=0.8, volume=1),
                    Mock(sentiment_score=0.6, volume=2)
                ],
                "twitter_sentiment": [
                    Mock(sentiment_score=-0.2, volume=1)
                ],
                "reddit_sentiment": [],
                "macro_indicators": {},
                "options_flow": []
            }
            
            result = await manager.get_aggregated_sentiment("AAPL", days_back=7)
            
            # Should return aggregated scores
            assert "news_sentiment" in result
            assert "twitter_sentiment" in result
            assert "reddit_sentiment" in result
            assert "overall" in result

class TestKubernetesDeployer:
    """Test Kubernetes deployment script functionality."""
    
    def test_deployer_initialization(self):
        """Test deployer initialization."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        deployer = KubernetesDeployer(
            namespace="test-namespace",
            image="test-image",
            image_tag="v1.0.0",
            registry="test-registry.com"
        )
        
        assert deployer.namespace == "test-namespace"
        assert deployer.image == "test-image"
        assert deployer.image_tag == "v1.0.0"
        assert deployer.registry == "test-registry.com"
        assert deployer.full_image_name == "test-registry.com/test-image:v1.0.0"
    
    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock successful subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.run_command(["kubectl", "version"])
        
        assert result.returncode == 0
        assert result.stdout == "Success output"
    
    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test command execution failure."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock failed subprocess result
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: command failed"
        mock_run.return_value = mock_result
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        
        # Should exit with error code
        with pytest.raises(SystemExit) as exc_info:
            deployer.run_command(["kubectl", "invalid-command"])
        
        assert exc_info.value.code == 1
    
    @patch('subprocess.run')
    def test_run_command_timeout(self, mock_run):
        """Test command execution timeout."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock timeout exception
        mock_run.side_effect = subprocess.TimeoutExpired("kubectl", 30)
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        
        # Should exit with error code
        with pytest.raises(SystemExit) as exc_info:
            deployer.run_command(["kubectl", "version"])
        
        assert exc_info.value.code == 1
    
    @patch('subprocess.run')
    def test_check_prerequisites(self, mock_run):
        """Test prerequisites check."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock successful checks
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "version info"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.check_prerequisites()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_create_namespace(self, mock_run):
        """Test namespace creation."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock namespace doesn't exist, then creation succeeds
        mock_results = [
            Mock(returncode=1, stdout="", stderr="not found"),  # Namespace doesn't exist
            Mock(returncode=0, stdout="namespace created", stderr="")  # Creation succeeds
        ]
        mock_run.side_effect = mock_results
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.create_namespace()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_build_and_push_image(self, mock_run):
        """Test Docker image build and push."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock successful build and push
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.build_and_push_image()
        
        assert result is True
    
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists')
    def test_update_deployment_config(self, mock_exists, mock_open):
        """Test deployment configuration update."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock file exists and content
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "image: automation:latest"
        mock_open.return_value.__enter__.return_value.write = Mock()
        
        deployer = KubernetesDeployer("test-namespace", "test-image", registry="test-registry.com")
        result = deployer.update_deployment_config()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_apply_kubernetes_configs(self, mock_run):
        """Test Kubernetes configuration application."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock successful apply
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "applied"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.apply_kubernetes_configs()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_wait_for_deployment(self, mock_run):
        """Test deployment readiness wait."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock successful rollout status
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "deployment successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.wait_for_deployment()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_check_service_health(self, mock_run):
        """Test service health check."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock successful health check
        mock_results = [
            Mock(returncode=0, stdout="test-service.com", stderr=""),  # Get service URL
            Mock(returncode=0, stdout='{"status": "healthy"}', stderr="")  # Health check
        ]
        mock_run.side_effect = mock_results
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.check_service_health()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_rollback_deployment(self, mock_run):
        """Test deployment rollback."""
        from scripts.deploy_to_kube_batch import KubernetesDeployer
        
        # Mock successful rollback
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "rolled back"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        deployer = KubernetesDeployer("test-namespace", "test-image")
        result = deployer.rollback_deployment()
        
        assert result is True

class TestDeploymentScriptIntegration:
    """Test deployment script integration."""
    
    @patch('scripts.deploy_to_kube_batch.KubernetesDeployer')
    def test_main_function_success(self, mock_deployer_class):
        """Test successful main function execution."""
        from scripts.deploy_to_kube_batch import main
        
        # Mock deployer
        mock_deployer = Mock()
        mock_deployer.deploy.return_value = True
        mock_deployer_class.return_value = mock_deployer
        
        # Mock command line arguments
        with patch('sys.argv', ['deploy_to_kube_batch.py', '--namespace', 'test', '--image', 'test']):
            # Should not raise SystemExit
            main()
    
    @patch('scripts.deploy_to_kube_batch.KubernetesDeployer')
    def test_main_function_failure(self, mock_deployer_class):
        """Test failed main function execution."""
        from scripts.deploy_to_kube_batch import main
        
        # Mock deployer
        mock_deployer = Mock()
        mock_deployer.deploy.return_value = False
        mock_deployer_class.return_value = mock_deployer
        
        # Mock command line arguments
        with patch('sys.argv', ['deploy_to_kube_batch.py', '--namespace', 'test', '--image', 'test']):
            # Should raise SystemExit
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1

def run_performance_benchmark():
    """Run performance benchmark tests."""
    print("Running performance benchmarks...")
    
    # Benchmark async strategy wrapper
    import time
    
    @async_strategy_wrapper(timeout=1, fallback_value="timeout")
    async def benchmark_strategy():
        await asyncio.sleep(0.1)
        return "success"
    
    start_time = time.time()
    
    # Run multiple concurrent strategies
    async def run_benchmark():
        tasks = [benchmark_strategy() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        return results
    
    results = asyncio.run(run_benchmark())
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Async strategy benchmark: {execution_time:.2f}s for 100 concurrent strategies")
    print(f"Success rate: {results.count('success')}/{len(results)}")
    
    return execution_time < 10.0  # Should complete within 10 seconds

if __name__ == "__main__":
    # Run tests
    print("Running signal collector and Kubernetes deployment tests...")
    
    # Run performance benchmark
    benchmark_passed = run_performance_benchmark()
    print(f"Performance benchmark: {'PASSED' if benchmark_passed else 'FAILED'}")
    
    # Run pytest
    import pytest
    pytest.main([__file__, "-v"])
