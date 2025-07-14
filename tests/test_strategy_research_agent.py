"""
Tests for StrategyResearchAgent

Tests the internet-based strategy discovery and integration functionality.
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

from agents.strategy_research_agent import (
    StrategyResearchAgent, 
    StrategyDiscovery
)
from trading.strategies.base_strategy import BaseStrategy


class TestStrategyDiscovery:
    """Test StrategyDiscovery dataclass"""
    
    def test_strategy_discovery_creation(self):
        """Test creating a StrategyDiscovery instance"""
        discovery = StrategyDiscovery(
            source="arxiv",
            title="Test Strategy",
            description="A test trading strategy",
            authors=["Author 1", "Author 2"],
            url="https://example.com",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=["def test(): pass"],
            parameters={"lookback": 20},
            requirements=["pandas", "numpy"],
            tags=["python", "trading"]
        )
        
        assert discovery.source == "arxiv"
        assert discovery.title == "Test Strategy"
        assert discovery.confidence_score == 0.8
        assert len(discovery.authors) == 2
        assert discovery.strategy_type == "momentum"


class TestStrategyResearchAgent:
    """Test StrategyResearchAgent functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def agent(self, temp_dir):
        """Create StrategyResearchAgent instance for testing"""
        # Create minimal config
        config = {
            'strategy_research': {
                'sources': ['arxiv', 'github'],
                'scan_interval': 24
            }
        }
        
        with patch('agents.strategy_research_agent.load_config', return_value=config):
            agent = StrategyResearchAgent()
            agent.discovered_dir = Path(temp_dir) / "strategies" / "discovered"
            agent.discovered_dir.mkdir(parents=True, exist_ok=True)
            return agent
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.name == "StrategyResearchAgent"
        assert hasattr(agent, 'discovered_strategies')
        assert hasattr(agent, 'scan_history')
        assert hasattr(agent, 'test_results')
    
    def test_extract_strategy_from_text_momentum(self, agent):
        """Test strategy extraction for momentum strategies"""
        text = """
        This is a momentum trading strategy that uses SMA and EMA indicators.
        The strategy generates buy signals when price momentum exceeds threshold.
        def calculate_momentum(prices, lookback_period=20):
            return prices / prices.shift(lookback_period) - 1
        """
        
        result = agent._extract_strategy_from_text(text)
        
        assert result['strategy_type'] == "momentum"
        assert result['confidence_score'] > 0.3
        assert 'lookback_period' in result['parameters']
        assert 'python' in result['tags']
    
    def test_extract_strategy_from_text_mean_reversion(self, agent):
        """Test strategy extraction for mean reversion strategies"""
        text = """
        This is a mean reversion strategy using RSI and Bollinger Bands.
        The strategy generates signals when price deviates from moving average.
        threshold = 0.5
        window = 14
        """
        
        result = agent._extract_strategy_from_text(text)
        
        assert result['strategy_type'] == "mean_reversion"
        assert result['confidence_score'] > 0.3
        assert result['parameters']['threshold'] == 0.5
        assert result['parameters']['window'] == 14.0
    
    def test_extract_strategy_from_text_ml(self, agent):
        """Test strategy extraction for ML strategies"""
        text = """
        This is a machine learning strategy using neural networks.
        The model predicts price movements using deep learning.
        import tensorflow as tf
        import scikit-learn
        """
        
        result = agent._extract_strategy_from_text(text)
        
        assert result['strategy_type'] == "ml"
        assert result['confidence_score'] > 0.3
        assert 'tensorflow' in result['tags']
        assert 'scikit-learn' in result['tags']
    
    def test_similarity_score(self, agent):
        """Test text similarity calculation"""
        text1 = "Momentum trading strategy"
        text2 = "Momentum trading algorithm"
        text3 = "Mean reversion strategy"
        
        score1 = agent._similarity_score(text1, text2)
        score2 = agent._similarity_score(text1, text3)
        
        assert score1 > score2  # More similar texts should have higher score
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1
    
    def test_is_duplicate(self, agent):
        """Test duplicate detection"""
        discovery1 = StrategyDiscovery(
            source="arxiv",
            title="Momentum Trading Strategy",
            description="Test strategy",
            authors=["Author"],
            url="https://example.com/1",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=[],
            parameters={},
            requirements=[],
            tags=[]
        )
        
        discovery2 = StrategyDiscovery(
            source="arxiv",
            title="Momentum Trading Algorithm",  # Similar title
            description="Test strategy",
            authors=["Author"],
            url="https://example.com/2",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=[],
            parameters={},
            requirements=[],
            tags=[]
        )
        
        discovery3 = StrategyDiscovery(
            source="arxiv",
            title="Mean Reversion Strategy",  # Different title
            description="Test strategy",
            authors=["Author"],
            url="https://example.com/3",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="mean_reversion",
            confidence_score=0.8,
            code_snippets=[],
            parameters={},
            requirements=[],
            tags=[]
        )
        
        # Add first discovery
        agent.discovered_strategies.append(discovery1)
        
        # Check duplicates
        assert agent._is_duplicate(discovery2)  # Similar title
        assert not agent._is_duplicate(discovery3)  # Different title
    
    def test_save_discovery(self, agent, temp_dir):
        """Test saving discovery to disk"""
        discovery = StrategyDiscovery(
            source="arxiv",
            title="Test Strategy",
            description="A test strategy",
            authors=["Author"],
            url="https://example.com",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=[],
            parameters={},
            requirements=[],
            tags=[]
        )
        
        filepath = agent.save_discovery(discovery)
        
        assert filepath != ""
        assert Path(filepath).exists()
        
        # Check file content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert data['title'] == "Test Strategy"
        assert data['source'] == "arxiv"
        assert data['confidence_score'] == 0.8
    
    def test_generate_strategy_code_momentum(self, agent):
        """Test strategy code generation for momentum strategy"""
        discovery = StrategyDiscovery(
            source="arxiv",
            title="Momentum Strategy",
            description="A momentum trading strategy",
            authors=["Author"],
            url="https://example.com",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=[],
            parameters={"lookback": 20, "threshold": 0.5},
            requirements=[],
            tags=[]
        )
        
        code = agent.generate_strategy_code(discovery)
        
        assert "class MomentumStrategyStrategy(BaseStrategy)" in code
        assert "def generate_signals" in code
        assert "momentum" in code.lower()
        assert "lookback_period = 20" in code
        assert "threshold = 0.5" in code
    
    def test_generate_strategy_code_mean_reversion(self, agent):
        """Test strategy code generation for mean reversion strategy"""
        discovery = StrategyDiscovery(
            source="arxiv",
            title="Mean Reversion Strategy",
            description="A mean reversion trading strategy",
            authors=["Author"],
            url="https://example.com",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="mean_reversion",
            confidence_score=0.8,
            code_snippets=[],
            parameters={"threshold": 0.5, "window": 14},
            requirements=[],
            tags=[]
        )
        
        code = agent.generate_strategy_code(discovery)
        
        assert "class MeanReversionStrategyStrategy(BaseStrategy)" in code
        assert "mean reversion" in code.lower()
        assert "deviation" in code.lower()
    
    @patch('agents.strategy_research_agent.requests.Session')
    def test_search_arxiv_mock(self, mock_session, agent):
        """Test arXiv search with mocked response"""
        # Mock response
        mock_response = Mock()
        mock_response.content = '''
        <feed>
            <entry>
                <title>Momentum Trading Strategy</title>
                <summary>This paper presents a momentum trading strategy using SMA and EMA indicators.</summary>
                <author><name>Author 1</name></author>
                <id>http://arxiv.org/abs/1234.5678</id>
                <published>2024-01-01T00:00:00Z</published>
            </entry>
        </feed>
        '''
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        discoveries = agent.search_arxiv()
        
        assert len(discoveries) > 0
        assert discoveries[0].title == "Momentum Trading Strategy"
        assert discoveries[0].strategy_type == "momentum"
    
    @patch('agents.strategy_research_agent.requests.Session')
    def test_search_github_mock(self, mock_session, agent):
        """Test GitHub search with mocked response"""
        # Mock GitHub API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'items': [
                {
                    'full_name': 'user/trading-strategy',
                    'description': 'A momentum trading strategy',
                    'html_url': 'https://github.com/user/trading-strategy',
                    'stargazers_count': 100,
                    'language': 'Python',
                    'owner': {'login': 'user'}
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_session.return_value.get.return_value = mock_response
        
        # Mock repository contents
        mock_contents_response = Mock()
        mock_contents_response.json.return_value = [
            {
                'name': 'strategy.py',
                'type': 'file',
                'path': 'strategy.py'
            }
        ]
        mock_contents_response.raise_for_status.return_value = None
        
        # Mock file content
        mock_file_response = Mock()
        mock_file_response.json.return_value = {
            'content': 'ZGVmIHRlc3QoKToKICAgIHBhc3M='  # base64 encoded "def test(): pass"
        }
        mock_file_response.raise_for_status.return_value = None
        
        # Setup session to return different responses
        mock_session.return_value.get.side_effect = [
            mock_response,  # Search results
            mock_contents_response,  # Repository contents
            mock_file_response  # File content
        ]
        
        discoveries = agent.search_github()
        
        assert len(discoveries) > 0
        assert discoveries[0].source == "github"
    
    def test_get_discovery_summary(self, agent):
        """Test discovery summary generation"""
        # Add some test discoveries
        discovery1 = StrategyDiscovery(
            source="arxiv",
            title="Strategy 1",
            description="Test",
            authors=["Author"],
            url="https://example.com",
            discovered_date=datetime.now().isoformat(),
            strategy_type="momentum",
            confidence_score=0.9,
            code_snippets=[],
            parameters={},
            requirements=[],
            tags=[]
        )
        
        discovery2 = StrategyDiscovery(
            source="github",
            title="Strategy 2",
            description="Test",
            authors=["Author"],
            url="https://example.com",
            discovered_date=(datetime.now() - timedelta(days=10)).isoformat(),
            strategy_type="mean_reversion",
            confidence_score=0.3,
            code_snippets=[],
            parameters={},
            requirements=[],
            tags=[]
        )
        
        agent.discovered_strategies = [discovery1, discovery2]
        
        summary = agent.get_discovery_summary()
        
        assert summary['total_discoveries'] == 2
        assert summary['by_source']['arxiv'] == 1
        assert summary['by_source']['github'] == 1
        assert summary['by_type']['momentum'] == 1
        assert summary['by_type']['mean_reversion'] == 1
        assert summary['confidence_distribution']['high'] == 1
        assert summary['confidence_distribution']['low'] == 1
    
    @patch('agents.strategy_research_agent.BacktestEngine')
    def test_test_discovered_strategy(self, mock_backtester, agent):
        """Test strategy testing functionality"""
        discovery = StrategyDiscovery(
            source="arxiv",
            title="Test Strategy",
            description="A test strategy",
            authors=["Author"],
            url="https://example.com",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=[],
            parameters={"lookback": 20},
            requirements=[],
            tags=[]
        )
        
        # Mock backtest results
        mock_backtester.return_value.run_backtest.return_value = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05
        }
        
        results = agent.test_discovered_strategy(discovery)
        
        assert 'total_return' in results
        assert results['total_return'] == 0.15
        assert results['sharpe_ratio'] == 1.2
    
    def test_run_method(self, agent):
        """Test main agent run method"""
        # Mock search methods to return test discoveries
        test_discovery = StrategyDiscovery(
            source="arxiv",
            title="Test Strategy",
            description="A test strategy",
            authors=["Author"],
            url="https://example.com",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=[],
            parameters={},
            requirements=[],
            tags=[]
        )
        
        with patch.object(agent, 'search_arxiv', return_value=[test_discovery]):
            with patch.object(agent, 'search_ssrn', return_value=[]):
                with patch.object(agent, 'search_github', return_value=[]):
                    with patch.object(agent, 'search_quantconnect', return_value=[]):
                        with patch.object(agent, 'test_discovered_strategy', return_value={'total_return': 0.1}):
                            results = agent.run()
        
        assert results['status'] == 'success'
        assert results['discoveries_found'] == 1
        assert results['strategies_tested'] == 1


class TestStrategyResearchAgentIntegration:
    """Integration tests for StrategyResearchAgent"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def agent(self, temp_dir):
        """Create agent for integration testing"""
        config = {
            'strategy_research': {
                'sources': ['arxiv', 'github'],
                'scan_interval': 24
            }
        }
        
        with patch('agents.strategy_research_agent.load_config', return_value=config):
            agent = StrategyResearchAgent()
            agent.discovered_dir = Path(temp_dir) / "strategies" / "discovered"
            agent.discovered_dir.mkdir(parents=True, exist_ok=True)
            return agent
    
    def test_full_discovery_workflow(self, agent):
        """Test complete discovery workflow"""
        # Create test discovery
        discovery = StrategyDiscovery(
            source="arxiv",
            title="Integration Test Strategy",
            description="A strategy for integration testing",
            authors=["Test Author"],
            url="https://example.com",
            discovered_date=datetime.now().isoformat(),
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=["def test_strategy(): return 'buy'"],
            parameters={"lookback": 20, "threshold": 0.5},
            requirements=["pandas", "numpy"],
            tags=["python", "trading"]
        )
        
        # Test saving
        filepath = agent.save_discovery(discovery)
        assert Path(filepath).exists()
        
        # Test loading
        agent._load_existing_discoveries()
        assert len(agent.discovered_strategies) > 0
        
        # Test code generation
        code = agent.generate_strategy_code(discovery)
        assert "class IntegrationTestStrategyStrategy" in code
        assert "def generate_signals" in code
        
        # Test summary generation
        summary = agent.get_discovery_summary()
        assert summary['total_discoveries'] > 0
        assert 'arxiv' in summary['by_source']
    
    def test_strategy_code_execution(self, agent):
        """Test that generated strategy code can be executed"""
        discovery = StrategyDiscovery(
            source="arxiv",
            title="Executable Strategy",
            description="A strategy that can be executed",
            authors=["Author"],
            url="https://example.com",
            discovered_date="2024-01-01T00:00:00",
            strategy_type="momentum",
            confidence_score=0.8,
            code_snippets=[],
            parameters={"lookback": 20},
            requirements=[],
            tags=[]
        )
        
        # Generate code
        code = agent.generate_strategy_code(discovery)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_file.write(code)
        temp_file.close()
        
        try:
            # Import the generated module
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_strategy", temp_file.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find strategy class
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseStrategy) and 
                    attr != BaseStrategy):
                    strategy_class = attr
                    break
            
            assert strategy_class is not None
            
            # Test strategy instantiation
            strategy = strategy_class()
            assert isinstance(strategy, BaseStrategy)
            
            # Test with sample data
            df = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100,
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 102,
                'low': np.random.randn(100).cumsum() + 98,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            signals = strategy.generate_signals(df)
            assert 'signal' in signals.columns
            assert len(signals) == len(df)
            
        finally:
            # Clean up
            import os
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 