# Tests for Autonomous Alpha Agent System

# This module tests the complete alpha strategy generation and management system.
import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

from trading.agents.alpha import (
    AlphaGenAgent, Hypothesis,
    SignalTester, TestResult, TestConfig,
    RiskValidator, ValidationResult, ValidationConfig,
    SentimentIngestion, SentimentData, SentimentIndex,
    AlphaRegistry, StrategyRecord, DecayAnalysis,
    AlphaOrchestrator, OrchestrationCycle, DecisionLog
)
from trading.agents.base_agent_interface import AgentConfig, AgentResult, AgentState

logger = logging.getLogger(__name__)


class TestHypothesis:
    """Hypothesis data class."""
    
    def test_hypothesis_creation(self):
        """Test hypothesis creation."""
        hypothesis = Hypothesis(
            id="test_hyp_001",
            title="Test Strategy",
            description="A test trading strategy",
            strategy_type="momentum",
            asset_class="equity",
            timeframe="1d",
            entry_conditions=["RSI <30"],
            exit_conditions=["RSI >70"],
            risk_parameters={"stop_loss": 0.02},
            confidence_score=0.8,
            reasoning="Test reasoning",
            data_sources=["price_data"]
        )
        
        assert hypothesis.id == "test_hyp_001"
        assert hypothesis.title == "Test Strategy"
        assert hypothesis.strategy_type == "momentum"
        assert hypothesis.confidence_score == 0.8
    
    def test_hypothesis_serialization(self):
        """Test hypothesis serialization."""
        hypothesis = Hypothesis(
            id="test_hyp_002",
            title="Test Strategy 2",
            description="Another test strategy",
            strategy_type="mean_reversion",
            asset_class="crypto",
            timeframe="4h",
            entry_conditions=["Price > MA20"],
            exit_conditions=["Price < MA20"],
            risk_parameters={"take_profit": 0.05},
            confidence_score=0.7,
            reasoning="Test reasoning 2",
            data_sources=["price_data", "ume_data"]
        )
        
        # Test to_dict
        data = hypothesis.to_dict()
        assert data["id"] == "test_hyp_002"
        assert data["strategy_type"] == "mean_reversion"
        assert data["confidence_score"] == 0.7        
        # Test from_dict
        reconstructed = Hypothesis.from_dict(data)
        assert reconstructed.id == hypothesis.id
        assert reconstructed.title == hypothesis.title
        assert reconstructed.confidence_score == hypothesis.confidence_score


class TestAlphaGenAgent:
    """Test AlphaGenAgent."""
    @pytest.fixture
    def agent_config(self):
        """Fixture for agent configuration."""
        return AgentConfig(
            name="test_alphagen",
            custom_config={
               "openai_api_key": "test_key"
            }
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        """Fixture to create AlphaGenAgent instance."""
        with patch('trading.agents.alpha.alphagen_agent.OpenAI'):
            agent = AlphaGenAgent(agent_config)
            agent._setup()
            return agent
    
    def test_agent_creation(self, agent_config):
        """Test AlphaGenAgent creation."""
        with patch('trading.agents.alpha.alphagen_agent.OpenAI'):
            agent = AlphaGenAgent(agent_config)
            assert agent.config.name == "test_alphagen"
            assert agent.get_capabilities() == ["hypothesis_generation", "market_analysis", "llm_integration"]
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful execution."""
        with patch.object(agent, '_analyze_market_context') as mock_analyze, \
             patch.object(agent, '_generate_hypotheses') as mock_generate, \
             patch.object(agent, '_rank_hypotheses') as mock_rank:
            
            mock_analyze.return_value = {"market_regime": "trending"}
            mock_generate.return_value = [
                Hypothesis(
                    id="test_hyp",
                    title="Test Hypothesis",
                    description="Test",
                    strategy_type="momentum",
                    asset_class="equity",
                    timeframe="1d",
                    entry_conditions=[],
                    exit_conditions=[],
                    risk_parameters={},
                    confidence_score=0.8,
                    reasoning="Test",
                    data_sources=[]
                )
            ]
            mock_rank.return_value = mock_generate.return_value
            
            result = await agent.execute()
            
            assert result.success is True
            assert "hypotheses" in result.data
            assert len(result.data["hypotheses"]) == 1
    
    def test_validate_config(self, agent):
        """Test configuration validation."""
        assert agent.validate_config() is True
        
        # Test invalid config
        agent.config.custom_config = {"openai_api_key": "invalid_key"}
        assert agent.validate_config() is False

class TestSignalTester:
    """Test SignalTester."""
    @pytest.fixture
    def test_config(self):
        """Fixture for test configuration."""
        return TestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            tickers=["SPY", "QQQ"],
            timeframes=["1d", "1h"]
        )
    
    @pytest.fixture
    def agent_config(self):
        """Fixture for agent configuration."""
        return AgentConfig(
            name="test_signal_tester",
            custom_config={
                "test_config": {
                    "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "tickers": ["SPY", "QQQ"],
                    "timeframes": ["1d", "1h"]
                }
            }
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        """Fixture to create SignalTester instance."""
        agent = SignalTester(agent_config)
        agent._setup()
        return agent
    
    def test_agent_creation(self, agent_config):
        """Test SignalTester creation."""
        agent = SignalTester(agent_config)
        assert agent.config.name == "test_signal_tester"
        assert agent.get_capabilities() == ["hypothesis_testing", "performance_analysis", "risk_metrics"]
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful execution."""
        hypothesis = Hypothesis(
            id="test_hyp",
            title="Test Strategy",
            description="Test",
            strategy_type="momentum",
            asset_class="equity",
            timeframe="1d",
            entry_conditions=["RSI <30"],
            exit_conditions=["RSI >70"],
            risk_parameters={"stop_loss": 0.02},
            confidence_score=0.8,
            reasoning="Test",
            data_sources=[]
        )
        
        with patch.object(agent, '_test_hypothesis') as mock_test:
            mock_test.return_value = TestResult(
                    hypothesis_id="test_hyp",
                    ticker="SPY",
                    timeframe="1d",
                    regime="bull",
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    total_return=0.05,
                    sharpe_ratio=1.2,
                    max_drawdown=-0.02,
                    win_rate=0.6,
                    profit_factor=1.5,
                    calmar_ratio=2.5,
                    sortino_ratio=1.8,
                    total_trades=10,
                    winning_trades=6,
                    losing_trades=4,
                    avg_win=0.02,
                    avg_loss=-0.01,
                    largest_win=0.05,
                    largest_loss=-0.02,
                    volatility=0.15,
                    var_95=-0.03,
                    cvar_95=-0.04,
                    beta=1.0,
                    alpha=0.01,
                    information_ratio=0.8,
                    execution_time=1.0,
                    data_quality_score=0.9,
                    confidence_score=0.8,
                    equity_curve=pd.Series([100, 105]),
                    trade_log=[]
                )
            
            result = await agent.execute(hypotheses=[hypothesis])
            
            assert result.success is True
            assert "test_results" in result.data
            assert len(result.data["test_results"]) == 1
    
    def test_validate_config(self, agent):
        """Test configuration validation."""
        assert agent.validate_config() is True
        
        # Test invalid config
        agent.config.custom_config = {"test_config": {"start_date": "invalid_date"}}
        assert agent.validate_config() is False


class TestRiskValidator:
    """Test RiskValidator."""
    @pytest.fixture
    def validation_config(self):
        """Fixture to create validation configuration."""
        return ValidationConfig(
            max_correlation_threshold=0.7,
            min_sharpe_ratio=0.5,
            max_drawdown_threshold=0.15
        )
    
    @pytest.fixture
    def agent_config(self):
        """Fixture for agent configuration."""
        return AgentConfig(
            name="test_risk_validator",
            custom_config={
                "validation_config": {
                  "max_correlation_threshold": 0.7,
                   "min_sharpe_ratio": 0.5,
                  "max_drawdown_threshold": 0.15
                },
               "existing_strategies": {}
            }
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        """Fixture to create RiskValidator instance."""
        agent = RiskValidator(agent_config)
        agent._setup()
        return agent
    
    def test_agent_creation(self, agent_config):
        """Test RiskValidator creation."""
        agent = RiskValidator(agent_config)
        assert agent.config.name == "test_risk_validator"
        assert agent.get_capabilities() == ["risk_validation", "correlation_analysis", "stability_testing"]
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful execution."""
        test_result = TestResult(
            hypothesis_id="test_hyp",
            ticker="SPY",
            timeframe="1d",
            regime="bull",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.02,
            win_rate=0.6,
            profit_factor=1.5,
            calmar_ratio=2.5,
            sortino_ratio=1.8,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            avg_win=0.02,
            avg_loss=-0.01,
            largest_win=0.05,
            largest_loss=-0.02,
            volatility=0.15,
            var_95=0.03,
            cvar_95=-0.04,
            beta=1.0,
            alpha=0.01,
            information_ratio=0.8,
            execution_time=1.0,
            data_quality_score=0.9,
            confidence_score=0.8,
            equity_curve=pd.Series([100, 105]),
            trade_log=[]
        )
        
        with patch.object(agent, '_validate_test_result') as mock_validate:
            mock_validate.return_value = ValidationResult(
                hypothesis_id="test_hyp",
                test_result_id="test_result",
                correlation_score=0.8,
                stability_score=0.7,
                viability_score=0.9,
                overall_score=0.8,
                correlation_analysis={},
                stability_analysis={},
                viability_analysis={},
                risk_flags=[],
                warnings=[],
                recommendations=[],
                validation_date=datetime.now(),
                validator_version="1.0.0",
                confidence_level=0.8,
                is_approved=True,
                approval_reason="Strategy approved"
            )
            
            result = await agent.execute(test_results=[test_result])
            
            assert result.success is True
            assert "validation_results" in result.data
            assert len(result.data["validation_results"]) == 1


class TestSentimentIngestion:
    """TestSentimentIngestion."""
    @pytest.fixture
    def agent_config(self):
        """Fixture for agent configuration."""
        return AgentConfig(
            name="test_sentiment_ingestion",
            custom_config={
                "tickers": ["SPY", "QQQ"],
                "reddit": {
                    "client_id": "test_id",
                    "client_secret": "test_secret",
                    "user_agent": "test_agent"
                },
                "twitter": {
                    "bearer_token": "test_token"
                }
            }
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        """Fixture to create SentimentIngestion instance."""
        with patch('trading.agents.alpha.sentiment_ingestion.praw.Reddit'), patch('trading.agents.alpha.sentiment_ingestion.tweepy.Client'):
            agent = SentimentIngestion(agent_config)
            agent._setup()
            return agent
    
    def test_agent_creation(self, agent_config):
        """Test SentimentIngestion creation."""
        with patch('trading.agents.alpha.sentiment_ingestion.praw.Reddit'), patch('trading.agents.alpha.sentiment_ingestion.tweepy.Client'):
            agent = SentimentIngestion(agent_config)
            assert agent.config.name == "test_sentiment_ingestion"
            assert agent.get_capabilities() == ["sentiment_collection", "data_aggregation", "real_time_processing"]
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful execution."""
        with patch.object(agent, '_collect_reddit_sentiment') as mock_reddit, \
             patch.object(agent, '_collect_twitter_sentiment') as mock_twitter, \
             patch.object(agent, '_collect_news_sentiment') as mock_news, \
             patch.object(agent, '_collect_substack_sentiment') as mock_substack, \
             patch.object(agent, '_update_sentiment_index') as mock_update:
            
            mock_reddit.return_value = []
            mock_twitter.return_value = []
            mock_news.return_value = []
            mock_substack.return_value = []
            mock_update.return_value = None
            
            result = await agent.execute()
            
            assert result.success is True
            assert "sentiment_data" in result.data
            assert "sentiment_index" in result.data


class TestAlphaRegistry:
    """Test AlphaRegistry."""
    @pytest.fixture
    def agent_config(self):
        """Fixture for agent configuration."""
        return AgentConfig(
            name="test_alpha_registry",
            custom_config={
                "database": ":memory:"  # Use in-memory database for testing
            }
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        """Fixture to create AlphaRegistry instance."""
        agent = AlphaRegistry(agent_config)
        agent._setup()
        return agent
    
    def test_agent_creation(self, agent_config):
        """Test AlphaRegistry creation."""
        agent = AlphaRegistry(agent_config)
        assert agent.config.name == "test_alpha_registry"
        assert agent.get_capabilities() == ["strategy_registration", "lifecycle_management", "decay_detection"]
    
    @pytest.mark.asyncio
    async def test_register_strategy(self, agent):
        """Test strategy registration."""
        hypothesis = Hypothesis(
            id="test_hyp",
            title="Test Strategy",
            description="Test",
            strategy_type="momentum",
            asset_class="equity",
            timeframe="1d",
            entry_conditions=[],
            exit_conditions=[],
            risk_parameters={},
            confidence_score=0.8,
            reasoning="Test",
            data_sources=[]
        )
        
        test_result = TestResult(
            hypothesis_id="test_hyp",
            ticker="SPY",
            timeframe="1d",
            regime="bull",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.02,
            win_rate=0.6,
            profit_factor=1.5,
            calmar_ratio=2.5,
            sortino_ratio=1.8,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            avg_win=0.02,
            avg_loss=-0.01,
            largest_win=0.05,
            largest_loss=-0.02,
            volatility=0.15,
            var_95=0.03,
            cvar_95=-0.04,
            beta=1.0,
            alpha=0.01,
            information_ratio=0.8,
            execution_time=1.0,
            data_quality_score=0.9,
            confidence_score=0.8,
            equity_curve=pd.Series([100, 105]),
            trade_log=[]
        )
        
        validation_result = ValidationResult(
            hypothesis_id="test_hyp",
            test_result_id="test_result",
            correlation_score=0.8,
            stability_score=0.7,
            viability_score=0.9,
            overall_score=0.8,
            correlation_analysis={},
            stability_analysis={},
            viability_analysis={},
            risk_flags=[],
            warnings=[],
            recommendations=[],
            validation_date=datetime.now(),
            validator_version="1.0.0",
            confidence_level=0.8,
            is_approved=True,
            approval_reason="Strategy approved"
        )
        
        result = await agent.execute(
            operation="register",
            hypothesis=hypothesis,
            test_result=test_result,
            validation_result=validation_result
        )
        
        assert result.success is True
        assert "strategy_id" in result.data
        assert result.data["status"] == "registered" 
    @pytest.mark.asyncio
    async def test_decay_analysis(self, agent):
        """Test decay analysis."""
        # First register a strategy
        hypothesis = Hypothesis(
            id="test_hyp",
            title="Test Strategy",
            description="Test",
            strategy_type="momentum",
            asset_class="equity",
            timeframe="1d",
            entry_conditions=[],
            exit_conditions=[],
            risk_parameters={},
            confidence_score=0.8,
            reasoning="Test",
            data_sources=[]
        )
        
        test_result = TestResult(
            hypothesis_id="test_hyp",
            ticker="SPY",
            timeframe="1d",
            regime="bull",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.02,
            win_rate=0.6,
            profit_factor=1.5,
            calmar_ratio=2.5,
            sortino_ratio=1.8,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            avg_win=0.02,
            avg_loss=-0.01,
            largest_win=0.05,
            largest_loss=-0.02,
            volatility=0.15,
            var_95=0.03,
            cvar_95=-0.04,
            beta=1.0,
            alpha=0.01,
            information_ratio=0.8,
            execution_time=1.0,
            data_quality_score=0.9,
            confidence_score=0.8,
            equity_curve=pd.Series([100, 105]),
            trade_log=[]
        )
        
        validation_result = ValidationResult(
            hypothesis_id="test_hyp",
            test_result_id="test_result",
            correlation_score=0.8,
            stability_score=0.7,
            viability_score=0.9,
            overall_score=0.8,
            correlation_analysis={},
            stability_analysis={},
            viability_analysis={},
            risk_flags=[],
            warnings=[],
            recommendations=[],
            validation_date=datetime.now(),
            validator_version="1.0.0",
            confidence_level=0.8,
            is_approved=True,
            approval_reason="Strategy approved"
        )
        
        await agent.execute(
            operation="register",
            hypothesis=hypothesis,
            test_result=test_result,
            validation_result=validation_result
        )
        
        # Now perform decay analysis
        result = await agent.execute(operation="decay_analysis")
        
        assert result.success is True
        assert "analyses_performed" in result.data


class TestAlphaOrchestrator:
    """Test AlphaOrchestrator."""
    @pytest.fixture
    def agent_config(self):
        """Fixture for agent configuration."""
        return AgentConfig(
            name="test_alpha_orchestrator",
            custom_config={
                "alphagen": {
                    "openai_api_key": "test_key"
                },
                "signal_tester": {
                    "test_config": {
                        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                        "end_date": datetime.now().isoformat(),
                        "tickers": ["SPY", "QQQ"],
                        "timeframes": ["1d", "1h"]
                    }
                },
                "risk_validator": {
                    "validation_config": {
                      "max_correlation_threshold": 0.7,
                       "min_sharpe_ratio": 0.5,
                      "max_drawdown_threshold": 0.15
                    },
                   "existing_strategies": {}
                },
                "alpha_registry": {
                    "database": ":memory:"
                }
            },
            sentiment_ingestion={
                "tickers": []
            }
        )
    
    @pytest.fixture
    def agent(self, agent_config):
        """Fixture to create AlphaOrchestrator instance."""
        with patch('trading.agents.alpha.alpha_orchestrator.AlphaGenAgent'), patch('trading.agents.alpha.alpha_orchestrator.SignalTester'), patch('trading.agents.alpha.alpha_orchestrator.RiskValidator'), patch('trading.agents.alpha.alpha_orchestrator.AlphaRegistry'), patch('trading.agents.alpha.alpha_orchestrator.SentimentIngestion'):
            agent = AlphaOrchestrator(agent_config)
            agent._setup()
            return agent
    
    def test_agent_creation(self, agent_config):
        """Test AlphaOrchestrator creation."""
        with patch('trading.agents.alpha.alpha_orchestrator.AlphaGenAgent'), patch('trading.agents.alpha.alpha_orchestrator.SignalTester'), patch('trading.agents.alpha.alpha_orchestrator.RiskValidator'), patch('trading.agents.alpha.alpha_orchestrator.AlphaRegistry'), patch('trading.agents.alpha.alpha_orchestrator.SentimentIngestion'):
            agent = AlphaOrchestrator(agent_config)
            assert agent.config.name == "test_alpha_orchestrator"
            assert agent.get_capabilities() == ["full_lifecycle_orchestration", "decision_logging", "agent_coordination"]
    
    @pytest.mark.asyncio
    async def test_execute_full_cycle(self, agent):
        """Test full cycle execution."""
        # Mock all agent executions
        with patch.object(agent.sentiment_ingestion, 'execute') as mock_sentiment, \
             patch.object(agent.alphagen_agent, 'execute') as mock_alphagen, \
             patch.object(agent.signal_tester, 'execute') as mock_tester, \
             patch.object(agent.risk_validator, 'execute') as mock_validator, \
             patch.object(agent.alpha_registry, 'execute') as mock_registry:
            
            # Mock sentiment result
            mock_sentiment.return_value = AgentResult(
                success=True,
                data={
                    "sentiment_index": {"SPY": {"overall_sentiment": 0.6}},
                    "sentiment_data": {}
                }
            )
            
            # Mock alphagen result
            mock_alphagen.return_value = AgentResult(
                success=True,
                data={
                  "hypotheses": [
                      {
                          "id": "test_hyp",
                          "title": "Test Strategy",
                          "strategy_type": "momentum",
                          "asset_class": "equity",
                          "timeframe": "1d",
                          "entry_conditions": [],
                          "exit_conditions": [],
                          "risk_parameters": {},
                          "confidence_score": 0.8,
                          "reasoning": "Test",
                          "data_sources": []
                      }
                  ]
                }
            )
            
            # Mock tester result
            mock_tester.return_value = AgentResult(
                success=True,
                data={
                    "test_results": [
                        {
                            "hypothesis_id": "test_hyp",
                            "ticker": "SPY",
                            "timeframe": "1d",
                            "sharpe_ratio": 1.2,
                            "total_return": 0.05,
                            "max_drawdown": -0.02,
                            "win_rate": 0.6
                        }
                    ]
                }
            )
            
            # Mock validator result
            mock_validator.return_value = AgentResult(
                success=True,
                data={
             "validation_results": [
                    {
                        "hypothesis_id": "test_hyp",
                        "test_result_id": "test_result",
                        "is_approved": True,
                        "overall_score": 0.8
                    }
                ]
            }
            )
            
            # Mock registry result
            mock_registry.return_value = AgentResult(
                success=True,
                data={
                  "strategy_id": "strategy_001",
             "status": "registered"
                }
            )
            
            result = await agent.execute(cycle_type="full")
            
            assert result.success is True
            assert "cycle" in result.data
            assert result.data[cycle]["status"] == "completed"
    
    def test_validate_input(self, agent):
        """Test input validation."""
        assert agent.validate_input(cycle_type="full") is True
        assert agent.validate_input(cycle_type="quick") is True
        assert agent.validate_input(cycle_type="invalid") is False
    
    def test_get_cycle_summary(self, agent):
        """Test cycle summary generation."""
        summary = agent.get_cycle_summary()
        assert "total_cycles" in summary
        assert "active_cycles" in summary
        assert "completed_cycles" in summary


class TestIntegration:
    """Integration tests for the alpha system."""
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test end-to-end workflow."""
        # This would test the complete workflow from hypothesis generation to deployment
        # For now, we'll just verify that all components can be imported and instantiated
        
        from trading.agents.alpha import (
            AlphaGenAgent, SignalTester, RiskValidator,
            SentimentIngestion, AlphaRegistry, AlphaOrchestrator
        )
        
        # Verify all components are available
        assert AlphaGenAgent is not None
        assert SignalTester is not None
        assert RiskValidator is not None
        assert SentimentIngestion is not None
        assert AlphaRegistry is not None
        assert AlphaOrchestrator is not None
    
    def test_data_class_serialization(self):
        """Test that all data classes can be serialized."""
        # Test Hypothesis
        hypothesis = Hypothesis(
            id="test",
            title="Test",
            description="Test",
            strategy_type="momentum",
            asset_class="equity",
            timeframe="1d",
            entry_conditions=[],
            exit_conditions=[],
            risk_parameters={},
            confidence_score=0.8,
            reasoning="Test",
            data_sources=[]
        )
        
        data = hypothesis.to_dict()
        reconstructed = Hypothesis.from_dict(data)
        assert reconstructed.id == hypothesis.id
        
        # Test TestResult
        test_result = TestResult(
            hypothesis_id="test",
            ticker="SPY",
            timeframe="1d",
            regime="bull",
            start_date=datetime.now(),
            end_date=datetime.now(),
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=0.02,
            win_rate=0.6,
            profit_factor=1.5,
            calmar_ratio=2.5,
            sortino_ratio=1.8,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            avg_win=0.02,
            avg_loss=-0.01,
            largest_win=0.05,
            largest_loss=-0.02,
            volatility=0.15,
            var_95=0.03,
            cvar_95=-0.04,
            beta=1.0,
            alpha=0.01,
            information_ratio=0.8,
            execution_time=1.0,
            data_quality_score=0.9,
            confidence_score=0.8,
            equity_curve=pd.Series([100, 105]),
            trade_log=[]
        )
        
        data = test_result.to_dict()
        reconstructed = TestResult.from_dict(data)
        assert reconstructed.hypothesis_id == test_result.hypothesis_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 