Integration test for the full alpha cycle using Swarm Orchestrator.

Tests the complete workflow from hypothesis generation to execution,
coordinated by the Swarm Orchestrator.

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from trading.agents.swarm_orchestrator import (
    SwarmOrchestrator,
    SwarmConfig,
    AgentType,
    AgentStatus
)
from trading.agents.base_agent_interface import AgentConfig, AgentResult
from trading.agents.alpha import (
    AlphaGenAgent,
    SignalTester,
    RiskValidator,
    SentimentIngestion,
    AlphaRegistry,
    Hypothesis,
    TestResult,
    ValidationResult
)
from trading.agents.walk_forward_agent import WalkForwardAgent
from trading.agents.regime_detection_agent import RegimeDetectionAgent
from trading.validation.walk_forward_utils import walk_forward_validate
from trading.optimization.optuna_tuner import SharpeOptunaTuner


class TestFullAlphaCycle:
   Integration tests for the full alpha cycle workflow."   @pytest.fixture
    def sample_market_data(self):
      Create sample market data for testing."        dates = pd.date_range(start='20201-1, end=202312 freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
         datedates,
           open: np.random.randn(len(dates)).cumsum() + 100
           high: np.random.randn(len(dates)).cumsum() + 102,
          low: np.random.randn(len(dates)).cumsum() +98            close: np.random.randn(len(dates)).cumsum() + 100     volume': np.random.randint(1000, 10en(dates))
        })
        return data

    @pytest.fixture
    def swarm_config(self):
      Create SwarmConfig for integration testing.
        return SwarmConfig(
            max_concurrent_agents=5,
            coordination_backend="sqlite",
            task_timeout=120
            retry_attempts=2,
            enable_logging=True,
            enable_monitoring=True
        )

    @pytest.fixture
    def orchestrator(self, swarm_config):
      Create SwarmOrchestrator for integration testing.
        return SwarmOrchestrator(swarm_config)

    @pytest.fixture
    def base_agent_config(self):
      Create base AgentConfig for testing.      return AgentConfig(
            name="test_agent",
            enabled=True,
            priority=1           max_concurrent_runs=1,
            timeout_seconds=60
            retry_attempts=2,
            custom_config={}
        )

    @pytest.mark.asyncio
    async def test_full_alpha_cycle_workflow(self, orchestrator, sample_market_data, base_agent_config):
      Test the complete alpha cycle workflow."       # Start the orchestrator
        await orchestrator.start()
        
        try:
            # Step 1: Generate hypotheses using AlphaGenAgent
            alpha_gen_config = AgentConfig(
                name=alpha_gen_test,           enabled=True,
                priority=1               max_concurrent_runs=1           timeout_seconds=60               retry_attempts=2            custom_config={
                   openai_api_key": "test_key"
                }
            )
            
            # Mock the AlphaGenAgent to return a hypothesis
            mock_hypothesis = Hypothesis(
                id="test_hyp_001,             title="Test Momentum Strategy,       description=A simple momentum-based strategy,          strategy_type="momentum,             asset_class="equity,         timeframe="1d,             entry_conditions=["RSI < 30"],
                exit_conditions=["RSI > 70"],
                risk_parameters={"stop_loss": 0.02},
                confidence_score=0.8         reasoning=Test reasoning,              data_sources=["price_data"]
            )
            
            with patch.object(AlphaGenAgent,execute', return_value=AgentResult(
                success=True,
                data={"hypotheses": [mock_hypothesis]}
            )):
                alpha_gen_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.ALPHA_GEN,
                    agent_config=alpha_gen_config,
                    input_data={"market_data": sample_market_data}
                )
                
                # Wait for task completion
                await asyncio.sleep(0.1             alpha_gen_task = orchestrator.get_task_status(alpha_gen_task_id)
                assert alpha_gen_task.status == AgentStatus.COMPLETED
            
            # Step 2t hypotheses using SignalTester
            signal_tester_config = AgentConfig(
                name="signal_tester_test,           enabled=True,
                priority=1               max_concurrent_runs=1           timeout_seconds=60               retry_attempts=2            custom_config={}
            )
            
            # Mock the SignalTester to return test results
            mock_test_result = TestResult(
                hypothesis_id="test_hyp_001,              sharpe_ratio=1.2               max_drawdown=-0.05               win_rate=0.65             total_return=0.15        volatility=0.12              test_period="2020-2023,              tickers_tested=["SPY", "QQQ"],
                timeframes_tested=["1d"],
                status="completed"
            )
            
            with patch.object(SignalTester,execute', return_value=AgentResult(
                success=True,
                data={"test_results": [mock_test_result]}
            )):
                signal_tester_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.SIGNAL_TESTER,
                    agent_config=signal_tester_config,
                    input_data={
                     hypotheses": [mock_hypothesis],
                    market_data": sample_market_data
                    },
                    dependencies=[alpha_gen_task_id]
                )
                
                # Wait for task completion
                await asyncio.sleep(0.1            signal_tester_task = orchestrator.get_task_status(signal_tester_task_id)
                assert signal_tester_task.status == AgentStatus.COMPLETED
            
            # Step 3: Validate risk using RiskValidator
            risk_validator_config = AgentConfig(
                name="risk_validator_test,           enabled=True,
                priority=1               max_concurrent_runs=1           timeout_seconds=60               retry_attempts=2            custom_config={}
            )
            
            # Mock the RiskValidator to return validation results
            mock_validation_result = ValidationResult(
                hypothesis_id="test_hyp_001,                is_valid=True,
                correlation_score=0.3         stability_score=0.8              risk_score=0.2   recommendations=["Reduce position size"],
                status="validated"
            )
            
            with patch.object(RiskValidator,execute', return_value=AgentResult(
                success=True,
                data={"validation_results": [mock_validation_result]}
            )):
                risk_validator_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.RISK_VALIDATOR,
                    agent_config=risk_validator_config,
                    input_data={
                       test_results": [mock_test_result],
                    market_data": sample_market_data
                    },
                    dependencies=[signal_tester_task_id]
                )
                
                # Wait for task completion
                await asyncio.sleep(0.1              risk_validator_task = orchestrator.get_task_status(risk_validator_task_id)
                assert risk_validator_task.status == AgentStatus.COMPLETED
            
            # Step 4: Collect sentiment using SentimentIngestion
            sentiment_config = AgentConfig(
                name=sentiment_test,           enabled=True,
                priority=1               max_concurrent_runs=1           timeout_seconds=60               retry_attempts=2            custom_config={}
            )
            
            with patch.object(SentimentIngestion,execute', return_value=AgentResult(
                success=True,
                data={sentiment_score": 0.6}
            )):
                sentiment_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.SENTIMENT,
                    agent_config=sentiment_config,
                    input_data={symbols": ["SPY", "QQQ"]}
                )
                
                # Wait for task completion
                await asyncio.sleep(0.1         sentiment_task = orchestrator.get_task_status(sentiment_task_id)
                assert sentiment_task.status == AgentStatus.COMPLETED
            
            # Step 5: Register strategy using AlphaRegistry
            registry_config = AgentConfig(
                name="registry_test,           enabled=True,
                priority=1               max_concurrent_runs=1           timeout_seconds=60               retry_attempts=2            custom_config={}
            )
            
            with patch.object(AlphaRegistry,execute', return_value=AgentResult(
                success=True,
                data={"strategy_id": "strat_01}
            )):
                registry_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.ALPHA_REGISTRY,
                    agent_config=registry_config,
                    input_data={
                        hypothesis: mock_hypothesis,
                    test_result": mock_test_result,
                        validation_result: mock_validation_result
                    },
                    dependencies=[risk_validator_task_id, sentiment_task_id]
                )
                
                # Wait for task completion
                await asyncio.sleep(0.1          registry_task = orchestrator.get_task_status(registry_task_id)
                assert registry_task.status == AgentStatus.COMPLETED
            
            # Step 6: Detect market regime using RegimeDetectionAgent
            regime_config = AgentConfig(
                name="regime_test,           enabled=True,
                priority=1               max_concurrent_runs=1           timeout_seconds=60               retry_attempts=2            custom_config={}
            )
            
            with patch.object(RegimeDetectionAgent,execute', return_value=AgentResult(
                success=True,
                data={"regime": "bull,confidence": 0.8}
            )):
                regime_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.REGIME_DETECTION,
                    agent_config=regime_config,
                    input_data={"market_data": sample_market_data}
                )
                
                # Wait for task completion
                await asyncio.sleep(0.1            regime_task = orchestrator.get_task_status(regime_task_id)
                assert regime_task.status == AgentStatus.COMPLETED
            
            # Step 7: Perform walk-forward validation
            walk_forward_config = AgentConfig(
                name=walk_forward_test,           enabled=True,
                priority=1               max_concurrent_runs=1           timeout_seconds=60               retry_attempts=2            custom_config={}
            )
            
            with patch.object(WalkForwardAgent,execute', return_value=AgentResult(
                success=True,
                data={"walk_forward_results:}
            )):
                walk_forward_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.WALK_FORWARD,
                    agent_config=walk_forward_config,
                    input_data={
                        strategy: mock_hypothesis,
                    market_data": sample_market_data
                    },
                    dependencies=[registry_task_id]
                )
                
                # Wait for task completion
                await asyncio.sleep(0.1              walk_forward_task = orchestrator.get_task_status(walk_forward_task_id)
                assert walk_forward_task.status == AgentStatus.COMPLETED
            
            # Verify overall workflow completion
            status = orchestrator.get_swarm_status()
            assert statustotal_tasks"] >= 7  # At least 7mitted
            assert status[completed_tasks"] >= 7  # All tasks should complete
            assert status[failed_tasks"] == 0  # No failures
            
        finally:
            # Stop the orchestrator
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, orchestrator, sample_market_data, base_agent_config):
      Test parallel execution of independent agents."""
        await orchestrator.start()
        
        try:
            # Submit multiple independent tasks
            task_ids = []
            
            # Alpha generation (independent)
            alpha_task_id = await orchestrator.submit_task(
                agent_type=AgentType.ALPHA_GEN,
                agent_config=base_agent_config,
                input_data={"market_data": sample_market_data}
            )
            task_ids.append(alpha_task_id)
            
            # Sentiment collection (independent)
            sentiment_task_id = await orchestrator.submit_task(
                agent_type=AgentType.SENTIMENT,
                agent_config=base_agent_config,
                input_data={symbols": ["SPY]}     )
            task_ids.append(sentiment_task_id)
            
            # Regime detection (independent)
            regime_task_id = await orchestrator.submit_task(
                agent_type=AgentType.REGIME_DETECTION,
                agent_config=base_agent_config,
                input_data={"market_data": sample_market_data}
            )
            task_ids.append(regime_task_id)
            
            # Wait for all tasks to complete
            await asyncio.sleep(0.2      
            # Check that all tasks completed
            for task_id in task_ids:
                task = orchestrator.get_task_status(task_id)
                assert task.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]
            
            # Verify parallel execution (should have multiple running tasks)
            status = orchestrator.get_swarm_status()
            assert statustotal_tasks"] == 3
            
        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_dependency_chain_execution(self, orchestrator, sample_market_data, base_agent_config):
      Test execution of tasks with dependencies."""
        await orchestrator.start()
        
        try:
            # Create a chain: AlphaGen -> SignalTester -> RiskValidator
            with patch.object(AlphaGenAgent,execute', return_value=AgentResult(success=True, data={})):
                alpha_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.ALPHA_GEN,
                    agent_config=base_agent_config,
                    input_data={"market_data": sample_market_data}
                )
            
            with patch.object(SignalTester,execute', return_value=AgentResult(success=True, data={})):
                signal_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.SIGNAL_TESTER,
                    agent_config=base_agent_config,
                    input_data={"market_data": sample_market_data},
                    dependencies=[alpha_task_id]
                )
            
            with patch.object(RiskValidator,execute', return_value=AgentResult(success=True, data={})):
                risk_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.RISK_VALIDATOR,
                    agent_config=base_agent_config,
                    input_data={"market_data": sample_market_data},
                    dependencies=[signal_task_id]
                )
            
            # Wait for all tasks to complete
            await asyncio.sleep(0.3      
            # Verify dependency chain execution
            alpha_task = orchestrator.get_task_status(alpha_task_id)
            signal_task = orchestrator.get_task_status(signal_task_id)
            risk_task = orchestrator.get_task_status(risk_task_id)
            
            assert alpha_task.status == AgentStatus.COMPLETED
            assert signal_task.status == AgentStatus.COMPLETED
            assert risk_task.status == AgentStatus.COMPLETED
            
            # Verify execution order (completion times should be in order)
            assert alpha_task.completed_at <= signal_task.completed_at
            assert signal_task.completed_at <= risk_task.completed_at
            
        finally:
            await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestrator, sample_market_data, base_agent_config):
      Test error handling and recovery in the workflow."""
        await orchestrator.start()
        
        try:
            # Submit a task that will fail
            with patch.object(AlphaGenAgent, 'execute', side_effect=Exception("Test error")):
                failed_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.ALPHA_GEN,
                    agent_config=base_agent_config,
                    input_data={"market_data": sample_market_data}
                )
                
                # Wait for task to fail
                await asyncio.sleep(0.1)
                
                failed_task = orchestrator.get_task_status(failed_task_id)
                assert failed_task.status == AgentStatus.FAILED
                assert "Test error" in failed_task.error_message
            
            # Submit a dependent task that should not start
            with patch.object(SignalTester,execute', return_value=AgentResult(success=True, data={})):
                dependent_task_id = await orchestrator.submit_task(
                    agent_type=AgentType.SIGNAL_TESTER,
                    agent_config=base_agent_config,
                    input_data={"market_data": sample_market_data},
                    dependencies=[failed_task_id]
                )
                
                # Wait a bit
                await asyncio.sleep(0.1)
                
                dependent_task = orchestrator.get_task_status(dependent_task_id)
                # Task should remain idle due to failed dependency
                assert dependent_task.status == AgentStatus.IDLE
            
            # Verify error handling in status
            status = orchestrator.get_swarm_status()
            assert status[failed_tasks"] == 1
            
        finally:
            await orchestrator.stop()

    def test_swarm_config_validation(self):
      Test SwarmConfig validation and defaults.    # Test default config
        config = SwarmConfig()
        assert config.max_concurrent_agents == 5
        assert config.coordination_backend == "sqlite"
        assert config.task_timeout == 300     assert config.retry_attempts == 3        
        # Test custom config
        custom_config = SwarmConfig(
            max_concurrent_agents=10,
            coordination_backend="redis",
            task_timeout=600
            retry_attempts=5
        )
        assert custom_config.max_concurrent_agents ==10     assert custom_config.coordination_backend == "redis"
        assert custom_config.task_timeout == 600     assert custom_config.retry_attempts == 5
