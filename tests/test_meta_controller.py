"""
Tests for MetaControllerAgent

Tests the system orchestrator functionality including trigger evaluation,
action execution, and system monitoring.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio

from meta.meta_controller import (
    MetaControllerAgent,
    ActionType,
    TriggerCondition,
    SystemMetrics,
    ActionDecision,
    ActionResult,
    create_meta_controller,
    run_meta_controller
)


class TestActionType:
    """Test ActionType enum"""
    
    def test_action_type_values(self):
        """Test ActionType enum values"""
        assert ActionType.MODEL_REBUILD.value == "model_rebuild"
        assert ActionType.STRATEGY_TUNE.value == "strategy_tune"
        assert ActionType.SENTIMENT_FETCH.value == "sentiment_fetch"
        assert ActionType.REPORT_GENERATE.value == "report_generate"
        assert ActionType.SYSTEM_HEALTH_CHECK.value == "system_health_check"


class TestTriggerCondition:
    """Test TriggerCondition enum"""
    
    def test_trigger_condition_values(self):
        """Test TriggerCondition enum values"""
        assert TriggerCondition.TIME_BASED.value == "time_based"
        assert TriggerCondition.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert TriggerCondition.MARKET_VOLATILITY.value == "market_volatility"
        assert TriggerCondition.SENTIMENT_SHIFT.value == "sentiment_shift"
        assert TriggerCondition.ERROR_THRESHOLD.value == "error_threshold"


class TestSystemMetrics:
    """Test SystemMetrics dataclass"""
    
    def test_system_metrics_creation(self):
        """Test creating a SystemMetrics instance"""
        metrics = SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={"model1": 0.8, "model2": 0.7},
            strategy_performance={"strategy1": 0.6},
            sentiment_scores={"AAPL": 0.5, "TSLA": 0.3},
            system_health={"cpu_usage": 50.0, "memory_usage": 60.0},
            market_conditions={"volatility": 0.2, "trend": 0.1},
            error_count=2,
            active_trades=15,
            portfolio_value=50000.0
        )
        
        assert metrics.timestamp == "2024-01-01T12:00:00"
        assert len(metrics.model_performance) == 2
        assert metrics.error_count == 2
        assert metrics.portfolio_value == 50000.0


class TestActionDecision:
    """Test ActionDecision dataclass"""
    
    def test_action_decision_creation(self):
        """Test creating an ActionDecision instance"""
        decision = ActionDecision(
            action_type=ActionType.MODEL_REBUILD,
            trigger_condition=TriggerCondition.PERFORMANCE_DEGRADATION,
            timestamp="2024-01-01T12:00:00",
            reason="Model performance below threshold",
            priority=4,
            estimated_duration=120,
            affected_components=["models", "forecasting"],
            parameters={"performance_threshold": 0.6}
        )
        
        assert decision.action_type == ActionType.MODEL_REBUILD
        assert decision.trigger_condition == TriggerCondition.PERFORMANCE_DEGRADATION
        assert decision.priority == 4
        assert decision.estimated_duration == 120
        assert "models" in decision.affected_components


class TestActionResult:
    """Test ActionResult dataclass"""
    
    def test_action_result_creation(self):
        """Test creating an ActionResult instance"""
        result = ActionResult(
            action_id="model_rebuild_1234567890",
            action_type=ActionType.MODEL_REBUILD,
            start_time="2024-01-01T12:00:00",
            end_time="2024-01-01T14:00:00",
            success=True,
            duration_minutes=120.0,
            results={"models_updated": 3, "performance_improved": 0.1},
            errors=[],
            recommendations=["Monitor new model performance"]
        )
        
        assert result.action_id == "model_rebuild_1234567890"
        assert result.success is True
        assert result.duration_minutes == 120.0
        assert len(result.recommendations) == 1


class TestMetaControllerAgent:
    """Test MetaControllerAgent functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def controller(self, temp_dir):
        """Create MetaControllerAgent instance for testing"""
        config = {
            'meta_controller': {
                'enabled': True,
                'log_level': 'INFO'
            }
        }
        
        with patch('meta.meta_controller.load_config', return_value=config):
            controller = MetaControllerAgent()
            controller.log_dir = Path(temp_dir) / "logs" / "meta_controller"
            controller.log_dir.mkdir(parents=True, exist_ok=True)
            return controller
    
    def test_controller_initialization(self, controller):
        """Test controller initialization"""
        assert controller.name == "MetaControllerAgent"
        assert hasattr(controller, 'thresholds')
        assert hasattr(controller, 'system_metrics')
        assert hasattr(controller, 'action_history')
        assert hasattr(controller, 'pending_actions')
    
    def test_load_trigger_thresholds(self, controller):
        """Test trigger thresholds loading"""
        # Test with existing file
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='{"test": "data"}')):
                thresholds = controller._load_trigger_thresholds()
                assert thresholds == {"test": "data"}
        
        # Test with missing file (should return defaults)
        with patch('pathlib.Path.exists', return_value=False):
            thresholds = controller._load_trigger_thresholds()
            assert 'model_rebuild' in thresholds
            assert 'strategy_tune' in thresholds
            assert 'sentiment_fetch' in thresholds
    
    def test_collect_system_metrics(self, controller):
        """Test system metrics collection"""
        with patch.object(controller, '_get_model_performance', return_value={'model1': 0.8}):
            with patch.object(controller, '_get_strategy_performance', return_value={'strategy1': 0.6}):
                with patch.object(controller, '_get_sentiment_metrics', return_value={'AAPL': 0.5}):
                    with patch.object(controller, '_get_system_health', return_value={'cpu_usage': 50.0}):
                        with patch.object(controller, '_get_market_conditions', return_value={'volatility': 0.2}):
                            with patch.object(controller, '_get_error_count', return_value=2):
                                with patch.object(controller, '_get_active_trades', return_value=15):
                                    with patch.object(controller, '_get_portfolio_value', return_value=50000.0):
                                        metrics = controller.collect_system_metrics()
                                        
                                        assert metrics is not None
                                        assert metrics.model_performance == {'model1': 0.8}
                                        assert metrics.strategy_performance == {'strategy1': 0.6}
                                        assert metrics.error_count == 2
                                        assert metrics.active_trades == 15
                                        assert metrics.portfolio_value == 50000.0
    
    def test_evaluate_model_rebuild_trigger(self, controller):
        """Test model rebuild trigger evaluation"""
        # Test performance degradation trigger
        metrics = SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={"model1": 0.4, "model2": 0.3},  # Below threshold
            strategy_performance={},
            sentiment_scores={},
            system_health={},
            market_conditions={},
            error_count=0,
            active_trades=0,
            portfolio_value=0.0
        )
        
        decision = controller._evaluate_model_rebuild(metrics)
        
        assert decision is not None
        assert decision.action_type == ActionType.MODEL_REBUILD
        assert decision.trigger_condition == TriggerCondition.PERFORMANCE_DEGRADATION
        assert decision.priority == 4
        assert "performance" in decision.reason.lower()
    
    def test_evaluate_strategy_tune_trigger(self, controller):
        """Test strategy tune trigger evaluation"""
        # Test performance degradation trigger
        metrics = SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={},
            strategy_performance={"strategy1": 0.3, "strategy2": 0.4},  # Below threshold
            sentiment_scores={},
            system_health={},
            market_conditions={},
            error_count=0,
            active_trades=0,
            portfolio_value=0.0
        )
        
        decision = controller._evaluate_strategy_tune(metrics)
        
        assert decision is not None
        assert decision.action_type == ActionType.STRATEGY_TUNE
        assert decision.trigger_condition == TriggerCondition.PERFORMANCE_DEGRADATION
        assert decision.priority == 3
        assert "performance" in decision.reason.lower()
    
    def test_evaluate_sentiment_fetch_trigger(self, controller):
        """Test sentiment fetch trigger evaluation"""
        # Set last fetch time to be old
        controller.last_action_times[ActionType.SENTIMENT_FETCH] = datetime.now() - timedelta(hours=2)
        
        metrics = SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={},
            strategy_performance={},
            sentiment_scores={},
            system_health={},
            market_conditions={},
            error_count=0,
            active_trades=0,
            portfolio_value=0.0
        )
        
        decision = controller._evaluate_sentiment_fetch(metrics)
        
        assert decision is not None
        assert decision.action_type == ActionType.SENTIMENT_FETCH
        assert decision.trigger_condition == TriggerCondition.TIME_BASED
        assert decision.priority == 2
        assert "stale" in decision.reason.lower()
    
    def test_evaluate_report_generation_trigger(self, controller):
        """Test report generation trigger evaluation"""
        # Set last report time to be old
        controller.last_action_times[ActionType.REPORT_GENERATE] = datetime.now() - timedelta(hours=8)
        
        metrics = SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={},
            strategy_performance={},
            sentiment_scores={},
            system_health={},
            market_conditions={},
            error_count=0,
            active_trades=0,
            portfolio_value=0.0
        )
        
        decision = controller._evaluate_report_generation(metrics)
        
        assert decision is not None
        assert decision.action_type == ActionType.REPORT_GENERATE
        assert decision.trigger_condition == TriggerCondition.TIME_BASED
        assert decision.priority == 2
        assert "due" in decision.reason.lower()
    
    def test_evaluate_system_health_trigger(self, controller):
        """Test system health trigger evaluation"""
        # Test high error count trigger
        metrics = SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={},
            strategy_performance={},
            sentiment_scores={},
            system_health={},
            market_conditions={},
            error_count=15,  # High error count
            active_trades=0,
            portfolio_value=0.0
        )
        
        decision = controller._evaluate_system_health(metrics)
        
        assert decision is not None
        assert decision.action_type == ActionType.SYSTEM_HEALTH_CHECK
        assert decision.trigger_condition == TriggerCondition.ERROR_THRESHOLD
        assert decision.priority == 5
        assert "error" in decision.reason.lower()
    
    def test_evaluate_triggers(self, controller):
        """Test comprehensive trigger evaluation"""
        # Create metrics that should trigger multiple actions
        controller.last_action_times[ActionType.SENTIMENT_FETCH] = datetime.now() - timedelta(hours=2)
        controller.last_action_times[ActionType.REPORT_GENERATE] = datetime.now() - timedelta(hours=8)
        
        metrics = SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={"model1": 0.4},  # Below threshold
            strategy_performance={"strategy1": 0.3},  # Below threshold
            sentiment_scores={},
            system_health={},
            market_conditions={},
            error_count=15,  # High error count
            active_trades=0,
            portfolio_value=0.0
        )
        
        decisions = controller.evaluate_triggers(metrics)
        
        assert len(decisions) >= 4  # Should trigger multiple actions
        action_types = [d.action_type for d in decisions]
        assert ActionType.MODEL_REBUILD in action_types
        assert ActionType.STRATEGY_TUNE in action_types
        assert ActionType.SENTIMENT_FETCH in action_types
        assert ActionType.REPORT_GENERATE in action_types
        assert ActionType.SYSTEM_HEALTH_CHECK in action_types
    
    @pytest.mark.asyncio
    async def test_execute_model_rebuild(self, controller):
        """Test model rebuild action execution"""
        # Mock model agent
        mock_model_agent = Mock()
        mock_model_agent.run.return_value = {'status': 'success', 'models_updated': 3}
        controller.model_agent = mock_model_agent
        
        decision = ActionDecision(
            action_type=ActionType.MODEL_REBUILD,
            trigger_condition=TriggerCondition.PERFORMANCE_DEGRADATION,
            timestamp=datetime.now().isoformat(),
            reason="Test model rebuild",
            priority=4,
            estimated_duration=120,
            affected_components=["models"],
            parameters={}
        )
        
        result = await controller._execute_model_rebuild(decision)
        
        assert result['success'] is True
        assert 'models_updated' in result['results']
        assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_execute_strategy_tune(self, controller):
        """Test strategy tune action execution"""
        # Mock strategy agent
        mock_strategy_agent = Mock()
        mock_strategy_agent.run.return_value = {'status': 'success', 'strategies_tested': 5}
        controller.strategy_agent = mock_strategy_agent
        
        decision = ActionDecision(
            action_type=ActionType.STRATEGY_TUNE,
            trigger_condition=TriggerCondition.PERFORMANCE_DEGRADATION,
            timestamp=datetime.now().isoformat(),
            reason="Test strategy tune",
            priority=3,
            estimated_duration=60,
            affected_components=["strategies"],
            parameters={}
        )
        
        result = await controller._execute_strategy_tune(decision)
        
        assert result['success'] is True
        assert 'strategies_tested' in result['results']
        assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_execute_sentiment_fetch(self, controller):
        """Test sentiment fetch action execution"""
        # Mock sentiment fetcher
        mock_fetcher = Mock()
        mock_fetcher.fetch_all_sentiment.return_value = {
            'news': [Mock()],
            'reddit': [Mock(), Mock()],
            'twitter': [Mock()]
        }
        controller.sentiment_fetcher = mock_fetcher
        
        decision = ActionDecision(
            action_type=ActionType.SENTIMENT_FETCH,
            trigger_condition=TriggerCondition.TIME_BASED,
            timestamp=datetime.now().isoformat(),
            reason="Test sentiment fetch",
            priority=2,
            estimated_duration=15,
            affected_components=["sentiment"],
            parameters={}
        )
        
        result = await controller._execute_sentiment_fetch(decision)
        
        assert result['success'] is True
        assert 'AAPL' in result['results']
        assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_execute_report_generation(self, controller):
        """Test report generation action execution"""
        # Add some test data
        controller.system_metrics.append(SystemMetrics(
            timestamp="2024-01-01T12:00:00",
            model_performance={"model1": 0.8},
            strategy_performance={"strategy1": 0.6},
            sentiment_scores={"AAPL": 0.5},
            system_health={"cpu_usage": 50.0},
            market_conditions={"volatility": 0.2},
            error_count=2,
            active_trades=15,
            portfolio_value=50000.0
        ))
        
        decision = ActionDecision(
            action_type=ActionType.REPORT_GENERATE,
            trigger_condition=TriggerCondition.TIME_BASED,
            timestamp=datetime.now().isoformat(),
            reason="Test report generation",
            priority=2,
            estimated_duration=30,
            affected_components=["reporting"],
            parameters={}
        )
        
        result = await controller._execute_report_generation(decision)
        
        assert result['success'] is True
        assert 'report_path' in result['results']
        assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_execute_system_health_check(self, controller):
        """Test system health check action execution"""
        decision = ActionDecision(
            action_type=ActionType.SYSTEM_HEALTH_CHECK,
            trigger_condition=TriggerCondition.ERROR_THRESHOLD,
            timestamp=datetime.now().isoformat(),
            reason="Test health check",
            priority=5,
            estimated_duration=10,
            affected_components=["system"],
            parameters={}
        )
        
        result = await controller._execute_system_health_check(decision)
        
        assert result['success'] is True
        assert 'system_status' in result['results']
        assert 'components' in result['results']
        assert len(result['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_execute_action(self, controller):
        """Test action execution wrapper"""
        # Mock sub-agent
        mock_model_agent = Mock()
        mock_model_agent.run.return_value = {'status': 'success'}
        controller.model_agent = mock_model_agent
        
        decision = ActionDecision(
            action_type=ActionType.MODEL_REBUILD,
            trigger_condition=TriggerCondition.PERFORMANCE_DEGRADATION,
            timestamp=datetime.now().isoformat(),
            reason="Test action execution",
            priority=4,
            estimated_duration=120,
            affected_components=["models"],
            parameters={}
        )
        
        result = await controller.execute_action(decision)
        
        assert result.action_id is not None
        assert result.action_type == ActionType.MODEL_REBUILD
        assert result.success is True
        assert result.duration_minutes > 0
        assert len(result.recommendations) > 0
    
    def test_get_performance_summary(self, controller):
        """Test performance summary generation"""
        # Add test metrics
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=f"2024-01-01T{i:02d}:00:00",
                model_performance={"model1": 0.7 + i * 0.01},
                strategy_performance={"strategy1": 0.6 + i * 0.01},
                sentiment_scores={"AAPL": 0.5},
                system_health={"cpu_usage": 50.0},
                market_conditions={"volatility": 0.2},
                error_count=2,
                active_trades=15,
                portfolio_value=50000.0
            )
            controller.system_metrics.append(metrics)
        
        # Add test actions
        for i in range(5):
            action = ActionResult(
                action_id=f"action_{i}",
                action_type=ActionType.MODEL_REBUILD,
                start_time=f"2024-01-01T{i:02d}:00:00",
                end_time=f"2024-01-01T{i:02d}:30:00",
                success=i < 4,  # 4 out of 5 successful
                duration_minutes=30.0,
                results={},
                errors=[],
                recommendations=[]
            )
            controller.action_history.append(action)
        
        summary = controller._get_performance_summary()
        
        assert 'model_performance' in summary
        assert 'strategy_performance' in summary
        assert 'action_success_rate' in summary
        assert summary['action_success_rate'] == 0.8  # 4/5 successful
    
    def test_generate_recommendations(self, controller):
        """Test recommendation generation"""
        # Add test data that should generate recommendations
        for i in range(5):
            metrics = SystemMetrics(
                timestamp=f"2024-01-01T{i:02d}:00:00",
                model_performance={"model1": 0.4},  # Low performance
                strategy_performance={"strategy1": 0.3},  # Low performance
                sentiment_scores={"AAPL": 0.5},
                system_health={"cpu_usage": 85.0, "memory_usage": 90.0},  # High usage
                market_conditions={"volatility": 0.2},
                error_count=10,  # High error count
                active_trades=15,
                portfolio_value=50000.0
            )
            controller.system_metrics.append(metrics)
        
        recommendations = controller._generate_recommendations()
        
        assert len(recommendations) > 0
        assert any("performance" in rec.lower() for rec in recommendations)
        assert any("error" in rec.lower() for rec in recommendations)
        assert any("usage" in rec.lower() for rec in recommendations)
    
    def test_run_method(self, controller):
        """Test main run method"""
        with patch.object(controller, 'collect_system_metrics') as mock_collect:
            with patch.object(controller, 'evaluate_triggers') as mock_evaluate:
                with patch.object(controller, 'execute_action') as mock_execute:
                    # Mock successful execution
                    mock_result = ActionResult(
                        action_id="test",
                        action_type=ActionType.MODEL_REBUILD,
                        start_time="2024-01-01T12:00:00",
                        end_time="2024-01-01T12:30:00",
                        success=True,
                        duration_minutes=30.0,
                        results={},
                        errors=[],
                        recommendations=[]
                    )
                    mock_execute.return_value = mock_result
                    
                    mock_collect.return_value = SystemMetrics(
                        timestamp="2024-01-01T12:00:00",
                        model_performance={"model1": 0.4},
                        strategy_performance={},
                        sentiment_scores={},
                        system_health={},
                        market_conditions={},
                        error_count=0,
                        active_trades=0,
                        portfolio_value=0.0
                    )
                    
                    mock_evaluate.return_value = [
                        ActionDecision(
                            action_type=ActionType.MODEL_REBUILD,
                            trigger_condition=TriggerCondition.PERFORMANCE_DEGRADATION,
                            timestamp=datetime.now().isoformat(),
                            reason="Test",
                            priority=4,
                            estimated_duration=120,
                            affected_components=["models"],
                            parameters={}
                        )
                    ]
                    
                    result = controller.run()
                    
                    assert result['status'] == 'success'
                    assert result['metrics_collected'] is True
                    assert result['decisions_made'] == 1
                    assert result['actions_executed'] == 1


class TestMetaControllerIntegration:
    """Integration tests for MetaControllerAgent"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_controller_workflow(self, temp_dir):
        """Test complete controller workflow"""
        config = {
            'meta_controller': {
                'enabled': True,
                'log_level': 'INFO'
            }
        }
        
        with patch('meta.meta_controller.load_config', return_value=config):
            controller = MetaControllerAgent()
            controller.log_dir = Path(temp_dir) / "logs" / "meta_controller"
            controller.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock sub-agents
        controller.model_agent = Mock()
        controller.model_agent.run.return_value = {'status': 'success'}
        
        controller.strategy_agent = Mock()
        controller.strategy_agent.run.return_value = {'status': 'success'}
        
        controller.sentiment_fetcher = Mock()
        controller.sentiment_fetcher.fetch_all_sentiment.return_value = {
            'news': [Mock()],
            'reddit': [Mock()],
            'twitter': [Mock()]
        }
        
        # Test metrics collection
        metrics = controller.collect_system_metrics()
        assert metrics is not None
        
        # Test trigger evaluation
        decisions = controller.evaluate_triggers(metrics)
        assert isinstance(decisions, list)
        
        # Test action execution
        if decisions:
            decision = decisions[0]
            result = asyncio.run(controller.execute_action(decision))
            assert result.success is True
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test create_meta_controller
        with patch('meta.meta_controller.MetaControllerAgent') as mock_controller:
            mock_instance = Mock()
            mock_controller.return_value = mock_instance
            
            controller = create_meta_controller()
            assert controller == mock_instance
        
        # Test run_meta_controller
        with patch('meta.meta_controller.MetaControllerAgent') as mock_controller:
            mock_instance = Mock()
            mock_instance.run.return_value = {'status': 'success'}
            mock_controller.return_value = mock_instance
            
            result = run_meta_controller()
            assert result['status'] == 'success'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 