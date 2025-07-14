"""
Tests for Explainability and Audit Trail

Tests both reporting/audit_logger.py and reporting/explainer_agent.py modules.
"""

import pytest
import json
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from reporting.audit_logger import (
    AuditLogger,
    AuditEventType,
    DecisionLevel,
    OrderSide,
    OrderType,
    OrderStatus,
    AuditEvent,
    SignalEvent,
    ModelEvent,
    WeightEvent,
    ForecastEvent,
    TradeEvent,
    OrderEvent,
    RiskEvent,
    create_audit_logger,
    log_trading_decision
)
from reporting.explainer_agent import (
    ExplainerAgent,
    ExplanationType,
    ExplanationLevel,
    Explanation,
    ModelExplanation,
    FeatureExplanation,
    ForecastExplanation,
    TradeExplanation,
    create_explainer_agent,
    explain_trading_decision
)


class TestAuditEventType:
    """Test AuditEventType enum"""
    
    def test_audit_event_type_values(self):
        """Test AuditEventType enum values"""
        assert AuditEventType.SIGNAL_GENERATED.value == "signal_generated"
        assert AuditEventType.MODEL_SELECTED.value == "model_selected"
        assert AuditEventType.WEIGHT_UPDATED.value == "weight_updated"
        assert AuditEventType.FORECAST_MADE.value == "forecast_made"
        assert AuditEventType.TRADE_DECISION.value == "trade_decision"
        assert AuditEventType.ORDER_SUBMITTED.value == "order_submitted"
        assert AuditEventType.ORDER_EXECUTED.value == "order_executed"


class TestDecisionLevel:
    """Test DecisionLevel enum"""
    
    def test_decision_level_values(self):
        """Test DecisionLevel enum values"""
        assert DecisionLevel.SYSTEM.value == "system"
        assert DecisionLevel.STRATEGY.value == "strategy"
        assert DecisionLevel.MODEL.value == "model"
        assert DecisionLevel.TRADE.value == "trade"
        assert DecisionLevel.USER.value == "user"


class TestAuditEvent:
    """Test AuditEvent dataclass"""
    
    def test_audit_event_creation(self):
        """Test creating an AuditEvent instance"""
        event = AuditEvent(
            event_id="test_event",
            event_type=AuditEventType.SIGNAL_GENERATED,
            timestamp=datetime.now().isoformat(),
            session_id="test_session",
            decision_level=DecisionLevel.STRATEGY,
            ticker="AAPL",
            description="Test signal generated",
            metadata={"test": "data"},
            confidence_score=0.8,
            risk_score=0.2,
            tags=["test", "signal"]
        )
        
        assert event.event_id == "test_event"
        assert event.event_type == AuditEventType.SIGNAL_GENERATED
        assert event.ticker == "AAPL"
        assert event.confidence_score == 0.8
        assert event.risk_score == 0.2


class TestSignalEvent:
    """Test SignalEvent dataclass"""
    
    def test_signal_event_creation(self):
        """Test creating a SignalEvent instance"""
        event = SignalEvent(
            event_id="signal_event",
            event_type=AuditEventType.SIGNAL_GENERATED,
            timestamp=datetime.now().isoformat(),
            session_id="test_session",
            decision_level=DecisionLevel.STRATEGY,
            ticker="AAPL",
            signal_type="momentum",
            signal_value=0.75,
            signal_strength=0.8,
            features_used=["rsi", "macd"],
            feature_importance={"rsi": 0.6, "macd": 0.4}
        )
        
        assert event.signal_type == "momentum"
        assert event.signal_value == 0.75
        assert event.signal_strength == 0.8
        assert len(event.features_used) == 2
        assert event.feature_importance["rsi"] == 0.6


class TestModelEvent:
    """Test ModelEvent dataclass"""
    
    def test_model_event_creation(self):
        """Test creating a ModelEvent instance"""
        event = ModelEvent(
            event_id="model_event",
            event_type=AuditEventType.MODEL_SELECTED,
            timestamp=datetime.now().isoformat(),
            session_id="test_session",
            decision_level=DecisionLevel.MODEL,
            ticker="AAPL",
            model_name="LSTM_Ensemble",
            model_type="Neural Network",
            model_version="v2.1",
            model_parameters={"layers": 3, "units": 64},
            model_performance={"accuracy": 0.85, "sharpe": 1.2},
            selection_reason="Best performance on validation set"
        )
        
        assert event.model_name == "LSTM_Ensemble"
        assert event.model_type == "Neural Network"
        assert event.model_performance["accuracy"] == 0.85


class TestForecastEvent:
    """Test ForecastEvent dataclass"""
    
    def test_forecast_event_creation(self):
        """Test creating a ForecastEvent instance"""
        event = ForecastEvent(
            event_id="forecast_event",
            event_type=AuditEventType.FORECAST_MADE,
            timestamp=datetime.now().isoformat(),
            session_id="test_session",
            decision_level=DecisionLevel.MODEL,
            ticker="AAPL",
            forecast_horizon=5,
            forecast_value=155.0,
            forecast_confidence=0.8,
            model_used="LSTM_Ensemble",
            features_contributing=["price", "volume"],
            market_conditions={"volatility": 0.02}
        )
        
        assert event.forecast_value == 155.0
        assert event.forecast_horizon == 5
        assert event.forecast_confidence == 0.8
        assert event.model_used == "LSTM_Ensemble"


class TestTradeEvent:
    """Test TradeEvent dataclass"""
    
    def test_trade_event_creation(self):
        """Test creating a TradeEvent instance"""
        event = TradeEvent(
            event_id="trade_event",
            event_type=AuditEventType.TRADE_DECISION,
            timestamp=datetime.now().isoformat(),
            session_id="test_session",
            decision_level=DecisionLevel.TRADE,
            ticker="AAPL",
            trade_type="buy",
            trade_reason="Strong momentum signal",
            trade_confidence=0.75,
            expected_return=0.05,
            expected_risk=0.02,
            position_size=1000.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        assert event.trade_type == "buy"
        assert event.trade_confidence == 0.75
        assert event.expected_return == 0.05
        assert event.position_size == 1000.0


class TestAuditLogger:
    """Test AuditLogger functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def audit_logger(self, temp_dir):
        """Create AuditLogger instance for testing"""
        config = {
            'audit': {
                'output_formats': ['json', 'csv'],
                'batch_size': 10,
                'flush_interval': 30,
                'max_events': 1000,
                'real_time_logging': True,
                'log_to_console': False,
                'enabled_event_types': ['signal_generated', 'model_selected', 'forecast_made', 'trade_decision'],
                'min_confidence_threshold': 0.0
            }
        }
        
        with patch('reporting.audit_logger.load_config', return_value=config):
            logger = AuditLogger()
            return logger
    
    def test_audit_logger_initialization(self, audit_logger):
        """Test audit logger initialization"""
        assert audit_logger.session_id is not None
        assert audit_logger.session_start is not None
        assert len(audit_logger.events) == 1  # Session start event
        assert audit_logger.output_formats == ['json', 'csv']
        assert audit_logger.batch_size == 10
    
    def test_log_signal_generated(self, audit_logger):
        """Test signal generation logging"""
        event_id = audit_logger.log_signal_generated(
            ticker="AAPL",
            signal_type="momentum",
            signal_value=0.75,
            signal_strength=0.8,
            features_used=["rsi", "macd", "volume"],
            feature_importance={"rsi": 0.4, "macd": 0.4, "volume": 0.2},
            confidence_score=0.85
        )
        
        assert event_id is not None
        assert len(audit_logger.events) == 2  # Session start + signal event
        
        event = audit_logger.get_order_status(event_id)
        assert event is None  # This is not an order event
        
        # Check the actual event
        signal_events = [e for e in audit_logger.events if e.event_type == AuditEventType.SIGNAL_GENERATED]
        assert len(signal_events) == 1
        assert signal_events[0].ticker == "AAPL"
        assert signal_events[0].signal_type == "momentum"
    
    def test_log_model_selected(self, audit_logger):
        """Test model selection logging"""
        event_id = audit_logger.log_model_selected(
            ticker="AAPL",
            model_name="LSTM_Ensemble",
            model_version="v2.1",
            model_type="Neural Network",
            model_parameters={"layers": 3, "units": 64},
            model_performance={"accuracy": 0.85, "sharpe": 1.2},
            selection_reason="Best validation performance"
        )
        
        assert event_id is not None
        
        model_events = [e for e in audit_logger.events if e.event_type == AuditEventType.MODEL_SELECTED]
        assert len(model_events) == 1
        assert model_events[0].model_name == "LSTM_Ensemble"
        assert model_events[0].model_performance["accuracy"] == 0.85
    
    def test_log_forecast_made(self, audit_logger):
        """Test forecast logging"""
        event_id = audit_logger.log_forecast_made(
            ticker="AAPL",
            forecast_horizon=5,
            forecast_value=155.0,
            forecast_confidence=0.8,
            model_used="LSTM_Ensemble",
            features_contributing=["price", "volume", "sentiment"],
            market_conditions={"volatility": 0.02, "trend": "bullish"}
        )
        
        assert event_id is not None
        
        forecast_events = [e for e in audit_logger.events if e.event_type == AuditEventType.FORECAST_MADE]
        assert len(forecast_events) == 1
        assert forecast_events[0].forecast_value == 155.0
        assert forecast_events[0].forecast_confidence == 0.8
    
    def test_log_trade_decision(self, audit_logger):
        """Test trade decision logging"""
        event_id = audit_logger.log_trade_decision(
            ticker="AAPL",
            trade_type="buy",
            trade_reason="Strong momentum signal with positive forecast",
            trade_confidence=0.75,
            expected_return=0.05,
            expected_risk=0.02,
            position_size=1000.0,
            stop_loss=150.0,
            take_profit=160.0
        )
        
        assert event_id is not None
        
        trade_events = [e for e in audit_logger.events if e.event_type == AuditEventType.TRADE_DECISION]
        assert len(trade_events) == 1
        assert trade_events[0].trade_type == "buy"
        assert trade_events[0].trade_confidence == 0.75
        assert trade_events[0].position_size == 1000.0
    
    def test_log_order_submitted(self, audit_logger):
        """Test order submission logging"""
        event_id = audit_logger.log_order_submitted(
            order_id="order_123",
            ticker="AAPL",
            order_type="market",
            order_side="buy",
            quantity=100,
            price=150.0
        )
        
        assert event_id is not None
        
        order_events = [e for e in audit_logger.events if e.event_type == AuditEventType.ORDER_SUBMITTED]
        assert len(order_events) == 1
        assert order_events[0].order_id == "order_123"
        assert order_events[0].quantity == 100
    
    def test_log_risk_check(self, audit_logger):
        """Test risk check logging"""
        event_id = audit_logger.log_risk_check(
            risk_type="position_size",
            risk_level="medium",
            risk_value=0.15,
            risk_threshold=0.1,
            risk_action="warning",
            risk_mitigation="Reduce position size",
            ticker="AAPL"
        )
        
        assert event_id is not None
        
        risk_events = [e for e in audit_logger.events if e.event_type == AuditEventType.RISK_CHECK]
        assert len(risk_events) == 1
        assert risk_events[0].risk_type == "position_size"
        assert risk_events[0].risk_action == "warning"
    
    def test_get_events_filtering(self, audit_logger):
        """Test event filtering"""
        # Log multiple events
        audit_logger.log_signal_generated(
            ticker="AAPL",
            signal_type="momentum",
            signal_value=0.75,
            signal_strength=0.8,
            features_used=["rsi"],
            feature_importance={"rsi": 1.0}
        )
        
        audit_logger.log_signal_generated(
            ticker="TSLA",
            signal_type="momentum",
            signal_value=0.6,
            signal_strength=0.7,
            features_used=["rsi"],
            feature_importance={"rsi": 1.0}
        )
        
        # Filter by ticker
        aapl_events = audit_logger.get_events(ticker="AAPL")
        assert len(aapl_events) == 1
        assert aapl_events[0].ticker == "AAPL"
        
        # Filter by event type
        signal_events = audit_logger.get_events(event_types=[AuditEventType.SIGNAL_GENERATED])
        assert len(signal_events) == 2
        
        # Filter by decision level
        strategy_events = audit_logger.get_events(decision_levels=[DecisionLevel.STRATEGY])
        assert len(strategy_events) == 2
    
    def test_get_performance_summary(self, audit_logger):
        """Test performance summary generation"""
        # Log some events
        audit_logger.log_signal_generated(
            ticker="AAPL",
            signal_type="momentum",
            signal_value=0.75,
            signal_strength=0.8,
            features_used=["rsi"],
            feature_importance={"rsi": 1.0},
            confidence_score=0.85
        )
        
        audit_logger.log_trade_decision(
            ticker="AAPL",
            trade_type="buy",
            trade_reason="Test",
            trade_confidence=0.75,
            expected_return=0.05,
            expected_risk=0.02,
            position_size=1000.0
        )
        
        summary = audit_logger.get_performance_summary()
        
        assert summary['session_id'] == audit_logger.session_id
        assert summary['total_events'] == 3  # Session start + 2 events
        assert summary['events_by_type']['signal_generated'] == 1
        assert summary['events_by_type']['trade_decision'] == 1
    
    def test_export_session_report(self, audit_logger):
        """Test session report export"""
        # Log some events
        audit_logger.log_signal_generated(
            ticker="AAPL",
            signal_type="momentum",
            signal_value=0.75,
            signal_strength=0.8,
            features_used=["rsi"],
            feature_importance={"rsi": 1.0}
        )
        
        # Export report
        report_path = audit_logger.export_session_report()
        
        assert Path(report_path).exists()
        
        # Check report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        assert 'session_metadata' in report_data
        assert 'performance_summary' in report_data
        assert 'events' in report_data
        assert len(report_data['events']) == 2  # Session start + signal event
    
    def test_close_session(self, audit_logger):
        """Test session closing"""
        # Log some events
        audit_logger.log_signal_generated(
            ticker="AAPL",
            signal_type="momentum",
            signal_value=0.75,
            signal_strength=0.8,
            features_used=["rsi"],
            feature_importance={"rsi": 1.0}
        )
        
        # Close session
        audit_logger.close_session()
        
        # Check that session end event was added
        session_events = [e for e in audit_logger.events if 'session_end' in e.tags]
        assert len(session_events) == 1


class TestExplanationType:
    """Test ExplanationType enum"""
    
    def test_explanation_type_values(self):
        """Test ExplanationType enum values"""
        assert ExplanationType.MODEL_SELECTION.value == "model_selection"
        assert ExplanationType.FEATURE_IMPORTANCE.value == "feature_importance"
        assert ExplanationType.FORECAST_REASONING.value == "forecast_reasoning"
        assert ExplanationType.TRADE_DECISION.value == "trade_decision"
        assert ExplanationType.RISK_ASSESSMENT.value == "risk_assessment"


class TestExplanationLevel:
    """Test ExplanationLevel enum"""
    
    def test_explanation_level_values(self):
        """Test ExplanationLevel enum values"""
        assert ExplanationLevel.BASIC.value == "basic"
        assert ExplanationLevel.DETAILED.value == "detailed"
        assert ExplanationLevel.EXPERT.value == "expert"


class TestExplanation:
    """Test Explanation dataclass"""
    
    def test_explanation_creation(self):
        """Test creating an Explanation instance"""
        explanation = Explanation(
            explanation_id="test_explanation",
            explanation_type=ExplanationType.MODEL_SELECTION,
            timestamp=datetime.now().isoformat(),
            ticker="AAPL",
            title="Model Selection for AAPL",
            summary="Selected LSTM model for forecasting",
            details={"model": "LSTM", "accuracy": 0.85},
            confidence_score=0.8,
            key_points=["High accuracy", "Good stability"],
            recommendations=["Monitor performance", "Update regularly"],
            tags=["model", "selection"]
        )
        
        assert explanation.explanation_id == "test_explanation"
        assert explanation.explanation_type == ExplanationType.MODEL_SELECTION
        assert explanation.ticker == "AAPL"
        assert explanation.confidence_score == 0.8
        assert len(explanation.key_points) == 2


class TestModelExplanation:
    """Test ModelExplanation dataclass"""
    
    def test_model_explanation_creation(self):
        """Test creating a ModelExplanation instance"""
        explanation = ModelExplanation(
            explanation_id="model_explanation",
            explanation_type=ExplanationType.MODEL_SELECTION,
            timestamp=datetime.now().isoformat(),
            ticker="AAPL",
            model_name="LSTM_Ensemble",
            model_type="Neural Network",
            model_version="v2.1",
            selection_criteria=["accuracy", "stability"],
            model_performance={"accuracy": 0.85, "sharpe": 1.2},
            alternative_models=["Random Forest", "XGBoost"],
            model_limitations=["Black box", "Large dataset required"]
        )
        
        assert explanation.model_name == "LSTM_Ensemble"
        assert explanation.model_type == "Neural Network"
        assert len(explanation.selection_criteria) == 2
        assert explanation.model_performance["accuracy"] == 0.85


class TestForecastExplanation:
    """Test ForecastExplanation dataclass"""
    
    def test_forecast_explanation_creation(self):
        """Test creating a ForecastExplanation instance"""
        explanation = ForecastExplanation(
            explanation_id="forecast_explanation",
            explanation_type=ExplanationType.FORECAST_REASONING,
            timestamp=datetime.now().isoformat(),
            ticker="AAPL",
            forecast_value=155.0,
            forecast_horizon=5,
            forecast_confidence=0.8,
            model_used="LSTM_Ensemble",
            features_contributing=["price", "volume"],
            market_conditions={"volatility": 0.02},
            risk_factors=["Market volatility", "Earnings uncertainty"],
            assumptions=["Stable market", "No major news"]
        )
        
        assert explanation.forecast_value == 155.0
        assert explanation.forecast_horizon == 5
        assert explanation.forecast_confidence == 0.8
        assert len(explanation.risk_factors) == 2


class TestTradeExplanation:
    """Test TradeExplanation dataclass"""
    
    def test_trade_explanation_creation(self):
        """Test creating a TradeExplanation instance"""
        explanation = TradeExplanation(
            explanation_id="trade_explanation",
            explanation_type=ExplanationType.TRADE_DECISION,
            timestamp=datetime.now().isoformat(),
            ticker="AAPL",
            trade_type="buy",
            trade_reason="Strong momentum signal",
            expected_return=0.05,
            expected_risk=0.02,
            position_size=1000.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            risk_reward_ratio=2.5,
            technical_signals=["RSI oversold", "MACD crossover"],
            fundamental_factors=["Strong earnings"]
        )
        
        assert explanation.trade_type == "buy"
        assert explanation.expected_return == 0.05
        assert explanation.position_size == 1000.0
        assert explanation.risk_reward_ratio == 2.5
        assert len(explanation.technical_signals) == 2


class TestExplainerAgent:
    """Test ExplainerAgent functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def explainer_agent(self, temp_dir):
        """Create ExplainerAgent instance for testing"""
        config = {
            'explainer': {
                'llm_enabled': False,
                'llm_model': 'gpt-3.5-turbo',
                'llm_max_tokens': 500,
                'llm_temperature': 0.7
            }
        }
        
        with patch('reporting.explainer_agent.load_config', return_value=config):
            agent = ExplainerAgent()
            return agent
    
    def test_explainer_agent_initialization(self, explainer_agent):
        """Test explainer agent initialization"""
        assert explainer_agent.llm_enabled is False
        assert explainer_agent.llm_model == 'gpt-3.5-turbo'
        assert len(explainer_agent.explanations) == 0
        assert len(explainer_agent.templates) > 0
    
    def test_explain_model_selection(self, explainer_agent):
        """Test model selection explanation"""
        explanation_id = explainer_agent.explain_model_selection(
            ticker="AAPL",
            model_name="LSTM_Ensemble",
            model_type="Neural Network",
            model_version="v2.1",
            selection_criteria=["accuracy", "stability", "interpretability"],
            model_performance={"accuracy": 0.85, "sharpe_ratio": 1.2, "stability": 0.9},
            alternative_models=["Random Forest", "XGBoost", "Prophet"],
            model_limitations=["Requires large dataset", "Black box model"],
            confidence_score=0.8
        )
        
        assert explanation_id is not None
        
        explanation = explainer_agent.get_explanation(explanation_id)
        assert explanation is not None
        assert explanation.explanation_type == ExplanationType.MODEL_SELECTION
        assert explanation.model_name == "LSTM_Ensemble"
        assert explanation.confidence_score == 0.8
        assert len(explanation.key_points) > 0
    
    def test_explain_feature_importance(self, explainer_agent):
        """Test feature importance explanation"""
        explanation_id = explainer_agent.explain_feature_importance(
            ticker="AAPL",
            features_analyzed=["rsi", "macd", "volume", "price"],
            feature_importance={"rsi": 0.4, "macd": 0.3, "volume": 0.2, "price": 0.1},
            feature_contributions={"rsi": 0.3, "macd": 0.25, "volume": 0.2, "price": 0.15},
            feature_correlations={"rsi_macd": 0.6, "volume_price": 0.4},
            feature_trends={"rsi": "increasing", "macd": "decreasing"},
            market_context={"volatility": 0.02, "trend": "bullish"}
        )
        
        assert explanation_id is not None
        
        explanation = explainer_agent.get_explanation(explanation_id)
        assert explanation is not None
        assert explanation.explanation_type == ExplanationType.FEATURE_IMPORTANCE
        assert len(explanation.features_analyzed) == 4
        assert explanation.feature_importance["rsi"] == 0.4
        assert len(explanation.key_drivers) > 0
    
    def test_explain_forecast(self, explainer_agent):
        """Test forecast explanation"""
        explanation_id = explainer_agent.explain_forecast(
            ticker="AAPL",
            forecast_value=155.0,
            forecast_horizon=5,
            forecast_confidence=0.8,
            model_used="LSTM_Ensemble",
            key_factors=["Technical momentum", "Earnings growth", "Market sentiment"],
            market_conditions={"volatility": 0.02, "trend": "bullish"},
            risk_factors=["Market volatility", "Earnings uncertainty"],
            assumptions=["Stable market conditions", "No major news events"],
            scenario_analysis={"bullish": 160.0, "bearish": 150.0, "neutral": 155.0}
        )
        
        assert explanation_id is not None
        
        explanation = explainer_agent.get_explanation(explanation_id)
        assert explanation is not None
        assert explanation.explanation_type == ExplanationType.FORECAST_REASONING
        assert explanation.forecast_value == 155.0
        assert explanation.forecast_confidence == 0.8
        assert len(explanation.risk_factors) == 2
    
    def test_explain_trade_decision(self, explainer_agent):
        """Test trade decision explanation"""
        explanation_id = explainer_agent.explain_trade_decision(
            ticker="AAPL",
            trade_type="buy",
            trade_reason="Strong momentum signal with positive forecast",
            expected_return=0.05,
            expected_risk=0.02,
            position_size=1000.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            technical_signals=["RSI oversold", "MACD crossover", "Volume spike"],
            fundamental_factors=["Strong earnings", "Market leadership"],
            market_timing="Early morning",
            risk_reward_ratio=2.5
        )
        
        assert explanation_id is not None
        
        explanation = explainer_agent.get_explanation(explanation_id)
        assert explanation is not None
        assert explanation.explanation_type == ExplanationType.TRADE_DECISION
        assert explanation.trade_type == "buy"
        assert explanation.expected_return == 0.05
        assert explanation.risk_reward_ratio == 2.5
        assert len(explanation.technical_signals) == 3
    
    def test_get_explanations_filtering(self, explainer_agent):
        """Test explanation filtering"""
        # Create multiple explanations
        explainer_agent.explain_model_selection(
            ticker="AAPL",
            model_name="LSTM",
            model_type="Neural Network",
            model_version="v1.0",
            selection_criteria=["accuracy"],
            model_performance={"accuracy": 0.8},
            alternative_models=[],
            model_limitations=[]
        )
        
        explainer_agent.explain_forecast(
            ticker="AAPL",
            forecast_value=150.0,
            forecast_horizon=1,
            forecast_confidence=0.7,
            model_used="LSTM",
            key_factors=["momentum"],
            market_conditions={},
            risk_factors=[],
            assumptions=[]
        )
        
        explainer_agent.explain_trade_decision(
            ticker="TSLA",
            trade_type="sell",
            trade_reason="Test",
            expected_return=0.02,
            expected_risk=0.01,
            position_size=500.0,
            technical_signals=[],
            fundamental_factors=[]
        )
        
        # Filter by ticker
        aapl_explanations = explainer_agent.get_explanations(ticker="AAPL")
        assert len(aapl_explanations) == 2
        
        # Filter by type
        model_explanations = explainer_agent.get_explanations(
            explanation_types=[ExplanationType.MODEL_SELECTION]
        )
        assert len(model_explanations) == 1
        
        # Filter by tags
        trade_explanations = explainer_agent.get_explanations(tags=["trade"])
        assert len(trade_explanations) == 1
    
    def test_generate_summary_report(self, explainer_agent):
        """Test summary report generation"""
        # Create some explanations
        explainer_agent.explain_model_selection(
            ticker="AAPL",
            model_name="LSTM",
            model_type="Neural Network",
            model_version="v1.0",
            selection_criteria=["accuracy"],
            model_performance={"accuracy": 0.8},
            alternative_models=[],
            model_limitations=[]
        )
        
        explainer_agent.explain_forecast(
            ticker="AAPL",
            forecast_value=150.0,
            forecast_horizon=1,
            forecast_confidence=0.7,
            model_used="LSTM",
            key_factors=["momentum"],
            market_conditions={},
            risk_factors=[],
            assumptions=[]
        )
        
        # Generate summary
        summary = explainer_agent.generate_summary_report()
        
        assert summary['summary']['total_explanations'] == 2
        assert 'AAPL' in summary['summary']['tickers_covered']
        assert summary['metrics']['explanations_by_type']['model_selection'] == 1
        assert summary['metrics']['explanations_by_type']['forecast_reasoning'] == 1
    
    def test_export_explanations(self, explainer_agent):
        """Test explanation export"""
        # Create explanation
        explainer_agent.explain_model_selection(
            ticker="AAPL",
            model_name="LSTM",
            model_type="Neural Network",
            model_version="v1.0",
            selection_criteria=["accuracy"],
            model_performance={"accuracy": 0.8},
            alternative_models=[],
            model_limitations=[]
        )
        
        # Export JSON
        json_path = explainer_agent.export_explanations(format="json")
        assert Path(json_path).exists()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'explanations' in data
        assert len(data['explanations']) == 1
        
        # Export CSV
        csv_path = explainer_agent.export_explanations(format="csv")
        assert Path(csv_path).exists()
    
    @patch('reporting.explainer_agent.openai.OpenAI')
    def test_llm_explanation_generation(self, mock_openai, explainer_agent):
        """Test LLM explanation generation"""
        # Enable LLM
        explainer_agent.llm_enabled = True
        explainer_agent.llm_client = Mock()
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is an LLM-generated explanation."
        mock_response.usage.total_tokens = 100
        explainer_agent.llm_client.chat.completions.create.return_value = mock_response
        
        # Create explanation with LLM
        explanation_id = explainer_agent.explain_model_selection(
            ticker="AAPL",
            model_name="LSTM",
            model_type="Neural Network",
            model_version="v1.0",
            selection_criteria=["accuracy"],
            model_performance={"accuracy": 0.8},
            alternative_models=[],
            model_limitations=[]
        )
        
        explanation = explainer_agent.get_explanation(explanation_id)
        assert explanation.llm_explanation is not None
        assert "LLM-generated explanation" in explanation.llm_explanation


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_audit_logger(self):
        """Test create_audit_logger function"""
        with patch('reporting.audit_logger.AuditLogger') as mock_logger:
            mock_instance = Mock()
            mock_logger.return_value = mock_instance
            
            logger = create_audit_logger()
            assert logger == mock_instance
    
    def test_create_explainer_agent(self):
        """Test create_explainer_agent function"""
        with patch('reporting.explainer_agent.ExplainerAgent') as mock_agent:
            mock_instance = Mock()
            mock_agent.return_value = mock_instance
            
            agent = create_explainer_agent()
            assert agent == mock_instance
    
    def test_log_trading_decision(self):
        """Test log_trading_decision function"""
        mock_logger = Mock()
        
        decision_data = {
            'signal_type': 'momentum',
            'signal_value': 0.75,
            'signal_strength': 0.8,
            'features_used': ['rsi'],
            'feature_importance': {'rsi': 1.0},
            'confidence_score': 0.85
        }
        
        log_trading_decision(mock_logger, "AAPL", "signal", decision_data)
        
        mock_logger.log_signal_generated.assert_called_once()
    
    def test_explain_trading_decision(self):
        """Test explain_trading_decision function"""
        mock_agent = Mock()
        
        decision_data = {
            'model_name': 'LSTM',
            'model_type': 'Neural Network',
            'model_version': 'v1.0',
            'selection_criteria': ['accuracy'],
            'model_performance': {'accuracy': 0.8},
            'alternative_models': [],
            'model_limitations': []
        }
        
        explain_trading_decision(mock_agent, "model_selection", "AAPL", decision_data)
        
        mock_agent.explain_model_selection.assert_called_once()


class TestIntegration:
    """Integration tests for audit and explainability"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_audit_and_explain_integration(self, temp_dir):
        """Test integration between audit logger and explainer agent"""
        # Create both agents
        config = {
            'audit': {
                'output_formats': ['json'],
                'batch_size': 10,
                'real_time_logging': False
            },
            'explainer': {
                'llm_enabled': False
            }
        }
        
        with patch('reporting.audit_logger.load_config', return_value=config), \
             patch('reporting.explainer_agent.load_config', return_value=config):
            
            audit_logger = AuditLogger()
            explainer_agent = ExplainerAgent()
            
            # Log a signal
            signal_id = audit_logger.log_signal_generated(
                ticker="AAPL",
                signal_type="momentum",
                signal_value=0.75,
                signal_strength=0.8,
                features_used=["rsi", "macd"],
                feature_importance={"rsi": 0.6, "macd": 0.4},
                confidence_score=0.85
            )
            
            # Explain the signal
            explanation_id = explainer_agent.explain_feature_importance(
                ticker="AAPL",
                features_analyzed=["rsi", "macd"],
                feature_importance={"rsi": 0.6, "macd": 0.4},
                feature_contributions={"rsi": 0.5, "macd": 0.3},
                feature_correlations={"rsi_macd": 0.5},
                feature_trends={"rsi": "increasing", "macd": "decreasing"},
                market_context={"volatility": 0.02}
            )
            
            # Verify both systems work together
            signal_events = audit_logger.get_events(event_types=[AuditEventType.SIGNAL_GENERATED])
            assert len(signal_events) == 1
            
            explanations = explainer_agent.get_explanations(explanation_types=[ExplanationType.FEATURE_IMPORTANCE])
            assert len(explanations) == 1
            
            # Both should reference the same ticker
            assert signal_events[0].ticker == explanations[0].ticker == "AAPL"
            
            # Close audit session
            audit_logger.close_session()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 