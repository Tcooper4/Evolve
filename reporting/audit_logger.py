"""
Audit Logger

This module provides comprehensive audit trail functionality for the Evolve trading platform:
- Records every signal, model choice, weight update, and trade decision
- Outputs JSON and CSV audit logs per session
- Tracks decision metadata, timestamps, and performance metrics
- Supports real-time logging and batch processing
- Enables compliance and explainability requirements
"""

import os
import json
import csv
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# Local imports
from utils.common_helpers import safe_json_saver, load_config
from utils.cache_utils import cache_result


class AuditEventType(Enum):
    """Types of audit events"""
    SIGNAL_GENERATED = "signal_generated"
    MODEL_SELECTED = "model_selected"
    WEIGHT_UPDATED = "weight_updated"
    FORECAST_MADE = "forecast_made"
    TRADE_DECISION = "trade_decision"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_EXECUTED = "order_executed"
    POSITION_UPDATED = "position_updated"
    RISK_CHECK = "risk_check"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    STRATEGY_ACTIVATED = "strategy_activated"
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_METRIC = "system_metric"
    ERROR_OCCURRED = "error_occurred"
    USER_ACTION = "user_action"


class DecisionLevel(Enum):
    """Decision levels for audit events"""
    SYSTEM = "system"
    STRATEGY = "strategy"
    MODEL = "model"
    TRADE = "trade"
    USER = "user"


@dataclass
class AuditEvent:
    """Audit event structure"""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    session_id: str
    decision_level: DecisionLevel
    ticker: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_score: Optional[float] = None
    risk_score: Optional[float] = None
    user_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    child_event_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class SignalEvent(AuditEvent):
    """Signal generation event"""
    signal_type: str = ""
    signal_value: float = 0.0
    signal_strength: float = 0.0
    features_used: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelEvent(AuditEvent):
    """Model selection and usage event"""
    model_name: str = ""
    model_version: str = ""
    model_type: str = ""
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    model_performance: Dict[str, float] = field(default_factory=dict)
    selection_reason: str = ""
    confidence_interval: Optional[tuple] = None


@dataclass
class WeightEvent(AuditEvent):
    """Weight update event"""
    weight_type: str = ""  # model_weight, position_weight, etc.
    old_weight: float = 0.0
    new_weight: float = 0.0
    weight_change: float = 0.0
    update_reason: str = ""
    performance_contribution: float = 0.0


@dataclass
class ForecastEvent(AuditEvent):
    """Forecast generation event"""
    forecast_horizon: int = 0
    forecast_value: float = 0.0
    forecast_confidence: float = 0.0
    forecast_interval: Optional[tuple] = None
    model_used: str = ""
    features_contributing: List[str] = field(default_factory=list)
    market_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeEvent(AuditEvent):
    """Trade decision event"""
    trade_type: str = ""  # buy, sell, hold
    trade_reason: str = ""
    trade_confidence: float = 0.0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    position_size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrderEvent(AuditEvent):
    """Order submission and execution event"""
    order_id: str = ""
    order_type: str = ""
    order_side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    executed_quantity: float = 0.0
    executed_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    execution_quality: float = 0.0


@dataclass
class RiskEvent(AuditEvent):
    """Risk management event"""
    risk_type: str = ""
    risk_level: str = ""
    risk_value: float = 0.0
    risk_threshold: float = 0.0
    risk_action: str = ""
    risk_mitigation: str = ""


class AuditLogger:
    """
    Comprehensive audit logger for trading decisions and actions
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.audit_config = self.config.get('audit', {})
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create audit directory
        self.audit_dir = Path("logs/audit")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        self.session_metadata = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'config': self.audit_config,
            'version': '1.0.0'
        }
        
        # Event storage
        self.events: List[AuditEvent] = []
        self.event_index: Dict[str, AuditEvent] = {}
        self.event_hierarchy: Dict[str, List[str]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.risk_metrics = defaultdict(list)
        self.decision_metrics = defaultdict(list)
        
        # Output configuration
        self.output_formats = self.audit_config.get('output_formats', ['json', 'csv'])
        self.batch_size = self.audit_config.get('batch_size', 100)
        self.flush_interval = self.audit_config.get('flush_interval', 60)  # seconds
        self.max_events = self.audit_config.get('max_events', 10000)
        
        # Real-time logging
        self.real_time_logging = self.audit_config.get('real_time_logging', True)
        self.log_to_console = self.audit_config.get('log_to_console', False)
        
        # Event filtering
        self.enabled_event_types = set(
            self.audit_config.get('enabled_event_types', [e.value for e in AuditEventType])
        )
        self.min_confidence_threshold = self.audit_config.get('min_confidence_threshold', 0.0)
        
        # Initialize output files
        self._initialize_output_files()
        
        # Log session start
        self.log_session_start()
    
    def _initialize_output_files(self):
        """Initialize output files for the session"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        self.output_files = {}
        
        if 'json' in self.output_formats:
            json_file = self.audit_dir / f"audit_log_{timestamp}_{self.session_id}.json"
            self.output_files['json'] = json_file
            
            # Initialize JSON file
            with open(json_file, 'w') as f:
                json.dump({
                    'session_metadata': self.session_metadata,
                    'events': []
                }, f, indent=2)
        
        if 'csv' in self.output_formats:
            csv_file = self.audit_dir / f"audit_log_{timestamp}_{self.session_id}.csv"
            self.output_files['csv'] = csv_file
            
            # Initialize CSV file with headers
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'event_id', 'event_type', 'timestamp', 'session_id', 'decision_level',
                    'ticker', 'description', 'confidence_score', 'risk_score', 'user_id',
                    'parent_event_id', 'tags', 'metadata', 'performance_metrics'
                ])
    
    def log_session_start(self):
        """Log session start event"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.SYSTEM_METRIC,
            timestamp=self.session_start.isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.SYSTEM,
            description="Audit session started",
            metadata=self.session_metadata,
            tags=['session_start', 'system']
        )
        self._add_event(event)
    
    def log_signal_generated(self, 
                           ticker: str,
                           signal_type: str,
                           signal_value: float,
                           signal_strength: float,
                           features_used: List[str],
                           feature_importance: Dict[str, float],
                           confidence_score: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log signal generation event"""
        event = SignalEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.SIGNAL_GENERATED,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.STRATEGY,
            ticker=ticker,
            description=f"Signal generated: {signal_type} for {ticker}",
            signal_type=signal_type,
            signal_value=signal_value,
            signal_strength=signal_strength,
            features_used=features_used,
            feature_importance=feature_importance,
            confidence_score=confidence_score,
            metadata=metadata or {},
            tags=['signal', 'strategy', ticker.lower()]
        )
        
        return self._add_event(event)
    
    def log_model_selected(self,
                          ticker: str,
                          model_name: str,
                          model_version: str,
                          model_type: str,
                          model_parameters: Dict[str, Any],
                          model_performance: Dict[str, float],
                          selection_reason: str,
                          confidence_interval: Optional[tuple] = None,
                          parent_event_id: Optional[str] = None) -> str:
        """Log model selection event"""
        event = ModelEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.MODEL_SELECTED,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.MODEL,
            ticker=ticker,
            description=f"Model selected: {model_name} for {ticker}",
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            model_parameters=model_parameters,
            model_performance=model_performance,
            selection_reason=selection_reason,
            confidence_interval=confidence_interval,
            parent_event_id=parent_event_id,
            tags=['model', 'selection', ticker.lower()]
        )
        
        return self._add_event(event)
    
    def log_weight_updated(self,
                          weight_type: str,
                          ticker: Optional[str] = None,
                          old_weight: float = 0.0,
                          new_weight: float = 0.0,
                          update_reason: str = "",
                          performance_contribution: float = 0.0,
                          parent_event_id: Optional[str] = None) -> str:
        """Log weight update event"""
        event = WeightEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.WEIGHT_UPDATED,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.STRATEGY,
            ticker=ticker,
            description=f"Weight updated: {weight_type}",
            weight_type=weight_type,
            old_weight=old_weight,
            new_weight=new_weight,
            weight_change=new_weight - old_weight,
            update_reason=update_reason,
            performance_contribution=performance_contribution,
            parent_event_id=parent_event_id,
            tags=['weight', 'update', weight_type.lower()]
        )
        
        return self._add_event(event)
    
    def log_forecast_made(self,
                         ticker: str,
                         forecast_horizon: int,
                         forecast_value: float,
                         forecast_confidence: float,
                         model_used: str,
                         features_contributing: List[str],
                         market_conditions: Dict[str, Any],
                         forecast_interval: Optional[tuple] = None,
                         parent_event_id: Optional[str] = None) -> str:
        """Log forecast generation event"""
        event = ForecastEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.FORECAST_MADE,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.MODEL,
            ticker=ticker,
            description=f"Forecast made: {forecast_value:.2f} for {ticker} ({forecast_horizon}d)",
            forecast_horizon=forecast_horizon,
            forecast_value=forecast_value,
            forecast_confidence=forecast_confidence,
            forecast_interval=forecast_interval,
            model_used=model_used,
            features_contributing=features_contributing,
            market_conditions=market_conditions,
            parent_event_id=parent_event_id,
            tags=['forecast', 'model', ticker.lower()]
        )
        
        return self._add_event(event)
    
    def log_trade_decision(self,
                          ticker: str,
                          trade_type: str,
                          trade_reason: str,
                          trade_confidence: float,
                          expected_return: float,
                          expected_risk: float,
                          position_size: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None,
                          risk_metrics: Optional[Dict[str, float]] = None,
                          parent_event_id: Optional[str] = None) -> str:
        """Log trade decision event"""
        event = TradeEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.TRADE_DECISION,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.TRADE,
            ticker=ticker,
            description=f"Trade decision: {trade_type} {ticker}",
            trade_type=trade_type,
            trade_reason=trade_reason,
            trade_confidence=trade_confidence,
            expected_return=expected_return,
            expected_risk=expected_risk,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_metrics=risk_metrics or {},
            parent_event_id=parent_event_id,
            tags=['trade', 'decision', trade_type.lower(), ticker.lower()]
        )
        
        return self._add_event(event)
    
    def log_order_submitted(self,
                           order_id: str,
                           ticker: str,
                           order_type: str,
                           order_side: str,
                           quantity: float,
                           price: float,
                           parent_event_id: Optional[str] = None) -> str:
        """Log order submission event"""
        event = OrderEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.ORDER_SUBMITTED,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.TRADE,
            ticker=ticker,
            description=f"Order submitted: {order_side} {quantity} {ticker} @ {price}",
            order_id=order_id,
            order_type=order_type,
            order_side=order_side,
            quantity=quantity,
            price=price,
            parent_event_id=parent_event_id,
            tags=['order', 'submission', order_side.lower(), ticker.lower()]
        )
        
        return self._add_event(event)
    
    def log_order_executed(self,
                          order_id: str,
                          ticker: str,
                          executed_quantity: float,
                          executed_price: float,
                          commission: float,
                          slippage: float,
                          execution_quality: float,
                          parent_event_id: Optional[str] = None) -> str:
        """Log order execution event"""
        event = OrderEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.ORDER_EXECUTED,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.TRADE,
            ticker=ticker,
            description=f"Order executed: {executed_quantity} {ticker} @ {executed_price}",
            order_id=order_id,
            executed_quantity=executed_quantity,
            executed_price=executed_price,
            commission=commission,
            slippage=slippage,
            execution_quality=execution_quality,
            parent_event_id=parent_event_id,
            tags=['order', 'execution', ticker.lower()]
        )
        
        return self._add_event(event)
    
    def log_risk_check(self,
                      risk_type: str,
                      risk_level: str,
                      risk_value: float,
                      risk_threshold: float,
                      risk_action: str,
                      risk_mitigation: str = "",
                      ticker: Optional[str] = None) -> str:
        """Log risk management event"""
        event = RiskEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.RISK_CHECK,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.SYSTEM,
            ticker=ticker,
            description=f"Risk check: {risk_type} - {risk_action}",
            risk_type=risk_type,
            risk_level=risk_level,
            risk_value=risk_value,
            risk_threshold=risk_threshold,
            risk_action=risk_action,
            risk_mitigation=risk_mitigation,
            tags=['risk', 'check', risk_type.lower()]
        )
        
        return self._add_event(event)
    
    def log_portfolio_rebalance(self,
                               rebalance_type: str,
                               changes: Dict[str, float],
                               reason: str,
                               performance_impact: float = 0.0) -> str:
        """Log portfolio rebalancing event"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.PORTFOLIO_REBALANCE,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.STRATEGY,
            description=f"Portfolio rebalance: {rebalance_type}",
            metadata={
                'rebalance_type': rebalance_type,
                'changes': changes,
                'reason': reason,
                'performance_impact': performance_impact
            },
            tags=['portfolio', 'rebalance', rebalance_type.lower()]
        )
        
        return self._add_event(event)
    
    def log_error(self,
                 error_type: str,
                 error_message: str,
                 error_details: Optional[Dict[str, Any]] = None,
                 ticker: Optional[str] = None) -> str:
        """Log error event"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.ERROR_OCCURRED,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.SYSTEM,
            ticker=ticker,
            description=f"Error: {error_type}",
            metadata={
                'error_type': error_type,
                'error_message': error_message,
                'error_details': error_details or {}
            },
            tags=['error', error_type.lower()]
        )
        
        return self._add_event(event)
    
    def log_user_action(self,
                       action_type: str,
                       action_description: str,
                       user_id: str,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log user action event"""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.USER_ACTION,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.USER,
            description=f"User action: {action_description}",
            user_id=user_id,
            metadata=metadata or {},
            tags=['user', 'action', action_type.lower()]
        )
        
        return self._add_event(event)
    
    def _add_event(self, event: AuditEvent) -> str:
        """Add event to audit log"""
        # Check if event type is enabled
        if event.event_type.value not in self.enabled_event_types:
            return event.event_id
        
        # Check confidence threshold
        if event.confidence_score is not None and event.confidence_score < self.min_confidence_threshold:
            return event.event_id
        
        # Add event to storage
        self.events.append(event)
        self.event_index[event.event_id] = event
        
        # Update hierarchy
        if event.parent_event_id:
            self.event_hierarchy[event.parent_event_id].append(event.event_id)
        
        # Update metrics
        self._update_metrics(event)
        
        # Real-time logging
        if self.real_time_logging:
            self._log_event_real_time(event)
        
        # Batch processing
        if len(self.events) >= self.batch_size:
            self.flush_events()
        
        # Limit events in memory
        if len(self.events) > self.max_events:
            self._prune_old_events()
        
        return event.event_id
    
    def _update_metrics(self, event: AuditEvent):
        """Update performance and risk metrics"""
        # Performance metrics
        if event.performance_metrics:
            for metric, value in event.performance_metrics.items():
                self.performance_metrics[metric].append({
                    'timestamp': event.timestamp,
                    'value': value,
                    'event_id': event.event_id
                })
        
        # Risk metrics
        if event.risk_score is not None:
            self.risk_metrics[event.event_type.value].append({
                'timestamp': event.timestamp,
                'risk_score': event.risk_score,
                'event_id': event.event_id
            })
        
        # Decision metrics
        self.decision_metrics[event.decision_level.value].append({
            'timestamp': event.timestamp,
            'event_type': event.event_type.value,
            'confidence_score': event.confidence_score,
            'event_id': event.event_id
        })
    
    def _log_event_real_time(self, event: AuditEvent):
        """Log event in real-time"""
        if self.log_to_console:
            self.logger.info(f"AUDIT: {event.event_type.value} - {event.description}")
    
    def _prune_old_events(self):
        """Remove old events to manage memory"""
        # Keep only the most recent events
        keep_count = self.max_events // 2
        self.events = self.events[-keep_count:]
        
        # Rebuild index
        self.event_index.clear()
        for event in self.events:
            self.event_index[event.event_id] = event
    
    def flush_events(self):
        """Flush events to output files"""
        try:
            # JSON output
            if 'json' in self.output_files:
                with open(self.output_files['json'], 'r') as f:
                    data = json.load(f)
                
                # Add new events
                new_events = [asdict(event) for event in self.events[-self.batch_size:]]
                data['events'].extend(new_events)
                
                # Save updated file
                safe_json_saver(str(self.output_files['json']), data)
            
            # CSV output
            if 'csv' in self.output_files:
                with open(self.output_files['csv'], 'a', newline='') as f:
                    writer = csv.writer(f)
                    
                    for event in self.events[-self.batch_size:]:
                        writer.writerow([
                            event.event_id,
                            event.event_type.value,
                            event.timestamp,
                            event.session_id,
                            event.decision_level.value,
                            event.ticker,
                            event.description,
                            event.confidence_score,
                            event.risk_score,
                            event.user_id,
                            event.parent_event_id,
                            ','.join(event.tags),
                            json.dumps(event.metadata),
                            json.dumps(event.performance_metrics)
                        ])
            
            self.logger.debug(f"Flushed {len(self.events[-self.batch_size:])} events to output files")
            
        except Exception as e:
            self.logger.error(f"Failed to flush events: {e}")
    
    def get_events(self,
                  event_types: Optional[List[AuditEventType]] = None,
                  ticker: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  decision_levels: Optional[List[DecisionLevel]] = None,
                  tags: Optional[List[str]] = None) -> List[AuditEvent]:
        """Get filtered events"""
        filtered_events = self.events
        
        # Filter by event type
        if event_types:
            filtered_events = [e for e in filtered_events if e.event_type in event_types]
        
        # Filter by ticker
        if ticker:
            filtered_events = [e for e in filtered_events if e.ticker == ticker]
        
        # Filter by time range
        if start_time:
            filtered_events = [e for e in filtered_events if datetime.fromisoformat(e.timestamp) >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if datetime.fromisoformat(e.timestamp) <= end_time]
        
        # Filter by decision level
        if decision_levels:
            filtered_events = [e for e in filtered_events if e.decision_level in decision_levels]
        
        # Filter by tags
        if tags:
            filtered_events = [e for e in filtered_events if any(tag in e.tags for tag in tags)]
        
        return filtered_events
    
    def get_event_hierarchy(self, event_id: str) -> Dict[str, Any]:
        """Get event hierarchy for a specific event"""
        event = self.event_index.get(event_id)
        if not event:
            return {}
        
        hierarchy = {
            'event': asdict(event),
            'children': [],
            'ancestors': []
        }
        
        # Get children
        children_ids = self.event_hierarchy.get(event_id, [])
        for child_id in children_ids:
            child_event = self.event_index.get(child_id)
            if child_event:
                hierarchy['children'].append(asdict(child_event))
        
        # Get ancestors
        current_event = event
        while current_event.parent_event_id:
            parent_event = self.event_index.get(current_event.parent_event_id)
            if parent_event:
                hierarchy['ancestors'].append(asdict(parent_event))
                current_event = parent_event
            else:
                break
        
        return hierarchy
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the session"""
        summary = {
            'session_id': self.session_id,
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'total_events': len(self.events),
            'events_by_type': defaultdict(int),
            'events_by_level': defaultdict(int),
            'performance_metrics': {},
            'risk_metrics': {},
            'decision_metrics': {}
        }
        
        # Count events by type and level
        for event in self.events:
            summary['events_by_type'][event.event_type.value] += 1
            summary['events_by_level'][event.decision_level.value] += 1
        
        # Calculate performance metrics
        for metric, values in self.performance_metrics.items():
            if values:
                summary['performance_metrics'][metric] = {
                    'count': len(values),
                    'mean': np.mean([v['value'] for v in values]),
                    'std': np.std([v['value'] for v in values]),
                    'min': min([v['value'] for v in values]),
                    'max': max([v['value'] for v in values])
                }
        
        # Calculate risk metrics
        for risk_type, values in self.risk_metrics.items():
            if values:
                summary['risk_metrics'][risk_type] = {
                    'count': len(values),
                    'mean_risk': np.mean([v['risk_score'] for v in values]),
                    'max_risk': max([v['risk_score'] for v in values])
                }
        
        # Calculate decision metrics
        for level, values in self.decision_metrics.items():
            if values:
                confidence_scores = [v['confidence_score'] for v in values if v['confidence_score'] is not None]
                summary['decision_metrics'][level] = {
                    'count': len(values),
                    'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0.0
                }
        
        return summary
    
    def export_session_report(self, output_path: Optional[str] = None) -> str:
        """Export comprehensive session report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.audit_dir / f"session_report_{timestamp}_{self.session_id}.json"
        
        report = {
            'session_metadata': self.session_metadata,
            'performance_summary': self.get_performance_summary(),
            'events': [asdict(event) for event in self.events],
            'event_hierarchies': {
                event_id: self.get_event_hierarchy(event_id)
                for event_id in self.event_index.keys()
            }
        }
        
        safe_json_saver(str(output_path), report)
        return str(output_path)
    
    def close_session(self):
        """Close audit session and finalize logs"""
        # Flush remaining events
        self.flush_events()
        
        # Log session end
        session_end_event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.SYSTEM_METRIC,
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            decision_level=DecisionLevel.SYSTEM,
            description="Audit session ended",
            metadata={
                'session_duration': (datetime.now() - self.session_start).total_seconds(),
                'total_events': len(self.events)
            },
            tags=['session_end', 'system']
        )
        self._add_event(session_end_event)
        
        # Final flush
        self.flush_events()
        
        # Export final report
        self.export_session_report()
        
        self.logger.info(f"Audit session {self.session_id} closed with {len(self.events)} events")


# Convenience functions
def create_audit_logger(config_path: str = "config/app_config.yaml") -> AuditLogger:
    """Create an audit logger instance"""
    return AuditLogger(config_path)


def log_trading_decision(audit_logger: AuditLogger,
                        ticker: str,
                        decision_type: str,
                        decision_data: Dict[str, Any],
                        parent_event_id: Optional[str] = None) -> str:
    """Convenience function to log trading decisions"""
    if decision_type == "signal":
        return audit_logger.log_signal_generated(
            ticker=ticker,
            signal_type=decision_data.get('signal_type', ''),
            signal_value=decision_data.get('signal_value', 0.0),
            signal_strength=decision_data.get('signal_strength', 0.0),
            features_used=decision_data.get('features_used', []),
            feature_importance=decision_data.get('feature_importance', {}),
            confidence_score=decision_data.get('confidence_score'),
            metadata=decision_data.get('metadata')
        )
    elif decision_type == "forecast":
        return audit_logger.log_forecast_made(
            ticker=ticker,
            forecast_horizon=decision_data.get('forecast_horizon', 1),
            forecast_value=decision_data.get('forecast_value', 0.0),
            forecast_confidence=decision_data.get('forecast_confidence', 0.0),
            model_used=decision_data.get('model_used', ''),
            features_contributing=decision_data.get('features_contributing', []),
            market_conditions=decision_data.get('market_conditions', {}),
            forecast_interval=decision_data.get('forecast_interval'),
            parent_event_id=parent_event_id
        )
    elif decision_type == "trade":
        return audit_logger.log_trade_decision(
            ticker=ticker,
            trade_type=decision_data.get('trade_type', ''),
            trade_reason=decision_data.get('trade_reason', ''),
            trade_confidence=decision_data.get('trade_confidence', 0.0),
            expected_return=decision_data.get('expected_return', 0.0),
            expected_risk=decision_data.get('expected_risk', 0.0),
            position_size=decision_data.get('position_size', 0.0),
            stop_loss=decision_data.get('stop_loss'),
            take_profit=decision_data.get('take_profit'),
            risk_metrics=decision_data.get('risk_metrics'),
            parent_event_id=parent_event_id
        )
    else:
        raise ValueError(f"Unknown decision type: {decision_type}")


if __name__ == "__main__":
    # Example usage
    logger = create_audit_logger()
    
    # Log some events
    signal_id = logger.log_signal_generated(
        ticker="AAPL",
        signal_type="momentum",
        signal_value=0.75,
        signal_strength=0.8,
        features_used=["rsi", "macd", "volume"],
        feature_importance={"rsi": 0.4, "macd": 0.4, "volume": 0.2},
        confidence_score=0.85
    )
    
    forecast_id = logger.log_forecast_made(
        ticker="AAPL",
        forecast_horizon=5,
        forecast_value=155.0,
        forecast_confidence=0.8,
        model_used="lstm_ensemble",
        features_contributing=["price", "volume", "sentiment"],
        market_conditions={"volatility": 0.02, "trend": "bullish"}
    )
    
    trade_id = logger.log_trade_decision(
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
    
    # Close session
    logger.close_session()
