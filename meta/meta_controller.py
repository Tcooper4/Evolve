"""
MetaControllerAgent - System Orchestrator

This agent oversees the entire Evolve trading system and makes intelligent decisions
about when to trigger various operations based on system performance, market conditions,
and configurable thresholds.

Responsibilities:
- Monitor system performance and health
- Decide when to rebuild/update models
- Trigger strategy tuning and optimization
- Schedule sentiment data collection
- Generate and distribute reports
- Maintain system logs and audit trails
- Portfolio allocation and risk management
"""

import os
import json
import time
import logging
import asyncio
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum

# Local imports
from agents.base_agent import BaseAgent
from agents.model_innovation_agent import ModelInnovationAgent
from agents.strategy_research_agent import StrategyResearchAgent
from data.sentiment.sentiment_fetcher import SentimentFetcher
from features.sentiment_features import SentimentAnalyzer
from trading.backtesting.backtest_engine import BacktestEngine
from trading.strategies.base_strategy import BaseStrategy
from portfolio import (
    PortfolioAllocator,
    PortfolioRiskManager,
    AllocationStrategy,
    AssetMetrics,
    create_allocator,
    create_risk_manager
)
from utils.cache_utils import cache_result
from utils.common_helpers import safe_json_save, load_config
from utils.weight_registry import WeightRegistry


class ActionType(Enum):
    """Types of actions the meta controller can take"""
    MODEL_REBUILD = "model_rebuild"
    STRATEGY_TUNE = "strategy_tune"
    SENTIMENT_FETCH = "sentiment_fetch"
    REPORT_GENERATE = "report_generate"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    RISK_MANAGEMENT = "risk_management"
    ALLOCATION_OPTIMIZATION = "allocation_optimization"


class TriggerCondition(Enum):
    """Conditions that can trigger actions"""
    TIME_BASED = "time_based"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MARKET_VOLATILITY = "market_volatility"
    SENTIMENT_SHIFT = "sentiment_shift"
    ERROR_THRESHOLD = "error_threshold"
    MANUAL_REQUEST = "manual_request"
    RISK_VIOLATION = "risk_violation"
    PORTFOLIO_DRIFT = "portfolio_drift"
    ALLOCATION_INEFFICIENCY = "allocation_inefficiency"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    model_performance: Dict[str, float]
    strategy_performance: Dict[str, float]
    sentiment_scores: Dict[str, float]
    system_health: Dict[str, float]
    market_conditions: Dict[str, float]
    portfolio_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    error_count: int
    active_trades: int
    portfolio_value: float


@dataclass
class ActionDecision:
    """Decision made by the meta controller"""
    action_type: ActionType
    trigger_condition: TriggerCondition
    timestamp: str
    reason: str
    priority: int  # 1-5, 5 being highest
    estimated_duration: int  # minutes
    affected_components: List[str]
    parameters: Dict[str, Any]


@dataclass
class ActionResult:
    """Result of an executed action"""
    action_id: str
    action_type: ActionType
    start_time: str
    end_time: str
    success: bool
    duration_minutes: float
    results: Dict[str, Any]
    errors: List[str]
    recommendations: List[str]


class MetaControllerAgent(BaseAgent):
    """
    Central orchestrator for the Evolve trading system
    """
    
    def __init__(self, config_path: str = "config/app_config.yaml"):
        super().__init__("MetaControllerAgent")
        
        # Load configuration
        self.config = load_config(config_path)
        self.meta_config = self.config.get('meta_controller', {})
        
        # Load trigger thresholds
        self.thresholds = self._load_trigger_thresholds()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        self.log_dir = Path("logs/meta_controller")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize system state
        self.system_metrics: List[SystemMetrics] = []
        self.action_history: List[ActionResult] = []
        self.pending_actions: List[ActionDecision] = []
        self.last_action_times: Dict[ActionType, datetime] = {}
        
        # Initialize sub-agents
        self.model_agent = None
        self.strategy_agent = None
        self.sentiment_fetcher = None
        self.sentiment_analyzer = None
        self.backtester = None
        self.weight_registry = None
        
        # Initialize portfolio components
        self.portfolio_allocator = None
        self.portfolio_risk_manager = None
        
        self._initialize_sub_agents()
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {
            'model_accuracy': [],
            'strategy_returns': [],
            'sentiment_correlation': [],
            'system_uptime': [],
            'portfolio_sharpe': [],
            'portfolio_drawdown': []
        }
        
        # Market state tracking
        self.market_state = {
            'volatility': 0.0,
            'trend': 'neutral',
            'sentiment': 0.0,
            'volume': 0.0
        }
        
        # Portfolio state tracking
        self.portfolio_state = {
            'current_allocation': {},
            'target_allocation': {},
            'risk_violations': [],
            'last_rebalance': datetime.now(),
            'allocation_strategy': 'maximum_sharpe'
        }
        
        # Action scheduling
        self.scheduler = schedule.Scheduler()
        self._setup_scheduled_actions()
        
        # Health monitoring
        self.health_metrics = {
            'last_health_check': datetime.now(),
            'system_status': 'healthy',
            'error_rate': 0.0,
            'response_time': 0.0
        }
    
    def _load_trigger_thresholds(self) -> Dict[str, Any]:
        """Load trigger thresholds from JSON file"""
        thresholds_path = Path("config/trigger_thresholds.json")
        
        if thresholds_path.exists():
            try:
                with open(thresholds_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load trigger thresholds: {e}")
        
        # Default thresholds
        return {
            "model_rebuild": {
                "performance_threshold": 0.6,
                "time_threshold_hours": 24,
                "error_threshold": 0.1,
                "market_volatility_threshold": 0.3
            },
            "strategy_tune": {
                "performance_threshold": 0.5,
                "time_threshold_hours": 12,
                "drawdown_threshold": 0.1,
                "sharpe_threshold": 1.0
            },
            "sentiment_fetch": {
                "time_threshold_hours": 1,
                "sentiment_volatility_threshold": 0.2,
                "market_impact_threshold": 0.1
            },
            "report_generation": {
                "time_threshold_hours": 24,
                "performance_change_threshold": 0.05,
                "error_rate_threshold": 0.05
            },
            "system_health_check": {
                "time_threshold_hours": 1,
                "error_rate_threshold": 0.1,
                "performance_threshold": 0.8
            },
            "portfolio_rebalance": {
                "time_threshold_hours": 24,
                "drift_threshold": 0.05,
                "risk_violation_threshold": 1,
                "performance_threshold": 0.6
            },
            "risk_management": {
                "drawdown_threshold": 0.15,
                "volatility_threshold": 0.25,
                "var_threshold": 0.02,
                "exposure_threshold": 0.3
            },
            "allocation_optimization": {
                "sharpe_degradation_threshold": 0.1,
                "time_threshold_hours": 168,  # 1 week
                "market_regime_change_threshold": 0.2
            }
        }
    
    def _initialize_sub_agents(self):
        """Initialize sub-agents and components"""
        try:
            # Initialize existing agents
            self.model_agent = ModelInnovationAgent()
            self.strategy_agent = StrategyResearchAgent()
            self.sentiment_fetcher = SentimentFetcher()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.backtester = BacktestEngine()
            self.weight_registry = WeightRegistry()
            
            # Initialize portfolio components
            self.portfolio_allocator = create_allocator()
            self.portfolio_risk_manager = create_risk_manager()
            
            self.logger.info("All sub-agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sub-agents: {e}")
            raise
    
    def _setup_scheduled_actions(self):
        """Setup scheduled actions"""
        # Existing scheduled actions
        self.scheduler.every().hour.do(self._health_check)
        self.scheduler.every().day.at("09:00").do(self._daily_optimization)
        self.scheduler.every().week.do(self._weekly_analysis)
        
        # Portfolio-specific scheduled actions
        self.scheduler.every().day.at("16:00").do(self._daily_portfolio_check)
        self.scheduler.every().week.do(self._weekly_allocation_optimization)
        self.scheduler.every(4).hours.do(self._risk_monitoring)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics including portfolio metrics"""
        try:
            # Collect existing metrics
            model_performance = self._get_model_performance()
            strategy_performance = self._get_strategy_performance()
            sentiment_scores = self._get_sentiment_metrics()
            system_health = self._get_system_health()
            market_conditions = self._get_market_conditions()
            
            # Collect portfolio metrics
            portfolio_metrics = self._get_portfolio_metrics()
            risk_metrics = self._get_risk_metrics()
            
            # Collect other metrics
            error_count = self._get_error_count()
            active_trades = self._get_active_trades()
            portfolio_value = self._get_portfolio_value()
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                model_performance=model_performance,
                strategy_performance=strategy_performance,
                sentiment_scores=sentiment_scores,
                system_health=system_health,
                market_conditions=market_conditions,
                portfolio_metrics=portfolio_metrics,
                risk_metrics=risk_metrics,
                error_count=error_count,
                active_trades=active_trades,
                portfolio_value=portfolio_value
            )
            
            self.system_metrics.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def _get_portfolio_metrics(self) -> Dict[str, float]:
        """Get portfolio performance metrics"""
        try:
            # This would typically get data from your portfolio tracking system
            # For now, return sample metrics
            return {
                'total_return': 0.08,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.05,
                'volatility': 0.15,
                'diversification_ratio': 1.3,
                'allocation_drift': 0.02
            }
        except Exception as e:
            self.logger.error(f"Failed to get portfolio metrics: {e}")
            return {}
    
    def _get_risk_metrics(self) -> Dict[str, float]:
        """Get portfolio risk metrics"""
        try:
            # This would typically get data from your risk management system
            # For now, return sample metrics
            return {
                'var_95': -0.015,
                'cvar_95': -0.025,
                'current_drawdown': -0.03,
                'exposure_concentration': 0.25,
                'sector_concentration': 0.35,
                'correlation_risk': 0.45
            }
        except Exception as e:
            self.logger.error(f"Failed to get risk metrics: {e}")
            return {}
    
    def evaluate_triggers(self, metrics: SystemMetrics) -> List[ActionDecision]:
        """Evaluate all triggers and generate action decisions"""
        decisions = []
        
        try:
            # Evaluate existing triggers
            model_rebuild = self._evaluate_model_rebuild(metrics)
            if model_rebuild:
                decisions.append(model_rebuild)
            
            strategy_tune = self._evaluate_strategy_tune(metrics)
            if strategy_tune:
                decisions.append(strategy_tune)
            
            sentiment_fetch = self._evaluate_sentiment_fetch(metrics)
            if sentiment_fetch:
                decisions.append(sentiment_fetch)
            
            report_generation = self._evaluate_report_generation(metrics)
            if report_generation:
                decisions.append(report_generation)
            
            system_health = self._evaluate_system_health(metrics)
            if system_health:
                decisions.append(system_health)
            
            # Evaluate portfolio-specific triggers
            portfolio_rebalance = self._evaluate_portfolio_rebalance(metrics)
            if portfolio_rebalance:
                decisions.append(portfolio_rebalance)
            
            risk_management = self._evaluate_risk_management(metrics)
            if risk_management:
                decisions.append(risk_management)
            
            allocation_optimization = self._evaluate_allocation_optimization(metrics)
            if allocation_optimization:
                decisions.append(allocation_optimization)
            
            # Sort by priority
            decisions.sort(key=lambda x: x.priority, reverse=True)
            
            return decisions
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate triggers: {e}")
            return []
    
    def _evaluate_portfolio_rebalance(self, metrics: SystemMetrics) -> Optional[ActionDecision]:
        """Evaluate if portfolio rebalancing is needed"""
        try:
            thresholds = self.thresholds.get('portfolio_rebalance', {})
            
            # Check time-based trigger
            last_rebalance = self.portfolio_state.get('last_rebalance', datetime.now() - timedelta(days=30))
            time_threshold = thresholds.get('time_threshold_hours', 24)
            
            if datetime.now() - last_rebalance > timedelta(hours=time_threshold):
                return ActionDecision(
                    action_type=ActionType.PORTFOLIO_REBALANCE,
                    trigger_condition=TriggerCondition.TIME_BASED,
                    timestamp=datetime.now().isoformat(),
                    reason=f"Portfolio rebalancing due after {time_threshold} hours",
                    priority=3,
                    estimated_duration=30,
                    affected_components=['portfolio', 'risk_manager'],
                    parameters={'rebalance_type': 'scheduled'}
                )
            
            # Check drift-based trigger
            drift_threshold = thresholds.get('drift_threshold', 0.05)
            current_drift = metrics.portfolio_metrics.get('allocation_drift', 0)
            
            if current_drift > drift_threshold:
                return ActionDecision(
                    action_type=ActionType.PORTFOLIO_REBALANCE,
                    trigger_condition=TriggerCondition.PORTFOLIO_DRIFT,
                    timestamp=datetime.now().isoformat(),
                    reason=f"Portfolio drift {current_drift:.2%} exceeds threshold {drift_threshold:.2%}",
                    priority=4,
                    estimated_duration=30,
                    affected_components=['portfolio', 'risk_manager'],
                    parameters={'rebalance_type': 'drift_correction'}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate portfolio rebalance: {e}")
            return None
    
    def _evaluate_risk_management(self, metrics: SystemMetrics) -> Optional[ActionDecision]:
        """Evaluate if risk management actions are needed"""
        try:
            thresholds = self.thresholds.get('risk_management', {})
            
            # Check drawdown threshold
            drawdown_threshold = thresholds.get('drawdown_threshold', 0.15)
            current_drawdown = abs(metrics.risk_metrics.get('current_drawdown', 0))
            
            if current_drawdown > drawdown_threshold:
                return ActionDecision(
                    action_type=ActionType.RISK_MANAGEMENT,
                    trigger_condition=TriggerCondition.RISK_VIOLATION,
                    timestamp=datetime.now().isoformat(),
                    reason=f"Drawdown {current_drawdown:.2%} exceeds threshold {drawdown_threshold:.2%}",
                    priority=5,
                    estimated_duration=15,
                    affected_components=['portfolio', 'risk_manager'],
                    parameters={'action_type': 'drawdown_protection'}
                )
            
            # Check volatility threshold
            volatility_threshold = thresholds.get('volatility_threshold', 0.25)
            current_volatility = metrics.portfolio_metrics.get('volatility', 0)
            
            if current_volatility > volatility_threshold:
                return ActionDecision(
                    action_type=ActionType.RISK_MANAGEMENT,
                    trigger_condition=TriggerCondition.RISK_VIOLATION,
                    timestamp=datetime.now().isoformat(),
                    reason=f"Volatility {current_volatility:.2%} exceeds threshold {volatility_threshold:.2%}",
                    priority=4,
                    estimated_duration=20,
                    affected_components=['portfolio', 'risk_manager'],
                    parameters={'action_type': 'volatility_reduction'}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate risk management: {e}")
            return None
    
    def _evaluate_allocation_optimization(self, metrics: SystemMetrics) -> Optional[ActionDecision]:
        """Evaluate if allocation optimization is needed"""
        try:
            thresholds = self.thresholds.get('allocation_optimization', {})
            
            # Check Sharpe ratio degradation
            sharpe_threshold = thresholds.get('sharpe_degradation_threshold', 0.1)
            current_sharpe = metrics.portfolio_metrics.get('sharpe_ratio', 0)
            target_sharpe = 1.5  # This would come from historical performance
            
            if target_sharpe - current_sharpe > sharpe_threshold:
                return ActionDecision(
                    action_type=ActionType.ALLOCATION_OPTIMIZATION,
                    trigger_condition=TriggerCondition.ALLOCATION_INEFFICIENCY,
                    timestamp=datetime.now().isoformat(),
                    reason=f"Sharpe ratio degradation {target_sharpe - current_sharpe:.2f} exceeds threshold {sharpe_threshold:.2f}",
                    priority=3,
                    estimated_duration=60,
                    affected_components=['portfolio', 'allocator'],
                    parameters={'optimization_type': 'sharpe_improvement'}
                )
            
            # Check time-based optimization
            last_optimization = self.portfolio_state.get('last_optimization', datetime.now() - timedelta(days=30))
            time_threshold = thresholds.get('time_threshold_hours', 168)
            
            if datetime.now() - last_optimization > timedelta(hours=time_threshold):
                return ActionDecision(
                    action_type=ActionType.ALLOCATION_OPTIMIZATION,
                    trigger_condition=TriggerCondition.TIME_BASED,
                    timestamp=datetime.now().isoformat(),
                    reason=f"Allocation optimization due after {time_threshold} hours",
                    priority=2,
                    estimated_duration=60,
                    affected_components=['portfolio', 'allocator'],
                    parameters={'optimization_type': 'scheduled'}
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate allocation optimization: {e}")
            return None
    
    async def execute_action(self, decision: ActionDecision) -> ActionResult:
        """Execute an action decision"""
        action_id = f"{decision.action_type.value}_{int(time.time())}"
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing action: {decision.action_type.value}")
            
            # Execute based on action type
            if decision.action_type == ActionType.MODEL_REBUILD:
                results = await self._execute_model_rebuild(decision)
            elif decision.action_type == ActionType.STRATEGY_TUNE:
                results = await self._execute_strategy_tune(decision)
            elif decision.action_type == ActionType.SENTIMENT_FETCH:
                results = await self._execute_sentiment_fetch(decision)
            elif decision.action_type == ActionType.REPORT_GENERATE:
                results = await self._execute_report_generation(decision)
            elif decision.action_type == ActionType.SYSTEM_HEALTH_CHECK:
                results = await self._execute_system_health_check(decision)
            elif decision.action_type == ActionType.PORTFOLIO_REBALANCE:
                results = await self._execute_portfolio_rebalance(decision)
            elif decision.action_type == ActionType.RISK_MANAGEMENT:
                results = await self._execute_risk_management(decision)
            elif decision.action_type == ActionType.ALLOCATION_OPTIMIZATION:
                results = await self._execute_allocation_optimization(decision)
            else:
                results = {"error": f"Unknown action type: {decision.action_type}"}
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            # Create action result
            action_result = ActionResult(
                action_id=action_id,
                action_type=decision.action_type,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                success=not results.get('error'),
                duration_minutes=duration,
                results=results,
                errors=results.get('errors', []),
                recommendations=results.get('recommendations', [])
            )
            
            # Log and store result
            self._log_action_result(action_result)
            self.action_history.append(action_result)
            
            # Update last action time
            self.last_action_times[decision.action_type] = end_time
            
            return action_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute action {decision.action_type.value}: {e}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            return ActionResult(
                action_id=action_id,
                action_type=decision.action_type,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                success=False,
                duration_minutes=duration,
                results={},
                errors=[str(e)],
                recommendations=["Check system logs for details"]
            )
    
    async def _execute_model_rebuild(self, decision: ActionDecision) -> Dict[str, Any]:
        """Execute model rebuild action"""
        try:
            if self.model_agent:
                # Run model innovation process
                result = self.model_agent.run()
                
                return {
                    'success': result.get('status') == 'success',
                    'results': result,
                    'errors': [],
                    'recommendations': [
                        'Monitor new model performance',
                        'Update model weights if needed'
                    ]
                }
            else:
                return {
                    'success': False,
                    'results': {},
                    'errors': ['Model agent not available'],
                    'recommendations': ['Initialize model agent']
                }
        except Exception as e:
            return {
                'success': False,
                'results': {},
                'errors': [str(e)],
                'recommendations': ['Check model agent configuration']
            }
    
    async def _execute_strategy_tune(self, decision: ActionDecision) -> Dict[str, Any]:
        """Execute strategy tuning action"""
        try:
            if self.strategy_agent:
                # Run strategy research and testing
                result = self.strategy_agent.run()
                
                return {
                    'success': result.get('status') == 'success',
                    'results': result,
                    'errors': [],
                    'recommendations': [
                        'Test new strategies in backtest',
                        'Monitor strategy performance'
                    ]
                }
            else:
                return {
                    'success': False,
                    'results': {},
                    'errors': ['Strategy agent not available'],
                    'recommendations': ['Initialize strategy agent']
                }
        except Exception as e:
            return {
                'success': False,
                'results': {},
                'errors': [str(e)],
                'recommendations': ['Check strategy agent configuration']
            }
    
    async def _execute_sentiment_fetch(self, decision: ActionDecision) -> Dict[str, Any]:
        """Execute sentiment fetch action"""
        try:
            if self.sentiment_fetcher:
                # Fetch sentiment for major tickers
                tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
                results = {}
                
                for ticker in tickers:
                    try:
                        sentiment_data = self.sentiment_fetcher.fetch_all_sentiment(ticker, hours_back=6)
                        results[ticker] = {
                            'news_count': len(sentiment_data.get('news', [])),
                            'reddit_count': len(sentiment_data.get('reddit', [])),
                            'twitter_count': len(sentiment_data.get('twitter', []))
                        }
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch sentiment for {ticker}: {e}")
                
                return {
                    'success': True,
                    'results': results,
                    'errors': [],
                    'recommendations': [
                        'Process sentiment data for features',
                        'Update sentiment-based signals'
                    ]
                }
            else:
                return {
                    'success': False,
                    'results': {},
                    'errors': ['Sentiment fetcher not available'],
                    'recommendations': ['Initialize sentiment fetcher']
                }
        except Exception as e:
            return {
                'success': False,
                'results': {},
                'errors': [str(e)],
                'recommendations': ['Check sentiment fetcher configuration']
            }
    
    async def _execute_report_generation(self, decision: ActionDecision) -> Dict[str, Any]:
        """Execute report generation action"""
        try:
            # Generate comprehensive system report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': self._get_latest_metrics(),
                'action_history': self._get_recent_actions(),
                'performance_summary': self._get_performance_summary(),
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            report_filename = f"system_report_{int(time.time())}.json"
            report_path = self.log_dir / report_filename
            safe_json_save(str(report_path), report_data)
            
            return {
                'success': True,
                'results': {'report_path': str(report_path)},
                'errors': [],
                'recommendations': [
                    'Review system performance',
                    'Consider manual interventions if needed'
                ]
            }
        except Exception as e:
            return {
                'success': False,
                'results': {},
                'errors': [str(e)],
                'recommendations': ['Check report generation configuration']
            }
    
    async def _execute_system_health_check(self, decision: ActionDecision) -> Dict[str, Any]:
        """Execute system health check action"""
        try:
            # Perform comprehensive health check
            health_results = {
                'system_status': 'healthy',
                'components': {},
                'issues': [],
                'recommendations': []
            }
            
            # Check each component
            components = ['models', 'strategies', 'sentiment', 'backtester']
            
            for component in components:
                try:
                    status = await self._check_component_health(component)
                    health_results['components'][component] = status
                    
                    if status['status'] != 'healthy':
                        health_results['issues'].append(f"{component}: {status['message']}")
                except Exception as e:
                    health_results['components'][component] = {
                        'status': 'error',
                        'message': str(e)
                    }
                    health_results['issues'].append(f"{component}: {str(e)}")
            
            # Update overall status
            if health_results['issues']:
                health_results['system_status'] = 'degraded'
                health_results['recommendations'].append('Address identified issues')
            else:
                health_results['recommendations'].append('System operating normally')
            
            return {
                'success': True,
                'results': health_results,
                'errors': [],
                'recommendations': health_results['recommendations']
            }
        except Exception as e:
            return {
                'success': False,
                'results': {},
                'errors': [str(e)],
                'recommendations': ['Check system configuration']
            }
    
    async def _check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of a specific component"""
        try:
            if component == 'models':
                if self.weight_registry:
                    weights = self.weight_registry.get_current_weights()
                    if weights:
                        return {'status': 'healthy', 'message': f'{len(weights)} models active'}
                    else:
                        return {'status': 'warning', 'message': 'No models found'}
                else:
                    return {'status': 'error', 'message': 'Weight registry not available'}
            
            elif component == 'strategies':
                if self.strategy_agent:
                    return {'status': 'healthy', 'message': 'Strategy agent active'}
                else:
                    return {'status': 'error', 'message': 'Strategy agent not available'}
            
            elif component == 'sentiment':
                if self.sentiment_fetcher and self.sentiment_analyzer:
                    return {'status': 'healthy', 'message': 'Sentiment components active'}
                else:
                    return {'status': 'error', 'message': 'Sentiment components not available'}
            
            elif component == 'backtester':
                if self.backtester:
                    return {'status': 'healthy', 'message': 'Backtester active'}
                else:
                    return {'status': 'error', 'message': 'Backtester not available'}
            
            else:
                return {'status': 'unknown', 'message': f'Unknown component: {component}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics"""
        return self.system_metrics[-1] if self.system_metrics else None
    
    def _get_recent_actions(self, hours: int = 24) -> List[ActionResult]:
        """Get recent action history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            action for action in self.action_history
            if datetime.fromisoformat(action.start_time) > cutoff_time
        ]
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.system_metrics:
            return {}
        
        recent_metrics = self.system_metrics[-100:]  # Last 100 metrics
        
        summary = {
            'model_performance': {},
            'strategy_performance': {},
            'system_health': {},
            'action_success_rate': 0.0
        }
        
        # Calculate averages
        if recent_metrics:
            # Model performance
            all_model_perf = []
            for metrics in recent_metrics:
                all_model_perf.extend(metrics.model_performance.values())
            if all_model_perf:
                summary['model_performance']['average'] = np.mean(all_model_perf)
                summary['model_performance']['trend'] = 'stable'  # Simplified
            
            # Strategy performance
            all_strategy_perf = []
            for metrics in recent_metrics:
                all_strategy_perf.extend(metrics.strategy_performance.values())
            if all_strategy_perf:
                summary['strategy_performance']['average'] = np.mean(all_strategy_perf)
                summary['strategy_performance']['trend'] = 'stable'  # Simplified
        
        # Action success rate
        recent_actions = self._get_recent_actions(24)
        if recent_actions:
            success_count = sum(1 for action in recent_actions if action.success)
            summary['action_success_rate'] = success_count / len(recent_actions)
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Check performance trends
        if self.system_metrics:
            recent_metrics = self.system_metrics[-10:]
            if len(recent_metrics) >= 2:
                # Check for performance degradation
                latest_perf = np.mean(list(recent_metrics[-1].model_performance.values())) if recent_metrics[-1].model_performance else 0
                prev_perf = np.mean(list(recent_metrics[-2].model_performance.values())) if recent_metrics[-2].model_performance else 0
                
                if latest_perf < prev_perf * 0.9:  # 10% degradation
                    recommendations.append("Consider model retraining due to performance degradation")
        
        # Check error rates
        recent_actions = self._get_recent_actions(1)  # Last hour
        if recent_actions:
            error_rate = sum(1 for action in recent_actions if not action.success) / len(recent_actions)
            if error_rate > 0.2:  # 20% error rate
                recommendations.append("High error rate detected - investigate system issues")
        
        # Check system resources
        latest_metrics = self._get_latest_metrics()
        if latest_metrics and latest_metrics.system_health:
            cpu_usage = latest_metrics.system_health.get('cpu_usage', 0)
            memory_usage = latest_metrics.system_health.get('memory_usage', 0)
            
            if cpu_usage > 80:
                recommendations.append("High CPU usage - consider optimization")
            if memory_usage > 80:
                recommendations.append("High memory usage - consider cleanup")
        
        if not recommendations:
            recommendations.append("System operating normally - no immediate action required")
        
        return recommendations
    
    def _log_action_result(self, action_result: ActionResult):
        """Log action result to file"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action_id': action_result.action_id,
                'action_type': action_result.action_type.value,
                'success': action_result.success,
                'duration_minutes': action_result.duration_minutes,
                'errors': action_result.errors,
                'recommendations': action_result.recommendations
            }
            
            log_filename = f"action_log_{datetime.now().strftime('%Y%m%d')}.json"
            log_path = self.log_dir / log_filename
            
            # Append to existing log or create new
            if log_path.exists():
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            log_data.append(log_entry)
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to log action result: {e}")
    
    def _health_check(self):
        """Scheduled health check"""
        try:
            metrics = self.collect_system_metrics()
            if metrics:
                decisions = self.evaluate_triggers(metrics)
                
                for decision in decisions:
                    if decision.priority >= 4:  # High priority actions
                        asyncio.create_task(self.execute_action(decision))
            
            self.health_metrics['last_health_check'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def _monitor_performance(self):
        """Scheduled performance monitoring"""
        try:
            metrics = self.collect_system_metrics()
            if metrics:
                # Update performance history
                if metrics.model_performance:
                    avg_perf = np.mean(list(metrics.model_performance.values()))
                    self.performance_history['model_accuracy'].append(avg_perf)
                
                # Keep only last 1000 entries
                for key in self.performance_history:
                    if len(self.performance_history[key]) > 1000:
                        self.performance_history[key] = self.performance_history[key][-1000:]
            
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
    
    def _update_market_state(self):
        """Scheduled market state update"""
        try:
            # Update market state based on latest metrics
            latest_metrics = self._get_latest_metrics()
            if latest_metrics and latest_metrics.market_conditions:
                self.market_state['volatility'] = latest_metrics.market_conditions.get('volatility', 0)
                self.market_state['trend'] = 'bullish' if latest_metrics.market_conditions.get('trend_strength', 0) > 0.5 else 'bearish'
                self.market_state['volume'] = latest_metrics.market_conditions.get('volume_ratio', 1.0)
            
        except Exception as e:
            self.logger.error(f"Market state update failed: {e}")
    
    def _rotate_logs(self):
        """Scheduled log rotation"""
        try:
            # Archive old logs (older than 30 days)
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for log_file in self.log_dir.glob("*.json"):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        # Move to archive directory
                        archive_dir = self.log_dir / "archive"
                        archive_dir.mkdir(exist_ok=True)
                        
                        log_file.rename(archive_dir / log_file.name)
                        self.logger.info(f"Archived old log: {log_file.name}")
                except Exception as e:
                    self.logger.error(f"Failed to archive {log_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Log rotation failed: {e}")
    
    def run_scheduler(self):
        """Run the action scheduler"""
        self.logger.info("Starting meta controller scheduler")
        
        while True:
            try:
                self.scheduler.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                self.logger.info("Stopping meta controller scheduler")
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Main execution method
        """
        try:
            # Collect initial metrics
            metrics = self.collect_system_metrics()
            
            # Evaluate triggers
            decisions = self.evaluate_triggers(metrics)
            
            # Execute high-priority actions
            executed_actions = []
            for decision in decisions:
                if decision.priority >= 3:  # Medium and high priority
                    result = asyncio.run(self.execute_action(decision))
                    executed_actions.append(result)
            
            # Start scheduler in background
            scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            scheduler_thread.start()
            
            return {
                'status': 'success',
                'metrics_collected': metrics is not None,
                'decisions_made': len(decisions),
                'actions_executed': len(executed_actions),
                'scheduler_started': True,
                'system_status': self.health_metrics['system_status']
            }
            
        except Exception as e:
            self.logger.error(f"Meta controller failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# Convenience functions
def create_meta_controller(config_path: str = "config/app_config.yaml") -> MetaControllerAgent:
    """Create a meta controller instance"""
    return MetaControllerAgent(config_path)


def run_meta_controller(config_path: str = "config/app_config.yaml") -> Dict[str, Any]:
    """Quick function to run meta controller"""
    controller = MetaControllerAgent(config_path)
    return controller.run()


if __name__ == "__main__":
    # Example usage
    controller = MetaControllerAgent()
    
    # Run meta controller
    results = controller.run()
    print(json.dumps(results, indent=2))
    
    # Keep running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping meta controller...")
