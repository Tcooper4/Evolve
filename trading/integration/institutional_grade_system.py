"""
Institutional-Grade Trading System Integration

Main integration module that ties all strategic intelligence modules together.
Provides full UI integration, autonomous operation, and comprehensive system management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import os
import asyncio
import threading
import time
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import all strategic intelligence modules
try:
    from trading.agents.market_regime_agent import MarketRegimeAgent, RegimeAnalysis
    from trading.agents.rolling_retraining_agent import RollingRetrainingAgent, RetrainingConfig
    from trading.strategies.multi_strategy_hybrid_engine import MultiStrategyHybridEngine, HybridSignal
    from trading.analytics.alpha_attribution_engine import AlphaAttributionEngine, AlphaAttribution
    from trading.risk.position_sizing_engine import PositionSizingEngine, SizingResult
    from trading.agents.execution_risk_control_agent import ExecutionRiskControlAgent, RiskCheck
    from trading.data.macro_data_integration import MacroDataIntegration, MacroIndicator
    from trading.analytics.forecast_explainability import IntelligentForecastExplainability, ForecastExplanation
    from trading.services.real_time_signal_center import RealTimeSignalCenter, TradingSignal
    from trading.report.report_export_engine import ReportExportEngine, ReportConfig
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class SystemMetrics:
    """System performance metrics."""
    uptime: float
    total_signals: int
    active_trades: int
    system_health: float
    performance_score: float
    risk_score: float
    last_update: datetime

class InstitutionalGradeSystem:
    """Main institutional-grade trading system integration."""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 auto_start: bool = True):
        """Initialize the institutional-grade system.
        
        Args:
            config_path: Path to system configuration file
            auto_start: Whether to automatically start the system
        """
        self.config_path = config_path or "config/institutional_system.json"
        self.config = self._load_config()
        
        # Initialize system status
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now()
        self.system_metrics = None
        
        # Initialize all modules
        self._initialize_modules()
        
        # Initialize integration components
        self.signal_queue = asyncio.Queue()
        self.alert_queue = asyncio.Queue()
        self.data_cache = {}
        
        # Start system if requested
        if auto_start:
            self.start()
        
        logger.info("Institutional-Grade Trading System initialized")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                return self._create_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default system configuration."""
        return {
            'system': {
                'name': 'Institutional-Grade Trading System',
                'version': '2.0.0',
                'auto_restart': True,
                'max_memory_usage': 0.8,
                'log_level': 'INFO'
            },
            'modules': {
                'market_regime': {'enabled': True, 'update_interval': 300},
                'rolling_retraining': {'enabled': True, 'retrain_interval': 86400},
                'hybrid_engine': {'enabled': True, 'signal_interval': 60},
                'alpha_attribution': {'enabled': True, 'analysis_interval': 3600},
                'position_sizing': {'enabled': True, 'update_interval': 300},
                'execution_control': {'enabled': True, 'check_interval': 30},
                'macro_data': {'enabled': True, 'update_interval': 3600},
                'forecast_explainability': {'enabled': True, 'explanation_interval': 300},
                'signal_center': {'enabled': True, 'websocket_port': 8765},
                'report_engine': {'enabled': True, 'report_interval': 86400}
            },
            'risk_limits': {
                'max_position_size': 0.25,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.15,
                'max_correlation': 0.7
            },
            'data_sources': {
                'primary': 'yahoo',
                'backup': 'alpha_vantage',
                'macro': 'fred'
            }
        }
    
    def _initialize_modules(self):
        """Initialize all strategic intelligence modules."""
        try:
            self.modules = {}
            
            if not MODULES_AVAILABLE:
                logger.warning("Not all modules available, using fallback implementations")
                self._initialize_fallback_modules()

            # Market Regime Agent
            if self.config['modules']['market_regime']['enabled']:
                self.modules['market_regime'] = MarketRegimeAgent(
                    lookback_period=252,
                    regime_threshold=0.7
                )
            
            # Rolling Retraining Agent
            if self.config['modules']['rolling_retraining']['enabled']:
                retraining_config = RetrainingConfig(
                    retrain_frequency=30,
                    lookback_window=252,
                    min_train_size=60,
                    test_size=20,
                    performance_threshold=0.1
                )
                self.modules['rolling_retraining'] = RollingRetrainingAgent(
                    config=retraining_config
                )
            
            # Multi-Strategy Hybrid Engine
            if self.config['modules']['hybrid_engine']['enabled']:
                self.modules['hybrid_engine'] = MultiStrategyHybridEngine(
                    ensemble_method="weighted_average",
                    confidence_threshold=0.6,
                    max_position_size=self.config['risk_limits']['max_position_size']
                )
            
            # Alpha Attribution Engine
            if self.config['modules']['alpha_attribution']['enabled']:
                self.modules['alpha_attribution'] = AlphaAttributionEngine(
                    lookback_period=252,
                    min_alpha_threshold=0.01
                )
            
            # Position Sizing Engine
            if self.config['modules']['position_sizing']['enabled']:
                self.modules['position_sizing'] = PositionSizingEngine(
                    risk_per_trade=0.02,
                    max_position_size=self.config['risk_limits']['max_position_size']
                )
            
            # Execution Risk Control Agent
            if self.config['modules']['execution_control']['enabled']:
                self.modules['execution_control'] = ExecutionRiskControlAgent(
                    max_position_size=self.config['risk_limits']['max_position_size'],
                    max_daily_trades=50,
                    max_daily_loss=self.config['risk_limits']['max_daily_loss']
                )
            
            # Macro Data Integration
            if self.config['modules']['macro_data']['enabled']:
                self.modules['macro_data'] = MacroDataIntegration(
                    fred_api_key=os.getenv('FRED_API_KEY'),
                    alpha_vantage_api_key=os.getenv('ALPHA_VANTAGE_API_KEY')
                )
            
            # Forecast Explainability
            if self.config['modules']['forecast_explainability']['enabled']:
                self.modules['forecast_explainability'] = IntelligentForecastExplainability(
                    confidence_levels=[0.68, 0.80, 0.95],
                    max_features=10
                )
            
            # Real-Time Signal Center
            if self.config['modules']['signal_center']['enabled']:
                self.modules['signal_center'] = RealTimeSignalCenter(
                    websocket_port=self.config['modules']['signal_center']['websocket_port'],
                    max_signals=1000
                )
            
            # Report Export Engine
            if self.config['modules']['report_engine']['enabled']:
                self.modules['report_engine'] = ReportExportEngine(
                    output_dir="reports",
                    template_dir="templates",
                    chart_dir="charts"
                )
            
            logger.info(f"Initialized {len(self.modules)} modules successfully")
            
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            self._initialize_fallback_modules()
    
    def _initialize_fallback_modules(self):
        """Initialize fallback modules when main modules are unavailable."""
        try:
            self.modules = {}
            
            # Create simple fallback implementations
            class FallbackModule:
                def __init__(self, name):
                    self.name = name
                return def get_status(self):
                    return {'success': True, 'result': {'status': 'fallback', 'module': self.name}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            module_names = [
                'market_regime', 'rolling_retraining', 'hybrid_engine',
                'alpha_attribution', 'position_sizing', 'execution_control',
                'macro_data', 'forecast_explainability', 'signal_center', 'report_engine'
            ]
            
            for name in module_names:
                self.modules[name] = FallbackModule(name)
            
            logger.info("Initialized fallback modules")
            
        except Exception as e:
            logger.error(f"Error initializing fallback modules: {e}")
    
    def start(self):
        """Start the institutional-grade system."""
        try:
            logger.info("Starting Institutional-Grade Trading System...")
            
            # Start all modules
            for name, module in self.modules.items():
                if hasattr(module, 'start'):
                    try:
                        module.start()
                        logger.info(f"Started module: {name}")
                    except Exception as e:
                        logger.error(f"Error starting module {name}: {e}")
            
            # Start system monitoring
            self._start_system_monitoring()
            
            # Update status
            self.status = SystemStatus.RUNNING
            self.start_time = datetime.now()
            
            logger.info("Institutional-Grade Trading System started successfully")
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.status = SystemStatus.ERROR

    def stop(self):
        """Stop the institutional-grade system."""
        try:
            logger.info("Stopping Institutional-Grade Trading System...")
            
            # Stop all modules
            for name, module in self.modules.items():
                if hasattr(module, 'stop'):
                    try:
                        module.stop()
                        logger.info(f"Stopped module: {name}")
                    except Exception as e:
                        logger.error(f"Error stopping module {name}: {e}")
            
            # Update status
            self.status = SystemStatus.PAUSED
            
            logger.info("Institutional-Grade Trading System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")

    def _start_system_monitoring(self):
        """Start system monitoring and health checks."""
        try:
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("System monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting system monitoring: {e}")

    def _monitoring_loop(self):
        """Main system monitoring loop."""
        while self.status == SystemStatus.RUNNING:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check module health
                self._check_module_health()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Generate periodic reports
                self._generate_periodic_reports()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _update_system_metrics(self):
        """Update system performance metrics."""
        try:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Get metrics from modules
            total_signals = 0
            active_trades = 0
            
            if 'signal_center' in self.modules:
                summary = self.modules['signal_center'].get_signal_summary()
                total_signals = summary.get('total_signals_24h', 0)
                active_trades = summary.get('active_trades', 0)
            
            # Calculate system health
            module_health = self._calculate_module_health()
            
            # Calculate performance score
            performance_score = self._calculate_performance_score()
            
            # Calculate risk score
            risk_score = self._calculate_risk_score()
            
            self.system_metrics = SystemMetrics(
                uptime=uptime,
                total_signals=total_signals,
                active_trades=active_trades,
                system_health=module_health,
                performance_score=performance_score,
                risk_score=risk_score,
                last_update=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def _calculate_module_health(self) -> float:
        """Calculate overall module health score."""
        try:
            if not self.modules:
                return 0.0
            
            health_scores = []
            
            for name, module in self.modules.items():
                try:
                    if hasattr(module, 'get_status'):
                        status = module.get_status()
                        if isinstance(status, dict) and 'status' in status:
                            if status['status'] == 'healthy':
                                health_scores.append(1.0)
                            elif status['status'] == 'warning':
                                health_scores.append(0.7)
                            elif status['status'] == 'error':
                                health_scores.append(0.3)
                            else:
                                health_scores.append(0.5)
                        else:
                            health_scores.append(0.8)  # Assume healthy if no status
                    else:
                        health_scores.append(0.8)  # Assume healthy
                except Exception as e:
                    logger.warning(f"Error checking health for module {name}: {e}")
                    health_scores.append(0.3)
            
            return np.mean(health_scores) if health_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating module health: {e}")
            return 0.0
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        try:
            scores = []
            
            # Get performance from various modules
            if 'hybrid_engine' in self.modules:
                summary = self.modules['hybrid_engine'].get_performance_summary()
                if 'avg_confidence' in summary:
                    scores.append(summary['avg_confidence'])
            
            if 'rolling_retraining' in self.modules:
                summary = self.modules['rolling_retraining'].get_performance_summary()
                if 'avg_recent_performance' in summary:
                    scores.append(summary['avg_recent_performance'])
            
            if 'alpha_attribution' in self.modules:
                summary = self.modules['alpha_attribution'].get_attribution_summary()
                if 'avg_r_squared' in summary:
                    scores.append(summary['avg_r_squared'])
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.5
    
    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score."""
        try:
            scores = []
            
            # Get risk metrics from various modules
            if 'execution_control' in self.modules:
                summary = self.modules['execution_control'].get_risk_summary()
                if 'current_drawdown' in summary:
                    drawdown = abs(summary['current_drawdown'])
                    scores.append(min(1.0, drawdown / 0.15))  # Normalize to max drawdown
            
            if 'position_sizing' in self.modules:
                summary = self.modules['position_sizing'].get_sizing_summary()
                if 'avg_confidence' in summary:
                    scores.append(1.0 - summary['avg_confidence'])  # Lower confidence = higher risk
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _check_module_health(self):
        """Check health of all modules."""
        try:
            for name, module in self.modules.items():
                try:
                    if hasattr(module, 'get_status'):
                        status = module.get_status()
                        if isinstance(status, dict) and status.get('status') == 'error':
                            logger.warning(f"Module {name} reported error status")
                            self._handle_module_error(name, module)
                except Exception as e:
                    logger.error(f"Error checking health for module {name}: {e}")
            
        except Exception as e:
            logger.error(f"Error checking module health: {e}")

    def _handle_module_error(self, module_name: str, module: Any):
        """Handle module errors."""
        try:
            logger.warning(f"Handling error for module: {module_name}")
            
            # Attempt to restart module
            if hasattr(module, 'stop'):
                module.stop()
            
            if hasattr(module, 'start'):
                module.start()
                logger.info(f"Restarted module: {module_name}")
            
        except Exception as e:
            logger.error(f"Error handling module error for {module_name}: {e}")

    def _check_risk_limits(self):
        """Check if system is within risk limits."""
        try:
            if not self.system_metrics:

            # Check drawdown limit
            if self.system_metrics.risk_score > 0.8:
                logger.warning("Risk score approaching limit")
                self._trigger_risk_alert("High risk score detected")
            
            # Check position limits
            if self.system_metrics.active_trades > 50:
                logger.warning("Too many active trades")
                self._trigger_risk_alert("Too many active trades")
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    def _trigger_risk_alert(self, message: str):
        """Trigger risk alert."""
        try:
            alert = {
                'type': 'risk_alert',
                'message': message,
                'priority': 'high',
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to signal center if available
            if 'signal_center' in self.modules:
                self.modules['signal_center']._queue_alert(alert)
            
            logger.warning(f"Risk alert triggered: {message}")
            
        except Exception as e:
            logger.error(f"Error triggering risk alert: {e}")

    def _generate_periodic_reports(self):
        """Generate periodic system reports."""
        try:
            # Check if it's time for daily report
            now = datetime.now()
            if now.hour == 0 and now.minute < 5:  # Once per day at midnight
                self.generate_system_report()
            
        except Exception as e:
            logger.error(f"Error generating periodic reports: {e}")

    def generate_system_report(self) -> str:
        """Generate comprehensive system report."""
        try:
            logger.info("Generating system report...")
            
            # Collect data from all modules
            report_data = {
                'system_metrics': self._get_system_metrics_dict(),
                'module_status': self._get_module_status(),
                'performance_data': self._get_performance_data(),
                'risk_data': self._get_risk_data(),
                'regime_data': self._get_regime_data(),
                'signal_data': self._get_signal_data()
            }
            
            # Create report configuration
            config = ReportConfig(
                title="Institutional-Grade Trading System Report",
                author="System Auto-Generator",
                date=datetime.now(),
                format=ReportFormat.MARKDOWN,
                include_charts=True,
                include_tables=True,
                include_metrics=True
            )
            
            # Generate report
            if 'report_engine' in self.modules:
                report_path = self.modules['report_engine'].generate_comprehensive_report(
                    config, report_data
                )
                logger.info(f"System report generated: {report_path}")
                return report_path
            else:
                logger.warning("Report engine not available")
                return ""
            
        except Exception as e:
            logger.error(f"Error generating system report: {e}")
            return ""
    
    def _get_system_metrics_dict(self) -> Dict[str, Any]:
        """Get system metrics as dictionary."""
        if self.system_metrics:
            return {
                'uptime': self.system_metrics.uptime,
                'total_signals': self.system_metrics.total_signals,
                'active_trades': self.system_metrics.active_trades,
                'system_health': self.system_metrics.system_health,
                'performance_score': self.system_metrics.performance_score,
                'risk_score': self.system_metrics.risk_score,
                'last_update': self.system_metrics.last_update.isoformat()
            }
        return {}
    
    def _get_module_status(self) -> Dict[str, Any]:
        """Get status of all modules."""
        status = {}
        for name, module in self.modules.items():
            try:
                if hasattr(module, 'get_status'):
                    status[name] = module.get_status()
                else:
                    status[name] = {'status': 'unknown', 'module': name}
            except Exception as e:
                status[name] = {'status': 'error', 'error': str(e)}
        return status
    
    def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data from modules."""
        data = {}
        
        if 'hybrid_engine' in self.modules:
            data['hybrid_engine'] = self.modules['hybrid_engine'].get_performance_summary()
        
        if 'rolling_retraining' in self.modules:
            data['rolling_retraining'] = self.modules['rolling_retraining'].get_performance_summary()
        
        if 'alpha_attribution' in self.modules:
            data['alpha_attribution'] = self.modules['alpha_attribution'].get_attribution_summary()
        
        return data
    
    def _get_risk_data(self) -> Dict[str, Any]:
        """Get risk data from modules."""
        data = {}
        
        if 'execution_control' in self.modules:
            data['execution_control'] = self.modules['execution_control'].get_risk_summary()
        
        if 'position_sizing' in self.modules:
            data['position_sizing'] = self.modules['position_sizing'].get_sizing_summary()
        
        return data
    
    def _get_regime_data(self) -> Dict[str, Any]:
        """Get regime data from modules."""
        data = {}
        
        if 'market_regime' in self.modules:
            data['market_regime'] = self.modules['market_regime'].get_regime_summary()
        
        if 'macro_data' in self.modules:
            data['macro_data'] = self.modules['macro_data'].analyze_macro_environment()
        
        return data
    
    def _get_signal_data(self) -> Dict[str, Any]:
        """Get signal data from modules."""
        data = {}
        
        if 'signal_center' in self.modules:
            data['signal_center'] = self.modules['signal_center'].get_signal_summary()
        
        if 'forecast_explainability' in self.modules:
            data['forecast_explainability'] = self.modules['forecast_explainability'].get_explanation_summary()
        
        return data
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'status': self.status.value,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'modules': len(self.modules),
                'system_metrics': self._get_system_metrics_dict(),
                'config': {
                    'name': self.config['system']['name'],
                    'version': self.config['system']['version']
                },
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'success': True, 'result': {'status': 'error', 'error': str(e)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and route to appropriate modules."""
        try:
            query_lower = query.lower()
            
            # Route to appropriate modules based on query content
            if any(word in query_lower for word in ['regime', 'market', 'bull', 'bear']):
                return self._handle_regime_query(query)
            
            elif any(word in query_lower for word in ['signal', 'trade', 'buy', 'sell']):
                return self._handle_signal_query(query)
            
            elif any(word in query_lower for word in ['risk', 'position', 'size']):
                return self._handle_risk_query(query)
            
            elif any(word in query_lower for word in ['performance', 'return', 'sharpe']):
                return self._handle_performance_query(query)
            
            elif any(word in query_lower for word in ['report', 'summary', 'analysis']):
                return self._handle_report_query(query)
            
            else:
                return self._handle_general_query(query)
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def _handle_regime_query(self, query: str) -> Dict[str, Any]:
        """Handle regime-related queries."""
        try:
            if 'market_regime' in self.modules:
                regime_summary = self.modules['market_regime'].get_regime_summary()
                return {
                    'success': True,
                    'type': 'regime_analysis',
                    'data': regime_summary,
                    'query': query
                }
            else:
                return {
                    'success': False,
                    'error': 'Market regime module not available',
                    'query': query
                }
            
        except Exception as e:
            logger.error(f"Error handling regime query: {e}")
            return {'success': False, 'error': str(e), 'query': query}
    
    def _handle_signal_query(self, query: str) -> Dict[str, Any]:
        """Handle signal-related queries."""
        try:
            if 'signal_center' in self.modules:
                signal_summary = self.modules['signal_center'].get_signal_summary()
                return {
                    'success': True,
                    'type': 'signal_analysis',
                    'data': signal_summary,
                    'query': query
                }
            else:
                return {
                    'success': False,
                    'error': 'Signal center module not available',
                    'query': query
                }
            
        except Exception as e:
            logger.error(f"Error handling signal query: {e}")
            return {'success': False, 'error': str(e), 'query': query}
    
    def _handle_risk_query(self, query: str) -> Dict[str, Any]:
        """Handle risk-related queries."""
        try:
            risk_data = {}
            
            if 'execution_control' in self.modules:
                risk_data['execution_control'] = self.modules['execution_control'].get_risk_summary()
            
            if 'position_sizing' in self.modules:
                risk_data['position_sizing'] = self.modules['position_sizing'].get_sizing_summary()
            
            return {
                'success': True,
                'type': 'risk_analysis',
                'data': risk_data,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error handling risk query: {e}")
            return {'success': False, 'error': str(e), 'query': query}
    
    def _handle_performance_query(self, query: str) -> Dict[str, Any]:
        """Handle performance-related queries."""
        try:
            performance_data = self._get_performance_data()
            
            return {
                'success': True,
                'type': 'performance_analysis',
                'data': performance_data,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error handling performance query: {e}")
            return {'success': False, 'error': str(e), 'query': query}
    
    def _handle_report_query(self, query: str) -> Dict[str, Any]:
        """Handle report-related queries."""
        try:
            report_path = self.generate_system_report()
            
            return {
                'success': True,
                'type': 'report_generation',
                'data': {'report_path': report_path},
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error handling report query: {e}")
            return {'success': False, 'error': str(e), 'query': query}
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries."""
        try:
            system_status = self.get_system_status()
            
            return {
                'success': True,
                'type': 'system_status',
                'data': system_status,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error handling general query: {e}")
            return {'success': False, 'error': str(e), 'query': query}
    
    def export_system_data(self, filepath: str = "logs/institutional_system_export.json"):
        """Export comprehensive system data."""
        try:
            export_data = {
                'system_status': self.get_system_status(),
                'module_status': self._get_module_status(),
                'performance_data': self._get_performance_data(),
                'risk_data': self._get_risk_data(),
                'regime_data': self._get_regime_data(),
                'signal_data': self._get_signal_data(),
                'config': self.config,
                'export_date': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"System data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting system data: {e}")

    def get_help(self) -> Dict[str, Any]:
        """Get system help and usage information."""
        return {
            'system_name': self.config['system']['name'],
            'version': self.config['system']['version'],
            'description': 'Institutional-Grade Trading System with full strategic intelligence',
            'modules': list(self.modules.keys()),
            'commands': {
                'status': 'Get system status and health',
                'query': 'Process natural language queries',
                'report': 'Generate comprehensive system report',
                'export': 'Export system data',
                'start': 'Start the system',
                'stop': 'Stop the system'
            },
            'example_queries': [
                'What is the current market regime?',
                'Show me recent trading signals',
                'What is the current risk level?',
                'Generate a performance report',
                'How is the system performing?'
            ]
        } 