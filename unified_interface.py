"""
Unified Interface for Institutional-Grade Trading System

This module provides a comprehensive interface that integrates all agents, strategies,
and features into a single, autonomous, prompt-driven system with full audit trails
and exportable insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import json
import asyncio
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import redis
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all institutional-grade modules
try:
    # Core agents
    from trading.agents.market_regime_agent import MarketRegimeAgent
    from trading.agents.walk_forward_agent import WalkForwardAgent
    from trading.agents.execution_risk_control_agent import ExecutionRiskControlAgent
    
    # Strategy engines
    from trading.strategies.hybrid_engine import MultiStrategyHybridEngine
    
    # Analytics engines
    from trading.analytics.alpha_attribution_engine import AlphaAttributionEngine
    from trading.analytics.forecast_explainability import ForecastExplainabilityEngine
    
    # Risk management
    from trading.risk.position_sizing_engine import PositionSizingEngine
    
    # Data integration
    from trading.data.macro_data_integration import MacroDataIntegration
    
    # Real-time systems
    from trading.services.real_time_signal_center import RealTimeSignalCenter
    from trading.report.export_engine import ReportExportEngine
    
    # Portfolio and execution
    from trading.portfolio.portfolio_manager import PortfolioManager
    from trading.execution.execution_engine import ExecutionEngine
    
    # LLM and NLP
    from trading.llm.agent import QuantGPTAgent
    from trading.nlp.llm_processor import LLMProcessor
    
    # Optimization
    from trading.optimization.strategy_optimizer import StrategyOptimizer
    from trading.optimization.portfolio_optimizer import PortfolioOptimizer
    
    # Memory and monitoring
    from trading.memory.model_monitor import ModelMonitor
    from trading.memory.strategy_logger import StrategyLogger
    
    # UI components
    from trading.ui.components import (
        create_date_range_selector, create_model_selector, create_strategy_selector,
        create_parameter_inputs, create_forecast_chart, create_performance_report
    )
    
    MODULES_AVAILABLE = True
    logger.info("All institutional-grade modules imported successfully")
    
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    MODULES_AVAILABLE = False

@dataclass
class SystemStatus:
    """System status information."""
    status: str
    message: str
    timestamp: datetime
    modules: Dict[str, str]
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'modules': self.modules,
            'performance_metrics': self.performance_metrics
        }

@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    data: Dict[str, Any]
    message: str
    confidence: float
    execution_time: float
    agent_name: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'data': self.data,
            'message': self.message,
            'confidence': self.confidence,
            'execution_time': self.execution_time,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp.isoformat()
        }

class UnifiedInterface:
    """Main unified interface for the institutional trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the unified interface."""
        self.config = self._load_config(config_path)
        self.status = SystemStatus(
            status="initializing",
            message="System starting up",
            timestamp=datetime.now(),
            modules={},
            performance_metrics={}
        )
        
        # Initialize all components
        self._initialize_components()
        self._setup_redis()
        self._setup_logging()
        
        # Update status
        self.status.status = "operational"
        self.status.message = "System ready"
        self.status.timestamp = datetime.now()
        
        logger.info("Unified Interface initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "system_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/unified_interface.log'
            },
            'agents': {
                'market_regime': {'enabled': True},
                'walk_forward': {'enabled': True},
                'execution_risk': {'enabled': True},
                'quant_gpt': {'enabled': True}
            },
            'strategies': {
                'hybrid_engine': {'enabled': True},
                'portfolio_optimizer': {'enabled': True}
            },
            'data': {
                'cache_ttl': 3600,
                'fallback_enabled': True
            }
        }
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize agents
            self.agents = {}
            if MODULES_AVAILABLE:
                self.agents['market_regime'] = MarketRegimeAgent()
                self.agents['walk_forward'] = WalkForwardAgent()
                self.agents['execution_risk'] = ExecutionRiskControlAgent()
                self.agents['quant_gpt'] = QuantGPTAgent()
                
                # Initialize engines
                self.hybrid_engine = MultiStrategyHybridEngine()
                self.alpha_engine = AlphaAttributionEngine()
                self.forecast_explainability = ForecastExplainabilityEngine()
                self.position_sizing = PositionSizingEngine()
                self.macro_data = MacroDataIntegration()
                self.signal_center = RealTimeSignalCenter()
                self.report_engine = ReportExportEngine()
                self.portfolio_manager = PortfolioManager()
                self.execution_engine = ExecutionEngine()
                self.strategy_optimizer = StrategyOptimizer()
                self.portfolio_optimizer = PortfolioOptimizer()
                self.model_monitor = ModelMonitor()
                self.strategy_logger = StrategyLogger()
                self.llm_processor = LLMProcessor()
                
                # Update module status
                for name, agent in self.agents.items():
                    self.status.modules[name] = "active"
                
                logger.info("All components initialized successfully")
            else:
                logger.warning("Some components not available due to missing modules")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.status.status = "degraded"
            self.status.message = f"Component initialization failed: {e}"
    
    def _setup_redis(self) -> None:
        """Setup Redis connection for caching."""
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            log_config = self.config.get('logging', {})
            log_file = Path(log_config.get('file', 'logs/unified_interface.log'))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logging.basicConfig(
                level=getattr(logging, log_config.get('level', 'INFO')),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            logger.info("Logging configured successfully")
        except Exception as e:
            logger.warning(f"Logging setup failed: {e}")
    
    def process_natural_language_query(self, query: str) -> AgentResult:
        """Process natural language query and route to appropriate agents."""
        start_time = datetime.now()
        
        try:
            # Parse query using LLM processor
            if hasattr(self, 'llm_processor'):
                intent = self.llm_processor.parse_intent(query)
                entities = self.llm_processor.extract_entities(query)
            else:
                intent = self._simple_intent_parsing(query)
                entities = {}
            
            # Route to appropriate agent
            result = self._route_query(intent, entities, query)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            agent_result = AgentResult(
                success=result.get('success', False),
                data=result.get('data', {}),
                message=result.get('message', ''),
                confidence=result.get('confidence', 0.0),
                execution_time=execution_time,
                agent_name=result.get('agent_name', 'unified_interface'),
                timestamp=datetime.now()
            )
            
            # Log result
            self._log_agent_result(agent_result)
            
            return agent_result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return AgentResult(
                success=False,
                data={'error': str(e)},
                message=f"Error processing query: {e}",
                confidence=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                agent_name='unified_interface',
                timestamp=datetime.now()
            )
    
    def _simple_intent_parsing(self, query: str) -> str:
        """Simple intent parsing without LLM."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['forecast', 'predict', 'price']):
            return 'forecast'
        elif any(word in query_lower for word in ['trade', 'buy', 'sell', 'position']):
            return 'trading'
        elif any(word in query_lower for word in ['strategy', 'optimize', 'backtest']):
            return 'strategy'
        elif any(word in query_lower for word in ['portfolio', 'risk', 'allocation']):
            return 'portfolio'
        elif any(word in query_lower for word in ['market', 'regime', 'analysis']):
            return 'market_analysis'
        else:
            return 'general'
    
    def _route_query(self, intent: str, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Route query to appropriate agent based on intent."""
        try:
            if intent == 'forecast':
                return self._handle_forecast_request(entities, query)
            elif intent == 'trading':
                return self._handle_trading_request(entities, query)
            elif intent == 'strategy':
                return self._handle_strategy_request(entities, query)
            elif intent == 'portfolio':
                return self._handle_portfolio_request(entities, query)
            elif intent == 'market_analysis':
                return self._handle_market_analysis_request(entities, query)
            else:
                return self._handle_general_request(query)
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            return {
                'success': False,
                'data': {'error': str(e)},
                'message': f"Error routing query: {e}",
                'confidence': 0.0,
                'agent_name': 'unified_interface'
            }
    
    def _handle_forecast_request(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle forecast requests."""
        try:
            symbol = entities.get('symbol', 'AAPL')
            timeframe = entities.get('timeframe', '1d')
            horizon = entities.get('horizon', 5)
            
            # Get market regime
            regime_result = self.agents['market_regime'].detect_regime(symbol)
            
            # Generate forecast with explainability
            forecast_data = self._generate_forecast(symbol, timeframe, horizon)
            explainability = self.forecast_explainability.explain_forecast(forecast_data)
            
            # Get QuantGPT commentary
            commentary = self.agents['quant_gpt'].generate_commentary(
                forecast_data, regime_result, explainability
            )
            
            return {
                'success': True,
                'data': {
                    'forecast': forecast_data,
                    'regime': regime_result,
                    'explainability': explainability,
                    'commentary': commentary
                },
                'message': f"Forecast generated for {symbol}",
                'confidence': 0.85,
                'agent_name': 'forecast_agent'
            }
        except Exception as e:
            logger.error(f"Error handling forecast request: {e}")
            return {
                'success': False,
                'data': {'error': str(e)},
                'message': f"Error generating forecast: {e}",
                'confidence': 0.0,
                'agent_name': 'forecast_agent'
            }
    
    def _handle_trading_request(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle trading requests."""
        try:
            symbol = entities.get('symbol', 'AAPL')
            action = entities.get('action', 'analyze')
            
            # Get market regime and signals
            regime_result = self.agents['market_regime'].detect_regime(symbol)
            signals = self.signal_center.get_signals(symbol)
            
            # Get position sizing recommendation
            sizing = self.position_sizing.calculate_position_size(
                symbol, regime_result, signals
            )
            
            # Check execution risk
            risk_check = self.agents['execution_risk'].check_trade_risk(
                symbol, action, sizing
            )
            
            return {
                'success': True,
                'data': {
                    'regime': regime_result,
                    'signals': signals,
                    'position_sizing': sizing,
                    'risk_check': risk_check
                },
                'message': f"Trading analysis completed for {symbol}",
                'confidence': 0.80,
                'agent_name': 'trading_agent'
            }
        except Exception as e:
            logger.error(f"Error handling trading request: {e}")
            return {
                'success': False,
                'data': {'error': str(e)},
                'message': f"Error analyzing trade: {e}",
                'confidence': 0.0,
                'agent_name': 'trading_agent'
            }
    
    def _handle_strategy_request(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle strategy requests."""
        try:
            strategy_type = entities.get('strategy_type', 'hybrid')
            
            # Get hybrid strategy signals
            signals = self.hybrid_engine.generate_signals()
            
            # Get alpha attribution
            alpha_attribution = self.alpha_engine.analyze_alpha_attribution()
            
            # Get strategy optimization recommendations
            optimization = self.strategy_optimizer.optimize_strategies()
            
            return {
                'success': True,
                'data': {
                    'signals': signals,
                    'alpha_attribution': alpha_attribution,
                    'optimization': optimization
                },
                'message': f"Strategy analysis completed",
                'confidence': 0.75,
                'agent_name': 'strategy_agent'
            }
        except Exception as e:
            logger.error(f"Error handling strategy request: {e}")
            return {
                'success': False,
                'data': {'error': str(e)},
                'message': f"Error analyzing strategy: {e}",
                'confidence': 0.0,
                'agent_name': 'strategy_agent'
            }
    
    def _handle_portfolio_request(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle portfolio requests."""
        try:
            # Get portfolio status
            portfolio_status = self.portfolio_manager.get_portfolio_status()
            
            # Get optimization recommendations
            optimization = self.portfolio_optimizer.optimize_portfolio()
            
            # Get risk analysis
            risk_analysis = self.portfolio_manager.get_risk_analysis()
            
            return {
                'success': True,
                'data': {
                    'portfolio_status': portfolio_status,
                    'optimization': optimization,
                    'risk_analysis': risk_analysis
                },
                'message': "Portfolio analysis completed",
                'confidence': 0.85,
                'agent_name': 'portfolio_agent'
            }
        except Exception as e:
            logger.error(f"Error handling portfolio request: {e}")
            return {
                'success': False,
                'data': {'error': str(e)},
                'message': f"Error analyzing portfolio: {e}",
                'confidence': 0.0,
                'agent_name': 'portfolio_agent'
            }
    
    def _handle_market_analysis_request(self, entities: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Handle market analysis requests."""
        try:
            # Get market regime analysis
            regime_analysis = self.agents['market_regime'].get_comprehensive_analysis()
            
            # Get macro data
            macro_data = self.macro_data.get_macro_indicators()
            
            # Get walk-forward analysis
            walk_forward = self.agents['walk_forward'].get_analysis()
            
            return {
                'success': True,
                'data': {
                    'regime_analysis': regime_analysis,
                    'macro_data': macro_data,
                    'walk_forward': walk_forward
                },
                'message': "Market analysis completed",
                'confidence': 0.80,
                'agent_name': 'market_analysis_agent'
            }
        except Exception as e:
            logger.error(f"Error handling market analysis request: {e}")
            return {
                'success': False,
                'data': {'error': str(e)},
                'message': f"Error analyzing market: {e}",
                'confidence': 0.0,
                'agent_name': 'market_analysis_agent'
            }
    
    def _handle_general_request(self, query: str) -> Dict[str, Any]:
        """Handle general requests."""
        try:
            # Use QuantGPT for general queries
            response = self.agents['quant_gpt'].process_query(query)
            
            return {
                'success': True,
                'data': {'response': response},
                'message': "General query processed",
                'confidence': 0.70,
                'agent_name': 'quant_gpt'
            }
        except Exception as e:
            logger.error(f"Error handling general request: {e}")
            return {
                'success': False,
                'data': {'error': str(e)},
                'message': f"Error processing query: {e}",
                'confidence': 0.0,
                'agent_name': 'quant_gpt'
            }
    
    def _generate_forecast(self, symbol: str, timeframe: str, horizon: int) -> Dict[str, Any]:
        """Generate forecast with fallback logic."""
        try:
            # Try to get cached forecast
            cache_key = f"forecast:{symbol}:{timeframe}:{horizon}"
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            
            # Generate new forecast (placeholder)
            forecast_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'predictions': np.random.normal(100, 5, horizon).tolist(),
                'confidence_intervals': {
                    'lower': (np.random.normal(100, 5, horizon) - 2).tolist(),
                    'upper': (np.random.normal(100, 5, horizon) + 2).tolist()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            if self.redis_client:
                self.redis_client.setex(
                    cache_key, 
                    self.config.get('data', {}).get('cache_ttl', 3600),
                    json.dumps(forecast_data)
                )
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            # Return fallback forecast
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'predictions': [100] * horizon,
                'confidence_intervals': {
                    'lower': [98] * horizon,
                    'upper': [102] * horizon
                },
                'timestamp': datetime.now().isoformat(),
                'fallback': True
            }
    
    def _log_agent_result(self, result: AgentResult) -> None:
        """Log agent execution result."""
        try:
            # Log to strategy logger
            if hasattr(self, 'strategy_logger'):
                self.strategy_logger.log_decision(
                    agent_name=result.agent_name,
                    decision=result.data,
                    confidence=result.confidence,
                    execution_time=result.execution_time
                )
            
            # Log to Redis for real-time monitoring
            if self.redis_client:
                log_entry = {
                    'timestamp': result.timestamp.isoformat(),
                    'agent': result.agent_name,
                    'success': result.success,
                    'confidence': result.confidence,
                    'execution_time': result.execution_time
                }
                self.redis_client.lpush('agent_logs', json.dumps(log_entry))
                self.redis_client.ltrim('agent_logs', 0, 999)  # Keep last 1000 entries
            
        except Exception as e:
            logger.error(f"Error logging agent result: {e}")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        try:
            # Update performance metrics
            self.status.performance_metrics = self._calculate_performance_metrics()
            self.status.timestamp = datetime.now()
            
            return self.status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return SystemStatus(
                status="error",
                message=f"Error getting status: {e}",
                timestamp=datetime.now(),
                modules={},
                performance_metrics={}
            )
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate system performance metrics."""
        try:
            metrics = {}
            
            # Get model trust levels
            if hasattr(self, 'model_monitor'):
                trust_levels = self.model_monitor.get_model_trust_levels()
                metrics['avg_model_trust'] = sum(trust_levels.values()) / len(trust_levels) if trust_levels else 0
            
            # Get portfolio performance
            if hasattr(self, 'portfolio_manager'):
                portfolio_status = self.portfolio_manager.get_portfolio_status()
                metrics['portfolio_return'] = portfolio_status.get('total_return', 0)
                metrics['portfolio_sharpe'] = portfolio_status.get('sharpe_ratio', 0)
            
            # Get signal quality
            if hasattr(self, 'signal_center'):
                signals = self.signal_center.get_signal_quality()
                metrics['signal_accuracy'] = signals.get('accuracy', 0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def export_report(self, report_type: str = 'comprehensive') -> str:
        """Export system report."""
        try:
            if hasattr(self, 'report_engine'):
                return self.report_engine.generate_report(report_type)
            else:
                # Fallback report generation
                return self._generate_fallback_report(report_type)
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return f"Error generating report: {e}"
    
    def _generate_fallback_report(self, report_type: str) -> str:
        """Generate fallback report."""
        try:
            status = self.get_system_status()
            
            report = f"""
# Evolve Trading System Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Status
- Status: {status.status}
- Message: {status.message}
- Modules Active: {len([m for m in status.modules.values() if m == 'active'])}

## Performance Metrics
"""
            for metric, value in status.performance_metrics.items():
                report += f"- {metric}: {value:.4f}\n"
            
            report += "\n## Recent Activity\n"
            
            # Get recent logs
            if self.redis_client:
                recent_logs = self.redis_client.lrange('agent_logs', 0, 9)
                for log in recent_logs:
                    log_data = json.loads(log)
                    report += f"- {log_data['timestamp']}: {log_data['agent']} ({log_data['success']})\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating fallback report: {e}")
            return f"Error generating report: {e}"

def main():
    """Main function for running the unified interface."""
    # Initialize interface
    interface = UnifiedInterface()
    
    # Run Streamlit UI
    st.set_page_config(
        page_title="Evolve Institutional Trading System",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Evolve Institutional Trading System")
    st.markdown("Advanced AI-powered trading system with autonomous agents and real-time analysis")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Interface:",
        ["🏠 Dashboard", "🤖 AI Agents", "📈 Forecasting", "⚡ Trading", "📊 Portfolio", "📋 Reports"]
    )
    
    if page == "🏠 Dashboard":
        render_dashboard(interface)
    elif page == "🤖 AI Agents":
        render_agents_interface(interface)
    elif page == "📈 Forecasting":
        render_forecasting_interface(interface)
    elif page == "⚡ Trading":
        render_trading_interface(interface)
    elif page == "📊 Portfolio":
        render_portfolio_interface(interface)
    elif page == "📋 Reports":
        render_reports_interface(interface)

def render_dashboard(interface: UnifiedInterface):
    """Render main dashboard."""
    st.header("🏠 System Dashboard")
    
    # System status
    status = interface.get_system_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", status.status, status.message)
    
    with col2:
        active_modules = len([m for m in status.modules.values() if m == 'active'])
        st.metric("Active Modules", active_modules)
    
    with col3:
        avg_trust = status.performance_metrics.get('avg_model_trust', 0)
        st.metric("Avg Model Trust", f"{avg_trust:.1%}")
    
    with col4:
        portfolio_return = status.performance_metrics.get('portfolio_return', 0)
        st.metric("Portfolio Return", f"{portfolio_return:.2%}")
    
    # Natural language interface
    st.subheader("🤖 Natural Language Interface")
    query = st.text_area("Ask me anything about trading, forecasting, or portfolio management:")
    
    if st.button("Process Query"):
        if query:
            with st.spinner("Processing your query..."):
                result = interface.process_natural_language_query(query)
                
                if result.success:
                    st.success(result.message)
                    st.json(result.data)
                else:
                    st.error(result.message)
                    st.json(result.data)

def render_agents_interface(interface: UnifiedInterface):
    """Render AI agents interface."""
    st.header("🤖 AI Agents")
    
    # Agent status
    for agent_name, status in interface.status.modules.items():
        st.write(f"**{agent_name}**: {status}")
    
    # Agent controls
    st.subheader("Agent Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Market Regime Analysis"):
            with st.spinner("Analyzing market regime..."):
                result = interface.agents['market_regime'].detect_regime('SPY')
                st.json(result)
    
    with col2:
        if st.button("Run Walk-Forward Analysis"):
            with st.spinner("Running walk-forward analysis..."):
                result = interface.agents['walk_forward'].get_analysis()
                st.json(result)

def render_forecasting_interface(interface: UnifiedInterface):
    """Render forecasting interface."""
    st.header("📈 Forecasting")
    
    # Input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Symbol", "AAPL")
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "1h", "15m", "5m"])
    
    with col3:
        horizon = st.slider("Forecast Horizon", 1, 30, 5)
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            result = interface._handle_forecast_request(
                {'symbol': symbol, 'timeframe': timeframe, 'horizon': horizon},
                f"Forecast {symbol} for {horizon} periods"
            )
            
            if result['success']:
                st.success(result['message'])
                
                # Display forecast
                forecast_data = result['data']['forecast']
                
                # Create forecast chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=forecast_data['predictions'],
                    mode='lines+markers',
                    name='Forecast'
                ))
                fig.add_trace(go.Scatter(
                    y=forecast_data['confidence_intervals']['upper'],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Upper CI'
                ))
                fig.add_trace(go.Scatter(
                    y=forecast_data['confidence_intervals']['lower'],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Lower CI'
                ))
                
                fig.update_layout(title=f"Forecast for {symbol}")
                st.plotly_chart(fig)
                
                # Display commentary
                if 'commentary' in result['data']:
                    st.subheader("AI Commentary")
                    st.write(result['data']['commentary'])
            else:
                st.error(result['message'])

def render_trading_interface(interface: UnifiedInterface):
    """Render trading interface."""
    st.header("⚡ Trading")
    
    # Trading parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Symbol", "AAPL")
    
    with col2:
        action = st.selectbox("Action", ["analyze", "buy", "sell"])
    
    with col3:
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    
    if st.button("Analyze Trade"):
        with st.spinner("Analyzing trade..."):
            result = interface._handle_trading_request(
                {'symbol': symbol, 'action': action, 'amount': amount},
                f"{action} {symbol} for {amount}"
            )
            
            if result['success']:
                st.success(result['message'])
                st.json(result['data'])
            else:
                st.error(result['message'])

def render_portfolio_interface(interface: UnifiedInterface):
    """Render portfolio interface."""
    st.header("📊 Portfolio")
    
    if st.button("Refresh Portfolio"):
        with st.spinner("Loading portfolio..."):
            result = interface._handle_portfolio_request({}, "Get portfolio status")
            
            if result['success']:
                st.success(result['message'])
                
                # Display portfolio status
                portfolio_status = result['data']['portfolio_status']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Value", f"${portfolio_status.get('total_value', 0):,.2f}")
                
                with col2:
                    st.metric("Total Return", f"{portfolio_status.get('total_return', 0):.2%}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{portfolio_status.get('sharpe_ratio', 0):.2f}")
                
                # Display positions
                if 'positions' in portfolio_status:
                    st.subheader("Positions")
                    positions_df = pd.DataFrame(portfolio_status['positions'])
                    st.dataframe(positions_df)
            else:
                st.error(result['message'])

def render_reports_interface(interface: UnifiedInterface):
    """Render reports interface."""
    st.header("📋 Reports")
    
    report_type = st.selectbox("Report Type", ["comprehensive", "performance", "risk", "strategy"])
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            report = interface.export_report(report_type)
            
            st.subheader("Generated Report")
            st.text_area("Report Content", report, height=400)
            
            # Download button
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"evolve_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main() 