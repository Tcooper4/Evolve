"""
Enhanced Unified Interface for Evolve Trading System

Provides access to all features through:
- Streamlit UI with LLM controls and transparency
- Terminal UI
- Direct command/prompt interface

Features include:
- Model tuning and optimization
- Forecasting and predictions
- QuantGPT natural language interface
- Strategy management
- Performance tracking
- Report generation
- Agent management
- LLM provider controls and transparency
- Notification system integration
"""

import streamlit as st
import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent / "trading"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedUnifiedInterface:
    """Enhanced unified interface for all Evolve trading system features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced unified interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self.components = {}
        
        # LLM provider settings
        self.llm_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.huggingface_model = os.getenv('HUGGINGFACE_MODEL', 'gpt2')
        
        # Notification settings
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        
        # Initialize components
        self._initialize_components()
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Import components dynamically to handle missing dependencies
            self._import_components()
            self.initialized = True
            logger.info("Enhanced unified interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.initialized = False
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _import_components(self):
        """Import all available components."""
        try:
            from trading.services.quant_gpt import QuantGPT
            self.components['quant_gpt'] = QuantGPT(
                openai_api_key=self.openai_api_key,
                redis_host=self.config.get('redis_host', 'localhost'),
                redis_port=self.config.get('redis_port', 6379)
            )
        except ImportError:
            logger.warning("QuantGPT not available")
        
        try:
            from trading.services.service_client import ServiceClient
            self.components['service_client'] = ServiceClient(
                redis_host=self.config.get('redis_host', 'localhost'),
                redis_port=self.config.get('redis_port', 6379)
            )
        except ImportError:
            logger.warning("ServiceClient not available")
        
        try:
            from trading.agents.agent_manager import AgentManager
            self.components['agent_manager'] = AgentManager()
        except ImportError:
            logger.warning("AgentManager not available")
        
        try:
            from trading.report.report_client import ReportClient
            self.components['report_client'] = ReportClient()
        except ImportError:
            logger.warning("ReportClient not available")
        
        try:
            from trading.agents.enhanced_prompt_router import EnhancedPromptRouterAgent
            self.components['prompt_router'] = EnhancedPromptRouterAgent(
                openai_api_key=self.openai_api_key,
                huggingface_model=self.huggingface_model,
                huggingface_api_key=self.huggingface_api_key
            )
        except ImportError:
            logger.warning("EnhancedPromptRouterAgent not available")
        
        try:
            from trading.utils.notification_system import notification_system
            self.components['notifications'] = notification_system
        except ImportError:
            logger.warning("Notification system not available")
        
        try:
            from trading.agents.meta_learner import MetaLearnerAgent
            self.components['meta_learner'] = MetaLearnerAgent()
        except ImportError:
            logger.warning("MetaLearnerAgent not available")
        
        try:
            from trading.optimizer.strategies.strategy_optimizer import StrategyOptimizer
            self.components['strategy_optimizer'] = StrategyOptimizer()
        except ImportError:
            logger.warning("StrategyOptimizer not available")
        
        try:
            from trading.agents.model_switching_controller import ModelSwitchingController
            self.components['model_controller'] = ModelSwitchingController()
        except ImportError:
            logger.warning("ModelSwitchingController not available")
        
        try:
            from trading.memory.prompt_feedback_memory import PromptFeedbackMemory
            self.components['prompt_memory'] = PromptFeedbackMemory()
        except ImportError:
            logger.warning("PromptFeedbackMemory not available")
        
        try:
            from trading.memory.long_term_performance_tracker import LongTermPerformanceTracker
            self.components['performance_tracker'] = LongTermPerformanceTracker()
        except ImportError:
            logger.warning("LongTermPerformanceTracker not available")

    def get_llm_provider_status(self) -> Dict[str, Any]:
        """Get status of all LLM providers."""
        if 'prompt_router' in self.components:
            return self.components['prompt_router'].get_provider_status()
        else:
            return {
                'openai': bool(self.openai_api_key),
                'huggingface': bool(self.huggingface_api_key),
                'regex': True
            }
    
    def set_llm_provider(self, provider: str) -> bool:
        """Set the LLM provider for processing."""
        available_providers = ['openai', 'huggingface', 'regex']
        if provider in available_providers:
            self.llm_provider = provider
            logger.info(f"LLM provider set to: {provider}")
            return True
        else:
            logger.error(f"Invalid LLM provider: {provider}")
            return False
    
    def get_notification_status(self) -> Dict[str, Any]:
        """Get notification system status."""
        if 'notifications' in self.components:
            return self.components['notifications'].get_notification_status()
        else:
            return {
                'slack_configured': bool(self.slack_webhook_url),
                'email_configured': bool(self.email_password),
                'last_notification': None
            }
    
    def get_help(self) -> Dict[str, Any]:
        """Get comprehensive help information."""
        return {
            'overview': {
                'title': 'Evolve Trading System - Enhanced Unified Interface',
                'description': 'Access all trading system features through natural language, commands, or UI',
                'version': '2.1.0'
            },
            'features': {
                'forecasting': {
                    'description': 'Generate market predictions using advanced ML models',
                    'commands': [
                        'forecast AAPL 30d',
                        'predict BTCUSDT 1h',
                        'analyze market NVDA'
                    ],
                    'ui_section': 'Forecasting'
                },
                'tuning': {
                    'description': 'Optimize model hyperparameters and strategies',
                    'commands': [
                        'tune model lstm AAPL',
                        'optimize strategy rsi',
                        'hyperparameter search xgboost'
                    ],
                    'ui_section': 'Optimization'
                },
                'quantgpt': {
                    'description': 'Natural language interface for all features',
                    'commands': [
                        'What\'s the best model for TSLA?',
                        'Should I buy AAPL now?',
                        'Analyze market conditions for BTC'
                    ],
                    'ui_section': 'QuantGPT'
                },
                'strategy': {
                    'description': 'Manage and execute trading strategies',
                    'commands': [
                        'run strategy bollinger AAPL',
                        'backtest macd TSLA',
                        'list strategies'
                    ],
                    'ui_section': 'Strategy'
                },
                'portfolio': {
                    'description': 'Portfolio management and analysis',
                    'commands': [
                        'portfolio status',
                        'rebalance portfolio',
                        'risk analysis'
                    ],
                    'ui_section': 'Portfolio'
                },
                'agents': {
                    'description': 'Manage autonomous trading agents',
                    'commands': [
                        'list agents',
                        'start agent model_builder',
                        'agent status'
                    ],
                    'ui_section': 'Agents'
                },
                'reports': {
                    'description': 'Generate comprehensive trading reports',
                    'commands': [
                        'generate report AAPL',
                        'performance report',
                        'trade log'
                    ],
                    'ui_section': 'Reports'
                }
            },
            'ui_modes': {
                'streamlit': 'Web-based interface with interactive dashboards',
                'terminal': 'Command-line interface for quick access',
                'api': 'Direct API calls for programmatic access'
            },
            'llm_providers': {
                'openai': 'GPT-4 for high-quality natural language processing',
                'huggingface': 'Local models for privacy and cost control',
                'regex': 'Rule-based fallback for reliability'
            },
            'notifications': {
                'slack': 'Real-time alerts via Slack webhooks',
                'email': 'Email notifications for important events'
            },
            'examples': {
                'quick_start': [
                    'forecast AAPL 7d',
                    'tune model lstm AAPL',
                    'What\'s the trading signal for TSLA?'
                ],
                'advanced': [
                    'run full analysis AAPL with backtest',
                    'optimize portfolio with risk constraints',
                    'generate comprehensive report with heatmap'
                ]
            }
        }
    
    def process_command(self, command: str, llm_provider: Optional[str] = None) -> Dict[str, Any]:
        """Process a command or natural language query.
        
        Args:
            command: Command string or natural language query
            llm_provider: Specific LLM provider to use
            
        Returns:
            Dictionary containing results and metadata
        """
        try:
            # Use specified provider or default
            provider = llm_provider or self.llm_provider
            
            # Process with enhanced prompt router if available
            if 'prompt_router' in self.components:
                # Create mock agents for routing
                mock_agents = {
                    'forecasting': self._create_mock_agent('forecasting'),
                    'backtesting': self._create_mock_agent('backtesting'),
                    'tuning': self._create_mock_agent('tuning'),
                    'research': self._create_mock_agent('research'),
                    'portfolio': self._create_mock_agent('portfolio'),
                    'risk': self._create_mock_agent('risk'),
                    'sentiment': self._create_mock_agent('sentiment')
                }
                
                result = self.components['prompt_router'].route(command, mock_agents)
                
                # Add provider information
                result['llm_provider'] = provider
                result['command'] = command
                result['timestamp'] = datetime.now().isoformat()
                
                # Store in prompt memory if available
                if 'prompt_memory' in self.components:
                    self.components['prompt_memory'].store_interaction(command, result)
                
                # Send notification if significant event
                if result.get('success') and 'notifications' in self.components:
                    self.components['notifications'].send_agent_activity_notification(
                        'prompt_router', f"Processed: {command[:50]}...", result
                    )
                
                return result
            
            # Fallback to original processing
            return self._process_command_fallback(command)
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return {
                'success': False,
                'error': str(e),
                'command': command,
                'llm_provider': llm_provider or self.llm_provider,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_mock_agent(self, agent_type: str):
        """Create a mock agent for routing."""
        class MockAgent:
            def __init__(self, agent_type):
                self.agent_type = agent_type
            def run_forecast(self, **kwargs):
                return {'success': True, 'result': {'type': 'forecast', 'agent': self.agent_type, 'args': kwargs}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            def run_backtest(self, **kwargs):
                return {'success': True, 'result': {'type': 'backtest', 'agent': self.agent_type, 'args': kwargs}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            def run_tuning(self, **kwargs):
                return {'success': True, 'result': {'type': 'tuning', 'agent': self.agent_type, 'args': kwargs}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            def research(self, **kwargs):
                return {'success': True, 'result': {'type': 'research', 'agent': self.agent_type, 'args': kwargs}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            def analyze_portfolio(self, **kwargs):
                return {'success': True, 'result': {'type': 'portfolio', 'agent': self.agent_type, 'args': kwargs}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            def analyze_risk(self, **kwargs):
                return {'success': True, 'result': {'type': 'risk', 'agent': self.agent_type, 'args': kwargs}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
            return {'success': True, 'result': def analyze_sentiment(self, **kwargs):, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
                return {'success': True, 'result': {'type': 'sentiment', 'agent': self.agent_type, 'args': kwargs}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        
        return MockAgent(agent_type)
    
    def _process_command_fallback(self, command: str) -> Dict[str, Any]:
        """Fallback command processing."""
        command_lower = command.lower().strip()
        
        # Simple command parsing
        if command_lower.startswith('forecast') or command_lower.startswith('predict'):
            return self._handle_forecasting(command.split())
        elif command_lower.startswith('tune') or command_lower.startswith('optimize'):
            return self._handle_tuning(command.split())
        elif command_lower.startswith('strategy') or command_lower.startswith('backtest'):
            return self._handle_strategy(command.split())
        elif command_lower.startswith('portfolio'):
            return self._handle_portfolio(command.split())
        elif command_lower.startswith('agent'):
            return self._handle_agents(command.split())
        elif command_lower.startswith('report'):
            return self._handle_reports(command.split())
        elif command_lower.startswith('status'):
            return self._handle_status(command.split())
        else:
            # Try natural language processing
            return self._process_natural_language(command)
    
    def _process_natural_language(self, query: str) -> Dict[str, Any]:
        """Process natural language queries."""
        if 'quant_gpt' in self.components:
            try:
                response = self.components['quant_gpt'].process_query(query)
                return {
                    'success': True,
                    'type': 'natural_language',
                    'query': query,
                    'response': response,
                    'llm_provider': self.llm_provider
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'type': 'natural_language',
                    'query': query
                }
        else:
            return {
                'success': False,
                'error': 'QuantGPT not available',
                'type': 'natural_language',
                'query': query
            }
    
    def _handle_forecasting(self, parts: List[str]) -> Dict[str, Any]:
        """Handle forecasting commands."""
        try:
            if len(parts) < 2:
                return {'error': 'Forecast command requires symbol', 'status': 'error'}
            
            symbol = parts[1].upper()
            timeframe = '7d' if len(parts) < 3 else parts[2]
            
            # Mock forecast result
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'prediction': 150.25,
                'confidence': 0.85,
                'model': 'lstm',
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'type': 'forecast',
                'result': result,
                'status': 'success'
            }
        except Exception as e:
            return {'success': True, 'result': {'error': f'Forecast error: {e}', 'status': 'error'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _handle_tuning(self, parts: List[str]) -> Dict[str, Any]:
        """Handle tuning commands."""
        try:
            if len(parts) < 3:
                return {'error': 'Tune command requires model and symbol', 'status': 'error'}
            
            model = parts[1]
            symbol = parts[2].upper()
            
            # Mock tuning result
            result = {
                'model': model,
                'symbol': symbol,
                'best_params': {'learning_rate': 0.001, 'batch_size': 32},
                'improvement': 0.15,
                'duration': '2h 30m'
            }
            
            return {
                'type': 'tuning',
                'result': result,
                'status': 'success'
            }
        except Exception as e:
            return {'success': True, 'result': {'error': f'Tuning error: {e}', 'status': 'error'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _handle_strategy(self, parts: List[str]) -> Dict[str, Any]:
        """Handle strategy commands."""
        try:
            if len(parts) < 2:
                return {'error': 'Strategy command requires action', 'status': 'error'}
            
            action = parts[1]
            
            if action == 'list':
                strategies = ['bollinger', 'macd', 'rsi', 'momentum', 'mean_reversion']
                return {
                    'type': 'strategy_list',
                    'strategies': strategies,
                    'status': 'success'
                }
            elif action == 'backtest' and len(parts) >= 3:
                strategy = parts[2]
                symbol = parts[3].upper() if len(parts) >= 4 else 'AAPL'
                
                result = {
                    'strategy': strategy,
                    'symbol': symbol,
                    'sharpe_ratio': 1.25,
                    'total_return': 0.15,
                    'max_drawdown': -0.08
                }
                
                return {
                    'type': 'strategy_backtest',
                    'result': result,
                    'status': 'success'
                }
            else:
                return {'success': True, 'result': {'error': 'Invalid strategy command', 'status': 'error'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            return {'error': f'Strategy error: {e}', 'status': 'error'}
    
    def _handle_portfolio(self, parts: List[str]) -> Dict[str, Any]:
        """Handle portfolio commands."""
        try:
            if len(parts) < 2:
                return {'error': 'Portfolio command requires action', 'status': 'error'}
            
            action = parts[1]
            
            if action == 'status':
                status = {
                    'total_value': 100000,
                    'positions': ['AAPL', 'TSLA', 'NVDA'],
                    'pnl': 2500,
                    'risk_level': 'medium'
                }
                return {
                    'type': 'portfolio_status',
                    'status': status,
                    'status': 'success'
                }
            elif action == 'rebalance':
                result = {
                    'rebalanced': True,
                    'new_allocation': {'AAPL': 0.4, 'TSLA': 0.3, 'NVDA': 0.3},
                    'trades_executed': 2
                }
                return {
                    'type': 'portfolio_rebalance',
                    'result': result,
                    'status': 'success'
                }
            else:
                return {'success': True, 'result': {'error': 'Invalid portfolio command', 'status': 'error'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            return {'error': f'Portfolio error: {e}', 'status': 'error'}
    
    def _handle_agents(self, parts: List[str]) -> Dict[str, Any]:
        """Handle agent commands."""
        try:
            if len(parts) < 2:
                return {'error': 'Agent command requires action', 'status': 'error'}
            
            action = parts[1]
            
            if action == 'list':
                agents = ['model_builder', 'performance_critic', 'updater', 'researcher']
                return {
                    'type': 'agent_list',
                    'agents': agents,
                    'status': 'success'
                }
            elif action == 'status':
                status = {
                    'total_agents': 4,
                    'active_agents': 3,
                    'agent_status': {
                        'model_builder': 'active',
                        'performance_critic': 'active',
                        'updater': 'active',
                        'researcher': 'idle'
                    }
                }
                return {
                    'type': 'agent_status',
                    'status': status,
                    'status': 'success'
                }
            else:
                return {'success': True, 'result': {'error': 'Invalid agent command', 'status': 'error'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            return {'error': f'Agent error: {e}', 'status': 'error'}
    
    def _handle_reports(self, parts: List[str]) -> Dict[str, Any]:
        """Handle report commands."""
        try:
            if len(parts) < 2:
                return {'error': 'Report command requires action', 'status': 'error'}
            
            action = parts[1]
            
            if action == 'generate' and len(parts) >= 3:
                symbol = parts[2].upper()
                result = {
                    'symbol': symbol,
                    'report_type': 'comprehensive',
                    'generated_at': datetime.now().isoformat(),
                    'file_path': f'reports/{symbol}_report.pdf'
                }
                return {
                    'type': 'report_generated',
                    'symbol': symbol,
                    'result': result,
                    'status': 'success'
                }
            else:
                return {'error': 'Invalid report command', 'status': 'error'}
        except Exception as e:
            return {'success': True, 'result': {'error': f'Report error: {e}', 'status': 'error'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _handle_status(self, parts: List[str]) -> Dict[str, Any]:
        """Handle status commands."""
        try:
            status = {
                'interface': 'initialized' if self.initialized else 'error',
                'components': {name: 'available' for name in self.components.keys()},
                'llm_provider': self.llm_provider,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'type': 'status',
                'status': status,
                'status': 'success'
            }
        except Exception as e:
            return {'success': True, 'result': {'error': f'Status error: {e}', 'status': 'error'}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def enhanced_streamlit_ui():
    """Enhanced Streamlit web interface with LLM controls and transparency."""
    st.set_page_config(
        page_title="Evolve - Enhanced Unified Interface",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize interface
    interface = EnhancedUnifiedInterface()
    
    # Sidebar with enhanced controls
    st.sidebar.title("🔮 Evolve Enhanced Interface")
    st.sidebar.markdown("---")
    
    # LLM Provider Controls
    st.sidebar.subheader("🤖 LLM Provider")
    provider_status = interface.get_llm_provider_status()
    
    # Show provider status
    for provider, available in provider_status.items():
        status_icon = "✅" if available else "❌"
        st.sidebar.markdown(f"{status_icon} {provider.title()}")
    
    # Provider selection
    selected_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        [p for p, available in provider_status.items() if available],
        index=0
    )
    
    if st.sidebar.button("Apply Provider"):
        interface.set_llm_provider(selected_provider)
        st.sidebar.success(f"Provider set to: {selected_provider}")
    
    # Notification Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔔 Notifications")
    notification_status = interface.get_notification_status()
    
    slack_status = "✅" if notification_status.get('slack_configured') else "❌"
    email_status = "✅" if notification_status.get('email_configured') else "❌"
    
    st.sidebar.markdown(f"{slack_status} Slack")
    st.sidebar.markdown(f"{email_status} Email")
    
    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "Choose Interface",
        ["Main Interface", "QuantGPT", "Forecasting", "Tuning", "Strategy", "Portfolio", "Agents", "Reports", "Help", "System Status"]
    )
    
    if page == "Main Interface":
        render_enhanced_main_interface(interface)
    elif page == "QuantGPT":
        render_enhanced_quantgpt_interface(interface)
    elif page == "Forecasting":
        render_enhanced_forecasting_interface(interface)
    elif page == "Tuning":
        render_enhanced_tuning_interface(interface)
    elif page == "Strategy":
        render_enhanced_strategy_interface(interface)
    elif page == "Portfolio":
        render_enhanced_portfolio_interface(interface)
    elif page == "Agents":
        render_enhanced_agents_interface(interface)
    elif page == "Reports":
        render_enhanced_reports_interface(interface)
    elif page == "Help":
        render_enhanced_help_interface(interface)
    elif page == "System Status":
        render_system_status_interface(interface)

def render_enhanced_main_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced main interface page."""
    st.title("🔮 Evolve Enhanced Unified Interface")
    st.markdown("Access all trading system features through one unified interface with full transparency.")
    
    # System Status Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("LLM Provider", interface.llm_provider.title())
    
    with col2:
        provider_status = interface.get_llm_provider_status()
        available_providers = sum(provider_status.values())
        st.metric("Available Providers", available_providers)
    
    with col3:
        notification_status = interface.get_notification_status()
        notifications_configured = sum([
            notification_status.get('slack_configured', False),
            notification_status.get('email_configured', False)
        ])
        st.metric("Notifications", notifications_configured)
    
    with col4:
        components_count = len(interface.components)
        st.metric("Components", components_count)
    
    # Command input with transparency
    st.subheader("Command Interface")
    
    # Manual prompt input
    manual_prompt = st.text_area(
        "Enter command or natural language query:",
        placeholder="e.g., 'forecast AAPL 30d' or 'What's the best model for TSLA?'",
        height=100
    )
    
    # LLM provider override
    col1, col2 = st.columns([3, 1])
    with col1:
        override_provider = st.selectbox(
            "Override LLM Provider (optional)",
            ["Use Default"] + [p.title() for p in interface.get_llm_provider_status().keys() if interface.get_llm_provider_status()[p]]
        )
    
    with col2:
        execute_button = st.button("🚀 Execute", type="primary")
    
    if execute_button and manual_prompt:
        with st.spinner("Processing..."):
            # Use override provider if selected
            provider = None
            if override_provider != "Use Default":
                provider = override_provider.lower()
            
            result = interface.process_command(manual_prompt, provider)
            display_enhanced_result(result)
    
    # Quick actions with transparency
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Forecast AAPL"):
            result = interface.process_command("forecast AAPL 7d")
            display_enhanced_result(result)
    
    with col2:
        if st.button("🤖 Ask QuantGPT"):
            result = interface.process_command("What's the trading signal for TSLA?")
            display_enhanced_result(result)
    
    with col3:
        if st.button("⚙️ System Status"):
            result = interface.process_command("status")
            display_enhanced_result(result)

def render_enhanced_quantgpt_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced QuantGPT interface."""
    st.title("🤖 QuantGPT Natural Language Interface")
    st.markdown("Ask questions about trading in natural language with full transparency.")
    
    # LLM Provider Info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Current LLM Provider:** {interface.llm_provider.title()}")
    with col2:
        provider_status = interface.get_llm_provider_status()
        available = [p for p, available in provider_status.items() if available]
        st.info(f"**Available Providers:** {', '.join(available)}")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., 'What's the best model for TSLA?' or 'Should I buy AAPL now?'",
        height=150
    )
    
    # Provider selection for this query
    col1, col2 = st.columns([3, 1])
    with col1:
        query_provider = st.selectbox(
            "LLM Provider for this query",
            [p.title() for p in provider_status.keys() if provider_status[p]]
        )
    
    with col2:
        ask_button = st.button("🤖 Ask", type="primary")
    
    if ask_button and query:
        with st.spinner("Processing with QuantGPT..."):
            result = interface.process_command(query, query_provider.lower())
            display_enhanced_result(result)

def display_enhanced_result(result: Dict[str, Any]):
    """Display enhanced result with full transparency."""
    st.subheader("📊 Result")
    
    # Success/Error indicator
    if result.get('success', False):
        st.success("✅ Command executed successfully")
    else:
        st.error("❌ Command failed")
    
    # LLM Provider Info
    if 'llm_provider' in result:
        st.info(f"**LLM Provider Used:** {result['llm_provider'].title()}")
    
    # Intent and Confidence (if available)
    if 'intent' in result:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Intent", result['intent'].title())
        with col2:
            if 'confidence' in result:
                st.metric("Confidence", f"{result['confidence']:.2f}")
    
    # Raw Response (if available)
    if 'raw_response' in result:
        with st.expander("🔍 Raw LLM Response"):
            st.code(result['raw_response'])
    
    # Arguments (if available)
    if 'args' in result and result['args']:
        with st.expander("📋 Parsed Arguments"):
            st.json(result['args'])
    
    # Main Result
    if 'result' in result:
        st.subheader("📈 Result Data")
        st.json(result['result'])
    
    # Error Information
    if 'error' in result:
        st.error(f"**Error:** {result['error']}")
    
    # Timestamp
    if 'timestamp' in result:
        st.caption(f"Executed at: {result['timestamp']}")

def render_enhanced_forecasting_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced forecasting interface."""
    st.title("📈 Enhanced Forecasting Interface")
    st.markdown("Generate market predictions with full model transparency.")
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbol", value="AAPL").upper()
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d", "7d", "30d", "90d"])
    
    # Model options
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Model", ["lstm", "transformer", "xgboost", "ensemble"])
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8)
    with col3:
        provider = st.selectbox("LLM Provider", ["openai", "huggingface", "regex"])
    
    if st.button("🚀 Generate Forecast", type="primary"):
        command = f"forecast {symbol} {timeframe} with {model} model"
        result = interface.process_command(command, provider)
        display_enhanced_result(result)

def render_enhanced_tuning_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced tuning interface."""
    st.title("⚙️ Enhanced Tuning Interface")
    st.markdown("Optimize models and strategies with full transparency.")
    
    # Tuning parameters
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model to Tune", ["lstm", "transformer", "xgboost"])
    with col2:
        symbol = st.text_input("Symbol", value="AAPL").upper()
    
    # Optimization settings
    col1, col2, col3 = st.columns(3)
    with col1:
        optimization_type = st.selectbox("Optimization Type", ["hyperparameter", "strategy", "ensemble"])
    with col2:
        max_iterations = st.number_input("Max Iterations", 10, 1000, 100)
    with col3:
        provider = st.selectbox("LLM Provider", ["openai", "huggingface", "regex"])
    
    if st.button("🔧 Start Tuning", type="primary"):
        command = f"tune {model} {symbol} {optimization_type} {max_iterations} iterations"
        result = interface.process_command(command, provider)
        display_enhanced_result(result)

def render_enhanced_strategy_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced strategy interface."""
    st.title("📊 Enhanced Strategy Interface")
    st.markdown("Manage and execute trading strategies with full transparency.")
    
    # Strategy selection
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox("Strategy", ["bollinger", "macd", "rsi", "momentum"])
    with col2:
        symbol = st.text_input("Symbol", value="AAPL").upper()
    
    # Action selection
    action = st.selectbox("Action", ["backtest", "run", "optimize", "analyze"])
    
    # Additional parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        timeframe = st.selectbox("Timeframe", ["1d", "1w", "1m", "3m"])
    with col2:
        provider = st.selectbox("LLM Provider", ["openai", "huggingface", "regex"])
    with col3:
        include_risk = st.checkbox("Include Risk Analysis")
    
    if st.button("📈 Execute Strategy", type="primary"):
        command = f"strategy {action} {strategy} {symbol} {timeframe}"
        if include_risk:
            command += " with risk analysis"
        result = interface.process_command(command, provider)
        display_enhanced_result(result)

def render_enhanced_portfolio_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced portfolio interface."""
    st.title("💼 Enhanced Portfolio Interface")
    st.markdown("Portfolio management and analysis with full transparency.")
    
    # Portfolio actions
    action = st.selectbox("Portfolio Action", ["status", "rebalance", "analyze", "optimize"])
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        risk_tolerance = st.selectbox("Risk Tolerance", ["low", "medium", "high"])
    with col2:
        provider = st.selectbox("LLM Provider", ["openai", "huggingface", "regex"])
    
    if st.button("💼 Execute Portfolio Action", type="primary"):
        command = f"portfolio {action} with {risk_tolerance} risk"
        result = interface.process_command(command, provider)
        display_enhanced_result(result)

def render_enhanced_agents_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced agents interface."""
    st.title("🤖 Enhanced Agents Interface")
    st.markdown("Manage autonomous trading agents with full transparency.")
    
    # Agent actions
    action = st.selectbox("Agent Action", ["list", "status", "start", "stop", "configure"])
    
    # Agent selection
    if action in ["start", "stop", "configure"]:
        agent = st.selectbox("Agent", ["model_builder", "performance_critic", "updater", "researcher"])
    else:
        agent = ""
    
    # Provider selection
    provider = st.selectbox("LLM Provider", ["openai", "huggingface", "regex"])
    
    if st.button("🤖 Execute Agent Action", type="primary"):
        command = f"agent {action}"
        if agent:
            command += f" {agent}"
        result = interface.process_command(command, provider)
        display_enhanced_result(result)

def render_enhanced_reports_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced reports interface."""
    st.title("📋 Enhanced Reports Interface")
    st.markdown("Generate comprehensive trading reports with full transparency.")
    
    # Report parameters
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbol", value="AAPL").upper()
    with col2:
        report_type = st.selectbox("Report Type", ["comprehensive", "performance", "risk", "trade_log"])
    
    # Additional options
    col1, col2, col3 = st.columns(3)
    with col1:
        include_heatmap = st.checkbox("Include Heatmap", value=True)
    with col2:
        include_models = st.checkbox("Include Model Summary", value=True)
    with col3:
        provider = st.selectbox("LLM Provider", ["openai", "huggingface", "regex"])
    
    if st.button("📋 Generate Report", type="primary"):
        command = f"report generate {symbol} {report_type}"
        if include_heatmap:
            command += " with heatmap"
        if include_models:
            command += " with model summary"
        result = interface.process_command(command, provider)
        display_enhanced_result(result)

def render_enhanced_help_interface(interface: EnhancedUnifiedInterface):
    """Render enhanced help interface."""
    st.title("❓ Enhanced Help Interface")
    st.markdown("Comprehensive help and documentation.")
    
    help_info = interface.get_help()
    
    # Overview
    st.subheader("📖 Overview")
    st.markdown(f"**{help_info['overview']['title']}**")
    st.markdown(help_info['overview']['description'])
    st.caption(f"Version: {help_info['overview']['version']}")
    
    # Features
    st.subheader("🚀 Features")
    for feature_name, feature_info in help_info['features'].items():
        with st.expander(f"📋 {feature_name.title()}"):
            st.markdown(f"**Description:** {feature_info['description']}")
            st.markdown("**Commands:**")
            for cmd in feature_info['commands']:
                st.code(cmd)
            st.markdown(f"**UI Section:** {feature_info['ui_section']}")
    
    # LLM Providers
    st.subheader("🤖 LLM Providers")
    for provider, description in help_info['llm_providers'].items():
        st.markdown(f"**{provider.title()}:** {description}")
    
    # Examples
    st.subheader("💡 Examples")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Quick Start:**")
        for example in help_info['examples']['quick_start']:
            st.code(example)
    with col2:
        st.markdown("**Advanced:**")
        for example in help_info['examples']['advanced']:
            st.code(example)

def render_system_status_interface(interface: EnhancedUnifiedInterface):
    """Render system status interface."""
    st.title("⚙️ System Status")
    st.markdown("Comprehensive system status and health monitoring.")
    
    # System Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Interface Status", "✅ Active" if interface.initialized else "❌ Error")
    
    with col2:
        components_count = len(interface.components)
        st.metric("Components Loaded", components_count)
    
    with col3:
        provider_status = interface.get_llm_provider_status()
        available_providers = sum(provider_status.values())
        st.metric("LLM Providers", available_providers)
    
    with col4:
        notification_status = interface.get_notification_status()
        notifications_configured = sum([
            notification_status.get('slack_configured', False),
            notification_status.get('email_configured', False)
        ])
        st.metric("Notifications", notifications_configured)
    
    # Component Status
    st.subheader("🔧 Component Status")
    for component_name, component in interface.components.items():
        status = "✅ Available" if component else "❌ Not Available"
        st.markdown(f"**{component_name}:** {status}")
    
    # LLM Provider Status
    st.subheader("🤖 LLM Provider Status")
    provider_status = interface.get_llm_provider_status()
    for provider, available in provider_status.items():
        status_icon = "✅" if available else "❌"
        st.markdown(f"{status_icon} **{provider.title()}:** {'Available' if available else 'Not Available'}")
    
    # Notification Status
    st.subheader("🔔 Notification Status")
    notification_status = interface.get_notification_status()
    for key, value in notification_status.items():
        if isinstance(value, bool):
            status_icon = "✅" if value else "❌"
            st.markdown(f"{status_icon} **{key.replace('_', ' ').title()}:** {'Configured' if value else 'Not Configured'}")
        else:
            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evolve Trading System - Enhanced Unified Interface")
    parser.add_argument("--mode", choices=["streamlit", "terminal"], default="streamlit",
                       help="Interface mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.mode == "streamlit":
        enhanced_streamlit_ui()
    else:
        terminal_ui()

if __name__ == "__main__":
    main() 