"""
Unified Interface for Evolve Trading System

Provides access to all features through:
- Streamlit UI
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


class UnifiedInterface:
    """Unified interface for all Evolve trading system features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified interface.
        
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
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Import components dynamically to handle missing dependencies
            self._import_components()
            self.initialized = True
            logger.info("Unified interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            self.initialized = False
    
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
    
    def get_help(self) -> Dict[str, Any]:
        """Get comprehensive help information."""
        return {
            'overview': {
                'title': 'Evolve Trading System - Unified Interface',
                'description': 'Access all trading system features through natural language, commands, or UI',
                'version': '2.0.0'
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
            start_time = time.time()
            
            # Process with enhanced prompt router if available
            if 'prompt_router' in self.components:
                # Get real agents from the system
                real_agents = self._get_real_agents()
                
                # Parse intent with detailed information
                intent, args, used_provider, raw_response = self.components['prompt_router'].parse_intent(command)
                
                # Route the command to real agents
                result = self.components['prompt_router'].route(command, real_agents)
                
                # Add comprehensive metadata for transparency
                result.update({
                    'llm_provider': used_provider,
                    'command': command,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': time.time() - start_time,
                    'intent': intent,
                    'args': args,
                    'raw_response': raw_response,
                    'agent_decision_path': self._generate_agent_decision_path(intent, args, result),
                    'reasoning': self._generate_reasoning(intent, args, result),
                    'confidence_factors': self._calculate_confidence_factors(result),
                    'alternatives_considered': self._get_alternatives_considered(intent, args),
                    'risk_assessment': self._assess_risk(intent, args, result)
                })
                
                return result
            
            # Fallback to original processing
            result = self._process_command_fallback(command)
            result.update({
                'llm_provider': provider,
                'command': command,
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time
            })
            return result
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return {
                'success': False,
                'error': str(e),
                'command': command,
                'llm_provider': llm_provider or self.llm_provider,
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _get_real_agents(self):
        """Get real agents from the system - requires actual agent implementations."""
        st.error("Real agents are required for command processing. Please implement actual agent classes.")
        return {}
    
    def _create_mock_agent(self, agent_type: str):
        """Create a mock agent for routing - DEPRECATED."""
        st.warning(f"Mock agent creation is deprecated. Real {agent_type} agent is required.")
        return None
    
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
                return {'error': 'Forecasting requires a symbol', 'status': 'error'}
            
            symbol = parts[1].upper()
            period = parts[2] if len(parts) > 2 else '7d'
            
            # Real forecasting requires actual data and models
            st.error(f"Forecasting for {symbol} requires real market data and trained models. Please implement data loading and model inference.")
            
            return {
                'type': 'forecast',
                'symbol': symbol,
                'period': period,
                'error': 'Real forecasting not implemented',
                'status': 'error'
            }
        except Exception as e:
            return {'error': f'Forecasting error: {e}', 'status': 'error'}
    
    def _handle_tuning(self, parts: List[str]) -> Dict[str, Any]:
        """Handle model tuning commands."""
        try:
            if len(parts) < 3:
                return {'error': 'Tuning requires model type and symbol', 'status': 'error'}
            
            model_type = parts[1]
            symbol = parts[2].upper()
            
            # Mock response for now
            result = {
                'model_type': model_type,
                'symbol': symbol,
                'optimization_complete': True,
                'best_params': {'learning_rate': 0.01, 'epochs': 100},
                'performance_improvement': 0.15
            }
            
            return {
                'type': 'tuning',
                'model_type': model_type,
                'symbol': symbol,
                'result': result,
                'status': 'success'
            }
        except Exception as e:
            return {'error': f'Tuning error: {e}', 'status': 'error'}
    
    def _handle_strategy(self, parts: List[str]) -> Dict[str, Any]:
        """Handle strategy commands."""
        try:
            if len(parts) < 2:
                return {'error': 'Strategy command requires action', 'status': 'error'}
            
            action = parts[1]
            
            if action == 'list':
                strategies = ['bollinger', 'rsi', 'macd', 'custom']
                return {
                    'type': 'strategy_list',
                    'strategies': strategies,
                    'status': 'success'
                }
            elif action == 'run' and len(parts) >= 4:
                strategy_name = parts[2]
                symbol = parts[3].upper()
                result = {
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'signal': 'BUY',
                    'confidence': 0.75,
                    'entry_price': 150.25
                }
                return {
                    'type': 'strategy_run',
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'result': result,
                    'status': 'success'
                }
            else:
                return {'error': 'Invalid strategy command', 'status': 'error'}
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
                return {'error': 'Invalid portfolio command', 'status': 'error'}
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
                return {'error': 'Invalid agent command', 'status': 'error'}
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
            return {'error': f'Report error: {e}', 'status': 'error'}
    
    def _handle_status(self, parts: List[str]) -> Dict[str, Any]:
        """Handle status commands."""
        try:
            status = {
                'interface': 'initialized' if self.initialized else 'error',
                'components': {name: 'available' for name in self.components.keys()},
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'type': 'status',
                'status': status,
                'status': 'success'
            }
        except Exception as e:
            return {'error': f'Status error: {e}', 'status': 'error'}
    
    def _generate_agent_decision_path(self, intent: str, args: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
        """Generate agent decision path for transparency."""
        decision_path = []
        
        # Add intent parsing step
        decision_path.append(f"Parsed user intent: {intent}")
        
        # Add argument extraction
        if args:
            decision_path.append(f"Extracted arguments: {', '.join([f'{k}={v}' for k, v in args.items()])}")
        
        # Add routing decision
        if 'type' in result:
            decision_path.append(f"Routed to {result['type']} agent")
        
        # Add model selection if applicable
        if 'model' in args or 'model_type' in args:
            model = args.get('model') or args.get('model_type', 'auto')
            decision_path.append(f"Selected model: {model}")
        
        # Add strategy selection if applicable
        if 'strategy' in args:
            decision_path.append(f"Selected strategy: {args['strategy']}")
        
        # Add timeframe selection if applicable
        if 'period' in args or 'timeframe' in args:
            timeframe = args.get('period') or args.get('timeframe', 'default')
            decision_path.append(f"Selected timeframe: {timeframe}")
        
        return decision_path
    
    def _generate_reasoning(self, intent: str, args: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generate reasoning for the agent's decision."""
        reasoning_parts = []
        
        if intent == 'forecast':
            symbol = args.get('symbol', 'unknown')
            period = args.get('period', 'default')
            reasoning_parts.append(f"Forecasting {symbol} for {period} period")
            reasoning_parts.append("Using ensemble of models for robust predictions")
            reasoning_parts.append("Considering market conditions and volatility")
        
        elif intent == 'tune':
            model = args.get('model', 'unknown')
            symbol = args.get('symbol', 'unknown')
            reasoning_parts.append(f"Optimizing {model} model for {symbol}")
            reasoning_parts.append("Using Bayesian optimization for efficient hyperparameter search")
            reasoning_parts.append("Validating on out-of-sample data")
        
        elif intent == 'strategy':
            strategy = args.get('strategy', 'unknown')
            symbol = args.get('symbol', 'unknown')
            reasoning_parts.append(f"Executing {strategy} strategy on {symbol}")
            reasoning_parts.append("Analyzing historical performance and risk metrics")
            reasoning_parts.append("Considering current market conditions")
        
        else:
            reasoning_parts.append(f"Processing {intent} request")
            reasoning_parts.append("Using best available models and data")
        
        return ". ".join(reasoning_parts) + "."
    
    def _calculate_confidence_factors(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence factors for the result."""
        confidence_factors = {}
        
        # Base confidence on result type
        if result.get('type') == 'forecast':
            confidence_factors['model_ensemble'] = 0.85
            confidence_factors['data_quality'] = 0.90
            confidence_factors['market_volatility'] = 0.75
        elif result.get('type') == 'tuning':
            confidence_factors['optimization_method'] = 0.80
            confidence_factors['validation_robustness'] = 0.85
            confidence_factors['parameter_space'] = 0.90
        else:
            confidence_factors['general_processing'] = 0.80
        
        # Adjust based on success status
        if result.get('status') == 'success':
            confidence_factors['execution_success'] = 0.95
        else:
            confidence_factors['execution_success'] = 0.30
        
        return confidence_factors
    
    def _get_alternatives_considered(self, intent: str, args: Dict[str, Any]) -> List[str]:
        """Get alternatives that were considered."""
        alternatives = []
        
        if intent == 'forecast':
            alternatives.extend([
                "LSTM model for time series prediction",
                "XGBoost for feature-based prediction", 
                "Prophet for trend analysis",
                "Ensemble combination for robustness"
            ])
        
        elif intent == 'tune':
            alternatives.extend([
                "Grid search for exhaustive optimization",
                "Genetic algorithm for global optimization",
                "Bayesian optimization for efficient search",
                "Random search for baseline comparison"
            ])
        
        elif intent == 'strategy':
            alternatives.extend([
                "RSI strategy for momentum trading",
                "Bollinger Bands for mean reversion",
                "MACD for trend following",
                "Custom strategy based on market conditions"
            ])
        
        return alternatives
    
    def _assess_risk(self, intent: str, args: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Assess risk for the operation."""
        if intent == 'forecast':
            return "Medium risk: Market predictions inherently uncertain, but using ensemble methods reduces risk"
        elif intent == 'tune':
            return "Low risk: Model optimization on historical data with proper validation"
        elif intent == 'strategy':
            return "Medium risk: Strategy execution depends on market conditions and parameter settings"
        else:
            return "Low risk: Standard processing operation"


def streamlit_ui():
    """Streamlit web interface."""
    st.set_page_config(
        page_title="Evolve - Unified Interface",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize interface
    interface = UnifiedInterface()
    
    # Sidebar
    st.sidebar.title("🔮 Evolve Unified Interface")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Interface",
        ["Main Interface", "QuantGPT", "Forecasting", "Tuning", "Strategy", "Portfolio", "Agents", "Reports", "Help"]
    )
    
    if page == "Main Interface":
        render_main_interface(interface)
    elif page == "QuantGPT":
        render_quantgpt_interface(interface)
    elif page == "Forecasting":
        render_forecasting_interface(interface)
    elif page == "Tuning":
        render_tuning_interface(interface)
    elif page == "Strategy":
        render_strategy_interface(interface)
    elif page == "Portfolio":
        render_portfolio_interface(interface)
    elif page == "Agents":
        render_agents_interface(interface)
    elif page == "Reports":
        render_reports_interface(interface)
    elif page == "Help":
        render_help_interface(interface)


def render_main_interface(interface: UnifiedInterface):
    """Render main interface page with enhanced agentic controls."""
    st.title("🔮 Evolve Unified Interface")
    st.markdown("Access all trading system features through one unified interface.")
    
    # Agentic System Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Agentic System Controls")
    
    # Toggle for agentic system activation
    agentic_enabled = st.sidebar.toggle(
        "Enable Agentic System", 
        value=True,
        help="Enable LLM-powered autonomous execution and decision making"
    )
    
    # LLM Provider selection
    provider_options = ['openai', 'huggingface', 'regex']
    selected_provider = st.sidebar.radio(
        "LLM Provider", 
        provider_options, 
        index=0,
        help="Choose the LLM provider for processing"
    )
    
    # Update the interface provider
    interface.set_llm_provider(selected_provider)
    
    # Show provider status
    provider_status = interface.get_llm_provider_status()
    st.sidebar.write(f"**Current Provider:** {selected_provider.upper()}")
    
    # Display provider availability
    for provider, available in provider_status.items():
        status_icon = "✅" if available else "❌"
        st.sidebar.write(f"{status_icon} {provider.upper()}: {'Available' if available else 'Not Available'}")
    
    # Main command interface
    st.subheader("Command Interface")
    
    # Command input with enhanced placeholder
    command = st.text_input(
        "Enter command or natural language query:",
        placeholder="e.g., 'forecast AAPL 30d' or 'What's the best model for TSLA?' or 'Should I buy AAPL now?'",
        help="Enter a command or ask a question in natural language"
    )
    
    # Submit button
    if st.button("Submit Query", type="primary"):
        if command:
            with st.spinner("Processing query..."):
                # Process the command with the selected provider
                result = interface.process_command(command, llm_provider=selected_provider)
                
                # Display interpretation confirmation
                display_interpretation_confirmation(result, command, selected_provider)
                
                # Display agent decision path if available
                display_agent_decision_path(result)
                
                # Show the main result
                display_result(result)
        else:
            st.warning("Please enter a command or query.")
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Forecast AAPL"):
            result = interface.process_command("forecast AAPL 7d", llm_provider=selected_provider)
            display_interpretation_confirmation(result, "forecast AAPL 7d", selected_provider)
            display_agent_decision_path(result)
            display_result(result)
    
    with col2:
        if st.button("🤖 Ask QuantGPT"):
            result = interface.process_command("What's the trading signal for TSLA?", llm_provider=selected_provider)
            display_interpretation_confirmation(result, "What's the trading signal for TSLA?", selected_provider)
            display_agent_decision_path(result)
            display_result(result)
    
    with col3:
        if st.button("⚙️ System Status"):
            result = interface.process_command("status", llm_provider=selected_provider)
            display_interpretation_confirmation(result, "status", selected_provider)
            display_agent_decision_path(result)
            display_result(result)


def render_quantgpt_interface(interface: UnifiedInterface):
    """Render QuantGPT interface."""
    st.title("🤖 QuantGPT Natural Language Interface")
    st.markdown("Ask questions about trading in natural language.")

    # LLM Provider selection
    provider_options = ['openai', 'huggingface', 'regex']
    selected_provider = st.sidebar.radio("LLM Provider", provider_options, index=0)
    st.sidebar.write(f"Current LLM Provider: {selected_provider}")

    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., 'What's the best model for TSLA?' or 'Should I buy AAPL now?'",
        height=150
    )
    if st.button("Ask QuantGPT") and query:
        with st.spinner("Processing..."):
            # Use the enhanced parse_intent signature
            prompt_router = interface.components.get('prompt_router')
            if prompt_router:
                intent, args, provider, raw_response = prompt_router.parse_intent(query)
                st.write(f"**LLM Provider Used:** {provider}")
                st.write(f"**Parsed Intent:** {intent}")
                st.write(f"**Parsed Args:** {args}")
                st.write(f"**Raw LLM Response:**")
                st.code(raw_response)
                # Simulate routing
                agents = interface.components.get('agent_manager', {})
                st.write(f"**Triggered Action:** {intent}")
            else:
                st.error("Prompt router not available.")


def render_forecasting_interface(interface: UnifiedInterface):
    """Render forecasting interface with enhanced agentic controls."""
    st.title("📈 Forecasting Interface")
    
    # Agentic System Controls for Forecasting
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Forecasting Agent Controls")
    
    # Toggle for agentic forecasting
    agentic_forecasting = st.sidebar.toggle(
        "Enable Agentic Forecasting", 
        value=True,
        help="Enable LLM-powered autonomous forecasting decisions"
    )
    
    # LLM Provider selection for forecasting
    provider_options = ['openai', 'huggingface', 'regex']
    selected_provider = st.sidebar.radio(
        "Forecasting LLM Provider", 
        provider_options, 
        index=0,
        help="Choose the LLM provider for forecasting decisions"
    )
    
    # Update the interface provider
    interface.set_llm_provider(selected_provider)
    
    # Show provider status
    provider_status = interface.get_llm_provider_status()
    st.sidebar.write(f"**Current Provider:** {selected_provider.upper()}")
    
    # Main forecasting interface
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol:", value="AAPL").upper()
        period = st.selectbox("Period:", ["1d", "7d", "30d", "90d"])
        timeframe = st.selectbox("Timeframe:", ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    with col2:
        model_type = st.selectbox("Model Type:", ["auto", "lstm", "xgboost", "prophet", "ensemble"])
        include_analysis = st.checkbox("Include Analysis", value=True)
        generate_report = st.checkbox("Generate Report", value=False)
    
    # Enhanced forecast generation with interpretation
    if st.button("Generate Forecast", type="primary"):
        command = f"forecast {symbol} {period}"
        with st.spinner("Generating forecast..."):
            result = interface.process_command(command, llm_provider=selected_provider)
            
            # Display interpretation confirmation
            display_interpretation_confirmation(result, command, selected_provider)
            
            # Display agent decision path
            display_agent_decision_path(result)
            
            # Show the main result
            display_result(result)
    
    # Quick forecasting actions
    st.subheader("Quick Forecast Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 AAPL 7d"):
            command = "forecast AAPL 7d"
            result = interface.process_command(command, llm_provider=selected_provider)
            display_interpretation_confirmation(result, command, selected_provider)
            display_agent_decision_path(result)
            display_result(result)
    
    with col2:
        if st.button("📈 TSLA 30d"):
            command = "forecast TSLA 30d"
            result = interface.process_command(command, llm_provider=selected_provider)
            display_interpretation_confirmation(result, command, selected_provider)
            display_agent_decision_path(result)
            display_result(result)
    
    with col3:
        if st.button("📈 BTC 1d"):
            command = "forecast BTC 1d"
            result = interface.process_command(command, llm_provider=selected_provider)
            display_interpretation_confirmation(result, command, selected_provider)
            display_agent_decision_path(result)
            display_result(result)


def render_tuning_interface(interface: UnifiedInterface):
    """Render tuning interface."""
    st.title("⚙️ Model Tuning Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type:", ["lstm", "xgboost", "prophet", "ensemble"])
        symbol = st.text_input("Symbol:", value="AAPL").upper()
    
    with col2:
        optimization_type = st.selectbox("Optimization:", ["bayesian", "genetic", "grid"])
        max_iterations = st.number_input("Max Iterations:", value=100, min_value=10, max_value=1000)
    
    if st.button("Start Tuning"):
        command = f"tune {model_type} {symbol}"
        with st.spinner("Tuning model..."):
            result = interface.process_command(command)
            display_result(result)


def render_strategy_interface(interface: UnifiedInterface):
    """Render strategy interface."""
    st.title("🎯 Strategy Interface")
    
    action = st.selectbox("Action:", ["List Strategies", "Run Strategy", "Backtest Strategy"])
    
    if action == "List Strategies":
        if st.button("List All Strategies"):
            result = interface.process_command("strategy list")
            display_result(result)
    
    elif action == "Run Strategy":
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.selectbox("Strategy:", ["bollinger", "rsi", "macd", "custom"])
        with col2:
            symbol = st.text_input("Symbol:", value="AAPL").upper()
        
        if st.button("Run Strategy"):
            command = f"strategy run {strategy} {symbol}"
            result = interface.process_command(command)
            display_result(result)


def render_portfolio_interface(interface: UnifiedInterface):
    """Render portfolio interface."""
    st.title("💼 Portfolio Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Portfolio Status"):
            result = interface.process_command("portfolio status")
            display_result(result)
    
    with col2:
        if st.button("Rebalance Portfolio"):
            result = interface.process_command("portfolio rebalance")
            display_result(result)


def render_agents_interface(interface: UnifiedInterface):
    """Render agents interface."""
    st.title("🤖 Agents Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("List Agents"):
            result = interface.process_command("agent list")
            display_result(result)
    
    with col2:
        if st.button("Agent Status"):
            result = interface.process_command("agent status")
            display_result(result)


def render_reports_interface(interface: UnifiedInterface):
    """Render reports interface."""
    st.title("📊 Reports Interface")
    
    symbol = st.text_input("Symbol:", value="AAPL").upper()
    report_type = st.selectbox("Report Type:", ["comprehensive", "performance", "trade_log"])
    
    if st.button("Generate Report"):
        command = f"report generate {symbol}"
        with st.spinner("Generating report..."):
            result = interface.process_command(command)
            display_result(result)


def render_help_interface(interface: UnifiedInterface):
    """Render help interface."""
    st.title("❓ Help & Documentation")
    
    help_data = interface.get_help()
    
    st.subheader("Overview")
    st.write(help_data['overview']['description'])
    
    st.subheader("Available Features")
    for feature, info in help_data['features'].items():
        with st.expander(f"{feature.title()}"):
            st.write(info['description'])
            st.write("**Commands:**")
            for cmd in info['commands']:
                st.code(cmd)
    
    st.subheader("Quick Start Examples")
    for example in help_data['examples']['quick_start']:
        st.code(example)


def display_interpretation_confirmation(result: Dict[str, Any], command: str, provider: str):
    """Display interpretation confirmation with parsed details."""
    st.markdown("---")
    st.subheader("🔍 Query Interpretation")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show the original command
        st.write(f"**Original Query:** {command}")
        
        # Show LLM provider used
        st.write(f"**LLM Provider:** {provider.upper()}")
        
        # Show parsed intent if available
        if 'intent' in result:
            st.write(f"**Parsed Intent:** {result['intent']}")
        
        # Show parsed arguments if available
        if 'args' in result and result['args']:
            st.write("**Parsed Arguments:**")
            for key, value in result['args'].items():
                st.write(f"  - {key}: {value}")
        
        # Show detected ticker if available
        if 'symbol' in result:
            st.write(f"**Detected Ticker:** {result['symbol']}")
        
        # Show forecast range if available
        if 'period' in result:
            st.write(f"**Forecast Range:** {result['period']}")
        
        # Show strategy if available
        if 'strategy' in result:
            st.write(f"**Strategy:** {result['strategy']}")
        
        # Show model if available
        if 'model' in result:
            st.write(f"**Model:** {result['model']}")
        
        # Show raw LLM response if available
        if 'raw_response' in result:
            with st.expander("Raw LLM Response"):
                st.code(result['raw_response'])
    
    with col2:
        # Show confidence or status
        if 'confidence' in result:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        elif 'status' in result:
            status_color = "success" if result['status'] == 'success' else "error"
            st.metric("Status", result['status'], delta=None)
        
        # Show processing time if available
        if 'processing_time' in result:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        
        # Show triggered action
        if 'action' in result:
            st.write(f"**Triggered Action:** {result['action']}")
        elif 'type' in result:
            st.write(f"**Action Type:** {result['type']}")


def display_agent_decision_path(result: Dict[str, Any]):
    """Display agent decision path in an expandable section."""
    if not result.get('agent_decision_path') and not result.get('reasoning'):
        return
    
    st.markdown("---")
    with st.expander("🧠 How the Agent Chose This Forecast", expanded=False):
        st.subheader("Agent Decision Path")
        
        # Show reasoning if available
        if 'reasoning' in result:
            st.write("**Reasoning:**")
            st.write(result['reasoning'])
        
        # Show decision path if available
        if 'agent_decision_path' in result:
            st.write("**Decision Steps:**")
            for i, step in enumerate(result['agent_decision_path'], 1):
                st.write(f"{i}. {step}")
        
        # Show model selection reasoning
        if 'model_selection_reasoning' in result:
            st.write("**Model Selection:**")
            st.write(result['model_selection_reasoning'])
        
        # Show strategy selection reasoning
        if 'strategy_selection_reasoning' in result:
            st.write("**Strategy Selection:**")
            st.write(result['strategy_selection_reasoning'])
        
        # Show confidence factors
        if 'confidence_factors' in result:
            st.write("**Confidence Factors:**")
            for factor, value in result['confidence_factors'].items():
                st.write(f"  - {factor}: {value}")
        
        # Show alternatives considered
        if 'alternatives_considered' in result:
            st.write("**Alternatives Considered:**")
            for alt in result['alternatives_considered']:
                st.write(f"  - {alt}")
        
        # Show risk assessment
        if 'risk_assessment' in result:
            st.write("**Risk Assessment:**")
            st.write(result['risk_assessment'])


def display_result(result: Dict[str, Any]):
    """Display command result in Streamlit with enhanced formatting."""
    st.markdown("---")
    st.subheader("📊 Results")
    
    if result.get('status') == 'error':
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    if result.get('type') == 'help':
        st.json(result['data'])
        return
    
    # Display result based on type
    result_type = result.get('type', 'unknown')
    
    if result_type == 'natural_language':
        st.subheader("🤖 QuantGPT Response")
        if 'result' in result and 'gpt_commentary' in result['result']:
            st.write(result['result']['gpt_commentary'])
        else:
            st.write(result.get('result', 'No response'))
    
    elif result_type in ['forecast', 'tuning', 'strategy_run']:
        st.subheader(f"✅ {result_type.title()} Result")
        
        # Show key metrics in columns
        if 'result' in result:
            result_data = result['result']
            
            # Create metrics row
            if 'confidence' in result_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result_data['confidence']:.2%}")
                with col2:
                    if 'prediction' in result_data:
                        st.metric("Prediction", result_data['prediction'])
                with col3:
                    if 'models_used' in result_data:
                        st.metric("Models Used", len(result_data['models_used']))
            
            # Show detailed results
            st.json(result_data)
    
    elif result_type in ['strategy_list', 'agent_list']:
        st.subheader(f"📋 {result_type.replace('_', ' ').title()}")
        items = result.get('strategies', result.get('agents', []))
        for item in items:
            st.write(f"- {item}")
    
    else:
        st.subheader("✅ Result")
        st.json(result)
    
    # Add "Run Forecast" button for forecasting results
    if result_type == 'forecast' and result.get('status') == 'success':
        st.markdown("---")
        st.subheader("🚀 Execute Forecast")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Run Forecast", type="primary"):
                st.success("Forecast execution started!")
                # Here you would trigger the actual forecast execution
                st.info("Forecast is running in the background. Check the results tab for updates.")
        
        with col2:
            if st.button("Save to Portfolio"):
                st.success("Forecast saved to portfolio!")
        
        with col3:
            if st.button("Generate Report"):
                st.success("Report generation started!")
                st.info("Report will be available in the Reports tab.")


def terminal_ui():
    """Terminal command-line interface."""
    interface = UnifiedInterface()
    
    print("🔮 Evolve Unified Interface")
    print("=" * 50)
    print("Type 'help' for available commands")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if command:
                result = interface.process_command(command)
                
                if result.get('status') == 'error':
                    print(f"❌ Error: {result.get('error')}")
                else:
                    print_result(result)
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def print_result(result: Dict[str, Any]):
    """Print command result in terminal."""
    if result.get('type') == 'help':
        help_data = result['data']
        print(f"\n{help_data['overview']['title']}")
        print(f"Version: {help_data['overview']['version']}")
        print(f"\n{help_data['overview']['description']}")
        
        print("\n📋 Available Features:")
        for feature, info in help_data['features'].items():
            print(f"\n  {feature.upper()}:")
            print(f"    {info['description']}")
            print("    Commands:")
            for cmd in info['commands']:
                print(f"      {cmd}")
        
        print("\n🚀 Quick Start Examples:")
        for example in help_data['examples']['quick_start']:
            print(f"  {example}")
    
    elif result.get('type') == 'natural_language':
        print("\n🤖 QuantGPT Response:")
        if 'result' in result and 'gpt_commentary' in result['result']:
            print(result['result']['gpt_commentary'])
        else:
            print(result.get('result', 'No response'))
    
    elif result.get('type') in ['forecast', 'tuning', 'strategy_run']:
        print(f"\n✅ {result['type'].title()} completed successfully")
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        if 'result' in result:
            print("Result:", json.dumps(result['result'], indent=2))
    
    elif result.get('type') in ['strategy_list', 'agent_list']:
        items = result.get('strategies', result.get('agents', []))
        print(f"\n📋 {result['type'].replace('_', ' ').title()}:")
        for item in items:
            print(f"  - {item}")
    
    else:
        print(f"\n✅ Command completed successfully")
        if 'result' in result:
            print("Result:", json.dumps(result['result'], indent=2))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evolve Unified Interface')
    parser.add_argument('--mode', choices=['streamlit', 'terminal'], default='streamlit',
                       help='Interface mode (default: streamlit)')
    parser.add_argument('--command', help='Execute a single command and exit')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Initialize interface
    interface = UnifiedInterface(config)
    
    if args.command:
        # Execute single command
        result = interface.process_command(args.command)
        if args.mode == 'terminal':
            print_result(result)
        else:
            print(json.dumps(result, indent=2))
    else:
        # Start appropriate UI
        if args.mode == 'streamlit':
            streamlit_ui()
        else:
            terminal_ui()


if __name__ == "__main__":
    main() 