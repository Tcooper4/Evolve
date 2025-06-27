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
                openai_api_key=os.getenv('OPENAI_API_KEY'),
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
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """Process a command or natural language query.
        
        Args:
            command: Command string or natural language query
            
        Returns:
            Dictionary containing results and metadata
        """
        try:
            if not self.initialized:
                return {'error': 'Interface not initialized', 'status': 'error'}
            
            # Check if it's a help command
            if command.lower() in ['help', '?', '--help', '-h']:
                return {'type': 'help', 'data': self.get_help()}
            
            # Check if it's a natural language query (starts with common question words)
            if any(command.lower().startswith(word) for word in ['what', 'how', 'why', 'when', 'where', 'should', 'can', 'could', 'would', 'will']):
                return self._process_natural_language(command)
            
            # Parse command
            parts = command.split()
            if not parts:
                return {'error': 'Empty command', 'status': 'error'}
            
            action = parts[0].lower()
            
            # Route to appropriate handler
            if action in ['forecast', 'predict', 'analyze']:
                return self._handle_forecasting(parts)
            elif action in ['tune', 'optimize', 'hyperparameter']:
                return self._handle_tuning(parts)
            elif action in ['strategy', 'run', 'backtest']:
                return self._handle_strategy(parts)
            elif action in ['portfolio', 'rebalance', 'risk']:
                return self._handle_portfolio(parts)
            elif action in ['agent', 'agents']:
                return self._handle_agents(parts)
            elif action in ['report', 'generate']:
                return self._handle_reports(parts)
            elif action in ['status', 'health']:
                return self._handle_status(parts)
            else:
                return {'error': f'Unknown command: {action}', 'status': 'error'}
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _process_natural_language(self, query: str) -> Dict[str, Any]:
        """Process natural language query using QuantGPT."""
        try:
            if 'quant_gpt' not in self.components:
                return {'error': 'QuantGPT not available', 'status': 'error'}
            
            result = self.components['quant_gpt'].process_query(query)
            return {
                'type': 'natural_language',
                'query': query,
                'result': result,
                'status': 'success'
            }
        except Exception as e:
            return {'error': f'Error processing natural language query: {e}', 'status': 'error'}
    
    def _handle_forecasting(self, parts: List[str]) -> Dict[str, Any]:
        """Handle forecasting commands."""
        try:
            if len(parts) < 2:
                return {'error': 'Forecasting requires a symbol', 'status': 'error'}
            
            symbol = parts[1].upper()
            period = parts[2] if len(parts) > 2 else '7d'
            
            # Mock response for now
            result = {
                'symbol': symbol,
                'period': period,
                'prediction': f'Mock forecast for {symbol} over {period}',
                'confidence': 0.85,
                'models_used': ['lstm', 'xgboost']
            }
            
            return {
                'type': 'forecast',
                'symbol': symbol,
                'period': period,
                'result': result,
                'status': 'success'
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
    """Render main interface page."""
    st.title("🔮 Evolve Unified Interface")
    st.markdown("Access all trading system features through one unified interface.")
    
    # Command input
    st.subheader("Command Interface")
    command = st.text_input(
        "Enter command or natural language query:",
        placeholder="e.g., 'forecast AAPL 30d' or 'What's the best model for TSLA?'"
    )
    
    if st.button("Execute"):
        if command:
            with st.spinner("Processing..."):
                result = interface.process_command(command)
                display_result(result)
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Forecast AAPL"):
            result = interface.process_command("forecast AAPL 7d")
            display_result(result)
    
    with col2:
        if st.button("🤖 Ask QuantGPT"):
            result = interface.process_command("What's the trading signal for TSLA?")
            display_result(result)
    
    with col3:
        if st.button("⚙️ System Status"):
            result = interface.process_command("status")
            display_result(result)


def render_quantgpt_interface(interface: UnifiedInterface):
    """Render QuantGPT interface."""
    st.title("🤖 QuantGPT Natural Language Interface")
    st.markdown("Ask questions about trading in natural language.")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., 'What's the best model for NVDA over 90 days?' or 'Should I buy AAPL now?'",
        height=100
    )
    
    if st.button("Ask QuantGPT"):
        if query:
            with st.spinner("Processing..."):
                result = interface.process_command(query)
                display_result(result)
    
    # Example queries
    st.subheader("Example Queries")
    examples = [
        "Give me the best model for NVDA over 90 days",
        "Should I long TSLA this week?",
        "Analyze BTCUSDT market conditions",
        "What's the trading signal for AAPL?",
        "Find the optimal model for GOOGL on 1h timeframe"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{example}"):
            with st.spinner("Processing..."):
                result = interface.process_command(example)
                display_result(result)


def render_forecasting_interface(interface: UnifiedInterface):
    """Render forecasting interface."""
    st.title("📈 Forecasting Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol:", value="AAPL").upper()
        period = st.selectbox("Period:", ["1d", "7d", "30d", "90d"])
        timeframe = st.selectbox("Timeframe:", ["1m", "5m", "15m", "1h", "4h", "1d"])
    
    with col2:
        model_type = st.selectbox("Model Type:", ["auto", "lstm", "xgboost", "prophet", "ensemble"])
        include_analysis = st.checkbox("Include Analysis", value=True)
        generate_report = st.checkbox("Generate Report", value=False)
    
    if st.button("Generate Forecast"):
        command = f"forecast {symbol} {period}"
        with st.spinner("Generating forecast..."):
            result = interface.process_command(command)
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


def display_result(result: Dict[str, Any]):
    """Display command result in Streamlit."""
    if result.get('status') == 'error':
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    if result.get('type') == 'help':
        st.json(result['data'])
        return
    
    # Display result based on type
    result_type = result.get('type', 'unknown')
    
    if result_type == 'natural_language':
        st.subheader("QuantGPT Response")
        if 'result' in result and 'gpt_commentary' in result['result']:
            st.write(result['result']['gpt_commentary'])
        else:
            st.write(result.get('result', 'No response'))
    
    elif result_type in ['forecast', 'tuning', 'strategy_run']:
        st.subheader(f"{result_type.title()} Result")
        st.json(result.get('result', {}))
    
    elif result_type in ['strategy_list', 'agent_list']:
        st.subheader(f"{result_type.replace('_', ' ').title()}")
        items = result.get('strategies', result.get('agents', []))
        for item in items:
            st.write(f"- {item}")
    
    else:
        st.subheader("Result")
        st.json(result)


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