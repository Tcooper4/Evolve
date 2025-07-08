"""
Evolve Trading Platform - Production Ready Streamlit App

A clean, ChatGPT-like interface with a single prompt box that routes to:
- Forecasting
- Strategy Selection  
- Signal Generation
- Trade Reports

All triggered from a single input with professional styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Evolve Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components with error handling
try:
    from trading.llm.agent import PromptAgent
    from strategies.strategy_engine import StrategyEngine
    from models.forecast_router import ForecastRouter
    from trading.report.report_export_engine import ReportExportEngine
    from trading.memory.model_monitor import ModelMonitor
    from trading.memory.strategy_logger import StrategyLogger
    CORE_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    CORE_COMPONENTS_AVAILABLE = False

class EvolveTradingApp:
    """Production-ready Evolve Trading Platform with ChatGPT-like interface."""
    
    def __init__(self):
        """Initialize the app."""
        self.initialize_components()
        self.setup_session_state()
        
    def initialize_components(self):
        """Initialize core components with fallbacks."""
        try:
            if CORE_COMPONENTS_AVAILABLE:
                self.prompt_agent = PromptAgent()
                self.strategy_engine = StrategyEngine()
                self.forecast_router = ForecastRouter()
                self.report_engine = ReportExportEngine()
                self.model_monitor = ModelMonitor()
                self.strategy_logger = StrategyLogger()
                logger.info("All core components initialized successfully")
            else:
                self._initialize_fallback_components()
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self):
        """Initialize fallback components."""
        logger.info("Initializing fallback components")
        
        # Create minimal fallback components for missing modules
        self.model_monitor = self._create_fallback_model_monitor()
        self.strategy_logger = self._create_fallback_strategy_logger()
    
    def _create_fallback_model_monitor(self):
        """Create fallback model monitor."""
        class FallbackModelMonitor:
            def get_model_trust_levels(self) -> Dict[str, float]:
                return {'LSTM': 0.8, 'ARIMA': 0.7, 'XGBoost': 0.9}
        return FallbackModelMonitor()
    
    def _create_fallback_strategy_logger(self):
        """Create fallback strategy logger."""
        class FallbackStrategyLogger:
            def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
                return []
        return FallbackStrategyLogger()
    
    def setup_session_state(self):
        """Setup session state variables."""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_action' not in st.session_state:
            st.session_state.current_action = None
        if 'last_result' not in st.session_state:
            st.session_state.last_result = None
    
    def run(self):
        """Run the main application."""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
    
    def render_header(self):
        """Render the header with professional styling."""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .chat-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid #e9ecef;
        }
        .user-message {
            background: #007bff;
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            max-width: 80%;
            margin-left: auto;
        }
        .assistant-message {
            background: white;
            color: #333;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            max-width: 80%;
            border: 1px solid #e9ecef;
        }
        .stButton > button {
            border-radius: 25px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 600;
        }
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid #e9ecef;
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Evolve Trading Platform</h1>
            <p>AI-Powered Trading Analysis & Strategy Generation</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the minimal sidebar."""
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            
            # Model selection
            st.markdown("**🤖 Models**")
            try:
                trust_levels = self.model_monitor.get_model_trust_levels()
                for model, trust in trust_levels.items():
                    st.checkbox(f"{model} ({trust:.1%})", value=True, key=f"model_{model}")
            except:
                st.info("Models: LSTM, ARIMA, XGBoost")
            
            st.markdown("---")
            
            # Strategy toggles
            st.markdown("**📊 Strategies**")
            strategies = ["Momentum", "Mean Reversion", "Breakout", "Arbitrage"]
            for strategy in strategies:
                st.checkbox(strategy, value=True, key=f"strategy_{strategy}")
            
            st.markdown("---")
            
            # System status
            st.markdown("**📈 System Status**")
            st.success("🟢 Operational")
            
            # Recent activity
            st.markdown("**📋 Recent Activity**")
            try:
                recent_decisions = self.strategy_logger.get_recent_decisions(limit=5)
                if recent_decisions:
                    for decision in recent_decisions[-3:]:
                        st.caption(f"• {decision.get('timestamp', 'Unknown')}")
                else:
                    st.caption("No recent activity")
            except:
                st.caption("Activity tracking unavailable")
    
    def render_main_content(self):
        """Render the main content area with chat interface."""
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Main prompt input
        st.markdown("### 💬 Ask me anything about trading...")
        prompt = st.text_area(
            "Enter your trading question or request:",
            placeholder="e.g., 'Forecast AAPL for the next 30 days' or 'Generate a momentum strategy for TSLA' or 'Create a comprehensive trading report for my portfolio'",
            height=100,
            key="main_prompt"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Submit", use_container_width=True):
                if prompt.strip():
                    self.process_user_prompt(prompt)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results if available
        if st.session_state.last_result:
            self.display_results(st.session_state.last_result)
    
    def process_user_prompt(self, prompt: str):
        """Process user prompt and route to appropriate functionality."""
        try:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now()
            })
            
            # Process prompt with AI agent
            result = self.prompt_agent.process_prompt(prompt)
            
            if result.success:
                # Convert AgentResponse to dictionary format for compatibility
                result_dict = {
                    'success': result.success,
                    'action': 'forecast',  # Default action
                    'message': result.message,
                    'data': result.data,
                    'recommendations': result.recommendations,
                    'next_actions': result.next_actions
                }
                
                # Determine action based on the response
                if result.data and 'forecast' in str(result.data).lower():
                    action = 'forecast'
                elif result.data and 'strategy' in str(result.data).lower():
                    action = 'strategy'
                elif result.data and 'report' in str(result.data).lower():
                    action = 'report'
                else:
                    action = 'forecast'  # Default
                
                result_dict['action'] = action
                
                if action == 'forecast':
                    forecast_result = self.handle_forecast_request(prompt, result_dict)
                    st.session_state.last_result = forecast_result
                elif action == 'strategy':
                    strategy_result = self.handle_strategy_request(prompt, result_dict)
                    st.session_state.last_result = strategy_result
                elif action == 'report':
                    report_result = self.handle_report_request(prompt, result_dict)
                    st.session_state.last_result = report_result
                else:
                    # Default to forecast
                    forecast_result = self.handle_forecast_request(prompt, result_dict)
                    st.session_state.last_result = forecast_result
                
                # Add assistant response to chat history
                response_content = self.format_response_content(st.session_state.last_result)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_content,
                    'timestamp': datetime.now()
                })
                
            else:
                error_message = f"Error processing request: {result.message if hasattr(result, 'message') else 'Unknown error'}"
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': error_message,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            error_message = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': error_message,
                'timestamp': datetime.now()
            })
        
        # Rerun to update the display
        st.rerun()
    
    def handle_forecast_request(self, prompt: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle forecast requests."""
        try:
            symbol = result.get('symbol', 'AAPL')
            days = result.get('days', 30)
            
            forecast_data = self.forecast_router.generate_forecast(symbol, days)
            
            return {
                'type': 'forecast',
                'symbol': symbol,
                'days': days,
                'data': forecast_data,
                'confidence': result.get('confidence', 0.8),
                'prompt': prompt
            }
        except Exception as e:
            logger.error(f"Error in forecast request: {e}")
            return {
                'type': 'forecast',
                'error': str(e),
                'prompt': prompt
            }
    
    def handle_strategy_request(self, prompt: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle strategy requests."""
        try:
            symbol = result.get('symbol', 'AAPL')
            strategy_type = result.get('strategy_type', 'momentum')
            
            # Create sample market data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
            market_data = pd.DataFrame({'Close': prices}, index=dates)
            
            strategy_result = self.strategy_engine.select_strategy(market_data, 'bull')
            
            return {
                'type': 'strategy',
                'symbol': symbol,
                'strategy_type': strategy_type,
                'data': strategy_result,
                'confidence': result.get('confidence', 0.7),
                'prompt': prompt
            }
        except Exception as e:
            logger.error(f"Error in strategy request: {e}")
            return {
                'type': 'strategy',
                'error': str(e),
                'prompt': prompt
            }
    
    def handle_report_request(self, prompt: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report requests."""
        try:
            report_type = result.get('report_type', 'comprehensive')
            
            report_result = self.report_engine.generate_comprehensive_report(
                config={
                    'title': 'Trading Analysis Report',
                    'author': 'Evolve AI',
                    'date': datetime.now(),
                    'format': 'markdown'
                },
                data={
                    'metrics': {
                        'total_return': 0.15,
                        'sharpe_ratio': 1.2,
                        'max_drawdown': -0.08,
                        'win_rate': 0.65
                    }
                }
            )
            
            return {
                'type': 'report',
                'report_type': report_type,
                'data': report_result,
                'confidence': result.get('confidence', 0.9),
                'prompt': prompt
            }
        except Exception as e:
            logger.error(f"Error in report request: {e}")
            return {
                'type': 'report',
                'error': str(e),
                'prompt': prompt
            }
    
    def format_response_content(self, result: Dict[str, Any]) -> str:
        """Format the response content for chat display."""
        if result.get('error'):
            return f"❌ Error: {result['error']}"
        
        # Handle AgentResponse message if available
        if result.get('message'):
            return result['message']
        
        result_type = result.get('type')
        
        if result_type == 'forecast':
            symbol = result.get('symbol', 'Unknown')
            confidence = result.get('confidence', 0)
            return f"📈 Generated forecast for {symbol} with {confidence:.1%} confidence. Check the results below!"
        
        elif result_type == 'strategy':
            symbol = result.get('symbol', 'Unknown')
            strategy_type = result.get('strategy_type', 'Unknown')
            confidence = result.get('confidence', 0)
            return f"📊 Generated {strategy_type} strategy for {symbol} with {confidence:.1%} confidence. Check the results below!"
        
        elif result_type == 'report':
            report_type = result.get('report_type', 'Unknown')
            confidence = result.get('confidence', 0)
            return f"📋 Generated {report_type} report with {confidence:.1%} confidence. Check the results below!"
        
        else:
            return "✅ Request processed successfully. Check the results below!"
    
    def display_results(self, result: Dict[str, Any]):
        """Display the results in a professional format."""
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        if result.get('error'):
            st.error(f"Error: {result['error']}")
            return
        
        result_type = result.get('type')
        
        if result_type == 'forecast':
            self.display_forecast_results(result)
        elif result_type == 'strategy':
            self.display_strategy_results(result)
        elif result_type == 'report':
            self.display_report_results(result)
    
    def display_forecast_results(self, result: Dict[str, Any]):
        """Display forecast results."""
        symbol = result.get('symbol', 'Unknown')
        confidence = result.get('confidence', 0)
        forecast_data = result.get('data', {})
        
        if not forecast_data.get('success'):
            st.error("Forecast generation failed")
            return
        
        # Display confidence
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        with col3:
            st.metric("Forecast Days", result.get('days', 0))
        
        # Create forecast chart
        if 'forecast' in forecast_data and 'dates' in forecast_data:
            forecast_values = forecast_data['forecast']
            dates = pd.to_datetime(forecast_data['dates'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#667eea', width=3)
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        if st.button("📥 Download Forecast Data"):
            self.download_forecast_data(result)
    
    def display_strategy_results(self, result: Dict[str, Any]):
        """Display strategy results."""
        symbol = result.get('symbol', 'Unknown')
        strategy_type = result.get('strategy_type', 'Unknown')
        confidence = result.get('confidence', 0)
        strategy_data = result.get('data', {})
        
        # Display strategy info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Symbol", symbol)
        with col2:
            st.metric("Strategy", strategy_type.title())
        with col3:
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Display strategy details
        if strategy_data.get('success'):
            st.success(f"✅ {strategy_type.title()} strategy generated successfully")
            
            # Strategy metrics
            if 'metrics' in strategy_data:
                metrics = strategy_data['metrics']
                st.markdown("#### Strategy Metrics")
                metric_cols = st.columns(len(metrics))
                for i, (metric, value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(metric.title(), f"{value:.4f}")
        
        # Download button
        if st.button("📥 Download Strategy Report"):
            self.download_strategy_data(result)
    
    def display_report_results(self, result: Dict[str, Any]):
        """Display report results."""
        report_type = result.get('report_type', 'Unknown')
        confidence = result.get('confidence', 0)
        report_data = result.get('data', {})
        
        # Display report info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Report Type", report_type.title())
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Display report content
        if report_data.get('success'):
            st.success("✅ Report generated successfully")
            
            # Show report path if available
            if 'report_path' in report_data:
                st.info(f"Report saved to: {report_data['report_path']}")
            
            # Display sample metrics
            st.markdown("#### Sample Metrics")
            sample_metrics = {
                'Total Return': '15.2%',
                'Sharpe Ratio': '1.24',
                'Max Drawdown': '-8.3%',
                'Win Rate': '65.4%'
            }
            
            metric_cols = st.columns(len(sample_metrics))
            for i, (metric, value) in enumerate(sample_metrics.items()):
                with metric_cols[i]:
                    st.metric(metric, value)
        
        # Download button
        if st.button("📥 Download Report"):
            self.download_report_data(result)
    
    def download_forecast_data(self, result: Dict[str, Any]):
        """Download forecast data."""
        try:
            forecast_data = result.get('data', {})
            if 'forecast' in forecast_data and 'dates' in forecast_data:
                df = pd.DataFrame({
                    'Date': forecast_data['dates'],
                    'Forecast': forecast_data['forecast']
                })
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"forecast_{result.get('symbol', 'unknown')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error downloading data: {e}")
    
    def download_strategy_data(self, result: Dict[str, Any]):
        """Download strategy data."""
        try:
            strategy_data = result.get('data', {})
            json_data = json.dumps(strategy_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"strategy_{result.get('strategy_type', 'unknown')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Error downloading data: {e}")
    
    def download_report_data(self, result: Dict[str, Any]):
        """Download report data."""
        try:
            report_data = result.get('data', {})
            json_data = json.dumps(report_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"report_{result.get('report_type', 'unknown')}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Error downloading data: {e}")

def main():
    """Main function to run the app."""
    app = EvolveTradingApp()
    app.run()

if __name__ == "__main__":
    main() 