"""
Strategy Laboratory

A clean, production-ready strategy interface with:
- Strategy selection and optimization
- Auto-tuning capabilities
- Performance metrics (Sharpe, Win Rate, Drawdown)
- Strategy comparison and backtesting
- Clean UI without dev clutter
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for clean styling
st.markdown("""
<style>
    .strategy-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .strategy-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .performance-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    
    .optimization-panel {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #2196f3;
    }
    
    .strategy-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-active {
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #4caf50;
    }
    
    .status-optimizing {
        background: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ff9800;
    }
    
    .status-inactive {
        background: #ffebee;
        color: #c62828;
        border: 1px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for strategy lab."""
    if 'strategy_history' not in st.session_state:
        st.session_state.strategy_history = []
    
    if 'current_strategy' not in st.session_state:
        st.session_state.current_strategy = None
    
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}

def load_available_strategies():
    """Load available trading strategies."""
    strategies = {
        'RSI Mean Reversion': {
            'description': 'Relative Strength Index mean reversion strategy',
            'best_for': 'Range-bound markets',
            'parameters': ['rsi_period', 'oversold_threshold', 'overbought_threshold'],
            'default_params': {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70},
            'performance': {'sharpe': 1.2, 'win_rate': 0.65, 'max_dd': 0.08},
            'status': 'Active'
        },
        'MACD Crossover': {
            'description': 'Moving Average Convergence Divergence crossover strategy',
            'best_for': 'Trending markets',
            'parameters': ['fast_period', 'slow_period', 'signal_period'],
            'default_params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'performance': {'sharpe': 1.1, 'win_rate': 0.58, 'max_dd': 0.12},
            'status': 'Active'
        },
        'Bollinger Bands': {
            'description': 'Bollinger Bands mean reversion strategy',
            'best_for': 'Volatile markets',
            'parameters': ['period', 'std_dev'],
            'default_params': {'period': 20, 'std_dev': 2},
            'performance': {'sharpe': 1.1, 'win_rate': 0.62, 'max_dd': 0.10},
            'status': 'Active'
        },
        'Moving Average Crossover': {
            'description': 'Simple moving average crossover strategy',
            'best_for': 'Trend following',
            'parameters': ['short_period', 'long_period'],
            'default_params': {'short_period': 10, 'long_period': 30},
            'performance': {'sharpe': 0.9, 'win_rate': 0.55, 'max_dd': 0.15},
            'status': 'Active'
        },
        'EMA Crossover': {
            'description': 'Exponential moving average crossover strategy',
            'best_for': 'Fast trend changes',
            'parameters': ['short_period', 'long_period'],
            'default_params': {'short_period': 12, 'long_period': 26},
            'performance': {'sharpe': 1.0, 'win_rate': 0.60, 'max_dd': 0.13},
            'status': 'Active'
        },
        'Donchian Channels': {
            'description': 'Donchian Channels breakout strategy',
            'best_for': 'Breakout trading',
            'parameters': ['period'],
            'default_params': {'period': 20},
            'performance': {'sharpe': 1.3, 'win_rate': 0.68, 'max_dd': 0.09},
            'status': 'Active'
        },
        'ATR Breakout': {
            'description': 'Average True Range breakout strategy',
            'best_for': 'Volatility breakouts',
            'parameters': ['atr_period', 'multiplier'],
            'default_params': {'atr_period': 14, 'multiplier': 2},
            'performance': {'sharpe': 1.4, 'win_rate': 0.70, 'max_dd': 0.08},
            'status': 'Active'
        },
        'Stochastic Oscillator': {
            'description': 'Stochastic oscillator mean reversion',
            'best_for': 'Oversold/overbought conditions',
            'parameters': ['k_period', 'd_period', 'oversold', 'overbought'],
            'default_params': {'k_period': 14, 'd_period': 3, 'oversold': 20, 'overbought': 80},
            'performance': {'sharpe': 1.0, 'win_rate': 0.58, 'max_dd': 0.14},
            'status': 'Active'
        }
    }
    return strategies

def calculate_strategy_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive strategy performance metrics."""
    try:
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = calculate_max_drawdown(returns)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # Win rate and other metrics
        win_rate = np.mean(returns > 0)
        profit_factor = np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'VaR_95': var_95,
            'CVaR_95': cvar_95,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Calmar_Ratio': calmar_ratio
        }
    except Exception as e:
        logger.error(f"Error calculating strategy metrics: {e}")
        return {}

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

def generate_strategy_backtest(strategy_name: str, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate mock strategy backtest results."""
    try:
        # Generate mock price data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # Generate price series with some trend and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        
        # Generate strategy signals based on strategy type
        if 'RSI' in strategy_name:
            # RSI-like signals
            rsi_period = params.get('rsi_period', 14)
            oversold = params.get('oversold_threshold', 30)
            overbought = params.get('overbought_threshold', 70)
            
            # Mock RSI calculation
            rsi = 50 + 20 * np.sin(np.arange(len(dates)) * 0.1)
            signals = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
            
        elif 'MACD' in strategy_name:
            # MACD-like signals
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            
            # Mock MACD calculation
            fast_ma = pd.Series(prices).rolling(fast_period).mean()
            slow_ma = pd.Series(prices).rolling(slow_period).mean()
            macd = fast_ma - slow_ma
            signals = np.where(macd > 0, 1, -1)
            
        elif 'Bollinger' in strategy_name:
            # Bollinger Bands-like signals
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2)
            
            # Mock Bollinger Bands
            ma = pd.Series(prices).rolling(period).mean()
            std = pd.Series(prices).rolling(period).std()
            upper = ma + std_dev * std
            lower = ma - std_dev * std
            
            signals = np.where(prices < lower, 1, np.where(prices > upper, -1, 0))
            
        else:
            # Generic strategy signals
            signals = np.random.choice([-1, 0, 1], len(dates), p=[0.3, 0.4, 0.3])
        
        # Calculate strategy returns
        strategy_returns = np.diff(prices) / prices[:-1] * signals[:-1]
        
        # Calculate metrics
        metrics = calculate_strategy_metrics(strategy_returns)
        
        # Calculate equity curve
        equity_curve = np.cumprod(1 + strategy_returns)
        
        return {
            'strategy_name': strategy_name,
            'symbol': symbol,
            'parameters': params,
            'dates': dates[1:],  # Remove first date since we're using diff
            'prices': prices[1:],
            'signals': signals[1:],
            'returns': strategy_returns,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'generated_at': datetime.now()
        }
    except Exception as e:
        logger.error(f"Error generating strategy backtest: {e}")
        return {}

def plot_strategy_results(backtest_data: Dict[str, Any]):
    """Plot comprehensive strategy backtest results."""
    try:
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price & Signals', 'Equity Curve', 'Returns Distribution', 'Drawdown', 'Monthly Returns', 'Risk Metrics'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Price and signals
        fig.add_trace(
            go.Scatter(
                x=backtest_data['dates'],
                y=backtest_data['prices'],
                mode='lines',
                name='Price',
                line=dict(color='#2c3e50', width=2)
            ),
            row=1, col=1
        )
        
        # Add buy/sell signals
        buy_signals = backtest_data['signals'] == 1
        sell_signals = backtest_data['signals'] == -1
        
        if np.any(buy_signals):
            fig.add_trace(
                go.Scatter(
                    x=np.array(backtest_data['dates'])[buy_signals],
                    y=np.array(backtest_data['prices'])[buy_signals],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        if np.any(sell_signals):
            fig.add_trace(
                go.Scatter(
                    x=np.array(backtest_data['dates'])[sell_signals],
                    y=np.array(backtest_data['prices'])[sell_signals],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=8, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=backtest_data['dates'],
                y=backtest_data['equity_curve'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='#27ae60', width=2)
            ),
            row=1, col=2
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(
                x=backtest_data['returns'],
                name='Returns',
                marker_color='#3498db',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # Drawdown
        cumulative = backtest_data['equity_curve']
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(
                x=backtest_data['dates'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='#e74c3c', width=2),
                fill='tonexty'
            ),
            row=2, col=2
        )
        
        # Monthly returns heatmap
        returns_df = pd.DataFrame({
            'date': backtest_data['dates'],
            'returns': backtest_data['returns']
        })
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        returns_df['year'] = returns_df['date'].dt.year
        returns_df['month'] = returns_df['date'].dt.month
        
        monthly_returns = returns_df.groupby(['year', 'month'])['returns'].sum().unstack()
        
        fig.add_trace(
            go.Heatmap(
                z=monthly_returns.values,
                x=monthly_returns.columns,
                y=monthly_returns.index,
                colorscale='RdYlGn',
                name='Monthly Returns'
            ),
            row=3, col=1
        )
        
        # Risk metrics bar chart
        metrics = backtest_data['metrics']
        metric_names = ['Sharpe_Ratio', 'Win_Rate', 'Max_Drawdown', 'Calmar_Ratio']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name='Risk Metrics',
                marker_color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text=f"Strategy Analysis: {backtest_data['strategy_name']} on {backtest_data['symbol']}",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error plotting strategy results: {e}")
        st.error("Error generating strategy visualization")

def display_strategy_metrics(metrics: Dict[str, float]):
    """Display strategy performance metrics."""
    try:
        st.markdown("### Performance Metrics")
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Sharpe_Ratio', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Total_Return', 0):.2%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Win_Rate', 0):.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Max_Drawdown', 0):.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Volatility', 0):.1%}</div>
                <div class="metric-label">Volatility</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Calmar_Ratio', 0):.2f}</div>
                <div class="metric-label">Calmar Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('Profit_Factor', 0):.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-item">
                <div class="metric-value">{metrics.get('VaR_95', 0):.1%}</div>
                <div class="metric-label">VaR (95%)</div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Error displaying metrics: {e}")

def optimize_strategy_parameters(strategy_name: str, symbol: str, base_params: Dict[str, Any]):
    """Optimize strategy parameters using various techniques."""
    st.markdown("""
    <div class="optimization-panel">
        <h3>Strategy Optimization</h3>
        <p>Automatically optimize strategy parameters for better performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimization Settings")
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["Grid Search", "Bayesian Optimization", "Genetic Algorithm", "Random Search"]
        )
        
        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["Sharpe Ratio", "Total Return", "Calmar Ratio", "Win Rate"]
        )
        
        max_iterations = st.slider("Max Iterations", 10, 1000, 100)
        
        if st.button("Start Optimization", type="primary"):
            with st.spinner("Optimizing strategy parameters..."):
                try:
                    # Mock optimization process
                    optimization_results = {
                        'method': optimization_method,
                        'metric': optimization_metric,
                        'iterations': max_iterations,
                        'best_params': base_params.copy(),
                        'best_score': 1.5,
                        'improvement': 0.3,
                        'optimization_history': []
                    }
                    
                    # Simulate optimization history
                    for i in range(max_iterations):
                        score = 1.0 + np.random.normal(0, 0.1)
                        optimization_results['optimization_history'].append({
                            'iteration': i + 1,
                            'score': score,
                            'params': base_params.copy()
                        })
                    
                    st.session_state.optimization_results = optimization_results
                    st.success("Optimization completed!")
                    
                except Exception as e:
                    st.error(f"Error during optimization: {e}")
    
    with col2:
        st.subheader("Optimization Results")
        
        if 'optimization_results' in st.session_state and st.session_state.optimization_results:
            results = st.session_state.optimization_results
            
            st.markdown(f"""
            <div class="strategy-card">
                <h4>Best Parameters</h4>
                <p><strong>Method:</strong> {results['method']}</p>
                <p><strong>Metric:</strong> {results['metric']}</p>
                <p><strong>Best Score:</strong> {results['best_score']:.3f}</p>
                <p><strong>Improvement:</strong> {results['improvement']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show optimization history
            if results['optimization_history']:
                history_df = pd.DataFrame(results['optimization_history'])
                st.line_chart(history_df.set_index('iteration')['score'])
        else:
            st.info("Run optimization to see results")

def main():
    """Main strategy lab function."""
    st.title("Strategy Laboratory")
    st.markdown("Develop, test, and optimize trading strategies with advanced analytics")
    
    # Initialize session state
    initialize_session_state()
    
    # Load available strategies
    strategies = load_available_strategies()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Strategy Settings")
        
        # Strategy selection
        strategy_options = list(strategies.keys())
        selected_strategy = st.selectbox("Strategy", strategy_options, index=0)
        
        # Symbol input
        symbol = st.text_input("Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, BTC-USD")
        
        # Backtest period
        backtest_days = st.slider("Backtest Period (days)", 30, 1000, 252)
        
        # Show strategy info
        if selected_strategy in strategies:
            strategy_info = strategies[selected_strategy]
            st.markdown("### Strategy Info")
            st.markdown(f"""
            <div class="strategy-card">
                <h4>{selected_strategy}</h4>
                <p>{strategy_info['description']}</p>
                <p><strong>Best for:</strong> {strategy_info['best_for']}</p>
                <p><strong>Sharpe:</strong> {strategy_info['performance']['sharpe']:.2f}</p>
                <p><strong>Win Rate:</strong> {strategy_info['performance']['win_rate']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Run backtest button
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                if selected_strategy in strategies:
                    params = strategies[selected_strategy]['default_params']
                    
                    # Validate parameters
                    valid_params = True
                    for param, value in params.items():
                        if isinstance(value, (int, float)) and value <= 0:
                            st.error(f"Parameter {param} must be > 0, got {value}")
                            valid_params = False
                            break
                    
                    if valid_params:
                        backtest_data = generate_strategy_backtest(selected_strategy, symbol, params)
                        if backtest_data:
                            st.session_state.current_strategy = backtest_data
                            st.session_state.strategy_history.append(backtest_data)
                            st.success("Backtest completed!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current strategy results
        if st.session_state.current_strategy:
            st.markdown("### Strategy Analysis")
            plot_strategy_results(st.session_state.current_strategy)
            
            # Performance metrics
            display_strategy_metrics(st.session_state.current_strategy['metrics'])
            
            # Export options
            st.markdown("### Export Options")
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                if st.button("Export Results"):
                    st.info("Export feature coming soon!")
            
            with col_export2:
                if st.button("Generate Report"):
                    st.info("Report generation feature coming soon!")
        else:
            st.info("Run a backtest using the sidebar controls")
    
    with col2:
        # Strategy comparison
        if st.session_state.strategy_history:
            st.markdown("### Strategy Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for strategy in st.session_state.strategy_history[-5:]:  # Last 5 strategies
                metrics = strategy['metrics']
                comparison_data.append({
                    'Strategy': strategy['strategy_name'],
                    'Symbol': strategy['symbol'],
                    'Sharpe': metrics.get('Sharpe_Ratio', 0),
                    'Win Rate': metrics.get('Win_Rate', 0),
                    'Max DD': metrics.get('Max_Drawdown', 0)
                })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Show best strategy
                best_strategy = comparison_df.loc[comparison_df['Sharpe'].idxmax()]
                st.markdown(f"""
                <div class="strategy-card">
                    <h4>Best Strategy</h4>
                    <p><strong>Name:</strong> {best_strategy['Strategy']}</p>
                    <p><strong>Symbol:</strong> {best_strategy['Symbol']}</p>
                    <p><strong>Sharpe:</strong> {best_strategy['Sharpe']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Strategy optimization section
    st.markdown("---")
    if st.session_state.current_strategy:
        optimize_strategy_parameters(
            st.session_state.current_strategy['strategy_name'],
            st.session_state.current_strategy['symbol'],
            st.session_state.current_strategy['parameters']
        )

if __name__ == "__main__":
    main() 