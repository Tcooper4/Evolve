Strategy Pipeline Demo Page for Evolve Trading Platform.""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading.ui.components import (
    create_strategy_pipeline_selector,
    execute_strategy_pipeline,
    create_date_range_selector,
    create_asset_selector
)

warnings.filterwarnings("ignore)

# Page config
st.set_page_config(page_title=Strategy Pipeline Demo, page_icon=🔗 layout=wide


def load_sample_data(symbol: str =AAPL, days: int = 365 -> pd.DataFrame:
Load sample market data for demonstration.""    try:
        # Generate sample data for demo purposes
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0050.02en(dates))
        prices =10* np.exp(np.cumsum(returns))
        
        # Add some trend and volatility clustering
        trend = np.linspace(0,0.1en(dates))
        prices = prices * (1 + trend)
        
        # Generate OHLCV data
        data = pd.DataFrame({
         datedates,
            open': prices * (1 + np.random.normal(0, 05es))),
            high': prices * (1 + np.abs(np.random.normal(001s)))),
            low': prices * (1 - np.abs(np.random.normal(001s)))),
           closerices,
      volume': np.random.randint(10000000en(dates))
        })
        
        data.set_index('date', inplace=True)
        return data
        
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame()


def plot_strategy_signals(data: pd.DataFrame, strategy_result: dict) -> go.Figure:
  ate a plot showing price data and strategy signals."""
    fig = go.Figure()
    
    # Add price data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close],       name='Price',
        line=dict(color='blue', width=2
    ))
    
    # Add buy signals
    buy_signals = strategy_result[signal] ==1    if buy_signals.any():
        fig.add_trace(go.Scatter(
            x=data.index[buy_signals],
            y=data.loc[buy_signals, 'close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green, size=10, symbol='triangle-up')
        ))
    
    # Add sell signals
    sell_signals = strategy_result['signal'] == -1   if sell_signals.any():
        fig.add_trace(go.Scatter(
            x=data.index[sell_signals],
            y=data.loc[sell_signals, 'close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red, size=10, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title=f"Strategy Signals - {strategy_result[signal_type']}",
        xaxis_title="Date",
        yaxis_title="Price,        height=500 )
    
    return fig


def plot_individual_signals(data: pd.DataFrame, strategy_result: dict) -> go.Figure:
  ate a plot showing individual strategy signals."""
    fig = go.Figure()
    
    # Add price data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close],       name='Price',
        line=dict(color=gray width=1),
        opacity=00.7
    ))
    
    # Add individual strategy signals
    colors = ['red', blue, green',orange',purple]  for i, (strategy_name, signal) in enumerate(strategy_result['individual_signals'].items()):
        color = colors[i % len(colors)]
        
        # Buy signals
        buy_signals = signal == 1
        if buy_signals.any():
            fig.add_trace(go.Scatter(
                x=data.index[buy_signals],
                y=data.loc[buy_signals, 'close'],
                mode='markers,              name=f'{strategy_name} Buy,            marker=dict(color=color, size=8, symbol='circle'),
                opacity=00.8      ))
        
        # Sell signals
        sell_signals = signal == -1      if sell_signals.any():
            fig.add_trace(go.Scatter(
                x=data.index[sell_signals],
                y=data.loc[sell_signals, 'close'],
                mode='markers,              name=f'{strategy_name} Sell,            marker=dict(color=color, size=8, symbol='x'),
                opacity=00.8          ))
    
    fig.update_layout(
        title="Individual Strategy Signals",
        xaxis_title="Date",
        yaxis_title="Price,        height=500 )
    
    return fig


def display_performance_metrics(performance: dict) -> None:
    """Display performance metrics in a nice format.
    if 'error in performance:
        st.error(fError calculating performance: {performance['error']}")
        return
    
    st.subheader("📊 Performance Metrics")
    
    # Create metrics columns
    col1, col2 col4t.columns(4)
    
    with col1:
        st.metric(
          Total Return,    f"{performancetotal_return']:0.2,
            help=Total return over the period"
        )
    
    with col2:
        st.metric(
          Sharpe Ratio,    f"{performancesharpe_ratio']:0.2,
            help="Risk-adjusted return metric"
        )
    
    with col3:
        st.metric(
          Max Drawdown,    f{performancemax_drawdown']:0.2,
            help=Maximum peak-to-trough decline"
        )
    
    with col4:
        st.metric(
      WinRate,    f{performance['win_rate']:0.1,
            help="Percentage of profitable trades"
        )
    
    # Signal statistics
    st.subheader("📈 Signal Statistics)   col1, col2 col4t.columns(4)
    
    with col1:
        st.metric("Buy Signals", performance['buy_signals'])
    
    with col2:
        st.metric("Sell Signals", performance['sell_signals'])
    
    with col3:
        st.metric("Hold Signals", performance['hold_signals'])
    
    with col4:
        st.metric("Total Signals", performance['total_signals])


def main():
  Main function for the Strategy Pipeline Demo page.
    st.title("🔗 Strategy Pipeline Demo)
    st.markdown(
       Combine multiple trading strategies using different signal combination modes. "Test how different strategy combinations perform together. )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.selectbox(
         Data Source",
            ["Sample Data", Real Data (Coming Soon)"],
            help="Choose data source for demonstration"
        )
        
        # Symbol selection (for future real data)
        symbol = st.text_input(
    Symbol",
            value="AAPL",
            help="Stock symbol for data loading"
        )
        
        # Date range
        start_date, end_date = create_date_range_selector(
            default_days=365
            key="pipeline_demo"
        )
        
        # Load data button
        if st.button("📊 Load Data", type="primary):        st.session_state.data_loaded = True
            st.session_state.market_data = load_sample_data(symbol, 365        st.success(f"Loaded {len(st.session_state.market_data)} days of data for {symbol})    # Main content
    if not st.session_state.get('data_loaded,false):
        st.info("👈 Please load data from the sidebar to get started.")
        return
    
    # Strategy pipeline selector
    st.header("🎯 Strategy Configuration")
    strategy_config = create_strategy_pipeline_selector(
        key="pipeline_demo_selector",
        allow_combos=True
    )
    
    if not strategy_config:
        st.warning("Please configure your strategy selection above.")
        return
    
    # Execute strategy button
    if st.button("🚀 Execute Strategy Pipeline", type="primary"):
        with st.spinner("Executing strategy pipeline..."):
            strategy_result = execute_strategy_pipeline(
                strategy_config,
                st.session_state.market_data,
                key="pipeline_demo_execution"
            )
            
            if strategy_result:
                st.session_state.strategy_result = strategy_result
                st.success(Strategy pipeline executed successfully!")
            else:
                st.error(Failed to execute strategy pipeline.")
    
    # Display results
    if st.session_state.get('strategy_result'):
        strategy_result = st.session_state.strategy_result
        
        st.header("📈 Results")
        
        # Performance metrics
        display_performance_metrics(strategy_result['performance'])
        
        # Strategy information
        st.subheader("🔍 Strategy Information")
        col1, col2lumns(2)
        
        with col1:
            st.write(**Strategies Used:**", , in(strategy_result[strategies_used']))
            st.write("**Signal Type:**, strategy_result['signal_type])       if strategy_result['combine_mode']:
                st.write(**Combine Mode:**, strategy_result['combine_mode'])
        
        with col2:
            st.write("**Execution Time:**, strategy_result['execution_time])       if strategy_result['weights']:
                st.write(**Weights:**, [f{w:.2f} for w in strategy_result['weights']])
        
        # Create tabs for different visualizations
        tab1 tab2, tab3st.tabs(["Combined Signals", "Individual Signals", "Raw Data"])
        
        with tab1:
            st.plotly_chart(
                plot_strategy_signals(st.session_state.market_data, strategy_result),
                use_container_width=True
            )
        
        with tab2:
            if len(strategy_result['individual_signals']) > 1                st.plotly_chart(
                    plot_individual_signals(st.session_state.market_data, strategy_result),
                    use_container_width=True
                )
            else:
                st.info(Only one strategy selected. Individual signals view is the same as combined signals.")
        
        with tab3:
            st.subheader("📋 Raw Signal Data")
            signal_df = pd.DataFrame({
                Date: st.session_state.market_data.index,
         Price': st.session_state.market_data['close'],
                Combined_Signal: strategy_result['signal]  })
            
            # Add individual signals
            for strategy_name, signal in strategy_result['individual_signals'].items():
                signal_df[f'{strategy_name}_Signal'] = signal
            
            st.dataframe(signal_df.tail(20se_container_width=True)
            
            # Download button
            csv = signal_df.to_csv(index=True)
            st.download_button(
                label="📥 Download Signal Data,              data=csv,
                file_name=f"strategy_signals_{datetime.now().strftime(%Y%m%d_%H%M%S')}.csv,
                mime="text/csv            )


if __name__ == "__main__":
    main() 