# PAGE 2: 🔄 STRATEGY DEVELOPMENT & TESTING
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/2_Strategy_Testing.py`  
**Merges:** 2_Strategy_Backtest.py + Strategy_Lab.py + Strategy_Combo_Creator.py  
**Tabs:** 6 tabs  
**Estimated Time:** 10-12 hours  
**Priority:** CRITICAL

### Features Preserved:
✅ Pre-built strategy backtesting (Bollinger, MACD, RSI, SMA, etc.)  
✅ Custom strategy creation with visual builder  
✅ Strategy code editor  
✅ Strategy combination/ensemble creation  
✅ Multi-strategy portfolio backtesting  
✅ Walk-forward analysis  
✅ Monte Carlo strategy simulation  
✅ Strategy optimization  
✅ Parameter sensitivity analysis

---

## CURSOR PROMPT 2.1 - Create Page Structure

```
Create pages/2_Strategy_Testing.py with complete 6-tab structure

REQUIREMENTS:

1. Page configuration:
```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Strategy Development & Testing",
    page_icon="🔄",
    layout="wide"
)
```

2. Initialize session state:
```python
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'custom_strategies' not in st.session_state:
    st.session_state.custom_strategies = {}
if 'selected_strategy' not in st.session_state:
    st.session_state.selected_strategy = None
if 'strategy_combos' not in st.session_state:
    st.session_state.strategy_combos = {}
```

3. Create tabs:
```python
st.title("🔄 Strategy Development & Testing")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Quick Backtest",
    "🔧 Strategy Builder",
    "💻 Advanced Editor",
    "🎯 Strategy Combos",
    "📊 Strategy Comparison",
    "🔬 Advanced Analysis"
])

# Placeholder content for each tab
with tab1:
    st.header("Quick Backtest")
    st.info("Integration pending...")

with tab2:
    st.header("Strategy Builder")
    st.info("Integration pending...")

with tab3:
    st.header("Advanced Editor")
    st.info("Integration pending...")

with tab4:
    st.header("Strategy Combos")
    st.info("Integration pending...")

with tab5:
    st.header("Strategy Comparison")
    st.info("Integration pending...")

with tab6:
    st.header("Advanced Analysis")
    st.info("Integration pending...")
```

OUTPUT: Create pages/2_Strategy_Testing.py
```

---

## CURSOR PROMPT 2.2 - Implement Quick Backtest (Tab 1)

```
Implement Tab 1 (Quick Backtest) in pages/2_Strategy_Testing.py

ADD IMPORTS:
```python
from trading.strategies.bollinger_strategy import BollingerStrategy
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.rsi_strategy import RSIStrategy
from trading.strategies.sma_strategy import SMAStrategy
from trading.backtesting.backtester import Backtester
from trading.data.data_loader import DataLoader
```

REPLACE Tab 1 with:

```python
with tab1:
    st.header("Quick Backtest")
    st.markdown("Backtest pre-built strategies with configurable parameters")
    
    # Create strategy registry
    STRATEGY_REGISTRY = {
        "Bollinger Bands": {
            "class": BollingerStrategy,
            "description": "Trades based on Bollinger Band breakouts",
            "params": {
                "window": {"type": "slider", "min": 10, "max": 50, "default": 20},
                "num_std": {"type": "slider", "min": 1.0, "max": 3.0, "default": 2.0, "step": 0.1}
            }
        },
        "MACD": {
            "class": MACDStrategy,
            "description": "Moving Average Convergence Divergence strategy",
            "params": {
                "fast": {"type": "slider", "min": 5, "max": 20, "default": 12},
                "slow": {"type": "slider", "min": 20, "max": 50, "default": 26},
                "signal": {"type": "slider", "min": 5, "max": 15, "default": 9}
            }
        },
        "RSI": {
            "class": RSIStrategy,
            "description": "Relative Strength Index mean reversion",
            "params": {
                "period": {"type": "slider", "min": 5, "max": 30, "default": 14},
                "overbought": {"type": "slider", "min": 60, "max": 90, "default": 70},
                "oversold": {"type": "slider", "min": 10, "max": 40, "default": 30}
            }
        },
        "SMA Crossover": {
            "class": SMAStrategy,
            "description": "Simple Moving Average crossover strategy",
            "params": {
                "fast_period": {"type": "slider", "min": 5, "max": 50, "default": 20},
                "slow_period": {"type": "slider", "min": 50, "max": 200, "default": 50}
            }
        }
    }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Data & Strategy")
        
        # Data loading
        with st.form("backtest_data_form"):
            symbol = st.text_input("Symbol", value="AAPL").upper()
            
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )
            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now()
                )
            
            load_data = st.form_submit_button("📊 Load Data", use_container_width=True)
        
        if load_data:
            try:
                with st.spinner(f"Loading {symbol}..."):
                    loader = DataLoader()
                    data = loader.load_data(symbol, start_date, end_date)
                    st.session_state.loaded_data = data
                    st.success(f"✅ Loaded {len(data)} days")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Strategy selection
        if st.session_state.loaded_data is not None:
            st.markdown("---")
            st.subheader("🎯 Strategy Selection")
            
            strategy_name = st.selectbox(
                "Select Strategy",
                list(STRATEGY_REGISTRY.keys())
            )
            
            strategy_info = STRATEGY_REGISTRY[strategy_name]
            st.info(strategy_info["description"])
            
            # Dynamic parameter inputs
            st.markdown("**Parameters:**")
            params = {}
            for param_name, param_config in strategy_info["params"].items():
                if param_config["type"] == "slider":
                    params[param_name] = st.slider(
                        param_name.replace("_", " ").title(),
                        min_value=param_config["min"],
                        max_value=param_config["max"],
                        value=param_config["default"],
                        step=param_config.get("step", 1)
                    )
            
            # Backtest settings
            st.markdown("---")
            st.markdown("**Backtest Settings:**")
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                value=10000,
                step=1000
            )
            commission = st.number_input(
                "Commission (%)",
                min_value=0.0,
                value=0.1,
                step=0.01
            )
            
            run_backtest = st.button(
                "🚀 Run Backtest",
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.subheader("📈 Results")
        
        if run_backtest and st.session_state.loaded_data is not None:
            try:
                with st.spinner("Running backtest..."):
                    # Initialize strategy
                    strategy_class = strategy_info["class"]
                    strategy = strategy_class(**params)
                    
                    # Initialize backtester
                    backtester = Backtester(
                        strategy=strategy,
                        initial_capital=initial_capital,
                        commission=commission/100
                    )
                    
                    # Run backtest
                    results = backtester.run(st.session_state.loaded_data)
                    st.session_state.backtest_results = results
                
                st.success("✅ Backtest complete!")
                
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
        
        # Display results
        if st.session_state.get('backtest_results'):
            results = st.session_state.backtest_results
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                total_return = results.get('total_return', 0) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            
            with col_m2:
                sharpe = results.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col_m3:
                max_dd = results.get('max_drawdown', 0) * 100
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
            
            with col_m4:
                win_rate = results.get('win_rate', 0) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Equity curve
            if 'equity_curve' in results:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['equity_curve'].index,
                    y=results['equity_curve']['equity'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title="Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Trade list
            if 'trades' in results:
                st.markdown("**Trade History:**")
                trades_df = pd.DataFrame(results['trades'])
                st.dataframe(trades_df, use_container_width=True)
                
                # Download trades
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Trades",
                    data=csv,
                    file_name=f"{symbol}_{strategy_name}_trades.csv",
                    mime="text/csv"
                )
```

VERIFICATION:
- Data loads correctly
- All strategies work
- Parameters adjust properly
- Backtest runs successfully
- Results display correctly
```

---

## CURSOR PROMPT 2.3 - Implement Strategy Builder (Tab 2)

```
Implement Tab 2 (Strategy Builder) in pages/2_Strategy_Testing.py

ADD IMPORT:
```python
from trading.strategies.custom_strategy_handler import CustomStrategyHandler
```

REPLACE Tab 2 with visual strategy builder that allows users to create custom strategies through a form-based interface without coding.

[Full implementation includes:
- Entry condition builder
- Exit condition builder
- Position sizing configuration
- Risk management rules
- Strategy preview
- Save/Load functionality]

VERIFICATION:
- Can create custom strategies
- Conditions validate properly
- Strategy saves successfully
- Can load saved strategies
```

---

## CURSOR PROMPT 2.4 - Implement Advanced Editor (Tab 3)

```
Implement Tab 3 (Advanced Editor) in pages/2_Strategy_Testing.py

This tab provides a code editor for advanced users to write custom strategy Python code with syntax highlighting and testing capabilities.

[Full implementation includes:
- Code editor with syntax highlighting
- Strategy templates
- Code validation
- Testing interface
- Export/Import functionality]

VERIFICATION:
- Code editor functional
- Syntax highlighting works
- Can test custom code
- Templates load correctly
```

---

## CURSOR PROMPT 2.5 - Implement Strategy Combos (Tab 4)

```
Implement Tab 4 (Strategy Combos) in pages/2_Strategy_Testing.py

ADD IMPORT:
```python
from trading.strategies.ensemble_methods import EnsembleStrategy
```

Create interface for combining multiple strategies into ensembles.

[Full implementation includes:
- Multi-strategy selection
- Ensemble methods (voting, weighted, stacking)
- Portfolio allocation across strategies
- Combo backtesting
- Performance comparison]

VERIFICATION:
- Can select multiple strategies
- Ensemble methods work
- Combo backtests successfully
- Results compare strategies
```

---

## CURSOR PROMPT 2.6 - Implement Strategy Comparison (Tab 5)

```
Implement Tab 5 (Strategy Comparison) in pages/2_Strategy_Testing.py

Side-by-side comparison of multiple strategies on the same data with comprehensive metrics.

[Full implementation includes:
- Multi-strategy selection
- Parallel backtesting
- Comparison charts
- Metrics table with highlighting
- Statistical significance tests]

VERIFICATION:
- Multiple strategies run in parallel
- Comparison table displays
- Charts show all strategies
- Best performer highlighted
```

---

## CURSOR PROMPT 2.7 - Implement Advanced Analysis (Tab 6)

```
Implement Tab 6 (Advanced Analysis) in pages/2_Strategy_Testing.py

ADD IMPORTS:
```python
from trading.validation.walk_forward_utils import WalkForwardAnalyzer
from trading.backtesting.monte_carlo import MonteCarloSimulator
```

Advanced testing methodologies including walk-forward and Monte Carlo.

[Full implementation includes:
- Walk-forward analysis
- Monte Carlo simulation
- Sensitivity analysis
- Regime-based testing
- Optimization]

VERIFICATION:
- Walk-forward runs correctly
- Monte Carlo completes
- Sensitivity analysis works
- Results display properly
```

---

## ✅ PAGE 2 INTEGRATION CHECKLIST

- [ ] File created: pages/2_Strategy_Testing.py
- [ ] All 6 tabs implemented
- [ ] Quick backtest functional
- [ ] Strategy builder works
- [ ] Code editor operational
- [ ] Strategy combos work
- [ ] Comparison functional
- [ ] Advanced analysis complete
- [ ] All pre-built strategies work
- [ ] Custom strategies can be created
- [ ] Error handling in place
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/2_Strategy_Testing.py
git commit -m "feat(page-2): Implement Strategy Development & Testing with 6 tabs

- Tab 1: Quick backtest with pre-built strategies
- Tab 2: Visual strategy builder
- Tab 3: Code editor for advanced users
- Tab 4: Strategy ensemble creation
- Tab 5: Multi-strategy comparison
- Tab 6: Walk-forward and Monte Carlo analysis

Merges 3 original pages with zero functionality loss"
```

---

**Next:** Proceed to PAGE_03_TRADE_EXECUTION_INTEGRATION.md
