from datetime import datetime, timedelta
import streamlit as st
import pandas as pd


def app() -> None:
    st.title("Backtesting")

    symbol = st.text_input("Symbol", "AAPL")
    start = st.date_input("Start", datetime.now() - timedelta(days=365))
    end = st.date_input("End", datetime.now())
    strategy = st.selectbox(
        "Strategy",
        ["Momentum", "Mean Reversion", "ML-Based"],
    )

    if st.button("Run Backtest"):
        try:
            data = st.session_state.market_data.get_data(symbol, str(start), str(end))
        except Exception as e:
            st.error(f"Data load failed: {e}")
            return

        if data is None or data.empty:
            st.warning("No data available")
            return

        st.session_state.backtest_engine.data = data
        results = st.session_state.backtest_engine.run_backtest(strategy, {})
        st.session_state.backtest_results = results
        st.write(results)
        if 'returns' in results:
            st.line_chart(results['returns'].cumsum())

