"""Backtest page for the trading dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from trading.ui.components import (
    create_prompt_input,
    create_sidebar,
    create_performance_report,
    create_error_block,
    create_loading_spinner
)

def render_backtest_page():
    """Render the backtest page."""
    st.title("Strategy Backtesting")
    
    # Create sidebar
    sidebar_config = create_sidebar()
    
    # Create prompt input
    prompt = create_prompt_input()
    
    # Process prompt if provided
    if prompt:
        try:
            with create_loading_spinner("Running backtest..."):
                # Get backtest results from router
                result = st.session_state.router.route(
                    prompt,
                    model=sidebar_config['model'],
                    strategies=sidebar_config['strategies'],
                    expert_mode=sidebar_config['expert_mode'],
                    data_source=sidebar_config['data_source'],
                    uploaded_file=sidebar_config['uploaded_file']
                )
                
                if result.status == "success":
                    # Display backtest explanation
                    if result.explanation:
                        st.info(result.explanation)
                    
                    # Display performance report
                    if result.data:
                        create_performance_report(result.data)
                    
                    # Display trade log if available
                    if result.data and 'trade_log' in result.data:
                        st.subheader("Trade Log")
                        trade_log = pd.DataFrame(result.data['trade_log'])
                        st.dataframe(trade_log)
                        
                        # Export trade log
                        if st.button("Export Trade Log"):
                            csv = trade_log.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="trade_log.csv",
                                mime="text/csv"
                            )
                else:
                    create_error_block(result.error or "Failed to run backtest")
                    
        except Exception as e:
            create_error_block(str(e))

if __name__ == "__main__":
    render_backtest_page()

