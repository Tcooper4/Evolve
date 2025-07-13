"""Strategy page for the trading dashboard."""


import streamlit as st

from trading.ui.components import (
    create_error_block,
    create_loading_spinner,
    create_prompt_input,
    create_sidebar,
)


def render_strategy_page():
    """Render the strategy page."""
    st.title("Trading Strategy")

    # Check if router is initialized in session state
    if "router" not in st.session_state:
        st.warning("Router not initialized. Please restart the application.")

    # Create sidebar
    sidebar_config = create_sidebar()

    # Create prompt input
    prompt = create_prompt_input()

    # Process prompt if provided
    if prompt:
        try:
            with create_loading_spinner("Analyzing strategy..."):
                # Get strategy analysis from router
                result = st.session_state.router.route(
                    prompt,
                    model=sidebar_config["model"],
                    strategies=sidebar_config["strategies"],
                    expert_mode=sidebar_config["expert_mode"],
                    data_source=sidebar_config["data_source"],
                    uploaded_file=sidebar_config["uploaded_file"],
                )

                if result.status == "success":
                    # Display strategy chart
                    if result.visual:
                        st.plotly_chart(result.visual, use_container_width=True)

                    # Display strategy explanation
                    if result.explanation:
                        st.info(result.explanation)

                    # Display strategy metrics
                    if result.data and "metrics" in result.data:
                        metrics = result.data["metrics"]

                        # Performance metrics
                        st.subheader("Performance Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Return", f"{metrics.get('total_return', 0):.1%}")
                            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                        with col2:
                            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
                            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")
                        with col3:
                            st.metric("Avg Trade", f"{metrics.get('avg_trade', 0):.1%}")
                            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

                        # Strategy parameters
                        if "parameters" in metrics:
                            st.subheader("Strategy Parameters")
                            params = metrics["parameters"]
                            for param, value in params.items():
                                st.text(f"{param}: {value}")
                else:
                    create_error_block(result.error or "Failed to analyze strategy")

        except Exception as e:
            create_error_block(str(e))


if __name__ == "__main__":
    render_strategy_page()


def main():
    """Main function for the strategy page."""
    render_strategy_page()
