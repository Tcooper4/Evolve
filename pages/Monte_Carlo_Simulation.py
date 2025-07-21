"""
Monte Carlo Simulation Dashboard

This page provides an interactive interface for running Monte Carlo simulations
on portfolio returns, with configurable parameters and comprehensive visualizations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trading.backtesting.monte_carlo import MonteCarloConfig, MonteCarloSimulator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Monte Carlo simulation


def generate_sample_returns(
    n_days: int = 252,
    mean_return: float = 0.0005,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.Series:
    """Generate sample returns for demonstration."""
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    returns = pd.Series(np.random.normal(mean_return, volatility, n_days), index=dates)
    return returns


def create_monte_carlo_plot(
    simulator: MonteCarloSimulator,
    show_individual_paths: bool = True,
    n_paths_to_show: int = 50,
) -> go.Figure:
    """Create an interactive Plotly visualization of Monte Carlo results."""

    if simulator.simulated_paths is None or simulator.percentiles is None:
        st.error("No simulation data available")
        return None

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Portfolio Value Paths", "Return Distribution"),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )

    # Plot 1: Portfolio Value Paths
    if show_individual_paths:
        # Show subset of individual paths
        paths_to_show = min(n_paths_to_show, simulator.simulated_paths.shape[1])
        for i in range(paths_to_show):
            fig.add_trace(
                go.Scatter(
                    x=simulator.simulated_paths.index,
                    y=simulator.simulated_paths.iloc[:, i],
                    mode="lines",
                    line=dict(color="lightgray", width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # Add confidence bands
    fig.add_trace(
        go.Scatter(
            x=simulator.percentiles.index,
            y=simulator.percentiles["P95"],
            mode="lines",
            line=dict(color="blue", width=2),
            name="95th Percentile",
            fill=None,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=simulator.percentiles.index,
            y=simulator.percentiles["P5"],
            mode="lines",
            line=dict(color="blue", width=2),
            name="5th Percentile",
            fill="tonexty",
            fillcolor="rgba(0, 100, 255, 0.2)",
        ),
        row=1,
        col=1,
    )

    # Add median and mean
    fig.add_trace(
        go.Scatter(
            x=simulator.percentiles.index,
            y=simulator.percentiles["P50"],
            mode="lines",
            line=dict(color="red", width=3),
            name="Median (50th percentile)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=simulator.percentiles.index,
            y=simulator.percentiles["Mean"],
            mode="lines",
            line=dict(color="green", width=3, dash="dash"),
            name="Mean Path",
        ),
        row=1,
        col=1,
    )

    # Plot 2: Return Distribution
    final_returns = (
        simulator.simulated_paths.iloc[-1] - simulator.results["initial_capital"]
    ) / simulator.results["initial_capital"]

    fig.add_trace(
        go.Histogram(
            x=final_returns,
            nbinsx=50,
            name="Return Distribution",
            marker_color="skyblue",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # Add vertical lines for key statistics
    mean_return = final_returns.mean()
    p5_return = np.percentile(final_returns, 5)
    p95_return = np.percentile(final_returns, 95)

    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_return:.2%}",
        row=2,
        col=1,
    )

    fig.add_vline(
        x=p5_return,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"5th percentile: {p5_return:.2%}",
        row=2,
        col=1,
    )

    fig.add_vline(
        x=p95_return,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"95th percentile: {p95_return:.2%}",
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title="Monte Carlo Simulation Results",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_xaxes(title_text="Total Return", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    return fig


def display_summary_statistics(stats: dict):
    """Display summary statistics in a clean format."""

    st.subheader("ðŸ“Š Summary Statistics")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean Final Value",
            f"${stats['mean_final_value']:,.0f}",
            f"{stats['mean_total_return']:.1%}",
        )

        st.metric(
            "5th Percentile", f"${stats['final_p5']:,.0f}", f"{stats['return_p5']:.1%}"
        )

    with col2:
        st.metric(
            "95th Percentile",
            f"${stats['final_p95']:,.0f}",
            f"{stats['return_p95']:.1%}",
        )

        st.metric("Probability of Loss", f"{stats['probability_of_loss']:.1%}")

    with col3:
        st.metric("95% VaR", f"{stats['var_95']:.1%}")

        st.metric("95% CVaR", f"{stats['cvar_95']:.1%}")

    with col4:
        st.metric("Volatility", f"{stats['std_total_return']:.1%}")

        st.metric("Max Drawdown (Median)", f"{stats.get('max_drawdown_p50', 0):.1%}")


def main():
    """Main function for the Monte Carlo Simulation dashboard."""

    st.set_page_config(
        page_title="ðŸŽ² Monte Carlo Simulation",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ðŸŽ² Monte Carlo Simulation Dashboard")
    st.markdown(
        "Simulate portfolio performance using bootstrapped historical returns "
        "and analyze risk through percentile bands and confidence intervals."
    )

    # Sidebar configuration
    st.sidebar.header("âš™ï¸ Simulation Configuration")

    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Sample Data", "Upload Returns File"],
        help="Choose data source for simulation",
    )

    if data_source == "Sample Data":
        # Sample data configuration
        st.sidebar.subheader("ðŸ“Š Sample Data Parameters")

        n_days = st.sidebar.slider(
            "Number of Days",
            min_value=30,
            max_value=1000,
            value=252,
            help="Number of trading days to simulate",
        )

        mean_return = st.sidebar.number_input(
            "Mean Daily Return",
            value=0.0005,
            format="%.4f",
            help="Expected daily return (e.g., 0.0005 = 0.05%)",
        )

        volatility = st.sidebar.number_input(
            "Daily Volatility",
            value=0.02,
            format="%.3f",
            help="Daily volatility (e.g., 0.02 = 2%)",
        )

        # Generate sample data
        returns = generate_sample_returns(n_days, mean_return, volatility)

    else:
        # File upload
        st.sidebar.subheader("ðŸ“ Upload Returns File")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file with returns",
            type=["csv"],
            help="File should have a 'returns' column or be a single column of returns",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if "returns" in df.columns:
                    returns = pd.Series(df["returns"].values)
                elif len(df.columns) == 1:
                    returns = pd.Series(df.iloc[:, 0].values)
                else:
                    st.error(
                        "File must have a 'returns' column or be a single column of returns"
                    )
                    return
                st.success(f"Loaded {len(returns)} return observations")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
        else:
            st.info("Please upload a returns file to continue")
            return

    # Simulation parameters
    st.sidebar.subheader("ðŸŽ¯ Simulation Parameters")

    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        value=10000.0,
        min_value=1000.0,
        max_value=1000000.0,
        step=1000.0,
        help="Starting portfolio value",
    )

    n_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of Monte Carlo paths to simulate",
    )

    bootstrap_method = st.sidebar.selectbox(
        "Bootstrap Method",
        ["historical", "block", "parametric"],
        help="Method for generating return paths",
    )

    if bootstrap_method == "block":
        block_size = st.sidebar.slider(
            "Block Size",
            min_value=5,
            max_value=50,
            value=20,
            help="Size of blocks for block bootstrap",
        )
    else:
        block_size = 20

    # Confidence levels
    st.sidebar.subheader("ðŸ“ˆ Confidence Levels")

    p5 = st.sidebar.checkbox("5th Percentile", value=True)
    p50 = st.sidebar.checkbox("50th Percentile (Median)", value=True)
    p95 = st.sidebar.checkbox("95th Percentile", value=True)

    confidence_levels = []
    if p5:
        confidence_levels.append(0.05)
    if p50:
        confidence_levels.append(0.50)
    if p95:
        confidence_levels.append(0.95)

    if not confidence_levels:
        st.warning("Please select at least one confidence level")
        return

    # Run simulation button
    if st.sidebar.button("ðŸš€ Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # Create configuration
                config = MonteCarloConfig(
                    n_simulations=n_simulations,
                    confidence_levels=confidence_levels,
                    bootstrap_method=bootstrap_method,
                    block_size=block_size,
                    initial_capital=initial_capital,
                )

                # Create and run simulator
                simulator = MonteCarloSimulator(config)
                simulator.simulate_portfolio_paths(
                    returns, initial_capital, n_simulations
                )
                simulator.calculate_percentiles(confidence_levels)

                # Store in session state
                st.session_state.simulator = simulator
                st.session_state.results = simulator.create_detailed_report()

                st.success(
                    f"âœ… Simulation completed! Generated {n_simulations} paths."
                )

            except Exception as e:
                st.error(f"âŒ Simulation failed: {str(e)}")
                return

    # Display results if available
    if (
        hasattr(st.session_state, "simulator")
        and st.session_state.simulator is not None
    ):
        simulator = st.session_state.simulator
        results = st.session_state.results

        # Display summary statistics
        display_summary_statistics(results["summary_statistics"])

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "ðŸ“ˆ Portfolio Paths",
                "ðŸ“Š Risk Analysis",
                "ðŸ“‹ Detailed Report",
                "ðŸŽ¨ Custom Visualization",
            ]
        )

        with tab1:
            st.subheader("Portfolio Value Paths")

            # Visualization options
            col1, col2 = st.columns(2)

            with col1:
                show_paths = st.checkbox("Show Individual Paths", value=True)
                n_paths_to_show = st.slider(
                    "Number of Paths to Show", min_value=10, max_value=200, value=50
                )

            with col2:
                show_confidence = st.checkbox("Show Confidence Bands", value=True)
                show_mean = st.checkbox("Show Mean Path", value=True)

            # Create plot
            fig = create_monte_carlo_plot(
                simulator,
                show_individual_paths=show_paths,
                n_paths_to_show=n_paths_to_show,
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Risk Analysis")

            # Risk metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ðŸ“‰ Value at Risk (VaR)")

                risk_data = {
                    "Metric": ["95% VaR", "99% VaR", "95% CVaR", "99% CVaR"],
                    "Value": [
                        f"{results['summary_statistics']['var_95']:.2%}",
                        f"{results['summary_statistics']['var_99']:.2%}",
                        f"{results['summary_statistics']['cvar_95']:.2%}",
                        f"{results['summary_statistics']['cvar_99']:.2%}",
                    ],
                }

                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)

            with col2:
                st.markdown("### ðŸŽ¯ Probability Analysis")

                prob_data = {
                    "Event": [
                        "Probability of Loss",
                        "Probability of 20% Loss",
                        "Probability of 50% Loss",
                    ],
                    "Probability": [
                        f"{results['summary_statistics']['probability_of_loss']:.2%}",
                        f"{results['summary_statistics']['probability_of_20_percent_loss']:.2%}",
                        f"{results['summary_statistics']['probability_of_50_percent_loss']:.2%}",
                    ],
                }

                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True)

            # Volatility analysis
            st.markdown("### ðŸ“Š Volatility Analysis")

            vol_analysis = results["percentile_analysis"]["volatility_analysis"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean Volatility", f"{vol_analysis['mean_volatility']:.2%}")
            with col2:
                st.metric("Volatility Std", f"{vol_analysis['volatility_std']:.2%}")
            with col3:
                st.metric("Min Volatility", f"{vol_analysis['min_volatility']:.2%}")
            with col4:
                st.metric("Max Volatility", f"{vol_analysis['max_volatility']:.2%}")

        with tab3:
            st.subheader("Detailed Report")

            # Display full report
            st.json(results)

            # Download report
            import json

            report_json = json.dumps(results, indent=2, default=str)

            st.download_button(
                label="ðŸ“¥ Download Report (JSON)",
                data=report_json,
                file_name=f"monte_carlo_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        with tab4:
            st.subheader("Custom Visualization")

            # Custom plot options
            col1, col2 = st.columns(2)

            with col1:
                plot_type = st.selectbox(
                    "Plot Type",
                    [
                        "Portfolio Paths",
                        "Return Distribution",
                        "Drawdown Analysis",
                        "Volatility Over Time",
                    ],
                )

            with col2:
                if plot_type == "Portfolio Paths":
                    show_bands = st.checkbox("Show Confidence Bands", value=True)
                elif plot_type == "Return Distribution":
                    show_stats = st.checkbox("Show Statistics", value=True)

            # Create custom plots
            if plot_type == "Portfolio Paths":
                fig = create_monte_carlo_plot(simulator, show_individual_paths=True)
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Return Distribution":
                final_returns = (
                    simulator.simulated_paths.iloc[-1]
                    - simulator.results["initial_capital"]
                ) / simulator.results["initial_capital"]

                fig = px.histogram(
                    x=final_returns,
                    nbins=50,
                    title="Distribution of Final Returns",
                    labels={"x": "Total Return", "y": "Frequency"},
                )

                if show_stats:
                    fig.add_vline(
                        x=final_returns.mean(), line_dash="dash", line_color="red"
                    )
                    fig.add_vline(
                        x=np.percentile(final_returns, 5),
                        line_dash="dash",
                        line_color="orange",
                    )
                    fig.add_vline(
                        x=np.percentile(final_returns, 95),
                        line_dash="dash",
                        line_color="orange",
                    )

                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Drawdown Analysis":
                # Calculate drawdowns for each path
                drawdowns = []
                for col in simulator.simulated_paths.columns:
                    equity_curve = simulator.simulated_paths[col]
                    running_max = equity_curve.cummax()
                    drawdown = (equity_curve - running_max) / running_max
                    drawdowns.append(drawdown.min())

                fig = px.histogram(
                    x=drawdowns,
                    nbins=30,
                    title="Distribution of Maximum Drawdowns",
                    labels={"x": "Maximum Drawdown", "y": "Frequency"},
                )

                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Volatility Over Time":
                # Calculate rolling volatility for each path
                daily_returns = simulator.simulated_paths.pct_change().dropna()
                rolling_vol = daily_returns.rolling(window=20).std()

                # Plot mean volatility over time
                mean_vol = rolling_vol.mean(axis=1)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=mean_vol.index,
                        y=mean_vol * 100,
                        mode="lines",
                        name="Mean Volatility (20-day)",
                    )
                )

                fig.update_layout(
                    title="Volatility Over Time",
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        # Show data preview
        st.subheader("ðŸ“Š Data Preview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Returns Statistics")
            returns_stats = {
                "Mean": f"{returns.mean():.4f}",
                "Std": f"{returns.std():.4f}",
                "Min": f"{returns.min():.4f}",
                "Max": f"{returns.max():.4f}",
                "Skewness": f"{returns.skew():.4f}",
                "Kurtosis": f"{returns.kurtosis():.4f}",
            }

            for stat, value in returns_stats.items():
                st.metric(stat, value)

        with col2:
            st.markdown("### Returns Distribution")
            fig = px.histogram(
                x=returns,
                nbins=50,
                title="Distribution of Returns",
                labels={"x": "Return", "y": "Frequency"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "ðŸ‘ˆ Configure simulation parameters in the sidebar and click 'Run Monte Carlo Simulation' to start."
        )


if __name__ == "__main__":
    main()
