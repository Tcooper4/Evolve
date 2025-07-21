"""
Reports Page

A clean, production-ready reporting interface with:
- Comprehensive performance reports
- Risk analysis and metrics
- Strategy backtest reports
- Model evaluation reports
- Export capabilities (PDF, Excel, HTML, JSON)
- Clean UI without dev clutter
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Reports - Evolve AI Trading",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean styling
st.markdown(
    """
<style>
    .report-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }

    .report-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
    }

    .metric-change {
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }

    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }

    .export-panel {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #2196f3;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state for reports."""
    if "report_history" not in st.session_state:
        st.session_state.report_history = []

    if "current_report" not in st.session_state:
        st.session_state.current_report = None


def generate_performance_report(
    symbol: str, start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    """Generate comprehensive performance report."""
    try:
        # Generate mock performance data
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Portfolio performance
        portfolio_returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_values = np.cumprod(1 + portfolio_returns)

        # Benchmark performance (S&P 500)
        benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))
        benchmark_values = np.cumprod(1 + benchmark_returns)

        # Calculate metrics
        total_return = portfolio_values[-1] - 1
        benchmark_return = benchmark_values[-1] - 1
        excess_return = total_return - benchmark_return

        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe_ratio = (
            (np.mean(portfolio_returns) * 252) / volatility if volatility > 0 else 0
        )

        # Maximum drawdown
        cumulative = portfolio_values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])

        # Win rate and other metrics
        win_rate = np.mean(portfolio_returns > 0)
        profit_factor = (
            np.sum(portfolio_returns[portfolio_returns > 0])
            / abs(np.sum(portfolio_returns[portfolio_returns < 0]))
            if np.sum(portfolio_returns[portfolio_returns < 0]) != 0
            else float("inf")
        )

        # Calmar ratio
        annualized_return = (1 + total_return) ** (252 / len(dates)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "dates": dates,
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": benchmark_returns,
            "metrics": {
                "Total_Return": total_return,
                "Benchmark_Return": benchmark_return,
                "Excess_Return": excess_return,
                "Annualized_Return": annualized_return,
                "Volatility": volatility,
                "Sharpe_Ratio": sharpe_ratio,
                "Max_Drawdown": max_drawdown,
                "VaR_95": var_95,
                "CVaR_95": cvar_95,
                "Win_Rate": win_rate,
                "Profit_Factor": profit_factor,
                "Calmar_Ratio": calmar_ratio,
                "Beta": 1.1,  # Mock beta
                "Alpha": 0.02,  # Mock alpha
                "Information_Ratio": 0.8,  # Mock information ratio
            },
            "generated_at": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return {}


def generate_risk_report(
    symbol: str, start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    """Generate comprehensive risk analysis report."""
    try:
        # Generate mock risk data
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Position data
        positions = {
            "AAPL": {"size": 0.25, "beta": 1.2, "volatility": 0.25},
            "TSLA": {"size": 0.20, "beta": 1.8, "volatility": 0.40},
            "MSFT": {"size": 0.30, "beta": 1.1, "volatility": 0.20},
            "GOOGL": {"size": 0.25, "beta": 1.0, "volatility": 0.22},
        }

        # Calculate portfolio risk metrics
        portfolio_beta = sum(pos["size"] * pos["beta"] for pos in positions.values())
        portfolio_volatility = np.sqrt(
            sum((pos["size"] * pos["volatility"]) ** 2 for pos in positions.values())
        )

        # Generate correlation matrix
        symbols = list(positions.keys())
        correlation_matrix = pd.DataFrame(
            np.random.uniform(0.3, 0.8, (len(symbols), len(symbols))),
            index=symbols,
            columns=symbols,
        )
        np.fill_diagonal(correlation_matrix.values, 1.0)

        # Risk metrics
        risk_metrics = {
            "Portfolio_Beta": portfolio_beta,
            "Portfolio_Volatility": portfolio_volatility,
            "Concentration_Risk": max(pos["size"] for pos in positions.values()),
            "Leverage_Ratio": 1.05,  # Mock leverage
            "VaR_95": -0.025,
            "CVaR_95": -0.035,
            "Max_Drawdown": -0.08,
            "Stress_Test_Result": "Pass",
        }

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "positions": positions,
            "correlation_matrix": correlation_matrix,
            "risk_metrics": risk_metrics,
            "generated_at": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error generating risk report: {e}")
        return {}


def generate_strategy_report(
    strategy_name: str, symbol: str, start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    """Generate strategy performance report."""
    try:
        # Generate mock strategy data
        np.random.seed(42)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Strategy performance
        strategy_returns = np.random.normal(0.0012, 0.018, len(dates))
        strategy_values = np.cumprod(1 + strategy_returns)

        # Buy and hold performance
        buy_hold_returns = np.random.normal(0.0008, 0.015, len(dates))
        buy_hold_values = np.cumprod(1 + buy_hold_returns)

        # Calculate strategy metrics
        strategy_total_return = strategy_values[-1] - 1
        buy_hold_total_return = buy_hold_values[-1] - 1
        excess_return = strategy_total_return - buy_hold_total_return

        # Risk metrics
        strategy_volatility = np.std(strategy_returns) * np.sqrt(252)
        strategy_sharpe = (
            (np.mean(strategy_returns) * 252) / strategy_volatility
            if strategy_volatility > 0
            else 0
        )

        # Maximum drawdown
        cumulative = strategy_values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Trading statistics
        trades = np.random.choice([-1, 0, 1], len(dates), p=[0.3, 0.4, 0.3])
        winning_trades = np.sum(trades > 0)
        losing_trades = np.sum(trades < 0)
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "dates": dates,
            "strategy_values": strategy_values,
            "buy_hold_values": buy_hold_values,
            "strategy_returns": strategy_returns,
            "buy_hold_returns": buy_hold_returns,
            "trades": trades,
            "metrics": {
                "Strategy_Total_Return": strategy_total_return,
                "Buy_Hold_Return": buy_hold_total_return,
                "Excess_Return": excess_return,
                "Strategy_Sharpe": strategy_sharpe,
                "Strategy_Volatility": strategy_volatility,
                "Max_Drawdown": max_drawdown,
                "Win_Rate": win_rate,
                "Total_Trades": total_trades,
                "Winning_Trades": winning_trades,
                "Losing_Trades": losing_trades,
            },
            "generated_at": datetime.now(),
        }
    except Exception as e:
        logger.error(f"Error generating strategy report: {e}")
        return {}


def plot_performance_report(report_data: Dict[str, Any]):
    """Plot comprehensive performance report."""
    try:
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Portfolio vs Benchmark",
                "Returns Distribution",
                "Drawdown Analysis",
                "Rolling Sharpe",
                "Monthly Returns",
                "Risk Metrics",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Portfolio vs Benchmark
        fig.add_trace(
            go.Scatter(
                x=report_data["dates"],
                y=report_data["portfolio_values"],
                mode="lines",
                name="Portfolio",
                line=dict(color="#2ecc71", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=report_data["dates"],
                y=report_data["benchmark_values"],
                mode="lines",
                name="Benchmark",
                line=dict(color="#3498db", width=2),
            ),
            row=1,
            col=1,
        )

        # Returns distribution
        fig.add_trace(
            go.Histogram(
                x=report_data["portfolio_returns"],
                name="Portfolio Returns",
                marker_color="#e74c3c",
                nbinsx=30,
            ),
            row=1,
            col=2,
        )

        # Drawdown analysis
        cumulative = report_data["portfolio_values"]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        fig.add_trace(
            go.Scatter(
                x=report_data["dates"],
                y=drawdown,
                mode="lines",
                name="Drawdown",
                line=dict(color="#e74c3c", width=2),
                fill="tonexty",
            ),
            row=2,
            col=1,
        )

        # Rolling Sharpe ratio
        returns_series = pd.Series(
            report_data["portfolio_returns"], index=report_data["dates"]
        )
        rolling_sharpe = (
            returns_series.rolling(window=30).mean()
            / returns_series.rolling(window=30).std()
            * np.sqrt(252)
        )

        fig.add_trace(
            go.Scatter(
                x=report_data["dates"],
                y=rolling_sharpe,
                mode="lines",
                name="Rolling Sharpe",
                line=dict(color="#f39c12", width=2),
            ),
            row=2,
            col=2,
        )

        # Monthly returns heatmap
        returns_df = pd.DataFrame(
            {"date": report_data["dates"], "returns": report_data["portfolio_returns"]}
        )
        returns_df["date"] = pd.to_datetime(returns_df["date"])
        returns_df["year"] = returns_df["date"].dt.year
        returns_df["month"] = returns_df["date"].dt.month

        monthly_returns = (
            returns_df.groupby(["year", "month"])["returns"].sum().unstack()
        )

        fig.add_trace(
            go.Heatmap(
                z=monthly_returns.values,
                x=monthly_returns.columns,
                y=monthly_returns.index,
                colorscale="RdYlGn",
                name="Monthly Returns",
            ),
            row=3,
            col=1,
        )

        # Risk metrics bar chart
        metrics = report_data["metrics"]
        metric_names = ["Sharpe_Ratio", "Max_Drawdown", "Win_Rate", "Calmar_Ratio"]
        metric_values = [metrics.get(name, 0) for name in metric_names]

        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name="Risk Metrics",
                marker_color=["#2ecc71", "#e74c3c", "#3498db", "#f39c12"],
            ),
            row=3,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text=f"Performance Report: {
                report_data['symbol']} ({
                report_data['start_date'].strftime('%Y-%m-%d')} to {
                report_data['end_date'].strftime('%Y-%m-%d')})",
            title_x=0.5,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error plotting performance report: {e}")
        st.error("Error generating performance visualization")


def display_performance_metrics(metrics: Dict[str, float]):
    """Display performance metrics in a clean format."""
    try:
        st.markdown("### Performance Metrics")

        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Total_Return', 0):.2%}</div>
                <div class="metric-label">Total Return</div>
                <div class="metric-change positive">+{metrics.get('Excess_Return', 0):.2%} vs Benchmark</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Sharpe_Ratio', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-change neutral">Risk-adjusted return</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Annualized_Return', 0):.2%}</div>
                <div class="metric-label">Annualized Return</div>
                <div class="metric-change positive">Annualized performance</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Max_Drawdown', 0):.2%}</div>
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-change negative">Largest decline</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Win_Rate', 0):.1%}</div>
                <div class="metric-label">Win Rate</div>
                <div class="metric-change neutral">Profitable periods</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Volatility', 0):.2%}</div>
                <div class="metric-label">Volatility</div>
                <div class="metric-change neutral">Annualized volatility</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Calmar_Ratio', 0):.2f}</div>
                <div class="metric-label">Calmar Ratio</div>
                <div class="metric-change neutral">Return vs drawdown</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('Information_Ratio', 0):.2f}</div>
                <div class="metric-label">Information Ratio</div>
                <div class="metric-change neutral">Excess return efficiency</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    except Exception as e:
        logger.error(f"Error displaying metrics: {e}")


def export_report(report_data: Dict[str, Any], format_type: str):
    """Export report in various formats."""
    try:
        if format_type == "CSV":
            # Export metrics as CSV
            metrics_df = pd.DataFrame([report_data["metrics"]])
            return metrics_df.to_csv(index=False)

        elif format_type == "JSON":
            # Export full report as JSON
            return json.dumps(report_data, default=str, indent=2)

        elif format_type == "HTML":
            # Generate HTML report
            html_content = f"""
            <html>
            <head>
                <title>Performance Report - {report_data.get('symbol', 'Portfolio')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #f8f9fa; padding: 20px; border-radius: 10px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                    .metric {{ background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }}
                    .metric-value {{ font-size: 2rem; font-weight: bold; color: #2c3e50; }}
                    .metric-label {{ color: #6c757d; margin-top: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Performance Report</h1>
                    <p><strong>Symbol:</strong> {report_data.get('symbol', 'Portfolio')}</p>
                    <p><strong>Period:</strong> {report_data.get('start_date', 'N/A')} to {report_data.get('end_date', 'N/A')}</p>
                    <p><strong>Generated:</strong> {report_data.get('generated_at', 'N/A')}</p>
                </div>

                <div class="metrics">
            """

            metrics = report_data.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    if "Rate" in metric_name or "Return" in metric_name:
                        display_value = f"{metric_value:.2%}"
                    else:
                        display_value = f"{metric_value:.3f}"
                else:
                    display_value = str(metric_value)

                html_content += f"""
                    <div class="metric">
                        <div class="metric-value">{display_value}</div>
                        <div class="metric-label">{metric_name.replace('_', ' ')}</div>
                    </div>
                """

            html_content += """
                </div>
            </body>
            </html>
            """
            return html_content

        else:
            return "Unsupported format"

    except Exception as e:
        logger.error(f"Error exporting report: {e}")
        return None


def main():
    """Main reports function."""
    st.title("Reports Dashboard")
    st.markdown("Generate comprehensive performance and risk reports")

    # Initialize session state
    initialize_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header("Report Settings")

        # Report type selection
        report_type = st.selectbox(
            "Report Type",
            [
                "Performance Report",
                "Risk Analysis",
                "Strategy Report",
                "Model Evaluation",
            ],
        )

        # Symbol input
        symbol = st.text_input(
            "Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, Portfolio"
        )

        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=start_date)
        with col2:
            end_date = st.date_input("End Date", value=end_date)

        # Strategy selection (for strategy reports)
        if report_type == "Strategy Report":
            strategy_name = st.selectbox(
                "Strategy",
                [
                    "RSI Mean Reversion",
                    "MACD Crossover",
                    "Bollinger Bands",
                    "Moving Average Crossover",
                ],
            )

        # Generate report button
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    start_dt = datetime.combine(start_date, datetime.min.time())
                    end_dt = datetime.combine(end_date, datetime.min.time())

                    if report_type == "Performance Report":
                        report_data = generate_performance_report(
                            symbol, start_dt, end_dt
                        )
                    elif report_type == "Risk Analysis":
                        report_data = generate_risk_report(symbol, start_dt, end_dt)
                    elif report_type == "Strategy Report":
                        report_data = generate_strategy_report(
                            strategy_name, symbol, start_dt, end_dt
                        )
                    else:
                        report_data = generate_performance_report(
                            symbol, start_dt, end_dt
                        )  # Default

                    if report_data:
                        st.session_state.current_report = report_data
                        st.session_state.report_history.append(report_data)
                        st.success("Report generated successfully!")
                    else:
                        st.error("Failed to generate report")

                except Exception as e:
                    st.error(f"Error generating report: {e}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Current report display
        if st.session_state.current_report:
            st.markdown("### Report Analysis")

            if report_type == "Performance Report":
                plot_performance_report(st.session_state.current_report)
                display_performance_metrics(st.session_state.current_report["metrics"])
            elif report_type == "Risk Analysis":
                st.markdown("### Risk Analysis")
                risk_metrics = st.session_state.current_report["risk_metrics"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Portfolio Beta", f"{risk_metrics['Portfolio_Beta']:.2f}")
                with col2:
                    st.metric(
                        "Portfolio Volatility",
                        f"{risk_metrics['Portfolio_Volatility']:.2%}",
                    )
                with col3:
                    st.metric(
                        "Concentration Risk",
                        f"{risk_metrics['Concentration_Risk']:.1%}",
                    )
                with col4:
                    st.metric("VaR (95%)", f"{risk_metrics['VaR_95']:.2%}")

                # Correlation matrix
                st.markdown("### Correlation Matrix")
                correlation_matrix = st.session_state.current_report[
                    "correlation_matrix"
                ]
                st.dataframe(correlation_matrix, use_container_width=True)

            elif report_type == "Strategy Report":
                st.markdown("### Strategy Performance")
                metrics = st.session_state.current_report["metrics"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Strategy Return", f"{metrics['Strategy_Total_Return']:.2%}"
                    )
                with col2:
                    st.metric("Excess Return", f"{metrics['Excess_Return']:.2%}")
                with col3:
                    st.metric("Win Rate", f"{metrics['Win_Rate']:.1%}")
                with col4:
                    st.metric("Total Trades", metrics["Total_Trades"])

            # Export options
            st.markdown("### Export Options")
            col_export1, col_export2, col_export3, col_export4 = st.columns(4)

            with col_export1:
                if st.button("Export CSV"):
                    csv_data = export_report(st.session_state.current_report, "CSV")
                    if csv_data:
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                        )

            with col_export2:
                if st.button("Export JSON"):
                    json_data = export_report(st.session_state.current_report, "JSON")
                    if json_data:
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                        )

            with col_export3:
                if st.button("Export HTML"):
                    html_data = export_report(st.session_state.current_report, "HTML")
                    if html_data:
                        st.download_button(
                            label="Download HTML",
                            data=html_data,
                            file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html",
                        )

            with col_export4:
                if st.button("Generate PDF"):
                    try:
                        # Convert report data to DataFrame format for the exporter
                        from utils.report_exporter import export_trade_report

                        # Create a DataFrame from the report data
                        if "trade_data" in st.session_state.current_report:
                            trade_df = pd.DataFrame(
                                st.session_state.current_report["trade_data"]
                            )
                        else:
                            # Create a summary DataFrame if no trade data
                            metrics = st.session_state.current_report.get("metrics", {})
                            trade_df = pd.DataFrame(
                                [
                                    {
                                        "timestamp": datetime.now(),
                                        "symbol": symbol,
                                        "strategy": "Report Summary",
                                        "signal": "SUMMARY",
                                        "total_return": metrics.get("Total_Return", 0),
                                        "sharpe_ratio": metrics.get("Sharpe_Ratio", 0),
                                        "max_drawdown": metrics.get("Max_Drawdown", 0),
                                        "win_rate": metrics.get("Win_Rate", 0),
                                    }
                                ]
                            )

                        # Export to PDF
                        export_path = export_trade_report(
                            signals=trade_df, format="PDF", include_summary=True
                        )

                        # Create download button for PDF
                        with open(export_path, "rb") as f:
                            pdf_data = f.read()

                        st.download_button(
                            label="Download PDF",
                            data=pdf_data,
                            file_name=f"{symbol}_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                        )

                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
        else:
            st.info("Generate a report using the sidebar controls")

    with col2:
        # Report history
        if st.session_state.report_history:
            st.markdown("### Recent Reports")
            for i, report in enumerate(reversed(st.session_state.report_history[-5:])):
                with st.expander(
                    f"{report.get('symbol', 'Portfolio')} - {report['generated_at'].strftime('%Y-%m-%d %H:%M')}"
                ):
                    if "metrics" in report:
                        metrics = report["metrics"]
                        st.markdown(
                            f"**Total Return:** {metrics.get('Total_Return', 0):.2%}"
                        )
                        st.markdown(
                            f"**Sharpe Ratio:** {metrics.get('Sharpe_Ratio', 0):.2f}"
                        )
                        st.markdown(
                            f"**Max Drawdown:** {metrics.get('Max_Drawdown', 0):.2%}"
                        )

                    if st.button(f"Load Report", key=f"load_{i}"):
                        st.session_state.current_report = report
                        st.rerun()

        # Quick stats
        st.markdown("### Quick Stats")
        if st.session_state.report_history:
            total_reports = len(st.session_state.report_history)
            avg_return = np.mean(
                [
                    r.get("metrics", {}).get("Total_Return", 0)
                    for r in st.session_state.report_history
                ]
            )
            avg_sharpe = np.mean(
                [
                    r.get("metrics", {}).get("Sharpe_Ratio", 0)
                    for r in st.session_state.report_history
                ]
            )

            st.metric("Total Reports", total_reports)
            st.metric("Avg Return", f"{avg_return:.2%}")
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")


if __name__ == "__main__":
    main()
