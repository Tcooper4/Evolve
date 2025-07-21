"""
Hybrid Model Page

This page provides a comprehensive interface for hybrid model management with
auto-adjusting weights based on past performance and real-time ensemble composition display.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)


class HybridModelManager:
    """
    Manager for hybrid model operations with auto-adjusting weights.

    Features:
    - Auto-adjust weights based on MSE or Sharpe ratio
    - Performance tracking and visualization
    - Real-time ensemble composition updates
    - Historical weight evolution tracking
    """

    def __init__(self):
        """Initialize the hybrid model manager."""
        self.performance_history = {}
        self.weight_history = {}
        self.current_weights = {}
        self.ensemble_model = None

    def calculate_adaptive_weights(
        self,
        model_performances: Dict[str, float],
        method: str = "sharpe",
        recency_weight: float = 0.7,
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on model performance.

        Args:
            model_performances: Dictionary of model performance metrics
            method: Weighting method ('sharpe', 'mse', 'inverse_mse')
            recency_weight: Weight for recent performance vs historical

        Returns:
            Dictionary of normalized weights
        """
        if not model_performances:
            return {}

        # Calculate base weights based on method
        if method == "sharpe":
            # Higher Sharpe ratio = higher weight
            weights = {
                model: max(0, perf) for model, perf in model_performances.items()
            }
        elif method == "mse":
            # Lower MSE = higher weight
            min_mse = min(model_performances.values())
            weights = {
                model: max(0, min_mse / max(perf, 1e-6))
                for model, perf in model_performances.items()
            }
        elif method == "inverse_mse":
            # Inverse MSE weighting
            weights = {
                model: 1.0 / max(perf, 1e-6)
                for model, perf in model_performances.items()
            }
        else:
            # Equal weights as fallback
            weights = {model: 1.0 for model in model_performances.keys()}

        # Apply recency weighting if historical data available
        if self.weight_history and recency_weight > 0:
            historical_weights = self._get_historical_weights(
                list(model_performances.keys())
            )
            if historical_weights:
                for model in weights:
                    if model in historical_weights:
                        weights[model] = (
                            recency_weight * weights[model]
                            + (1 - recency_weight) * historical_weights[model]
                        )

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {
                model: weight / total_weight for model, weight in weights.items()
            }
        else:
            # Equal weights if all are zero
            normalized_weights = {model: 1.0 / len(weights) for model in weights.keys()}

        return normalized_weights

    def _get_historical_weights(self, models: List[str]) -> Dict[str, float]:
        """Get historical weights for the specified models."""
        if not self.weight_history:
            return {}

        # Get most recent weights
        recent_weights = {}
        for model in models:
            if model in self.weight_history:
                weights = self.weight_history[model]
                if weights:
                    recent_weights[model] = weights[-1]

        return recent_weights

    def update_performance(
        self, model: str, performance: float, metric: str = "sharpe"
    ):
        """Update performance history for a model."""
        if model not in self.performance_history:
            self.performance_history[model] = []

        self.performance_history[model].append(
            {"timestamp": datetime.now(), "performance": performance, "metric": metric}
        )

        # Keep only recent history (last 100 entries)
        if len(self.performance_history[model]) > 100:
            self.performance_history[model] = self.performance_history[model][-100:]

    def update_weights(self, weights: Dict[str, float]):
        """Update weight history."""
        timestamp = datetime.now()

        for model, weight in weights.items():
            if model not in self.weight_history:
                self.weight_history[model] = []

            self.weight_history[model].append(
                {"timestamp": timestamp, "weight": weight}
            )

            # Keep only recent history (last 50 entries)
            if len(self.weight_history[model]) > 50:
                self.weight_history[model] = self.weight_history[model][-50:]

        self.current_weights = weights.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        summary = {
            "total_models": len(self.performance_history),
            "active_models": list(self.current_weights.keys()),
            "current_weights": self.current_weights.copy(),
            "performance_trends": {},
            "weight_stability": {},
        }

        # Calculate performance trends
        for model, history in self.performance_history.items():
            if len(history) >= 2:
                recent_perf = history[-1]["performance"]
                older_perf = history[-min(5, len(history))]["performance"]
                trend = (recent_perf - older_perf) / max(abs(older_perf), 1e-6)
                summary["performance_trends"][model] = trend

        # Calculate weight stability
        for model, history in self.weight_history.items():
            if len(history) >= 2:
                weights = [entry["weight"] for entry in history[-10:]]
                stability = 1.0 - np.std(weights) / max(np.mean(weights), 1e-6)
                summary["weight_stability"][model] = max(0, min(1, stability))

        return summary


def create_ensemble_composition_sidebar(manager: HybridModelManager):
    """Create sidebar showing active ensemble composition."""
    st.sidebar.header("ðŸŽ¯ Ensemble Composition")

    if not manager.current_weights:
        st.sidebar.info("No active ensemble weights")
        return

    # Display current weights
    st.sidebar.subheader("Current Weights")

    # Create weight bars
    for model, weight in manager.current_weights.items():
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.progress(weight)
        with col2:
            st.write(f"{weight:.1%}")

        # Show model name
        st.sidebar.caption(f"**{model}**")

    # Performance summary
    st.sidebar.subheader("ðŸ“Š Performance Summary")
    summary = manager.get_performance_summary()

    # Active models count
    st.sidebar.metric("Active Models", summary["total_models"])

    # Weight stability indicator
    if summary["weight_stability"]:
        avg_stability = np.mean(list(summary["weight_stability"].values()))
        st.sidebar.metric("Weight Stability", f"{avg_stability:.1%}")

    # Performance trends
    if summary["performance_trends"]:
        st.sidebar.subheader("ðŸ“ˆ Performance Trends")
        for model, trend in summary["performance_trends"].items():
            trend_icon = "ðŸ“ˆ" if trend > 0 else "ðŸ“‰" if trend < 0 else "âž¡ï¸"
            st.sidebar.write(f"{trend_icon} {model}: {trend:+.1%}")


def create_weight_evolution_chart(manager: HybridModelManager):
    """Create chart showing weight evolution over time."""
    if not manager.weight_history:
        st.warning("No weight history available")
        return

    # Prepare data for plotting
    data = []
    for model, history in manager.weight_history.items():
        for entry in history:
            data.append(
                {
                    "Model": model,
                    "Weight": entry["weight"],
                    "Timestamp": entry["timestamp"],
                }
            )

    if not data:
        return

    df = pd.DataFrame(data)

    # Create line chart
    fig = px.line(
        df,
        x="Timestamp",
        y="Weight",
        color="Model",
        title="Weight Evolution Over Time",
        labels={"Weight": "Model Weight", "Timestamp": "Time"},
    )

    fig.update_layout(xaxis_title="Time", yaxis_title="Weight", hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)


def create_performance_comparison_chart(manager: HybridModelManager):
    """Create chart comparing model performances."""
    if not manager.performance_history:
        st.warning("No performance history available")
        return

    # Prepare data for plotting
    data = []
    for model, history in manager.performance_history.items():
        for entry in history:
            data.append(
                {
                    "Model": model,
                    "Performance": entry["performance"],
                    "Metric": entry["metric"],
                    "Timestamp": entry["timestamp"],
                }
            )

    if not data:
        return

    df = pd.DataFrame(data)

    # Create performance comparison chart
    fig = px.line(
        df,
        x="Timestamp",
        y="Performance",
        color="Model",
        title="Model Performance Comparison",
        labels={"Performance": "Performance Metric", "Timestamp": "Time"},
    )

    fig.update_layout(
        xaxis_title="Time", yaxis_title="Performance", hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


def create_ensemble_optimization_interface(manager: HybridModelManager):
    """Create interface for ensemble optimization."""
    st.header("ðŸ”§ Ensemble Optimization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weighting Method")
        weighting_method = st.selectbox(
            "Select weighting method:",
            ["sharpe", "mse", "inverse_mse"],
            help="Method for calculating adaptive weights",
        )

        recency_weight = st.slider(
            "Recency Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Weight given to recent performance vs historical",
        )

    with col2:
        st.subheader("Performance Metrics")

        # Mock performance data (replace with actual data)
        mock_performances = {
            "LSTM": 0.85,
            "XGBoost": 0.92,
            "ARIMA": 0.78,
            "Prophet": 0.81,
        }

        # Display current performances
        for model, perf in mock_performances.items():
            st.metric(model, f"{perf:.3f}")

    # Calculate new weights
    if st.button("ðŸ”„ Recalculate Weights"):
        new_weights = manager.calculate_adaptive_weights(
            mock_performances, method=weighting_method, recency_weight=recency_weight
        )

        manager.update_weights(new_weights)

        # Update performance history
        for model, perf in mock_performances.items():
            manager.update_performance(model, perf, weighting_method)

        st.success("Weights updated successfully!")
        st.experimental_rerun()


def create_backtest_interface(manager: HybridModelManager):
    """Create interface for backtesting the ensemble."""
    st.header("ðŸ§ª Backtesting")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=365),
            max_value=datetime.now().date(),
        )

        end_date = st.date_input(
            "End Date", value=datetime.now().date(), max_value=datetime.now().date()
        )

    with col2:
        initial_capital = st.number_input(
            "Initial Capital ($)", min_value=1000, value=10000, step=1000
        )

        risk_free_rate = (
            st.number_input(
                "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1
            )
            / 100
        )

    if st.button("ðŸš€ Run Backtest"):
        with st.spinner("Running backtest..."):
            # Mock backtest results (replace with actual backtesting)
            backtest_results = {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "volatility": 0.12,
                "win_rate": 0.65,
            }

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Return", f"{backtest_results['total_return']:.1%}")
                st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")

            with col2:
                st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.1%}")
                st.metric("Volatility", f"{backtest_results['volatility']:.1%}")

            with col3:
                st.metric("Win Rate", f"{backtest_results['win_rate']:.1%}")

                # Update ensemble performance
                ensemble_performance = backtest_results["sharpe_ratio"]
                manager.update_performance("Ensemble", ensemble_performance, "sharpe")


def main():
    """Main function for the Hybrid Model page."""
    st.set_page_config(page_title="Hybrid Model", page_icon="ðŸŽ¯", layout="wide")

    st.title("ðŸŽ¯ Hybrid Model Management")
    st.markdown("---")

    # Initialize manager
    if "hybrid_manager" not in st.session_state:
        st.session_state.hybrid_manager = HybridModelManager()

    manager = st.session_state.hybrid_manager

    # Create sidebar
    create_ensemble_composition_sidebar(manager)

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(
        ["ðŸ“Š Overview", "ðŸ”§ Optimization", "ðŸ§ª Backtesting", "ðŸ“ˆ Analytics"]
    )

    with tab1:
        st.header("ðŸ“Š Ensemble Overview")

        if manager.current_weights:
            # Current ensemble status
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Current Ensemble")

                # Create weight distribution pie chart
                if manager.current_weights:
                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=list(manager.current_weights.keys()),
                                values=list(manager.current_weights.values()),
                                hole=0.3,
                            )
                        ]
                    )

                    fig.update_layout(
                        title="Current Weight Distribution", showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Performance Summary")
                summary = manager.get_performance_summary()

                # Display key metrics
                if summary["weight_stability"]:
                    avg_stability = np.mean(list(summary["weight_stability"].values()))
                    st.metric("Weight Stability", f"{avg_stability:.1%}")

                if summary["performance_trends"]:
                    avg_trend = np.mean(list(summary["performance_trends"].values()))
                    st.metric("Performance Trend", f"{avg_trend:+.1%}")

                st.metric("Active Models", summary["total_models"])
        else:
            st.info(
                "No ensemble configured. Use the Optimization tab to set up your ensemble."
            )

    with tab2:
        create_ensemble_optimization_interface(manager)

    with tab3:
        create_backtest_interface(manager)

    with tab4:
        st.header("ðŸ“ˆ Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Weight Evolution")
            create_weight_evolution_chart(manager)

        with col2:
            st.subheader("Performance Comparison")
            create_performance_comparison_chart(manager)

        # Additional analytics
        if manager.performance_history:
            st.subheader("ðŸ“‹ Detailed Performance")

            # Create performance table
            performance_data = []
            for model, history in manager.performance_history.items():
                if history:
                    latest = history[-1]
                    performance_data.append(
                        {
                            "Model": model,
                            "Latest Performance": latest["performance"],
                            "Metric": latest["metric"],
                            "Last Updated": latest["timestamp"].strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                        }
                    )

            if performance_data:
                df = pd.DataFrame(performance_data)
                st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
