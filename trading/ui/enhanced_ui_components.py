"""
Enhanced UI Components for Trading System

This module provides enhanced UI components including dynamic sliders,
confidence score displays, session summary bars, and log viewers.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ParameterConfig:
    """Configuration for a UI parameter."""

    name: str
    display_name: str
    param_type: str  # 'int', 'float', 'bool', 'select'
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    description: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None


class EnhancedUIComponents:
    """Enhanced UI components for the trading system."""

    def __init__(self):
        """Initialize enhanced UI components."""
        self.session_state = st.session_state
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables."""
        if "ui_session_start" not in self.session_state:
            self.session_state.ui_session_start = datetime.now()

        if "ui_selected_ticker" not in self.session_state:
            self.session_state.ui_selected_ticker = None

        if "ui_date_range" not in self.session_state:
            self.session_state.ui_date_range = None

        if "ui_active_strategy" not in self.session_state:
            self.session_state.ui_active_strategy = None

        if "ui_last_run" not in self.session_state:
            self.session_state.ui_last_run = None

        if "ui_logs" not in self.session_state:
            self.session_state.ui_logs = []

    def render_dynamic_sliders(
        self, parameter_configs: List[ParameterConfig], title: str = "Parameters"
    ) -> Dict[str, Any]:
        """
        Render dynamic sliders for model and strategy parameters.

        Args:
            parameter_configs: List of parameter configurations
            title: Title for the parameter section

        Returns:
            Dictionary of parameter values
        """
        st.subheader(f"🎛️ {title}")

        parameters = {}

        # Create columns for better layout
        num_columns = 2
        cols = st.columns(num_columns)

        for i, config in enumerate(parameter_configs):
            col_idx = i % num_columns

            with cols[col_idx]:
                try:
                    if config.param_type == "int":
                        value = st.slider(
                            config.display_name,
                            min_value=int(config.min_value) if config.min_value is not None else 0,
                            max_value=int(config.max_value) if config.max_value is not None else 100,
                            value=int(config.default_value),
                            step=int(config.step) if config.step else 1,
                            help=config.description,
                        )

                        # Validation
                        if config.validation_rules:
                            if "min" in config.validation_rules and value < config.validation_rules["min"]:
                                st.error(f"{config.display_name} must be at least {config.validation_rules['min']}")
                            elif "max" in config.validation_rules and value > config.validation_rules["max"]:
                                st.error(f"{config.display_name} must be at most {config.validation_rules['max']}")

                    elif config.param_type == "float":
                        value = st.slider(
                            config.display_name,
                            min_value=float(config.min_value) if config.min_value is not None else 0.0,
                            max_value=float(config.max_value) if config.max_value is not None else 1.0,
                            value=float(config.default_value),
                            step=float(config.step) if config.step else 0.01,
                            help=config.description,
                        )

                        # Validation
                        if config.validation_rules:
                            if "min" in config.validation_rules and value < config.validation_rules["min"]:
                                st.error(f"{config.display_name} must be at least {config.validation_rules['min']}")
                            elif "max" in config.validation_rules and value > config.validation_rules["max"]:
                                st.error(f"{config.display_name} must be at most {config.validation_rules['max']}")

                    elif config.param_type == "bool":
                        value = st.checkbox(
                            config.display_name, value=bool(config.default_value), help=config.description
                        )

                    elif config.param_type == "select":
                        if config.options:
                            value = st.selectbox(
                                config.display_name,
                                options=config.options,
                                index=config.options.index(config.default_value)
                                if config.default_value in config.options
                                else 0,
                                help=config.description,
                            )
                        else:
                            st.warning(f"No options provided for {config.display_name}")
                            value = config.default_value

                    else:
                        st.warning(f"Unknown parameter type: {config.param_type}")
                        value = config.default_value

                    parameters[config.name] = value

                except Exception as e:
                    st.error(f"Error rendering parameter {config.display_name}: {e}")
                    parameters[config.name] = config.default_value

        return parameters

    def render_confidence_score_display(
        self, confidence_data: Dict[str, Any], title: str = "Model Confidence Analysis"
    ):
        """
        Display confidence score and model selection breakdown.

        Args:
            confidence_data: Dictionary containing confidence information
            title: Title for the confidence display
        """
        st.subheader(f"🎯 {title}")

        # Overall confidence score
        overall_confidence = confidence_data.get("overall_confidence", 0.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Confidence",
                f"{overall_confidence:.1%}",
                delta=f"{overall_confidence - 0.5:.1%}"
                if overall_confidence > 0.5
                else f"{overall_confidence - 0.5:.1%}",
            )

        with col2:
            selected_model = confidence_data.get("selected_model", "Unknown")
            st.metric("Selected Model", selected_model)

        with col3:
            model_count = confidence_data.get("model_count", 0)
            st.metric("Available Models", model_count)

        # Model breakdown
        if "model_breakdown" in confidence_data:
            st.subheader("Model Selection Breakdown")

            breakdown_data = confidence_data["model_breakdown"]

            # Create a DataFrame for better display
            if isinstance(breakdown_data, dict):
                df = pd.DataFrame(
                    [
                        {
                            "Model": model,
                            "Confidence": f"{data.get('confidence', 0):.1%}",
                            "Score": data.get("score", 0),
                            "Reason": data.get("reason", "N/A"),
                        }
                        for model, data in breakdown_data.items()
                    ]
                )

                # Sort by confidence
                df["Confidence_Value"] = df["Confidence"].str.rstrip("%").astype(float) / 100
                df = df.sort_values("Confidence_Value", ascending=False)
                df = df.drop("Confidence_Value", axis=1)

                st.dataframe(df, use_container_width=True)

                # Confidence visualization
                st.subheader("Confidence Distribution")

                confidence_values = [data.get("confidence", 0) for data in breakdown_data.values()]
                model_names = list(breakdown_data.keys())

                # Create bar chart
                chart_data = pd.DataFrame({"Model": model_names, "Confidence": confidence_values})

                st.bar_chart(chart_data.set_index("Model"))

        # Strategy breakdown if available
        if "strategy_breakdown" in confidence_data:
            st.subheader("Strategy Confidence Breakdown")

            strategy_data = confidence_data["strategy_breakdown"]

            if isinstance(strategy_data, dict):
                strategy_df = pd.DataFrame(
                    [
                        {
                            "Strategy": strategy,
                            "Confidence": f"{data.get('confidence', 0):.1%}",
                            "Performance": f"{data.get('performance', 0):.1%}",
                            "Regime Fit": data.get("regime_fit", "N/A"),
                        }
                        for strategy, data in strategy_data.items()
                    ]
                )

                st.dataframe(strategy_df, use_container_width=True)

    def render_session_summary_bar(self):
        """Render session summary bar showing current session information."""
        st.markdown("---")

        # Session summary container
        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                # Session duration
                session_duration = datetime.now() - self.session_state.ui_session_start
                st.metric("Session Duration", f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s")

            with col2:
                # Selected ticker
                ticker = self.session_state.ui_selected_ticker or "None"
                st.metric("Selected Ticker", ticker)

            with col3:
                # Date range
                date_range = self.session_state.ui_date_range or "Not set"
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    date_range = f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)

            with col4:
                # Active strategy
                strategy = self.session_state.ui_active_strategy or "None"
                st.metric("Active Strategy", strategy)

            with col5:
                # Last run
                last_run = self.session_state.ui_last_run
                if last_run:
                    time_since = datetime.now() - last_run
                    if time_since.seconds < 60:
                        last_run_text = f"{time_since.seconds}s ago"
                    elif time_since.seconds < 3600:
                        last_run_text = f"{time_since.seconds // 60}m ago"
                    else:
                        last_run_text = f"{time_since.seconds // 3600}h ago"
                else:
                    last_run_text = "Never"

                st.metric("Last Run", last_run_text)

        st.markdown("---")

    def render_log_viewer(
        self, logs: Optional[List[Dict[str, Any]]] = None, max_logs: int = 100, title: str = "System Logs"
    ):
        """
        Render log viewer with expandable error details.

        Args:
            logs: List of log entries
            max_logs: Maximum number of logs to display
            title: Title for the log viewer
        """
        st.subheader(f"📋 {title}")

        # Use provided logs or session state logs
        if logs is None:
            logs = self.session_state.ui_logs

        if not logs:
            st.info("No logs available.")
            return

        # Filter options
        col1, col2, col3 = st.columns(3)

        with col1:
            log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR", "DEBUG"], index=0)

        with col2:
            search_term = st.text_input("Search Logs", "")

        with col3:
            auto_refresh = st.checkbox("Auto Refresh", value=True)

        # Filter logs
        filtered_logs = logs

        if log_level != "All":
            filtered_logs = [log for log in filtered_logs if log.get("level", "INFO") == log_level]

        if search_term:
            filtered_logs = [log for log in filtered_logs if search_term.lower() in log.get("message", "").lower()]

        # Limit number of logs
        filtered_logs = filtered_logs[-max_logs:]

        # Display logs
        if filtered_logs:
            for log in reversed(filtered_logs):  # Show newest first
                timestamp = log.get("timestamp", "Unknown")
                level = log.get("level", "INFO")
                message = log.get("message", "No message")
                details = log.get("details", {})

                # Color coding based on level
                if level == "ERROR":
                    st.error(f"**{timestamp}** [{level}] {message}")
                elif level == "WARNING":
                    st.warning(f"**{timestamp}** [{level}] {message}")
                elif level == "DEBUG":
                    st.info(f"**{timestamp}** [{level}] {message}")
                else:
                    st.write(f"**{timestamp}** [{level}] {message}")

                # Expandable details
                if details:
                    with st.expander("View Details"):
                        st.json(details)

                st.markdown("---")
        else:
            st.info("No logs match the current filters.")

        # Clear logs button
        if st.button("Clear Logs"):
            self.session_state.ui_logs = []
            st.success("Logs cleared!")
            st.rerun()

    def render_error_viewer(self, errors: Optional[List[Dict[str, Any]]] = None, title: str = "Error Viewer"):
        """
        Render expandable error viewer.

        Args:
            errors: List of error entries
            title: Title for the error viewer
        """
        st.subheader(f"🚨 {title}")

        if errors is None:
            # Get errors from logs
            errors = [log for log in self.session_state.ui_logs if log.get("level") == "ERROR"]

        if not errors:
            st.success("No errors found!")
            return

        # Group errors by type
        error_groups = {}
        for error in errors:
            error_type = error.get("error_type", "Unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(error)

        # Display grouped errors
        for error_type, error_list in error_groups.items():
            with st.expander(f"{error_type} ({len(error_list)} errors)", expanded=True):
                for error in error_list:
                    timestamp = error.get("timestamp", "Unknown")
                    message = error.get("message", "No message")
                    traceback = error.get("traceback", "")

                    st.error(f"**{timestamp}**: {message}")

                    if traceback:
                        with st.expander("View Traceback"):
                            st.code(traceback, language="python")

                    st.markdown("---")

    def update_session_state(
        self,
        ticker: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        strategy: Optional[str] = None,
    ):
        """
        Update session state with current selections.

        Args:
            ticker: Selected ticker
            date_range: Selected date range
            strategy: Active strategy
        """
        if ticker is not None:
            self.session_state.ui_selected_ticker = ticker

        if date_range is not None:
            self.session_state.ui_date_range = date_range

        if strategy is not None:
            self.session_state.ui_active_strategy = strategy

        # Update last run
        self.session_state.ui_last_run = datetime.now()

    def add_log_entry(self, message: str, level: str = "INFO", details: Optional[Dict[str, Any]] = None):
        """
        Add a log entry to the session state.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            details: Additional details
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "details": details or {},
        }

        self.session_state.ui_logs.append(log_entry)

        # Keep only recent logs
        if len(self.session_state.ui_logs) > 1000:
            self.session_state.ui_logs = self.session_state.ui_logs[-1000:]

    def render_parameter_validation_summary(
        self, parameters: Dict[str, Any], validation_rules: Dict[str, Dict[str, Any]]
    ):
        """
        Render parameter validation summary.

        Args:
            parameters: Dictionary of parameter values
            validation_rules: Dictionary of validation rules
        """
        st.subheader("✅ Parameter Validation")

        validation_results = []

        for param_name, value in parameters.items():
            if param_name in validation_rules:
                rules = validation_rules[param_name]
                is_valid = True
                issues = []

                # Check min/max values
                if "min" in rules and value < rules["min"]:
                    is_valid = False
                    issues.append(f"Below minimum ({rules['min']})")

                if "max" in rules and value > rules["max"]:
                    is_valid = False
                    issues.append(f"Above maximum ({rules['max']})")

                # Check required values
                if "required" in rules and rules["required"] and value is None:
                    is_valid = False
                    issues.append("Required but not set")

                validation_results.append(
                    {"parameter": param_name, "value": value, "valid": is_valid, "issues": issues}
                )

        # Display validation results
        if validation_results:
            for result in validation_results:
                if result["valid"]:
                    st.success(f"✅ {result['parameter']}: {result['value']}")
                else:
                    st.error(f"❌ {result['parameter']}: {result['value']} - {', '.join(result['issues'])}")
        else:
            st.info("No validation rules defined.")

    def render_performance_metrics(self, metrics: Dict[str, Any], title: str = "Performance Metrics"):
        """
        Render performance metrics with visual indicators.

        Args:
            metrics: Dictionary of performance metrics
            title: Title for the metrics display
        """
        st.subheader(f"📊 {title}")

        if not metrics:
            st.info("No performance metrics available.")
            return

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "sharpe_ratio" in metrics:
                sharpe = metrics["sharpe_ratio"]
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    delta=f"{sharpe - 1.0:.2f}" if sharpe > 1.0 else f"{sharpe - 1.0:.2f}",
                )

        with col2:
            if "total_return" in metrics:
                total_return = metrics["total_return"]
                st.metric("Total Return", f"{total_return:.1%}", delta=f"{total_return:.1%}")

        with col3:
            if "max_drawdown" in metrics:
                max_dd = metrics["max_drawdown"]
                st.metric("Max Drawdown", f"{max_dd:.1%}", delta=f"{max_dd:.1%}")

        with col4:
            if "win_rate" in metrics:
                win_rate = metrics["win_rate"]
                st.metric(
                    "Win Rate",
                    f"{win_rate:.1%}",
                    delta=f"{win_rate - 0.5:.1%}" if win_rate > 0.5 else f"{win_rate - 0.5:.1%}",
                )

        # Detailed metrics table
        if len(metrics) > 4:
            st.subheader("Detailed Metrics")

            metrics_df = pd.DataFrame([{"Metric": key, "Value": value} for key, value in metrics.items()])

            st.dataframe(metrics_df, use_container_width=True)
