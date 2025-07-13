"""
Agent Leaderboard Dashboard

Streamlit dashboard for visualizing and managing agent performance leaderboard.
Provides interactive charts, filtering, and agent management capabilities.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from trading.agents.agent_leaderboard import AgentLeaderboard
from trading.agents.agent_manager import AgentManager, AgentManagerConfig


class LeaderboardDashboard:
    """Streamlit dashboard for agent leaderboard visualization and management."""

    def __init__(self):
        """Initialize the dashboard."""
        self.manager = AgentManager()
        self.leaderboard = self.manager.leaderboard

        # Page configuration
        st.set_page_config(
            page_title="Agent Leaderboard",
            page_icon="🏆",
            layout="wide",
            initial_sidebar_state="expanded"
        )def run(self):
        """Run the dashboard."""
        st.title("🏆 Agent Performance Leaderboard")
        st.markdown("Track, analyze, and manage agent performance across the trading system.")

        # Sidebar controls
        self._render_sidebar()

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            self._render_leaderboard_table()
            self._render_performance_charts()

        with col2:
            self._render_summary_metrics()
            self._render_agent_status()
            self._render_deprecation_controls()

        # Bottom section
        self._render_performance_history()
        self._render_export_options()

    def _render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("📊 Dashboard Controls")

        # Filter options
        st.sidebar.subheader("Filters")

        # Status filter
        status_filter = st.sidebar.selectbox(
            "Agent Status",
            ["All", "Active", "Deprecated"],
            help="Filter agents by their current status"
        )

        # Sort options
        sort_by = st.sidebar.selectbox(
            "Sort By",
            ["sharpe_ratio", "total_return", "win_rate", "max_drawdown"],
            format_func=lambda x: x.replace("_", " ").title(),
            help="Choose metric to sort the leaderboard"
        )

        # Top N display
        top_n = st.sidebar.slider(
            "Show Top N Agents",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of top agents to display"
        )

        # Store filters in session state
        st.session_state.status_filter = status_filter
        st.session_state.sort_by = sort_by
        st.session_state.top_n = top_n

        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()

        # Add sample data button
        if st.sidebar.button("📈 Add Sample Data"):
            self._add_sample_data()

    def _render_leaderboard_table(self):
        """Render the main leaderboard table."""
        st.subheader("📋 Agent Leaderboard")

        # Get filtered data
        df = self._get_filtered_dataframe()

        if df.empty:
            st.info("No agents found matching the current filters.")
            return

        # Format the dataframe for display
        display_df = df.copy()
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].round(2)
        display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.2%}")
        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")

        # Add color coding for status
        def color_status(val):
            if val == "active":
                return "background-color: #d4edda"
            else:
                return {'success': True, 'result': {'success': True, 'result': "background-color: #f8d7da", 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

        styled_df = display_df.style.applymap(color_status, subset=['status'])

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    def _render_performance_charts(self):
        """Render performance visualization charts."""
        st.subheader("📈 Performance Analysis")

        df = self._get_filtered_dataframe()
        if df.empty:

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Sharpe Ratio Distribution", "Return vs Risk",
                          "Win Rate vs Drawdown", "Performance Heatmap"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )

        # 1. Sharpe Ratio Distribution
        fig.add_trace(
            go.Histogram(x=df['sharpe_ratio'], name="Sharpe Ratio", nbinsx=10),
            row=1, col=1
        )

        # 2. Return vs Risk (Return vs Drawdown)
        fig.add_trace(
            go.Scatter(
                x=df['max_drawdown'],
                y=df['total_return'],
                mode='markers+text',
                text=df['agent_name'],
                textposition="top center",
                name="Return vs Risk"
            ),
            row=1, col=2
        )

        # 3. Win Rate vs Drawdown
        fig.add_trace(
            go.Scatter(
                x=df['max_drawdown'],
                y=df['win_rate'],
                mode='markers+text',
                text=df['agent_name'],
                textposition="top center",
                name="Win Rate vs Risk"
            ),
            row=2, col=1
        )

        # 4. Performance Heatmap
        metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'max_drawdown']
        heatmap_data = df[metrics].values

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=metrics,
                y=df['agent_name'],
                colorscale='RdYlGn',
                name="Performance Heatmap"
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Agent Performance Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_summary_metrics(self):
        """Render summary metrics cards."""
        st.subheader("📊 Summary Metrics")

        df = self._get_filtered_dataframe()
        if df.empty:

        # Calculate summary metrics
        total_agents = len(df)
        active_agents = len(df[df['status'] == 'active'])
        deprecated_agents = len(df[df['status'] == 'deprecated'])

        avg_sharpe = df['sharpe_ratio'].mean()
        avg_return = df['total_return'].mean()
        avg_win_rate = df['win_rate'].mean()
        avg_drawdown = df['max_drawdown'].mean()

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Agents", total_agents)
            st.metric("Active Agents", active_agents)
            st.metric("Avg Sharpe", f"{avg_sharpe:.2f}")
            st.metric("Avg Return", f"{avg_return:.2%}")

        with col2:
            st.metric("Deprecated", deprecated_agents)
            st.metric("Success Rate", f"{active_agents/total_agents:.1%}" if total_agents > 0 else "0%")
            st.metric("Avg Win Rate", f"{avg_win_rate:.2%}")
            st.metric("Avg Drawdown", f"{avg_drawdown:.2%}")

    def _render_agent_status(self):
        """Render agent status breakdown."""
        st.subheader("🔍 Agent Status")

        df = self._get_filtered_dataframe()
        if df.empty:

        # Status breakdown
        status_counts = df['status'].value_counts()

        # Pie chart
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Agent Status Distribution",
            color_discrete_map={'active': '#28a745', 'deprecated': '#dc3545'}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Status details
        for status, count in status_counts.items():
            agents = df[df['status'] == status]['agent_name'].tolist()
            with st.expander(f"{status.title()} Agents ({count})"):
                for agent in agents:
                    st.write(f"• {agent}")

    def _render_deprecation_controls(self):
        """Render deprecation management controls."""
        st.subheader("⚙️ Deprecation Controls")

        # Show current thresholds
        thresholds = self.leaderboard.deprecation_thresholds

        st.write("**Current Deprecation Thresholds:**")
        st.write(f"• Sharpe Ratio < {thresholds['sharpe_ratio']}")
        st.write(f"• Max Drawdown > {thresholds['max_drawdown']:.1%}")
        st.write(f"• Win Rate < {thresholds['win_rate']:.1%}")

        # Manual deprecation
        st.write("**Manual Deprecation:**")
        active_agents = self.leaderboard.get_active_agents()

        if active_agents:
            agent_to_deprecate = st.selectbox(
                "Select agent to deprecate:",
                active_agents
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚫 Deprecate Agent"):
                    self._manually_deprecate_agent(agent_to_deprecate)
                    st.success(f"Agent '{agent_to_deprecate}' deprecated!")
                    st.rerun()

            with col2:
                if st.button("✅ Reactivate Agent"):
                    self._reactivate_agent(agent_to_deprecate)
                    st.success(f"Agent '{agent_to_deprecate}' reactivated!")
                    st.rerun()
        else:
            st.info("No active agents to manage.")

    def _render_performance_history(self):
        """Render performance history chart."""
        st.subheader("📅 Performance History")

        history = self.leaderboard.get_history(limit=50)
        if not history:
            st.info("No performance history available.")

        # Convert to DataFrame
        history_df = pd.DataFrame(history)
        history_df['last_updated'] = pd.to_datetime(history_df['last_updated'])

        # Plot performance over time
        fig = go.Figure()

        for agent in history_df['agent_name'].unique():
            agent_data = history_df[history_df['agent_name'] == agent]
            fig.add_trace(
                go.Scatter(
                    x=agent_data['last_updated'],
                    y=agent_data['sharpe_ratio'],
                    mode='lines+markers',
                    name=agent,
                    hovertemplate=f"{agent}<br>Sharpe: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>"
                )
            )

        fig.update_layout(
            title="Sharpe Ratio Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_export_options(self):
        """Render data export options."""
        st.subheader("💾 Export Options")

        df = self._get_filtered_dataframe()
        if df.empty:

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"agent_leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # JSON Export
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"agent_leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col3:
            # Summary Report
            if st.button("📊 Generate Report"):
                self._generate_summary_report(df)

    def _get_filtered_dataframe(self) -> pd.DataFrame:
        """Get filtered dataframe based on sidebar controls."""
        df = self.leaderboard.as_dataframe()

        if df.empty:
            return df

        # Apply status filter
        status_filter = st.session_state.get('status_filter', 'All')
        if status_filter != 'All':
            df = df[df['status'] == status_filter.lower()]

        # Sort by selected metric
        sort_by = st.session_state.get('sort_by', 'sharpe_ratio')
        df = df.sort_values(sort_by, ascending=False)

        # Limit to top N
        top_n = st.session_state.get('top_n', 10)
        df = df.head(top_n)

        return df

    def _add_sample_data(self):
        """Add sample performance data for demonstration."""
        import random

        sample_agents = [
            "model_builder_v1", "performance_critic_v2", "updater_v1",
            "execution_agent_v3", "optimizer_agent_v1", "research_agent_v2"
        ]

        for agent in sample_agents:
            # Generate realistic performance data
            sharpe = random.uniform(0.5, 3.0)
            drawdown = random.uniform(0.05, 0.30)
            win_rate = random.uniform(0.45, 0.80)
            total_return = random.uniform(0.10, 0.50)

            extra_metrics = {
                "volatility": random.uniform(0.15, 0.35),
                "calmar_ratio": sharpe / drawdown if drawdown > 0 else 0,
                "profit_factor": random.uniform(1.1, 3.0),
                "total_trades": random.randint(30, 150)
            }

            self.leaderboard.update_performance(
                agent_name=agent,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown,
                win_rate=win_rate,
                total_return=total_return,
                extra_metrics=extra_metrics
            )

        st.success("Sample data added successfully!")

    def _manually_deprecate_agent(self, agent_name: str):
        """Manually deprecate an agent."""
        if agent_name in self.leaderboard.leaderboard:
            self.leaderboard.leaderboard[agent_name].status = "deprecated"

    def _reactivate_agent(self, agent_name: str):
        """Reactivate a deprecated agent."""
        if agent_name in self.leaderboard.leaderboard:
            self.leaderboard.leaderboard[agent_name].status = "active"

    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate a summary report."""
        report = f"""

# Agent Performance Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- Total Agents: {len(df)}
- Active Agents: {len(df[df['status'] == 'active'])}
- Deprecated Agents: {len(df[df['status'] == 'deprecated'])}

## Performance Statistics
- Average Sharpe Ratio: {df['sharpe_ratio'].mean():.2f}
- Average Total Return: {df['total_return'].mean():.2%}
- Average Win Rate: {df['win_rate'].mean():.2%}
- Average Max Drawdown: {df['max_drawdown'].mean():.2%}

## Top Performers
{df.head(5)[['agent_name', 'sharpe_ratio', 'total_return', 'status']].to_string(index=False)}

## Recommendations
- Consider deprecating agents with Sharpe < 0.5
- Monitor agents with drawdown > 20%
- Focus on agents with win rate > 60%
        """

        st.text_area("📊 Summary Report", report, height=400)


def main():
    """Main function to run the dashboard."""
    dashboard = LeaderboardDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
