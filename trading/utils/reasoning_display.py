"""
Reasoning Display

Display components for showing agent reasoning logs in terminal and Streamlit UI.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from trading.utils.reasoning_logger import (
    AgentDecision,
    ConfidenceLevel,
    DecisionType,
    ReasoningLogger,
)


class ReasoningDisplay:
    """
    Display component for agent reasoning logs.

    Provides both terminal and Streamlit interfaces for viewing
    agent decisions and explanations.
    """

    def __init__(self, reasoning_logger: ReasoningLogger):
        """
        Initialize the ReasoningDisplay.

        Args:
            reasoning_logger: ReasoningLogger instance
        """
        self.logger = reasoning_logger

    def display_decision_terminal(self, decision: AgentDecision, show_explanation: bool = True):
        """
        Display a decision in the terminal.

        Args:
            decision: AgentDecision to display
            show_explanation: Whether to show the chat explanation
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"🤖 AGENT DECISION: {decision.agent_name.upper()}")
        logger.info("=" * 80)

        # Basic info
        logger.info(f"📅 Time: {decision.timestamp}")
        logger.info(f"🎯 Type: {decision.decision_type.value.replace('_', ' ').title()}")
        logger.info(f"📈 Symbol: {decision.context.symbol}")
        logger.info(f"⏱️  Timeframe: {decision.context.timeframe}")
        logger.info(f"🎯 Action: {decision.action_taken}")
        logger.info(f"🎯 Confidence: {decision.confidence_level.value.replace('_', ' ').title()}")

        # Reasoning
        logger.info(f"\n🧠 REASONING:")
        logger.info(f"Primary Reason: {decision.reasoning.primary_reason}")

        if decision.reasoning.supporting_factors:
            logger.info(f"\nSupporting Factors:")
            for factor in decision.reasoning.supporting_factors:
                logger.info(f"  • {factor}")

        if decision.reasoning.alternatives_considered:
            logger.info(f"\nAlternatives Considered:")
            for alt in decision.reasoning.alternatives_considered:
                logger.info(f"  • {alt}")

        if decision.reasoning.risks_assessed:
            logger.info(f"\nRisks Assessed:")
            for risk in decision.reasoning.risks_assessed:
                logger.info(f"  • {risk}")

        logger.info(f"\nExpected Outcome: {decision.reasoning.expected_outcome}")

        # Market conditions
        if decision.context.market_conditions:
            logger.info(f"\n📊 MARKET CONDITIONS:")
            for key, value in decision.context.market_conditions.items():
                logger.info(f"  {key}: {value}")

        # Chat explanation
        if show_explanation:
            explanation = self.logger.get_explanation(decision.decision_id)
            if explanation:
                logger.info(f"\n💬 CHAT EXPLANATION:")
                logger.info("-" * 40)
                logger.info(explanation)
                logger.info("-" * 40)

        logger.info("=" * 80 + "\n")

    def display_recent_decisions_terminal(self, limit: int = 10, agent_name: str = None):
        """
        Display recent decisions in the terminal.

        Args:
            limit: Number of decisions to show
            agent_name: Filter by specific agent
        """
        if agent_name:
            decisions = self.logger.get_agent_decisions(agent_name, limit=limit)
            logger.info(f"\n🤖 RECENT DECISIONS BY {agent_name.upper()}")
        else:
            # Get decisions from all agents
            decisions = []
            stats = self.logger.get_statistics()
            for decision_data in stats["recent_activity"][:limit]:
                decision = self.logger.get_decision(decision_data["decision_id"])
                if decision:
                    decisions.append(decision)
            logger.info(f"\n🤖 RECENT DECISIONS (ALL AGENTS)")

        logger.info("=" * 80)

        if not decisions:
            logger.info("No decisions found.")

        for i, decision in enumerate(decisions, 1):
            logger.info(f"\n{i}. {decision.agent_name} - {decision.decision_type.value}")
            logger.info(f"   📈 {decision.context.symbol} | {decision.action_taken}")
            logger.info(f"   🎯 {decision.reasoning.primary_reason}")
            logger.info(f"   ⏰ {decision.timestamp}")
            logger.info(f"   🎯 Confidence: {decision.confidence_level.value}")
            logger.info("-" * 40)

    def display_statistics_terminal(self):
        """Display reasoning statistics in the terminal."""
        stats = self.logger.get_statistics()

        logger.info("\n📊 REASONING STATISTICS")
        logger.info("=" * 50)

        logger.info(f"Total Decisions: {stats['total_decisions']}")

        logger.info(f"\nDecisions by Agent:")
        for agent, count in stats["decisions_by_agent"].items():
            logger.info(f"  {agent}: {count}")

        logger.info(f"\nDecisions by Type:")
        for decision_type, count in stats["decisions_by_type"].items():
            logger.info(f"  {decision_type}: {count}")

        logger.info(f"\nConfidence Distribution:")
        for confidence, count in stats["confidence_distribution"].items():
            logger.info(f"  {confidence}: {count}")

        logger.info(f"\nRecent Activity:")
        for activity in stats["recent_activity"][:5]:
            logger.info(f"  {activity['agent_name']} - {activity['decision_type']} - {activity['symbol']}")

    def display_decision_streamlit(self, decision: AgentDecision):
        """
        Display a decision in Streamlit.

        Args:
            decision: AgentDecision to display
        """
        # Header
        st.subheader(f"🤖 {decision.agent_name} Decision")

        # Basic info in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Symbol", decision.context.symbol)
            st.metric("Timeframe", decision.context.timeframe)

        with col2:
            st.metric("Type", decision.decision_type.value.replace("_", " ").title())
            st.metric("Confidence", decision.confidence_level.value.replace("_", " ").title())

        with col3:
            st.metric("Timestamp", decision.timestamp[:19])  # Remove microseconds

        # Action taken
        st.info(f"**Action:** {decision.action_taken}")

        # Reasoning
        st.subheader("🧠 Reasoning")

        st.write(f"**Primary Reason:** {decision.reasoning.primary_reason}")

        if decision.reasoning.supporting_factors:
            st.write("**Supporting Factors:**")
            for factor in decision.reasoning.supporting_factors:
                st.write(f"• {factor}")

        if decision.reasoning.alternatives_considered:
            st.write("**Alternatives Considered:**")
            for alt in decision.reasoning.alternatives_considered:
                st.write(f"• {alt}")

        if decision.reasoning.risks_assessed:
            st.write("**Risks Assessed:**")
            for risk in decision.reasoning.risks_assessed:
                st.write(f"• {risk}")

        st.write(f"**Expected Outcome:** {decision.reasoning.expected_outcome}")

        # Market conditions
        if decision.context.market_conditions:
            st.subheader("📊 Market Conditions")
            market_df = pd.DataFrame(list(decision.context.market_conditions.items()), columns=["Condition", "Value"])
            st.dataframe(market_df, use_container_width=True)

        # Chat explanation
        explanation = self.logger.get_explanation(decision.decision_id)
        if explanation:
            st.subheader("💬 Chat Explanation")
            st.text_area("Explanation", explanation, height=200, disabled=True)

    def display_recent_decisions_streamlit(self, limit: int = 10, agent_name: str = None):
        """
        Display recent decisions in Streamlit.

        Args:
            limit: Number of decisions to show
            agent_name: Filter by specific agent
        """
        if agent_name:
            decisions = self.logger.get_agent_decisions(agent_name, limit=limit)
            st.subheader(f"🤖 Recent Decisions by {agent_name}")
        else:
            # Get decisions from all agents
            decisions = []
            stats = self.logger.get_statistics()
            for decision_data in stats["recent_activity"][:limit]:
                decision = self.logger.get_decision(decision_data["decision_id"])
                if decision:
                    decisions.append(decision)
            st.subheader("🤖 Recent Decisions (All Agents)")

        if not decisions:
            st.info("No decisions found.")

        # Create dataframe for display
        decision_data = []
        for decision in decisions:
            decision_data.append(
                {
                    "Agent": decision.agent_name,
                    "Type": decision.decision_type.value.replace("_", " ").title(),
                    "Symbol": decision.context.symbol,
                    "Action": decision.action_taken[:50] + "..."
                    if len(decision.action_taken) > 50
                    else decision.action_taken,
                    "Confidence": decision.confidence_level.value.replace("_", " ").title(),
                    "Timestamp": decision.timestamp[:19],
                }
            )

        df = pd.DataFrame(decision_data)
        st.dataframe(df, use_container_width=True)

        # Allow user to click on a decision to see details
        if st.button("View Decision Details"):
            selected_decision = st.selectbox(
                "Select a decision to view details:",
                decisions,
                format_func=lambda x: f"{x.agent_name} - {x.decision_type.value} - {x.context.symbol}",
            )

            if selected_decision:
                self.display_decision_streamlit(selected_decision)

    def display_statistics_streamlit(self):
        """Display reasoning statistics in Streamlit."""
        stats = self.logger.get_statistics()

        st.subheader("📊 Reasoning Statistics")

        # Overall stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Decisions", stats["total_decisions"])

        with col2:
            st.metric("Active Agents", len(stats["decisions_by_agent"]))

        with col3:
            st.metric("Decision Types", len(stats["decisions_by_type"]))

        with col4:
            recent_count = len(stats["recent_activity"])
            st.metric("Recent Activity", recent_count)

        # Decisions by agent
        if stats["decisions_by_agent"]:
            st.subheader("Decisions by Agent")
            agent_df = pd.DataFrame(list(stats["decisions_by_agent"].items()), columns=["Agent", "Decisions"])
            st.bar_chart(agent_df.set_index("Agent"))

        # Decisions by type
        if stats["decisions_by_type"]:
            st.subheader("Decisions by Type")
            type_df = pd.DataFrame(list(stats["decisions_by_type"].items()), columns=["Type", "Count"])
            st.bar_chart(type_df.set_index("Type"))

        # Confidence distribution
        if stats["confidence_distribution"]:
            st.subheader("Confidence Distribution")
            conf_df = pd.DataFrame(list(stats["confidence_distribution"].items()), columns=["Confidence", "Count"])
            st.bar_chart(conf_df.set_index("Confidence"))

        # Recent activity
        if stats["recent_activity"]:
            st.subheader("Recent Activity")
            activity_df = pd.DataFrame(stats["recent_activity"])
            st.dataframe(activity_df, use_container_width=True)

    def create_streamlit_sidebar(self):
        """Create a sidebar for reasoning controls in Streamlit."""
        st.sidebar.subheader("🤖 Reasoning Controls")

        # Agent filter
        stats = self.logger.get_statistics()
        agent_names = list(stats["decisions_by_agent"].keys())

        if agent_names:
            selected_agent = st.sidebar.selectbox("Filter by Agent:", ["All Agents"] + agent_names)
        else:
            selected_agent = "All Agents"

        # Decision type filter
        decision_types = [dt.value for dt in DecisionType]
        selected_type = st.sidebar.selectbox("Filter by Type:", ["All Types"] + decision_types)

        # Confidence filter
        confidence_levels = [cl.value for cl in ConfidenceLevel]
        selected_confidence = st.sidebar.selectbox("Filter by Confidence:", ["All Levels"] + confidence_levels)

        # Limit
        limit = st.sidebar.slider("Number of decisions to show:", 5, 50, 10)

        # Refresh button
        if st.sidebar.button("🔄 Refresh"):
            st.rerun()

        return {
            "agent": selected_agent if selected_agent != "All Agents" else None,
            "type": selected_type if selected_type != "All Types" else None,
            "confidence": selected_confidence if selected_confidence != "All Levels" else None,
            "limit": limit,
        }

    def display_live_feed_streamlit(self):
        """Display a live feed of decisions in Streamlit."""
        st.subheader("🔴 Live Decision Feed")

        # Create a placeholder for live updates
        feed_placeholder = st.empty()

        # Get recent decisions
        stats = self.logger.get_statistics()
        recent_decisions = stats["recent_activity"][:10]

        if recent_decisions:
            feed_text = ""
            for decision_data in recent_decisions:
                decision = self.logger.get_decision(decision_data["decision_id"])
                if decision:
                    feed_text += f"**{decision.agent_name}** ({decision.timestamp[:19]})\n"
                    feed_text += f"📈 {decision.context.symbol} | {decision.action_taken}\n"
                    feed_text += f"🎯 {decision.reasoning.primary_reason}\n"
                    feed_text += "---\n"

            feed_placeholder.markdown(feed_text)
        else:
            feed_placeholder.info("No recent decisions to display.")

        # Auto-refresh every 30 seconds
        if st.button("🔄 Refresh Feed"):
            st.rerun()


def create_reasoning_page_streamlit():
    """Create a complete reasoning page in Streamlit."""
    st.title("🤖 Agent Reasoning Dashboard")

    # Initialize reasoning logger
    reasoning_logger = ReasoningLogger()
    display = ReasoningDisplay(reasoning_logger)

    # Sidebar controls
    filters = display.create_streamlit_sidebar()

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "📋 Recent Decisions", "🔴 Live Feed", "📄 Decision Details"])

    with tab1:
        display.display_statistics_streamlit()

    with tab2:
        display.display_recent_decisions_streamlit(limit=filters["limit"], agent_name=filters["agent"])

    with tab3:
        display.display_live_feed_streamlit()

    with tab4:
        st.subheader("📄 View Specific Decision")

        # Get all decisions for selection
        stats = reasoning_logger.get_statistics()
        all_decisions = []

        for decision_data in stats["recent_activity"]:
            decision = reasoning_logger.get_decision(decision_data["decision_id"])
            if decision:
                all_decisions.append(decision)

        if all_decisions:
            selected_decision = st.selectbox(
                "Select a decision to view:",
                all_decisions,
                format_func=lambda x: f"{x.agent_name} - {x.decision_type.value} - {x.context.symbol} - {x.timestamp[:19]}",
            )

            if selected_decision:
                display.display_decision_streamlit(selected_decision)
        else:
            st.info("No decisions available to view.")


if __name__ == "__main__":
    # Test the display components
    reasoning_logger = ReasoningLogger()
    display = ReasoningDisplay(reasoning_logger)

    # Display statistics
    display.display_statistics_terminal()

    # Display recent decisions
    display.display_recent_decisions_terminal(limit=5)
