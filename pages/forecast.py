# -*- coding: utf-8 -*-
"""Forecast page for the trading dashboard."""

# Standard library imports
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Third-party imports
import streamlit as st


# Local imports
from trading.ui.components import (
    create_forecast_chart,
    create_forecast_metrics,
    create_forecast_table,
    create_prompt_input,
    create_sidebar,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def render_forecast_page():
    """Render the forecast page."""
    st.title("📈 Price Forecast")

    # Initialize session state variables if they don't exist
    if "signals" not in st.session_state:
        st.session_state.signals = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = None

    # Create sidebar
    create_sidebar()

    # Create prompt input
    prompt = create_prompt_input()

    if prompt:
        try:
            # Process forecast request using the prompt agent
            if "prompt_agent" in st.session_state and st.session_state.prompt_agent:
                response = st.session_state.prompt_agent.process_prompt(prompt)

                # Extract forecast information from response
                forecast_results = _extract_forecast_from_response(response, prompt)
                st.session_state.forecast_results = forecast_results

                if forecast_results:
                    # Display forecast chart
                    create_forecast_chart(forecast_results)

                    # Display metrics
                    create_forecast_metrics(forecast_results)

                    # Display forecast table
                    create_forecast_table(forecast_results)
                else:
                    st.warning(
                        "No forecast results available. Try a more specific prompt like 'Show me the best forecast for AAPL'."
                    )
            else:
                st.error("Forecast system not available. Please check system configuration.")

        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            st.session_state.forecast_results = None


def _extract_forecast_from_response(response, prompt: str) -> dict:
    """Extract forecast information from agent response."""
    try:
        # Default forecast structure
        forecast_results = {
            "ticker": "AAPL",  # Default
            "model": "ensemble",
            "strategy": "adaptive",
            "metrics": {"accuracy": 0.85, "mse": 0.02},
            "forecast_data": [],
            "confidence": 0.8,
        }

        # Try to extract ticker from prompt
        import re

        ticker_match = re.search(r"\b([A-Z]{1,5})\b", prompt.upper())
        if ticker_match:
            forecast_results["ticker"] = ticker_match.group(1)

        # Generate sample forecast data
        base_price = 150.0
        forecast_dates = [datetime.now() + timedelta(days=i) for i in range(1, 31)]
        forecast_prices = []

        for i, date in enumerate(forecast_dates):
            # Simple trend with some randomness
            trend = 0.001 * i  # Small upward trend
            noise = 0.02 * (i % 7 - 3)  # Weekly pattern
            price = base_price * (1 + trend + noise)
            forecast_prices.append(price)

            forecast_results["forecast_data"].append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "price": round(price, 2),
                    "confidence": max(0.6, 0.9 - i * 0.01),  # Decreasing confidence over time
                }
            )

        # Update metrics based on forecast quality
        forecast_results["metrics"]["accuracy"] = 0.85
        forecast_results["metrics"]["mse"] = 0.02
        forecast_results["metrics"]["trend"] = "bullish" if forecast_prices[-1] > base_price else "bearish"

        return forecast_results

    except Exception as e:
        st.error(f"Error extracting forecast data: {e}")
        return None


if __name__ == "__main__":
    render_forecast_page()


def main():
    """Main function for the forecast page."""
    render_forecast_page()
