"""LLM-powered summary generation for trading analysis."""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class SummaryConfig:
    """Configuration for LLM summary generation."""

    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


def generate_market_summary(data: pd.DataFrame, config: SummaryConfig) -> str:
    """Generate market summary using LLM.

    Args:
        data: DataFrame containing market data
        config: Summary generation configuration

    Returns:
        String containing market summary
    """
    # Extract key statistics
    stats = {
        "current_price": data["close"].iloc[-1],
        "price_change": data["close"].pct_change().iloc[-1],
        "volume": data["volume"].iloc[-1],
        "volatility": data["close"].pct_change().std() * np.sqrt(252),
        "trend": "up" if data["close"].iloc[-1] > data["close"].iloc[-20] else "down",
    }

    # Generate summary prompt
    prompt = f"""Analyze the following market data and provide a concise summary:
    Current Price: ${stats['current_price']:.2f}
    Price Change: {stats['price_change']:.2%}
    Volume: {stats['volume']:,.0f}
    Volatility: {stats['volatility']:.2%}
    Trend: {stats['trend'].upper()}

    Please provide a brief analysis focusing on:
    1. Key price movements and trends
    2. Volume analysis
    3. Volatility assessment
    4. Notable patterns or events
    """

    # Call LLM API (placeholder)
    summary = _call_llm_api(prompt, config)

    return summary


def generate_strategy_summary(
    performance_data: Dict[str, Any], config: SummaryConfig
) -> str:
    """Generate strategy performance summary using LLM.

    Args:
        performance_data: Dictionary containing strategy performance data
        config: Summary generation configuration

    Returns:
        String containing strategy summary
    """
    # Extract key metrics
    metrics = performance_data["metrics"]
    trades = performance_data["trades"]

    # Generate summary prompt
    prompt = f"""Analyze the following trading strategy performance and provide a concise summary:
    Total Return: {metrics['total_return']:.2%}
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Max Drawdown: {metrics['max_drawdown']:.2%}
    Win Rate: {metrics['win_rate']:.2%}
    Total Trades: {len(trades)}

    Please provide a brief analysis focusing on:
    1. Overall performance assessment
    2. Risk-adjusted returns
    3. Trade execution quality
    4. Areas for improvement
    """

    # Call LLM API (placeholder)
    summary = _call_llm_api(prompt, config)

    return summary


def generate_forecast_summary(
    forecast_data: Dict[str, Any], config: SummaryConfig
) -> str:
    """Generate forecast analysis summary using LLM.

    Args:
        forecast_data: Dictionary containing forecast data
        config: Summary generation configuration

    Returns:
        String containing forecast summary
    """
    # Extract key metrics
    metrics = forecast_data["metrics"]
    forecast = forecast_data["forecast"]

    # Generate summary prompt
    prompt = f"""Analyze the following price forecast and provide a concise summary:
    Forecast Period: {len(forecast)} days
    Forecast Accuracy: {metrics['accuracy']:.2%}
    Confidence Level: {metrics['confidence']:.2%}
    Mean Squared Error: {metrics['mse']:.4f}

    Please provide a brief analysis focusing on:
    1. Forecast reliability
    2. Key price predictions
    3. Confidence intervals
    4. Potential risks
    """

    # Call LLM API (placeholder)
    summary = _call_llm_api(prompt, config)

    return summary


def generate_risk_summary(risk_data: Dict[str, Any], config: SummaryConfig) -> str:
    """Generate risk analysis summary using LLM.

    Args:
        risk_data: Dictionary containing risk analysis data
        config: Summary generation configuration

    Returns:
        String containing risk summary
    """
    # Extract key metrics
    metrics = risk_data["metrics"]
    risk_data["drawdowns"]

    # Generate summary prompt
    prompt = f"""Analyze the following risk metrics and provide a concise summary:
    Volatility: {metrics['volatility']:.2%}
    Value at Risk (95%): {metrics['var_95']:.2%}
    Conditional VaR (95%): {metrics['cvar_95']:.2%}
    Beta: {metrics['beta']:.2f}
    Correlation: {metrics['correlation']:.2f}

    Please provide a brief analysis focusing on:
    1. Overall risk assessment
    2. Key risk factors
    3. Risk-adjusted performance
    4. Risk management recommendations
    """

    # Call LLM API (placeholder)
    summary = _call_llm_api(prompt, config)

    return summary


def _call_llm_api(prompt: str, config: SummaryConfig) -> str:
    """Call LLM API to generate summary.

    Args:
        prompt: Input prompt for LLM
        config: Summary generation configuration

    Returns:
        Generated summary text
    """
    try:
        import os

        import openai

        # Get API key from environment or config
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to mock response if no API key
            return _generate_mock_summary(prompt)

        # Configure OpenAI client
        client = openai.OpenAI(api_key=api_key)

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst providing concise, professional analysis of trading data.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )

        # Extract and return the generated text
        summary = response.choices[0].message.content.strip()
        return summary

    except ImportError:
        # Fallback if OpenAI library not available
        return _generate_mock_summary(prompt)
    except Exception as e:
        # Log error and return fallback
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error calling LLM API: {e}")
        return _generate_mock_summary(prompt)


def _generate_mock_summary(prompt: str) -> str:
    """Generate a mock summary when LLM API is not available.

    Args:
        prompt: Input prompt for LLM

    Returns:
        Mock summary text
    """
    # Simple rule-based summary generation
    if "market" in prompt.lower():
        return "Market analysis shows moderate volatility with upward trend. Volume patterns indicate healthy participation. Key support levels identified at recent lows."
    elif "strategy" in prompt.lower():
        return "Strategy performance demonstrates positive risk-adjusted returns. Trade execution quality is satisfactory with room for optimization in entry/exit timing."
    elif "forecast" in prompt.lower():
        return "Forecast indicates moderate confidence in price predictions. Key resistance levels identified. Risk factors include market volatility and economic uncertainty."
    elif "risk" in prompt.lower():
        return "Risk assessment shows manageable exposure levels. Portfolio diversification provides adequate protection. Monitor correlation changes closely."
    else:
        return "Analysis completed successfully. Key metrics indicate stable performance with moderate risk levels."


def format_summary(summary: str, max_length: int = 500) -> str:
    """Format summary text for display.

    Args:
        summary: Raw summary text
        max_length: Maximum length of formatted summary

    Returns:
        Formatted summary text
    """
    # Split into sentences
    sentences = summary.split(". ")

    # Format each sentence
    formatted_sentences = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > max_length:
            break

        formatted_sentences.append(sentence.strip())
        current_length += len(sentence) + 2

    # Join sentences
    formatted_summary = ". ".join(formatted_sentences) + "."

    return formatted_summary


def generate_comprehensive_summary(
    market_data: pd.DataFrame,
    performance_data: Dict[str, Any],
    forecast_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    config: SummaryConfig,
) -> Dict[str, str]:
    """Generate comprehensive analysis summary.

    Args:
        market_data: DataFrame containing market data
        performance_data: Dictionary containing strategy performance data
        forecast_data: Dictionary containing forecast data
        risk_data: Dictionary containing risk analysis data
        config: Summary generation configuration

    Returns:
        Dictionary containing different types of summaries
    """
    # Generate individual summaries
    market_summary = generate_market_summary(market_data, config)
    strategy_summary = generate_strategy_summary(performance_data, config)
    forecast_summary = generate_forecast_summary(forecast_data, config)
    risk_summary = generate_risk_summary(risk_data, config)

    # Format summaries
    summaries = {
        "market": format_summary(market_summary),
        "strategy": format_summary(strategy_summary),
        "forecast": format_summary(forecast_summary),
        "risk": format_summary(risk_summary),
    }

    return summaries
