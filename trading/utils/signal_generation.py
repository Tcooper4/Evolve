"""Signal generation utilities for trading strategies."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    lookback_period: int = 14
    threshold: float = 0.7
    smoothing: int = 3
    min_volume: float = 1000
    max_spread: float = 0.02
    confidence_threshold: float = 0.6  # Minimum confidence for valid signals
    validate_completeness: bool = True  # Whether to validate signal completeness


@dataclass
class SignalResult:
    """Result of signal generation with validation and confidence."""

    signals: pd.Series
    confidence: pd.Series
    is_valid: bool
    validation_errors: List[str]
    completeness_score: float
    timestamp: datetime


def validate_market_data(data: pd.DataFrame) -> List[str]:
    """Validate market data for signal generation.

    Args:
        data: DataFrame containing market data

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    required_columns = ["open", "high", "low", "close", "volume"]

    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Check for null values
    if data[required_columns].isnull().any().any():
        errors.append("Market data contains null values")

    # Check for negative values
    if (data[["open", "high", "low", "close", "volume"]] < 0).any().any():
        errors.append("Market data contains negative values")

    # Check for high-low consistency
    if not (data["high"] >= data["low"]).all():
        errors.append("High prices must be greater than or equal to low prices")

    # Check for sufficient data points
    if len(data) < 50:
        errors.append("Insufficient data points for reliable signal generation")

    return errors


def validate_signals(
    signals: pd.Series, data: pd.DataFrame, config: SignalConfig
) -> Tuple[bool, List[str], float]:
    """
    Validate generated signals for completeness and quality.

    Args:
        signals: Series containing trading signals
        data: DataFrame containing market data
        config: Signal generation configuration

    Returns:
        Tuple of (is_valid, validation_errors, completeness_score)
    """
    errors = []

    # Check for NaN values
    if signals.isnull().any():
        errors.append(f"Signals contain {signals.isnull().sum()} NaN values")

    # Check for infinite values
    if np.isinf(signals).any():
        errors.append("Signals contain infinite values")

    # Check signal completeness (non-zero signals)
    total_signals = len(signals)
    non_zero_signals = (signals != 0).sum()
    completeness_score = non_zero_signals / total_signals if total_signals > 0 else 0.0

    if config.validate_completeness and completeness_score < 0.01:
        errors.append(
            f"Signal completeness too low: {completeness_score:.3f} (minimum 0.01)"
        )

    # Check for signal clustering (too many consecutive signals)
    consecutive_signals = 0
    max_consecutive = 0
    for signal in signals:
        if signal != 0:
            consecutive_signals += 1
            max_consecutive = max(max_consecutive, consecutive_signals)
        else:
            consecutive_signals = 0

    if max_consecutive > len(signals) * 0.3:  # More than 30% consecutive signals
        errors.append(f"Too many consecutive signals: {max_consecutive}")

    # Check signal magnitude consistency
    unique_signals = signals.unique()
    if len(unique_signals) > 10:  # Too many unique signal values
        errors.append(f"Too many unique signal values: {len(unique_signals)}")

    is_valid = len(errors) == 0
    return is_valid, errors, completeness_score


def calculate_signal_confidence(
    signals: pd.Series, data: pd.DataFrame, config: SignalConfig
) -> pd.Series:
    """
    Calculate confidence level for each signal based on multiple factors.

    Args:
        signals: Series containing trading signals
        data: DataFrame containing market data
        config: Signal generation configuration

    Returns:
        Series containing confidence scores (0-1)
    """
    confidence = pd.Series(0.5, index=signals.index)  # Base confidence

    # Volume-based confidence
    if "volume" in data.columns:
        volume_ratio = data["volume"] / data["volume"].rolling(window=20).mean()
        volume_confidence = np.clip(volume_ratio, 0.5, 2.0) / 2.0
        confidence *= volume_confidence

    # Volatility-based confidence (lower volatility = higher confidence)
    if "volatility" in data.columns:
        vol_confidence = 1.0 - np.clip(data["volatility"], 0, 0.5) / 0.5
        confidence *= vol_confidence

    # Signal strength confidence
    signal_strength = abs(signals)
    strength_confidence = np.clip(signal_strength, 0.1, 1.0)
    confidence *= strength_confidence

    # Trend consistency confidence
    if "sma_short" in data.columns and "sma_long" in data.columns:
        trend_alignment = np.where(
            (signals > 0) & (data["sma_short"] > data["sma_long"]),
            1.2,
            np.where((signals < 0) & (data["sma_short"] < data["sma_long"]), 1.2, 0.8),
        )
        confidence *= trend_alignment

    # RSI-based confidence
    if "rsi" in data.columns:
        rsi_confidence = np.where(
            (signals > 0) & (data["rsi"] < 30),
            1.3,  # Strong oversold buy signal
            np.where(
                (signals < 0) & (data["rsi"] > 70),
                1.3,  # Strong overbought sell signal
                np.where((data["rsi"] >= 30) & (data["rsi"] <= 70), 0.7, 1.0),
            ),  # Neutral zone
        )
        confidence *= rsi_confidence

    # Ensure confidence is within bounds
    confidence = np.clip(confidence, 0.1, 1.0)

    return confidence


def calculate_technical_indicators(
    data: pd.DataFrame, config: SignalConfig
) -> pd.DataFrame:
    """Calculate technical indicators for signal generation.

    Args:
        data: DataFrame containing market data
        config: Signal generation configuration

    Returns:
        DataFrame with added technical indicators
    """
    # Calculate returns
    data["returns"] = data["close"].pct_change()

    # Calculate volatility
    data["volatility"] = data["returns"].rolling(window=config.lookback_period).std()

    # Calculate moving averages
    data["sma_short"] = data["close"].rolling(window=config.lookback_period).mean()
    data["sma_long"] = data["close"].rolling(window=config.lookback_period * 2).mean()

    # Calculate RSI
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config.lookback_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config.lookback_period).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))

    # Calculate MACD with span validation
    span1, span2, signal_span = 12, 26, 9

    # Validate MACD spans
    if span1 >= span2:
        raise ValueError("MACD span1 must be less than span2.")

    exp1 = data["close"].ewm(span=span1, adjust=False).mean()
    exp2 = data["close"].ewm(span=span2, adjust=False).mean()
    data["macd"] = exp1 - exp2
    data["macd_signal"] = data["macd"].ewm(span=signal_span, adjust=False).mean()

    # Calculate Bollinger Bands
    data["bb_middle"] = data["close"].rolling(window=config.lookback_period).mean()
    data["bb_std"] = data["close"].rolling(window=config.lookback_period).std()
    data["bb_upper"] = data["bb_middle"] + (data["bb_std"] * 2)
    data["bb_lower"] = data["bb_middle"] - (data["bb_std"] * 2)

    return data


def apply_signal_filters(
    signals: pd.Series, data: pd.DataFrame, config: SignalConfig
) -> pd.Series:
    """Apply filters to raw trading signals.

    Args:
        signals: Series containing raw trading signals
        data: DataFrame containing market data
        config: Signal generation configuration

    Returns:
        Filtered trading signals
    """
    # Volume filter
    volume_filter = data["volume"] >= config.min_volume

    # Spread filter
    spread = (data["high"] - data["low"]) / data["low"]
    spread_filter = spread <= config.max_spread

    # Apply filters
    filtered_signals = signals.copy()
    filtered_signals[~volume_filter] = 0
    filtered_signals[~spread_filter] = 0

    # Apply smoothing if configured
    if config.smoothing > 1:
        filtered_signals = filtered_signals.rolling(window=config.smoothing).mean()

    return filtered_signals


def generate_signals(
    data: pd.DataFrame, config: SignalConfig
) -> Dict[str, SignalResult]:
    """Generate trading signals using multiple strategies with validation, confidence, and safeguards.

    Args:
        data: DataFrame containing market data
        config: Signal generation configuration

    Returns:
        Dictionary containing SignalResult for each strategy
    """
    # Enhanced data validation with safeguards
    validation_errors = validate_market_data(data)
    if validation_errors:
        logger.error(f"Market data validation failed: {validation_errors}")
        raise ValueError(f"Market data validation failed: {validation_errors}")

    # Guard against missing or short price series
    if data.empty:
        logger.error("Empty data provided for signal generation")
        raise ValueError("Empty data provided for signal generation")

    if len(data) < 3:
        logger.error(
            f"Insufficient data for signal generation: {len(data)} rows (minimum 3 required)"
        )
        raise ValueError(
            f"Insufficient data for signal generation: {len(data)} rows (minimum 3 required)"
        )

    # Validate Close column exists
    if "Close" not in data.columns:
        logger.error("Close column not found in data")
        raise ValueError("Close column not found in data")

    # Handle missing values in Close column
    data = data.dropna(subset=["Close"])
    if len(data) < 3:
        logger.error("Insufficient data after removing NaN values")
        raise ValueError("Insufficient data after removing NaN values")

    # Calculate indicators with error handling
    try:
        data = calculate_technical_indicators(data, config)
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise ValueError(f"Error calculating technical indicators: {e}")

    # Initialize results dictionary
    signal_results = {}

    # RSI strategy with safeguards
    rsi_signals = pd.Series(0, index=data.index)
    if "rsi" in data.columns and not data["rsi"].isna().all():
        rsi_signals[data["rsi"] < 30] = 1  # Oversold
        rsi_signals[data["rsi"] > 70] = -1  # Overbought
    else:
        logger.warning("RSI column missing or all NaN, skipping RSI signals")
    filtered_rsi = apply_signal_filters(rsi_signals, data, config)

    # Validate RSI signals
    is_valid, errors, completeness = validate_signals(filtered_rsi, data, config)
    confidence = calculate_signal_confidence(filtered_rsi, data, config)

    signal_results["rsi"] = SignalResult(
        signals=filtered_rsi,
        confidence=confidence,
        is_valid=is_valid,
        validation_errors=errors,
        completeness_score=completeness,
        timestamp=datetime.now(),
    )

    # MACD strategy with safeguards
    macd_signals = pd.Series(0, index=data.index)
    if (
        "macd" in data.columns
        and "macd_signal" in data.columns
        and not data["macd"].isna().all()
    ):
        macd_signals[data["macd"] > data["macd_signal"]] = 1  # Bullish crossover
        macd_signals[data["macd"] < data["macd_signal"]] = -1  # Bearish crossover
    else:
        logger.warning("MACD columns missing or all NaN, skipping MACD signals")
    filtered_macd = apply_signal_filters(macd_signals, data, config)

    # Validate MACD signals
    is_valid, errors, completeness = validate_signals(filtered_macd, data, config)
    confidence = calculate_signal_confidence(filtered_macd, data, config)

    signal_results["macd"] = SignalResult(
        signals=filtered_macd,
        confidence=confidence,
        is_valid=is_valid,
        validation_errors=errors,
        completeness_score=completeness,
        timestamp=datetime.now(),
    )

    # Bollinger Bands strategy with safeguards
    bb_signals = pd.Series(0, index=data.index)
    if (
        all(col in data.columns for col in ["close", "bb_lower", "bb_upper"])
        and not data["close"].isna().all()
    ):
        bb_signals[data["close"] < data["bb_lower"]] = 1  # Price below lower band
        bb_signals[data["close"] > data["bb_upper"]] = -1  # Price above upper band
    else:
        logger.warning(
            "Bollinger Bands columns missing or all NaN, skipping BB signals"
        )
    filtered_bb = apply_signal_filters(bb_signals, data, config)

    # Validate BB signals
    is_valid, errors, completeness = validate_signals(filtered_bb, data, config)
    confidence = calculate_signal_confidence(filtered_bb, data, config)

    signal_results["bollinger"] = SignalResult(
        signals=filtered_bb,
        confidence=confidence,
        is_valid=is_valid,
        validation_errors=errors,
        completeness_score=completeness,
        timestamp=datetime.now(),
    )

    # Moving Average Crossover strategy with safeguards
    ma_signals = pd.Series(0, index=data.index)
    if (
        all(col in data.columns for col in ["sma_short", "sma_long"])
        and not data["sma_short"].isna().all()
    ):
        ma_signals[data["sma_short"] > data["sma_long"]] = 1  # Bullish crossover
        ma_signals[data["sma_short"] < data["sma_long"]] = -1  # Bearish crossover
    else:
        logger.warning("Moving Average columns missing or all NaN, skipping MA signals")
    filtered_ma = apply_signal_filters(ma_signals, data, config)

    # Validate MA signals
    is_valid, errors, completeness = validate_signals(filtered_ma, data, config)
    confidence = calculate_signal_confidence(filtered_ma, data, config)

    signal_results["ma_crossover"] = SignalResult(
        signals=filtered_ma,
        confidence=confidence,
        is_valid=is_valid,
        validation_errors=errors,
        completeness_score=completeness,
        timestamp=datetime.now(),
    )

    # Log validation results
    for strategy, result in signal_results.items():
        if not result.is_valid:
            logger.warning(
                f"{strategy} signals validation failed: {result.validation_errors}"
            )
        else:
            logger.info(
                f"{strategy} signals generated successfully (completeness: {result.completeness_score:.3f})"
            )

    return signal_results


def generate_custom_signals(
    data: pd.DataFrame, rules: List[Dict[str, Any]], config: SignalConfig
) -> SignalResult:
    """Generate custom trading signals based on user-defined rules with validation.

    Args:
        data: DataFrame containing market data
        rules: List of dictionaries containing signal rules
        config: Signal generation configuration

    Returns:
        SignalResult containing custom trading signals with validation
    """
    # Validate data
    validation_errors = validate_market_data(data)
    if validation_errors:
        logger.error(f"Market data validation failed: {validation_errors}")
        raise ValueError(f"Market data validation failed: {validation_errors}")

    # Calculate indicators
    data = calculate_technical_indicators(data, config)

    # Initialize signals
    signals = pd.Series(0, index=data.index)

    # Apply each rule
    for rule in rules:
        condition = rule["condition"]
        value = rule["value"]
        signal = rule["signal"]

        # Evaluate condition
        if condition == "rsi":
            mask = data["rsi"] < value if signal == 1 else data["rsi"] > value
        elif condition == "macd":
            mask = data["macd"] > value if signal == 1 else data["macd"] < value
        elif condition == "bb":
            mask = (
                data["close"] < data["bb_lower"]
                if signal == 1
                else data["close"] > data["bb_upper"]
            )
        elif condition == "ma":
            mask = (
                data["sma_short"] > data["sma_long"]
                if signal == 1
                else data["sma_short"] < data["sma_long"]
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        # Apply signal
        signals[mask] = signal

    # Apply filters
    filtered_signals = apply_signal_filters(signals, data, config)

    # Validate signals
    is_valid, errors, completeness = validate_signals(filtered_signals, data, config)
    confidence = calculate_signal_confidence(filtered_signals, data, config)

    result = SignalResult(
        signals=filtered_signals,
        confidence=confidence,
        is_valid=is_valid,
        validation_errors=errors,
        completeness_score=completeness,
        timestamp=datetime.now(),
    )

    if not result.is_valid:
        logger.warning(f"Custom signals validation failed: {result.validation_errors}")
    else:
        logger.info(
            f"Custom signals generated successfully (completeness: {result.completeness_score:.3f})"
        )

    return result


def get_signal_summary(signal_results: Dict[str, SignalResult]) -> Dict[str, Any]:
    """
    Generate summary statistics for all signal strategies.

    Args:
        signal_results: Dictionary of SignalResult objects

    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        "total_strategies": len(signal_results),
        "valid_strategies": sum(1 for r in signal_results.values() if r.is_valid),
        "average_completeness": np.mean(
            [r.completeness_score for r in signal_results.values()]
        ),
        "average_confidence": np.mean(
            [r.confidence.mean() for r in signal_results.values()]
        ),
        "strategy_details": {},
    }

    for strategy, result in signal_results.items():
        summary["strategy_details"][strategy] = {
            "is_valid": result.is_valid,
            "completeness_score": result.completeness_score,
            "average_confidence": result.confidence.mean(),
            "signal_count": (result.signals != 0).sum(),
            "validation_errors": result.validation_errors,
        }

    return summary
