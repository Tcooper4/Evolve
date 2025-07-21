"""RSI strategy signal generator.
Enhanced with proper index alignment and empty slice validation.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .rsi_utils import generate_rsi_signals_core, validate_rsi_parameters

warnings.filterwarnings("ignore")


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def validate_dataframe_index(df: pd.DataFrame) -> bool:
    """Validate DataFrame index for proper alignment.

    Args:
        df: DataFrame to validate

    Returns:
        True if index is valid, False otherwise
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex, converting...")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert index to DatetimeIndex: {e}")
            return False

    # Check for duplicate indices
    if df.index.duplicated().any():
        logger.warning("Duplicate indices found, keeping first occurrence")
        df = df[~df.index.duplicated(keep="first")]

    # Check for monotonic index
    if not df.index.is_monotonic_increasing:
        logger.warning("Index is not monotonic, sorting...")
        df.sort_index(inplace=True)

    return True


def align_signals_with_index(
    signals: pd.DataFrame, original_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Ensure signals DataFrame aligns with original index after transformations.

    Args:
        signals: DataFrame with signals
        original_index: Original DataFrame index

    Returns:
        Aligned signals DataFrame
    """
    try:
        # Create a template DataFrame with the original index
        template = pd.DataFrame(index=original_index)

        # Reindex signals to match original index
        aligned_signals = signals.reindex(original_index)

        # Fill NaN values appropriately
        # For numeric columns, forward fill then backward fill
        numeric_columns = aligned_signals.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            aligned_signals[col] = (
                aligned_signals[col]
                .fillna(method="ffill")
                .fillna(method="bfill")
                .fillna(0)
            )

        # For non-numeric columns, fill with appropriate defaults
        non_numeric_columns = aligned_signals.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            if col == "signal":
                aligned_signals[col] = aligned_signals[col].fillna(0)
            else:
                aligned_signals[col] = aligned_signals[col].fillna("")

        logger.debug(
            f"Aligned signals from {len(signals)} to {len(aligned_signals)} rows"
        )
        return aligned_signals

    except Exception as e:
        logger.error(f"Error aligning signals with index: {e}")
        # Return original signals if alignment fails
        return signals


def validate_empty_slices(df: pd.DataFrame, period: int) -> bool:
    """Validate that data has sufficient length for the given period.

    Args:
        df: DataFrame to validate
        period: Period for calculations

    Returns:
        True if data is sufficient, False otherwise
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False

    if len(df) < period:
        logger.error(f"Insufficient data: {len(df)} rows, need at least {period} rows")
        return False

    # Check for sufficient non-null values
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            if non_null_count < period:
                logger.error(
                    f"Insufficient non-null values in {col}: {non_null_count}, need at least {period}"
                )
                return False

    return True


def load_optimized_settings(ticker: str) -> Dict[str, Any]:
    """Load optimized RSI settings for a ticker if available.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary of optimized settings or None if not found
    """
    try:
        settings_file = Path(f"memory/strategy_settings/rsi/{ticker}.json")
        if settings_file.exists():
            with open(settings_file, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading optimized settings for {ticker}: {str(e)}")
    return None


def generate_rsi_signals(
    df: pd.DataFrame,
    ticker: str = None,
    period: int = 14,
    buy_threshold: float = 30,
    sell_threshold: float = 70,
    user_config: dict = None,
    streamlit_session: dict = None,
) -> pd.DataFrame:
    """Generate RSI trading signals with user-configurable thresholds.
    Enhanced with index alignment and empty slice validation.

    RSI (Relative Strength Index) Strategy Logic:
    - Buy signals are triggered when RSI falls below buy_threshold (default 30)
      indicating oversold conditions where price may reverse upward
    - Sell signals are triggered when RSI rises above sell_threshold (default 70)
      indicating overbought conditions where price may reverse downward
    - RSI is calculated using a rolling window of 'period' periods (default 14)
      with exponential smoothing to reduce noise
    - Edge cases are handled including:
      * Insufficient data (less than period length)
      * All NaN values in price data
      * Non-monotonic or duplicate indices
      * Constant price series (zero variance)

    Args:
        df: Price data DataFrame with OHLCV columns (must include 'Close')
        ticker: Optional ticker symbol for loading optimized settings
        period: RSI calculation period (default 14, typical range 10-21)
        buy_threshold: RSI level for buy signals (default 30, typical range 20-40)
        sell_threshold: RSI level for sell signals (default 70, typical range 60-80)
        user_config: Optional user configuration dict with custom thresholds
        streamlit_session: Optional Streamlit session for UI integration

    Returns:
        DataFrame with RSI signals and returns including:
        - 'RSI': Calculated RSI values
        - 'signal': Trading signals (1=buy, -1=sell, 0=hold)
        - 'buy_signal': Boolean buy signal flags
        - 'sell_signal': Boolean sell signal flags
        - 'oversold': Boolean oversold condition flags
        - 'overbought': Boolean overbought condition flags

    Raises:
        ValueError: If DataFrame is invalid, missing required columns, or insufficient data
    """
    try:
        # Store original index for alignment
        original_index = df.index.copy()

        # Validate DataFrame index
        if not validate_dataframe_index(df):
            raise ValueError("Invalid DataFrame index")

        # Validate against empty slices
        if not validate_empty_slices(df, period):
            raise ValueError(f"Insufficient data for RSI period {period}")

        # Validate required columns
        if "Close" not in df.columns:
            raise ValueError("Missing 'Close' column in DataFrame")

        # Apply user configuration if provided
        if user_config:
            buy_threshold = user_config.get("buy_threshold", buy_threshold)
            sell_threshold = user_config.get("sell_threshold", sell_threshold)
            period = user_config.get("period", period)

        # Apply Streamlit session configuration if provided
        if streamlit_session:
            buy_threshold = streamlit_session.get("rsi_buy_threshold", buy_threshold)
            sell_threshold = streamlit_session.get("rsi_sell_threshold", sell_threshold)
            period = streamlit_session.get("rsi_period", period)

        # Load optimized settings if available and no user config
        if ticker and not user_config and not streamlit_session:
            optimized = load_optimized_settings(ticker)
            if optimized:
                period = optimized["optimal_period"]
                buy_threshold = optimized["buy_threshold"]
                sell_threshold = optimized["sell_threshold"]
                logger.info(f"Using optimized RSI settings for {ticker}")

        # Validate parameters
        is_valid, error_msg = validate_rsi_parameters(
            period, buy_threshold, sell_threshold
        )
        if not is_valid:
            raise ValueError(f"Invalid RSI parameters: {error_msg}")

        # Use shared core function
        result_df = generate_rsi_signals_core(
            df,
            period=period,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        # Ensure signals align with original index
        result_df = align_signals_with_index(result_df, original_index)

        # Validate final result
        if result_df is None or result_df.empty:
            raise ValueError("Generated signals DataFrame is empty")

        if len(result_df) != len(original_index):
            logger.warning(
                f"Signal length mismatch: {len(result_df)} vs {len(original_index)}"
            )

        return result_df

    except Exception as e:
        error_msg = f"Error generating RSI signals: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def generate_signals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Generate trading signals using RSI strategy.
    Enhanced with proper index alignment and validation.

    This function implements the shared strategy interface that returns a DataFrame
    with signal columns for consistent usage across the system.

    Args:
        df: Price data DataFrame with OHLCV columns
        **kwargs: Additional parameters including:
            - ticker: Optional ticker symbol for loading optimized settings
            - period: RSI period (default: 14)
            - buy_threshold: RSI level for buy signals (default: 30)
            - sell_threshold: RSI level for sell signals (default: 70)

    Returns:
        DataFrame with original data plus signal columns:
            - signal: 1 for buy, -1 for sell, 0 for hold
            - rsi: RSI indicator values
            - returns: Price returns
            - strategy_returns: Strategy returns
            - cumulative_returns: Cumulative price returns
            - strategy_cumulative_returns: Cumulative strategy returns

    Raises:
        RuntimeError: If signal generation fails
    """
    try:
        # Extract parameters
        ticker = kwargs.get("ticker")
        period = kwargs.get("period", 14)
        buy_threshold = kwargs.get("buy_threshold", 30)
        sell_threshold = kwargs.get("sell_threshold", 70)

        # Validate input data
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")

        # Generate RSI signals
        result_df = generate_rsi_signals(
            df,
            ticker=ticker,
            period=period,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        # Ensure we have all required columns
        required_columns = ["signal", "rsi", "returns", "strategy_returns"]
        for col in required_columns:
            if col not in result_df.columns:
                logger.warning(f"Missing required column {col} in RSI strategy output")
                result_df[col] = 0

        # Add metadata columns
        result_df["strategy_name"] = "RSI"
        result_df["strategy_params"] = json.dumps(
            {
                "period": period,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "ticker": ticker,
            }
        )

        # Final validation
        if result_df.index.duplicated().any():
            logger.warning("Duplicate indices in final result, removing duplicates")
            result_df = result_df[~result_df.index.duplicated(keep="first")]

        logger.info(
            f"Successfully generated RSI signals for {len(result_df)} data points"
        )
        return result_df

    except Exception as e:
        error_msg = f"Error generating RSI signals: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def calculate_rsi_fallback(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using a fallback implementation when pandas_ta is not available.
    Enhanced with index alignment and validation.

    Args:
        prices: Price series
        period: RSI period

    Returns:
        RSI values
    """
    try:
        # Validate input
        if prices is None or prices.empty:
            raise ValueError("Price series is None or empty")

        if len(prices) < period:
            raise ValueError(
                f"Insufficient data: {len(prices)} points, need at least {period}"
            )

        # Store original index
        original_index = prices.index.copy()

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Ensure alignment with original index
        rsi = rsi.reindex(original_index)

        return rsi

    except Exception as e:
        logger.error(f"Error in RSI fallback calculation: {e}")
        # Return NaN series with same index
        return pd.Series(index=prices.index, dtype=float)
