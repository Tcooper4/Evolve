"""
Shared feature engineering and price normalization for all forecasting models.

Ensures every model receives minimum meaningful features (returns, volatility, RSI, MACD, volume ratio)
and trains on normalized prices (divide by last known price) to learn dynamics, not price level.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_close(df: pd.DataFrame) -> str:
    """Return the name of the close/price column (case-insensitive)."""
    for c in ["close", "Close", "price", "Price"]:
        if c in df.columns:
            return c
    # First numeric column as fallback
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        return numeric.columns[0]
    return "close"


def _ensure_volume(df: pd.DataFrame) -> Optional[str]:
    """Return volume column name if present."""
    for c in ["volume", "Volume"]:
        if c in df.columns:
            return c
    return None


def add_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add minimum meaningful features for price forecasting. Call this before training any model.

    Adds:
    - returns: pct_change of close
    - rolling_vol_20: 20-day std of returns
    - rsi_14: 14-period RSI
    - macd_signal: MACD signal line (12/26/9)
    - volume_ratio: volume / 20-day avg volume

    Modifies df in place and returns it. Fills NaNs with 0 or neutral (RSI 50) so models can train.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    close_col = _ensure_close(df)
    if close_col not in df.columns:
        logger.warning("No close column found for forecast features")
        return df

    close = pd.Series(df[close_col].values, index=df.index)

    # 1. Returns (pct_change)
    returns = close.pct_change()
    df["returns"] = returns
    df["returns"] = df["returns"].fillna(0)

    # 2. Rolling volatility (20-day std of returns)
    df["rolling_vol_20"] = df["returns"].rolling(20, min_periods=1).std()
    df["rolling_vol_20"] = df["rolling_vol_20"].fillna(0)

    # 3. RSI 14-period
    try:
        from trading.utils.safe_math import safe_rsi
        rsi = safe_rsi(close, period=14)
        df["rsi_14"] = rsi if isinstance(rsi, pd.Series) else pd.Series(rsi, index=df.index)
    except Exception as e:
        logger.debug("RSI computation failed: %s", e)
        df["rsi_14"] = 50.0
    df["rsi_14"] = df["rsi_14"].fillna(50.0)

    # 4. MACD signal (12/26/9)
    try:
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_signal"] = signal
        df["macd_signal"] = df["macd_signal"].fillna(0)
    except Exception as e:
        logger.debug("MACD computation failed: %s", e)
        df["macd_signal"] = 0.0

    # 5. Volume ratio (volume / 20d avg volume)
    vol_col = _ensure_volume(df)
    if vol_col:
        vol = df[vol_col].replace(0, np.nan)
        vol_avg_20 = vol.rolling(20, min_periods=1).mean()
        df["volume_ratio"] = (vol / vol_avg_20).fillna(1.0)
    else:
        df["volume_ratio"] = 1.0

    return df


def prepare_forecast_data(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Add forecast features and normalize the target column by the last known price.
    Use the returned last_price to denormalize forecasts: forecast_price = forecast_normalized * last_price.

    IMPORTANT: last_price is extracted ONLY from the raw close column BEFORE any feature
    engineering or normalization. It is never modified. The router must multiply the
    final forecast array by this raw last_price exactly once.
    """
    if df is None or df.empty:
        return df, 1.0
    raw = df.copy()
    close_col = target_col or _ensure_close(raw)
    if close_col not in raw.columns:
        return df, 1.0
    # Extract raw last price BEFORE any feature engineering — never modify this value
    last_price = float(raw[close_col].iloc[-1])
    if last_price <= 0 or not np.isfinite(last_price):
        last_price = 1.0
    # Now add features (does not change close column values)
    df = add_forecast_features(raw)
    # Normalize only the target column by the stored raw last_price
    df[close_col] = df[close_col] / last_price
    return df, last_price
