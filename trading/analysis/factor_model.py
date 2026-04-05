"""
Basic factor exposure model for attribution.

Explains what % of a stock's return came from momentum, reversal, volatility, volume trend.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _close_col(df: pd.DataFrame):
    return df["Close"] if "Close" in df.columns else df["close"]


def _vol_col(df: pd.DataFrame):
    return df["Volume"] if "Volume" in df.columns else df["volume"]


def _factor_momentum(df: pd.DataFrame) -> pd.Series:
    """Long-horizon return; period capped when history < 252 bars."""
    c = _close_col(df)
    n = len(c)
    if n < 3:
        return pd.Series(np.nan, index=c.index)
    if n <= 22:
        periods = max(1, n - 1)
    else:
        periods = min(252, max(5, n // 2), n - 20)
    return c.pct_change(periods)


def _factor_short_term_reversal(df: pd.DataFrame) -> pd.Series:
    c = _close_col(df)
    n = len(c)
    if n < 2:
        return pd.Series(np.nan, index=c.index)
    p = max(1, min(5, n - 1))
    return c.pct_change(p) * -1.0


def _factor_volatility(df: pd.DataFrame) -> pd.Series:
    c = _close_col(df)
    n = len(c)
    if n < 3:
        return pd.Series(np.nan, index=c.index)
    win = max(2, min(20, n - 1))
    mp = max(2, min(win - 1, max(2, win // 2)))
    return (
        c.pct_change()
        .rolling(win, min_periods=mp)
        .std()
        * np.sqrt(252.0)
    )


def _factor_volume_trend(df: pd.DataFrame) -> pd.Series:
    v = _vol_col(df)
    n = len(v)
    if n < 4:
        return pd.Series(np.nan, index=v.index)
    w_long = max(3, min(60, n - 1))
    w_short = max(2, min(20, w_long - 1))
    mp_l = max(2, w_long // 3)
    mp_s = max(2, w_short // 3)
    num = v.rolling(w_short, min_periods=mp_s).mean()
    den = v.rolling(w_long, min_periods=mp_l).mean().replace(0, np.nan)
    return num / den


STANDARD_FACTORS = {
    "momentum": _factor_momentum,
    "short_term_reversal": _factor_short_term_reversal,
    "volatility": _factor_volatility,
    "volume_trend": _factor_volume_trend,
}


def compute_factor_exposures(
    df: pd.DataFrame,
    returns: pd.Series,
    window: int = 60,
) -> Dict[str, float]:
    """
    Regress returns on standard factor returns; return factor loadings (exposures).

    Args:
        df: OHLCV with Close, Volume.
        returns: Aligned daily returns (e.g. strategy or stock).
        window: Rolling window for regression.

    Returns:
        Dict factor_name -> coefficient (exposure).
    """
    if df is None or df.empty or ("Close" not in df.columns and "close" not in df.columns):
        return {k: 0.0 for k in STANDARD_FACTORS}
    if returns is None or returns.empty:
        return {k: 0.0 for k in STANDARD_FACTORS}

    out = {}
    try:
        factors_df = pd.DataFrame(index=df.index)
        for name, func in STANDARD_FACTORS.items():
            try:
                s = func(df)
                if s is not None and not s.empty:
                    factors_df[name] = s
            except Exception:
                pass
        factors_df = factors_df.dropna(how="all").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        _min_rows = min(window, max(15, len(factors_df)))
        if factors_df.empty or len(factors_df) < _min_rows:
            return {k: 0.0 for k in STANDARD_FACTORS}
        common = returns.index.intersection(factors_df.index)
        if len(common) < 20:
            return {k: 0.0 for k in STANDARD_FACTORS}
        y = returns.reindex(common).dropna()
        X = factors_df.reindex(common).dropna()
        common = y.index.intersection(X.index)
        y = y.reindex(common).dropna()
        X = X.reindex(common).dropna()
        common = y.index.intersection(X.index)
        if len(common) < 20:
            return {k: 0.0 for k in STANDARD_FACTORS}
        y = y.loc[common]
        X = X.loc[common]
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        for i, name in enumerate(X.columns):
            out[name] = float(reg.coef_[i])
        for k in STANDARD_FACTORS:
            if k not in out:
                out[k] = 0.0
    except Exception:
        out = {k: 0.0 for k in STANDARD_FACTORS}
    return out


def factor_attribution_pct(
    df: pd.DataFrame,
    returns: pd.Series,
    window: int = 60,
) -> Dict[str, float]:
    """
    What % of total return (over window) is attributed to each factor (approximate).

    Returns dict factor -> contribution as fraction of total return (sum ≈ 1 or 0).
    """
    exposures = compute_factor_exposures(df, returns, window)
    total_ret = returns.tail(window).sum()
    if abs(total_ret) < 1e-12:
        return {k: 0.0 for k in exposures}
    try:
        factors_df = pd.DataFrame(index=df.index)
        for name, func in STANDARD_FACTORS.items():
            try:
                s = func(df)
                if s is not None and not s.empty:
                    factors_df[name] = s
            except Exception:
                pass
        factors_df = factors_df.dropna(how="all").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        common = returns.index.intersection(factors_df.index).intersection(df.index)
        _min_c = min(window, max(15, len(common)))
        if len(common) < _min_c:
            return {k: 0.0 for k in exposures}
        r = returns.reindex(common).dropna().tail(window)
        F = factors_df.reindex(common).dropna().tail(window)
        common = r.index.intersection(F.index)
        r = r.loc[common]
        F = F.loc[common]
        contributions = {}
        for name in F.columns:
            exp = exposures.get(name, 0.0)
            contributions[name] = float(exp * F[name].sum())
        total_contrib = sum(contributions.values())
        if abs(total_ret) < 1e-12:
            return {k: 0.0 for k in contributions}
        return {k: (v / total_ret) for k, v in contributions.items()}
    except Exception:
        return {k: 0.0 for k in STANDARD_FACTORS}


class FactorModel:
    """
    Factor exposure and attribution model.
    Wraps compute_factor_exposures and factor_attribution_pct.
    """

    FACTORS = STANDARD_FACTORS

    def __init__(self):
        self._exposures: Dict[str, Dict[str, float]] = {}

    def _ohlcv_for_returns(
        self,
        returns: pd.Series,
        ohlcv: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if ohlcv is not None and not ohlcv.empty:
            _cm = {c.lower(): c for c in ohlcv.columns}
            cc = _cm.get("close")
            vc = _cm.get("volume")
            if cc is None:
                raise ValueError("ohlcv must include a close column")
            out = pd.DataFrame(index=ohlcv.index)
            out["close"] = pd.to_numeric(ohlcv[cc], errors="coerce")
            if vc is not None:
                out["volume"] = pd.to_numeric(ohlcv[vc], errors="coerce")
            else:
                out["volume"] = 1.0
            return out
        if returns is None or returns.empty:
            return pd.DataFrame()
        price = (1.0 + returns.fillna(0.0)).cumprod()
        return pd.DataFrame(
            {"close": price, "volume": 1.0},
            index=returns.index,
        )

    def compute_exposures(
        self,
        symbol: str,
        returns: pd.Series,
        factor_returns: Optional[Dict[str, Any]] = None,
        ohlcv: Optional[pd.DataFrame] = None,
        window: int = 60,
    ) -> Dict[str, float]:
        """Compute factor exposures for a symbol."""
        if factor_returns:
            logger.debug(
                "factor_returns is not used by STANDARD_FACTORS regression; ignoring."
            )
        try:
            df = self._ohlcv_for_returns(returns, ohlcv)
            _rs = returns if returns is not None else pd.Series(dtype=float)
            logger.debug(
                "compute_exposures(%s): returns len=%s, dtype=%s, NaN count=%s",
                symbol,
                len(_rs),
                getattr(_rs, "dtype", type(_rs)),
                int(_rs.isna().sum()) if len(_rs) else 0,
            )
            if ohlcv is not None:
                logger.debug(
                    "compute_exposures(%s): ohlcv columns=%s, len=%s",
                    symbol,
                    list(ohlcv.columns),
                    len(ohlcv),
                )
            result = compute_factor_exposures(df, returns, window=window)
            self._exposures[str(symbol)] = result
            return result
        except Exception as e:
            logger.warning(
                "Factor exposure failed for %s: %s",
                symbol,
                e,
            )
            return {}

    def get_attribution(
        self,
        symbol: str,
        returns: pd.Series,
        factor_returns: Optional[Dict[str, Any]] = None,
        ohlcv: Optional[pd.DataFrame] = None,
        window: int = 60,
    ) -> Dict[str, float]:
        """Get factor attribution breakdown."""
        if factor_returns:
            logger.debug(
                "factor_returns is not used by factor_attribution_pct; ignoring."
            )
        try:
            df = self._ohlcv_for_returns(returns, ohlcv)
            return factor_attribution_pct(df, returns, window=window)
        except Exception as e:
            logger.warning(
                "Factor attribution failed for %s: %s",
                symbol,
                e,
            )
            return {}
