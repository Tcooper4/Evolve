# -*- coding: utf-8 -*-
"""Portfolio concentration from pairwise return correlations.

Threshold rationale
-------------------
Flag a pair when |Pearson rho| of aligned daily returns exceeds
``HIGH_PAIRWISE_CORR = 0.70``.

Shared variance is rho^2. At 0.70 that is ~49% — nearly half the move
in one name is associated with the other, so the diversification
benefit of holding both is materially reduced. At 0.50 (25% shared)
there is still useful diversification; 0.70 is a common industry
"highly correlated" screen and matches that variance geometry rather
than a round 0.5 / 0.8 guess.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HIGH_PAIRWISE_CORR = 0.70
LOOKBACK_DAYS = 252

THRESHOLD_NOTE = (
    f"Pairs are flagged when |correlation| of daily returns over up to "
    f"{LOOKBACK_DAYS}d exceeds {HIGH_PAIRWISE_CORR:.2f} "
    f"(~{HIGH_PAIRWISE_CORR**2:.0%} shared variance) — diversification "
    f"benefit of holding both is materially reduced."
)


def correlation_matrix_for_symbols(
    symbols: Sequence[str],
    price_history: Dict[str, Union[pd.Series, pd.DataFrame, Sequence[float]]],
    *,
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Compute corr via ``PortfolioManager.calculate_correlation_matrix``."""
    syms = [str(s).strip().upper() for s in symbols if s and str(s).strip()]
    if len(syms) < 2:
        return pd.DataFrame()
    try:
        from trading.portfolio.portfolio_manager import PortfolioManager

        pm = PortfolioManager()
        pm.symbols = list(dict.fromkeys(syms))  # stable unique
        return pm.calculate_correlation_matrix(
            market_data={"prices": price_history},
            lookback_days=int(lookback_days),
        )
    except Exception as e:
        logger.debug("correlation_matrix_for_symbols failed: %s", e)
        return pd.DataFrame()


def flag_high_correlation_pairs(
    corr: pd.DataFrame,
    *,
    threshold: float = HIGH_PAIRWISE_CORR,
) -> List[Dict[str, Any]]:
    """Upper-triangle pairs with |rho| >= threshold (excludes diagonal)."""
    flags: List[Dict[str, Any]] = []
    if corr is None or corr.empty or corr.shape[0] < 2:
        return flags
    try:
        thr = float(threshold)
    except Exception:
        thr = HIGH_PAIRWISE_CORR
    cols = [str(c) for c in corr.columns]
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            try:
                rho = float(corr.loc[a, b]) if a in corr.index else float("nan")
            except Exception:
                try:
                    rho = float(corr.iloc[i, j])
                except Exception:
                    continue
            if not np.isfinite(rho):
                continue
            if abs(rho) >= thr:
                flags.append({
                    "symbol_a": a,
                    "symbol_b": b,
                    "correlation": round(rho, 4),
                    "abs_correlation": round(abs(rho), 4),
                    "message": (
                        f"{a} and {b} have moved together closely recently "
                        f"(corr {rho:+.2f}) — this reduces the diversification "
                        f"benefit of holding both."
                    ),
                })
    flags.sort(key=lambda x: -float(x["abs_correlation"]))
    return flags


def build_concentration_report(
    symbols: Sequence[str],
    price_history: Dict[str, Union[pd.Series, pd.DataFrame, Sequence[float]]],
    *,
    threshold: float = HIGH_PAIRWISE_CORR,
    lookback_days: int = LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """Full report for account-risk; never raises."""
    out: Dict[str, Any] = {
        "success": False,
        "threshold": float(threshold),
        "threshold_note": THRESHOLD_NOTE,
        "matrix": {},
        "high_pairs": [],
        "n_symbols": 0,
        "lookback_days": int(lookback_days),
        "error": None,
    }
    try:
        syms = [str(s).strip().upper() for s in symbols if s and str(s).strip()]
        syms = list(dict.fromkeys(syms))
        out["n_symbols"] = len(syms)
        if len(syms) < 2:
            out["success"] = True
            out["error"] = None
            out["note"] = "Need at least two holdings with price history to assess concentration."
            return out

        corr = correlation_matrix_for_symbols(
            syms, price_history, lookback_days=lookback_days
        )
        if corr is None or corr.empty:
            out["error"] = "insufficient aligned price history for correlation"
            out["success"] = True  # graceful empty
            return out

        # JSON-safe nested matrix
        matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for a in corr.index:
            row: Dict[str, Optional[float]] = {}
            for b in corr.columns:
                try:
                    v = float(corr.loc[a, b])
                    row[str(b)] = round(v, 4) if np.isfinite(v) else None
                except Exception:
                    row[str(b)] = None
            matrix[str(a)] = row
        out["matrix"] = matrix
        out["high_pairs"] = flag_high_correlation_pairs(corr, threshold=threshold)
        out["success"] = True
        out["note"] = (
            f"{len(out['high_pairs'])} high-correlation pair(s) above "
            f"{threshold:.2f}." if out["high_pairs"] else
            f"No pair above |corr| {threshold:.2f} — holdings look less "
            f"duplicative on this window."
        )
        return out
    except Exception as e:
        logger.debug("build_concentration_report failed: %s", e)
        out["error"] = str(e)
        out["success"] = True  # do not fail parent risk payload
        return out


__all__ = [
    "HIGH_PAIRWISE_CORR",
    "LOOKBACK_DAYS",
    "THRESHOLD_NOTE",
    "correlation_matrix_for_symbols",
    "flag_high_correlation_pairs",
    "build_concentration_report",
]
