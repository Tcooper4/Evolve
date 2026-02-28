# -*- coding: utf-8 -*-
"""
Monitoring tools — detect model/strategy degradation and write
human-readable recommendations to MemoryStore (long_term, category=recommendations).
Surfaced by the Home briefing. Replaces auto-executing MetaLearnerAgent and MetaTunerAgent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Data quality thresholds
MAX_GAP_DAYS = 5
STALE_DAYS = 2
ANOMALY_PCT = 0.50


def _get_memory_store():
    try:
        from trading.memory import get_memory_store
        return get_memory_store()
    except Exception as e:
        logger.warning("get_memory_store failed: %s", e)
        return None


def _write_recommendation(
    title: str,
    text: str,
    source: str = "monitoring_tools",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Write a recommendation to long_term memory, category=recommendations."""
    store = _get_memory_store()
    if not store:
        return False
    try:
        from trading.memory.memory_store import MemoryType

        value = {"title": title, "text": text, "source": source}
        if metadata:
            value["metadata"] = metadata
        store.add(
            MemoryType.LONG_TERM,
            namespace="monitoring",
            value=value,
            category="recommendations",
            metadata=metadata or {},
        )
        logger.info("Wrote recommendation: %s", title)
        return True
    except Exception as e:
        logger.exception("_write_recommendation failed: %s", e)
        return False


def check_model_degradation(
    model_id: Optional[str] = None,
    recent_sharpe: Optional[float] = None,
    recent_drawdown: Optional[float] = None,
    threshold_sharpe: float = 0.3,
    threshold_drawdown: float = -0.2,
) -> Dict[str, Any]:
    """
    Check for model degradation (e.g. Sharpe drop, drawdown worsening).
    If degradation is detected, write a recommendation to MemoryStore.
    If no metrics are passed (all performance params None), return immediately without writing.
    """
    # No data: do not write anything
    if recent_sharpe is None and recent_drawdown is None:
        return {"degradation_detected": False, "message": "No metrics provided; skipping check"}

    degraded = False
    reasons = []
    if recent_sharpe is not None and recent_sharpe < threshold_sharpe:
        degraded = True
        reasons.append(f"Sharpe ratio {recent_sharpe:.3f} below threshold {threshold_sharpe}")
    if recent_drawdown is not None and recent_drawdown < threshold_drawdown:
        degraded = True
        reasons.append(f"Drawdown {recent_drawdown:.1%} below threshold {threshold_drawdown:.1%}")

    if not degraded:
        return {"degradation_detected": False, "message": "No degradation detected"}

    title = f"Model degradation: {model_id or 'model'}"
    text = "Consider retraining or switching model. " + "; ".join(reasons)
    ok = _write_recommendation(
        title,
        text,
        source="check_model_degradation",
        metadata={"model_id": model_id, "recent_sharpe": recent_sharpe, "recent_drawdown": recent_drawdown},
    )
    return {"degradation_detected": True, "reasons": reasons, "written_to_memory": ok}


def check_strategy_degradation(
    strategy_name: Optional[str] = None,
    recent_sharpe: Optional[float] = None,
    recent_win_rate: Optional[float] = None,
    threshold_sharpe: float = 0.3,
    threshold_win_rate: float = 0.4,
) -> Dict[str, Any]:
    """
    Check for strategy degradation. If detected, write a recommendation to MemoryStore.
    If no metrics are passed (all performance params None), return immediately without writing.
    """
    # No data: do not write anything
    if recent_sharpe is None and recent_win_rate is None:
        return {"degradation_detected": False, "message": "No metrics provided; skipping check"}

    degraded = False
    reasons = []
    if recent_sharpe is not None and recent_sharpe < threshold_sharpe:
        degraded = True
        reasons.append(f"Sharpe ratio {recent_sharpe:.3f} below threshold {threshold_sharpe}")
    if recent_win_rate is not None and recent_win_rate < threshold_win_rate:
        degraded = True
        reasons.append(f"Win rate {recent_win_rate:.1%} below threshold {threshold_win_rate:.1%}")

    if not degraded:
        return {"degradation_detected": False, "message": "No degradation detected"}

    title = f"Strategy degradation: {strategy_name or 'strategy'}"
    text = "Consider re-optimizing parameters or switching strategy. " + "; ".join(reasons)
    ok = _write_recommendation(
        title,
        text,
        source="check_strategy_degradation",
        metadata={"strategy_name": strategy_name, "recent_sharpe": recent_sharpe, "recent_win_rate": recent_win_rate},
    )
    return {"degradation_detected": True, "reasons": reasons, "written_to_memory": ok}


def check_data_quality(
    symbol: str,
    data: Union[pd.DataFrame, Any],
    *,
    max_gap_days: int = MAX_GAP_DAYS,
    stale_days: int = STALE_DAYS,
    anomaly_pct: float = ANOMALY_PCT,
) -> Dict[str, Any]:
    """
    Check for obvious data issues: gaps > max_gap_days, stale data older than stale_days,
    single-day price move > anomaly_pct (e.g. 50%). If issues found, write a recommendation
    to MemoryStore with plain-English explanation.
    """
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        return {"issues_detected": False, "message": "No data to check"}

    if not isinstance(data, pd.DataFrame):
        return {"issues_detected": False, "message": "Data is not a DataFrame"}

    df = data.copy()
    issues: List[str] = []
    # Resolve date index or column
    if df.index.name in (None, "") and "date" not in df.columns and "Date" not in df.columns:
        try:
            if hasattr(df.index, "date"):
                df = df.reset_index()
        except Exception:
            pass
    date_col = None
    if hasattr(df.index, "date") or (hasattr(df.index, "is_all_dates") and df.index.is_all_dates):
        df = df.reset_index()
    for c in ["date", "Date", "datetime", "timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None and len(df.columns) > 0:
        # Try first column as dates
        date_col = df.columns[0]
    if date_col is None:
        return {"issues_detected": False, "message": "No date column found"}

    try:
        df["_dt"] = pd.to_datetime(df[date_col])
    except Exception as e:
        logger.debug("check_data_quality: could not parse dates: %s", e)
        return {"issues_detected": False, "message": "Could not parse dates"}

    df = df.sort_values("_dt").drop_duplicates(subset=["_dt"])
    price_col = None
    for c in ["close", "Close", "Adj Close", "adj close"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None and len(df.columns) > 0:
        for c in df.columns:
            if c not in ("_dt", date_col) and pd.api.types.is_numeric_dtype(df[c]):
                price_col = c
                break
    if not price_col:
        return {"issues_detected": False, "message": "No price column found"}

    # Gaps longer than max_gap_days
    diffs = df["_dt"].diff().dt.days
    big_gaps = diffs[diffs > max_gap_days]
    if not big_gaps.empty:
        first_gap_idx = big_gaps.index[0]
        try:
            pos = df.index.get_loc(first_gap_idx)
            gap_start = df["_dt"].iloc[pos - 1] if pos > 0 else df["_dt"].iloc[0]
        except Exception:
            gap_start = df["_dt"].iloc[0]
        gap_days = int(big_gaps.iloc[0])
        gap_start_str = pd.Timestamp(gap_start).strftime("%B %d") if pd.notna(gap_start) else "unknown"
        issues.append(f"{symbol} price data has a {gap_days}-day gap starting {gap_start_str} — this may affect backtest accuracy.")

    # Stale: last date older than stale_days
    last_ts = df["_dt"].max()
    if pd.notna(last_ts):
        try:
            last_dt = pd.Timestamp(last_ts)
            now = pd.Timestamp.now()
            age_days = (now - last_dt).days
            if age_days > stale_days:
                dt_str = last_dt.strftime("%Y-%m-%d") if hasattr(last_dt, "strftime") else str(last_dt)[:10]
                issues.append(f"{symbol} data is {age_days} days old (last date {dt_str}) — consider refreshing for accurate results.")
        except Exception as e:
            logger.debug("check_data_quality: stale check failed: %s", e)

    # Single-day move > anomaly_pct
    pct = df[price_col].pct_change()
    big_moves = pct[abs(pct) > anomaly_pct]
    if not big_moves.empty:
        idx = big_moves.index[0]
        dt_str = str(df.loc[idx, "_dt"].strftime("%Y-%m-%d")) if hasattr(df.loc[idx, "_dt"], "strftime") else str(df.loc[idx, "_dt"])
        issues.append(f"{symbol} had a single-day price move over {anomaly_pct:.0%} on {dt_str} — verify data source.")

    if not issues:
        return {"issues_detected": False, "message": "No data quality issues detected"}

    title = f"Data quality: {symbol}"
    text = " ".join(issues)
    ok = _write_recommendation(
        title,
        text,
        source="check_data_quality",
        metadata={"symbol": symbol, "issues_count": len(issues)},
    )
    return {"issues_detected": True, "issues": issues, "written_to_memory": ok}
