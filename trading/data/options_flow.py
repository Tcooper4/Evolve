# -*- coding: utf-8 -*-
"""Options flow / unusual activity from yfinance chains."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

OPTIONS_CACHE_TTL_SEC = 14400.0
_options_cache_lock = threading.Lock()


def _options_db_path() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    p = root / "data" / "options_cache.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _neutral_options(symbol: str) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    return {
        "success": False,
        "put_call_ratio": 0.0,
        "unusual_calls": [],
        "unusual_puts": [],
        "max_pain": 0.0,
        "net_flow": "NEUTRAL",
        "expiries": [],
        "error": "symbol required" if not sym else None,
        "source": "yfinance",
    }


def _get_options_cache(symbol: str, top_n: int) -> Optional[Dict[str, Any]]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    cache_key = f"{sym}|{top_n}"
    try:
        with _options_cache_lock:
            conn = sqlite3.connect(
                str(_options_db_path()),
                check_same_thread=False,
            )
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS options_cache (
                        symbol TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        ts REAL NOT NULL
                    )
                    """
                )
                cur = conn.execute(
                    "SELECT data, ts FROM options_cache WHERE symbol = ?",
                    (cache_key,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                raw, ts = row[0], float(row[1])
                if time.time() - ts > OPTIONS_CACHE_TTL_SEC:
                    conn.execute(
                        "DELETE FROM options_cache WHERE symbol = ?",
                        (cache_key,),
                    )
                    conn.commit()
                    return None
                return json.loads(raw)
            finally:
                conn.close()
    except Exception as e:
        logger.debug("options cache read failed: %s", e)
        return None


def _set_options_cache(symbol: str, top_n: int, data: Dict[str, Any]) -> None:
    sym = (symbol or "").strip().upper()
    if not sym:
        return
    cache_key = f"{sym}|{top_n}"
    try:
        with _options_cache_lock:
            conn = sqlite3.connect(
                str(_options_db_path()),
                check_same_thread=False,
            )
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS options_cache (
                        symbol TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        ts REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO options_cache (symbol, data, ts)
                    VALUES (?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        data = excluded.data,
                        ts = excluded.ts
                    """,
                    (cache_key, json.dumps(data), time.time()),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        logger.debug("options cache write failed: %s", e)


def _max_pain_strike(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """Strike minimizing total ITM intrinsic * open interest (writer pain)."""
    try:
        if calls.empty and puts.empty:
            return 0.0
        strikes: List[float] = []
        if not calls.empty and "strike" in calls.columns:
            strikes.extend(calls["strike"].astype(float).tolist())
        if not puts.empty and "strike" in puts.columns:
            strikes.extend(puts["strike"].astype(float).tolist())
        if not strikes:
            return 0.0
        candidates = sorted(set(strikes))
        best_k = candidates[0]
        min_pain = float("inf")
        for k in candidates:
            pain = 0.0
            if not calls.empty:
                for _, row in calls.iterrows():
                    oi = float(row.get("openInterest") or 0)
                    strike = float(row["strike"])
                    pain += max(0.0, k - strike) * oi * 100.0
            if not puts.empty:
                for _, row in puts.iterrows():
                    oi = float(row.get("openInterest") or 0)
                    strike = float(row["strike"])
                    pain += max(0.0, strike - k) * oi * 100.0
            if pain < min_pain:
                min_pain = pain
                best_k = k
        return float(best_k)
    except Exception as e:
        logger.debug("max pain calc failed: %s", e)
        return 0.0


def _unusual_for_expiry(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    expiry: str,
    top_n: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    unusual_calls: List[Dict[str, Any]] = []
    unusual_puts: List[Dict[str, Any]] = []
    try:
        for name, df, bucket in (
            ("call", calls, unusual_calls),
            ("put", puts, unusual_puts),
        ):
            if df is None or df.empty or "volume" not in df.columns:
                continue
            vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            pos = vol[vol > 0]
            avg_v = float(pos.mean()) if len(pos) > 0 else 0.0
            if avg_v <= 0:
                continue
            thresh = 2.0 * avg_v
            mask = vol > thresh
            sub = df.loc[mask].copy()
            sub = sub.assign(_v=vol.loc[mask])
            sub = sub.sort_values("_v", ascending=False).head(top_n)
            for _, row in sub.iterrows():
                bucket.append(
                    {
                        "strike": float(row["strike"]),
                        "expiry": expiry,
                        "volume": float(row.get("volume") or 0),
                        "openInterest": float(row.get("openInterest") or 0),
                        "type": name,
                    }
                )
    except Exception as e:
        logger.debug("unusual activity scan failed: %s", e)
    return unusual_calls, unusual_puts


def _fetch_options_data(symbol: str, top_n: int) -> Dict[str, Any]:
    """
    Blocking yfinance options fetch (run in a worker thread with timeout).
    """
    out: Dict[str, Any] = {
        "success": False,
        "put_call_ratio": 0.0,
        "unusual_calls": [],
        "unusual_puts": [],
        "max_pain": 0.0,
        "net_flow": "NEUTRAL",
        "expiries": [],
        "error": None,
        "source": "yfinance",
    }
    sym = (symbol or "").strip().upper()
    if not sym:
        out["error"] = "symbol required"
        return out
    import yfinance as yf

    t = yf.Ticker(sym)
    expiries = list(t.options or [])[:8]
    out["expiries"] = [str(e) for e in expiries]
    if not expiries:
        out["error"] = "no options expiries"
        return out

    total_cv = 0.0
    total_pv = 0.0
    all_uc: List[Dict[str, Any]] = []
    all_up: List[Dict[str, Any]] = []
    agg_calls_frames: List[pd.DataFrame] = []
    agg_puts_frames: List[pd.DataFrame] = []

    for exp in expiries:
        try:
            chain = t.option_chain(exp)
            c = chain.calls
            p = chain.puts
            if not c.empty:
                agg_calls_frames.append(c)
            if not p.empty:
                agg_puts_frames.append(p)
            if not c.empty:
                total_cv += float(
                    pd.to_numeric(c["volume"], errors="coerce").fillna(0).sum()
                )
            if not p.empty:
                total_pv += float(
                    pd.to_numeric(p["volume"], errors="coerce").fillna(0).sum()
                )
            uc, up = _unusual_for_expiry(c, p, str(exp), top_n)
            all_uc.extend(uc)
            all_up.extend(up)
        except Exception as e:
            logger.debug("option_chain %s %s: %s", sym, exp, e)
            continue

    out["put_call_ratio"] = (
        round(total_pv / total_cv, 4) if total_cv > 0 else 0.0
    )
    all_uc.sort(key=lambda x: x.get("volume", 0), reverse=True)
    all_up.sort(key=lambda x: x.get("volume", 0), reverse=True)
    out["unusual_calls"] = all_uc[:top_n]
    out["unusual_puts"] = all_up[:top_n]
    try:
        agg_calls = (
            pd.concat(agg_calls_frames, ignore_index=True)
            if agg_calls_frames
            else pd.DataFrame()
        )
        agg_puts = (
            pd.concat(agg_puts_frames, ignore_index=True)
            if agg_puts_frames
            else pd.DataFrame()
        )
        out["max_pain"] = _max_pain_strike(agg_calls, agg_puts)
    except Exception as e:
        logger.debug("max pain aggregate failed: %s", e)
        out["max_pain"] = 0.0

    uc_n = len(out["unusual_calls"])
    up_n = len(out["unusual_puts"])
    if uc_n > up_n * 1.25:
        out["net_flow"] = "BULLISH"
    elif up_n > uc_n * 1.25:
        out["net_flow"] = "BEARISH"
    else:
        pc = out["put_call_ratio"]
        if pc > 1.15:
            out["net_flow"] = "BEARISH"
        elif pc > 0 and pc < 0.85:
            out["net_flow"] = "BULLISH"
        else:
            out["net_flow"] = "NEUTRAL"

    out["success"] = True
    return out


def get_options_flow(symbol: str, top_n: int = 10) -> Dict[str, Any]:
    """
    Fetch options chains via yfinance and surface unusual volume vs expiry average.

    Returns:
        put_call_ratio, unusual_calls, unusual_puts, max_pain, net_flow,
        expiries, success/error.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _neutral_options(symbol)

    _cached = _get_options_cache(sym, top_n)
    if _cached is not None:
        return _cached

    try:
        import concurrent.futures as _cf

        with _cf.ThreadPoolExecutor(1) as _ex:
            _fut = _ex.submit(_fetch_options_data, sym, top_n)
            result = _fut.result(timeout=8)
        if result.get("success"):
            _set_options_cache(sym, top_n, result)
        return result
    except Exception as _e:
        logger.debug(
            "Options flow failed for "
            "%s: %s", sym, _e
        )
        return _neutral_options(sym)
