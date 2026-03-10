"""Streamlit watchlist widget for Evolve.

Renders a persistent, SQLite-backed watchlist with live prices, percentage
changes, RSI(14), and alert status using `WatchlistManager` from
`trading.data.watchlist`.
"""

from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from trading.data.watchlist import WatchlistManager
from trading.utils.safe_indicators import safe_rsi


def _fetch_price_and_rsi(symbols: Dict[str, None]) -> Dict[str, Dict[str, Optional[float]]]:
    """Fetch latest price and RSI(14) for each symbol."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if not symbols:
        return out
    # Fetch modest history for stable RSI
    for sym in symbols.keys():
        try:
            hist = yf.Ticker(sym).history(period="3mo")
            if hist is None or hist.empty:
                continue
            close = hist["Close"].dropna()
            if close.empty:
                continue
            price = float(close.iloc[-1])
            rsi_series = safe_rsi(close, period=14)
            rsi_val: Optional[float] = None
            if rsi_series is not None and len(rsi_series):
                try:
                    rsi_val = float(rsi_series.iloc[-1])
                except Exception:
                    rsi_val = None
            out[sym] = {"price": price, "rsi": rsi_val}
        except Exception:
            continue
    return out


def render_watchlist() -> None:
    """Render the main watchlist table and alert configuration UI."""
    mgr = WatchlistManager()

    st.markdown("Add tickers to monitor prices and RSI with optional alerts.")
    col_add1, col_add2 = st.columns([2, 3])
    with col_add1:
        sym_input = st.text_input("Symbol", placeholder="AAPL", key="watchlist_symbol_input")
    with col_add2:
        note_input = st.text_input("Note (optional)", key="watchlist_note_input")

    col_price_above, col_price_below = st.columns(2)
    with col_price_above:
        alert_above = st.number_input(
            "Alert if price ≥",
            min_value=0.0,
            value=0.0,
            step=0.5,
            format="%.2f",
            key="watchlist_price_above",
        )
    with col_price_below:
        alert_below = st.number_input(
            "Alert if price ≤",
            min_value=0.0,
            value=0.0,
            step=0.5,
            format="%.2f",
            key="watchlist_price_below",
        )

    col_rsi_low, col_rsi_high = st.columns(2)
    with col_rsi_low:
        rsi_below = st.number_input(
            "Alert if RSI ≤",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="watchlist_rsi_below",
        )
    with col_rsi_high:
        rsi_above = st.number_input(
            "Alert if RSI ≥",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="watchlist_rsi_above",
        )

    if st.button("Add / Update Ticker", type="primary", key="watchlist_add_button"):
        sym = (sym_input or "").strip().upper()
        if not sym:
            st.warning("Please enter a symbol.")
        else:
            mgr.add_ticker(
                sym,
                alert_price_above=alert_above or None,
                alert_price_below=alert_below or None,
                alert_rsi_below=rsi_below or None,
                alert_rsi_above=rsi_above or None,
                note=note_input or "",
            )
            st.success(f"Watchlist updated for {sym}")

    rows = mgr.get_all()
    if not rows:
        st.info("Your watchlist is empty. Add a ticker above to start tracking it.")
        return

    # Fetch current prices/RSI for status display
    symbols = {row["symbol"]: None for row in rows}
    live = _fetch_price_and_rsi(symbols)

    table_data = []
    for row in rows:
        sym = row["symbol"]
        info = live.get(sym, {})
        price = info.get("price")
        rsi_val = info.get("rsi")
        prev_close = None
        change_pct = None
        try:
            hist = yf.Ticker(sym).history(period="2d")
            if hist is not None and len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                if price is None:
                    price = float(hist["Close"].iloc[-1])
        except Exception:
            pass
        if price is not None and prev_close is not None and prev_close != 0:
            change_pct = (price / prev_close - 1.0) * 100.0

        status = "✅ OK"
        badge = "✅"
        above = row.get("alert_price_above")
        below = row.get("alert_price_below")
        rsi_low = row.get("alert_rsi_below")
        rsi_high = row.get("alert_rsi_above")

        if price is not None:
            if above is not None and price >= above:
                status = "🔴 TRIGGERED (price ≥ target)"
                badge = "🔴"
            elif below is not None and price <= below:
                status = "🔴 TRIGGERED (price ≤ target)"
                badge = "🔴"
            elif (
                above is not None
                and price >= above * 0.98
                or below is not None
                and price <= below * 1.02
            ):
                status = "🟡 NEAR price alert"
                badge = "🟡"

        if rsi_val is not None:
            if rsi_low is not None and rsi_val <= rsi_low:
                status = "🔴 TRIGGERED (RSI ≤ target)"
                badge = "🔴"
            elif rsi_high is not None and rsi_val >= rsi_high:
                status = "🔴 TRIGGERED (RSI ≥ target)"
                badge = "🔴"

        table_data.append(
            {
                "Symbol": sym,
                "Price": f"${price:.2f}" if price is not None else "—",
                "Change %": f"{change_pct:+.2f}%" if change_pct is not None else "—",
                "RSI(14)": f"{rsi_val:.1f}" if rsi_val is not None else "—",
                "Alert Status": status,
                "Note": row.get("note") or "",
                " ": badge,
            }
        )

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-row remove controls
    cols = st.columns(len(rows))
    for idx, row in enumerate(rows):
        sym = row["symbol"]
        with cols[idx]:
            if st.button(f"Remove {sym}", key=f"watchlist_remove_{sym}"):
                mgr.remove_ticker(sym)
                st.experimental_rerun()

