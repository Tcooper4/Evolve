"""
Multi-timeframe candlestick chart component.
Shows 1D, 1W, and 1M views side by side with key indicators overlaid.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from typing import Optional


def render_multi_timeframe_chart(
    symbol: str,
    hist_daily: Optional[pd.DataFrame] = None,
    show_volume: bool = True,
    show_sma: bool = True,
    height: int = 500,
) -> None:
    """
    Render a 3-panel multi-timeframe candlestick chart.

    Panels: 3-month daily | 1-year weekly | 5-year monthly
    Each panel shows: OHLC candles, SMA20, SMA50, volume bars
    """
    try:
        ticker = yf.Ticker(symbol)

        if hist_daily is not None and not hist_daily.empty:
            daily = hist_daily.copy()
        else:
            daily = ticker.history(period="3mo", interval="1d")

        weekly = ticker.history(period="1y", interval="1wk")
        monthly = ticker.history(period="5y", interval="1mo")

        if daily.empty and weekly.empty:
            st.warning(f"No price data available for {symbol}")
            return

        tab_labels = ["📅 3-Month Daily", "📆 1-Year Weekly", "🗓️ 5-Year Monthly"]
        tabs = st.tabs(tab_labels)

        for tab, (data, label) in zip(
            tabs,
            [(daily, "Daily"), (weekly, "Weekly"), (monthly, "Monthly")],
        ):
            with tab:
                if data is None or data.empty:
                    st.caption(f"No {label} data available")
                    continue
                _render_ohlc_panel(data, symbol, label, show_volume, show_sma, height)

    except Exception as e:
        st.error(f"Chart error: {e}")


def _render_ohlc_panel(
    data: pd.DataFrame,
    symbol: str,
    label: str,
    show_volume: bool,
    show_sma: bool,
    height: int,
) -> None:
    """Render a single OHLC panel with indicators."""
    data = data.copy()
    col_map = {}
    for std, alts in [
        ("Open", ["open"]),
        ("High", ["high"]),
        ("Low", ["low"]),
        ("Close", ["close", "Adj Close"]),
        ("Volume", ["volume"]),
    ]:
        if std not in data.columns:
            for alt in alts:
                if alt in data.columns:
                    col_map[alt] = std
    if col_map:
        data = data.rename(columns=col_map)

    required = ["Open", "High", "Low", "Close"]
    if not all(c in data.columns for c in required):
        st.caption(f"Missing OHLC columns for {label}")
        return

    close = data["Close"].values.astype(float)

    rows = 2 if show_volume and "Volume" in data.columns else 1
    row_heights = [0.75, 0.25] if rows == 2 else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=symbol,
            increasing_line_color="#00D4AA",
            decreasing_line_color="#FF4B4B",
        ),
        row=1,
        col=1,
    )

    if show_sma:
        for period, color, name in [(20, "#FFA500", "SMA20"), (50, "#4FC3F7", "SMA50")]:
            if len(close) >= period:
                sma = pd.Series(close).rolling(period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=sma,
                        mode="lines",
                        line=dict(color=color, width=1.2, dash="dot"),
                        name=name,
                        opacity=0.8,
                    ),
                    row=1,
                    col=1,
                )

    if show_volume and "Volume" in data.columns and rows == 2:
        colors = [
            "#00D4AA" if c >= o else "#FF4B4B"
            for c, o in zip(data["Close"], data["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

    last_close = float(close[-1])
    prev_close = float(close[-2]) if len(close) >= 2 else last_close
    chg = (last_close / prev_close - 1) * 100
    chg_str = f"{chg:+.2f}%"

    fig.update_layout(
        title=f"{symbol} — {label} | ${last_close:,.2f} {chg_str}",
        template="plotly_dark",
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    if rows == 2:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    if len(close) >= 2:
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("Last Close", f"${last_close:,.2f}", chg_str)
        if len(close) >= 20:
            sma20_val = float(np.mean(close[-20:]))
            _c2.metric(
                "SMA20",
                f"${sma20_val:,.2f}",
                f"{(last_close / sma20_val - 1) * 100:+.1f}%",
            )
        if len(close) >= 52:
            high_52 = float(np.max(close[-52:]))
            _c3.metric(
                "52w High",
                f"${high_52:,.2f}",
                f"{(last_close / high_52 - 1) * 100:+.1f}%",
            )
        vol_20 = (
            float(np.mean(data["Volume"].values[-20:]))
            if "Volume" in data.columns and len(data) >= 20
            else None
        )
        if vol_20:
            _c4.metric("Avg Vol (20d)", f"{vol_20 / 1e6:.1f}M")
