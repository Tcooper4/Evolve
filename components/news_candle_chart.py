"""
Interactive news-linked candlestick chart component.
Call render_news_candle_chart(symbol, period, interval) to display.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


@st.cache_data(ttl=300)  # 5-min cache
def _fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV history from yfinance with a short cache."""
    hist = yf.Ticker(symbol).history(period=period, interval=interval)
    if hist is None or hist.empty:
        return hist
    # Normalize index to naive timestamps for Plotly
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    return hist


def render_news_candle_chart(
    symbol: str = "AAPL",
    period: str = "3mo",
    interval: str = "1d",
    volume_threshold: float = 2.0,
    price_threshold: float = 0.02,
    show_annotations: bool = True,
) -> None:
    """
    Renders a full-width interactive candlestick + volume chart.

    Significant volume spikes are annotated with a 📰 marker.
    Hovering over 📰 shows relevant news headlines.
    """
    from trading.analysis.volume_news_linker import (
        detect_significant_candles,
        build_chart_annotations,
    )

    try:
        hist = _fetch_ohlcv(symbol, period, interval)
    except Exception as e:
        st.error(f"Could not fetch data for {symbol}: {e}")
        return

    if hist is None or len(hist) < 5:
        st.warning(f"Not enough data for {symbol} ({period} {interval})")
        return

    df = detect_significant_candles(hist, volume_threshold, price_threshold)

    # Build figure with 2 rows: candlestick + volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    # Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"] if "Open" in df.columns else df.get("open"),
            high=df["High"] if "High" in df.columns else df.get("high"),
            low=df["Low"] if "Low" in df.columns else df.get("low"),
            close=df["Close"] if "Close" in df.columns else df.get("close"),
            name=symbol,
            increasing_line_color="#00D4AA",
            decreasing_line_color="#FF4444",
        ),
        row=1,
        col=1,
    )

    # Volume bars — color by candle direction
    close_series = df["Close"] if "Close" in df.columns else df.get("close")
    open_series = df["Open"] if "Open" in df.columns else df.get("open", close_series)
    vol_series = df["Volume"] if "Volume" in df.columns else df.get("volume")

    vol_colors = [
        "#00D4AA" if (c is not None and o is not None and c >= o) else "#FF4444"
        for c, o in zip(close_series, open_series)
    ]

    # Highlight significant volume bars
    sig_flags = list(df.get("is_significant", pd.Series(False, index=df.index)))
    vol_colors_final = [
        "#FFD700" if sig else c  # gold for volume spikes
        for sig, c in zip(sig_flags, vol_colors)
    ]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=vol_series,
            name="Volume",
            marker_color=vol_colors_final,
            opacity=0.8,
        ),
        row=2,
        col=1,
    )

    # Add 20-day volume average line
    vol_avg = vol_series.rolling(20, min_periods=5).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vol_avg,
            mode="lines",
            name="Avg Volume",
            line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1),
        ),
        row=2,
        col=1,
    )

    # News annotation markers on the candlestick
    if show_annotations:
        try:
            annotations = build_chart_annotations(df, symbol, max_annotations=8)
            if annotations:
                ann_dates = [a["date"] for a in annotations]
                ann_prices = [a["price"] for a in annotations]
                ann_hovers = [a["hover"] for a in annotations]
                fig.add_trace(
                    go.Scatter(
                        x=ann_dates,
                        y=ann_prices,
                        mode="markers+text",
                        name="News Events",
                        text=["📰"] * len(annotations),
                        textposition="top center",
                        hovertext=ann_hovers,
                        hoverinfo="text",
                        marker=dict(
                            size=14,
                            color="rgba(255,215,0,0.8)",
                            symbol="diamond",
                        ),
                    ),
                    row=1,
                    col=1,
                )
        except Exception as e:
            st.caption(f"News annotations unavailable: {e}")

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=550,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        title=dict(text=f"{symbol} — News-Linked Chart", font=dict(size=16)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    # News panel below chart — show articles for selected event
    if show_annotations:
        try:
            sig_df = df[df.get("is_significant", False)].nlargest(
                5, "volume_ratio"
            )
            if not sig_df.empty:
                st.markdown("**Significant Events — Click to expand news**")
                from trading.analysis.volume_news_linker import get_news_for_date

                for date, row in sig_df.iterrows():
                    date_str = (
                        str(date.date()) if hasattr(date, "date") else str(date)[:10]
                    )
                    vol_r = row["volume_ratio"]
                    pct = row["price_change_pct"] * 100
                    icon = "🟢" if pct > 0 else "🔴"
                    label = (
                        f"{icon} {date_str} — Vol {vol_r:.1f}x avg, "
                        f"Price {pct:+.1f}%"
                    )
                    with st.expander(label, expanded=False):
                        news = get_news_for_date(symbol, date_str)
                        if news:
                            for article in news:
                                st.markdown(
                                    f"**[{article.get('source','')}]** "
                                    f"[{article.get('title','')}]({article.get('url','')})"
                                )
                                if article.get("summary"):
                                    st.caption(article["summary"][:200])
                        else:
                            st.caption(
                                "No news found for this date. Try a wider date range."
                            )
        except Exception as e:
            st.caption(f"Events panel error: {e}")

