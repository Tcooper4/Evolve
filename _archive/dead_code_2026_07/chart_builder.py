"""
Build Plotly candlestick + volume chart for a market event with spike annotation.
Shared by Home page (featured event) and Market Events page.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_event_chart(
    df: pd.DataFrame,
    symbol: str,
    spike_timestamp,
) -> go.Figure:
    """
    Build a 2-row Plotly chart: candlestick (top), volume (bottom).
    Adds a yellow vertical line and triangle marker at spike_timestamp.
    Uses plotly_dark template.
    """
    if df is None or len(df) == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title=f"{symbol} — No data")
        return fig

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} — Price", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], showlegend=False, marker_color="rgba(100,150,200,0.6)"),
        row=2,
        col=1,
    )

    # Align spike_timestamp with df index for vertical line
    try:
        ts = pd.Timestamp(spike_timestamp)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        elif df.index.tz is None and ts.tz is not None:
            ts = ts.tz_localize(None) if hasattr(ts, "tz_localize") else ts
        # Clamp to range
        if len(df) > 0 and ts >= df.index.min() and ts <= df.index.max():
            fig.add_vline(
                x=ts,
                line_dash="solid",
                line_color="yellow",
                line_width=2,
                row=1,
                col=1,
            )
            fig.add_vline(
                x=ts,
                line_dash="solid",
                line_color="yellow",
                line_width=1,
                opacity=0.5,
                row=2,
                col=1,
            )
            # Triangle marker at spike time (find closest close price)
            idx = df.index.get_indexer([ts], method="nearest")[0]
            close_val = df["Close"].iloc[idx]
            fig.add_trace(
                go.Scatter(
                    x=[ts],
                    y=[close_val],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=14, color="yellow", line=dict(width=1, color="orange")),
                    text=["Spike"],
                    textposition="top center",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
    except Exception:
        pass

    fig.update_layout(
        template="plotly_dark",
        title=f"{symbol} — Event",
        xaxis_rangeslider_visible=False,
        height=500,
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig
