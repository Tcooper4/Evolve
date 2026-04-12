# -*- coding: utf-8 -*-
"""Price chart and short-term / value signal strip for Analyze."""
import logging

import plotly.graph_objects as go
import streamlit as st

from components.analyze_common import _is_english
from components.news_candle_chart import render_news_candle_chart
from trading.data.price_cache import get_history, get_info, get_quote

logger = logging.getLogger(__name__)


def _intraday_rangebreaks_and_dtick(
    hist,
    ticker: str = "",
) -> tuple:
    """
    Returns (rangebreaks, dtick).
    For 24hr assets (crypto/forex): weekends only.
    For regular equities: weekends + overnight session gaps.
    """
    _24hr_suffixes = (
        "-USD", "-EUR", "=X", "=F",
    )
    t_up = str(ticker).upper()
    _is_24hr = any(
        t_up.endswith(s) for s in _24hr_suffixes
    ) or str(ticker).startswith("^")

    rb = [dict(bounds=["sat", "mon"])]
    if not _is_24hr:
        rb.append(dict(bounds=[16, 9.5], pattern="hour"))

    dtick = None
    if hist is not None and len(hist) >= 2:
        try:
            _td = hist.index[1] - hist.index[0]
            _secs = float(abs(_td.total_seconds()))
            if _secs >= 60.0:
                dtick = _secs * 1000.0
        except Exception:
            pass
    return rb, dtick


def render_price_chart(
    ticker: str,
    hist,
    *,
    period: str,
    period_label: str,
    _interval: str,
    _tf_label: str,
    trader_mode: str,
    st_ver: tuple,
) -> None:
    """Full chart UI (candle/line/area/news, intraday live fragment, pattern detector)."""
    if hist is None or hist.empty:
        st.caption("No price history for this symbol and period.")
        return
    try:
        chart_type = st.radio(
            "Chart type",
            ["Candle", "Line", "Area", "News + Volume"],
            horizontal=True,
            key="analyze_chart_type",
            index=0,
        )
        if chart_type == "News + Volume":
            with st.expander("News overlay settings", expanded=True):
                _news_lang = st.selectbox(
                    "News language",
                    ["English only", "All languages"],
                    index=0,
                    key="news_lang_filter",
                )
                _news_vol_thresh = st.slider(
                    "Min price move % to show news",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    key="news_price_threshold",
                    help=(
                        "Only show news near bars where price moved at least this much"
                    ),
                )
            try:
                _nc_interval = (
                    "5m" if period == "1d"
                    else "1h" if period == "5d"
                    else "1d"
                )
                render_news_candle_chart(
                    symbol=ticker,
                    period=period,
                    interval=_nc_interval,
                    volume_threshold=1.5,
                    price_threshold=0.02,
                    show_annotations=True,
                )
                try:
                    from trading.analysis.chart_pattern_detector import (
                        ChartPatternDetector,
                    )
                    with st.expander(
                        "🔍 Chart Patterns & Support/Resistance",
                        expanded=False,
                    ):
                        if hist is not None and not hist.empty:
                            detector = ChartPatternDetector(
                                ticker, hist
                            )
                            detector.render_streamlit()
                        else:
                            st.caption(
                                "Load a symbol to detect patterns."
                            )
                except Exception as e:
                    st.caption(
                        f"Pattern detection unavailable: {e}"
                    )
            except Exception as e:
                st.caption(f"News chart unavailable: {e}")
        else:
            if period in ("1d", "5d"):
                from plotly.subplots import make_subplots as _make_subplots

                fig_chart = _make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.75, 0.25],
                )
                _intraday_mode = True
            else:
                fig_chart = go.Figure()
                _intraday_mode = False

            if _intraday_mode:
                if chart_type == "Candle":
                    fig_chart.add_trace(
                        go.Candlestick(
                            x=hist.index,
                            open=hist["Open"],
                            high=hist["High"],
                            low=hist["Low"],
                            close=hist["Close"],
                            increasing_line_color="#26a69a",
                            increasing_fillcolor="#26a69a",
                            decreasing_line_color="#ef5350",
                            decreasing_fillcolor="#ef5350",
                            name=ticker,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )
                else:
                    fig_chart.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist["Close"],
                            mode="lines",
                            line=dict(color="#00d4ff", width=1.5),
                            name=ticker,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                # VWAP
                try:
                    _vwap = (
                        (hist["Close"] * hist["Volume"]).cumsum()
                        / hist["Volume"].cumsum()
                    )
                    fig_chart.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=_vwap,
                            mode="lines",
                            line=dict(
                                color="#ff9800",
                                width=1,
                                dash="dot",
                            ),
                            name="VWAP",
                        ),
                        row=1,
                        col=1,
                    )
                except Exception:
                    pass

                # Volume bars
                try:
                    _vcols = [
                        "#26a69a"
                        if hist["Close"].iloc[i] >= hist["Open"].iloc[i]
                        else "#ef5350"
                        for i in range(len(hist))
                    ]
                    fig_chart.add_trace(
                        go.Bar(
                            x=hist.index,
                            y=hist["Volume"],
                            marker_color=_vcols,
                            name="Volume",
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
                except Exception:
                    pass

                # Previous close
                try:
                    import yfinance as _yf2

                    _prev = _yf2.Ticker(ticker).history(
                        period="2d", interval="1d"
                    )
                    if len(_prev) >= 2:
                        _pc = float(_prev["Close"].iloc[-2])
                        fig_chart.add_hline(
                            y=_pc,
                            line_dash="dot",
                            line_color="#4a6080",
                            line_width=1,
                            annotation_text="prev close",
                            annotation_font_color="#4a6080",
                            annotation_font_size=10,
                            row=1,
                            col=1,
                        )
                except Exception as _e:
                    logger.warning(
                        "Analyze: previous close yfinance fetch failed: %s", _e
                    )

                # Current price hline (intraday)
                try:
                    _current = float(hist["Close"].dropna().iloc[-1])
                    if _current and _current == _current:
                        fig_chart.add_hline(
                            y=_current,
                            line_dash="dash",
                            line_color="#ef5350",
                            line_width=1,
                            annotation_text=f" ${_current:.2f}",
                            annotation_position="right",
                            annotation_font_color="#ef5350",
                            annotation_font_size=11,
                            row=1,
                            col=1,
                        )
                except Exception:
                    pass

                _rb, _dtick = _intraday_rangebreaks_and_dtick(
                    hist,
                    ticker=ticker,
                )
                _x1 = dict(
                    gridcolor="#1a2535",
                    showgrid=True,
                    zeroline=False,
                    tickfont=dict(color="#4a6080", size=10),
                    rangeslider=dict(visible=False),
                    rangebreaks=(
                        _rb if period in ("1d", "5d") else [dict(bounds=["sat", "mon"])]
                    ),
                )
                if _dtick and period in ("1d", "5d"):
                    _x1["dtick"] = _dtick
                _x2 = dict(
                    gridcolor="#1a2535",
                    showgrid=True,
                    zeroline=False,
                    tickfont=dict(color="#4a6080", size=10),
                    rangebreaks=(
                        _rb if period in ("1d", "5d") else [dict(bounds=["sat", "mon"])]
                    ),
                )
                if _dtick and period in ("1d", "5d"):
                    _x2["dtick"] = _dtick

                fig_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0f1525",
                    font=dict(
                        family="'Courier New',monospace",
                        color="#e0e6f0",
                        size=11,
                    ),
                    title=dict(
                        text=f"{ticker} — {period_label} ({_tf_label})",
                        font=dict(color="#e0e6f0", size=14),
                        x=0,
                    ),
                    xaxis=_x1,
                    xaxis2=_x2,
                    yaxis=dict(
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        tickprefix="$",
                        side="right",
                    ),
                    yaxis2=dict(
                        gridcolor="#1a2535",
                        showgrid=False,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=9),
                        side="right",
                    ),
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="#0f1525",
                        bordercolor="#1e2d45",
                        font=dict(color="#e0e6f0", size=11),
                    ),
                    margin=dict(l=0, r=60, t=30, b=20),
                    height=450,
                    showlegend=True,
                    legend=dict(
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#4a6080", size=10),
                        x=0,
                        y=1.0,
                    ),
                )
            else:
                if chart_type == "Candle":
                    fig_chart = go.Figure(
                        data=[
                            go.Candlestick(
                                x=hist.index,
                                open=hist["Open"],
                                high=hist["High"],
                                low=hist["Low"],
                                close=hist["Close"],
                                increasing_line_color="#26a69a",
                                increasing_fillcolor="#26a69a",
                                decreasing_line_color="#ef5350",
                                decreasing_fillcolor="#ef5350",
                                name=ticker,
                            )
                        ]
                    )
                elif chart_type == "Area":
                    fig_chart = go.Figure(
                        data=[
                            go.Scatter(
                                x=hist.index,
                                y=hist["Close"],
                                fill="tozeroy",
                                fillcolor="rgba(0,212,255,0.1)",
                                line=dict(color="#00d4ff", width=2),
                                name=ticker,
                            )
                        ]
                    )
                else:  # Line
                    fig_chart = go.Figure(
                        data=[
                            go.Scatter(
                                x=hist.index,
                                y=hist["Close"],
                                mode="lines",
                                line=dict(color="#00d4ff", width=2),
                                name=ticker,
                            )
                        ]
                    )
                try:
                    _current = float(hist["Close"].dropna().iloc[-1])
                    if _current and _current == _current:
                        fig_chart.add_hline(
                            y=_current,
                            line_dash="dash",
                            line_color="#ef5350",
                            line_width=1,
                            annotation_text=f" ${_current:.2f}",
                            annotation_position="right",
                            annotation_font_color="#ef5350",
                            annotation_font_size=11,
                        )
                except Exception:
                    pass
                fig_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0f1525",
                    font=dict(
                        family="'Courier New', monospace",
                        color="#e0e6f0",
                        size=11,
                    ),
                    title=dict(
                        text=f"{ticker} — {period_label}",
                        font=dict(color="#e0e6f0", size=14),
                        x=0,
                    ),
                    xaxis=dict(
                        title="Date",
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        rangeslider=dict(visible=False),
                        rangebreaks=(
                            [
                                dict(bounds=["sat", "mon"]),
                                dict(bounds=[16, 9.5], pattern="hour"),
                            ]
                            if period in ("1d", "5d")
                            else [dict(bounds=["sat", "mon"])]
                        ),
                    ),
                    yaxis=dict(
                        title="Price ($)",
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        tickprefix="$",
                        side="right",
                    ),
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="#0f1525",
                        bordercolor="#1e2d45",
                        font=dict(color="#e0e6f0", size=11),
                    ),
                    margin=dict(l=0, r=60, t=30, b=20),
                    height=380,
                    showlegend=False,
                )

            if st_ver >= (1, 37) and period in ("1d", "5d"):

                @st.fragment(run_every=60)
                def _live_chart():
                    _tkr = st.session_state.get("analyze_ticker", ticker)
                    _per = period
                    _tf = st.session_state.get(
                        "analyze_intraday_tf",
                        "5m" if _per == "1d" else "30m",
                    )
                    _ct = st.session_state.get("analyze_chart_type", "Candle")

                    _hist = get_history(_tkr, period=_per, interval=_tf)
                    if _hist.empty:
                        st.caption("No intraday data.")
                        return

                    import datetime as _dt

                    _now = _dt.datetime.now().strftime("%H:%M:%S")
                    st.caption(f"Last updated: {_now}")

                    from plotly.subplots import make_subplots as _make_subplots

                    _fig = _make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.75, 0.25],
                    )

                    if _ct == "Candle":
                        _fig.add_trace(
                            go.Candlestick(
                                x=_hist.index,
                                open=_hist["Open"],
                                high=_hist["High"],
                                low=_hist["Low"],
                                close=_hist["Close"],
                                increasing_line_color="#26a69a",
                                increasing_fillcolor="#26a69a",
                                decreasing_line_color="#ef5350",
                                decreasing_fillcolor="#ef5350",
                                name=_tkr,
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )
                    else:
                        _fig.add_trace(
                            go.Scatter(
                                x=_hist.index,
                                y=_hist["Close"],
                                mode="lines",
                                line=dict(color="#00d4ff", width=1.5),
                                name=_tkr,
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )

                    try:
                        _vwap_live = (
                            (_hist["Close"] * _hist["Volume"]).cumsum()
                            / _hist["Volume"].cumsum()
                        )
                        _fig.add_trace(
                            go.Scatter(
                                x=_hist.index,
                                y=_vwap_live,
                                mode="lines",
                                line=dict(
                                    color="#ff9800",
                                    width=1,
                                    dash="dot",
                                ),
                                name="VWAP",
                            ),
                            row=1,
                            col=1,
                        )
                    except Exception:
                        pass

                    try:
                        _vcols_live = [
                            "#26a69a"
                            if _hist["Close"].iloc[i] >= _hist["Open"].iloc[i]
                            else "#ef5350"
                            for i in range(len(_hist))
                        ]
                        _fig.add_trace(
                            go.Bar(
                                x=_hist.index,
                                y=_hist["Volume"],
                                marker_color=_vcols_live,
                                name="Volume",
                                showlegend=False,
                            ),
                            row=2,
                            col=1,
                        )
                    except Exception:
                        pass

                    try:
                        import yfinance as _yf_live

                        _prev_live = _yf_live.Ticker(_tkr).history(
                            period="2d", interval="1d"
                        )
                        if len(_prev_live) >= 2:
                            _pc_live = float(_prev_live["Close"].iloc[-2])
                            _fig.add_hline(
                                y=_pc_live,
                                line_dash="dot",
                                line_color="#4a6080",
                                line_width=1,
                                annotation_text="prev close",
                                annotation_font_color="#4a6080",
                                annotation_font_size=10,
                                row=1,
                                col=1,
                            )
                    except Exception as _e:
                        logger.warning(
                            "Analyze: intraday prev close fetch failed: %s", _e
                        )

                    try:
                        _current_live = float(_hist["Close"].dropna().iloc[-1])
                        if _current_live and _current_live == _current_live:
                            _fig.add_hline(
                                y=_current_live,
                                line_dash="dash",
                                line_color="#ef5350",
                                line_width=1,
                                annotation_text=f" ${_current_live:.2f}",
                                annotation_position="right",
                                annotation_font_color="#ef5350",
                                annotation_font_size=11,
                                row=1,
                                col=1,
                            )
                    except Exception:
                        pass

                    _rb_l, _dtick_l = _intraday_rangebreaks_and_dtick(
                        _hist,
                        ticker=ticker,
                    )
                    _xa_l = dict(
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        rangeslider=dict(visible=False),
                        rangebreaks=_rb_l,
                    )
                    if _dtick_l:
                        _xa_l["dtick"] = _dtick_l
                    _xa2_l = dict(
                        gridcolor="#1a2535",
                        showgrid=True,
                        zeroline=False,
                        tickfont=dict(color="#4a6080", size=10),
                        rangebreaks=_rb_l,
                    )
                    if _dtick_l:
                        _xa2_l["dtick"] = _dtick_l

                    _fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#0a0e1a",
                        plot_bgcolor="#0f1525",
                        font=dict(
                            family="'Courier New',monospace",
                            color="#e0e6f0",
                            size=11,
                        ),
                        title=dict(
                            text=f"{_tkr} — {_per} ({_tf})",
                            font=dict(color="#e0e6f0", size=14),
                            x=0,
                        ),
                        xaxis=_xa_l,
                        xaxis2=_xa2_l,
                        yaxis=dict(
                            gridcolor="#1a2535",
                            showgrid=True,
                            zeroline=False,
                            tickfont=dict(color="#4a6080", size=10),
                            tickprefix="$",
                            side="right",
                        ),
                        yaxis2=dict(
                            gridcolor="#1a2535",
                            showgrid=False,
                            zeroline=False,
                            tickfont=dict(color="#4a6080", size=9),
                            side="right",
                        ),
                        hovermode="x unified",
                        hoverlabel=dict(
                            bgcolor="#0f1525",
                            bordercolor="#1e2d45",
                            font=dict(color="#e0e6f0", size=11),
                        ),
                        margin=dict(l=0, r=60, t=30, b=20),
                        height=380,
                        showlegend=False,
                    )

                    st.plotly_chart(
                        _fig,
                        width='stretch',
                        key="analyze_main_chart_live",
                    )

                _live_chart()
                try:
                    from trading.analysis.chart_pattern_detector import (
                        ChartPatternDetector,
                    )
                    with st.expander(
                        "🔍 Chart Patterns & Support/Resistance",
                        expanded=False,
                    ):
                        if hist is not None and not hist.empty:
                            detector = ChartPatternDetector(
                                ticker, hist
                            )
                            detector.render_streamlit()
                        else:
                            st.caption(
                                "Load a symbol to detect patterns."
                            )
                except Exception as e:
                    st.caption(
                        f"Pattern detection unavailable: {e}"
                    )
            else:
                # News vlines on Candle/Line/Area when using static chart (no live
                # intraday fragment). Uses same session keys as News+Vol settings.
                try:
                    from trading.data.price_cache import get_news

                    _news_items = get_news(ticker)
                    _lang_filter = st.session_state.get(
                        "news_lang_filter", "English only"
                    )
                    _price_thresh = st.session_state.get(
                        "news_price_threshold", 0.5
                    )
                    # Scale threshold by period —
                    # intraday bars rarely move 0.5%
                    _period_thresh = {
                        "1d": 0.15,
                        "5d": 0.25,
                        "1mo": 0.5,
                        "3mo": 1.0,
                        "6mo": 1.5,
                        "1y": 2.0,
                        "5y": 3.0,
                    }
                    _price_thresh = _period_thresh.get(
                        period, _price_thresh)
                    _plotted = 0
                    if _news_items and not hist.empty:
                        from datetime import datetime

                        _prices = hist["Close"].dropna()

                        for _item in _news_items[:15]:
                            _content = _item.get("content") or {}
                            _title = (
                                _item.get("title")
                                or _content.get("title")
                                or _content.get("summary")
                                or ""
                            )
                            if not _title:
                                continue

                            if (
                                _lang_filter == "English only"
                                and not _is_english(_title)
                            ):
                                continue

                            _pub = (
                                _item.get("providerPublishTime")
                                or _content.get("pubDate")
                                or _item.get("published")
                            )
                            if not _pub:
                                continue
                            try:
                                if isinstance(_pub, str):
                                    _dt = datetime.fromisoformat(
                                        _pub.replace("Z", "+00:00")
                                    )
                                    _pub_dt = _dt.replace(tzinfo=None)
                                    _pub_ts = _dt.timestamp()
                                else:
                                    _pub_ts = float(_pub)
                                    _pub_dt = datetime.fromtimestamp(
                                        _pub_ts
                                    )
                            except Exception:
                                continue

                            if len(hist.index) > 0:
                                _idx_min = hist.index.min()
                                _idx_max = hist.index.max()
                                try:
                                    _idx_min = _idx_min.replace(tzinfo=None)
                                    _idx_max = _idx_max.replace(tzinfo=None)
                                except Exception:
                                    pass
                                if _pub_dt < _idx_min or _pub_dt > _idx_max:
                                    continue

                            if _price_thresh > 0:
                                try:
                                    if hasattr(hist.index, "tz"):
                                        _index_dt = (
                                            hist.index.tz_localize(
                                                None
                                            ).to_pydatetime()
                                        )
                                    else:
                                        _index_dt = (
                                            hist.index.to_pydatetime()
                                        )
                                    _diffs = abs(_index_dt - _pub_dt)
                                    _closest_idx = int(_diffs.argmin())
                                    if _closest_idx > 0:
                                        _p1 = float(
                                            _prices.iloc[_closest_idx]
                                        )
                                        _p0 = float(
                                            _prices.iloc[_closest_idx - 1]
                                        )
                                        if _p0 != 0:
                                            _move = abs(
                                                (_p1 - _p0) / _p0 * 100
                                            )
                                            if _move < _price_thresh:
                                                continue
                                except Exception:
                                    pass

                            _pos_kw = [
                                "beat",
                                "surge",
                                "raises",
                                "upgrade",
                                "strong",
                                "growth",
                            ]
                            _neg_kw = [
                                "miss",
                                "falls",
                                "cuts",
                                "downgrade",
                                "weak",
                                "loss",
                            ]
                            _tl = _title.lower()
                            _pos = sum(
                                1 for k in _pos_kw if k in _tl
                            )
                            _neg = sum(
                                1 for k in _neg_kw if k in _tl
                            )
                            _ann_color = (
                                "#26a69a"
                                if _pos > _neg
                                else "#ef5350"
                                if _neg > _pos
                                else "#ff9800"
                            )

                            _ann_text = (
                                _title[:30] + "…"
                                if len(_title) > 30
                                else _title
                            )
                            _vline_kwargs = dict(
                                x=_pub_dt,
                                line_dash="dot",
                                line_color=_ann_color,
                                line_width=1,
                                annotation_text=_ann_text,
                                annotation_position="top",
                                annotation_font_color=_ann_color,
                                annotation_font_size=10,
                            )
                            if _intraday_mode:
                                fig_chart.add_vline(
                                    row=1, col=1, **_vline_kwargs
                                )
                            else:
                                fig_chart.add_vline(**_vline_kwargs)
                            _plotted += 1

                    if _plotted == 0 and period in ("1d", "5d"):
                        st.caption(
                            "No recent English news found within chart timeframe."
                        )
                except Exception:
                    pass

                st.plotly_chart(
                    fig_chart,
                    width='stretch',
                    key="analyze_main_chart",
                )
                try:
                    from trading.analysis.chart_pattern_detector import (
                        ChartPatternDetector,
                    )
                    with st.expander(
                        "🔍 Chart Patterns & Support/Resistance",
                        expanded=False,
                    ):
                        if hist is not None and not hist.empty:
                            detector = ChartPatternDetector(
                                ticker, hist
                            )
                            detector.render_streamlit()
                        else:
                            st.caption(
                                "Load a symbol to detect patterns."
                            )
                except Exception as e:
                    st.caption(
                        f"Pattern detection unavailable: {e}"
                    )
        if trader_mode == "Short-term":
            try:
                st.markdown("**Short-term Signals**")
                _hist_1d = get_history(ticker, period="1d", interval="5m")
                if not _hist_1d.empty:
                    _close = _hist_1d["Close"]
                    _vol_s = _hist_1d["Volume"]
                    _vwap = (_hist_1d["Close"] * _vol_s).cumsum() / _vol_s.cumsum()
                    _last = float(_close.iloc[-1])
                    _vwap_last = float(_vwap.iloc[-1])
                    _vwap_dev = None
                    try:
                        import math as _math

                        if _math.isfinite(_vwap_last) and _vwap_last != 0.0:
                            _vwap_dev = float(
                                (_last / _vwap_last - 1) * 100,
                            )
                            if not _math.isfinite(_vwap_dev):
                                _vwap_dev = None
                    except Exception:
                        _vwap_dev = None
                    _vol_avg = float(_hist_1d["Volume"].mean())
                    _vol_last = float(_hist_1d["Volume"].iloc[-1])
                    _vol_ratio = _vol_last / _vol_avg if _vol_avg > 0 else 1.0
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if _vwap_dev is not None:
                            _sign = "+" if _vwap_dev >= 0 else ""
                            _vwap_disp = f"{_sign}{_vwap_dev:.2f}%"
                        else:
                            _vwap_disp = "N/A"
                        st.metric(
                            "vs VWAP",
                            _vwap_disp,
                            help="Price deviation from Volume Weighted Avg Price",
                        )
                    with c2:
                        st.metric(
                            "Volume Ratio",
                            f"{_vol_ratio:.1f}x",
                            help="Current bar volume vs average intraday volume",
                        )
                    with c3:
                        _momentum = float(_close.pct_change(5).iloc[-1] * 100) if len(_close) > 5 else 0.0
                        _sign2 = "+" if _momentum >= 0 else ""
                        st.metric(
                            "5-bar Momentum",
                            f"{_sign2}{_momentum:.2f}%",
                            help="Price change over last 5 intraday bars",
                        )
            except Exception as _e:
                logger.warning(
                    "Analyze: short-term signals failed to compute/load: %s", _e
                )
        elif trader_mode == "Long-term":
            try:
                st.markdown("**Value Signals**")
                _info = get_info(ticker)
                if _info:
                    _pe = _info.get("trailingPE") or _info.get("forwardPE")
                    _pb = _info.get("priceToBook")
                    _eps = _info.get("trailingEps") or _info.get("forwardEps")
                    _div = _info.get("dividendYield", 0) or 0
                    _52w_low = _info.get("fiftyTwoWeekLow")
                    _52w_high = _info.get("fiftyTwoWeekHigh")
                    _price = get_quote(ticker).get("price", 0)
                    _range_pct = None
                    if _52w_low and _52w_high and _price:
                        _range_pct = (_price - _52w_low) / (_52w_high - _52w_low) * 100
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "P/E Ratio",
                            f"{_pe:.1f}x" if _pe else "N/A",
                            help="Lower = cheaper relative to earnings",
                        )
                    with c2:
                        st.metric(
                            "P/B Ratio",
                            f"{_pb:.2f}x" if _pb else "N/A",
                            help="Below 1.0 may indicate undervaluation",
                        )
                    with c3:
                        st.metric(
                            "Div Yield",
                            f"{_div * 100:.2f}%" if _div else "N/A",
                        )
                    with c4:
                        if _range_pct is not None:
                            st.metric(
                                "52w Position",
                                f"{_range_pct:.0f}%",
                                help="Where price sits in 52-week range. Low % = near lows",
                            )
                        else:
                            st.metric("52w Position", "N/A")
                    _signals = []
                    if _pe and _pe < 15:
                        _signals.append("P/E below 15")
                    if _pb and _pb < 1.5:
                        _signals.append("P/B below 1.5")
                    if _range_pct is not None and _range_pct < 25:
                        _signals.append("Near 52-week low")
                    if _signals:
                        st.success("Value signals: " + ", ".join(_signals))
            except Exception as _e:
                logger.warning(
                    "Analyze: long-term value signals failed to compute/load: %s",
                    _e,
                )
    except Exception as e:
        st.caption(f"unavailable: {e}")
