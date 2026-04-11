# -*- coding: utf-8 -*-
"""Analyze page — correlation analysis (Research section)."""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trading.data.price_cache import get_history
from trading.data.ticker_resolver import resolve_ticker

logger = logging.getLogger(__name__)


def _default_tickers(ticker: str) -> str:
    base = (ticker or "AAPL").strip().upper() or "AAPL"
    pool = ["MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    lines = [base]
    for p in pool:
        if p != base and p not in lines:
            lines.append(p)
        if len(lines) >= 5:
            break
    return "\n".join(lines[:5])


def _parse_tickers(text: str) -> List[str]:
    out: List[str] = []
    for line in (text or "").splitlines():
        s = line.strip().upper()
        if not s:
            continue
        try:
            out.append(resolve_ticker(s, validate=False))
        except Exception as e:
            logger.debug("resolve_ticker %s: %s", s, e)
            out.append(s)
    seen = set()
    uniq: List[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    cm = {c.lower(): c for c in df.columns}
    col = cm.get("close", df.columns[0])
    return pd.to_numeric(df[col], errors="coerce")


def _strength_label(r: float) -> str:
    if r < -0.7:
        return "🔴 Strong inverse"
    if abs(r) < 0.3:
        return "🟡 Weak/uncorrelated"
    if r > 0.7:
        return "🟢 Strongly correlated"
    return "🔵 Moderately correlated"


def _rank_pairs(corr: pd.DataFrame) -> List[Tuple[str, str, float]]:
    cols = list(corr.columns)
    n = len(cols)
    pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def render(
    ticker: str,
    hist,
    period: str,
    period_label: str,
    trader_mode: str,
    _interval: str,
    _tf_label: str,
    *,
    score_mode: str = "Buy",
    scoring_style: str = "Balanced (default)",
    backend: dict,
) -> None:
    del backend
    st.markdown("---")
    st.header("📊 Correlation Analysis")
    st.caption(
        "Discover which assets move together — useful for hedging, "
        "pair trading, and risk management."
    )

    sym_input = st.text_area(
        "Tickers (one per line)",
        value=_default_tickers(ticker),
        height=140,
        key="causal_corr_tickers_input",
        help="Symbols are normalized with resolve_ticker.",
    )

    if st.button("Load Correlation Data", key="causal_load_corr_btn"):
        tickers = _parse_tickers(sym_input)
        if len(tickers) < 2:
            st.caption("Enter at least two tickers to build a correlation matrix.")
        else:
            closes = pd.DataFrame()
            errs: List[str] = []
            for t in tickers:
                try:
                    raw = get_history(t, period="1y", interval="1d")
                    s = _close_series(raw)
                    if s.empty:
                        errs.append(t)
                        continue
                    closes[t] = s
                except Exception as e:
                    errs.append(f"{t}")
                    logger.warning("Correlation tab: history failed for %s: %s", t, e)
                    st.caption(f"Fetch issue for {t}: {e}")
            if errs and closes.shape[1] < 2:
                st.caption(
                    "Could not load enough series. Check symbols and try again."
                )
            if closes.shape[1] < 2:
                st.caption("Need at least two valid price series after fetch.")
            else:
                try:
                    closes = closes.sort_index().dropna(how="any", axis=0)
                    if len(closes) < 30:
                        st.caption(
                            "Not enough overlapping daily rows for a stable correlation."
                        )
                    else:
                        rets = closes.pct_change().dropna(how="any")
                        corr = rets.corr()
                        st.session_state["causal_corr_matrix"] = corr
                        st.session_state["causal_corr_tickers"] = list(corr.columns)
                        st.session_state["causal_corr_rets"] = rets
                        st.success(
                            f"Loaded {corr.shape[0]} assets, {len(rets)} return rows."
                        )
                except Exception as e:
                    st.caption(f"Correlation build failed: {e}")
                    logger.warning("causal_corr build: %s", e)

    corr = st.session_state.get("causal_corr_matrix")
    rets = st.session_state.get("causal_corr_rets")

    if corr is None or not isinstance(corr, pd.DataFrame):
        return

    pairs = _rank_pairs(corr)

    st.markdown("### Section A — Heatmap")
    try:
        fig_h = px.imshow(
            corr,
            labels=dict(x="Asset", y="Asset", color="Correlation"),
            x=list(corr.columns),
            y=list(corr.columns),
            color_continuous_scale="RdYlGn",
            zmin=-1.0,
            zmax=1.0,
            aspect="auto",
            title="Return Correlation Matrix (1-Year Daily Returns)",
        )
        try:
            fig_h.update_traces(
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        except Exception:
            try:
                fig_h.update_traces(texttemplate="%{z:.2f}")
            except Exception:
                pass
        fig_h.update_layout(height=520)
        st.plotly_chart(fig_h, width="stretch")
    except Exception as e:
        st.caption(f"Heatmap unavailable: {e}")

    st.markdown("### Section B — Top pairs")
    try:
        top = pairs[:15]
        tab_df = pd.DataFrame(
            [
                {
                    "Asset A": a,
                    "Asset B": b,
                    "Correlation": round(r, 4),
                    "Strength": _strength_label(r),
                }
                for a, b, r in top
            ]
        )
        st.dataframe(tab_df, width="stretch", hide_index=True)
        st.caption(
            "Correlations above 0.7 suggest the assets tend to move together. "
            "Below -0.7 suggests they move in opposite directions — useful for hedging."
        )
    except Exception as e:
        st.caption(f"Pairs table unavailable: {e}")

    st.markdown("### Section C — Rolling correlation")
    if isinstance(rets, pd.DataFrame) and not rets.empty and len(pairs) > 0:
        try:
            fig_r = go.Figure()
            for a, b, _ in pairs[:2]:
                if a not in rets.columns or b not in rets.columns:
                    continue
                roll = rets[a].rolling(30).corr(rets[b])
                fig_r.add_trace(
                    go.Scatter(
                        x=roll.index,
                        y=roll.values,
                        mode="lines",
                        name=f"{a} vs {b}",
                    )
                )
            fig_r.update_layout(
                title="30-Day Rolling Correlation",
                xaxis_title="Date",
                yaxis_title="Correlation",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig_r, width="stretch")
            st.caption(
                "Correlation changes over time — a rising line means the assets are converging."
            )
        except Exception as e:
            st.caption(f"Rolling correlation unavailable: {e}")
    else:
        st.caption("Load correlation data to view rolling correlations.")
