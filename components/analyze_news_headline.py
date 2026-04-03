# -*- coding: utf-8 -*-
"""News headline sentiment strip + Deep Dive news."""
import streamlit as st
from trading.data.price_cache import get_news


def render_analyze_headline_news_panel(ticker: str) -> None:
    """Hot/pos/neg labels for recent headlines (legacy Analyze strip)."""
    try:
        news_items = get_news(ticker)
        if news_items:
            with st.expander("📰 News sentiment", expanded=False):
                pos_kw = [
                    "beat",
                    "surge",
                    "raises",
                    "upgrade",
                    "strong",
                    "growth",
                    "record",
                    "above",
                    "buy",
                    "bullish",
                ]
                neg_kw = [
                    "miss",
                    "falls",
                    "cuts",
                    "downgrade",
                    "weak",
                    "below",
                    "layoffs",
                    "investigation",
                    "sell",
                    "bearish",
                    "loss",
                ]
                for item in news_items[:8]:
                    content = item.get("content") or {}
                    raw_title = (
                        item.get("title")
                        or item.get("headline")
                        or content.get("title")
                        or content.get("summary")
                        or ""
                    )
                    if not raw_title:
                        continue
                    title_lower = raw_title.lower()
                    score = 0
                    score += sum(1 for k in pos_kw if k in title_lower)
                    score -= sum(1 for k in neg_kw if k in title_lower)
                    if score >= 3:
                        ns_label, ns_color = "HOT", "#ff9800"
                    elif score >= 1:
                        ns_label, ns_color = "POS", "#26a69a"
                    elif score <= -1:
                        ns_label, ns_color = "NEG", "#ef5350"
                    else:
                        ns_label, ns_color = "NEU", "#4a6080"
                    st.markdown(
                        f'<span style="color:{ns_color};font-size:10px;'
                        f"padding:1px 6px;border-radius:2px;"
                        f"background:{ns_color}22;font-family:monospace'>"
                        f"{ns_label}</span> "
                        f"{raw_title[:80]}",
                        unsafe_allow_html=True,
                    )
    except Exception as e:
        st.caption(f"unavailable: {e}")
