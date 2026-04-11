# -*- coding: utf-8 -*-
"""News headline sentiment strip + Deep Dive news."""
import streamlit as st

from components.analyze_common import sentiment_icon_for_label
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
                    _url = (
                        item.get("url")
                        or item.get("link")
                        or item.get("href")
                        or (item.get("content") or {}).get("url")
                        or (item.get("content") or {}).get("clickThroughUrl")
                        or ""
                    )
                    if not raw_title:
                        continue
                    title_lower = raw_title.lower()
                    score = 0
                    score += sum(1 for k in pos_kw if k in title_lower)
                    score -= sum(1 for k in neg_kw if k in title_lower)
                    if score >= 3:
                        ns_label = "HOT"
                    elif score >= 1:
                        ns_label = "POS"
                    elif score <= -1:
                        ns_label = "NEG"
                    else:
                        ns_label = "NEU"
                    _ic = sentiment_icon_for_label(ns_label)
                    _clean_title = str(raw_title)[:120].strip()
                    _url_s = str(_url or "").strip()
                    if _url_s:
                        st.markdown(
                            f"{_ic} **[{_clean_title}]({_url_s})**",
                        )
                    else:
                        st.markdown(f"{_ic} **{_clean_title}**")
    except Exception as e:
        st.caption(f"unavailable: {e}")
