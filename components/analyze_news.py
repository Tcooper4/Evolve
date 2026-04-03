# -*- coding: utf-8 -*-
"""News list / overlay for Analyze and Deep Dive."""
import streamlit as st

from components.analyze_news_headline import render_analyze_headline_news_panel


def render_news(ticker: str, *, max_items: int = 12) -> None:
    """Headlines for a ticker (price_cache) + optional sentiment strip."""
    try:
        from trading.data.price_cache import get_news

        items = get_news(ticker) or []
        if not items:
            st.caption("No recent headlines.")
            return
        st.subheader("Headlines")
        for item in items[:max_items]:
            content = item.get("content") or {}
            title = (
                item.get("title")
                or item.get("headline")
                or content.get("title")
                or content.get("summary")
                or ""
            )
            if not title:
                continue
            src = item.get("source") or item.get("publisher") or ""
            link = item.get("url") or item.get("link") or ""
            st.markdown(f"**{title[:160]}**")
            if src:
                st.caption(src)
            if link:
                st.caption(f"[Link]({link})")
    except Exception as e:
        st.caption(f"unavailable: {e}")


def render_news_overlay_strip(ticker: str) -> None:
    """Legacy HOT/POS/NEG strip used above tabs on Analyze."""
    render_analyze_headline_news_panel(ticker)
