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
            _title = (
                item.get("title")
                or item.get("headline")
                or content.get("title")
                or content.get("summary")
                or ""
            ).strip()
            if not _title:
                continue
            _url = (
                item.get("url")
                or item.get("link")
                or item.get("href")
                or content.get("url")
                or content.get("link")
                or ""
            )
            _show_title = _title[:160]
            src = item.get("source") or item.get("publisher") or ""
            if _url and _show_title:
                st.markdown(f"**[{_show_title}]({_url})**")
            elif _show_title:
                st.markdown(f"**{_show_title}**")
            if src:
                st.caption(src)
    except Exception as e:
        st.caption(f"unavailable: {e}")


def render_news_overlay_strip(ticker: str) -> None:
    """Legacy HOT/POS/NEG strip used above tabs on Analyze."""
    render_analyze_headline_news_panel(ticker)
