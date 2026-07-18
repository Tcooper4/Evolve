# -*- coding: utf-8 -*-
"""
Minimal NewsAPI fetcher for recent headlines. Used by Chat to add real-time news context.
"""
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Optional: map ticker to company name for better news search
SYMBOL_TO_QUERY = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "GOOGL": "Google Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Facebook",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
}


def fetch_recent_news(
    symbol: str,
    company_name: Optional[str] = None,
    max_items: int = 5,
) -> List[dict]:
    """
    Fetch recent news headlines from NewsAPI for the given symbol/company.

    Args:
        symbol: Ticker symbol (e.g. AAPL).
        company_name: Optional company name for search query; if None, uses symbol or SYMBOL_TO_QUERY.
        max_items: Max number of items to return (default 5).

    Returns:
        List of dicts with "title" and "description" (description may be None).
        Empty list if API key missing or request fails.
    """
    from config.api_keys import resolve_api_key
    api_key = resolve_api_key("NEWS_API_KEY")
    if not api_key or not api_key.strip():
        logger.debug("NEWS_API_KEY not set; skipping news fetch")
        return []

    query = company_name or SYMBOL_TO_QUERY.get(symbol.upper(), symbol)
    # Pass the key as a request param — never interpolate into a URL string
    # that exceptions / HTTPError might echo into logs.
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": max_items,
        "apiKey": api_key,
    }

    try:
        import requests
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        from config.secret_redact import redact_secrets

        logger.debug(
            "NewsAPI request failed: %s",
            redact_secrets(str(e), known=[api_key]),
        )
        return []

    articles = data.get("articles") or []
    out = []
    for a in articles[:max_items]:
        title = a.get("title") or a.get("name")
        desc = a.get("description")
        if title:
            out.append({"title": title, "description": desc})
    return out


def format_news_for_context(headlines: List[dict]) -> str:
    """Format headline list as a short string for LLM context."""
    if not headlines:
        return ""
    lines = ["\n[Recent headlines]"]
    for i, h in enumerate(headlines, 1):
        line = f"{i}. {h.get('title', '')}"
        if h.get("description"):
            line += f" — {h['description'][:120]}..."
        lines.append(line)
    return "\n".join(lines)
