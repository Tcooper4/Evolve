"""
Fetch news articles around a market event time for display and ranking.
Uses NewsAPI; returns articles with title, url, source, publishedAt for use by news_ranker.
"""
import logging
import os
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)


def fetch_news_around_event(
    symbol: str,
    company_name: str,
    event_time: datetime,
    max_items: int = 10,
    hours_window: int = 24,
) -> List[dict]:
    """
    Fetch recent news from NewsAPI for the given symbol/company.
    Articles are returned with title, url, source (dict with name), publishedAt
    for use by rank_news_by_relevance and the Home UI.

    Returns list of dicts compatible with NewsAPI article shape:
    { "title", "url", "source": {"name"}, "publishedAt", "description" }
    """
    api_key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
    if not api_key or not api_key.strip():
        logger.debug("NEWS_API_KEY not set; skipping news fetch")
        return []

    query = company_name or symbol
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={query!s}"
        "&sortBy=publishedAt"
        f"&pageSize={max_items}"
        f"&apiKey={api_key}"
    )
    try:
        import requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.debug("NewsAPI request failed: %s", e)
        return []

    articles = data.get("articles") or []
    out = []
    for a in articles[:max_items]:
        title = a.get("title") or a.get("name")
        if not title:
            continue
        source = a.get("source") or {}
        if isinstance(source, dict):
            name = source.get("name") or "Unknown"
        else:
            name = str(source)
        out.append({
            "title": title,
            "url": a.get("url") or "",
            "source": {"name": name},
            "publishedAt": a.get("publishedAt") or "",
            "description": a.get("description"),
        })
    return out
