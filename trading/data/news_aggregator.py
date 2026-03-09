"""
Multi-source news aggregator.

Sources (in priority order): yfinance, NewsAPI, RSS feeds, Reddit.
All sources are optional — the aggregator uses whatever is available.
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List

from trading.utils.data_manager import disk_cache_get, disk_cache_set

logger = logging.getLogger(__name__)


# ── Source 1: yfinance news (always available) ──────────────────────────────
def _fetch_yfinance_news(symbol: str, max_items: int = 10) -> List[Dict]:
    """Fetch news via yfinance.Ticker(symbol).news when available."""
    try:
        import yfinance as yf

        news = getattr(yf.Ticker(symbol), "news", None) or []
        results: List[Dict] = []
        for item in news[:max_items]:
            try:
                ts = item.get("providerPublishTime", 0) or 0
                published = (
                    datetime.fromtimestamp(ts).isoformat() if ts else datetime.utcnow().isoformat()
                )
            except Exception:
                published = datetime.utcnow().isoformat()

            results.append(
                {
                    "title": item.get("title", "") or "",
                    "url": item.get("link", "") or "",
                    "source": item.get("publisher", "yfinance") or "yfinance",
                    "published": published,
                    "summary": item.get("summary", item.get("title", "")) or "",
                    "symbols": item.get("relatedTickers", [symbol]) or [symbol],
                    "source_type": "yfinance",
                }
            )
        return results
    except Exception as e:  # pragma: no cover - defensive log
        logger.debug("yfinance news failed for %s: %s", symbol, e)
        return []


# ── Source 2: NewsAPI (requires NEWSAPI_KEY env var) ────────────────────────
def _fetch_newsapi(query: str, max_items: int = 10) -> List[Dict]:
    """Fetch news from NewsAPI.org when NEWSAPI_KEY is configured."""
    api_key = os.getenv("NEWSAPI_KEY") or os.getenv("NEWS_API_KEY")
    if not api_key:
        return []

    try:
        import requests

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": api_key,
            "pageSize": max_items,
            "sortBy": "publishedAt",
            "language": "en",
            "from": (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d"),
        }
        resp = requests.get(url, params=params, timeout=8)
        articles = resp.json().get("articles", []) if resp.status_code == 200 else []
        out: List[Dict] = []
        for a in articles[:max_items]:
            out.append(
                {
                    "title": a.get("title", "") or "",
                    "url": a.get("url", "") or "",
                    "source": (a.get("source") or {}).get("name", "NewsAPI") or "NewsAPI",
                    "published": a.get("publishedAt", "") or "",
                    "summary": a.get("description", "") or a.get("title", "") or "",
                    "symbols": [],
                    "source_type": "newsapi",
                }
            )
        return out
    except Exception as e:  # pragma: no cover - defensive log
        logger.debug("NewsAPI failed for query '%s': %s", query, e)
        return []


# ── Source 3: RSS feeds ─────────────────────────────────────────────────────
RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc_top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "cnbc_finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "seeking_alpha": "https://seekingalpha.com/feed.xml",
}


def _fetch_rss(query: str = "", max_items: int = 10) -> List[Dict]:
    """Fetch and filter RSS feed entries by a simple keyword query."""
    try:
        import feedparser
    except ImportError:
        return []

    results: List[Dict] = []
    query_lower = (query or "").lower()

    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "") or ""
                summary = entry.get("summary", entry.get("description", "")) or ""

                # Filter by query relevance if query provided
                if query_lower:
                    text = (title + " " + summary).lower()
                    words = [w for w in query_lower.split() if w]
                    if words and not any(w in text for w in words[:3]):
                        continue

                published = entry.get("published", entry.get("updated", "")) or ""
                results.append(
                    {
                        "title": title,
                        "url": entry.get("link", "") or "",
                        "source": feed_name.replace("_", " ").title(),
                        "published": published,
                        "summary": (summary or "")[:300],
                        "symbols": [],
                        "source_type": "rss",
                    }
                )
                if len(results) >= max_items:
                    break
        except Exception:
            continue

    return results[:max_items]


# ── Source 4: Reddit/WSB ────────────────────────────────────────────────────
def _fetch_reddit(symbol: str, max_items: int = 5) -> List[Dict]:
    """Fetch Reddit discussion from a few finance subs when credentials are set."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not (client_id and client_secret):
        return []

    try:
        import praw

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="Evolve/1.0",
        )
        results: List[Dict] = []
        for sub in ["wallstreetbets", "stocks", "investing"]:
            try:
                for post in reddit.subreddit(sub).search(
                    symbol, limit=3, sort="new", time_filter="day"
                ):
                    results.append(
                        {
                            "title": post.title,
                            "url": f"https://reddit.com{post.permalink}",
                            "source": f"r/{sub}",
                            "published": datetime.fromtimestamp(
                                getattr(post, "created_utc", 0) or 0
                            ).isoformat(),
                            "summary": (post.selftext or post.title or "")[:300],
                            "symbols": [symbol],
                            "source_type": "reddit",
                            "score": getattr(post, "score", 0),
                            "num_comments": getattr(post, "num_comments", 0),
                        }
                    )
                    if len(results) >= max_items:
                        break
            except Exception:
                continue
            if len(results) >= max_items:
                break
        return results
    except Exception as e:  # pragma: no cover
        logger.debug("Reddit fetch failed for %s: %s", symbol, e)
        return []


# ── NLP relevance scoring ───────────────────────────────────────────────────
def _score_relevance(article: Dict, query: str, symbol: str) -> float:
    """Score 0–1 based on title/summary relevance to query and symbol."""
    text = (
        (article.get("title", "") or "")
        + " "
        + (article.get("summary", "") or "")
    ).lower()
    score = 0.0

    sym_lower = (symbol or "").lower()
    if sym_lower and sym_lower in text:
        score += 0.4

    # Query term matches
    query_words = [w for w in (query or "").lower().split() if len(w) > 3]
    if query_words:
        matches = sum(1 for w in query_words if w in text)
        score += 0.4 * (matches / max(1, len(query_words)))

    # Recency boost (articles from last 48h score higher)
    try:
        pub = article.get("published", "") or ""
        if pub:
            # Handle both ISO and RSS-style timestamps
            s = pub.replace("Z", "+00:00")
            try:
                pub_dt = datetime.fromisoformat(s.split("+")[0])
            except Exception:
                pub_dt = datetime.fromisoformat(s.replace("+00:00", ""))
            hours_old = (datetime.utcnow() - pub_dt).total_seconds() / 3600.0
            if hours_old >= 0:
                score += 0.2 * max(0.0, 1.0 - hours_old / 48.0)
    except Exception:
        pass

    return min(1.0, score)


def _deduplicate(articles: List[Dict]) -> List[Dict]:
    """Deduplicate by title hash."""
    seen = set()
    out: List[Dict] = []
    for a in articles:
        title = a.get("title", "") or ""
        key = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


# ── Main public API ─────────────────────────────────────────────────────────
def get_news(
    symbol: str,
    query: str = "",
    max_items: int = 10,
    include_reddit: bool = False,
) -> List[Dict]:
    """
    Fetch and rank news from all available sources.

    Returns list of articles sorted by relevance score (highest first).
    Uses disk cache for 10 minutes to reduce external calls.
    """
    q = query or symbol
    cache_key = f"news:{symbol}:{q}:{max_items}:{int(include_reddit)}"
    cached = disk_cache_get(cache_key)
    if cached is not None:
        return cached

    articles: List[Dict] = []

    # Source aggregation
    articles += _fetch_yfinance_news(symbol, max_items)
    articles += _fetch_newsapi(q, max_items)
    articles += _fetch_rss(q, max_items)
    if include_reddit:
        articles += _fetch_reddit(symbol, 5)

    if not articles:
        return []

    # Deduplicate and score
    articles = _deduplicate(articles)
    for a in articles:
        a["relevance_score"] = _score_relevance(a, q, symbol)

    articles.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    result = articles[:max_items]
    disk_cache_set(cache_key, result, ttl=600)  # 10 min
    return result


def get_market_news(max_items: int = 15) -> List[Dict]:
    """General market news (no specific symbol)."""
    articles: List[Dict] = []
    articles += _fetch_newsapi("stock market finance economy", max_items)
    articles += _fetch_rss("", max_items)
    return _deduplicate(articles)[:max_items]

