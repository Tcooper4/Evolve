"""
Multi-source news aggregator.

Sources (in priority order): yfinance, NewsAPI, RSS feeds, Reddit.
All sources are optional — the aggregator uses whatever is available.
"""

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import email.utils

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
                content = item.get("content") or {}
                if not isinstance(content, dict):
                    content = {}

                # Newer yfinance nests title/publisher under content{}
                title = (
                    item.get("title")
                    or content.get("title")
                    or ""
                )
                if not str(title).strip():
                    continue

                provider = content.get("provider") or {}
                if not isinstance(provider, dict):
                    provider = {}
                publisher = (
                    item.get("publisher")
                    or provider.get("displayName")
                    or content.get("publisher")
                    or "Yahoo Finance"
                )

                ts = item.get("providerPublishTime", 0) or 0
                pub_date = content.get("pubDate") or content.get("displayTime")
                if pub_date:
                    published = str(pub_date)
                else:
                    published = (
                        datetime.fromtimestamp(ts).isoformat()
                        if ts
                        else datetime.utcnow().isoformat()
                    )

                click = content.get("clickThroughUrl") or {}
                if not isinstance(click, dict):
                    click = {}
                canonical = content.get("canonicalUrl") or {}
                if not isinstance(canonical, dict):
                    canonical = {}
                _url = (
                    item.get("url")
                    or item.get("link")
                    or item.get("href")
                    or click.get("url")
                    or canonical.get("url")
                    or content.get("url")
                    or content.get("link")
                    or content.get("href")
                    or ""
                )
                summary = (
                    item.get("summary")
                    or content.get("summary")
                    or content.get("description")
                    or title
                )
                results.append(
                    {
                        "title": str(title).strip(),
                        "url": str(_url or "").strip(),
                        "source": str(publisher or "Yahoo Finance").strip(),
                        "published": published,
                        "summary": str(summary or "").strip(),
                        "symbols": item.get("relatedTickers", [symbol]) or [symbol],
                        "source_type": "yfinance",
                    }
                )
            except Exception as e:
                logger.debug("yfinance news item parse skipped: %s", e)
                continue
        return results
    except Exception as e:  # pragma: no cover - defensive log
        logger.debug("yfinance news failed for %s: %s", symbol, e)
        return []


# ── Source 2: NewsAPI (requires NEWSAPI_KEY env var) ────────────────────────
def _fetch_newsapi(
    query: str,
    max_items: int = 10,
    *,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> List[Dict]:
    """Fetch news from NewsAPI.org when NEWSAPI_KEY is configured.

    Optional ``from_date`` / ``to_date`` (YYYY-MM-DD) pin the search window —
    critical for chart volume-day linkage so we don't always pull "last 3 days".
    """
    from config.api_keys import resolve_api_key
    api_key = resolve_api_key("NEWSAPI_KEY")
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
            "from": from_date or (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d"),
        }
        if to_date:
            params["to"] = to_date
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


# ── Source 3: RSS feeds (reputable wire/finance first; no SA in default mix) ─
RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "cnbc_top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "cnbc_finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}

# Publisher reputation nudges applied in relevance scoring (−0.2 .. +0.25)
_REPUTABLE_SOURCES = (
    ("reuters", 0.25),
    ("bloomberg", 0.25),
    ("wall street journal", 0.22),
    ("wsj", 0.22),
    ("financial times", 0.20),
    ("cnbc", 0.18),
    ("marketwatch", 0.15),
    ("barron", 0.15),
    ("yahoo finance", 0.08),
)
_DEMOTE_SOURCES = (
    ("seeking alpha", -0.18),
    ("motley fool", -0.12),
    ("benzinga", -0.05),
)

# Neutral baseline before reputable/demote bumps (shared with severity scoring).
_REPUTATION_BASE = 0.35


def source_reputation_factor(source: str) -> float:
    """Map a publisher name to [0.05, 1.0] using the same reputable/demote table
    as ``_score_relevance`` — single source of truth for trust weighting.
    """
    src = (source or "").lower()
    score = float(_REPUTATION_BASE)
    for needle, bump in _REPUTABLE_SOURCES:
        if needle in src:
            score += float(bump)
            break
    for needle, bump in _DEMOTE_SOURCES:
        if needle in src:
            score += float(bump)
            break
    return round(min(1.0, max(0.05, score)), 4)


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


# ── Source 4: Reddit/WSB (optional — not in default headline feed) ──────────
def _fetch_reddit(symbol: str, max_items: int = 5) -> List[Dict]:
    """Fetch Reddit discussion from a few finance subs when credentials are set."""
    try:
        from config.api_keys import resolve_api_key

        client_id = (resolve_api_key("REDDIT_CLIENT_ID") or "").strip()
        client_secret = (resolve_api_key("REDDIT_CLIENT_SECRET") or "").strip()
    except Exception:
        client_id = (os.getenv("REDDIT_CLIENT_ID") or "").strip()
        client_secret = (os.getenv("REDDIT_CLIENT_SECRET") or "").strip()
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

    # Source reputation (wire services up, tip-sheet / SA style down)
    src = (
        f"{article.get('source', '')} {article.get('source_type', '')}"
    ).lower()
    for needle, bump in _REPUTABLE_SOURCES:
        if needle in src:
            score += bump
            break
    for needle, bump in _DEMOTE_SOURCES:
        if needle in src:
            score += bump
            break
    # Twitter/X: fast but noisy — small boost only when cashtag/symbol hit
    if article.get("source_type") == "twitter" and sym_lower and sym_lower in text:
        score += 0.12

    return min(1.0, max(0.0, score))


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
    # v3: reputable RSS mix + optional Twitter/X when bearer configured
    cache_key = f"news:v3:{symbol}:{q}:{max_items}:{int(include_reddit)}"
    cached = disk_cache_get(cache_key)
    if cached is not None:
        return cached

    articles: List[Dict] = []

    # Source aggregation (Twitter when key present; Reddit only if explicitly asked)
    articles += _fetch_yfinance_news(symbol, max_items)
    articles += _fetch_newsapi(q, max_items)
    articles += _fetch_rss(q, max_items)
    try:
        from trading.data.twitter_headlines import (
            get_twitter_symbol_headlines,
            twitter_configured,
        )

        if twitter_configured():
            articles += get_twitter_symbol_headlines(
                symbol, max_items=min(8, max_items)
            )
    except Exception as e:
        logger.debug("Twitter headlines skipped: %s", e)
    if include_reddit:
        articles += _fetch_reddit(symbol, 5)

    if not articles:
        return []

    # Deduplicate and score — drop empty titles (bad cache / parse misses)
    articles = [a for a in articles if str(a.get("title") or "").strip()]
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


def _parse_pub_date(pub_str: str):
    try:
        return email.utils.parsedate_to_datetime(pub_str)
    except Exception:
        return None


def get_walter_bloomberg_headlines(max_items: int = 10) -> List[Dict]:
    """
    Fetch breaking financial headlines from Walter Bloomberg (@WalterBloomberg)
    via public RSS feed. Returns list of dicts with keys: title, published, source.
    """
    try:
        import feedparser
    except ImportError:
        return []

    urls = [
        "https://nitter.net/WalterBloomberg/rss",
        "https://nitter.privacydev.net/WalterBloomberg/rss",
    ]
    for url in urls:
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                continue
            items: List[Dict] = []
            for entry in feed.entries[:max_items]:
                pub_date = _parse_pub_date(entry.get("published", ""))
                if pub_date:
                    now = datetime.now(timezone.utc)
                    if (now - pub_date).days > 7:
                        continue
                items.append(
                    {
                        "title": entry.get("title", ""),
                        "published": entry.get("published", ""),
                        "source": "Walter Bloomberg",
                        "link": entry.get("link", ""),
                    }
                )
            if items:
                return items
        except Exception:
            continue
    return []


FINANCIAL_RSS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]


def get_financial_headlines(max_items: int = 10) -> List[Dict]:
    try:
        import feedparser

        items: List[Dict] = []
        for url in FINANCIAL_RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_items]:
                    pub_date = _parse_pub_date(entry.get("published", ""))
                    if pub_date:
                        age = (datetime.now(timezone.utc) - pub_date).days
                        if age > 7:
                            continue
                    items.append(
                        {
                            "title": entry.get("title", ""),
                            "published": entry.get("published", ""),
                            "source": "Financial News",
                            "link": entry.get("link", ""),
                        }
                    )
                if items:
                    return items[:max_items]
            except Exception:
                continue
        return items[:max_items] if items else []
    except Exception:
        return []

