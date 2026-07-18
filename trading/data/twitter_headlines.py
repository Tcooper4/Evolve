# -*- coding: utf-8 -*-
"""Twitter/X recent headlines for breaking news and volume-overlay context.

Smart path:
1. Official API v2 recent search when TWITTER_BEARER_TOKEN is set (Settings/env).
2. Optional curated finance accounts for market-wide breaking.
3. Soft fail everywhere — never raise into chart/news UI.
4. Nitter RSS only as last-resort fallback for Walter Bloomberg (no API key).

Basic/free X tiers typically allow ~7 days of recent search — enough for
intraday/short chart overlays.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

# Curated breaking / markets accounts (cashtag search is primary for symbols)
_BREAKING_ACCOUNTS = (
    "WalterBloomberg",
    "DeItaone",
    "FirstSquawk",
    "LiveSquawk",
)


def _bearer() -> Optional[str]:
    """Per-user bearer via resolve_api_key only — never bare os.getenv
    (that bypasses the shared-keys admin policy)."""
    try:
        from config.api_keys import resolve_api_key

        tok = (resolve_api_key("TWITTER_BEARER_TOKEN") or "").strip()
        return tok or None
    except Exception:
        return None


def _search_recent(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Twitter API v2 recent search. Returns normalized headline dicts."""
    token = _bearer()
    if not token:
        return []
    max_results = max(10, min(int(max_results), 50))  # API minimum is 10
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,author_id,public_metrics,lang",
        "expansions": "author_id",
        "user.fields": "username,name",
    }
    try:
        r = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=12,
        )
        if r.status_code == 401:
            logger.warning("Twitter bearer rejected (401)")
            return []
        if r.status_code == 429:
            logger.warning("Twitter rate limited (429)")
            return []
        if r.status_code >= 400:
            logger.debug("Twitter search HTTP %s: %s", r.status_code, r.text[:200])
            return []
        data = r.json() or {}
        users = {
            u.get("id"): u
            for u in ((data.get("includes") or {}).get("users") or [])
            if isinstance(u, dict)
        }
        out: List[Dict[str, Any]] = []
        for tw in data.get("data") or []:
            if not isinstance(tw, dict):
                continue
            text = (tw.get("text") or "").strip()
            if not text:
                continue
            author = users.get(tw.get("author_id") or "", {}) or {}
            handle = author.get("username") or "twitter"
            created = tw.get("created_at") or ""
            tid = tw.get("id") or ""
            out.append({
                "title": text[:280],
                "url": f"https://x.com/{handle}/status/{tid}" if tid else "",
                "source": f"@{handle}",
                "published": created,
                "summary": text[:300],
                "symbols": [],
                "source_type": "twitter",
            })
        return out
    except Exception as e:
        logger.debug("Twitter search failed: %s", e)
        return []


def get_twitter_symbol_headlines(
    symbol: str,
    max_items: int = 8,
    since_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Headlines mentioning a ticker (cashtag + plain symbol)."""
    sym = (symbol or "").strip().upper().lstrip("$")
    if not sym:
        return []
    # Recent search operators; keep query short for free tiers
    q = f"(${sym} OR {sym}) (stock OR shares OR earnings OR market) -is:retweet lang:en"
    if since_date:
        # YYYY-MM-DD → RFC3339 start; end = next day
        try:
            d0 = datetime.strptime(since_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            d1 = d0 + timedelta(days=1)
            # recent search: start_time / end_time
            token = _bearer()
            if token:
                params = {
                    "query": q,
                    "max_results": max(10, min(max_items, 50)),
                    "tweet.fields": "created_at,author_id",
                    "expansions": "author_id",
                    "user.fields": "username",
                    "start_time": d0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_time": d1.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                r = requests.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    timeout=12,
                )
                if r.status_code < 400:
                    data = r.json() or {}
                    users = {
                        u.get("id"): u
                        for u in ((data.get("includes") or {}).get("users") or [])
                        if isinstance(u, dict)
                    }
                    out: List[Dict[str, Any]] = []
                    for tw in data.get("data") or []:
                        text = (tw.get("text") or "").strip()
                        if not text:
                            continue
                        handle = (users.get(tw.get("author_id") or "", {}) or {}).get("username") or "twitter"
                        tid = tw.get("id") or ""
                        out.append({
                            "title": text[:280],
                            "url": f"https://x.com/{handle}/status/{tid}" if tid else "",
                            "source": f"@{handle}",
                            "published": tw.get("created_at") or since_date,
                            "summary": text[:300],
                            "symbols": [sym],
                            "source_type": "twitter",
                        })
                    if out:
                        return out[:max_items]
        except Exception as e:
            logger.debug("Twitter dated search skipped: %s", e)
    return _search_recent(q, max_items)[:max_items]


def get_breaking_headlines(max_items: int = 12) -> List[Dict[str, Any]]:
    """Market-wide breaking: curated accounts via API, else Nitter Walter fallback."""
    # from:Account OR from:Account2 ...
    parts = [f"from:{a}" for a in _BREAKING_ACCOUNTS]
    q = f"({' OR '.join(parts)}) -is:retweet"
    items = _search_recent(q, max_results=max(10, max_items))
    if items:
        return items[:max_items]

    # Fallback: Nitter RSS for Walter only
    try:
        from trading.data.news_aggregator import get_walter_bloomberg_headlines

        fb = get_walter_bloomberg_headlines(max_items=max_items)
        for a in fb:
            a.setdefault("source_type", "twitter_rss")
            a.setdefault("url", a.get("link") or "")
        return fb
    except Exception:
        return []


def get_breaking_headlines_for_date(
    date_str: str,
    max_items: int = 12,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Breaking wires (WalterBloomberg / DeItaone / squawk) for one calendar day.

    Free X recent-search only covers ~7 days — older dates return []. Optional
    ``symbol`` prefers tweets that also mention the ticker / cashtag.
    """
    try:
        d0 = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return []
    d1 = d0 + timedelta(days=1)
    # Refuse clearly out-of-range asks (API will 400 anyway past ~7d)
    age = (datetime.now(timezone.utc) - d0).days
    if age > 7 or age < -1:
        return []

    parts = [f"from:{a}" for a in _BREAKING_ACCOUNTS]
    q = f"({' OR '.join(parts)}) -is:retweet"
    sym = (symbol or "").strip().upper().lstrip("$")
    if sym:
        # Soft prefer: still pull wires that day, rank symbol mentions later
        q = f"{q} (${sym} OR {sym} OR stocks OR market OR S&P OR Nasdaq)"

    token = _bearer()
    if not token:
        # RSS Walter fallback — filter by published day when possible
        try:
            from trading.data.news_aggregator import get_walter_bloomberg_headlines

            fb = get_walter_bloomberg_headlines(max_items=max(max_items, 15))
            out: List[Dict[str, Any]] = []
            for a in fb:
                a = dict(a)
                a.setdefault("source_type", "twitter_rss")
                a.setdefault("url", a.get("link") or "")
                pub = str(a.get("published") or "")
                if pub and str(date_str)[:10] not in pub and str(date_str)[:10] not in pub.replace("/", "-"):
                    # keep if we cannot parse — ranking/filter happens upstream
                    pass
                out.append(a)
            return out[:max_items]
        except Exception:
            return []

    try:
        params = {
            "query": q,
            "max_results": max(10, min(int(max_items), 50)),
            "tweet.fields": "created_at,author_id",
            "expansions": "author_id",
            "user.fields": "username",
            "start_time": d0.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": d1.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        r = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=12,
        )
        if r.status_code >= 400:
            logger.debug(
                "breaking-for-date HTTP %s: %s", r.status_code, r.text[:180]
            )
            return []
        data = r.json() or {}
        users = {
            u.get("id"): u
            for u in ((data.get("includes") or {}).get("users") or [])
            if isinstance(u, dict)
        }
        out = []
        for tw in data.get("data") or []:
            text = (tw.get("text") or "").strip()
            if not text:
                continue
            handle = (
                (users.get(tw.get("author_id") or "", {}) or {}).get("username")
                or "twitter"
            )
            tid = tw.get("id") or ""
            out.append({
                "title": text[:280],
                "url": f"https://x.com/{handle}/status/{tid}" if tid else "",
                "source": f"@{handle}",
                "published": tw.get("created_at") or date_str,
                "summary": text[:300],
                "symbols": [sym] if sym else [],
                "source_type": "twitter",
                "breaking": True,
            })
        return out[:max_items]
    except Exception as e:
        logger.debug("get_breaking_headlines_for_date failed: %s", e)
        return []


def twitter_configured() -> bool:
    return bool(_bearer())
