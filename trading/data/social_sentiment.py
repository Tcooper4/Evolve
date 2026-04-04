# -*- coding: utf-8 -*-
"""Reddit mention sentiment: PRAW when credentials exist, else public JSON API."""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

_USER_AGENT = "EvolveTradingBot/3.23 (research; contact: local)"
_SUBREDDITS = ("wallstreetbets", "stocks")


def _fetch_subreddit_search(
    sub: str, query: str, limit: int
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    q = urllib.parse.urlencode(
        {
            "q": query,
            "restrict_sr": "1",
            "sort": "new",
            "limit": str(min(limit, 100)),
            "t": "day",
        }
    )
    url = f"https://www.reddit.com/r/{sub}/search.json?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        children = (
            data.get("data", {}).get("children", [])
        )
        posts = []
        for ch in children:
            d = ch.get("data") or {}
            posts.append(
                {
                    "id": d.get("id") or "",
                    "title": (d.get("title") or "")[:500],
                    "selftext": (d.get("selftext") or "")[:1500],
                    "score": int(d.get("score") or 0),
                    "subreddit": sub,
                }
            )
        return posts, None
    except urllib.error.HTTPError as e:
        _reason = f"HTTP {e.code} for r/{sub}: {e.reason or 'error'}"
        logger.warning("Reddit rate limit or HTTP error: %s", _reason)
        return [], _reason
    except Exception as e:
        _reason = f"Reddit fetch r/{sub} failed: {e}"
        logger.warning(_reason)
        return [], str(e)


def _reddit_creds_from_runtime() -> Tuple[str, str]:
    rid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    rsec = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if rid and rsec:
        return rid, rsec
    try:
        import streamlit as st

        rid = (st.session_state.get("user_key_REDDIT_CLIENT_ID") or "").strip()
        rsec = (st.session_state.get("user_key_REDDIT_CLIENT_SECRET") or "").strip()
    except Exception:
        pass
    return rid, rsec


def _fetch_via_praw(
    sub: str, query: str, limit: int
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    reddit_id, reddit_secret = _reddit_creds_from_runtime()
    if not reddit_id or not reddit_secret:
        return [], "missing Reddit credentials"
    try:
        import praw
    except ImportError:
        return [], "praw not installed"
    try:
        reddit = praw.Reddit(
            client_id=reddit_id,
            client_secret=reddit_secret,
            user_agent="Evolve/1.0",
        )
        posts: List[Dict[str, Any]] = []
        lim = min(limit, 100)
        for submission in reddit.subreddit(sub).search(
            query, limit=lim, sort="new", time_filter="day"
        ):
            posts.append(
                {
                    "id": submission.id or "",
                    "title": (submission.title or "")[:500],
                    "selftext": (getattr(submission, "selftext", None) or "")[:1500],
                    "score": int(submission.score or 0),
                    "subreddit": sub,
                }
            )
        return posts, None
    except Exception as e:
        _reason = f"PRAW fetch r/{sub} failed: {e}"
        logger.warning(_reason)
        return [], str(e)


def _collect_reddit_posts(
    sym: str, limit: int, use_praw: bool
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    seen = set()
    posts: List[Dict[str, Any]] = []
    _last_fetch_err: Optional[str] = None
    fetch_fn = _fetch_via_praw if use_praw else _fetch_subreddit_search
    for sub in _SUBREDDITS:
        batch, fetch_err = fetch_fn(sub, sym, limit)
        if fetch_err:
            _last_fetch_err = fetch_err
        for p in batch:
            pid = (p.get("id") or "").strip()
            key = pid or (sub, p.get("title"), p.get("score"))
            if key in seen:
                continue
            seen.add(key)
            posts.append(p)
        time.sleep(0.3)
        if len(batch) < 3:
            batch_d, fetch_err_d = fetch_fn(sub, f"${sym}", limit)
            if fetch_err_d:
                _last_fetch_err = fetch_err_d
            for p in batch_d:
                pid = (p.get("id") or "").strip()
                key = pid or (sub, p.get("title"), p.get("score"))
                if key in seen:
                    continue
                seen.add(key)
                posts.append(p)
            time.sleep(0.3)
    return posts, _last_fetch_err


@st.cache_data(ttl=300, show_spinner=False)
def _get_social_sentiment_impl(
    symbol: str, limit: int, _auth_mode: str
) -> Dict[str, Any]:
    """
    Internal cached implementation. _auth_mode is 'praw' or 'json' so cache
    invalidates when Reddit credentials are added or removed.
    """
    use_praw = _auth_mode == "praw"
    out: Dict[str, Any] = {
        "success": False,
        "sentiment_score": 0.0,
        "sentiment_label": "NEUTRAL",
        "mention_count": 0,
        "top_posts": [],
        "trending": False,
        "error": None,
        "source": "reddit",
        "reason": None,
    }
    sym = (symbol or "").strip().upper()
    if not sym:
        out["error"] = "symbol required"
        return out
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        logger.warning("VADER not available: %s", e)
        out["error"] = str(e)
        out["source"] = "unavailable"
        out["reason"] = str(e)
        return out

    try:
        posts, _last_fetch_err = _collect_reddit_posts(sym, limit, use_praw)

        if use_praw and not posts and _last_fetch_err:
            posts, _last_fetch_err = _collect_reddit_posts(sym, limit, False)

        if not posts:
            if _last_fetch_err:
                out["success"] = False
                out["sentiment_score"] = 0.0
                out["source"] = "unavailable"
                out["reason"] = _last_fetch_err
                out["error"] = _last_fetch_err
                return out
            out["success"] = True
            out["mention_count"] = 0
            return out

        compounds: List[float] = []
        top_with_sent: List[Dict[str, Any]] = []
        for p in posts[: max(limit * 2, 25)]:
            text = f"{p.get('title', '')} {p.get('selftext', '')}"
            vs = analyzer.polarity_scores(text)
            comp = float(vs.get("compound", 0.0))
            compounds.append(comp)
            top_with_sent.append(
                {
                    "title": p.get("title", ""),
                    "score": p.get("score", 0),
                    "sentiment": round(comp, 3),
                }
            )

        avg_c = float(statistics.mean(compounds)) if compounds else 0.0
        try:
            from trading.nlp.sentiment_processor import SentimentProcessor

            _sp = SentimentProcessor()
            _combo = " ".join(
                f"{p.get('title', '')} {p.get('selftext', '')}" for p in posts[:15]
            )
            if len(_combo.strip()) > 20:
                _sr = _sp.analyze_sentiment(_combo[:8000])
                _blend = float(getattr(_sr, "scaled_score", 0.0) or 0.0)
                avg_c = 0.6 * avg_c + 0.4 * max(-1.0, min(1.0, _blend))
        except Exception:
            pass
        out["sentiment_score"] = max(-1.0, min(1.0, avg_c))
        if avg_c >= 0.15:
            out["sentiment_label"] = "BULLISH"
        elif avg_c <= -0.15:
            out["sentiment_label"] = "BEARISH"
        else:
            out["sentiment_label"] = "NEUTRAL"

        out["mention_count"] = len(posts)
        top_with_sent.sort(key=lambda x: (x.get("score", 0), abs(x.get("sentiment", 0))), reverse=True)
        out["top_posts"] = top_with_sent[:10]
        out["trending"] = len(posts) >= max(8, limit // 2)
        out["success"] = True
        return out
    except Exception as e:
        logger.warning("get_social_sentiment failed for %s: %s", sym, e)
        out["error"] = str(e)
        return out


def get_social_sentiment(symbol: str, limit: int = 25) -> Dict[str, Any]:
    """
    Reddit sentiment from r/wallstreetbets and r/stocks (last day, search).

    Returns sentiment_score (-1..1), label, mention_count, top_posts, trending.
    """
    rid, rsec = _reddit_creds_from_runtime()
    mode = "praw" if (rid and rsec) else "json"
    return _get_social_sentiment_impl(symbol, limit, mode)
