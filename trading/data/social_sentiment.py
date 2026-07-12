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

from trading.utils.credential_placeholders import is_placeholder_credential

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
    """resolve_api_key first, then env, then Streamlit session (Cloud)."""
    rid_e = ""
    sec_e = ""
    try:
        from config.api_keys import resolve_api_key

        rid_e = (resolve_api_key("REDDIT_CLIENT_ID") or "").strip()
        sec_e = (resolve_api_key("REDDIT_CLIENT_SECRET") or "").strip()
    except Exception:
        rid_e = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
        sec_e = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    rid_s = ""
    sec_s = ""
    try:
        import streamlit as st

        rid_s = (st.session_state.get("user_key_REDDIT_CLIENT_ID") or "").strip()
        sec_s = (st.session_state.get("user_key_REDDIT_CLIENT_SECRET") or "").strip()
    except Exception:
        pass
    return rid_e or rid_s, sec_e or sec_s


def _reddit_use_praw(reddit_id: str, reddit_secret: str) -> bool:
    """PRAW only when both credentials are real (not empty or .env template placeholders)."""
    rid = str(reddit_id or "").strip()
    rsec = str(reddit_secret or "").strip()
    if not rid or not rsec:
        return False
    if is_placeholder_credential(rid):
        return False
    if is_placeholder_credential(rsec):
        return False
    return True


def _fetch_via_praw(
    sub: str, query: str, limit: int
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    reddit_id, reddit_secret = _reddit_creds_from_runtime()
    if not _reddit_use_praw(reddit_id, reddit_secret):
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
    rid, rsec = _reddit_creds_from_runtime()
    if use_praw and not _reddit_use_praw(rid, rsec):
        use_praw = False
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


def _label_from_score(avg_c: float) -> str:
    if avg_c >= 0.15:
        return "BULLISH"
    if avg_c <= -0.15:
        return "BEARISH"
    return "NEUTRAL"


def get_news_headline_sentiment(symbol: str, max_items: int = 15) -> Dict[str, Any]:
    """Primary sentiment: score aggregator headlines (news + Twitter when keyed)."""
    out: Dict[str, Any] = {
        "success": False,
        "sentiment_score": 0.0,
        "sentiment_label": "NEUTRAL",
        "confidence": 0.0,
        "mention_count": 0,
        "top_posts": [],
        "trending": False,
        "error": None,
        "source": "news_headlines",
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
        out["error"] = str(e)
        out["source"] = "unavailable"
        out["reason"] = str(e)
        return out
    try:
        from trading.data.news_aggregator import get_news

        articles = get_news(sym, max_items=max_items, include_reddit=False) or []
        scored: List[Dict[str, Any]] = []
        compounds: List[float] = []
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title:
                continue
            summary = (a.get("summary") or "")[:400]
            text = f"{title}. {summary}".strip()
            c = float(analyzer.polarity_scores(text).get("compound", 0.0) or 0.0)
            compounds.append(c)
            scored.append({
                "title": title[:200],
                "source": a.get("source") or a.get("source_type") or "news",
                "sentiment": round(c, 3),
                "url": a.get("url") or "",
            })
        if not compounds:
            out["reason"] = "no headlines"
            return out
        avg_c = float(statistics.mean(compounds))
        try:
            from trading.nlp.sentiment_processor import SentimentProcessor

            _sp = SentimentProcessor()
            _combo = " ".join((a.get("title") or "") for a in articles[:12])
            if len(_combo.strip()) > 20:
                _sr = _sp.analyze_sentiment(_combo[:8000])
                _blend = float(getattr(_sr, "scaled_score", 0.0) or 0.0)
                avg_c = 0.6 * avg_c + 0.4 * max(-1.0, min(1.0, _blend))
        except Exception:
            pass
        avg_c = max(-1.0, min(1.0, avg_c))
        out["sentiment_score"] = avg_c
        out["sentiment_label"] = _label_from_score(avg_c)
        out["mention_count"] = len(compounds)
        out["confidence"] = min(1.0, 0.35 + 0.05 * len(compounds))
        out["top_posts"] = sorted(
            scored, key=lambda x: abs(x.get("sentiment", 0)), reverse=True
        )[:8]
        out["trending"] = len(compounds) >= 8
        out["success"] = True
        return out
    except Exception as e:
        logger.warning("get_news_headline_sentiment failed for %s: %s", sym, e)
        out["error"] = str(e)
        return out


def get_social_sentiment(symbol: str, limit: int = 25) -> Dict[str, Any]:
    """
    Market sentiment for AI Score.

    Primary: news/Twitter aggregator headlines.
    Secondary: Reddit when credentials exist (30% blend).
    Last resort: public Reddit JSON only if news is empty.
    """
    news = get_news_headline_sentiment(symbol, max_items=15)
    news_ok = bool(news.get("success") and int(news.get("mention_count") or 0) > 0)

    rid, rsec = _reddit_creds_from_runtime()
    reddit: Optional[Dict[str, Any]] = None
    try:
        if _reddit_use_praw(rid, rsec):
            reddit = _get_social_sentiment_impl(symbol, limit, "praw")
        elif not news_ok:
            # No Reddit keys — soft public JSON only when headlines failed
            reddit = _get_social_sentiment_impl(symbol, limit, "json")
    except Exception as e:
        logger.debug("Reddit sentiment optional path: %s", e)
        reddit = None

    reddit_ok = bool(
        reddit
        and reddit.get("success")
        and int(reddit.get("mention_count") or 0) > 0
        and not reddit.get("error")
    )

    if news_ok and reddit_ok and reddit is not None:
        blended = max(
            -1.0,
            min(
                1.0,
                0.70 * float(news["sentiment_score"])
                + 0.30 * float(reddit["sentiment_score"]),
            ),
        )
        return {
            "success": True,
            "sentiment_score": blended,
            "sentiment_label": _label_from_score(blended),
            "confidence": max(
                float(news.get("confidence") or 0.5),
                float(reddit.get("confidence") or 0.0),
            ),
            "mention_count": int(news.get("mention_count") or 0)
            + int(reddit.get("mention_count") or 0),
            "top_posts": (news.get("top_posts") or [])[:5]
            + (reddit.get("top_posts") or [])[:3],
            "trending": bool(news.get("trending") or reddit.get("trending")),
            "error": None,
            "source": "news+reddit",
            "reason": None,
            "news_component": news.get("sentiment_score"),
            "reddit_component": reddit.get("sentiment_score"),
        }

    if news_ok:
        return news
    if reddit_ok and reddit is not None:
        return reddit

    return {
        "symbol": symbol,
        "sentiment_score": 0.0,
        "sentiment_label": "NEUTRAL",
        "confidence": 0.0,
        "source": "unavailable",
        "reason": (
            (reddit or {}).get("reason")
            or news.get("reason")
            or news.get("error")
            or "no headlines available"
        ),
        "success": False,
        "mention_count": 0,
    }
