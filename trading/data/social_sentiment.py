# -*- coding: utf-8 -*-
"""Reddit + news headline sentiment.

News path: FinBERT (finance-tuned) blended with VADER when the transformer
loads; VADER-only if FinBERT is unavailable. Headline sentiment alone has
modest standalone predictive power — treat scores as context, not a trade
signal.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from trading.utils.credential_placeholders import is_placeholder_credential
from trading.utils.plain_language import sentiment_label_plain_language

logger = logging.getLogger(__name__)

_USER_AGENT = "EvolveTradingBot/3.23 (research; contact: local)"
_SUBREDDITS = ("wallstreetbets", "stocks")

# FinBERT / VADER blend when FinBERT is available (literature favors domain-
# tuned transformers on financial text; keep VADER as a soft anchor).
_FINBERT_WEIGHT = 0.70
_VADER_WEIGHT = 0.30
_FINBERT_MODEL_ID = "yiyanghkust/finbert-tone"

_finbert_pipe: Any = None
_finbert_failed: bool = False
_finbert_lock = threading.Lock()
_finbert_score_cache: Dict[str, float] = {}
_FINBERT_CACHE_MAX = 2048


def _get_finbert_pipeline() -> Any:
    """Lazy singleton FinBERT pipeline — never load per request."""
    global _finbert_pipe, _finbert_failed
    if _finbert_failed:
        return None
    if _finbert_pipe is not None:
        return _finbert_pipe
    with _finbert_lock:
        if _finbert_failed:
            return None
        if _finbert_pipe is not None:
            return _finbert_pipe
        try:
            from transformers import pipeline as hf_pipeline

            device = -1
            try:
                import torch

                if torch.cuda.is_available():
                    device = 0
            except Exception:
                device = -1
            _finbert_pipe = hf_pipeline(
                "sentiment-analysis",
                model=_FINBERT_MODEL_ID,
                tokenizer=_FINBERT_MODEL_ID,
                truncation=True,
                max_length=128,
                device=device,
            )
            logger.info(
                "FinBERT loaded (%s, device=%s)",
                _FINBERT_MODEL_ID,
                "cuda" if device >= 0 else "cpu",
            )
        except Exception as e:
            _finbert_failed = True
            _finbert_pipe = None
            logger.warning(
                "FinBERT unavailable — news sentiment falls back to VADER: %s", e
            )
            return None
    return _finbert_pipe


def _finbert_label_to_score(label: str, score: float) -> float:
    """Map FinBERT label + confidence to [-1, 1]."""
    lab = (label or "").strip().lower()
    conf = max(0.0, min(1.0, float(score)))
    if lab in ("positive", "pos", "bullish"):
        return conf
    if lab in ("negative", "neg", "bearish"):
        return -conf
    # neutral / unknown
    return 0.0


def score_texts_finbert(texts: List[str]) -> Optional[List[float]]:
    """Batch-score texts with FinBERT → list of [-1, 1] or None if unavailable.

    Results are cached by exact text. Never raises — returns None on failure.
    """
    if not texts:
        return []
    pipe = _get_finbert_pipeline()
    if pipe is None:
        return None
    out: List[Optional[float]] = [None] * len(texts)
    to_run: List[Tuple[int, str]] = []
    for i, t in enumerate(texts):
        key = (t or "").strip()
        if not key:
            out[i] = 0.0
            continue
        if key in _finbert_score_cache:
            out[i] = _finbert_score_cache[key]
        else:
            to_run.append((i, key[:512]))
    if to_run:
        try:
            batch = [t for _, t in to_run]
            raw = pipe(batch)
            if not isinstance(raw, list):
                raw = [raw]
            for (i, key), item in zip(to_run, raw):
                if isinstance(item, list) and item:
                    item = item[0]
                label = str((item or {}).get("label") or "")
                score = float((item or {}).get("score") or 0.0)
                val = _finbert_label_to_score(label, score)
                out[i] = val
                if len(_finbert_score_cache) >= _FINBERT_CACHE_MAX:
                    # Drop an arbitrary oldest-ish entry (FIFO-ish via iter)
                    try:
                        _finbert_score_cache.pop(next(iter(_finbert_score_cache)))
                    except Exception:
                        _finbert_score_cache.clear()
                _finbert_score_cache[key] = val
        except Exception as e:
            logger.warning("FinBERT batch scoring failed: %s", e)
            return None
    return [float(x if x is not None else 0.0) for x in out]


def blend_finbert_vader(
    finbert_scores: Optional[List[float]],
    vader_scores: List[float],
) -> Tuple[List[float], str]:
    """Per-item 0.7 FinBERT / 0.3 VADER when FinBERT present; else VADER-only."""
    if not vader_scores:
        return [], "empty"
    if finbert_scores is None or len(finbert_scores) != len(vader_scores):
        return [max(-1.0, min(1.0, float(v))) for v in vader_scores], "vader"
    blended = [
        max(
            -1.0,
            min(
                1.0,
                _FINBERT_WEIGHT * float(f) + _VADER_WEIGHT * float(v),
            ),
        )
        for f, v in zip(finbert_scores, vader_scores)
    ]
    return blended, "finbert+vader"


def benchmark_finbert_latency(
    n_headlines: int = 15,
    warmup: bool = True,
) -> Dict[str, Any]:
    """Time FinBERT on a synthetic batch (warm inference, not download).

    Returns latency stats for the Phase-1 cost check — does not raise.
    """
    sample = [
        f"Company beats guided-down estimates in quarter {i}"
        for i in range(max(1, int(n_headlines)))
    ]
    t0 = time.perf_counter()
    pipe = _get_finbert_pipeline()
    load_s = time.perf_counter() - t0
    if pipe is None:
        return {
            "available": False,
            "n_headlines": n_headlines,
            "load_s": round(load_s, 3),
            "error": "FinBERT pipeline unavailable",
        }
    if warmup:
        try:
            score_texts_finbert(sample[:2])
        except Exception:
            pass
    # Clear cache entries for these samples so we measure inference
    for t in sample:
        _finbert_score_cache.pop(t, None)
    t1 = time.perf_counter()
    scores = score_texts_finbert(sample)
    infer_s = time.perf_counter() - t1
    return {
        "available": scores is not None,
        "n_headlines": n_headlines,
        "load_s": round(load_s, 3),
        "infer_batch_s": round(infer_s, 3),
        "infer_per_headline_ms": round(1000.0 * infer_s / max(1, n_headlines), 1),
        "model": _FINBERT_MODEL_ID,
    }


def reset_finbert_for_tests() -> None:
    """Test helper — clear singleton/cache state."""
    global _finbert_pipe, _finbert_failed
    with _finbert_lock:
        _finbert_pipe = None
        _finbert_failed = False
        _finbert_score_cache.clear()


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
    """Per-user Reddit OAuth via resolve_api_key only (shared-keys aware)."""
    try:
        from config.api_keys import resolve_api_key

        rid = (resolve_api_key("REDDIT_CLIENT_ID") or "").strip()
        sec = (resolve_api_key("REDDIT_CLIENT_SECRET") or "").strip()
        return rid, sec
    except Exception:
        return "", ""


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
        "plain_language": sentiment_label_plain_language("NEUTRAL"),
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

        out["plain_language"] = sentiment_label_plain_language(
            out["sentiment_label"], out.get("sentiment_score")
        )
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
    """Score aggregator headlines (news + Twitter when keyed).

    Uses FinBERT blended with VADER when the model loads (0.7 / 0.3);
    otherwise VADER-only. Headline sentiment is contextual signal with
    modest standalone predictive power — not a trade decision.
    """
    out: Dict[str, Any] = {
        "success": False,
        "sentiment_score": 0.0,
        "sentiment_label": "NEUTRAL",
        "plain_language": sentiment_label_plain_language("NEUTRAL"),
        "confidence": 0.0,
        "mention_count": 0,
        "top_posts": [],
        "trending": False,
        "error": None,
        "source": "news_headlines",
        "reason": None,
        "engine": "vader",
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
        texts: List[str] = []
        titles: List[str] = []
        meta: List[Dict[str, Any]] = []
        vader_scores: List[float] = []
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title:
                continue
            summary = (a.get("summary") or "")[:400]
            text = f"{title}. {summary}".strip()
            c = float(analyzer.polarity_scores(text).get("compound", 0.0) or 0.0)
            texts.append(text)
            titles.append(title[:200])
            vader_scores.append(c)
            meta.append({
                "source": a.get("source") or a.get("source_type") or "news",
                "url": a.get("url") or "",
            })
        if not vader_scores:
            out["reason"] = "no headlines"
            return out

        finbert_scores = None
        try:
            finbert_scores = score_texts_finbert(texts)
        except Exception as e:
            logger.warning("FinBERT path failed (using VADER): %s", e)
            finbert_scores = None

        blended, engine = blend_finbert_vader(finbert_scores, vader_scores)
        avg_c = float(statistics.mean(blended)) if blended else 0.0
        avg_c = max(-1.0, min(1.0, avg_c))

        scored: List[Dict[str, Any]] = []
        for i, title in enumerate(titles):
            row = {
                "title": title,
                "source": meta[i]["source"],
                "sentiment": round(blended[i], 3),
                "vader": round(vader_scores[i], 3),
                "url": meta[i]["url"],
            }
            if finbert_scores is not None and i < len(finbert_scores):
                row["finbert"] = round(float(finbert_scores[i]), 3)
            scored.append(row)

        out["sentiment_score"] = avg_c
        out["sentiment_label"] = _label_from_score(avg_c)
        out["plain_language"] = sentiment_label_plain_language(
            out["sentiment_label"], avg_c
        )
        out["mention_count"] = len(blended)
        out["confidence"] = min(1.0, 0.35 + 0.05 * len(blended))
        out["top_posts"] = sorted(
            scored, key=lambda x: abs(x.get("sentiment", 0)), reverse=True
        )[:8]
        out["trending"] = len(blended) >= 8
        out["engine"] = engine
        out["source"] = (
            "news_headlines_finbert" if engine == "finbert+vader" else "news_headlines"
        )
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
            "plain_language": sentiment_label_plain_language(
                _label_from_score(blended), blended
            ),
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
        "plain_language": sentiment_label_plain_language("NEUTRAL"),
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
