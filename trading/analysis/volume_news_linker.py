# -*- coding: utf-8 -*-
"""
Links significant volume/price events to contemporaneous news.
Used for the news-annotated candlestick chart.

Honesty (Phase 2+): every linked headline carries ``link_quality`` —
``same_day`` when it falls in the target date window, or
``fallback_recent`` when nothing matched and we fall back to recent
articles. Chart annotations prefer empty/"no dated news" over
re-attaching the same recent Yahoo blurb to every historical volume day
(feeds are recent-only; they are not a news archive).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from trading.utils.data_manager import disk_cache_get, disk_cache_set

logger = logging.getLogger(__name__)

LINK_SAME_DAY = "same_day"
LINK_FALLBACK = "fallback_recent"

# Handles that often print the *why* of a move before wire recycle pieces.
_BREAKING_SOURCE_HINTS = (
    "walterbloomberg",
    "walter bloomberg",
    "dei taone",
    "deltaone",
    "firstsquawk",
    "livesquawk",
    "@walter",
    "@dei",
)


def meets_spike_thresholds(
    volume_ratio: float,
    price_change_pct: float,
    volume_threshold: float = 2.0,
    price_threshold: float = 0.02,
) -> bool:
    """Shared significant-volume rule: (2× vol AND |move|≥2%) OR ≥3× vol."""
    import math
    try:
        vr = float(volume_ratio)
        pc = float(price_change_pct)
    except (TypeError, ValueError):
        return False
    if math.isnan(vr) or math.isnan(pc):
        return False
    vol_and_move = (vr >= volume_threshold) and (abs(pc) >= price_threshold)
    vol_extreme = vr >= max(volume_threshold * 1.5, 3.0)
    return bool(vol_and_move or vol_extreme)


def detect_significant_candles(
    hist: pd.DataFrame,
    volume_threshold: float = 2.0,
    price_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Returns rows where volume > N×avg AND |price change| > threshold.
    Adds columns: volume_ratio, price_change_pct, is_significant, candle_type.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    df = hist.copy()

    # Normalize column names (work with either Close/Volume or lowercase)
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower

    # If we lost Close due to mismatch, fall back to original
    if "close" not in df.columns and "Close" in hist.columns:
        df = hist.copy()
        df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]

    if "Close" in df.columns and "close" not in df.columns:
        df["close"] = df["Close"]
    if "Volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["Volume"]

    if "close" not in df.columns or "volume" not in df.columns:
        return df

    df["price_change_pct"] = df["close"].pct_change()
    rolling_vol = df["volume"].rolling(20, min_periods=5).mean()
    df["volume_ratio"] = df["volume"] / rolling_vol.clip(lower=1)
    df["is_significant"] = [
        meets_spike_thresholds(
            vr, pc, volume_threshold=volume_threshold, price_threshold=price_threshold
        )
        for vr, pc in zip(df["volume_ratio"].tolist(), df["price_change_pct"].tolist())
    ]
    df["candle_type"] = np.where(
        df["price_change_pct"] > 0,
        "bullish",
        np.where(df["price_change_pct"] < 0, "bearish", "neutral"),
    )
    return df


def _tag_articles(
    articles: List[Dict[str, Any]],
    link_quality: str,
) -> List[Dict[str, Any]]:
    """Copy articles with honesty fields (never mutate caller list in place)."""
    out: List[Dict[str, Any]] = []
    confirmed = link_quality == LINK_SAME_DAY
    for a in articles:
        row = dict(a) if isinstance(a, dict) else {"title": str(a)}
        row["link_quality"] = link_quality
        row["date_confirmed"] = confirmed
        out.append(row)
    return out


def _normalize_title(title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip().lower())
    return t[:160]


def _parse_published(pub_raw: Any) -> Optional[datetime]:
    if not pub_raw:
        return None
    try:
        s = str(pub_raw).replace("Z", "+00:00")
        # email-style dates
        if "," in s and ":" in s and not s[0].isdigit():
            import email.utils

            dt = email.utils.parsedate_to_datetime(str(pub_raw))
            if dt is not None and getattr(dt, "tzinfo", None) is not None:
                return dt.replace(tzinfo=None)
            return dt
        pub_dt = datetime.fromisoformat(s.split("+")[0].split(".")[0])
        if getattr(pub_dt, "tzinfo", None) is not None:
            pub_dt = pub_dt.replace(tzinfo=None)
        return pub_dt
    except Exception:
        return None


def score_article_for_event(
    article: Dict[str, Any],
    symbol: str,
    date_str: str,
    price_change_pct: Optional[float] = None,
    *,
    use_finbert: bool = False,
) -> float:
    """
    How likely is this headline explaining *this* session's move?

    Not full causal NLP — ranks date proximity, ticker mention, breaking
    wires (Walter / squawk), and optional FinBERT polarity vs the move.
    """
    if not isinstance(article, dict):
        return 0.0
    title = str(article.get("title") or "")
    summary = str(article.get("summary") or "")
    text = f"{title} {summary}".lower()
    sym = (symbol or "").strip().upper().lstrip("$")
    score = 0.0

    # Date proximity to the event day (tight window preferred)
    try:
        target = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except Exception:
        target = None
    pub = _parse_published(article.get("published"))
    if target is not None and pub is not None:
        delta_h = abs((pub - target).total_seconds()) / 3600.0
        if delta_h <= 24:
            score += 0.40
        elif delta_h <= 48:
            score += 0.25
        elif delta_h <= 72:
            score += 0.10
        else:
            score -= 0.15
    elif article.get("source_type") in ("twitter", "twitter_rss") and article.get("breaking"):
        score += 0.15  # dated search already windowed

    # Ticker / cashtag in headline
    if sym and (sym.lower() in text or f"${sym.lower()}" in text):
        score += 0.35
    elif sym == "SPY" and any(k in text for k in ("s&p", "spx", "spy", "wall street", "stocks")):
        score += 0.20
    elif sym == "QQQ" and any(k in text for k in ("nasdaq", "qqq", "tech stocks")):
        score += 0.20

    # Breaking wires beat evergreen tip-sheet recycle
    src = f"{article.get('source', '')} {article.get('source_type', '')}".lower()
    if article.get("breaking") or any(h in src for h in _BREAKING_SOURCE_HINTS):
        score += 0.28
    if "twitter" in str(article.get("source_type") or ""):
        score += 0.08
    if article.get("archive") or article.get("source_type") == "gdelt":
        score += 0.18  # dated archive hit — prefer over recycled recent wire

    # Soft polarity vs session move — opt-in (FinBERT cold-start is slow).
    if (
        use_finbert
        and price_change_pct is not None
        and title.strip()
    ):
        try:
            from trading.data.social_sentiment import score_texts_finbert

            fb = score_texts_finbert([title[:400]])
            if fb and fb[0] is not None:
                pol = float(fb[0])
                move = float(price_change_pct)
                if move > 0.005 and pol > 0.15:
                    score += 0.12
                elif move < -0.005 and pol < -0.15:
                    score += 0.12
                elif abs(move) > 0.01 and pol * move < 0:
                    score -= 0.08
        except Exception:
            pass

    # Evergreen / tip-sheet patterns that rarely explain a specific volume day
    evergreen = (
        "everyone ignored", "yield actually", "is that fat", "income investors",
        "here are ", "top stocks to", "should you buy",
    )
    if any(p in text for p in evergreen):
        score -= 0.25

    return float(max(0.0, min(1.0, score)))


def classify_news_for_date(
    articles: List[Dict[str, Any]],
    date_str: str,
    n_articles: int = 5,
    *,
    allow_fallback: bool = True,
    price_change_pct: Optional[float] = None,
    symbol: str = "",
    exclude_titles: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Pure classifier: partition fetched articles into same-day window vs fallback.

    Returns ``(tagged_articles, link_quality)``. When ``allow_fallback`` is
    False and nothing is in-window, returns ``([], same_day)`` rather than
    recycling unrelated recent headlines.
    """
    if not articles:
        return [], LINK_SAME_DAY

    exclude = {_normalize_title(t) for t in (exclude_titles or []) if t}
    try:
        target = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except Exception:
        return [], LINK_SAME_DAY
    # Tight primary window: event day and next calendar day (after-hours print)
    window_start = target - timedelta(hours=6)  # prior evening prints still ok
    window_end = target + timedelta(days=1, hours=20)

    relevant: List[Dict[str, Any]] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        title_key = _normalize_title(str(a.get("title") or ""))
        if title_key and title_key in exclude:
            continue
        pub = _parse_published(a.get("published"))
        if pub is None:
            # Dated Twitter/breaking search already windowed — keep
            if a.get("source_type") in ("twitter", "twitter_rss") or a.get("breaking"):
                relevant.append(a)
            continue
        if window_start <= pub <= window_end:
            relevant.append(a)

    def _rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for a in rows:
            s = score_article_for_event(
                a, symbol, date_str, price_change_pct=price_change_pct
            )
            row = dict(a)
            row["event_link_score"] = round(s, 3)
            scored.append(row)
        scored.sort(key=lambda x: float(x.get("event_link_score") or 0), reverse=True)
        # Drop near-zero junk even if it fell in the date window
        scored = [a for a in scored if float(a.get("event_link_score") or 0) >= 0.18]
        return scored[:n_articles]

    if relevant:
        ranked = _rank(relevant)
        if ranked:
            return _tag_articles(ranked, LINK_SAME_DAY), LINK_SAME_DAY

    if not allow_fallback:
        return [], LINK_SAME_DAY

    # Fallback: recent articles, but never ones already claimed by another mark
    pool = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        title_key = _normalize_title(str(a.get("title") or ""))
        if title_key and title_key in exclude:
            continue
        pool.append(a)
    ranked_fb = _rank(pool)[:3]
    if not ranked_fb:
        return [], LINK_FALLBACK
    return _tag_articles(ranked_fb, LINK_FALLBACK), LINK_FALLBACK


def get_news_for_date(
    symbol: str,
    date_str: str,
    n_articles: int = 5,
    *,
    allow_fallback: bool = True,
    price_change_pct: Optional[float] = None,
    exclude_titles: Optional[Sequence[str]] = None,
    include_archives: bool = True,
) -> List[Dict]:
    """
    Fetch news articles from around a specific date.
    date_str: ISO format 'YYYY-MM-DD'.

    Prefers Walter/squawk breaking tweets (when bearer + ≤7d) and NewsAPI
    day windows over recycled "recent" Yahoo/RSS. Each returned article
    includes ``date_confirmed`` / ``link_quality``.

    ``include_archives=False`` skips GDELT (slow) — use for short chart windows
    where recent Twitter/NewsAPI/Yahoo cover the lookback.
    """
    # Cache key must ignore ephemeral exclude set (caller re-filters)
    cache_key = (
        f"news_date:v5:{symbol}:{date_str}:{n_articles}:"
        f"{int(allow_fallback)}:{int(include_archives)}:"
        f"{round(float(price_change_pct or 0), 4)}"
    )
    cached = disk_cache_get(cache_key)
    articles: List[Dict] = []
    if cached is not None:
        articles = list(cached)
    else:
        try:
            # 1) Breaking wires for that day (Walter et al.) — highest priority
            try:
                from trading.data.twitter_headlines import (
                    get_breaking_headlines_for_date,
                    get_twitter_symbol_headlines,
                )

                articles.extend(
                    get_breaking_headlines_for_date(
                        date_str, max_items=max(n_articles, 8), symbol=symbol
                    )
                    or []
                )
                articles.extend(
                    get_twitter_symbol_headlines(
                        symbol, max_items=n_articles, since_date=date_str
                    )
                    or []
                )
            except Exception as e:
                logger.debug("Twitter/breaking for date skipped: %s", e)

            # 2) NewsAPI pinned to the event window (real dated search when keyed)
            try:
                from trading.data.news_aggregator import _fetch_newsapi

                d0 = (
                    datetime.strptime(date_str[:10], "%Y-%m-%d") - timedelta(days=1)
                ).strftime("%Y-%m-%d")
                d1 = (
                    datetime.strptime(date_str[:10], "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                articles.extend(
                    _fetch_newsapi(
                        symbol, max_items=max(n_articles, 8), from_date=d0, to_date=d1
                    )
                    or []
                )
            except Exception as e:
                logger.debug("NewsAPI dated fetch skipped: %s", e)

        # 2b) GDELT dated archive for older sessions (free; recent Yahoo can't)
            if include_archives:
                try:
                    age = (
                        datetime.utcnow().date()
                        - datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                    ).days
                    if age >= 2:
                        from trading.data.gdelt_archive import fetch_gdelt_headlines

                        # One GDELT query per day (ticker only) — broad index
                        # query was a second ~14s tail on every SPY mark.
                        articles.extend(
                            fetch_gdelt_headlines(
                                symbol, date_str, max_items=max(n_articles, 6)
                            )
                            or []
                        )
                except Exception as e:
                    logger.debug("GDELT archive skipped: %s", e)

            # 3) General aggregator — will be date-filtered in classify; for
            #    older marks these are mostly noise (recent-only feeds).
            try:
                from trading.data.news_aggregator import get_news

                age = (
                    datetime.utcnow().date()
                    - datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                ).days
                if age <= 5:
                    articles.extend(
                        list(get_news(symbol, max_items=max(n_articles, 8)) or [])
                    )
            except Exception as e:
                logger.debug("get_news for date skipped: %s", e)

            # Deduplicate by title before classify
            seen: Set[str] = set()
            deduped: List[Dict] = []
            for a in articles:
                if not isinstance(a, dict):
                    continue
                key = _normalize_title(str(a.get("title") or ""))
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(a)
            articles = deduped
            disk_cache_set(cache_key, articles, ttl=1800)
        except Exception as e:  # pragma: no cover
            logger.debug("News for date failed for %s on %s: %s", symbol, date_str, e)
            return []

    result, _quality = classify_news_for_date(
        articles,
        date_str,
        n_articles,
        allow_fallback=allow_fallback,
        price_change_pct=price_change_pct,
        symbol=symbol,
        exclude_titles=exclude_titles,
    )
    return result


def select_chart_event_rows(
    df: pd.DataFrame,
    max_annotations: int = 14,
    *,
    min_visible: int = 8,
    notable_volume: float = 1.25,
    notable_price: float = 0.008,
    event_move: float = 0.012,
    win_start: Optional[Any] = None,
) -> List[Tuple[Any, Any, str]]:
    """Pick volume/move event rows — no network I/O.

    ``win_start`` (datetime.date) when set keeps only that calendar window so
    short charts (1D/5D) never pay for news fetches they will discard.
    """
    if df is None or df.empty or "is_significant" not in df.columns:
        return []
    if "volume_ratio" not in df.columns or "price_change_pct" not in df.columns:
        return []

    used_dates: set = set()
    rows: List[Tuple[Any, Any, str]] = []

    def _in_window(date_str: str) -> bool:
        if win_start is None:
            return True
        try:
            d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except Exception:
            return False
        return d >= win_start

    sig = df[df["is_significant"]].copy()
    if not sig.empty:
        sig = sig.nlargest(max_annotations * 2, "volume_ratio")
        for date, row in sig.iterrows():
            date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
            if date_str in used_dates or not _in_window(date_str):
                continue
            used_dates.add(date_str)
            rows.append((date, row, "significant"))
            if len(rows) >= max_annotations:
                return rows

    need = max(0, min_visible - len(rows))
    if need <= 0 and len(rows) >= max_annotations:
        return rows[:max_annotations]

    if len(rows) < max(min_visible, 1) or len(rows) < max_annotations:
        need = max_annotations - len(rows)
        notable_mask = (
            (df["volume_ratio"] >= notable_volume)
            & (df["price_change_pct"].abs() >= notable_price)
            & (~df["is_significant"].fillna(False))
        )
        event_mask = (
            (df["price_change_pct"].abs() >= event_move)
            & (~df["is_significant"].fillna(False))
            & (~notable_mask.fillna(False))
        )
        pool = df[notable_mask | event_mask].copy()
        if not pool.empty:
            pool = pool.assign(
                _rank=pool["volume_ratio"].fillna(1)
                * (1.0 + pool["price_change_pct"].abs().fillna(0) * 20.0)
            )
            pool = pool.nlargest(max(need * 3, need), "_rank")
            for date, row in pool.iterrows():
                if len(rows) >= max_annotations:
                    break
                date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
                if date_str in used_dates or not _in_window(date_str):
                    continue
                used_dates.add(date_str)
                try:
                    is_event = abs(float(row["price_change_pct"])) >= event_move and not (
                        float(row["volume_ratio"]) >= notable_volume
                        and abs(float(row["price_change_pct"])) >= notable_price
                    )
                except Exception:
                    is_event = False
                rows.append((date, row, "event_move" if is_event else "notable"))

    return rows


def annotations_from_rows(
    rows: List[Tuple[Any, Any, str]],
    symbol: str,
    *,
    skip_news: bool = False,
    include_archives: bool = True,
) -> List[Dict]:
    """Attach headlines (unless ``skip_news``) and build annotation dicts."""
    if not rows:
        return []

    claimed_titles: Set[str] = set()
    today = datetime.utcnow().date()
    annotations: List[Dict] = []
    for date, row, tier in rows:
        date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
        try:
            age_days = (today - datetime.strptime(date_str, "%Y-%m-%d").date()).days
        except Exception:
            age_days = 999
        allow_fb = age_days <= 2
        try:
            pc = float(row["price_change_pct"])
        except Exception:
            pc = None

        news: List[Dict] = []
        if not skip_news:
            news = get_news_for_date(
                symbol,
                date_str,
                n_articles=4,
                allow_fallback=allow_fb,
                price_change_pct=pc,
                exclude_titles=list(claimed_titles),
                include_archives=include_archives,
            )

        link_quality = LINK_SAME_DAY
        if news:
            link_quality = str(news[0].get("link_quality") or LINK_SAME_DAY)
            for n in news:
                claimed_titles.add(_normalize_title(str(n.get("title") or "")))

        bullish = row.get("candle_type") == "bullish"
        if tier == "significant":
            color = "#00FF88" if bullish else "#FF4444"
            mark_text = "N"
            mark_name = "Full volume spike"
            color_meaning = (
                "green = up-day spike" if bullish else "red = down-day spike"
            )
            tier_note = (
                f"<br><b>[{mark_text}] {mark_name}</b> — session hit the full "
                "volume/move spike bar (≥~2× vol with ≥~2% move, or ≥~3× vol). "
                f"Color: {color_meaning}."
            )
        elif tier == "event_move":
            color = "#7EB6FF" if bullish else "#C084FC"
            mark_text = "E"
            mark_name = "Large session move"
            color_meaning = (
                "blue = up-day move" if bullish else "purple = down-day move"
            )
            tier_note = (
                f"<br><b>[{mark_text}] {mark_name}</b> — big price change "
                "(≥~1.2%) without extreme volume. "
                f"Color: {color_meaning}."
            )
        else:
            color = "#F0C75E" if bullish else "#E89B6B"
            mark_text = "n"
            mark_name = "Notable volume"
            color_meaning = (
                "gold = up-day notable" if bullish else "orange = down-day notable"
            )
            tier_note = (
                f"<br><b>[{mark_text}] {mark_name}</b> — elevated volume "
                "(below the full 2×/2% or 3× spike bar). "
                f"Color: {color_meaning}."
            )

        if news:
            news_lines = []
            for a in news[:4]:
                title = str(a.get("title", ""))[:80]
                src = a.get("source", "")
                q = a.get("link_quality") or LINK_SAME_DAY
                tag = "" if q == LINK_SAME_DAY else " [may not be same-day]"
                score = a.get("event_link_score")
                score_s = f" · link {score:.2f}" if isinstance(score, (int, float)) else ""
                news_lines.append(f"• {title} [{src}]{tag}{score_s}")
            honesty = (
                ""
                if link_quality == LINK_SAME_DAY
                else "<br><i>Headlines may not be same-day (fallback)</i>"
            )
            hover = (
                f"<b>{date_str}</b><br>"
                f"Vol: {row['volume_ratio']:.1f}x avg<br>"
                f"Move: {row['price_change_pct'] * 100:+.1f}%<br><br>"
                + "<br>".join(news_lines)
                + honesty
                + tier_note
            )
        else:
            hover = (
                f"<b>{date_str}</b><br>"
                f"Vol: {row['volume_ratio']:.1f}x avg<br>"
                f"Move: {row['price_change_pct'] * 100:+.1f}%<br>"
                + (
                    "No dated breaking news found for this session"
                    " (feeds are recent-only — not a historical archive)"
                    if not skip_news
                    else "Volume/price mark"
                )
                + tier_note
            )
            link_quality = LINK_SAME_DAY

        high_val = row.get("High") or row.get("high") or row.get("close")
        try:
            price_val = float(high_val) * 1.005 if high_val is not None else float(
                row.get("close", 0)
            )
        except Exception:
            price_val = float(row.get("close", 0) or 0)

        annotations.append(
            {
                "date": date_str,
                "price": price_val,
                "text": mark_text,
                "hover": hover,
                "color": color,
                "volume_ratio": float(row["volume_ratio"]),
                "price_change_pct": float(row["price_change_pct"]),
                "news": news,
                "link_quality": link_quality,
                "date_confirmed": link_quality == LINK_SAME_DAY and bool(news),
                "tier": tier,
            }
        )

    return annotations


def build_chart_annotations(
    df: pd.DataFrame,
    symbol: str,
    max_annotations: int = 14,
    *,
    min_visible: int = 8,
    notable_volume: float = 1.25,
    notable_price: float = 0.008,
    event_move: float = 0.012,
    win_start: Optional[Any] = None,
    skip_news: bool = False,
    include_archives: bool = True,
) -> List[Dict]:
    """
    For each significant candle, build an annotation dict with
    news headlines as hover text + honesty flags.

    Pass ``win_start`` (date) to clip candidates *before* any news I/O — required
    for fast 1D/5D chart loads. Set ``skip_news=True`` for mark placement only.
    ``include_archives=False`` skips GDELT (use for ≤1W charts).
    """
    rows = select_chart_event_rows(
        df,
        max_annotations=max_annotations,
        min_visible=min_visible,
        notable_volume=notable_volume,
        notable_price=notable_price,
        event_move=event_move,
        win_start=win_start,
    )
    return annotations_from_rows(
        rows,
        symbol,
        skip_news=skip_news,
        include_archives=include_archives,
    )
