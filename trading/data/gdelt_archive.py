# -*- coding: utf-8 -*-
"""Free dated news archive via GDELT 2.1 DOC API.

Yahoo/RSS are recent-only. GDELT indexes global news continuously and
supports ``startdatetime`` / ``enddatetime`` queries — the cheapest
practical archive for chart volume-day linkage without paid terminals.
Soft-fails everywhere (timeouts / empty / malformed).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from trading.utils.data_manager import disk_cache_get, disk_cache_set

logger = logging.getLogger(__name__)


def fetch_gdelt_headlines(
    query: str,
    date_str: str,
    max_items: int = 8,
    *,
    window_hours: int = 18,
) -> List[Dict[str, Any]]:
    """
    Headlines for ``query`` (ticker / company) around ``date_str`` (YYYY-MM-DD).

    Uses GDELT DOC ArtList JSON. Returns normalized article dicts compatible
    with ``volume_news_linker``.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        day = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except Exception:
        return []

    cache_key = f"gdelt:v1:{q}:{date_str[:10]}:{max_items}:{window_hours}"
    cached = disk_cache_get(cache_key)
    if cached is not None:
        return list(cached)

    start = day - timedelta(hours=6)
    end = day + timedelta(hours=window_hours)
    params = {
        "query": f"({q}) sourcelang:english",
        "mode": "ArtList",
        "maxrecords": max(1, min(int(max_items), 75)),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
        "format": "json",
    }
    out: List[Dict[str, Any]] = []
    try:
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params,
            timeout=14,
        )
        if r.status_code >= 400:
            logger.debug("GDELT HTTP %s: %s", r.status_code, r.text[:160])
            disk_cache_set(cache_key, [], ttl=1800)
            return []
        # GDELT sometimes returns bare text on errors
        try:
            data = r.json() or {}
        except Exception:
            disk_cache_set(cache_key, [], ttl=900)
            return []
        arts = data.get("articles") or []
        for a in arts:
            if not isinstance(a, dict):
                continue
            title = str(a.get("title") or "").strip()
            if not title:
                continue
            # seendate is YYYYMMDDHHMMSS
            seen = str(a.get("seendate") or "")
            published = ""
            if len(seen) >= 8:
                try:
                    published = datetime.strptime(seen[:14], "%Y%m%d%H%M%S").isoformat()
                except Exception:
                    published = f"{seen[0:4]}-{seen[4:6]}-{seen[6:8]}"
            out.append({
                "title": title[:280],
                "url": str(a.get("url") or ""),
                "source": str(a.get("domain") or "GDELT"),
                "published": published or date_str,
                "summary": title[:300],
                "symbols": [q.upper()] if len(q) <= 6 else [],
                "source_type": "gdelt",
                "archive": True,
            })
            if len(out) >= max_items:
                break
    except Exception as e:
        logger.debug("GDELT fetch failed for %s @ %s: %s", q, date_str, e)
        return []

    disk_cache_set(cache_key, out, ttl=3600)
    return out
