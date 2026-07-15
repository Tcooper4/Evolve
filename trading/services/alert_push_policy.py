# -*- coding: utf-8 -*-
"""Notification-layer policy for alert pushes (not execution).

- ``watch``: visible in-app / still evaluates & one-shots; no WebSocket push.
- ``action``: may push via notification_hub (subject to per-symbol rate limit).

Legacy alerts with no ``mode`` keep prior behavior (treated as ``action``).
New alerts should default to ``watch`` at upsert time.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

MODE_WATCH = "watch"
MODE_ACTION = "action"
VALID_MODES = frozenset({MODE_WATCH, MODE_ACTION})

# Conservative default: a few distinct action alerts on one ticker / hour.
DEFAULT_PUSH_PER_SYMBOL_HOUR = 3


def normalize_alert_mode(raw: Any, *, default: str = MODE_ACTION) -> str:
    """Return watch|action. Missing/unknown → ``default`` (action preserves legacy)."""
    mode = str(raw or "").strip().lower()
    if mode in VALID_MODES:
        return mode
    return default if default in VALID_MODES else MODE_ACTION


def alert_allows_push(alert_or_row: Dict[str, Any]) -> bool:
    """True only for action-mode alerts (legacy missing mode = action)."""
    if not isinstance(alert_or_row, dict):
        return False
    return normalize_alert_mode(alert_or_row.get("mode"), default=MODE_ACTION) == MODE_ACTION


def push_rate_limit_max() -> int:
    """``EVOLVE_ALERT_PUSH_PER_SYMBOL_HOUR`` (default 3, min 1)."""
    raw = (os.getenv("EVOLVE_ALERT_PUSH_PER_SYMBOL_HOUR") or "").strip()
    if not raw:
        return DEFAULT_PUSH_PER_SYMBOL_HOUR
    try:
        return max(1, int(raw))
    except Exception:
        return DEFAULT_PUSH_PER_SYMBOL_HOUR


class AlertPushRateLimiter:
    """Per-user, per-symbol sliding window for *pushes only*."""

    def __init__(
        self,
        max_per_symbol_hour: Optional[int] = None,
        window_seconds: float = 3600.0,
    ) -> None:
        self.max_per_symbol_hour = (
            int(max_per_symbol_hour)
            if max_per_symbol_hour is not None
            else push_rate_limit_max()
        )
        self.window_seconds = float(window_seconds)
        self._events: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def _prune(self, key: Tuple[str, str], now: float) -> None:
        cutoff = now - self.window_seconds
        self._events[key] = [t for t in self._events[key] if t >= cutoff]

    def allow(
        self,
        username: str,
        symbol: str,
        *,
        now: Optional[float] = None,
        record: bool = True,
    ) -> bool:
        """Return True if a push is within quota. When True and ``record``, count it."""
        user = (username or "").strip()
        sym = (symbol or "").strip().upper()
        if not user or not sym:
            return False
        ts = float(now if now is not None else time.time())
        key = (user, sym)
        self._prune(key, ts)
        if len(self._events[key]) >= self.max_per_symbol_hour:
            return False
        if record:
            self._events[key].append(ts)
        return True

    def reset(self) -> None:
        self._events.clear()


# Process-wide limiter shared by the background loop.
alert_push_rate_limiter = AlertPushRateLimiter()


def should_push_alert_notification(
    username: str,
    row: Dict[str, Any],
    *,
    limiter: Optional[AlertPushRateLimiter] = None,
    now: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Notification gate after an alert has already executed/triggered.

    Returns ``(push?, reason)`` where reason is ``ok`` | ``watch`` | ``rate_limited``.
    Execution must already have happened — this never blocks one-shot stamping.
    """
    if not alert_allows_push(row):
        return False, "watch"
    lim = limiter if limiter is not None else alert_push_rate_limiter
    sym = str(row.get("symbol") or "")
    if not lim.allow(username, sym, now=now, record=True):
        return False, "rate_limited"
    return True, "ok"
