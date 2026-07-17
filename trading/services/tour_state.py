# -*- coding: utf-8 -*-
"""Per-user first-visit spotlight tour progress (server-side prefs).

Storage lives in ``config.user_store`` preferences under ``tours_seen``, the
same pattern as stated ``risk_tolerance``. Progress is account-scoped so
desktop + PWA share it and a cleared browser does not reset it.

Page ids must match ``web/frontend/src/App.tsx`` ``PAGES`` exactly.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional

# Exact ids from App.tsx PAGES — do not invent parallel naming.
TOUR_PAGE_IDS = (
    "dashboard",
    "analyze",
    "scanner",
    "portfolio",
    "backtest",
    "chat",
    "settings",
)
TOUR_PAGE_ID_SET = frozenset(TOUR_PAGE_IDS)
TOURS_SEEN_KEY = "tours_seen"


def normalize_tours_seen(raw: Any) -> Dict[str, bool]:
    """Coerce prefs blob → ``{page_id: bool}`` for known pages only."""
    out: Dict[str, bool] = {}
    if not isinstance(raw, Mapping):
        return out
    for page_id in TOUR_PAGE_IDS:
        if page_id not in raw:
            continue
        val = raw[page_id]
        if val is True or val is False:
            out[page_id] = bool(val)
        elif isinstance(val, (int, float)) and val in (0, 1):
            out[page_id] = bool(val)
        elif isinstance(val, str) and val.strip().lower() in ("true", "1", "yes"):
            out[page_id] = True
        elif isinstance(val, str) and val.strip().lower() in ("false", "0", "no"):
            out[page_id] = False
    return out


def get_tours_seen_from_prefs(prefs: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    """Read ``tours_seen`` from a prefs dict (missing → empty)."""
    if not prefs:
        return {}
    return normalize_tours_seen(prefs.get(TOURS_SEEN_KEY))


def mark_page_seen(
    tours_seen: Optional[Mapping[str, bool]],
    page_id: str,
) -> Dict[str, bool]:
    """Mark one page seen; leave every other page unchanged.

    Unknown page ids raise ``ValueError`` so callers cannot invent keys.
    """
    pid = str(page_id or "").strip().lower()
    if pid not in TOUR_PAGE_ID_SET:
        raise ValueError(
            f"Unknown tour page_id {page_id!r}; expected one of {list(TOUR_PAGE_IDS)}"
        )
    out = dict(normalize_tours_seen(tours_seen))
    out[pid] = True
    return out


def reset_tours_seen() -> Dict[str, bool]:
    """Empty dict — full first-time experience replays on next visits."""
    return {}


def apply_mark_page_seen(
    prefs: MutableMapping[str, Any],
    page_id: str,
) -> Dict[str, bool]:
    """Mutate prefs in place: set ``tours_seen[page_id]=true``. Returns new map."""
    updated = mark_page_seen(get_tours_seen_from_prefs(prefs), page_id)
    prefs[TOURS_SEEN_KEY] = updated
    return updated


def apply_reset_tours(prefs: MutableMapping[str, Any]) -> Dict[str, bool]:
    """Mutate prefs in place: clear ``tours_seen``. Returns empty map."""
    cleared = reset_tours_seen()
    prefs[TOURS_SEEN_KEY] = cleared
    return cleared


def load_tours_seen(session_id: str) -> Dict[str, bool]:
    """Load tours_seen for ``user:<name>`` (or raw session id). Fail → {}."""
    try:
        from config.user_store import load_user_preferences

        prefs = load_user_preferences(session_id) or {}
        return get_tours_seen_from_prefs(prefs)
    except Exception:
        return {}


def save_mark_page_seen(session_id: str, page_id: str) -> Dict[str, bool]:
    """Persist mark-one-page; returns the updated tours_seen map."""
    from config.user_store import load_user_preferences, save_user_preferences

    prefs = dict(load_user_preferences(session_id) or {})
    updated = apply_mark_page_seen(prefs, page_id)
    save_user_preferences(session_id, prefs)
    return updated


def save_reset_tours(session_id: str) -> Dict[str, bool]:
    """Persist reset-all; returns empty tours_seen."""
    from config.user_store import load_user_preferences, save_user_preferences

    prefs = dict(load_user_preferences(session_id) or {})
    updated = apply_reset_tours(prefs)
    save_user_preferences(session_id, prefs)
    return updated
