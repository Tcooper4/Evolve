# -*- coding: utf-8 -*-
"""Detect template / placeholder API credentials (e.g. from .env examples)."""

from __future__ import annotations

PLACEHOLDER_PATTERNS = frozenset(
    {
        "your-reddit-client-id",
        "your-reddit-client-secret",
        "your_reddit_client_id",
        "your_reddit_client_secret",
        "placeholder",
        "changeme",
        "replace_me",
        "your-",
        "your_",
        "<",
    }
)


def is_placeholder_credential(value: str | None) -> bool:
    """True if value is empty or matches known placeholder / template patterns."""
    if value is None:
        return True
    v = str(value).lower().strip()
    if not v:
        return True
    return any(v == p or v.startswith(p) for p in PLACEHOLDER_PATTERNS)
