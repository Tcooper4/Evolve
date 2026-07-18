# -*- coding: utf-8 -*-
"""Redact API keys / tokens from strings before they hit logs or error text.

Used when logging exceptions that may embed request URLs (``apiKey=…``)
or provider error bodies. Never rely on this as the only control —
prefer not putting secrets in URLs or exception messages at all.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# Common shapes: NewsAPI query params, sk- / sk-ant- / Bearer tokens.
_PATTERNS = (
    re.compile(r"(apiKey=)([^&\s]+)", re.IGNORECASE),
    re.compile(r"(access_token=)([^&\s]+)", re.IGNORECASE),
    re.compile(r"(Bearer\s+)([A-Za-z0-9\-._~+/]+=*)", re.IGNORECASE),
    re.compile(r"\b(sk-ant-api\d{2}-)[A-Za-z0-9\-_]+"),
    re.compile(r"\b(sk-)[A-Za-z0-9]{20,}"),
    re.compile(r"\b(xox[baprs]-)[A-Za-z0-9-]+"),  # slack-ish
)


def redact_secrets(
    text: str,
    known: Optional[Iterable[str]] = None,
) -> str:
    """Return ``text`` with known secrets and common key patterns masked."""
    if not text:
        return text
    out = str(text)
    for secret in known or ():
        if secret and isinstance(secret, str) and len(secret) >= 6:
            out = out.replace(secret, "***REDACTED***")
    for pat in _PATTERNS:
        out = pat.sub(
            lambda m: (
                f"{m.group(1)}***REDACTED***"
                if m.lastindex and m.lastindex >= 2
                else "***REDACTED***"
            ),
            out,
        )
    return out


__all__ = ["redact_secrets"]
