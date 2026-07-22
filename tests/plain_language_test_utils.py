# -*- coding: utf-8 -*-
"""Shared assertions for plain_language fields on technical outputs."""

from __future__ import annotations

from trading.utils.plain_language import (
    BANNED_JARGON_TERMS,
    find_banned_jargon,
    is_plain_language_length_ok,
)


def assert_plain_language_field(text: str) -> None:
    assert text and str(text).strip(), "plain_language must be non-empty"
    banned = find_banned_jargon(str(text), BANNED_JARGON_TERMS)
    assert not banned, f"plain_language contains banned jargon: {banned!r} in {text!r}"
    assert is_plain_language_length_ok(str(text)), (
        f"plain_language too long ({len(str(text))} chars): {text!r}"
    )
