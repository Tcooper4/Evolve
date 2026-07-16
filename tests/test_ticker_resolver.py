# -*- coding: utf-8 -*-
"""Ticker normalize / typo / near-miss resolution."""

from __future__ import annotations

from trading.data.ticker_resolver import (
    normalize_ticker,
    resolve_ticker,
    suggest_ticker,
)


class TestNormalizeTypos:
    def test_appl_maps_to_aapl(self):
        assert normalize_ticker("appl") == "AAPL"
        assert normalize_ticker("APPL") == "AAPL"

    def test_valid_passthrough(self):
        assert normalize_ticker("aapl") == "AAPL"
        assert normalize_ticker("MSFT") == "MSFT"
        assert normalize_ticker("spy") == "SPY"

    def test_index_alias(self):
        assert normalize_ticker("SPX") == "^GSPC"


class TestSuggestNearMiss:
    def test_appl_suggests_aapl(self):
        # Even without typo table, edit distance 1 should hit AAPL
        assert suggest_ticker("APPL") == "AAPL"

    def test_exact_not_suggested(self):
        assert suggest_ticker("AAPL") is None or suggest_ticker("AAPL") != "AAPL"


class TestResolveWithoutNetwork:
    def test_resolve_validate_false_uses_typo_table(self):
        assert resolve_ticker("appl", validate=False) == "AAPL"
