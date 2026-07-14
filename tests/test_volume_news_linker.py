# -*- coding: utf-8 -*-
"""Honesty flags + event-link ranking for volume↔news overlay."""

from __future__ import annotations

from trading.analysis.volume_news_linker import (
    LINK_FALLBACK,
    LINK_SAME_DAY,
    classify_news_for_date,
    score_article_for_event,
)


def _art(title: str, published: str, source: str = "wire", **extra) -> dict:
    row = {"title": title, "published": published, "source": source}
    row.update(extra)
    return row


class TestClassifySameDay:
    def test_window_match_is_same_day(self):
        articles = [
            _art("Earnings beat", "2024-06-14T15:00:00", source="Reuters"),
            _art("Unrelated last week", "2024-06-01T12:00:00"),
            _art("Also near day", "2024-06-15T09:00:00", source="CNBC"),
        ]
        tagged, quality = classify_news_for_date(
            articles, "2024-06-14", n_articles=5, symbol="AAPL",
        )
        assert quality == LINK_SAME_DAY
        assert len(tagged) >= 1
        assert all(a["link_quality"] == LINK_SAME_DAY for a in tagged)
        assert all(a["date_confirmed"] is True for a in tagged)
        titles = {a["title"] for a in tagged}
        assert "Unrelated last week" not in titles

    def test_no_window_match_falls_back_with_flag(self):
        articles = [
            _art("Old story A about AAPL", "2024-01-01T10:00:00"),
            _art("Old story B about AAPL", "2024-01-02T10:00:00"),
            _art("Old story C about AAPL", "2024-01-03T10:00:00"),
            _art("Old story D about AAPL", "2024-01-04T10:00:00"),
        ]
        tagged, quality = classify_news_for_date(
            articles, "2024-06-14", n_articles=5, symbol="AAPL", allow_fallback=True,
        )
        assert quality == LINK_FALLBACK
        assert len(tagged) >= 1
        assert all(a["link_quality"] == LINK_FALLBACK for a in tagged)
        assert all(a["date_confirmed"] is False for a in tagged)

    def test_no_fallback_when_disabled(self):
        articles = [
            _art("Old story A about AAPL", "2024-01-01T10:00:00"),
        ]
        tagged, quality = classify_news_for_date(
            articles, "2024-06-14", n_articles=5, symbol="AAPL", allow_fallback=False,
        )
        assert tagged == []
        assert quality == LINK_SAME_DAY

    def test_exclude_titles_prevents_reuse(self):
        articles = [
            _art("AAPL jumps on earnings", "2024-06-14T12:00:00", source="Reuters"),
            _art("Second AAPL story", "2024-06-14T13:00:00", source="CNBC"),
        ]
        tagged, _ = classify_news_for_date(
            articles,
            "2024-06-14",
            symbol="AAPL",
            exclude_titles=["AAPL jumps on earnings"],
        )
        titles = {a["title"] for a in tagged}
        assert "AAPL jumps on earnings" not in titles
        assert "Second AAPL story" in titles

    def test_empty_articles(self):
        tagged, quality = classify_news_for_date([], "2024-06-14")
        assert tagged == []
        assert quality == LINK_SAME_DAY  # nothing claimed


class TestEventLinkScore:
    def test_breaking_wire_beats_evergreen(self):
        breaking = _art(
            "$SPY sinking after CPI miss",
            "2024-06-14T14:00:00",
            source="@WalterBloomberg",
            source_type="twitter",
            breaking=True,
        )
        fluff = _art(
            "The Boring Dividend ETF Everyone Ignored Just Crushed the S&P",
            "2024-06-14T14:00:00",
            source="24/7 Wall St.",
        )
        s_break = score_article_for_event(breaking, "SPY", "2024-06-14", -0.02)
        s_fluff = score_article_for_event(fluff, "SPY", "2024-06-14", -0.02)
        assert s_break > s_fluff
