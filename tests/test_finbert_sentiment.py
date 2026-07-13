# -*- coding: utf-8 -*-
"""FinBERT + VADER headline sentiment (Phase 1).

Hand-picked headlines where general lexicons often miss finance-positive
framing; blend math is verified with a mocked FinBERT path so CI does not
require model weights.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from trading.data import social_sentiment as SS


# Finance-positive framing that VADER often reads flat/negative;
# finbert-tone reliably marks the first two Positive (verified).
HAND_PICKED_POSITIVE = [
    "Acme beats guided-down estimates despite soft revenue print",
    "In-line quarter amid sector headwinds; margins hold",
    "Company reports record quarterly profit and raises guidance",
]

HAND_PICKED_NEGATIVE = [
    "Regulator opens probe into accounting irregularities",
    "Sudden CFO resignation after restatement warning",
]


class TestBlendMath:
    def test_finbert_weighted_blend(self):
        # 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        blended, eng = SS.blend_finbert_vader([1.0, -1.0], [0.0, 0.0])
        assert eng == "finbert+vader"
        assert blended[0] == pytest.approx(0.7)
        assert blended[1] == pytest.approx(-0.7)

    def test_vader_only_when_finbert_missing(self):
        blended, eng = SS.blend_finbert_vader(None, [0.5, -0.2])
        assert eng == "vader"
        assert blended == [0.5, -0.2]

    def test_length_mismatch_falls_back_to_vader(self):
        blended, eng = SS.blend_finbert_vader([0.1], [0.5, -0.2])
        assert eng == "vader"
        assert blended == [0.5, -0.2]


class TestFinbertLabelMap:
    def test_labels(self):
        assert SS._finbert_label_to_score("positive", 0.9) == pytest.approx(0.9)
        assert SS._finbert_label_to_score("negative", 0.8) == pytest.approx(-0.8)
        assert SS._finbert_label_to_score("neutral", 0.95) == pytest.approx(0.0)


class TestScoreTextsFinbertMocked:
    def setup_method(self):
        SS.reset_finbert_for_tests()

    def teardown_method(self):
        SS.reset_finbert_for_tests()

    def test_batch_scoring_and_cache(self):
        mock_pipe = MagicMock(
            return_value=[
                {"label": "positive", "score": 0.91},
                {"label": "negative", "score": 0.88},
            ]
        )
        with patch.object(SS, "_get_finbert_pipeline", return_value=mock_pipe):
            scores = SS.score_texts_finbert(
                ["beats guided-down estimates", "accounting probe opened"]
            )
            assert scores is not None
            assert scores[0] == pytest.approx(0.91)
            assert scores[1] == pytest.approx(-0.88)
            # second call hits cache — pipe not called again for same texts
            mock_pipe.reset_mock()
            scores2 = SS.score_texts_finbert(
                ["beats guided-down estimates", "accounting probe opened"]
            )
            assert scores2 == scores
            mock_pipe.assert_not_called()

    def test_pipeline_failure_returns_none(self):
        with patch.object(SS, "_get_finbert_pipeline", return_value=None):
            assert SS.score_texts_finbert(["anything"]) is None


class TestNewsPathGraceful:
    def setup_method(self):
        SS.reset_finbert_for_tests()

    def teardown_method(self):
        SS.reset_finbert_for_tests()

    def test_news_sentiment_falls_back_when_finbert_down(self):
        articles = [
            {
                "title": "Acme beats guided-down estimates",
                "summary": "",
                "source": "test",
                "url": "",
            }
        ]
        with patch(
            "trading.data.news_aggregator.get_news", return_value=articles
        ), patch.object(SS, "score_texts_finbert", return_value=None):
            out = SS.get_news_headline_sentiment("ACME", max_items=5)
        assert out["success"] is True
        assert out["engine"] == "vader"
        assert out["mention_count"] == 1

    def test_news_sentiment_blends_when_finbert_ok(self):
        articles = [
            {
                "title": "Acme beats guided-down estimates",
                "summary": "",
                "source": "test",
                "url": "",
            }
        ]
        with patch(
            "trading.data.news_aggregator.get_news", return_value=articles
        ), patch.object(SS, "score_texts_finbert", return_value=[0.9]):
            out = SS.get_news_headline_sentiment("ACME", max_items=5)
        assert out["success"] is True
        assert out["engine"] == "finbert+vader"
        assert "finbert" in (out["top_posts"][0] or {})


class TestHandPickedHeadlines:
    """If FinBERT weights are present, show domain edge vs VADER.

    Skips cleanly when the model is not downloadable in CI.
    """

    def setup_method(self):
        SS.reset_finbert_for_tests()

    def teardown_method(self):
        SS.reset_finbert_for_tests()

    def test_finbert_more_positive_on_finance_jargon(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        pipe = SS._get_finbert_pipeline()
        if pipe is None:
            pytest.skip("FinBERT weights/pipeline not available in this environment")

        analyzer = SentimentIntensityAnalyzer()
        fb = SS.score_texts_finbert(HAND_PICKED_POSITIVE)
        assert fb is not None and len(fb) == len(HAND_PICKED_POSITIVE)
        vader = [
            float(analyzer.polarity_scores(t).get("compound", 0.0))
            for t in HAND_PICKED_POSITIVE
        ]
        # Domain model should read the batch as at least as constructive as
        # VADER on average (finance-positive framing).
        assert sum(fb) / len(fb) > sum(vader) / len(vader) - 0.05
        # At least one headline where FinBERT is clearly more positive
        assert any(f > v + 0.15 for f, v in zip(fb, vader)) or (
            sum(fb) / len(fb) > sum(vader) / len(vader) + 0.05
        )

    def test_finbert_negative_on_probe_headline(self):
        pipe = SS._get_finbert_pipeline()
        if pipe is None:
            pytest.skip("FinBERT weights/pipeline not available in this environment")
        fb = SS.score_texts_finbert(HAND_PICKED_NEGATIVE)
        assert fb is not None
        assert sum(fb) / len(fb) < 0.0


class TestLatencyBenchmark:
    def test_benchmark_reports_shape(self):
        SS.reset_finbert_for_tests()
        # Mock unavailable — still returns a structured report
        with patch.object(SS, "_get_finbert_pipeline", return_value=None):
            rep = SS.benchmark_finbert_latency(n_headlines=15, warmup=False)
        assert rep["available"] is False
        assert rep["n_headlines"] == 15
        assert "load_s" in rep
