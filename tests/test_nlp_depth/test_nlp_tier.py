# -*- coding: utf-8 -*-
"""NLP-tier depth tests: sentiment processor behavior and the
ticker-extraction false-positive fix (re.IGNORECASE had nullified the
uppercase pattern, so every short word became a 'ticker')."""

import pytest


class TestSentimentProcessor:
    @pytest.fixture()
    def sp(self):
        from trading.nlp.sentiment_processor import SentimentProcessor
        return SentimentProcessor()

    def test_polarity_directions(self, sp):
        pos = sp.analyze_sentiment(
            "Earnings beat expectations, strong growth and record profits"
        )
        neg = sp.analyze_sentiment(
            "Revenue collapsed, massive losses, guidance cut"
        )
        assert pos.base_score > 0.3
        assert neg.base_score < -0.3

    def test_negation_flips_polarity(self, sp):
        r = sp.analyze_sentiment("The results were not strong and growth stalled")
        assert r.base_score < 0

    def test_tweet_impact_bounded(self, sp):
        from trading.nlp.sentiment_processor import TweetMetrics
        tm = TweetMetrics(retweet_count=500, like_count=2000, reply_count=100,
                          quote_count=50, followers_count=100000, verified=True,
                          account_age_days=2000, tweet_age_hours=2)
        assert 0.0 <= sp.calculate_tweet_impact_score(tm) <= 1.0


class TestTickerExtraction:
    @pytest.fixture()
    def nli(self):
        from nlp.natural_language_insights import NaturalLanguageInsights
        return NaturalLanguageInsights()

    def test_no_short_word_false_positives(self, nli):
        out = nli.extract_tickers(
            "I'm bullish on $AAPL and TSLA after earnings. "
            "The CEO said IT and AI are key."
        )
        assert sorted(t["ticker"] for t in out) == ["AAPL", "TSLA"]

    def test_cashtag_bypasses_stoplist_and_case(self, nli):
        out = nli.extract_tickers("Loading up on $ai calls before the print")
        assert [t["ticker"] for t in out] == ["AI"]

    def test_cashtag_and_bare_dedupe(self, nli):
        out = nli.extract_tickers("$NVDA is ripping. NVDA calls printing.")
        assert [t["ticker"] for t in out] == ["NVDA"]

    def test_lowercase_prose_yields_nothing(self, nli):
        assert nli.extract_tickers("the quick brown fox jumps over it") == []
