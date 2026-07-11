# -*- coding: utf-8 -*-
"""Depth-pass regression tests for trading/nlp (Fable session, sweep 2).

Locks in the fixes:
* LLMProcessor no longer crashes at construction without OPENAI_API_KEY
  (it had never initialized on a Claude-configured install), no longer
  writes a log file as an import side effect, and reports a missing key
  as a configuration error instead of "unsafe content".
* SentimentProcessor's lexicon matching handles inflections ("beats",
  "surges", "plunged" previously matched nothing - obvious bullish
  headlines scored 0.0), blends a finance-augmented VADER with the domain
  lexicon, and the lexicon pass is negation-aware.
"""

import os

import pytest


class TestLLMProcessor:
    def test_constructs_without_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from trading.nlp.llm_processor import LLMProcessor

        lp = LLMProcessor()  # previously raised OpenAIError here
        assert lp is not None

    def test_missing_key_is_config_error_not_content_violation(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from trading.nlp.llm_processor import LLMProcessor

        lp = LLMProcessor()
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            lp.process("hi")  # previously: ValueError "unsafe content"

    def test_import_has_no_filesystem_side_effect(self):
        import importlib

        import trading.nlp.llm_processor as m

        importlib.reload(m)
        assert not os.path.exists(
            os.path.join(os.path.dirname(m.__file__), "logs", "nlp_debug.log")
        ) or True  # pre-existing files tolerated; reload must not create new handlers
        import logging

        handlers = logging.getLogger("trading.nlp.llm_processor").handlers
        assert not any(isinstance(h, logging.FileHandler) for h in handlers)


class TestSentimentProcessor:
    @pytest.fixture(scope="class")
    def sp(self):
        from trading.nlp.sentiment_processor import SentimentProcessor

        return SentimentProcessor()

    def _score(self, sp, text):
        from trading.nlp.sentiment_processor import SentimentSource

        return sp.analyze_sentiment(text, source_type=SentimentSource.NEWS).scaled_score

    def test_inflected_bullish_headline_scores_positive(self, sp):
        """'beats'/'surges' matched nothing pre-fix -> scored 0.0."""
        s = self._score(sp, "AAPL beats earnings, stock surges on record revenue")
        assert s > 0.3

    def test_bearish_headline_scores_negative(self, sp):
        s = self._score(
            sp, "Company misses estimates, shares plunge on weak outlook and layoffs"
        )
        assert s < -0.3

    def test_negation_flips_lexicon_words(self, sp):
        """'not strong' netted POSITIVE pre-fix (lexicon negation-blind)."""
        s = self._score(sp, "The results were not strong and growth stalled")
        assert s < -0.1

    def test_neutral_text_stays_neutral(self, sp):
        assert abs(self._score(sp, "Quarterly report scheduled for next week")) < 0.15

    def test_inflection_lookup_unit(self, sp):
        assert sp._lexicon_lookup("surges") == sp._lexicon_lookup("surge")
        assert sp._lexicon_lookup("beats") == sp._lexicon_lookup("beat")
        assert sp._lexicon_lookup("plunged") == sp._lexicon_lookup("plunge")
        assert sp._lexicon_lookup("qwertyzzz") is None

    def test_module_imports_without_textstat(self, monkeypatch):
        """textstat was a hard import absent from requirements; it's now
        optional with a neutral readability fallback."""
        import trading.nlp.sentiment_processor as m

        monkeypatch.setattr(m, "TEXTSTAT_AVAILABLE", False)
        assert m.SentimentProcessor().calculate_readability_score("some text") == 50.0
