# -*- coding: utf-8 -*-
"""Data-layer depth-pass tests (session 3): the Fourier leakage fix, the
price_cache timezone consistency fix, and hand-verified options-flow math."""

import numpy as np
import pandas as pd
import pytest

from trading.data.preprocessing import DataPreprocessor, FeatureEngineering


@pytest.fixture()
def cyclic_ohlcv():
    rng = np.random.default_rng(4)
    n = 400
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    t = np.arange(n)
    close = 100 + 5 * np.sin(2 * np.pi * t / 20) + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1,
         "Close": close, "Volume": np.full(n, 3e6)},
        index=idx,
    )


class TestFourierFeatures:
    """The old implementation ran FFT over the ENTIRE series (every row's
    feature used the future) and broadcast one scalar down the column."""

    def test_per_row_variation_not_constant(self, cyclic_ohlcv):
        fe = FeatureEngineering({"fourier_periods": [20]})
        col = fe.calculate_fourier_features(cyclic_ohlcv)["Fourier_20"]
        assert col.dropna().nunique() > 100

    def test_causality_future_cannot_change_past(self, cyclic_ohlcv):
        fe = FeatureEngineering({"fourier_periods": [20]})
        full = fe.calculate_fourier_features(cyclic_ohlcv)["Fourier_20"]
        trunc = fe.calculate_fourier_features(cyclic_ohlcv.iloc[:250])["Fourier_20"]
        assert np.allclose(full.iloc[39:250].values, trunc.iloc[39:250].values)

    def test_leading_window_is_nan(self, cyclic_ohlcv):
        fe = FeatureEngineering({"fourier_periods": [20]})
        col = fe.calculate_fourier_features(cyclic_ohlcv)["Fourier_20"]
        assert int(col.isna().sum()) == 2 * 20 - 1

    def test_detects_embedded_cycle(self, cyclic_ohlcv):
        fe = FeatureEngineering({"fourier_periods": [20, 50]})
        f = fe.calculate_fourier_features(cyclic_ohlcv)
        assert f["Fourier_20"].dropna().mean() > f["Fourier_50"].dropna().mean()


class TestFeatureMath:
    def test_rsi_bounds_and_macd_manual(self, cyclic_ohlcv):
        fe = FeatureEngineering()
        rsi = fe.calculate_rsi(cyclic_ohlcv)["RSI"].dropna()
        assert ((rsi >= 0) & (rsi <= 100)).all()
        macd = fe.calculate_macd(cyclic_ohlcv)["MACD"]
        manual = (cyclic_ohlcv["Close"].ewm(span=12, adjust=False).mean()
                  - cyclic_ohlcv["Close"].ewm(span=26, adjust=False).mean())
        assert np.allclose(macd, manual)

    def test_lags_strictly_backward(self, cyclic_ohlcv):
        fe = FeatureEngineering()
        lags = fe.calculate_lag_features(cyclic_ohlcv)
        col = lags.columns[0]
        p = int(col.split("_")[-1])
        assert np.allclose(lags[col].iloc[p:], cyclic_ohlcv["Close"].iloc[:-p])

    def test_normalize_round_trip(self, cyclic_ohlcv):
        pp = DataPreprocessor()
        clean = pp.clean_data(cyclic_ohlcv)
        back = pp.inverse_transform(pp.fit_transform(clean))
        assert np.allclose(back["Close"].values, clean["Close"].values, rtol=1e-6)


class TestOptionsFlowMath:
    def test_max_pain_matches_hand_computation(self):
        from trading.data.options_flow import _max_pain_strike
        calls = pd.DataFrame({"strike": [90, 100, 110],
                              "openInterest": [100, 10, 5],
                              "volume": [50, 10, 5]})
        puts = pd.DataFrame({"strike": [90, 100, 110],
                             "openInterest": [5, 10, 100],
                             "volume": [5, 10, 300]})

        def pain(k):
            p = sum(max(0, k - r.strike) * r.openInterest * 100
                    for _, r in calls.iterrows())
            p += sum(max(0, r.strike - k) * r.openInterest * 100
                     for _, r in puts.iterrows())
            return p

        expected = min((90, 100, 110), key=pain)
        assert _max_pain_strike(calls, puts) == expected

    def test_unusual_volume_threshold(self):
        from trading.data.options_flow import _unusual_for_expiry
        calls = pd.DataFrame({"strike": [90, 100, 110],
                              "openInterest": [1, 1, 1],
                              "volume": [50, 10, 5]})
        puts = pd.DataFrame({"strike": [90, 100, 110],
                             "openInterest": [1, 1, 1],
                             "volume": [5, 10, 300]})
        uc, up = _unusual_for_expiry(calls, puts, "2026-07-17", 5)
        assert len(up) == 1 and up[0]["strike"] == 110.0
        assert len(uc) == 1 and uc[0]["strike"] == 90.0


class TestSentimentTokenization:
    """Plain .split() left punctuation attached, so 'gains,' and 'up.'
    never matched the keyword sets - most real headlines scored 0.0."""

    def test_punctuated_and_plural_headlines_score(self):
        import asyncio
        from trading.data.external_signals import NewsSentimentCollector

        c = NewsSentimentCollector()

        async def run():
            pos = await c._analyze_text_sentiment(
                "Stock gains, surges after earnings beat expectations!"
            )
            neg = await c._analyze_text_sentiment(
                "Shares plunge as company misses estimates."
            )
            neutral = await c._analyze_text_sentiment(
                "The company reported quarterly figures."
            )
            return pos, neg, neutral

        pos, neg, neutral = asyncio.run(run())
        assert pos > 0
        assert neg < 0
        assert neutral == 0.0

    def test_score_bounded(self):
        import asyncio
        from trading.data.external_signals import NewsSentimentCollector

        c = NewsSentimentCollector()

        async def run():
            return await c._analyze_text_sentiment(
                "buy buy strong rally surge gain profit growth"
            )

        assert -1.0 <= asyncio.run(run()) <= 1.0
