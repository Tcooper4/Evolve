# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.feature_engineering import FeatureGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime classification"""

    name: str
    description: str
    conditions: Dict
    metrics: Dict
    confidence: float


@dataclass
class MarketCondition:
    """Market condition analysis"""

    name: str
    description: str
    indicators: Dict
    signals: Dict
    strength: float


class MarketAnalyzer:
    """Market analyzer class for compatibility"""

    def __init__(self, config: Optional[Dict] = None):
        self.analysis = MarketAnalysis(config)

    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions using the underlying MarketAnalysis"""
        return self.analysis.analyze_market(data)


class MarketAnalysis:
    """Comprehensive market analysis system"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_generator = FeatureGenerator()
        self._setup_logging()
        self._initialize_indicators()

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def _initialize_indicators(self):
        """Initialize technical indicators"""
        self.indicators = {
            "trend": {
                "sma": self._calculate_sma,
                "ema": self._calculate_ema,
                "macd": self._calculate_macd,
                "adx": self._calculate_adx,
                "ichimoku": self._calculate_ichimoku,
            },
            "momentum": {
                "rsi": self._calculate_rsi,
                "stochastic": self._calculate_stochastic,
                "cci": self._calculate_cci,
                "mfi": self._calculate_mfi,
                "roc": self._calculate_roc,
            },
            "volatility": {
                "bollinger_bands": self._calculate_bollinger_bands,
                "atr": self._calculate_atr,
                "keltner_channels": self._calculate_keltner_channels,
                "donchian_channels": self._calculate_donchian_channels,
            },
            "volume": {
                "obv": self._calculate_obv,
                "vpt": self._calculate_vpt,
                "cmf": self._calculate_cmf,
                "mfi": self._calculate_mfi,
                "ad": self._calculate_ad,
            },
            "support_resistance": {
                "pivot_points": self._calculate_pivot_points,
                "fibonacci_retracement": self._calculate_fibonacci_retracement,
                "price_channels": self._calculate_price_channels,
            },
        }

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """Perform comprehensive market analysis"""
        try:
            # Calculate all indicators
            indicators = self._calculate_indicators(data)

            # Analyze market regime
            regime = self._analyze_market_regime(data, indicators)

            # Analyze market conditions
            conditions = self._analyze_market_conditions(data, indicators)

            # Generate signals
            signals = self._generate_signals(data, indicators, regime, conditions)

            return {
                "indicators": indicators,
                "regime": regime,
                "conditions": conditions,
                "signals": signals,
            }

        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            raise

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        indicators = {}

        for category, category_indicators in self.indicators.items():
            indicators[category] = {}
            for name, func in category_indicators.items():
                try:
                    indicators[category][name] = func(data)
                except Exception as e:
                    self.logger.error(f"Error calculating {name}: {str(e)}")
                    indicators[category][name] = None

        return indicators

    def _analyze_market_regime(
        self, data: pd.DataFrame, indicators: Dict
    ) -> MarketRegime:
        """Analyze current market regime"""
        # Calculate regime metrics
        trend_strength = self._calculate_trend_strength(data, indicators)
        volatility = self._calculate_volatility(data, indicators)
        self._calculate_momentum(data, indicators)
        self._calculate_volume_profile(data, indicators)

        # Determine regime
        if trend_strength > 0.7 and volatility < 0.3:
            regime = MarketRegime(
                name="Strong Bull",
                description="Strong uptrend with low volatility",
                conditions={"trend": "up", "volatility": "low"},
                metrics={"trend_strength": trend_strength, "volatility": volatility},
                confidence=0.8,
            )
        elif trend_strength > 0.5 and volatility < 0.5:
            regime = MarketRegime(
                name="Moderate Bull",
                description="Moderate uptrend with normal volatility",
                conditions={"trend": "up", "volatility": "normal"},
                metrics={"trend_strength": trend_strength, "volatility": volatility},
                confidence=0.6,
            )
        elif trend_strength < -0.7 and volatility < 0.3:
            regime = MarketRegime(
                name="Strong Bear",
                description="Strong downtrend with low volatility",
                conditions={"trend": "down", "volatility": "low"},
                metrics={"trend_strength": trend_strength, "volatility": volatility},
                confidence=0.8,
            )
        elif trend_strength < -0.5 and volatility < 0.5:
            regime = MarketRegime(
                name="Moderate Bear",
                description="Moderate downtrend with normal volatility",
                conditions={"trend": "down", "volatility": "normal"},
                metrics={"trend_strength": trend_strength, "volatility": volatility},
                confidence=0.6,
            )
        elif volatility > 0.7:
            regime = MarketRegime(
                name="High Volatility",
                description="High volatility with unclear trend",
                conditions={"trend": "unclear", "volatility": "high"},
                metrics={"trend_strength": trend_strength, "volatility": volatility},
                confidence=0.7,
            )
        else:
            regime = MarketRegime(
                name="Sideways",
                description="Sideways market with normal volatility",
                conditions={"trend": "sideways", "volatility": "normal"},
                metrics={"trend_strength": trend_strength, "volatility": volatility},
                confidence=0.5,
            )

        return regime

    def _analyze_market_conditions(
        self, data: pd.DataFrame, indicators: Dict
    ) -> List[MarketCondition]:
        """Analyze current market conditions"""
        conditions = []

        # Analyze trend conditions
        trend_condition = self._analyze_trend_condition(data, indicators)
        if trend_condition:
            conditions.append(trend_condition)

        # Analyze momentum conditions
        momentum_condition = self._analyze_momentum_condition(data, indicators)
        if momentum_condition:
            conditions.append(momentum_condition)

        # Analyze volatility conditions
        volatility_condition = self._analyze_volatility_condition(data, indicators)
        if volatility_condition:
            conditions.append(volatility_condition)

        # Analyze volume conditions
        volume_condition = self._analyze_volume_condition(data, indicators)
        if volume_condition:
            conditions.append(volume_condition)

        # Analyze support/resistance conditions
        sr_condition = self._analyze_support_resistance_condition(data, indicators)
        if sr_condition:
            conditions.append(sr_condition)

        return conditions

    def _generate_signals(
        self,
        data: pd.DataFrame,
        indicators: Dict,
        regime: MarketRegime,
        conditions: List[MarketCondition],
    ) -> Dict:
        """Generate trading signals based on analysis"""
        signals = {
            "trend": self._generate_trend_signals(data, indicators, regime),
            "momentum": self._generate_momentum_signals(data, indicators, regime),
            "volatility": self._generate_volatility_signals(data, indicators, regime),
            "volume": self._generate_volume_signals(data, indicators, regime),
            "support_resistance": self._generate_support_resistance_signals(
                data, indicators, regime
            ),
        }

        # Combine signals
        combined_signal = self._combine_signals(signals, regime, conditions)

        return {"individual": signals, "combined": combined_signal}

    # Technical indicator calculations
    def _calculate_sma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data["close"].rolling(window=window).mean()

    def _calculate_ema(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data["close"].ewm(span=window, adjust=False).mean()

    def _calculate_macd(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = data["close"].ewm(span=12, adjust=False).mean()
        exp2 = data["close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def _calculate_adx(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Calculate smoothed averages
        tr_smoothed = tr.rolling(window=window).sum()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=window).sum() / tr_smoothed
        minus_di = 100 * pd.Series(minus_dm).rolling(window=window).sum() / tr_smoothed

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()

        return adx

    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud"""
        high = data["high"]
        low = data["low"]

        # Calculate components
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = (
            (high.rolling(window=52).max() + low.rolling(window=52).min()) / 2
        ).shift(26)
        chikou_span = data["close"].shift(-26)

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }

    def _calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(
        self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data["low"].rolling(window=k_window).min()
        high_max = data["high"].rolling(window=k_window).max()

        k = 100 * ((data["close"] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_window).mean()

        return k, d

    def _calculate_cci(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        tp = (data["high"] + data["low"] + data["close"]) / 3
        tp_sma = tp.rolling(window=window).mean()
        tp_std = tp.rolling(window=window).std()
        return (tp - tp_sma) / (0.015 * tp_std)

    def _calculate_mfi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        money_flow = typical_price * data["volume"]

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi

    def _calculate_roc(self, data: pd.DataFrame, window: int = 12) -> pd.Series:
        """Calculate Rate of Change"""
        return (
            (data["close"] - data["close"].shift(window)) / data["close"].shift(window)
        ) * 100

    def _calculate_bollinger_bands(
        self, data: pd.DataFrame, window: int = 20, num_std: float = 2
    ) -> Dict:
        """Calculate Bollinger Bands"""
        middle_band = data["close"].rolling(window=window).mean()
        std = data["close"].rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return {
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "std": std,
        }

    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    def _calculate_keltner_channels(
        self, data: pd.DataFrame, window: int = 20, num_std: float = 2
    ) -> Dict:
        """Calculate Keltner Channels"""
        middle_band = data["close"].ewm(span=window).mean()
        atr = self._calculate_atr(data, window)
        upper_band = middle_band + (atr * num_std)
        lower_band = middle_band - (atr * num_std)

        return {
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "atr": atr,
        }

    def _calculate_donchian_channels(
        self, data: pd.DataFrame, window: int = 20
    ) -> Dict:
        """Calculate Donchian Channels"""
        upper_band = data["high"].rolling(window=window).max()
        lower_band = data["low"].rolling(window=window).min()
        middle_band = (upper_band + lower_band) / 2

        return {
            "upper_band": upper_band,
            "middle_band": middle_band,
            "lower_band": lower_band,
        }

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        close_diff = data["close"].diff()
        volume = data["volume"]

        obv = pd.Series(0, index=data.index)
        obv[close_diff > 0] = volume[close_diff > 0]
        obv[close_diff < 0] = -volume[close_diff < 0]

        return obv.cumsum()

    def _calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        close_diff = data["close"].diff()
        volume = data["volume"]

        vpt = (close_diff / data["close"].shift(1)) * volume
        return vpt.cumsum()

    def _calculate_cmf(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mfv = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (
            data["high"] - data["low"]
        )
        mfv = mfv.fillna(0)
        mfv *= data["volume"]

        return (
            mfv.rolling(window=window).sum()
            / data["volume"].rolling(window=window).sum()
        )

    def _calculate_ad(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (
            data["high"] - data["low"]
        )
        clv = clv.fillna(0)
        return (clv * data["volume"]).cumsum()

    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict:
        """Calculate Pivot Points"""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)

        return {"pp": pp, "r1": r1, "s1": s1, "r2": r2, "s2": s2, "r3": r3, "s3": s3}

    def _calculate_fibonacci_retracement(
        self, data: pd.DataFrame, window: int = 20
    ) -> Dict:
        """Calculate Fibonacci Retracement Levels"""
        high = data["high"].rolling(window=window).max()
        low = data["low"].rolling(window=window).min()
        diff = high - low

        levels = {
            "0.0": low,
            "0.236": low + 0.236 * diff,
            "0.382": low + 0.382 * diff,
            "0.5": low + 0.5 * diff,
            "0.618": low + 0.618 * diff,
            "0.786": low + 0.786 * diff,
            "1.0": high,
        }

        return levels

    def _calculate_price_channels(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Calculate Price Channels"""
        high_channel = data["high"].rolling(window=window).max()
        low_channel = data["low"].rolling(window=window).min()
        middle_channel = (high_channel + low_channel) / 2

        return {
            "high_channel": high_channel,
            "middle_channel": middle_channel,
            "low_channel": low_channel,
        }

    # Market condition analysis methods
    def _analyze_trend_condition(
        self, data: pd.DataFrame, indicators: Dict
    ) -> Optional[MarketCondition]:
        """Analyze trend condition"""
        trend_indicators = indicators["trend"]

        # Get trend indicators
        adx = trend_indicators["adx"]
        macd, signal, _ = trend_indicators["macd"]

        if adx is None or macd is None:
            return None

        # Determine trend condition
        if adx.iloc[-1] > 25:
            if macd.iloc[-1] > signal.iloc[-1]:
                return MarketCondition(
                    name="Strong Uptrend",
                    description="Strong uptrend with high ADX and positive MACD",
                    indicators={"adx": adx.iloc[-1], "macd": macd.iloc[-1]},
                    signals={"trend": "up", "strength": "strong"},
                    strength=0.8,
                )
            else:
                return MarketCondition(
                    name="Strong Downtrend",
                    description="Strong downtrend with high ADX and negative MACD",
                    indicators={"adx": adx.iloc[-1], "macd": macd.iloc[-1]},
                    signals={"trend": "down", "strength": "strong"},
                    strength=0.8,
                )
        elif adx.iloc[-1] > 20:
            if macd.iloc[-1] > signal.iloc[-1]:
                return MarketCondition(
                    name="Moderate Uptrend",
                    description="Moderate uptrend with medium ADX and positive MACD",
                    indicators={"adx": adx.iloc[-1], "macd": macd.iloc[-1]},
                    signals={"trend": "up", "strength": "moderate"},
                    strength=0.6,
                )
            else:
                return MarketCondition(
                    name="Moderate Downtrend",
                    description="Moderate downtrend with medium ADX and negative MACD",
                    indicators={"adx": adx.iloc[-1], "macd": macd.iloc[-1]},
                    signals={"trend": "down", "strength": "moderate"},
                    strength=0.6,
                )

        return None

    def _analyze_momentum_condition(
        self, data: pd.DataFrame, indicators: Dict
    ) -> Optional[MarketCondition]:
        """Analyze momentum condition"""
        momentum_indicators = indicators["momentum"]

        # Get momentum indicators
        rsi = momentum_indicators["rsi"]
        k, d = momentum_indicators["stochastic"]
        cci = momentum_indicators["cci"]

        if rsi is None or k is None or cci is None:
            return None

        # Determine momentum condition
        if rsi.iloc[-1] > 70 and k.iloc[-1] > 80 and cci.iloc[-1] > 100:
            return MarketCondition(
                name="Overbought",
                description="Market is overbought with high RSI, Stochastic, and CCI",
                indicators={
                    "rsi": rsi.iloc[-1],
                    "stochastic": k.iloc[-1],
                    "cci": cci.iloc[-1],
                },
                signals={"momentum": "overbought"},
                strength=0.8,
            )
        elif rsi.iloc[-1] < 30 and k.iloc[-1] < 20 and cci.iloc[-1] < -100:
            return MarketCondition(
                name="Oversold",
                description="Market is oversold with low RSI, Stochastic, and CCI",
                indicators={
                    "rsi": rsi.iloc[-1],
                    "stochastic": k.iloc[-1],
                    "cci": cci.iloc[-1],
                },
                signals={"momentum": "oversold"},
                strength=0.8,
            )

        return None

    def _analyze_volatility_condition(
        self, data: pd.DataFrame, indicators: Dict
    ) -> Optional[MarketCondition]:
        """Analyze volatility condition"""
        volatility_indicators = indicators["volatility"]

        # Get volatility indicators
        bb = volatility_indicators["bollinger_bands"]
        atr = volatility_indicators["atr"]
        kc = volatility_indicators["keltner_channels"]

        if bb is None or atr is None or kc is None:
            return None

        # Calculate volatility metrics
        bb_width = (bb["upper_band"] - bb["lower_band"]) / bb["middle_band"]
        kc_width = (kc["upper_band"] - kc["lower_band"]) / kc["middle_band"]

        # Determine volatility condition
        if bb_width.iloc[-1] > 0.1 and kc_width.iloc[-1] > 0.1:
            return MarketCondition(
                name="High Volatility",
                description="Market is experiencing high volatility",
                indicators={
                    "bb_width": bb_width.iloc[-1],
                    "kc_width": kc_width.iloc[-1],
                    "atr": atr.iloc[-1],
                },
                signals={"volatility": "high"},
                strength=0.7,
            )
        elif bb_width.iloc[-1] < 0.05 and kc_width.iloc[-1] < 0.05:
            return MarketCondition(
                name="Low Volatility",
                description="Market is experiencing low volatility",
                indicators={
                    "bb_width": bb_width.iloc[-1],
                    "kc_width": kc_width.iloc[-1],
                    "atr": atr.iloc[-1],
                },
                signals={"volatility": "low"},
                strength=0.7,
            )

        return None

    def _analyze_volume_condition(
        self, data: pd.DataFrame, indicators: Dict
    ) -> Optional[MarketCondition]:
        """Analyze volume condition"""
        volume_indicators = indicators["volume"]

        # Get volume indicators
        obv = volume_indicators["obv"]
        vpt = volume_indicators["vpt"]
        cmf = volume_indicators["cmf"]

        if obv is None or vpt is None or cmf is None:
            return None

        # Calculate volume metrics
        obv_change = obv.pct_change()
        vpt_change = vpt.pct_change()

        # Determine volume condition
        if (
            obv_change.iloc[-1] > 0.1
            and vpt_change.iloc[-1] > 0.1
            and cmf.iloc[-1] > 0.2
        ):
            return MarketCondition(
                name="High Volume",
                description="Market is experiencing high volume with positive flow",
                indicators={
                    "obv_change": obv_change.iloc[-1],
                    "vpt_change": vpt_change.iloc[-1],
                    "cmf": cmf.iloc[-1],
                },
                signals={"volume": "high", "flow": "positive"},
                strength=0.7,
            )
        elif (
            obv_change.iloc[-1] < -0.1
            and vpt_change.iloc[-1] < -0.1
            and cmf.iloc[-1] < -0.2
        ):
            return MarketCondition(
                name="Low Volume",
                description="Market is experiencing low volume with negative flow",
                indicators={
                    "obv_change": obv_change.iloc[-1],
                    "vpt_change": vpt_change.iloc[-1],
                    "cmf": cmf.iloc[-1],
                },
                signals={"volume": "low", "flow": "negative"},
                strength=0.7,
            )

        return None

    def _analyze_support_resistance_condition(
        self, data: pd.DataFrame, indicators: Dict
    ) -> Optional[MarketCondition]:
        """Analyze support/resistance condition"""
        sr_indicators = indicators["support_resistance"]

        # Get support/resistance indicators
        pp = sr_indicators["pivot_points"]
        fib = sr_indicators["fibonacci_retracement"]
        pc = sr_indicators["price_channels"]

        if pp is None or fib is None or pc is None:
            return None

        # Calculate price position relative to levels
        current_price = data["close"].iloc[-1]

        # Determine support/resistance condition
        if current_price > pp["r2"].iloc[-1]:
            return MarketCondition(
                name="Above Resistance",
                description="Price is above major resistance level",
                indicators={"price": current_price, "resistance": pp["r2"].iloc[-1]},
                signals={"position": "above_resistance"},
                strength=0.7,
            )
        elif current_price < pp["s2"].iloc[-1]:
            return MarketCondition(
                name="Below Support",
                description="Price is below major support level",
                indicators={"price": current_price, "support": pp["s2"].iloc[-1]},
                signals={"position": "below_support"},
                strength=0.7,
            )

        return None

    # Signal generation methods
    def _generate_trend_signals(
        self, data: pd.DataFrame, indicators: Dict, regime: MarketRegime
    ) -> Dict:
        """Generate trend signals"""
        trend_indicators = indicators["trend"]

        # Get trend indicators
        macd, signal, histogram = trend_indicators["macd"]
        adx = trend_indicators["adx"]
        ichimoku = trend_indicators["ichimoku"]

        if macd is None or adx is None or ichimoku is None:
            return {"signal": "neutral", "strength": 0}

        # Generate signals
        signals = {
            "macd": "buy" if macd.iloc[-1] > signal.iloc[-1] else "sell",
            "adx": "strong" if adx.iloc[-1] > 25 else "weak",
            "ichimoku": (
                "buy"
                if data["close"].iloc[-1] > ichimoku["senkou_span_a"].iloc[-1]
                else "sell"
            ),
        }

        # Combine signals
        if (
            signals["macd"] == "buy"
            and signals["ichimoku"] == "buy"
            and signals["adx"] == "strong"
        ):
            return {"signal": "strong_buy", "strength": 0.8}
        elif (
            signals["macd"] == "sell"
            and signals["ichimoku"] == "sell"
            and signals["adx"] == "strong"
        ):
            return {"signal": "strong_sell", "strength": 0.8}
        elif signals["macd"] == "buy" and signals["ichimoku"] == "buy":
            return {"signal": "buy", "strength": 0.6}
        elif signals["macd"] == "sell" and signals["ichimoku"] == "sell":
            return {"signal": "sell", "strength": 0.6}

        return {"signal": "neutral", "strength": 0.4}

    def _generate_momentum_signals(
        self, data: pd.DataFrame, indicators: Dict, regime: MarketRegime
    ) -> Dict:
        """Generate momentum signals"""
        momentum_indicators = indicators["momentum"]

        # Get momentum indicators
        rsi = momentum_indicators["rsi"]
        k, d = momentum_indicators["stochastic"]
        cci = momentum_indicators["cci"]

        if rsi is None or k is None or cci is None:
            return {"signal": "neutral", "strength": 0}

        # Generate signals
        signals = {
            "rsi": (
                "oversold"
                if rsi.iloc[-1] < 30
                else "overbought"
                if rsi.iloc[-1] > 70
                else "neutral"
            ),
            "stochastic": (
                "oversold"
                if k.iloc[-1] < 20
                else "overbought"
                if k.iloc[-1] > 80
                else "neutral"
            ),
            "cci": (
                "oversold"
                if cci.iloc[-1] < -100
                else "overbought"
                if cci.iloc[-1] > 100
                else "neutral"
            ),
        }

        # Combine signals
        if (
            signals["rsi"] == "oversold"
            and signals["stochastic"] == "oversold"
            and signals["cci"] == "oversold"
        ):
            return {"signal": "strong_buy", "strength": 0.8}
        elif (
            signals["rsi"] == "overbought"
            and signals["stochastic"] == "overbought"
            and signals["cci"] == "overbought"
        ):
            return {"signal": "strong_sell", "strength": 0.8}
        elif signals["rsi"] == "oversold" and signals["stochastic"] == "oversold":
            return {"signal": "buy", "strength": 0.6}
        elif signals["rsi"] == "overbought" and signals["stochastic"] == "overbought":
            return {"signal": "sell", "strength": 0.6}

        return {"signal": "neutral", "strength": 0.4}

    def _generate_volatility_signals(
        self, data: pd.DataFrame, indicators: Dict, regime: MarketRegime
    ) -> Dict:
        """Generate volatility signals"""
        volatility_indicators = indicators["volatility"]

        # Get volatility indicators
        bb = volatility_indicators["bollinger_bands"]
        atr = volatility_indicators["atr"]
        kc = volatility_indicators["keltner_channels"]

        if bb is None or atr is None or kc is None:
            return {"signal": "neutral", "strength": 0}

        # Generate signals
        current_price = data["close"].iloc[-1]
        signals = {
            "bb": (
                "oversold"
                if current_price < bb["lower_band"].iloc[-1]
                else (
                    "overbought"
                    if current_price > bb["upper_band"].iloc[-1]
                    else "neutral"
                )
            ),
            "kc": (
                "oversold"
                if current_price < kc["lower_band"].iloc[-1]
                else (
                    "overbought"
                    if current_price > kc["upper_band"].iloc[-1]
                    else "neutral"
                )
            ),
            "atr": "high" if atr.iloc[-1] > atr.mean() else "low",
        }

        # Combine signals
        if signals["bb"] == "oversold" and signals["kc"] == "oversold":
            return {"signal": "strong_buy", "strength": 0.8}
        elif signals["bb"] == "overbought" and signals["kc"] == "overbought":
            return {"signal": "strong_sell", "strength": 0.8}
        elif signals["bb"] == "oversold":
            return {"signal": "buy", "strength": 0.6}
        elif signals["bb"] == "overbought":
            return {"signal": "sell", "strength": 0.6}

        return {"signal": "neutral", "strength": 0.4}

    def _generate_volume_signals(
        self, data: pd.DataFrame, indicators: Dict, regime: MarketRegime
    ) -> Dict:
        """Generate volume signals"""
        volume_indicators = indicators["volume"]

        # Get volume indicators
        obv = volume_indicators["obv"]
        vpt = volume_indicators["vpt"]
        cmf = volume_indicators["cmf"]

        if obv is None or vpt is None or cmf is None:
            return {"signal": "neutral", "strength": 0}

        # Generate signals
        signals = {
            "obv": "positive" if obv.iloc[-1] > obv.iloc[-2] else "negative",
            "vpt": "positive" if vpt.iloc[-1] > vpt.iloc[-2] else "negative",
            "cmf": "positive" if cmf.iloc[-1] > 0 else "negative",
        }

        # Combine signals
        if (
            signals["obv"] == "positive"
            and signals["vpt"] == "positive"
            and signals["cmf"] == "positive"
        ):
            return {"signal": "strong_buy", "strength": 0.8}
        elif (
            signals["obv"] == "negative"
            and signals["vpt"] == "negative"
            and signals["cmf"] == "negative"
        ):
            return {"signal": "strong_sell", "strength": 0.8}
        elif signals["obv"] == "positive" and signals["vpt"] == "positive":
            return {"signal": "buy", "strength": 0.6}
        elif signals["obv"] == "negative" and signals["vpt"] == "negative":
            return {"signal": "sell", "strength": 0.6}

        return {"signal": "neutral", "strength": 0.4}

    def _generate_support_resistance_signals(
        self, data: pd.DataFrame, indicators: Dict, regime: MarketRegime
    ) -> Dict:
        """Generate support/resistance signals"""
        sr_indicators = indicators["support_resistance"]

        # Get support/resistance indicators
        pp = sr_indicators["pivot_points"]
        fib = sr_indicators["fibonacci_retracement"]
        pc = sr_indicators["price_channels"]

        if pp is None or fib is None or pc is None:
            return {"signal": "neutral", "strength": 0}

        # Generate signals
        current_price = data["close"].iloc[-1]
        signals = {
            "pp": "support" if current_price < pp["pp"].iloc[-1] else "resistance",
            "fib": "support" if current_price < fib["0.5"].iloc[-1] else "resistance",
            "pc": (
                "support"
                if current_price < pc["middle_channel"].iloc[-1]
                else "resistance"
            ),
        }

        # Combine signals
        if (
            signals["pp"] == "support"
            and signals["fib"] == "support"
            and signals["pc"] == "support"
        ):
            return {"signal": "strong_buy", "strength": 0.8}
        elif (
            signals["pp"] == "resistance"
            and signals["fib"] == "resistance"
            and signals["pc"] == "resistance"
        ):
            return {"signal": "strong_sell", "strength": 0.8}
        elif signals["pp"] == "support" and signals["fib"] == "support":
            return {"signal": "buy", "strength": 0.6}
        elif signals["pp"] == "resistance" and signals["fib"] == "resistance":
            return {"signal": "sell", "strength": 0.6}

        return {"signal": "neutral", "strength": 0.4}

    def _combine_signals(
        self, signals: Dict, regime: MarketRegime, conditions: List[MarketCondition]
    ) -> Dict:
        """Combine all signals into a final trading signal"""
        # Get individual signals
        trend_signal = signals["trend"]
        momentum_signal = signals["momentum"]
        volatility_signal = signals["volatility"]
        volume_signal = signals["volume"]
        sr_signal = signals["support_resistance"]

        # Calculate signal strengths
        strengths = {
            "trend": trend_signal["strength"],
            "momentum": momentum_signal["strength"],
            "volatility": volatility_signal["strength"],
            "volume": volume_signal["strength"],
            "sr": sr_signal["strength"],
        }

        # Count buy and sell signals
        buy_signals = sum(
            1
            for signal in [
                trend_signal,
                momentum_signal,
                volatility_signal,
                volume_signal,
                sr_signal,
            ]
            if signal["signal"] in ["buy", "strong_buy"]
        )
        sell_signals = sum(
            1
            for signal in [
                trend_signal,
                momentum_signal,
                volatility_signal,
                volume_signal,
                sr_signal,
            ]
            if signal["signal"] in ["sell", "strong_sell"]
        )

        # Calculate average strength
        avg_strength = sum(strengths.values()) / len(strengths)

        # Generate final signal
        if buy_signals >= 4 and avg_strength >= 0.7:
            return {"signal": "strong_buy", "strength": avg_strength, "confidence": 0.9}
        elif sell_signals >= 4 and avg_strength >= 0.7:
            return {
                "signal": "strong_sell",
                "strength": avg_strength,
                "confidence": 0.9,
            }
        elif buy_signals >= 3 and avg_strength >= 0.6:
            return {"signal": "buy", "strength": avg_strength, "confidence": 0.7}
        elif sell_signals >= 3 and avg_strength >= 0.6:
            return {"signal": "sell", "strength": avg_strength, "confidence": 0.7}

        return {"signal": "neutral", "strength": avg_strength, "confidence": 0.5}


def analyze_market_conditions(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze market conditions using the MarketAnalysis class.

    Args:
        data: Market data DataFrame

    Returns:
        Dictionary with market analysis results
    """
    try:
        analyzer = MarketAnalyzer()
        return analyzer.analyze_market_conditions(data)
    except Exception as e:
        logger.error(f"Error analyzing market conditions: {str(e)}")
        return {
            "error": str(e),
            "indicators": {},
            "regime": None,
            "conditions": [],
            "signals": {},
        }
