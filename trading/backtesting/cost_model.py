"""
Robust Cost Model for Backtesting

This module provides realistic cost modeling for backtesting including:
- Trading fees (fixed, percentage, tiered, bps)
- Bid-ask spreads (fixed, proportional, volatility-based, market-based)
- Slippage (fixed, proportional, volatility-based, volume/market impact)
- Market impact modeling
- Configurable presets for different market types
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeeModel(Enum):
    FIXED = "fixed"  # Fixed fee per trade
    PERCENTAGE = "percentage"  # Percentage of trade value
    TIERED = "tiered"  # Tiered fee structure
    BPS = "bps"  # Basis points


class SpreadModel(Enum):
    FIXED = "fixed"  # Fixed spread
    PROPORTIONAL = "proportional"  # Proportional to price
    VOLATILITY = "volatility"  # Based on price volatility
    MARKET = "market"  # Market-based spreads


class SlippageModel(Enum):
    FIXED = "fixed"  # Fixed slippage
    PROPORTIONAL = "proportional"  # Proportional to trade size
    VOLATILITY = "volatility"  # Based on volatility
    VOLUME = "volume"  # Based on volume
    MARKET_IMPACT = "market_impact"  # Market impact model


@dataclass
class CostConfig:
    # Fee configuration
    fee_model: FeeModel = FeeModel.PERCENTAGE
    fee_rate: float = 0.001  # 0.1% or 10 bps
    fixed_fee: float = 1.0  # $1 per trade
    min_fee: float = 0.0  # Minimum fee
    max_fee: float = 1000.0  # Maximum fee
    tiered_fees: Optional[List[Tuple[float, float]]] = field(
        default_factory=lambda: [  # (threshold, rate)
            (10000, 0.001),  # 0.1% for trades up to $10k
            (100000, 0.0005),  # 0.05% for trades up to $100k
            (1000000, 0.0002),  # 0.02% for trades up to $1M
            (float("inf"), 0.0001),  # 0.01% above $1M
        ]
    )
    # Spread configuration
    spread_model: SpreadModel = SpreadModel.PROPORTIONAL
    spread_rate: float = 0.0005  # 0.05% or 5 bps
    min_spread: float = 0.0
    max_spread: float = 0.01
    # Slippage configuration
    slippage_model: SlippageModel = SlippageModel.PROPORTIONAL
    slippage_rate: float = 0.0002  # 0.02% or 2 bps
    min_slippage: float = 0.0
    max_slippage: float = 0.005
    # Market impact parameters
    market_impact_factor: float = 0.1
    volume_threshold: float = 1_000_000
    # Volatility parameters
    volatility_window: int = 20  # Days for volatility calculation
    volatility_multiplier: float = 1.0


class CostModel:
    """
    Comprehensive cost model for backtesting.
    """

    def __init__(self, config: CostConfig, price_data: Optional[pd.DataFrame] = None):
        self.config = config
        self.price_data = price_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self.volatility_cache = {}
        if price_data is not None:
            self._precalculate_volatility()

    def _precalculate_volatility(self) -> None:
        if self.price_data is None:
            return
        for column in self.price_data.columns:
            if "close" in column.lower() or "price" in column.lower():
                returns = self.price_data[column].pct_change().dropna()
                volatility = returns.rolling(window=self.config.volatility_window).std()
                self.volatility_cache[column] = volatility

    def calculate_fees(self, trade_value: float) -> float:
        if self.config.fee_model == FeeModel.FIXED:
            fee = self.config.fixed_fee
        elif self.config.fee_model == FeeModel.PERCENTAGE:
            fee = trade_value * self.config.fee_rate
        elif self.config.fee_model == FeeModel.BPS:
            fee = trade_value * (self.config.fee_rate / 10000)
        elif self.config.fee_model == FeeModel.TIERED:
            fee = self._calculate_tiered_fee(trade_value)
        else:
            fee = trade_value * self.config.fee_rate
        fee = max(self.config.min_fee, min(fee, self.config.max_fee))
        return fee

    def _calculate_tiered_fee(self, trade_value: float) -> float:
        fee = 0.0
        remaining_value = trade_value
        for threshold, rate in self.config.tiered_fees:
            if remaining_value <= 0:
                break
            tier_amount = min(remaining_value, threshold)
            fee += tier_amount * rate
            remaining_value -= tier_amount
        return fee

    def calculate_spread(
        self,
        price: float,
        asset: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> float:
        if self.config.spread_model == SpreadModel.FIXED:
            spread = self.config.spread_rate
        elif self.config.spread_model == SpreadModel.PROPORTIONAL:
            spread = price * self.config.spread_rate
        elif self.config.spread_model == SpreadModel.VOLATILITY:
            spread = self._calculate_volatility_spread(price, asset, timestamp)
        elif self.config.spread_model == SpreadModel.MARKET:
            spread = self._calculate_market_spread(price, asset, timestamp)
        else:
            spread = price * self.config.spread_rate
        spread = max(self.config.min_spread, min(spread, self.config.max_spread))
        return spread

    def _calculate_volatility_spread(
        self, price: float, asset: Optional[str], timestamp: Optional[datetime]
    ) -> float:
        if asset not in self.volatility_cache or timestamp is None:
            return price * self.config.spread_rate
        vol_idx = self.volatility_cache[asset].index.get_loc(
            timestamp, method="nearest"
        )
        volatility = self.volatility_cache[asset].iloc[vol_idx]
        volatility_adjustment = 1 + (volatility * self.config.volatility_multiplier)
        return price * self.config.spread_rate * volatility_adjustment

    def _calculate_market_spread(
        self, price: float, asset: Optional[str], timestamp: Optional[datetime]
    ) -> float:
        base_spread = price * self.config.spread_rate
        hour = timestamp.hour if timestamp else 12
        if 9 <= hour <= 16:
            return base_spread
        else:
            return base_spread * 2  # Wider spreads after hours

    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        trade_type: str,
        asset: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        volume: Optional[float] = None,
    ) -> float:
        if self.config.slippage_model == SlippageModel.FIXED:
            slippage = self.config.slippage_rate
        elif self.config.slippage_model == SlippageModel.PROPORTIONAL:
            slippage = price * self.config.slippage_rate
        elif self.config.slippage_model == SlippageModel.VOLATILITY:
            slippage = self._calculate_volatility_slippage(price, asset, timestamp)
        elif self.config.slippage_model == SlippageModel.VOLUME:
            slippage = self._calculate_volume_slippage(price, quantity, volume)
        elif self.config.slippage_model == SlippageModel.MARKET_IMPACT:
            slippage = self._calculate_market_impact_slippage(price, quantity, volume)
        else:
            slippage = price * self.config.slippage_rate
        slippage = max(
            self.config.min_slippage, min(slippage, self.config.max_slippage)
        )
        return slippage

    def _calculate_volatility_slippage(
        self, price: float, asset: Optional[str], timestamp: Optional[datetime]
    ) -> float:
        if asset not in self.volatility_cache or timestamp is None:
            return price * self.config.slippage_rate
        vol_idx = self.volatility_cache[asset].index.get_loc(
            timestamp, method="nearest"
        )
        volatility = self.volatility_cache[asset].iloc[vol_idx]
        volatility_adjustment = 1 + (volatility * self.config.volatility_multiplier)
        return price * self.config.slippage_rate * volatility_adjustment

    def _calculate_volume_slippage(
        self, price: float, quantity: float, volume: Optional[float]
    ) -> float:
        if volume is None or volume == 0:
            return price * self.config.slippage_rate
        volume_ratio = quantity / volume
        volume_adjustment = 1 + (volume_ratio * 10)
        return price * self.config.slippage_rate * volume_adjustment

    def _calculate_market_impact_slippage(
        self, price: float, quantity: float, volume: Optional[float]
    ) -> float:
        if volume is None or volume == 0:
            return price * self.config.slippage_rate
        trade_volume_ratio = quantity / volume
        impact_factor = np.sqrt(trade_volume_ratio) * self.config.market_impact_factor
        return price * impact_factor

    def calculate_total_cost(
        self,
        price: float,
        quantity: float,
        trade_type: str,
        asset: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        volume: Optional[float] = None,
    ) -> Dict[str, float]:
        trade_value = price * quantity
        fees = self.calculate_fees(trade_value)
        spread = self.calculate_spread(price, asset, timestamp)
        slippage = self.calculate_slippage(
            price, quantity, trade_type, asset, timestamp, volume
        )
        total_cost = fees + (spread * quantity) + (slippage * quantity)
        if trade_type.lower() == "buy":
            effective_price = price + spread + slippage
        else:
            effective_price = price - spread - slippage
        return {
            "fees": fees,
            "spread": spread * quantity,
            "slippage": slippage * quantity,
            "total_cost": total_cost,
            "effective_price": effective_price,
            "cost_percentage": (
                (total_cost / trade_value) * 100 if trade_value > 0 else 0
            ),
        }

    def get_cost_summary(self, trades: List[Dict]) -> Dict[str, float]:
        if not trades:
            return {}
        total_fees = sum(trade.get("fees", 0) for trade in trades)
        total_spread = sum(trade.get("spread", 0) for trade in trades)
        total_slippage = sum(trade.get("slippage", 0) for trade in trades)
        total_cost = sum(trade.get("total_cost", 0) for trade in trades)
        total_value = sum(trade.get("trade_value", 0) for trade in trades)
        return {
            "total_fees": total_fees,
            "total_spread": total_spread,
            "total_slippage": total_slippage,
            "total_cost": total_cost,
            "total_trade_value": total_value,
            "average_cost_percentage": (
                (total_cost / total_value) * 100 if total_value > 0 else 0
            ),
            "cost_per_trade": total_cost / len(trades) if trades else 0,
            "number_of_trades": len(trades),
        }


# Preset configurations for different market types
def get_retail_cost_config() -> CostConfig:
    return CostConfig(
        fee_model=FeeModel.PERCENTAGE,
        fee_rate=0.001,
        spread_model=SpreadModel.PROPORTIONAL,
        spread_rate=0.0005,
        slippage_model=SlippageModel.PROPORTIONAL,
        slippage_rate=0.0002,
    )


def get_institutional_cost_config() -> CostConfig:
    return CostConfig(
        fee_model=FeeModel.TIERED,
        spread_model=SpreadModel.VOLATILITY,
        spread_rate=0.0002,
        slippage_model=SlippageModel.MARKET_IMPACT,
        slippage_rate=0.0001,
        market_impact_factor=0.05,
    )


def get_high_frequency_cost_config() -> CostConfig:
    return CostConfig(
        fee_model=FeeModel.BPS,
        fee_rate=1.0,  # 1 bps
        spread_model=SpreadModel.MARKET,
        spread_rate=0.0001,
        slippage_model=SlippageModel.VOLUME,
        slippage_rate=0.00005,
        market_impact_factor=0.02,
    )


def get_crypto_cost_config() -> CostConfig:
    return CostConfig(
        fee_model=FeeModel.PERCENTAGE,
        fee_rate=0.002,
        spread_model=SpreadModel.PROPORTIONAL,
        spread_rate=0.001,
        slippage_model=SlippageModel.VOLATILITY,
        slippage_rate=0.0005,
        volatility_multiplier=2.0,
    )
