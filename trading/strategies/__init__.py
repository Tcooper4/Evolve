"""Trading strategies — import submodules directly; barrel avoids eager-loading all strategies."""

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def get_signals(strategy_name: str, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Get trading signals for a specific strategy (lazy imports)."""
    try:
        if strategy_name.lower() == "rsi":
            from trading.strategies.rsi_signals import generate_signals

            return generate_signals(data, **kwargs)
        elif strategy_name.lower() == "bollinger":
            from trading.strategies.bollinger_strategy import BollingerStrategy

            strategy = BollingerStrategy()
            return strategy.generate_signals(data, **kwargs)
        elif strategy_name.lower() == "macd":
            from trading.strategies.macd_strategy import MACDStrategy

            strategy = MACDStrategy()
            return strategy.generate_signals(data, **kwargs)
        elif strategy_name.lower() == "sma":
            from trading.strategies.sma_strategy import SMAStrategy

            strategy = SMAStrategy()
            return {
                "success": True,
                "result": strategy.generate_signals(data, **kwargs),
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
        elif strategy_name.lower() == "cci":
            from trading.strategies.cci_strategy import generate_cci_signals

            return generate_cci_signals(data, **kwargs)
        elif strategy_name.lower() == "atr":
            from trading.strategies.atr_strategy import generate_atr_signals

            return generate_atr_signals(data, **kwargs)
        elif strategy_name.lower() == "ensemble":
            from trading.strategies.ensemble import create_ensemble_strategy

            strategy_weights = kwargs.get(
                "strategy_weights", {"rsi": 0.4, "macd": 0.4, "bollinger": 0.2}
            )
            combination_method = kwargs.get("combination_method", "weighted_average")
            ensemble = create_ensemble_strategy(
                strategy_weights, combination_method, **kwargs
            )
            strategy_signals = {}
            for sn in strategy_weights.keys():
                try:
                    individual_signals = get_signals(sn, data, **kwargs)
                    if (
                        isinstance(individual_signals, dict)
                        and "result" in individual_signals
                    ):
                        strategy_signals[sn] = individual_signals["result"]
                    else:
                        strategy_signals[sn] = individual_signals
                except Exception as e:
                    logging.warning(
                        f"Failed to generate signals for {sn}: {e}"
                    )
            combined_signals = ensemble.combine_signals(strategy_signals)
            return {
                "success": True,
                "result": combined_signals,
                "message": "Ensemble signals generated successfully",
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    except Exception as e:
        logging.error(f"Error getting signals for {strategy_name}: {e}")
        raise RuntimeError(f"Signal generation failed for {strategy_name}: {e}")


def get_available_strategies() -> List[str]:
    """Get list of available strategies."""
    return ["rsi", "bollinger", "macd", "sma", "cci", "atr", "ensemble"]


def create_strategy(strategy_name: str, **kwargs) -> Any:
    """Create a strategy instance (lazy imports)."""
    sn = strategy_name.lower()
    if sn == "rsi":
        return None
    if sn == "bollinger":
        from trading.strategies.bollinger_strategy import BollingerConfig, BollingerStrategy

        return BollingerStrategy(BollingerConfig(**kwargs))
    if sn == "macd":
        from trading.strategies.macd_strategy import MACDConfig, MACDStrategy

        return MACDStrategy(MACDConfig(**kwargs))
    if sn == "sma":
        from trading.strategies.sma_strategy import SMAConfig, SMAStrategy

        return SMAStrategy(SMAConfig(**kwargs))
    if sn == "cci":
        from trading.strategies.cci_strategy import CCIConfig, CCIStrategy

        return CCIStrategy(CCIConfig(**kwargs))
    if sn == "atr":
        from trading.strategies.atr_strategy import ATRConfig, ATRStrategy

        return ATRStrategy(ATRConfig(**kwargs))
    raise ValueError(f"Unknown strategy: {strategy_name}")


__all__ = [
    "get_signals",
    "get_available_strategies",
    "create_strategy",
]
