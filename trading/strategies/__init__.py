"""Trading strategies module."""

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import functools

from trading.strategies.atr_strategy import ATRConfig, ATRStrategy, generate_atr_signals
from trading.strategies.bollinger_strategy import BollingerConfig, BollingerStrategy
from trading.strategies.cci_strategy import CCIConfig, CCIStrategy, generate_cci_signals
from trading.strategies.ensemble import (
    EnsembleConfig,
    WeightedEnsembleStrategy,
    create_ensemble_strategy,
    create_rsi_macd_bollinger_ensemble,
    create_balanced_ensemble,
    create_conservative_ensemble,
)
from trading.strategies.gatekeeper import (
    GatekeeperDecision,
    MarketRegime,
    RegimeClassifier,
    RegimeMetrics,
    StrategyGatekeeper,
    StrategyPerformance,
    StrategyStatus,
    create_strategy_gatekeeper,
)
from trading.strategies.macd_strategy import MACDConfig, MACDStrategy
from trading.strategies.rsi_signals import (
    generate_rsi_signals,
    generate_signals,
    load_optimized_settings,
)
from trading.strategies.sma_strategy import SMAConfig, SMAStrategy
from trading.strategies.strategy_manager import (
    Strategy,
    StrategyManager,
    StrategyMetrics,
)
from trading.strategies.strategy_runner import (
    AsyncStrategyRunner,
    run_strategies_parallel_example,
)


def get_signals(strategy_name: str, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Get trading signals for a specific strategy.

    Args:
        strategy_name: Name of the strategy to use
        data: Price data DataFrame
        **kwargs: Strategy-specific parameters

    Returns:
        Dictionary containing signals and metadata
    """
    try:
        if strategy_name.lower() == "rsi":
            return generate_signals(data, **kwargs)
        elif strategy_name.lower() == "bollinger":
            strategy = BollingerStrategy()
            return strategy.generate_signals(data, **kwargs)
        elif strategy_name.lower() == "macd":
            strategy = MACDStrategy()
            return strategy.generate_signals(data, **kwargs)
        elif strategy_name.lower() == "sma":
            strategy = SMAStrategy()
            return {
                "success": True,
                "result": strategy.generate_signals(data, **kwargs),
                "message": "Operation completed successfully",
                "timestamp": datetime.now().isoformat(),
            }
        elif strategy_name.lower() == "cci":
            return generate_cci_signals(data, **kwargs)
        elif strategy_name.lower() == "atr":
            return generate_atr_signals(data, **kwargs)
        elif strategy_name.lower() == "ensemble":
            # Handle ensemble strategy
            strategy_weights = kwargs.get("strategy_weights", {"rsi": 0.4, "macd": 0.4, "bollinger": 0.2})
            combination_method = kwargs.get("combination_method", "weighted_average")
            
            # Create ensemble strategy
            ensemble = create_ensemble_strategy(strategy_weights, combination_method, **kwargs)
            
            # Generate individual strategy signals
            strategy_signals = {}
            for strategy_name in strategy_weights.keys():
                try:
                    individual_signals = get_signals(strategy_name, data, **kwargs)
                    if isinstance(individual_signals, dict) and "result" in individual_signals:
                        strategy_signals[strategy_name] = individual_signals["result"]
                    else:
                        strategy_signals[strategy_name] = individual_signals
                except Exception as e:
                    logging.warning(f"Failed to generate signals for {strategy_name}: {e}")
            
            # Combine signals
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
    """Get list of available strategies.

    Returns:
        List[str]: Available strategy names
    """
    return ["rsi", "bollinger", "macd", "sma", "cci", "atr", "ensemble"]


@functools.lru_cache(maxsize=32)
def create_strategy(strategy_name: str, **kwargs) -> Any:
    """Create a strategy instance.

    Args:
        strategy_name: Name of the strategy
        **kwargs: Strategy parameters

    Returns:
        Strategy instance
    """
    strategy_map = {
        "rsi": lambda: None,  # RSI uses function-based approach
        "bollinger": lambda: BollingerStrategy(BollingerConfig(**kwargs)),
        "macd": lambda: MACDStrategy(MACDConfig(**kwargs)),
        "sma": lambda: SMAStrategy(SMAConfig(**kwargs)),
        "cci": lambda: CCIStrategy(CCIConfig(**kwargs)),
        "atr": lambda: ATRStrategy(ATRConfig(**kwargs)),
    }

    if strategy_name.lower() not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategy_map[strategy_name.lower()]()


__all__ = [
    "StrategyManager",
    "Strategy",
    "StrategyMetrics",
    "generate_rsi_signals",
    "load_optimized_settings",
    "generate_signals",
    "get_signals",
    "get_available_strategies",
    "create_strategy",
    "BollingerStrategy",
    "BollingerConfig",
    "MACDStrategy",
    "MACDConfig",
    "SMAStrategy",
    "SMAConfig",
    "CCIStrategy",
    "CCIConfig",
    "generate_cci_signals",
    "ATRStrategy",
    "ATRConfig",
    "generate_atr_signals",
    "StrategyGatekeeper",
    "RegimeClassifier",
    "MarketRegime",
    "StrategyStatus",
    "GatekeeperDecision",
    "RegimeMetrics",
    "StrategyPerformance",
    "create_strategy_gatekeeper",
    "WeightedEnsembleStrategy",
    "EnsembleConfig",
    "create_ensemble_strategy",
    "create_rsi_macd_bollinger_ensemble",
    "create_balanced_ensemble",
    "create_conservative_ensemble",
    "AsyncStrategyRunner",
    "run_strategies_parallel_example",
]
