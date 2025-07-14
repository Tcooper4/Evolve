"""
Fallback Strategy Selector Implementation

Provides fallback functionality for strategy selection when
the primary strategy selector is unavailable.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FallbackStrategySelector:
    """
    Fallback implementation of the Strategy Selector.

    Provides basic strategy selection functionality when the primary
    strategy selector is unavailable.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback strategy selector.

        Sets up basic logging and initializes strategy configurations for
        fallback operations.
        """
        self._status = "fallback"
        self._strategies = self._initialize_strategies()
        logger.info("FallbackStrategySelector initialized")

    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available strategies for fallback selection.

        Returns:
            Dict[str, Dict[str, Any]]: Strategy configurations
        """
        return {
            "rsi": {
                "name": "RSI Strategy",
                "description": "Relative Strength Index mean reversion strategy",
                "parameters": {"period": 14, "overbought": 70, "oversold": 30},
                "performance": {
                    "sharpe_ratio": 0.85,
                    "win_rate": 0.62,
                    "max_drawdown": 0.12,
                    "avg_return": 0.08,
                },
                "best_regime": "mean_reversion",
                "risk_level": "medium",
            },
            "macd": {
                "name": "MACD Strategy",
                "description": "Moving Average Convergence Divergence trend following",
                "parameters": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                },
                "performance": {
                    "sharpe_ratio": 0.92,
                    "win_rate": 0.58,
                    "max_drawdown": 0.15,
                    "avg_return": 0.12,
                },
                "best_regime": "trending",
                "risk_level": "medium",
            },
            "bollinger": {
                "name": "Bollinger Bands Strategy",
                "description": "Bollinger Bands volatility-based strategy",
                "parameters": {"period": 20, "std_dev": 2},
                "performance": {
                    "sharpe_ratio": 0.78,
                    "win_rate": 0.55,
                    "max_drawdown": 0.18,
                    "avg_return": 0.09,
                },
                "best_regime": "volatile",
                "risk_level": "high",
            },
            "sma_crossover": {
                "name": "SMA Crossover Strategy",
                "description": "Simple Moving Average crossover strategy",
                "parameters": {"fast_period": 10, "slow_period": 30},
                "performance": {
                    "sharpe_ratio": 0.75,
                    "win_rate": 0.52,
                    "max_drawdown": 0.20,
                    "avg_return": 0.07,
                },
                "best_regime": "trending",
                "risk_level": "low",
            },
        }

    def select_strategy(
        self, market_data: pd.DataFrame, regime: str, risk_tolerance: str = "medium"
    ) -> Optional[Dict[str, Any]]:
        """
        Select the best strategy based on market data and regime (fallback implementation).

        Args:
            market_data: Historical market data
            regime: Current market regime
            risk_tolerance: User's risk tolerance level

        Returns:
            Optional[Dict[str, Any]]: Selected strategy configuration
        """
        try:
            logger.info(
                f"Selecting strategy for regime: {regime}, risk: {risk_tolerance}"
            )

            # Filter strategies based on regime and risk tolerance
            suitable_strategies = []

            for strategy_name, strategy_config in self._strategies.items():
                if (
                    strategy_config["best_regime"] == regime
                    and strategy_config["risk_level"] == risk_tolerance
                ):
                    suitable_strategies.append((strategy_name, strategy_config))

            # If no exact match, find closest matches
            if not suitable_strategies:
                for strategy_name, strategy_config in self._strategies.items():
                    if strategy_config["risk_level"] == risk_tolerance:
                        suitable_strategies.append((strategy_name, strategy_config))

            # If still no matches, use all strategies
            if not suitable_strategies:
                suitable_strategies = list(self._strategies.items())

            # Select best strategy based on Sharpe ratio
            best_strategy = None
            best_sharpe = float("-inf")

            for strategy_name, strategy_config in suitable_strategies:
                sharpe = strategy_config["performance"]["sharpe_ratio"]
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_strategy = (strategy_name, strategy_config)

            if best_strategy:
                strategy_name, strategy_config = best_strategy
                result = {
                    "selected_strategy": strategy_name,
                    "strategy_config": strategy_config,
                    "selection_reason": f"Best Sharpe ratio ({best_sharpe:.2f}) for {regime} regime",
                    "confidence": 0.7,
                    "timestamp": datetime.now().isoformat(),
                    "fallback_mode": True,
                }

                logger.info(
                    f"Selected strategy: {strategy_name} with Sharpe {best_sharpe:.2f}"
                )
                return result

            return None

        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return None

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """
        Get list of available strategies (fallback implementation).

        Returns:
            List[Dict[str, Any]]: List of available strategies
        """
        try:
            strategies = []
            for strategy_name, config in self._strategies.items():
                strategy_info = {
                    "name": strategy_name,
                    "display_name": config["name"],
                    "description": config["description"],
                    "risk_level": config["risk_level"],
                    "best_regime": config["best_regime"],
                    "performance": config["performance"],
                }
                strategies.append(strategy_info)

            return strategies

        except Exception as e:
            logger.error(f"Error getting available strategies: {e}")
            return []

    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a specific strategy (fallback implementation).

        Args:
            strategy_name: Name of the strategy

        Returns:
            Optional[Dict[str, Any]]: Strategy performance metrics
        """
        try:
            if strategy_name in self._strategies:
                performance = self._strategies[strategy_name]["performance"].copy()
                performance["strategy_name"] = strategy_name
                performance["timestamp"] = datetime.now().isoformat()
                performance["fallback_mode"] = True
                return performance
            else:
                logger.warning(f"Unknown strategy: {strategy_name}")
                return None

        except Exception as e:
            logger.error(f"Error getting strategy performance for {strategy_name}: {e}")
            return None

    def optimize_strategy_parameters(
        self, strategy_name: str, market_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Optimize strategy parameters (fallback implementation).

        Args:
            strategy_name: Name of the strategy to optimize
            market_data: Historical market data for optimization

        Returns:
            Optional[Dict[str, Any]]: Optimized parameters
        """
        try:
            logger.info(f"Optimizing parameters for strategy: {strategy_name}")

            if strategy_name not in self._strategies:
                logger.warning(f"Unknown strategy: {strategy_name}")
                return None

            # Simple parameter optimization based on data characteristics
            volatility = market_data["Close"].pct_change().std()

            if strategy_name == "rsi":
                # Adjust RSI parameters based on volatility
                if volatility > 0.03:  # High volatility
                    optimized_params = {"period": 10, "overbought": 75, "oversold": 25}
                else:  # Low volatility
                    optimized_params = {"period": 20, "overbought": 65, "oversold": 35}

            elif strategy_name == "macd":
                # Adjust MACD parameters based on volatility
                if volatility > 0.03:
                    optimized_params = {
                        "fast_period": 8,
                        "slow_period": 21,
                        "signal_period": 7,
                    }
                else:
                    optimized_params = {
                        "fast_period": 15,
                        "slow_period": 30,
                        "signal_period": 12,
                    }

            elif strategy_name == "bollinger":
                # Adjust Bollinger parameters based on volatility
                if volatility > 0.03:
                    optimized_params = {"period": 15, "std_dev": 2.5}
                else:
                    optimized_params = {"period": 25, "std_dev": 1.8}

            else:
                # Use default parameters
                optimized_params = self._strategies[strategy_name]["parameters"]

            result = {
                "strategy_name": strategy_name,
                "original_params": self._strategies[strategy_name]["parameters"],
                "optimized_params": optimized_params,
                "optimization_reason": f"Adjusted for volatility level: {volatility:.4f}",
                "timestamp": datetime.now().isoformat(),
                "fallback_mode": True,
            }

            logger.info(f"Optimized parameters for {strategy_name}")
            return result

        except Exception as e:
            logger.error(
                f"Error optimizing strategy parameters for {strategy_name}: {e}"
            )
            return None

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback strategy selector.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                "status": self._status,
                "available_strategies": len(self._strategies),
                "strategies": list(self._strategies.keys()),
                "fallback_mode": True,
                "message": "Using fallback strategy selector",
            }
        except Exception as e:
            logger.error(f"Error getting fallback strategy selector health: {e}")
            return {
                "status": "error",
                "available_strategies": 0,
                "fallback_mode": True,
                "error": str(e),
            }
