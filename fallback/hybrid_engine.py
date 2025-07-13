"""
Fallback Hybrid Engine Implementation

Provides fallback functionality for strategy execution when
the primary hybrid engine is unavailable.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FallbackHybridEngine:
    """
    Fallback implementation of the Hybrid Engine.

    Provides basic strategy execution functionality when the primary
    hybrid engine is unavailable.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback hybrid engine.

        Sets up basic logging and initializes strategy execution parameters
        for fallback operations.
        """
        self._status = "fallback"
        self._strategies = self._initialize_strategies()
        logger.info("FallbackHybridEngine initialized")

    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available strategies for fallback execution.

        Returns:
            Dict[str, Dict[str, Any]]: Strategy configurations
        """
        return {
            "rsi": {
                "name": "RSI Strategy",
                "function": self._execute_rsi_strategy,
                "parameters": {"period": 14, "overbought": 70, "oversold": 30},
            },
            "macd": {
                "name": "MACD Strategy",
                "function": self._execute_macd_strategy,
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            },
            "bollinger": {
                "name": "Bollinger Bands Strategy",
                "function": self._execute_bollinger_strategy,
                "parameters": {"period": 20, "std_dev": 2},
            },
            "sma_crossover": {
                "name": "SMA Crossover Strategy",
                "function": self._execute_sma_crossover_strategy,
                "parameters": {"fast_period": 10, "slow_period": 30},
            },
        }

    def run_strategy(
        self, data: pd.DataFrame, strategy_name: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run a specific strategy (fallback implementation).

        Args:
            data: Historical market data
            strategy_name: Name of the strategy to run
            parameters: Strategy parameters (optional)

        Returns:
            Optional[Dict[str, Any]]: Strategy execution results
        """
        try:
            logger.info(f"Running strategy: {strategy_name}")

            if strategy_name not in self._strategies:
                logger.warning(f"Unknown strategy: {strategy_name}")
                return None

            strategy_config = self._strategies[strategy_name]
            strategy_func = strategy_config["function"]

            # Use provided parameters or defaults
            if parameters is None:
                parameters = strategy_config["parameters"]

            # Execute strategy
            result = strategy_func(data, parameters)

            if result:
                result["strategy_name"] = strategy_name
                result["parameters_used"] = parameters
                result["timestamp"] = datetime.now().isoformat()
                result["fallback_mode"] = True

                logger.info(f"Successfully executed {strategy_name}")
                return result

            return None

        except Exception as e:
            logger.error(f"Error running strategy {strategy_name}: {e}")
            return None

    def _execute_rsi_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute RSI strategy (fallback implementation).

        Args:
            data: Historical market data
            parameters: Strategy parameters

        Returns:
            Optional[Dict[str, Any]]: RSI strategy results
        """
        try:
            period = parameters.get("period", 14)
            overbought = parameters.get("overbought", 70)
            oversold = parameters.get("oversold", 30)

            # Calculate RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Generate signals
            signals = pd.Series(index=data.index, data=0)
            signals[rsi < oversold] = 1  # Buy signal
            signals[rsi > overbought] = -1  # Sell signal

            # Calculate performance metrics
            returns = data["Close"].pct_change()
            strategy_returns = signals.shift(1) * returns

            result = {
                "signals": signals.tolist(),
                "rsi_values": rsi.tolist(),
                "total_return": strategy_returns.sum(),
                "sharpe_ratio": strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
                "win_rate": (strategy_returns > 0).mean(),
                "max_drawdown": self._calculate_max_drawdown(strategy_returns),
                "signal_count": len(signals[signals != 0]),
            }

            return result

        except Exception as e:
            logger.error(f"Error executing RSI strategy: {e}")
            return None

    def _execute_macd_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute MACD strategy (fallback implementation).

        Args:
            data: Historical market data
            parameters: Strategy parameters

        Returns:
            Optional[Dict[str, Any]]: MACD strategy results
        """
        try:
            fast_period = parameters.get("fast_period", 12)
            slow_period = parameters.get("slow_period", 26)
            signal_period = parameters.get("signal_period", 9)

            # Calculate MACD
            ema_fast = data["Close"].ewm(span=fast_period).mean()
            ema_slow = data["Close"].ewm(span=slow_period).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line

            # Generate signals
            signals = pd.Series(index=data.index, data=0)
            signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1  # Buy
            signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1  # Sell

            # Calculate performance metrics
            returns = data["Close"].pct_change()
            strategy_returns = signals.shift(1) * returns

            result = {
                "signals": signals.tolist(),
                "macd_line": macd_line.tolist(),
                "signal_line": signal_line.tolist(),
                "histogram": histogram.tolist(),
                "total_return": strategy_returns.sum(),
                "sharpe_ratio": strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
                "win_rate": (strategy_returns > 0).mean(),
                "max_drawdown": self._calculate_max_drawdown(strategy_returns),
                "signal_count": len(signals[signals != 0]),
            }

            return result

        except Exception as e:
            logger.error(f"Error executing MACD strategy: {e}")
            return None

    def _execute_bollinger_strategy(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute Bollinger Bands strategy (fallback implementation).

        Args:
            data: Historical market data
            parameters: Strategy parameters

        Returns:
            Optional[Dict[str, Any]]: Bollinger Bands strategy results
        """
        try:
            period = parameters.get("period", 20)
            std_dev = parameters.get("std_dev", 2)

            # Calculate Bollinger Bands
            sma = data["Close"].rolling(window=period).mean()
            std = data["Close"].rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            # Generate signals
            signals = pd.Series(index=data.index, data=0)
            signals[data["Close"] <= lower_band] = 1  # Buy signal
            signals[data["Close"] >= upper_band] = -1  # Sell signal

            # Calculate performance metrics
            returns = data["Close"].pct_change()
            strategy_returns = signals.shift(1) * returns

            result = {
                "signals": signals.tolist(),
                "upper_band": upper_band.tolist(),
                "lower_band": lower_band.tolist(),
                "sma": sma.tolist(),
                "total_return": strategy_returns.sum(),
                "sharpe_ratio": strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
                "win_rate": (strategy_returns > 0).mean(),
                "max_drawdown": self._calculate_max_drawdown(strategy_returns),
                "signal_count": len(signals[signals != 0]),
            }

            return result

        except Exception as e:
            logger.error(f"Error executing Bollinger Bands strategy: {e}")
            return None

    def _execute_sma_crossover_strategy(
        self, data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute SMA Crossover strategy (fallback implementation).

        Args:
            data: Historical market data
            parameters: Strategy parameters

        Returns:
            Optional[Dict[str, Any]]: SMA Crossover strategy results
        """
        try:
            fast_period = parameters.get("fast_period", 10)
            slow_period = parameters.get("slow_period", 30)

            # Calculate SMAs
            sma_fast = data["Close"].rolling(window=fast_period).mean()
            sma_slow = data["Close"].rolling(window=slow_period).mean()

            # Generate signals
            signals = pd.Series(index=data.index, data=0)
            signals[(sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))] = 1  # Buy
            signals[(sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))] = -1  # Sell

            # Calculate performance metrics
            returns = data["Close"].pct_change()
            strategy_returns = signals.shift(1) * returns

            result = {
                "signals": signals.tolist(),
                "sma_fast": sma_fast.tolist(),
                "sma_slow": sma_slow.tolist(),
                "total_return": strategy_returns.sum(),
                "sharpe_ratio": strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0,
                "win_rate": (strategy_returns > 0).mean(),
                "max_drawdown": self._calculate_max_drawdown(strategy_returns),
                "signal_count": len(signals[signals != 0]),
            }

            return result

        except Exception as e:
            logger.error(f"Error executing SMA Crossover strategy: {e}")
            return None

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown from returns series.

        Args:
            returns: Returns series

        Returns:
            float: Maximum drawdown
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategies (fallback implementation).

        Returns:
            List[str]: List of available strategy names
        """
        try:
            return list(self._strategies.keys())
        except Exception as e:
            logger.error(f"Error getting available strategies: {e}")
            return []

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback hybrid engine.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            return {
                "status": self._status,
                "available_strategies": len(self._strategies),
                "strategies": list(self._strategies.keys()),
                "fallback_mode": True,
                "message": "Using fallback hybrid engine",
            }
        except Exception as e:
            logger.error(f"Error getting fallback hybrid engine health: {e}")
            return {"status": "error", "available_strategies": 0, "fallback_mode": True, "error": str(e)}
