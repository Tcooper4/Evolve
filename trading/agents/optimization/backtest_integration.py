"""
Backtest Integration Module

This module contains backtesting integration functionality for the optimizer agent.
Extracted from the original optimizer_agent.py for modularity.
"""

import logging
from typing import Any, Dict, List, Optional

from trading.backtesting.backtester import Backtester
from trading.evaluation.metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
)

from .strategy_optimizer import StrategyConfig


class BacktestIntegration:
    """Handles backtesting integration for optimization."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.backtester = Backtester()

    async def run_backtest(
        self,
        strategies: List,
        symbol: str,
        time_period: Dict[str, Any],
        config: Any,
    ) -> Optional[Dict[str, Any]]:
        """Run a backtest for optimization."""
        try:
            # Prepare strategies for backtesting
            strategy_instances = self._prepare_strategies(strategies)

            if not strategy_instances:
                self.logger.warning(f"No valid strategies for {symbol}")
                return None

            # Run backtest
            backtest_result = await self.backtester.run_backtest(
                symbol=symbol,
                strategies=strategy_instances,
                start_date=time_period.get("start_date"),
                end_date=time_period.get("end_date"),
                initial_capital=time_period.get("initial_capital", 100000),
                commission=time_period.get("commission", 0.001),
            )

            if not backtest_result:
                return None

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(backtest_result)

            # Check minimum trades requirement
            if performance_metrics.get("total_trades", 0) < config.min_trades:
                return None

            return {
                "symbol": symbol,
                "time_period": time_period,
                "backtest_result": backtest_result,
                "performance_metrics": performance_metrics,
            }

        except Exception as e:
            self.logger.error(f"Backtest failed for {symbol}: {e}")
            return None

    def _prepare_strategies(self, strategies: List) -> List[Any]:
        """Prepare strategy instances for backtesting."""
        strategy_instances = []

        for strategy_config in strategies:
            if isinstance(strategy_config, StrategyConfig):
                strategy_instance = self._create_strategy_instance(strategy_config)
                if strategy_instance:
                    strategy_instances.append(strategy_instance)
            else:
                # Handle other strategy formats
                strategy_instances.append(strategy_config)

        return strategy_instances

    def _create_strategy_instance(
        self, strategy_config: StrategyConfig
    ) -> Optional[Any]:
        """Create a strategy instance from configuration."""
        try:
            strategy_name = strategy_config.strategy_name.lower()
            parameters = strategy_config.parameters or {}

            if strategy_name == "bollinger":
                from trading.strategies.bollinger_strategy import BollingerStrategy

                return BollingerStrategy(**parameters)

            elif strategy_name == "macd":
                from trading.strategies.macd_strategy import MACDStrategy

                return MACDStrategy(**parameters)

            elif strategy_name == "rsi":
                from trading.strategies.rsi_strategy import RSIStrategy

                return RSIStrategy(**parameters)

            else:
                self.logger.warning(f"Unknown strategy: {strategy_name}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to create strategy instance: {e}")
            return None

    def _calculate_performance_metrics(
        self, backtest_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate performance metrics from backtest results."""
        try:
            # Extract returns from backtest result
            returns = backtest_result.get("returns", [])
            trades = backtest_result.get("trades", [])

            if not returns:
                return {}

            # Calculate basic metrics
            total_return = (
                (returns[-1] - returns[0]) / returns[0] if returns[0] > 0 else 0.0
            )

            # Calculate Sharpe ratio
            sharpe_ratio = calculate_sharpe_ratio(returns)

            # Calculate maximum drawdown
            max_drawdown = calculate_max_drawdown(returns)

            # Calculate win rate
            win_rate = calculate_win_rate(trades) if trades else 0.0

            # Calculate profit factor
            profit_factor = self._calculate_profit_factor(trades)

            # Calculate Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0

            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "calmar_ratio": calmar_ratio,
                "total_trades": len(trades),
                "avg_trade_return": total_return / len(trades) if trades else 0.0,
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate performance metrics: {e}")
            return {}

    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor from trades."""
        try:
            if not trades:
                return 0.0

            gross_profit = sum(
                trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) > 0
            )
            gross_loss = abs(
                sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) < 0)
            )

            return gross_profit / gross_loss if gross_loss > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Failed to calculate profit factor: {e}")
            return 0.0

    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtest configuration."""
        return self.config.copy()

    def update_backtest_config(self, updates: Dict[str, Any]) -> None:
        """Update backtest configuration."""
        self.config.update(updates)
