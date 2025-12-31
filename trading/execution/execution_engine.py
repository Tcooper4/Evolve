"""Execution Engine for Trade Execution.
Enhanced with Batch 10 features: detailed logging for skipped/failure cases.
"""

import logging
import warnings
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

# Try to import execution libraries with fallbacks
try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = True
    ccxt = None

# Import from trading package
try:
    from trading.config.configuration import TradingConfig
    from trading.evaluation.metrics import calculate_metrics
    from trading.memory.agent_memory import AgentMemory
    from trading.utils.common import get_logger
except ImportError:
    # Fallback imports
    def get_logger(name):
        return logging.getLogger(name)

    class TradingConfig:
        def __init__(self):
            self.execution_mode = "simulation"
            self.broker_api_key = None
            self.broker_secret_key = None


class AgentMemory:
    def __init__(self):
        self.memory = []


def calculate_metrics(returns):
    # Removed return statement - __init__ should not return values
    def __init__(self, config: TradingConfig = None):
                # Removed return statement - __init__ should not return values
                # Removed return statement - __init__ should not return values
            # Removed return statement - __init__ should not return values
            # Removed return statement - __init__ should not return values
            # Removed return statement - __init__ should not return values
                # Removed return statement - __init__ should not return values

            alpaca_order = self.brokers["alpaca"].submit_order(order_request)

            return {
                "status": "success",
                "order_id": alpaca_order.id,
                "symbol": alpaca_order.symbol,
                "quantity": alpaca_order.qty,
                "side": alpaca_order.side,
                "type": alpaca_order.type,
                "status": alpaca_order.status,
            }

        except Exception as e:
            error_msg = f"Alpaca execution error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def _execute_crypto_order(self, order: dict) -> dict:
        """Execute crypto order via CCXT with enhanced error handling."""
        if "binance" not in self.brokers:
            error_msg = "Binance broker not available"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        try:
            # Prepare order parameters
            order_params = {
                "symbol": order["symbol"],
                "type": order["order_type"],
                "side": order["side"],
                "amount": order["quantity"],
            }

            if order["limit_price"]:
                order_params["price"] = order["limit_price"]

            # Submit order
            ccxt_order = self.brokers["binance"].create_order(**order_params)

            return {
                "status": "success",
                "order_id": ccxt_order["id"],
                "symbol": ccxt_order["symbol"],
                "quantity": ccxt_order["amount"],
                "side": ccxt_order["side"],
                "type": ccxt_order["type"],
                "status": ccxt_order["status"],
            }

        except Exception as e:
            error_msg = f"CCXT execution error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def _execute_simulation_order(self, order: dict) -> dict:
        """Execute order in simulation mode with enhanced logging."""
        try:
            # Get simulated price
            simulated_price = self._get_simulated_price(order["symbol"])
            if simulated_price is None:
                error_msg = "Unable to get simulated price"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}

            # Calculate simulated execution
            execution_price = simulated_price
            if order["order_type"] == "limit" and order["limit_price"]:
                if order["side"] == "buy" and simulated_price > order["limit_price"]:
                    return {"status": "skipped", "message": "Limit price not met"}
                elif order["side"] == "sell" and simulated_price < order["limit_price"]:
                    return {"status": "skipped", "message": "Limit price not met"}
                execution_price = order["limit_price"]

            # Simulate execution delay and slippage
            import time

            time.sleep(0.1)  # Simulate execution delay

            # Calculate slippage
            slippage = abs(execution_price - simulated_price) / simulated_price
            if slippage > self.max_slippage_threshold:
                skip_msg = f"Simulated slippage too high: {slippage:.2%}"
                logger.warning(skip_msg)
                return {"status": "skipped", "message": skip_msg}

            return {
                "status": "success",
                "order_id": f"sim_{int(time.time())}",
                "symbol": order["symbol"],
                "quantity": order["quantity"],
                "side": order["side"],
                "type": order["order_type"],
                "execution_price": execution_price,
                "slippage": slippage,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            error_msg = f"Simulation execution error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for symbol."""
        # Simple price simulation - in real implementation, this would use market data
        import random

        base_price = 100.0
        if symbol.endswith("USD"):
            base_price = 50000.0  # Crypto prices

        # Add some randomness
        variation = random.uniform(-0.02, 0.02)  # Â±2% variation
        return base_price * (1 + variation)

    def get_failure_summary(self) -> dict:
        """Get summary of execution failures."""
        if not self.failed_orders:
            return {"total_failures": 0}

        failure_types = {}
        symbols_failed = {}

        for failure in self.failed_orders:
            failure_type = failure["failure_type"]
            symbol = failure["symbol"]

            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
            symbols_failed[symbol] = symbols_failed.get(symbol, 0) + 1

        return {
            "total_failures": len(self.failed_orders),
            "failure_types": failure_types,
            "symbols_failed": symbols_failed,
            "recent_failures": (
                self.failed_orders[-10:]
                if len(self.failed_orders) > 10
                else self.failed_orders
            ),
        }

    def get_skip_summary(self) -> dict:
        """Get summary of skipped orders."""
        if not self.skipped_orders:
            return {"total_skips": 0}

        skip_reasons = {}
        symbols_skipped = {}

        for skip in self.skipped_orders:
            reason = skip["reason"]
            symbol = skip["symbol"]

            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            symbols_skipped[symbol] = symbols_skipped.get(symbol, 0) + 1

        return {
            "total_skips": len(self.skipped_orders),
            "skip_reasons": skip_reasons,
            "symbols_skipped": symbols_skipped,
            "recent_skips": (
                self.skipped_orders[-10:]
                if len(self.skipped_orders) > 10
                else self.skipped_orders
            ),
        }

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an active order."""
        try:
            # Find order in active orders
            if order_id in self.active_orders:
                order = self.active_orders[order_id]

                # Cancel on broker if live
                if self.config.execution_mode == "live":
                    if order["symbol"].endswith("USD"):  # Crypto
                        if "binance" in self.brokers:
                            self.brokers["binance"].cancel_order(
                                order_id, order["symbol"]
                            )
                    else:  # Stock
                        if "alpaca" in self.brokers:
                            self.brokers["alpaca"].cancel_order(order_id)

                # Remove from active orders
                del self.active_orders[order_id]

                return {"status": "success", "message": "Order cancelled"}
            else:
                return {"status": "error", "message": "Order not found"}

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {"status": "error", "message": str(e)}

    def get_order_status(self, order_id: str) -> dict:
        """Get order status."""
        try:
            # Check active orders
            if order_id in self.active_orders:
                return self.active_orders[order_id]

            # Check execution history
            for execution in self.execution_history:
                if execution["result"].get("order_id") == order_id:
                    return execution["result"]

            return {"status": "error", "message": "Order not found"}

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"status": "error", "message": str(e)}

    def get_execution_summary(self) -> dict:
        """Get execution summary."""
        if not self.execution_history:
            return {
                "total_orders": 0,
                "successful_orders": 0,
                "failed_orders": 0,
                "total_volume": 0.0,
                "avg_execution_time": 0.0,
            }

        total_orders = len(self.execution_history)
        successful_orders = sum(
            1 for ex in self.execution_history if ex["result"]["status"] == "success"
        )
        failed_orders = total_orders - successful_orders

        total_volume = sum(
            ex["result"].get("filled_quantity", 0)
            * ex["result"].get("average_price", 0)
            for ex in self.execution_history
            if ex["result"]["status"] == "success"
        )

        return {
            "total_orders": total_orders,
            "successful_orders": successful_orders,
            "failed_orders": failed_orders,
            "success_rate": (
                successful_orders / total_orders if total_orders > 0 else 0.0
            ),
            "total_volume": total_volume,
            "avg_execution_time": 0.0,  # Would need timing data
        }

    def get_active_orders(self) -> list:
        """Get list of active orders."""
        return list(self.active_orders.values())

    def clear_history(self) -> dict:
        """Clear execution history.

        Returns:
            Dictionary with clear status and details
        """
        try:
            history_count = len(self.execution_history)
            self.execution_history = []
            logger.info("Execution history cleared")

            return {
                "success": True,
                "message": f"Execution history cleared successfully",
                "cleared_entries": history_count,
                "timestamp": "now",
            }

        except Exception as e:
            logger.error(f"Error clearing execution history: {e}")
            return {
                "success": False,
                "message": f"Error clearing execution history: {str(e)}",
                "timestamp": "now",
            }


# Global execution engine instance
execution_engine = ExecutionEngine()


def get_execution_engine() -> dict:
    """Get the global execution engine instance.

    Returns:
        Dictionary with execution engine status and instance
    """
    try:
        return {
            "success": True,
            "execution_engine": execution_engine,
            "execution_mode": execution_engine.config.execution_mode,
            "total_orders": len(execution_engine.execution_history),
            "active_orders": len(execution_engine.active_orders),
            "available_brokers": list(execution_engine.brokers.keys()),
            "timestamp": "now",
        }
    except Exception as e:
        logger.error(f"Error getting execution engine: {e}")
        return {
            "success": False,
            "message": f"Error getting execution engine: {str(e)}",
            "timestamp": "now",
        }
