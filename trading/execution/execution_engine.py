"""Execution Engine for Trade Execution."""

import logging
import warnings

warnings.filterwarnings("ignore")

# Try to import execution libraries with fallbacks
try:
    import alpaca_trade_api as tradeapi

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    tradeapi = None

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
    return {"sharpe_ratio": 0.0, "total_return": 0.0}


logger = get_logger(__name__)


class ExecutionEngine:
    """Trade execution engine."""

    def __init__(self, config: TradingConfig = None):
        """Initialize execution engine."""
        self.config = config or TradingConfig()
        self.memory = AgentMemory()
        self.execution_history = []
        self.active_orders = {}

        # Initialize broker connections
        self._init_brokers()

    def _init_brokers(self):
        """Initialize broker connections."""
        self.brokers = {}

        # Alpaca
        if ALPACA_AVAILABLE and self.config.broker_api_key:
            try:
                self.brokers["alpaca"] = tradeapi.REST(
                    self.config.broker_api_key,
                    self.config.broker_secret_key,
                    "https://paper-api.alpaca.markets",  # Use paper trading
                )
                logger.info("Alpaca broker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca: {e}")

        # CCXT (for crypto)
        if CCXT_AVAILABLE:
            try:
                self.brokers["binance"] = ccxt.binance(
                    {
                        "apiKey": self.config.broker_api_key,
                        "secret": self.config.broker_secret_key,
                        "sandbox": True,  # Use testnet
                    }
                )
                logger.info("Binance broker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Binance: {e}")

    def execute_order(self, order: dict) -> dict:
        """Execute a trading order."""
        try:
            # Validate order
            validated_order = self._validate_order(order)
            if not validated_order:
                return {"status": "error", "message": "Invalid order"}

            # Execute based on mode
            if self.config.execution_mode == "live":
                result = self._execute_live_order(validated_order)
            else:
                result = self._execute_simulation_order(validated_order)

            # Store execution
            self.execution_history.append(
                {"timestamp": "now", "order": validated_order, "result": result}
            )

            # Update memory
            self.memory.memory.append(
                {"timestamp": "now", "type": "order_execution", "data": result}
            )

            return result

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return {"status": "error", "message": str(e)}

    def _validate_order(self, order: dict) -> dict:
        """Validate order parameters."""
        required_fields = ["symbol", "side", "quantity", "order_type"]

        for field in required_fields:
            if field not in order:
                logger.error(f"Missing required field: {field}")

        # Validate symbol
        if not isinstance(order["symbol"], str) or len(order["symbol"]) == 0:
            logger.error("Invalid symbol")
            return None

        # Validate side
        if order["side"] not in ["buy", "sell"]:
            logger.error("Invalid side")
            return None

        # Validate quantity
        if not isinstance(order["quantity"], (int, float)) or order["quantity"] <= 0:
            logger.error("Invalid quantity")
            return None

        # Validate order type
        if order["order_type"] not in ["market", "limit", "stop", "stop_limit"]:
            logger.error("Invalid order type")
            return None

        # Add default values
        order.setdefault("time_in_force", "day")
        order.setdefault("limit_price", None)
        order.setdefault("stop_price", None)

        return order

    def _execute_live_order(self, order: dict) -> dict:
        """Execute order on live broker."""
        try:
            if order["symbol"].endswith("USD"):  # Crypto
                return self._execute_crypto_order(order)
            else:  # Stock
                return self._execute_stock_order(order)

        except Exception as e:
            logger.error(f"Live execution error: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_stock_order(self, order: dict) -> dict:
        """Execute stock order via Alpaca."""
        if "alpaca" not in self.brokers:
            return {"status": "error", "message": "Alpaca broker not available"}

        try:
            # Prepare order parameters
            order_params = {
                "symbol": order["symbol"],
                "qty": order["quantity"],
                "side": order["side"],
                "type": order["order_type"],
                "time_in_force": order["time_in_force"],
            }

            if order["limit_price"]:
                order_params["limit_price"] = order["limit_price"]

            if order["stop_price"]:
                order_params["stop_price"] = order["stop_price"]

            # Submit order
            alpaca_order = self.brokers["alpaca"].submit_order(**order_params)

            return {
                "status": "success",
                "order_id": alpaca_order.id,
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "filled_quantity": alpaca_order.filled_qty,
                "average_price": alpaca_order.filled_avg_price,
                "status": alpaca_order.status,
                "timestamp": alpaca_order.submitted_at,
            }

        except Exception as e:
            logger.error(f"Alpaca execution error: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_crypto_order(self, order: dict) -> dict:
        """Execute crypto order via CCXT."""
        if "binance" not in self.brokers:
            return {"status": "error", "message": "Binance broker not available"}

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
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "filled_quantity": ccxt_order.get("filled", 0),
                "average_price": ccxt_order.get("average", 0),
                "status": ccxt_order["status"],
                "timestamp": ccxt_order["timestamp"],
            }

        except Exception as e:
            logger.error(f"CCXT execution error: {e}")
            return {"status": "error", "message": str(e)}

    def _execute_simulation_order(self, order: dict) -> dict:
        """Execute order in simulation mode."""
        try:
            # Simulate market conditions
            current_price = self._get_simulated_price(order["symbol"])

            # Simulate execution
            if order["order_type"] == "market":
                executed_price = current_price
                filled_quantity = order["quantity"]
            elif order["order_type"] == "limit":
                if order["side"] == "buy" and order["limit_price"] >= current_price:
                    executed_price = order["limit_price"]
                    filled_quantity = order["quantity"]
                elif order["side"] == "sell" and order["limit_price"] <= current_price:
                    executed_price = order["limit_price"]
                    filled_quantity = order["quantity"]
                else:
                    executed_price = 0
                    filled_quantity = 0
            else:
                executed_price = current_price
                filled_quantity = order["quantity"]

            # Simulate order ID
            import uuid

            order_id = str(uuid.uuid4())

            return {
                "status": "success",
                "order_id": order_id,
                "symbol": order["symbol"],
                "side": order["side"],
                "quantity": order["quantity"],
                "filled_quantity": filled_quantity,
                "average_price": executed_price,
                "status": "filled" if filled_quantity > 0 else "rejected",
                "timestamp": "now",
                "simulation": True,
            }

        except Exception as e:
            logger.error(f"Simulation execution error: {e}")
            return {"status": "error", "message": str(e)}

    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for symbol."""
        # Simple price simulation
        import random

        base_prices = {
            "AAPL": 150.0,
            "GOOGL": 2800.0,
            "MSFT": 300.0,
            "TSLA": 800.0,
            "BTCUSD": 45000.0,
            "ETHUSD": 3000.0,
        }

        base_price = base_prices.get(symbol, 100.0)
        variation = random.uniform(-0.02, 0.02)  # ±2% variation

        return base_price * (1 + variation)

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
            "success_rate": successful_orders / total_orders
            if total_orders > 0
            else 0.0,
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
