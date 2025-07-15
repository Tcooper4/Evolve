"""Execution Engine for Trade Execution.
Enhanced with Batch 10 features: detailed logging for skipped/failure cases.
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any

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
    """Trade execution engine with enhanced logging for failure cases."""

    def __init__(self, config: TradingConfig = None):
        """Initialize execution engine."""
        self.config = config or TradingConfig()
        self.memory = AgentMemory()
        self.execution_history = []
        self.active_orders = {}
        self.failed_orders = []  # Track failed orders for analysis
        self.skipped_orders = []  # Track skipped orders for analysis

        # Enhanced logging configuration
        self.log_failures = getattr(self.config, 'log_failures', True)
        self.log_skips = getattr(self.config, 'log_skips', True)
        self.max_slippage_threshold = getattr(self.config, 'max_slippage_threshold', 0.05)  # 5%
        self.min_price_threshold = getattr(self.config, 'min_price_threshold', 0.01)  # $0.01

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
                self._log_failure("broker_init", "alpaca", str(e))

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
                self._log_failure("broker_init", "binance", str(e))

    def execute_order(self, order: dict) -> dict:
        """Execute a trading order with enhanced failure logging."""
        try:
            # Validate order
            validated_order = self._validate_order(order)
            if not validated_order:
                failure_reason = "Invalid order parameters"
                self._log_failure("validation", order.get("symbol", "unknown"), failure_reason)
                return {"status": "error", "message": failure_reason}

            # Check pre-execution conditions
            skip_reason = self._check_execution_conditions(validated_order)
            if skip_reason:
                self._log_skip(validated_order["symbol"], skip_reason)
                return {"status": "skipped", "message": skip_reason}

            # Execute based on mode
            if self.config.execution_mode == "live":
                result = self._execute_live_order(validated_order)
            else:
                result = self._execute_simulation_order(validated_order)

            # Log execution result
            if result.get("status") == "error":
                self._log_failure("execution", validated_order["symbol"], result.get("message", "Unknown error"))
            elif result.get("status") == "skipped":
                self._log_skip(validated_order["symbol"], result.get("message", "Unknown skip reason"))

            # Store execution
            execution_record = {
                "timestamp": datetime.now().isoformat(),
                "order": validated_order,
                "result": result,
                "failure_reason": result.get("message") if result.get("status") in ["error", "skipped"] else None
            }
            self.execution_history.append(execution_record)

            # Update memory
            self.memory.memory.append({
                "timestamp": datetime.now().isoformat(),
                "type": "order_execution",
                "data": result,
                "failure_logged": result.get("status") in ["error", "skipped"]
            })

            return result

        except Exception as e:
            error_msg = f"Unexpected error during execution: {str(e)}"
            logger.error(error_msg)
            self._log_failure("unexpected", order.get("symbol", "unknown"), error_msg)
            return {"status": "error", "message": error_msg}

    def _check_execution_conditions(self, order: dict) -> Optional[str]:
        """Check if order should be skipped due to market conditions."""
        try:
            # Check price availability
            current_price = self._get_current_price(order["symbol"])
            if current_price is None:
                return "Price unavailable for symbol"
            
            if current_price < self.min_price_threshold:
                return f"Price too low: ${current_price:.4f} < ${self.min_price_threshold}"

            # Check slippage conditions
            if "limit_price" in order and order["limit_price"]:
                slippage = abs(current_price - order["limit_price"]) / current_price
                if slippage > self.max_slippage_threshold:
                    return f"Slippage too high: {slippage:.2%} > {self.max_slippage_threshold:.2%}"

            # Check market hours (for stocks)
            if not order["symbol"].endswith("USD") and not self._is_market_open():
                return "Market closed"

            # Check liquidity (simplified)
            if order["quantity"] > self._get_available_liquidity(order["symbol"]):
                return "Insufficient liquidity"

            return None

        except Exception as e:
            logger.error(f"Error checking execution conditions: {e}")
            return f"Error checking conditions: {str(e)}"

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        try:
            if symbol.endswith("USD"):  # Crypto
                if "binance" in self.brokers:
                    ticker = self.brokers["binance"].fetch_ticker(symbol)
                    return ticker.get("last")
            else:  # Stock
                if "alpaca" in self.brokers:
                    bar = self.brokers["alpaca"].get_latest_bar(symbol)
                    return bar.c if bar else None
            
            # Fallback to simulated price
            return self._get_simulated_price(symbol)

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def _is_market_open(self) -> bool:
        """Check if market is open."""
        try:
            if "alpaca" in self.brokers:
                clock = self.brokers["alpaca"].get_clock()
                return clock.is_open
            return True  # Assume open for simulation
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return True

    def _get_available_liquidity(self, symbol: str) -> float:
        """Get available liquidity for symbol (simplified)."""
        # Simplified liquidity check - in real implementation, this would check order book
        return 1000000.0  # Assume $1M liquidity

    def _log_failure(self, failure_type: str, symbol: str, reason: str):
        """Log execution failure with detailed information."""
        if not self.log_failures:
            return

        failure_record = {
            "timestamp": datetime.now().isoformat(),
            "failure_type": failure_type,
            "symbol": symbol,
            "reason": reason,
            "execution_mode": self.config.execution_mode,
            "brokers_available": list(self.brokers.keys())
        }

        self.failed_orders.append(failure_record)
        
        logger.error(f"EXECUTION FAILURE - Type: {failure_type}, Symbol: {symbol}, Reason: {reason}")
        
        # Log to file for analysis
        try:
            import json
            with open("logs/execution_failures.json", "a") as f:
                f.write(json.dumps(failure_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log failure to file: {e}")

    def _log_skip(self, symbol: str, reason: str):
        """Log skipped order with detailed information."""
        if not self.log_skips:
            return

        skip_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "reason": reason,
            "execution_mode": self.config.execution_mode,
            "current_price": self._get_current_price(symbol),
            "market_open": self._is_market_open()
        }

        self.skipped_orders.append(skip_record)
        
        logger.warning(f"ORDER SKIPPED - Symbol: {symbol}, Reason: {reason}")
        
        # Log to file for analysis
        try:
            import json
            with open("logs/execution_skips.json", "a") as f:
                f.write(json.dumps(skip_record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log skip to file: {e}")

    def _validate_order(self, order: dict) -> dict:
        """Validate order parameters with enhanced logging."""
        required_fields = ["symbol", "side", "quantity", "order_type"]

        for field in required_fields:
            if field not in order:
                logger.error(f"Missing required field: {field}")
                return None

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
        """Execute order on live broker with enhanced error handling."""
        try:
            if order["symbol"].endswith("USD"):  # Crypto
                return self._execute_crypto_order(order)
            else:  # Stock
                return self._execute_stock_order(order)

        except Exception as e:
            error_msg = f"Live execution error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

    def _execute_stock_order(self, order: dict) -> dict:
        """Execute stock order via Alpaca with enhanced error handling."""
        if "alpaca" not in self.brokers:
            error_msg = "Alpaca broker not available"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

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
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
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
            "recent_failures": self.failed_orders[-10:] if len(self.failed_orders) > 10 else self.failed_orders
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
            "recent_skips": self.skipped_orders[-10:] if len(self.skipped_orders) > 10 else self.skipped_orders
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
