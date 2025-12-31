"""
Execution Providers Module

This module contains execution provider classes for different trading platforms.
Extracted from the original execution_agent.py for modularity.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict

from .trade_signals import TradeSignal


class ExecutionMode(Enum):
    """Execution mode enum."""

    SIMULATION = "simulation"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ROBINHOOD = "robinhood"


class ExecutionProvider(ABC):
    """Abstract base class for execution providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the execution platform."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the execution platform."""

    @abstractmethod
    async def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade based on the signal."""

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""

    @abstractmethod
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""


class SimulationProvider(ExecutionProvider):
    """Simulation execution provider for testing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trades = []
        self.positions = {}
        self.account_balance = config.get("initial_balance", 100000.0)

    async def connect(self) -> bool:
        """Connect to simulation environment."""
        self.is_connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from simulation environment."""
        self.is_connected = False

    async def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a simulated trade."""
        if not self.is_connected:
            raise RuntimeError("Simulation provider not connected")

        # Simulate trade execution
        execution_price = signal.entry_price
        fees = execution_price * 0.001  # 0.1% fee simulation

        trade_result = {
            "success": True,
            "execution_price": execution_price,
            "fees": fees,
            "timestamp": datetime.utcnow().isoformat(),
            "order_id": f"sim_{len(self.trades) + 1}",
            "slippage": 0.0,
        }

        self.trades.append(trade_result)
        return trade_result

    async def get_account_info(self) -> Dict[str, Any]:
        """Get simulated account information."""
        # Removed return statement - __init__ should not return values

    async def get_positions(self) -> Dict[str, Any]:
        """Get simulated positions."""
        return self.positions


class AlpacaProvider(ExecutionProvider):
    """Alpaca execution provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.base_url = config.get("base_url", "https://paper-api.alpaca.markets")

    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            # Import alpaca-py here to avoid dependency issues
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.trading.client import TradingClient
            except ImportError as e:
                print("âš ï¸ alpaca-py not available. Cannot connect to Alpaca.")
                print(f"   Missing: {e}")
                return False

            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=True if "paper" in self.base_url else False,
            )
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key, secret_key=self.secret_key
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self.is_connected = False

    async def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade via Alpaca."""
        if not self.is_connected:
            raise RuntimeError("Alpaca provider not connected")

        try:
            # Import alpaca-py components
            try:
                from alpaca.trading.enums import OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest
            except ImportError as e:
                print("âš ï¸ alpaca-py not available. Cannot execute trade.")
                print(f"   Missing: {e}")
                # Removed return statement - __init__ should not return values

            # Create market order request
            order_data = MarketOrderRequest(
                symbol=signal.symbol,
                qty=signal.size,
                side=(
                    OrderSide.BUY
                    if signal.direction.value == "long"
                    else OrderSide.SELL
                ),
                time_in_force=TimeInForce.DAY,
            )

            # Submit order
            order = self.trading_client.submit_order(order_data)

            return {
                "success": True,
                "execution_price": signal.entry_price,
                "fees": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
                "order_id": order.id,
                "slippage": 0.0,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_account_info(self) -> Dict[str, Any]:
        """Get Alpaca account information."""
        if not self.is_connected:
            return {}

        try:
            account = self.trading_client.get_account()
            return {
                "balance": float(account.cash),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "cash": float(account.cash),
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_positions(self) -> Dict[str, Any]:
        """Get Alpaca positions."""
        if not self.is_connected:
            return {}

        try:
            positions = self.trading_client.get_all_positions()
            return {
                pos.symbol: {
                    "quantity": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "market_value": float(pos.market_value),
                }
                for pos in positions
            }
        except Exception as e:
            return {"error": str(e)}


class IBProvider(ExecutionProvider):
    """Interactive Brokers execution provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 7497)
        self.client_id = config.get("client_id", 1)

    async def connect(self) -> bool:
        """Connect to Interactive Brokers TWS/Gateway."""
        try:
            # Import ib_insync here to avoid dependency issues
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to IB: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Interactive Brokers."""
        if hasattr(self, "ib"):
            self.ib.disconnect()
        self.is_connected = False

    async def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade via Interactive Brokers."""
        if not self.is_connected:
            raise RuntimeError("IB provider not connected")

        try:
            # Place order via IB API
            from ib_insync import MarketOrder, Stock

            contract = Stock(signal.symbol, "SMART", "USD")
            order = MarketOrder(
                "BUY" if signal.direction.value == "long" else "SELL", signal.size
            )

            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Wait for order to be processed

            # Removed return statement - __init__ should not return values
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_account_info(self) -> Dict[str, Any]:
        """Get IB account information."""
        if not self.is_connected:
            return {}

        try:
            account_values = self.ib.accountSummary()
            return {
                "balance": float(
                    [av for av in account_values if av.tag == "NetLiquidation"][0].value
                ),
                "buying_power": float(
                    [av for av in account_values if av.tag == "BuyingPower"][0].value
                ),
                "equity": float(
                    [av for av in account_values if av.tag == "NetLiquidation"][0].value
                ),
                "cash": float(
                    [av for av in account_values if av.tag == "AvailableFunds"][0].value
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_positions(self) -> Dict[str, Any]:
        """Get IB positions."""
        if not self.is_connected:
            return {}

        try:
            positions = self.ib.positions()
            return {
                pos.contract.symbol: {
                    "quantity": pos.position,
                    "avg_entry_price": pos.avgCost,
                    "market_value": pos.position * pos.avgCost,
                }
                for pos in positions
                if pos.position != 0
            }
        except Exception as e:
            return {"error": str(e)}


class RobinhoodProvider(ExecutionProvider):
    """Robinhood execution provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.username = config.get("username")
        self.password = config.get("password")
        self.mfa_code = config.get("mfa_code")

    async def connect(self) -> bool:
        """Connect to Robinhood API."""
        try:
            # Import robin_stocks here to avoid dependency issues
            import robin_stocks.robinhood as rh

            rh.login(self.username, self.password, mfa_code=self.mfa_code)
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Robinhood: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Robinhood API."""
        self.is_connected = False

    async def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a trade via Robinhood."""
        if not self.is_connected:
            raise RuntimeError("Robinhood provider not connected")

        try:
            # Place order via Robinhood API
            import robin_stocks.robinhood as rh

            side = "buy" if signal.direction.value == "long" else "sell"
            order = rh.order_market(
                symbol=signal.symbol, side=side, quantity=signal.size
            )

            # Removed return statement - __init__ should not return values
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_account_info(self) -> Dict[str, Any]:
        """Get Robinhood account information."""
        if not self.is_connected:
            return {}

        try:
            import robin_stocks.robinhood as rh

            account = rh.load_account_profile()
            return {
                "balance": float(account.get("cash", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "equity": float(account.get("equity", 0)),
                "cash": float(account.get("cash", 0)),
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_positions(self) -> Dict[str, Any]:
        """Get Robinhood positions."""
        if not self.is_connected:
            return {}

        try:
            import robin_stocks.robinhood as rh

            positions = rh.get_open_stock_positions()
            return {
                pos.get("symbol"): {
                    "quantity": float(pos.get("quantity", 0)),
                    "avg_entry_price": float(pos.get("average_buy_price", 0)),
                    "market_value": float(pos.get("quantity", 0))
                    * float(pos.get("average_buy_price", 0)),
                }
                for pos in positions
                if float(pos.get("quantity", 0)) > 0
            }
        except Exception as e:
            return {"error": str(e)}


def create_execution_provider(
    mode: ExecutionMode, config: Dict[str, Any]
) -> ExecutionProvider:
    """Factory function to create execution providers."""
    if mode == ExecutionMode.SIMULATION:
        return SimulationProvider(config)
    elif mode == ExecutionMode.ALPACA:
        return AlpacaProvider(config)
    elif mode == ExecutionMode.INTERACTIVE_BROKERS:
        return IBProvider(config)
    elif mode == ExecutionMode.ROBINHOOD:
        return RobinhoodProvider(config)
    else:
        raise ValueError(f"Unsupported execution mode: {mode}")
