"""
Execution Agent

This agent handles trade execution, position tracking, and portfolio management.
It uses modular components for risk controls, trade signals, execution providers, and position management.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Local imports
from trading.agents.base_agent_interface import AgentConfig, AgentResult, BaseAgent
from trading.memory.agent_memory import AgentMemory
from trading.portfolio.portfolio_manager import PortfolioManager, Position
from trading.portfolio.position_sizer import (
    MarketContext,
    PortfolioContext,
    PositionSizer,
    SignalContext,
    SizingParameters,
    SizingStrategy,
)

from .execution_providers import (
    ExecutionMode,
    ExecutionProvider,
    create_execution_provider,
)
from .position_manager import PositionManager

# Modular imports
from .risk_controls import RiskControls, create_default_risk_controls
from .trade_signals import ExecutionResult, TradeSignal


class ExecutionAgent(BaseAgent):
    """
    Execution Agent for trade execution and portfolio management.

    This agent handles:
    - Trade signal processing and execution
    - Risk management and position monitoring
    - Portfolio tracking and performance analysis
    - Multi-provider execution (simulation, Alpaca, IB, Robinhood)
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.execution_mode = ExecutionMode(config.get("execution_mode", "simulation"))
        self.risk_controls = self._load_risk_controls()

        # Initialize components
        self.portfolio_manager = PortfolioManager(config.get("portfolio_config", {}))
        self.position_sizer = PositionSizer(config.get("sizing_config", {}))
        self.position_manager = PositionManager(config.get("position_config", {}))
        self.agent_memory = AgentMemory(config.get("memory_config", {}))

        # Initialize execution providers
        self.execution_providers: Dict[ExecutionMode, ExecutionProvider] = {}
        self._initialize_execution_providers()

        # Market data cache
        self.market_data_cache = {}
        self.global_metrics = {}

        # Performance tracking
        self.execution_history = []
        self.trade_log = []

        # Initialize storage
        self._initialize_storage()

    def _load_risk_controls(self) -> RiskControls:
        """Load risk controls from configuration."""
        risk_config = self.config.get("risk_controls", {})
        if risk_config:
            return RiskControls.from_dict(risk_config)
        return create_default_risk_controls()

    def _initialize_execution_providers(self) -> None:
        """Initialize execution providers for different platforms."""
        provider_configs = self.config.get("execution_providers", {})

        # Initialize simulation provider (always available)
        sim_config = provider_configs.get("simulation", {})
        self.execution_providers[ExecutionMode.SIMULATION] = create_execution_provider(
            ExecutionMode.SIMULATION, sim_config
        )

        # Initialize other providers based on configuration
        if self.config.get("enable_alpaca", False):
            alpaca_config = provider_configs.get("alpaca", {})
            self.execution_providers[ExecutionMode.ALPACA] = create_execution_provider(
                ExecutionMode.ALPACA, alpaca_config
            )

        if self.config.get("enable_ib", False):
            ib_config = provider_configs.get("interactive_brokers", {})
            self.execution_providers[
                ExecutionMode.INTERACTIVE_BROKERS
            ] = create_execution_provider(ExecutionMode.INTERACTIVE_BROKERS, ib_config)

        if self.config.get("enable_robinhood", False):
            rh_config = provider_configs.get("robinhood", {})
            self.execution_providers[
                ExecutionMode.ROBINHOOD
            ] = create_execution_provider(ExecutionMode.ROBINHOOD, rh_config)

    def _initialize_storage(self) -> None:
        """Initialize storage for execution data."""
        storage_dir = Path("data/execution")
        storage_dir.mkdir(parents=True, exist_ok=True)

        self.execution_history_file = storage_dir / "execution_history.json"
        self.trade_log_file = storage_dir / "trade_log.json"

        # Load existing data
        self._load_execution_history()
        self._load_trade_log()

    def _load_execution_history(self) -> None:
        """Load execution history from storage."""
        try:
            if self.execution_history_file.exists():
                with open(self.execution_history_file, "r") as f:
                    self.execution_history = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load execution history: {e}")
            self.execution_history = []

    def _load_trade_log(self) -> None:
        """Load trade log from storage."""
        try:
            if self.trade_log_file.exists():
                with open(self.trade_log_file, "r") as f:
                    self.trade_log = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load trade log: {e}")
            self.trade_log = []

    def _save_execution_history(self) -> None:
        """Save execution history to storage."""
        try:
            with open(self.execution_history_file, "w") as f:
                json.dump(self.execution_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save execution history: {e}")

    def _save_trade_log(self) -> None:
        """Save trade log to storage."""
        try:
            with open(self.trade_log_file, "w") as f:
                json.dump(self.trade_log, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save trade log: {e}")

    async def execute(self, **kwargs) -> AgentResult:
        """
        Main execution method for the agent.

        Args:
            **kwargs: Execution parameters including:
                - signal: TradeSignal to execute
                - market_data: Current market data
                - force_execution: Force execution regardless of conditions

        Returns:
            AgentResult with execution status and results
        """
        try:
            signal = kwargs.get("signal")
            market_data = kwargs.get("market_data", {})
            _unused_var = kwargs.get(
                "force_execution", False
            )  # Placeholder, flake8 ignore: F841

            if not signal:
                return AgentResult(
                    success=False, message="No trade signal provided", data={}
                )

            # Update market data cache
            self._update_market_data_cache(market_data)

            # Process the trade signal
            result = await self._process_trade_signal(signal, market_data)

            # Log execution result
            self._log_execution_result(result)

            # Update execution history
            self.execution_history.append(result.to_dict())
            self._save_execution_history()

            return AgentResult(
                success=result.success, message=result.message, data=result.to_dict()
            )

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return AgentResult(
                success=False,
                message=f"Execution failed: {str(e)}",
                data={"error": str(e)},
            )

    async def _process_trade_signal(
        self, signal: TradeSignal, market_data: Dict[str, Any]
    ) -> ExecutionResult:
        """Process a trade signal and execute the trade."""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return ExecutionResult(
                    success=False,
                    signal=signal,
                    message="Invalid trade signal",
                    error="Signal validation failed",
                )

            # Check position limits
            if not self._check_position_limits(signal):
                return ExecutionResult(
                    success=False,
                    signal=signal,
                    message="Position limits exceeded",
                    error="Position limit check failed",
                )

            # Calculate execution price
            execution_price = self._calculate_execution_price(signal, market_data)
            
            # Log decision for parity checking
            try:
                from testing.parity_checker import get_parity_checker
                from datetime import datetime
                
                parity_checker = get_parity_checker()
                
                # Extract features from market data
                features = {
                    "price": execution_price,
                    "quantity": signal.size or 1.0,
                    "confidence": signal.confidence,
                    "strategy": signal.strategy,
                }
                
                # Add market data features if available
                symbol_key = f"{signal.symbol}_price"
                if symbol_key in market_data:
                    features["market_price"] = market_data[symbol_key]
                
                # Create signal dict
                signal_dict = {
                    "action": signal.direction.value if hasattr(signal.direction, 'value') else str(signal.direction),
                    "quantity": signal.size or 1.0,
                    "price": execution_price,
                    "entry_price": signal.entry_price,
                }
                
                # Log live decision
                parity_checker.log_live_decision(
                    timestamp=datetime.now(),
                    symbol=signal.symbol,
                    signal=signal_dict,
                    features=features,
                    context={"strategy": signal.strategy, "live": True},
                )
            except Exception as e:
                # Don't fail if parity checker not available
                self.logger.debug(f"Could not log live decision for parity: {e}")

            # Execute trade based on mode
            if self.execution_mode == ExecutionMode.SIMULATION:
                position = await self._execute_simulation_trade(signal, execution_price)
            else:
                position = await self._execute_real_trade(signal, execution_price)

            # Calculate fees
            fees = self._calculate_fees(signal, execution_price)

            # Calculate risk metrics
            risk_metrics = self.position_manager.calculate_position_risk_metrics(
                position, execution_price, market_data
            )

            return ExecutionResult(
                success=True,
                signal=signal,
                position=position,
                execution_price=execution_price,
                fees=fees,
                message="Trade executed successfully",
                risk_metrics=risk_metrics,
            )

        except Exception as e:
            self.logger.error(f"Failed to process trade signal: {e}")
            return ExecutionResult(
                success=False,
                signal=signal,
                message=f"Trade processing failed: {str(e)}",
                error=str(e),
            )

    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate a trade signal."""
        if not signal.symbol or not signal.direction:
            return False

        if signal.confidence < 0.0 or signal.confidence > 1.0:
            return False

        if signal.entry_price <= 0:
            return False

        return True

    def _check_position_limits(self, signal: TradeSignal) -> bool:
        """Check if the signal complies with position limits."""
        # Check existing positions for the symbol
        existing_positions = self.position_manager.get_positions_by_symbol(
            signal.symbol
        )

        # Check if we already have a position in the opposite direction
        for position in existing_positions:
            if position.direction != signal.direction:
                return False

        # Check portfolio risk limits
        portfolio_risk = self.position_manager.calculate_portfolio_risk_metrics({})
        if portfolio_risk.get("total_value", 0) > self.risk_controls.max_portfolio_risk:
            return False

        return True

    def _calculate_execution_price(
        self, signal: TradeSignal, market_data: Dict[str, Any]
    ) -> float:
        """Calculate execution price for the trade."""
        # Use market data if available
        market_price = market_data.get(f"{signal.symbol}_price")
        if market_price:
            return market_price

        # Fall back to signal entry price
        return signal.entry_price

    async def _execute_simulation_trade(
        self, signal: TradeSignal, execution_price: float
    ) -> Position:
        """Execute a trade in simulation mode."""
        # Calculate position size
        size, sizing_info = self._calculate_position_size(signal, execution_price)

        # Create position
        position = Position(
            position_id=f"sim_{len(self.position_manager.positions) + 1}",
            symbol=signal.symbol,
            direction=signal.direction,
            size=size,
            entry_price=execution_price,
            entry_time=datetime.utcnow(),
            strategy=signal.strategy,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
        )

        # Add to position manager
        self.position_manager.add_position(position)

        # Log trade
        trade_log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "size": size,
            "price": execution_price,
            "strategy": signal.strategy,
            "confidence": signal.confidence,
            "sizing_info": sizing_info,
        }
        self.trade_log.append(trade_log_entry)
        self._save_trade_log()

        return position

    async def _execute_real_trade(
        self, signal: TradeSignal, execution_price: float
    ) -> Position:
        """Execute a trade using the configured execution provider."""
        provider = self.execution_providers.get(self.execution_mode)
        if not provider:
            raise RuntimeError(
                f"No execution provider available for mode: {self.execution_mode}"
            )

        # Execute via provider
        result = await provider.execute_trade(signal)

        if not result.get("success", False):
            raise RuntimeError(
                f"Trade execution failed: {result.get('error', 'Unknown error')}"
            )

        # Create position from execution result
        position = Position(
            position_id=result.get(
                "order_id", f"real_{len(self.position_manager.positions) + 1}"
            ),
            symbol=signal.symbol,
            direction=signal.direction,
            size=signal.size or 1.0,
            entry_price=result.get("execution_price", execution_price),
            entry_time=datetime.utcnow(),
            strategy=signal.strategy,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
        )

        # Add to position manager
        self.position_manager.add_position(position)

        return position

    def _calculate_position_size(
        self,
        signal: TradeSignal,
        execution_price: float,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate position size using the position sizer."""
        # Create contexts for position sizing
        market_context = self._create_market_context(signal, market_data or {})
        signal_context = self._create_signal_context(signal)
        portfolio_context = self._create_portfolio_context()

        # Get sizing parameters
        sizing_params = self._get_sizing_parameters(signal)

        # Calculate size
        size = self.position_sizer.calculate_position_size(
            market_context, signal_context, portfolio_context, sizing_params
        )

        sizing_info = {
            "market_context": market_context.to_dict(),
            "signal_context": signal_context.to_dict(),
            "portfolio_context": portfolio_context.to_dict(),
            "sizing_params": sizing_params.to_dict() if sizing_params else None,
        }

        return size, sizing_info

    def _create_market_context(
        self, signal: TradeSignal, market_data: Dict[str, Any]
    ) -> MarketContext:
        """Create market context for position sizing."""
        return MarketContext(
            symbol=signal.symbol,
            current_price=signal.entry_price,
            volatility=market_data.get(f"{signal.symbol}_volatility", 0.0),
            volume=market_data.get(f"{signal.symbol}_volume", 0),
            market_cap=market_data.get(f"{signal.symbol}_market_cap", 0),
            sector=market_data.get(f"{signal.symbol}_sector", "unknown"),
        )

    def _create_signal_context(self, signal: TradeSignal) -> SignalContext:
        """Create signal context for position sizing."""
        return SignalContext(
            strategy=signal.strategy,
            confidence=signal.confidence,
            direction=signal.direction,
            entry_price=signal.entry_price,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
        )

    def _create_portfolio_context(self) -> PortfolioContext:
        """Create portfolio context for position sizing."""
        positions = self.position_manager.get_all_positions()
        total_value = sum(pos.entry_price * pos.size for pos in positions)

        return PortfolioContext(
            total_value=total_value,
            cash_available=self.portfolio_manager.get_cash_balance(),
            position_count=len(positions),
            current_risk=self.position_manager.calculate_portfolio_risk_metrics({}),
        )

    def _get_sizing_parameters(self, signal: TradeSignal) -> Optional[SizingParameters]:
        """Get sizing parameters for the signal."""
        strategy_performance = self._get_strategy_performance(signal.strategy)

        return SizingParameters(
            strategy=SizingStrategy.KELLY_CRITERION,
            risk_per_trade=self.risk_controls.max_position_size,
            max_position_size=self.risk_controls.max_position_size,
            strategy_performance=strategy_performance,
        )

    def _get_strategy_performance(self, strategy_name: str) -> Dict[str, float]:
        """Get performance metrics for a strategy."""
        # This would typically query a strategy performance database
        # For now, return default metrics
        return {
            "win_rate": 0.6,
            "avg_win": 0.05,
            "avg_loss": 0.03,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15,
        }

    def _calculate_fees(self, signal: TradeSignal, execution_price: float) -> float:
        """Calculate trading fees."""
        # Simple fee calculation - can be enhanced based on provider
        trade_value = execution_price * (signal.size or 1.0)
        return trade_value * 0.001  # 0.1% fee

    def _update_market_data_cache(self, market_data: Dict[str, Any]) -> None:
        """Update market data cache."""
        self.market_data_cache.update(market_data)

        # Keep cache size manageable
        if len(self.market_data_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.market_data_cache.keys())[:100]
            for key in keys_to_remove:
                del self.market_data_cache[key]

    def _log_execution_result(self, result: ExecutionResult) -> None:
        """Log execution result."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "success": result.success,
            "symbol": result.signal.symbol,
            "direction": result.signal.direction.value,
            "execution_price": result.execution_price,
            "fees": result.fees,
            "message": result.message,
            "error": result.error,
        }

        self.logger.info(f"Execution result: {log_entry}")

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return self.position_manager.get_risk_summary()

    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history[-limit:]

    def get_trade_log(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get trade log within a date range."""
        trades = self.trade_log

        if start_date:
            trades = [
                t
                for t in trades
                if datetime.fromisoformat(t["timestamp"]) >= start_date
            ]
        if end_date:
            trades = [
                t for t in trades if datetime.fromisoformat(t["timestamp"]) <= end_date
            ]

        return trades

    def clear_trade_log(self) -> None:
        """Clear the trade log."""
        self.trade_log = []
        self._save_trade_log()


def create_execution_agent(config: Optional[Dict[str, Any]] = None) -> ExecutionAgent:
    """Factory function to create an execution agent."""
    if config is None:
        config = {}

    agent_config = AgentConfig(
        name="ExecutionAgent", agent_type="execution", config=config
    )

    return ExecutionAgent(agent_config)
