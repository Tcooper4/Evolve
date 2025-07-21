"""Execution Risk Control Agent.

This agent enforces trade constraints including max exposure per asset,
pause trading on major losses, and cooling periods for risk management.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_agent_interface import AgentConfig, AgentResult, BaseAgent

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradeStatus(Enum):
    """Trade status."""

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    PENDING = "pending"


@dataclass
class RiskCheck:
    """Risk check result."""

    check_name: str
    passed: bool
    risk_level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime


@dataclass
class TradeApproval:
    """Trade approval result."""

    trade_id: str
    symbol: str
    status: TradeStatus
    original_size: float
    approved_size: float
    risk_checks: List[RiskCheck]
    warnings: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class RiskAssessmentRequest:
    """Request for risk assessment operations."""

    operation_type: str  # 'approve_trade', 'get_risk_summary', 'assess_portfolio'
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    size: Optional[float] = None
    side: Optional[str] = None
    price: Optional[float] = None
    portfolio_context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()


@dataclass
class RiskAssessmentResult:
    """Result of risk assessment operations."""

    success: bool
    operation_type: str
    trade_approval: Optional[TradeApproval] = None
    risk_summary: Optional[Dict[str, Any]] = None
    risk_checks: Optional[List[RiskCheck]] = None
    error_message: Optional[str] = None
    timestamp: datetime = datetime.now()


class ExecutionRiskAgent(BaseAgent):
    """Execution risk control agent for trade approval and risk management."""

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ExecutionRiskAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={},
            )
        super().__init__(config)
        self.config_dict = config.custom_config or {}
        # Risk limits
        self.risk_limits = self.config_dict.get(
            "risk_limits",
            {
                "max_position_size": 0.2,
                "max_sector_exposure": 0.3,
                "max_portfolio_risk": 0.15,
                "max_daily_loss": 0.05,
                "max_drawdown": 0.10,
                "max_leverage": 2.0,
                "min_liquidity": 1000000,
            },
        )
        # Cooling periods
        self.cooling_periods = self.config_dict.get(
            "cooling_periods",
            {
                "major_loss_hours": 24,
                "high_volatility_hours": 4,
                "consecutive_losses_hours": 12,
                "market_close_hours": 2,
            },
        )
        # Risk thresholds
        self.risk_thresholds = self.config_dict.get(
            "risk_thresholds",
            {
                "volatility_threshold": 0.03,
                "correlation_threshold": 0.7,
                "concentration_threshold": 0.4,
                "liquidity_threshold": 0.1,
            },
        )
        # State tracking
        self.trade_history = []
        self.risk_violations = []
        self.cooling_periods_active = {}
        self.portfolio_state = {
            "current_positions": {},
            "sector_exposures": {},
            "total_exposure": 0.0,
            "daily_pnl": 0.0,
            "current_drawdown": 0.0,
        }
        logger.info("Execution Risk Agent initialized")

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the risk agent logic.
        Args:
            **kwargs: action, trade_id, symbol, size, side, price, portfolio_context, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get("action", "approve_trade")
            if action == "approve_trade":
                trade_id = kwargs.get("trade_id")
                symbol = kwargs.get("symbol")
                size = kwargs.get("size")
                side = kwargs.get("side")
                price = kwargs.get("price")
                portfolio_context = kwargs.get("portfolio_context")
                approval = self.approve_trade(
                    trade_id, symbol, size, side, price, portfolio_context
                )
                return AgentResult(
                    success=True,
                    data={
                        "approval": {
                            "trade_id": approval.trade_id,
                            "symbol": approval.symbol,
                            "status": approval.status.value,
                            "original_size": approval.original_size,
                            "approved_size": approval.approved_size,
                            "warnings": approval.warnings,
                            "timestamp": approval.timestamp.isoformat(),
                            "metadata": approval.metadata,
                        }
                    },
                )
            elif action == "get_risk_summary":
                summary = self.get_risk_summary()
                return AgentResult(success=True, data={"risk_summary": summary})
            else:
                return AgentResult(
                    success=False, error_message=f"Unknown action: {action}"
                )
        except Exception as e:
            return self.handle_error(e)

    def approve_trade(
        self,
        trade_id: str,
        symbol: str,
        size: float,
        side: str,
        price: float,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> TradeApproval:
        """Approve or reject a trade based on risk checks.

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            size: Trade size
            side: Trade side (buy/sell)
            price: Trade price
            portfolio_context: Portfolio context information

        Returns:
            Trade approval result
        """
        try:
            # Initialize risk checks
            risk_checks = []
            warnings = []

            # Check if in cooling period
            if self._is_in_cooling_period(symbol):
                return TradeApproval(
                    trade_id=trade_id,
                    symbol=symbol,
                    status=TradeStatus.REJECTED,
                    original_size=size,
                    approved_size=0.0,
                    risk_checks=risk_checks,
                    warnings=["Symbol in cooling period"],
                    timestamp=datetime.now(),
                    metadata={"reason": "cooling_period"},
                )

            # Perform risk checks
            checks = [
                self._check_position_size(symbol, size, portfolio_context),
                self._check_sector_exposure(symbol, size, portfolio_context),
                self._check_portfolio_risk(size, portfolio_context),
                self._check_daily_loss_limit(portfolio_context),
                self._check_drawdown_limit(portfolio_context),
                self._check_leverage_limit(size, portfolio_context),
                self._check_liquidity(symbol, size, portfolio_context),
                self._check_volatility(symbol, portfolio_context),
                self._check_correlation(symbol, portfolio_context),
                self._check_concentration(symbol, size, portfolio_context),
            ]

            risk_checks.extend([check for check in checks if check is not None])

            # Determine approval status
            failed_checks = [check for check in risk_checks if not check.passed]
            critical_failures = [
                check
                for check in failed_checks
                if check.risk_level == RiskLevel.CRITICAL
            ]

            if critical_failures:
                status = TradeStatus.REJECTED
                approved_size = 0.0
                warnings.extend(
                    [
                        f"Critical risk: {check.check_name}"
                        for check in critical_failures
                    ]
                )
            elif failed_checks:
                status = TradeStatus.MODIFIED
                approved_size = self._calculate_modified_size(size, failed_checks)
                warnings.extend(
                    [f"Risk limit: {check.check_name}" for check in failed_checks]
                )
            else:
                status = TradeStatus.APPROVED
                approved_size = size

            # Create approval result
            approval = TradeApproval(
                trade_id=trade_id,
                symbol=symbol,
                status=status,
                original_size=size,
                approved_size=approved_size,
                risk_checks=risk_checks,
                warnings=warnings,
                timestamp=datetime.now(),
                metadata={
                    "side": side,
                    "price": price,
                    "risk_score": self._calculate_risk_score(risk_checks),
                },
            )

            # Update state
            self.trade_history.append(approval)
            if status == TradeStatus.APPROVED:
                self._update_portfolio_state(symbol, approved_size, side, price)

            logger.info(
                f"Trade {trade_id} {status.value}: {approved_size:.2%} of {size:.2%} approved"
            )

            return approval

        except Exception as e:
            logger.error(f"Error approving trade {trade_id}: {e}")
            return TradeApproval(
                trade_id=trade_id,
                symbol=symbol,
                status=TradeStatus.REJECTED,
                original_size=size,
                approved_size=0.0,
                risk_checks=[],
                warnings=[f"System error: {str(e)}"],
                timestamp=datetime.now(),
                metadata={"error": str(e)},
            )

    def _is_in_cooling_period(self, symbol: str) -> bool:
        """Check if symbol is in cooling period.

        Args:
            symbol: Trading symbol

        Returns:
            True if in cooling period
        """
        if symbol not in self.cooling_periods_active:
            return False

        cooling_info = self.cooling_periods_active[symbol]
        end_time = cooling_info["end_time"]

        if datetime.now() > end_time:
            # Remove expired cooling period
            del self.cooling_periods_active[symbol]
            return True

        return True

    def _check_position_size(
        self, symbol: str, size: float, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check position size limit.

        Args:
            symbol: Trading symbol
            size: Trade size
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            current_position = self.portfolio_state["current_positions"].get(symbol, 0)
            new_position = current_position + size

            threshold = self.risk_limits["max_position_size"]
            passed = new_position <= threshold

            risk_level = (
                RiskLevel.CRITICAL if new_position > threshold * 1.5 else RiskLevel.HIGH
            )

            return RiskCheck(
                check_name="Position Size Limit",
                passed=passed,
                risk_level=risk_level,
                message=f"Position size {new_position:.2%} vs limit {threshold:.2%}",
                value=new_position,
                threshold=threshold,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error checking position size: {e}")
            return True

    def _check_sector_exposure(
        self, symbol: str, size: float, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check sector exposure limit.

        Args:
            symbol: Trading symbol
            size: Trade size
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            if portfolio_context and "sector" in portfolio_context:
                sector = portfolio_context["sector"]
                current_exposure = self.portfolio_state["sector_exposures"].get(
                    sector, 0
                )
                new_exposure = current_exposure + size

                threshold = self.risk_limits["max_sector_exposure"]
                passed = new_exposure <= threshold

                risk_level = (
                    RiskLevel.HIGH if new_exposure > threshold else RiskLevel.MEDIUM
                )

                return RiskCheck(
                    check_name="Sector Exposure Limit",
                    passed=passed,
                    risk_level=risk_level,
                    message=f"Sector {sector} exposure {new_exposure:.2%} vs limit {threshold:.2%}",
                    value=new_exposure,
                    threshold=threshold,
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.error(f"Error checking sector exposure: {e}")
            return True

    def _check_portfolio_risk(
        self, size: float, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check portfolio risk limit.

        Args:
            size: Trade size
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            current_risk = self.portfolio_state["total_exposure"]
            new_risk = current_risk + size

            threshold = self.risk_limits["max_portfolio_risk"]
            passed = new_risk <= threshold

            risk_level = (
                RiskLevel.CRITICAL if new_risk > threshold * 1.2 else RiskLevel.HIGH
            )

            return RiskCheck(
                check_name="Portfolio Risk Limit",
                passed=passed,
                risk_level=risk_level,
                message=f"Portfolio risk {new_risk:.2%} vs limit {threshold:.2%}",
                value=new_risk,
                threshold=threshold,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return RiskCheck(
                check_name="Portfolio Risk Limit",
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                message=f"Error checking portfolio risk: {e}",
                value=0.0,
                threshold=0.0,
                timestamp=datetime.now(),
            )

    def _check_daily_loss_limit(
        self, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check daily loss limit.

        Args:
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            daily_loss = abs(min(self.portfolio_state["daily_pnl"], 0))
            threshold = self.risk_limits["max_daily_loss"]
            passed = daily_loss <= threshold

            risk_level = (
                RiskLevel.CRITICAL if daily_loss > threshold else RiskLevel.HIGH
            )

            return RiskCheck(
                check_name="Daily Loss Limit",
                passed=passed,
                risk_level=risk_level,
                message=f"Daily loss {daily_loss:.2%} vs limit {threshold:.2%}",
                value=daily_loss,
                threshold=threshold,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error checking daily loss limit: {e}")
            return RiskCheck(
                check_name="Daily Loss Limit",
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                message=f"Error checking daily loss limit: {e}",
                value=0.0,
                threshold=0.0,
                timestamp=datetime.now(),
            )

    def _check_drawdown_limit(
        self, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check drawdown limit.

        Args:
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            current_drawdown = self.portfolio_state["current_drawdown"]
            threshold = self.risk_limits["max_drawdown"]
            passed = current_drawdown <= threshold

            risk_level = (
                RiskLevel.CRITICAL if current_drawdown > threshold else RiskLevel.HIGH
            )

            return RiskCheck(
                check_name="Drawdown Limit",
                passed=passed,
                risk_level=risk_level,
                message=f"Current drawdown {current_drawdown:.2%} vs limit {threshold:.2%}",
                value=current_drawdown,
                threshold=threshold,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error checking drawdown limit: {e}")
            return RiskCheck(
                check_name="Drawdown Limit",
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                message=f"Error checking drawdown limit: {e}",
                value=0.0,
                threshold=0.0,
                timestamp=datetime.now(),
            )

    def _check_leverage_limit(
        self, size: float, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check leverage limit.

        Args:
            size: Trade size
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            # Simplified leverage calculation
            total_exposure = self.portfolio_state["total_exposure"] + size
            leverage = total_exposure / max(
                1.0, total_exposure - size
            )  # Avoid division by zero

            threshold = self.risk_limits["max_leverage"]
            passed = leverage <= threshold

            risk_level = RiskLevel.HIGH if leverage > threshold else RiskLevel.MEDIUM

            return RiskCheck(
                check_name="Leverage Limit",
                passed=passed,
                risk_level=risk_level,
                message=f"Leverage {leverage:.2f}x vs limit {threshold:.2f}x",
                value=leverage,
                threshold=threshold,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error checking leverage limit: {e}")
            return True

    def _check_liquidity(
        self, symbol: str, size: float, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check liquidity requirements.

        Args:
            symbol: Trading symbol
            size: Trade size
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            if portfolio_context and "liquidity" in portfolio_context:
                liquidity = portfolio_context["liquidity"].get(symbol, 0)
                threshold = self.risk_limits["min_liquidity"]
                passed = liquidity >= threshold

                risk_level = (
                    RiskLevel.MEDIUM if liquidity < threshold else RiskLevel.LOW
                )

                return RiskCheck(
                    check_name="Liquidity Requirement",
                    passed=passed,
                    risk_level=risk_level,
                    message=f"Liquidity ${liquidity:,.0f} vs minimum ${threshold:,.0f}",
                    value=liquidity,
                    threshold=threshold,
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.error(f"Error checking liquidity: {e}")
            return True

    def _check_volatility(
        self, symbol: str, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check volatility threshold.

        Args:
            symbol: Trading symbol
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            if portfolio_context and "volatility" in portfolio_context:
                volatility = portfolio_context["volatility"].get(symbol, 0)
                threshold = self.risk_thresholds["volatility_threshold"]
                passed = volatility <= threshold

                risk_level = (
                    RiskLevel.HIGH if volatility > threshold * 1.5 else RiskLevel.MEDIUM
                )

                return RiskCheck(
                    check_name="Volatility Threshold",
                    passed=passed,
                    risk_level=risk_level,
                    message=f"Volatility {volatility:.2%} vs threshold {threshold:.2%}",
                    value=volatility,
                    threshold=threshold,
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.error(f"Error checking volatility: {e}")
            return True

    def _check_correlation(
        self, symbol: str, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check correlation threshold.

        Args:
            symbol: Trading symbol
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            if portfolio_context and "correlation" in portfolio_context:
                correlation = portfolio_context["correlation"].get(symbol, 0)
                threshold = self.risk_thresholds["correlation_threshold"]
                passed = abs(correlation) <= threshold

                risk_level = (
                    RiskLevel.MEDIUM if abs(correlation) > threshold else RiskLevel.LOW
                )

                return RiskCheck(
                    check_name="Correlation Threshold",
                    passed=passed,
                    risk_level=risk_level,
                    message=f"Correlation {correlation:.2f} vs threshold {threshold:.2f}",
                    value=abs(correlation),
                    threshold=threshold,
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.error(f"Error checking correlation: {e}")
            return True

    def _check_concentration(
        self, symbol: str, size: float, portfolio_context: Optional[Dict[str, Any]]
    ) -> Optional[RiskCheck]:
        """Check concentration threshold.

        Args:
            symbol: Trading symbol
            size: Trade size
            portfolio_context: Portfolio context

        Returns:
            Risk check result
        """
        try:
            current_position = self.portfolio_state["current_positions"].get(symbol, 0)
            new_position = current_position + size
            total_exposure = self.portfolio_state["total_exposure"] + size

            if total_exposure > 0:
                concentration = new_position / total_exposure
                threshold = self.risk_thresholds["concentration_threshold"]
                passed = concentration <= threshold

                risk_level = (
                    RiskLevel.HIGH if concentration > threshold else RiskLevel.MEDIUM
                )

                return RiskCheck(
                    check_name="Concentration Threshold",
                    passed=passed,
                    risk_level=risk_level,
                    message=f"Concentration {concentration:.2%} vs threshold {threshold:.2%}",
                    value=concentration,
                    threshold=threshold,
                    timestamp=datetime.now(),
                )

            return None

        except Exception as e:
            logger.error(f"Error checking concentration: {e}")
            return True

    def _calculate_modified_size(
        self, original_size: float, failed_checks: List[RiskCheck]
    ) -> float:
        """Calculate modified trade size based on failed checks.

        Args:
            original_size: Original trade size
            failed_checks: List of failed risk checks

        Returns:
            Modified trade size
        """
        try:
            # Find the most restrictive constraint
            min_size = original_size

            for check in failed_checks:
                if check.check_name == "Position Size Limit":
                    min_size = min(min_size, check.threshold)
                elif check.check_name == "Sector Exposure Limit":
                    min_size = min(min_size, check.threshold)
                elif check.check_name == "Portfolio Risk Limit":
                    min_size = min(min_size, check.threshold)
                elif check.check_name == "Leverage Limit":
                    min_size = min(min_size, original_size * 0.8)  # Reduce by 20%

            return max(min_size, 0.01)  # Minimum 1%

        except Exception as e:
            logger.error(f"Error calculating modified size: {e}")
            return original_size * 0.5  # Default to 50%

    def _calculate_risk_score(self, risk_checks: List[RiskCheck]) -> float:
        """Calculate overall risk score.

        Args:
            risk_checks: List of risk checks

        Returns:
            Risk score (0-1)
        """
        try:
            if not risk_checks:
                return 0.0

            # Weight risk levels
            risk_weights = {
                RiskLevel.LOW: 0.1,
                RiskLevel.MEDIUM: 0.3,
                RiskLevel.HIGH: 0.6,
                RiskLevel.CRITICAL: 1.0,
            }

            total_score = 0.0
            total_weight = 0.0

            for check in risk_checks:
                weight = risk_weights.get(check.risk_level, 0.5)
                score = 1.0 if not check.passed else 0.0
                total_score += score * weight
                total_weight += weight

            return total_score / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5

    def _update_portfolio_state(
        self, symbol: str, size: float, side: str, price: float
    ):
        """Update portfolio state after trade.

        Args:
            symbol: Trading symbol
            size: Trade size
            side: Trade side
            price: Trade price
        """
        try:
            # Update position
            current_position = self.portfolio_state["current_positions"].get(symbol, 0)
            if side.lower() == "buy":
                self.portfolio_state["current_positions"][symbol] = (
                    current_position + size
                )
            else:
                self.portfolio_state["current_positions"][symbol] = (
                    current_position - size
                )

            # Update total exposure
            total_exposure = sum(
                abs(pos) for pos in self.portfolio_state["current_positions"].values()
            )
            self.portfolio_state["total_exposure"] = total_exposure

        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")

    def trigger_cooling_period(
        self, symbol: str, reason: str, hours: Optional[int] = None
    ):
        """Trigger cooling period for a symbol.

        Args:
            symbol: Trading symbol
            reason: Reason for cooling period
            hours: Cooling period duration (optional)
        """
        try:
            if hours is None:
                hours = self.cooling_periods.get("major_loss_hours", 24)

            end_time = datetime.now() + timedelta(hours=hours)

            self.cooling_periods_active[symbol] = {
                "reason": reason,
                "start_time": datetime.now(),
                "end_time": end_time,
                "duration_hours": hours,
            }

            logger.warning(
                f"Cooling period triggered for {symbol}: {reason} for {hours} hours"
            )

        except Exception as e:
            logger.error(f"Error triggering cooling period: {e}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary.

        Returns:
            Risk summary dictionary
        """
        try:
            recent_trades = [
                t
                for t in self.trade_history
                if t.timestamp > datetime.now() - timedelta(days=1)
            ]

            summary = {
                "total_trades": len(recent_trades),
                "approved_trades": len(
                    [t for t in recent_trades if t.status == TradeStatus.APPROVED]
                ),
                "rejected_trades": len(
                    [t for t in recent_trades if t.status == TradeStatus.REJECTED]
                ),
                "modified_trades": len(
                    [t for t in recent_trades if t.status == TradeStatus.MODIFIED]
                ),
                "active_cooling_periods": len(self.cooling_periods_active),
                "portfolio_exposure": self.portfolio_state["total_exposure"],
                "daily_pnl": self.portfolio_state["daily_pnl"],
                "current_drawdown": self.portfolio_state["current_drawdown"],
                "risk_violations": len(self.risk_violations),
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}

    def export_risk_report(self, filepath: str) -> bool:
        """Export risk report.

        Args:
            filepath: Output file path

        Returns:
            True if export successful
        """
        try:
            report_data = []

            for trade in self.trade_history:
                row = {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "status": trade.status.value,
                    "original_size": trade.original_size,
                    "approved_size": trade.approved_size,
                    "timestamp": trade.timestamp,
                    "risk_score": trade.metadata.get("risk_score", 0),
                    "warnings": "; ".join(trade.warnings),
                }
                report_data.append(row)

            df = pd.DataFrame(report_data)
            df.to_csv(filepath, index=False)

            logger.info(f"Risk report exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting risk report: {e}")
            return False


# Global execution risk agent instance
execution_risk_agent = ExecutionRiskAgent()


def get_execution_risk_agent() -> ExecutionRiskAgent:
    """Get the global execution risk agent instance."""
    return execution_risk_agent
