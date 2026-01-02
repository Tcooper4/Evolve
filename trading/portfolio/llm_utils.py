"""LLM utilities for trade rationale and commentary generation."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import openai
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler for debug logs
debug_handler = logging.FileHandler("trading/portfolio/logs/portfolio_debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
debug_handler.setFormatter(debug_formatter)
logger.addHandler(debug_handler)


class TradeRationale(BaseModel):
    """Trade rationale data model."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    direction: str
    strategy: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    market_context: Dict[str, Any]
    risk_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "strategy": self.strategy,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "market_context": self.market_context,
            "risk_factors": self.risk_factors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRationale":
        """Create from dictionary."""
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class DailyCommentary(BaseModel):
    """Daily commentary data model."""

    date: datetime = Field(default_factory=datetime.utcnow)
    summary: str
    trades: List[Dict[str, Any]]
    pnl_summary: Dict[str, float]
    strategy_shifts: List[Dict[str, Any]]
    market_conditions: Dict[str, Any]
    risk_assessment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "summary": self.summary,
            "trades": self.trades,
            "pnl_summary": self.pnl_summary,
            "strategy_shifts": self.strategy_shifts,
            "market_conditions": self.market_conditions,
            "risk_assessment": self.risk_assessment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DailyCommentary":
        """Create from dictionary."""
        if isinstance(data["date"], str):
            data["date"] = datetime.fromisoformat(data["date"])
        return cls(**data)


class LLMInterface:
    """Interface for LLM-based trade rationale and commentary generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM interface.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logger.warning("OpenAI API key not found. LLM features will be disabled.")
            self.enabled = False
        else:
            self.enabled = True

        # Create necessary directories
        try:
            os.makedirs("trading/portfolio/logs", exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create trading/portfolio/logs: {e}")
        try:
            os.makedirs("trading/portfolio/data", exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create trading/portfolio/data: {e}")

        logger.info("Initialized LLMInterface")

    def generate_trade_rationale(
        self, symbol: str, direction: str, strategy: str, market_data: Dict[str, Any]
    ) -> Optional[TradeRationale]:
        """Generate trade rationale using LLM.

        Args:
            symbol: Trading symbol
            direction: Trade direction
            strategy: Strategy name
            market_data: Market data and context

        Returns:
            TradeRationale object or None if LLM is disabled
        """
        if not self.enabled:
            return None

        try:
            # Prepare prompt
            prompt = self._create_trade_rationale_prompt(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                market_data=market_data,
            )

            # Call LLM
            response = openai.ChatCompletion.create(
                model=self.config.get("model", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading assistant that provides clear, concise rationales for trading decisions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            # Parse response
            rationale_text = response.choices[0].message.content

            # Extract confidence and risk factors
            confidence = self._extract_confidence(rationale_text)
            risk_factors = self._extract_risk_factors(rationale_text)

            # Create rationale object
            rationale = TradeRationale(
                symbol=symbol,
                direction=direction,
                strategy=strategy,
                rationale=rationale_text,
                confidence=confidence,
                market_context=market_data,
                risk_factors=risk_factors,
            )

            # Log rationale
            self._log_rationale(rationale)

            return rationale

        except Exception as e:
            logger.error(f"Error generating trade rationale: {e}")

    def generate_daily_commentary(
        self,
        portfolio_state: Dict[str, Any],
        trades: List[Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> Optional[DailyCommentary]:
        """Generate daily commentary using LLM.

        Args:
            portfolio_state: Current portfolio state
            trades: List of trades for the day
            market_data: Market data and context

        Returns:
            DailyCommentary object or None if LLM is disabled
        """
        if not self.enabled:
            return None

        try:
            # Prepare prompt
            prompt = self._create_daily_commentary_prompt(
                portfolio_state=portfolio_state, trades=trades, market_data=market_data
            )

            # Call LLM
            response = openai.ChatCompletion.create(
                model=self.config.get("model", "gpt-4"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading assistant that provides daily summaries of trading activity and market conditions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            # Parse response
            commentary_text = response.choices[0].message.content

            # Create commentary object
            commentary = DailyCommentary(
                summary=commentary_text,
                trades=trades,
                pnl_summary=self._calculate_pnl_summary(trades),
                strategy_shifts=self._identify_strategy_shifts(trades),
                market_conditions=market_data,
                risk_assessment=self._assess_risk(portfolio_state, market_data),
            )

            # Log commentary
            self._log_commentary(commentary)

            return commentary

        except Exception as e:
            logger.error(f"Error generating daily commentary: {e}")

    def _create_trade_rationale_prompt(
        self, symbol: str, direction: str, strategy: str, market_data: Dict[str, Any]
    ) -> str:
        """Create prompt for trade rationale generation.

        Args:
            symbol: Trading symbol
            direction: Trade direction
            strategy: Strategy name
            market_data: Market data and context

        Returns:
            Prompt string
        """
        return f"""Generate a trade rationale for the following trade:
Symbol: {symbol}
Direction: {direction}
Strategy: {strategy}

Market Context:
{json.dumps(market_data, indent=2)}

Please provide:
1. Clear rationale for the trade
2. Confidence level (0-1)
3. Key risk factors to monitor
4. Market conditions supporting the trade
5. Potential exit scenarios

Format the response as a structured analysis."""

    def _create_daily_commentary_prompt(
        self,
        portfolio_state: Dict[str, Any],
        trades: List[Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> str:
        """Create prompt for daily commentary generation.

        Args:
            portfolio_state: Current portfolio state
            trades: List of trades for the day
            market_data: Market data and context

        Returns:
            Prompt string
        """
        return f"""Generate a daily trading commentary for:
Date: {datetime.utcnow().date()}

Portfolio State:
{json.dumps(portfolio_state, indent=2)}

Trades:
{json.dumps(trades, indent=2)}

Market Data:
{json.dumps(market_data, indent=2)}

Please provide:
1. Summary of trading activity
2. Analysis of PnL and performance
3. Notable strategy shifts or changes
4. Market conditions and their impact
5. Risk assessment and recommendations

Format the response as a comprehensive daily report."""

    def _extract_confidence(self, rationale: str) -> float:
        """Extract confidence from rationale text.

        Args:
            rationale: Rationale text

        Returns:
            Confidence value between 0 and 1
        """
        try:
            # Look for confidence in text
            if "confidence" in rationale.lower():
                # Extract number after "confidence"
                confidence_text = rationale.lower().split("confidence")[1]
                confidence = float(confidence_text.split()[0])
                return {
                    "success": True,
                    "result": max(0.0, min(1.0, confidence)),
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat(),
                }
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not extract confidence from rationale: {e}")

        # Default to moderate confidence
        return 0.5

    def _extract_risk_factors(self, rationale: str) -> List[str]:
        """Extract risk factors from rationale text.

        Args:
            rationale: Rationale text

        Returns:
            List of risk factors
        """
        risk_factors = []

        # Look for risk factors in text
        if "risk factors" in rationale.lower():
            # Extract list after "risk factors"
            risk_text = rationale.lower().split("risk factors")[1]
            # Split by newlines or numbers
            factors = [f.strip() for f in risk_text.split("\n") if f.strip()]
            risk_factors.extend(factors)

        return {
            "success": True,
            "result": risk_factors
            or ["Market volatility", "Liquidity risk", "Execution risk"],
            "message": "Operation completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_pnl_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate PnL summary from trades.

        Args:
            trades: List of trades

        Returns:
            Dictionary with PnL summary
        """
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        strategy_pnl = {}

        for trade in trades:
            strategy = trade.get("strategy", "unknown")
            strategy_pnl[strategy] = strategy_pnl.get(strategy, 0) + trade.get("pnl", 0)

        return {"total_pnl": total_pnl, "strategy_pnl": strategy_pnl}

    def _identify_strategy_shifts(
        self, trades: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify strategy shifts from trades.

        Args:
            trades: List of trades

        Returns:
            List of strategy shifts
        """
        shifts = []

        # Group trades by strategy
        strategy_trades = {}
        for trade in trades:
            strategy = trade.get("strategy", "unknown")
            if strategy not in strategy_trades:
                strategy_trades[strategy] = []
            strategy_trades[strategy].append(trade)

        # Analyze strategy performance
        for strategy, trades in strategy_trades.items():
            if not trades:
                continue
                
            pnl = sum(t.get("pnl", 0) for t in trades)
            num_trades = len(trades)
            
            if num_trades > 0:
                win_rate = sum(1 for t in trades if t.get("pnl", 0) > 0) / num_trades
            else:
                continue

            if pnl > 0 and win_rate > 0.6:
                shifts.append(
                    {
                        "strategy": strategy,
                        "type": "increased",
                        "reason": "Strong performance",
                        "metrics": {"pnl": pnl, "win_rate": win_rate},
                    }
                )
            elif pnl < 0 and win_rate < 0.4:
                shifts.append(
                    {
                        "strategy": strategy,
                        "type": "decreased",
                        "reason": "Poor performance",
                        "metrics": {"pnl": pnl, "win_rate": win_rate},
                    }
                )

        return shifts

    def _assess_risk(
        self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess portfolio risk.

        Args:
            portfolio_state: Current portfolio state
            market_data: Market data and context

        Returns:
            Dictionary with risk assessment
        """
        return {
            "portfolio_risk": {
                "var_95": self._calculate_var(portfolio_state, 0.95),
                "var_99": self._calculate_var(portfolio_state, 0.99),
                "volatility": self._calculate_volatility(portfolio_state),
                "correlation": self._calculate_correlation(portfolio_state),
            },
            "market_risk": {
                "volatility": market_data.get("volatility", 0),
                "trend": market_data.get("trend", "neutral"),
                "liquidity": market_data.get("liquidity", "normal"),
            },
        }

    def _calculate_var(
        self, portfolio_state: Dict[str, Any], confidence: float
    ) -> float:
        """Calculate Value at Risk.

        Args:
            portfolio_state: Current portfolio state
            confidence: Confidence level

        Returns:
            VaR value
        """
        # Simple VaR calculation
        returns = [p.get("pnl", 0) for p in portfolio_state.get("closed_positions", [])]
        if not returns:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_volatility(self, portfolio_state: Dict[str, Any]) -> float:
        """Calculate portfolio volatility.

        Args:
            portfolio_state: Current portfolio state

        Returns:
            Volatility value
        """
        returns = [p.get("pnl", 0) for p in portfolio_state.get("closed_positions", [])]
        if not returns:
            return 0.0

        return np.std(returns) * np.sqrt(252)

    def _calculate_correlation(
        self, portfolio_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate position correlations.

        Args:
            portfolio_state: Current portfolio state

        Returns:
            Dictionary of correlations
        """
        positions = portfolio_state.get("open_positions", [])
        if len(positions) < 2:
            return {}

        # Calculate returns for each position
        returns = {}
        for pos in positions:
            symbol = pos.get("symbol")
            if symbol:
                returns[symbol] = pos.get("pnl", 0)

        # Calculate correlations
        correlations = {}
        symbols = list(returns.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                correlations[f"{sym1}-{sym2}"] = np.corrcoef(
                    [returns[sym1]], [returns[sym2]]
                )[0, 1]

        return correlations

    def _log_rationale(self, rationale: TradeRationale) -> None:
        """Log trade rationale.

        Args:
            rationale: Trade rationale to log
        """
        # Save to JSON
        log_path = f"trading/portfolio/logs/trade_rationales_{datetime.utcnow().strftime('%Y%m%d')}.json"
        with open(log_path, "a") as f:
            f.write(json.dumps(rationale.to_dict()) + "\n")

        logger.info(f"Logged trade rationale for {rationale.symbol}")

    def _log_commentary(self, commentary: DailyCommentary) -> None:
        """Log daily commentary.

        Args:
            commentary: Daily commentary to log
        """
        # Save to JSON
        log_path = f"trading/portfolio/logs/daily_commentary_{commentary.date.strftime('%Y%m%d')}.json"
        with open(log_path, "w") as f:
            json.dump(commentary.to_dict(), f, indent=2)

        logger.info(f"Logged daily commentary for {commentary.date.date()}")


__all__ = ["LLMInterface", "TradeRationale", "DailyCommentary"]
