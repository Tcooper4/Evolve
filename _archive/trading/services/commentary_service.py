"""
Commentary Service

This service provides a high-level interface for generating commentary
throughout the trading system. It integrates with the Commentary Engine
and provides convenient methods for different types of commentary.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from trading.agents.commentary_agent import create_commentary_agent
from trading.commentary import (
    CommentaryRequest,
    CommentaryType,
    create_commentary_engine,
)
from trading.memory.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class CommentaryService:
    """
    High-level commentary service for the trading system.

    This service provides convenient methods for generating different types
    of commentary and integrates with the agent system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the commentary service."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.commentary_engine = create_commentary_engine(
            self.config.get("engine_config", {})
        )
        self.commentary_agent = create_commentary_agent(
            self.config.get("agent_config", {})
        )
        self.memory = AgentMemory()

        # Service statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "commentary_types": {},
        }

        self.logger.info("CommentaryService initialized successfully")

    async def explain_trade(
        self,
        symbol: str,
        trade_data: Dict[str, Any],
        market_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Explain a trading decision.

        Args:
            symbol: Trading symbol
            trade_data: Trade details
            market_data: Market data (optional)

        Returns:
            Dictionary with trade explanation
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["commentary_types"]["trade"] = (
                self.stats["commentary_types"].get("trade", 0) + 1
            )

            request = CommentaryRequest(
                commentary_type=CommentaryType.TRADE_EXPLANATION,
                symbol=symbol,
                timestamp=datetime.now(),
                trade_data=trade_data,
                market_data=market_data,
            )

            response = await self.commentary_engine.generate_commentary(request)

            self.stats["successful_requests"] += 1

            return {
                "success": True,
                "commentary": response,
                "type": "trade_explanation",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Error explaining trade: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "trade_explanation",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

    async def analyze_performance(
        self, symbol: str, performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze trading performance.

        Args:
            symbol: Trading symbol
            performance_data: Performance metrics

        Returns:
            Dictionary with performance analysis
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["commentary_types"]["performance"] = (
                self.stats["commentary_types"].get("performance", 0) + 1
            )

            request = CommentaryRequest(
                commentary_type=CommentaryType.PERFORMANCE_ANALYSIS,
                symbol=symbol,
                timestamp=datetime.now(),
                performance_data=performance_data,
            )

            response = await self.commentary_engine.generate_commentary(request)

            self.stats["successful_requests"] += 1

            return {
                "success": True,
                "commentary": response,
                "type": "performance_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "performance_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

    async def assess_risk(
        self,
        symbol: str,
        trade_data: Optional[Dict[str, Any]] = None,
        portfolio_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assess trading risks.

        Args:
            symbol: Trading symbol
            trade_data: Trade details (optional)
            portfolio_data: Portfolio data (optional)

        Returns:
            Dictionary with risk assessment
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["commentary_types"]["risk"] = (
                self.stats["commentary_types"].get("risk", 0) + 1
            )

            request = CommentaryRequest(
                commentary_type=CommentaryType.RISK_ASSESSMENT,
                symbol=symbol,
                timestamp=datetime.now(),
                trade_data=trade_data,
                portfolio_data=portfolio_data,
            )

            response = await self.commentary_engine.generate_commentary(request)

            self.stats["successful_requests"] += 1

            return {
                "success": True,
                "commentary": response,
                "type": "risk_assessment",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Error assessing risk: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "risk_assessment",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

    async def generate_daily_summary(
        self,
        portfolio_data: Dict[str, Any],
        trades: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate daily trading summary.

        Args:
            portfolio_data: Portfolio state
            trades: List of trades for the day
            market_data: Market data (optional)

        Returns:
            Dictionary with daily summary
        """
        try:
            self.stats["total_requests"] += 1
            self.stats["commentary_types"]["daily"] = (
                self.stats["commentary_types"].get("daily", 0) + 1
            )

            request = CommentaryRequest(
                commentary_type=CommentaryType.DAILY_SUMMARY,
                symbol="PORTFOLIO",
                timestamp=datetime.now(),
                portfolio_data=portfolio_data,
                context={"trades": trades, "market_data": market_data},
            )

            response = await self.commentary_engine.generate_commentary(request)

            self.stats["successful_requests"] += 1

            return {
                "success": True,
                "commentary": response,
                "type": "daily_summary",
                "symbol": "PORTFOLIO",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Error generating daily summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "type": "daily_summary",
                "symbol": "PORTFOLIO",
                "timestamp": datetime.now().isoformat(),
            }

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        engine_stats = self.commentary_engine.get_commentary_statistics()

        return {
            "service_stats": self.stats,
            "engine_stats": engine_stats,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
            )
            * 100,
            "total_commentaries": engine_stats.get("total_commentaries", 0),
        }

    def get_commentary_history(
        self, symbol: Optional[str] = None, commentary_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get commentary history."""
        try:
            # This would typically query the memory or database
            # For now, return engine statistics
            engine_stats = self.commentary_engine.get_commentary_statistics()
            return engine_stats.get("recent_commentaries", [])
        except Exception as e:
            self.logger.error(f"Error getting commentary history: {str(e)}")
            return []


# Convenience function for creating commentary service


def create_commentary_service(
    config: Optional[Dict[str, Any]] = None,
) -> CommentaryService:
    """Create a configured commentary service."""
    return CommentaryService(config)
