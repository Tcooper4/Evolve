"""Strategy logging utilities."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.safe_json_saver import safe_save_historical_data

logger = logging.getLogger(__name__)


class StrategyLogger:
    """Strategy logging class for tracking strategy decisions and performance."""

    def __init__(self, log_dir: str = "logs/strategy"):
        """Initialize the strategy logger.

        Args:
            log_dir: Directory to store strategy logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "strategy_decisions.jsonl"
        self.regime_mapping_file = self.log_dir / "strategy_regime_mappings.json"

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_decision(
        self, strategy_name: str, decision: str, confidence: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a strategy decision.

        Args:
            strategy_name: Name of the strategy
            decision: Decision made (buy/sell/hold)
            confidence: Confidence level (0-1)
            metadata: Additional metadata
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "decision": decision,
            "confidence": confidence,
            "metadata": metadata or {},
        }

        # Log to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # Log to console
        self.logger.info(f"Strategy Decision: {json.dumps(log_data)}")

    def save_strategy_regime_mapping(
        self, strategy_name: str, regime: str, performance_metrics: Dict[str, float], market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save strategy performance mapping for a specific market regime.

        Args:
            strategy_name: Name of the strategy
            regime: Market regime (bull, bear, sideways, volatile, etc.)
            performance_metrics: Performance metrics for this regime
            market_conditions: Market conditions when this mapping was created

        Returns:
            Dictionary with save operation status
        """
        try:
            # Load existing mappings
            mappings = self.load_strategy_regime_mappings()

            # Create or update mapping
            if strategy_name not in mappings:
                mappings[strategy_name] = {}

            if regime not in mappings[strategy_name]:
                mappings[strategy_name][regime] = []

            # Add new mapping record
            mapping_record = {
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": performance_metrics,
                "market_conditions": market_conditions,
                "confidence": performance_metrics.get("confidence", 0.5),
                "win_rate": performance_metrics.get("win_rate", 0.0),
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0.0),
                "total_return": performance_metrics.get("total_return", 0.0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0.0),
            }

            mappings[strategy_name][regime].append(mapping_record)

            # Keep only recent mappings (last 50 per regime)
            if len(mappings[strategy_name][regime]) > 50:
                mappings[strategy_name][regime] = mappings[strategy_name][regime][-50:]

            # Save mappings using safe JSON saving
            result = safe_save_historical_data(mappings, self.regime_mapping_file)
            if not result["success"]:
                self.logger.error(f"Failed to save strategy-regime mappings: {result['error']}")
                return {
                    "success": False,
                    "error": f"Failed to save mappings: {result['error']}",
                    "timestamp": datetime.now().isoformat(),
                }

            self.logger.info(f"Strategy-regime mapping saved: {strategy_name} -> {regime}")

            return {
                "success": True,
                "strategy": strategy_name,
                "regime": regime,
                "message": "Strategy-regime mapping saved successfully",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error saving strategy-regime mapping: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def load_strategy_regime_mappings(self) -> Dict[str, Any]:
        """Load strategy-regime mappings from file.

        Returns:
            Dictionary of strategy-regime mappings
        """
        try:
            if self.regime_mapping_file.exists():
                with open(self.regime_mapping_file, "r") as f:
                    return json.load(f)
            else:
                return {}

        except Exception as e:
            self.logger.error(f"Error loading strategy-regime mappings: {e}")
            return {}

    def get_best_strategy_for_regime(self, regime: str, metric: str = "sharpe_ratio") -> Dict[str, Any]:
        """Get the best performing strategy for a specific market regime.

        Args:
            regime: Market regime to analyze
            metric: Performance metric to optimize for

        Returns:
            Dictionary with best strategy information
        """
        try:
            mappings = self.load_strategy_regime_mappings()

            best_strategy = None
            best_performance = float("-inf")
            best_mapping = None

            for strategy_name, regime_mappings in mappings.items():
                if regime in regime_mappings:
                    # Get average performance for this strategy in this regime
                    regime_performance = []

                    for mapping in regime_mappings[regime]:
                        if metric in mapping["performance_metrics"]:
                            regime_performance.append(mapping["performance_metrics"][metric])

                    if regime_performance:
                        avg_performance = sum(regime_performance) / len(regime_performance)

                        if avg_performance > best_performance:
                            best_performance = avg_performance
                            best_strategy = strategy_name
                            best_mapping = regime_mappings[regime][-1]  # Most recent

            if best_strategy:
                return {
                    "success": True,
                    "regime": regime,
                    "best_strategy": best_strategy,
                    "performance": best_performance,
                    "metric": metric,
                    "mapping_data": best_mapping,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "success": False,
                    "regime": regime,
                    "message": f"No strategy mappings found for regime: {regime}",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error getting best strategy for regime: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def get_strategy_regime_performance(self, strategy_name: str, regime: str = None) -> Dict[str, Any]:
        """Get performance summary for a strategy across different regimes.

        Args:
            strategy_name: Name of the strategy
            regime: Specific regime to analyze (None for all regimes)

        Returns:
            Dictionary with performance summary
        """
        try:
            mappings = self.load_strategy_regime_mappings()

            if strategy_name not in mappings:
                return {
                    "success": False,
                    "strategy": strategy_name,
                    "message": f"No mappings found for strategy: {strategy_name}",
                    "timestamp": datetime.now().isoformat(),
                }

            strategy_mappings = mappings[strategy_name]

            if regime:
                # Analyze specific regime
                if regime not in strategy_mappings:
                    return {
                        "success": False,
                        "strategy": strategy_name,
                        "regime": regime,
                        "message": f"No mappings found for regime: {regime}",
                        "timestamp": datetime.now().isoformat(),
                    }

                regime_data = strategy_mappings[regime]
                return self._analyze_regime_performance(strategy_name, regime, regime_data)
            else:
                # Analyze all regimes
                regime_summaries = {}
                overall_performance = {
                    "total_mappings": 0,
                    "avg_confidence": 0.0,
                    "avg_win_rate": 0.0,
                    "avg_sharpe_ratio": 0.0,
                    "avg_total_return": 0.0,
                    "avg_max_drawdown": 0.0,
                }

                for regime_name, regime_data in strategy_mappings.items():
                    regime_summary = self._analyze_regime_performance(strategy_name, regime_name, regime_data)
                    regime_summaries[regime_name] = regime_summary

                    # Aggregate overall performance
                    if regime_summary["success"]:
                        overall_performance["total_mappings"] += regime_summary["mapping_count"]
                        overall_performance["avg_confidence"] += regime_summary["avg_confidence"]
                        overall_performance["avg_win_rate"] += regime_summary["avg_win_rate"]
                        overall_performance["avg_sharpe_ratio"] += regime_summary["avg_sharpe_ratio"]
                        overall_performance["avg_total_return"] += regime_summary["avg_total_return"]
                        overall_performance["avg_max_drawdown"] += regime_summary["avg_max_drawdown"]

                # Calculate averages
                regime_count = len(regime_summaries)
                if regime_count > 0:
                    for key in [
                        "avg_confidence",
                        "avg_win_rate",
                        "avg_sharpe_ratio",
                        "avg_total_return",
                        "avg_max_drawdown",
                    ]:
                        overall_performance[key] /= regime_count

                return {
                    "success": True,
                    "strategy": strategy_name,
                    "regime_summaries": regime_summaries,
                    "overall_performance": overall_performance,
                    "regime_count": regime_count,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error getting strategy regime performance: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _analyze_regime_performance(
        self, strategy_name: str, regime: str, regime_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance for a specific strategy-regime combination.

        Args:
            strategy_name: Name of the strategy
            regime: Market regime
            regime_data: List of mapping records for this regime

        Returns:
            Dictionary with performance analysis
        """
        try:
            if not regime_data:
                return {
                    "success": False,
                    "strategy": strategy_name,
                    "regime": regime,
                    "message": "No data available",
                    "timestamp": datetime.now().isoformat(),
                }

            # Calculate averages
            total_confidence = 0.0
            total_win_rate = 0.0
            total_sharpe_ratio = 0.0
            total_return = 0.0
            total_max_drawdown = 0.0
            valid_records = 0

            for record in regime_data:
                total_confidence += record.get("confidence", 0.0)
                total_win_rate += record.get("win_rate", 0.0)
                total_sharpe_ratio += record.get("sharpe_ratio", 0.0)
                total_return += record.get("total_return", 0.0)
                total_max_drawdown += record.get("max_drawdown", 0.0)
                valid_records += 1

            if valid_records > 0:
                avg_confidence = total_confidence / valid_records
                avg_win_rate = total_win_rate / valid_records
                avg_sharpe_ratio = total_sharpe_ratio / valid_records
                avg_total_return = total_return / valid_records
                avg_max_drawdown = total_max_drawdown / valid_records
            else:
                avg_confidence = avg_win_rate = avg_sharpe_ratio = avg_total_return = avg_max_drawdown = 0.0

            # Get most recent mapping
            latest_mapping = regime_data[-1] if regime_data else None

            return {
                "success": True,
                "strategy": strategy_name,
                "regime": regime,
                "mapping_count": len(regime_data),
                "avg_confidence": avg_confidence,
                "avg_win_rate": avg_win_rate,
                "avg_sharpe_ratio": avg_sharpe_ratio,
                "avg_total_return": avg_total_return,
                "avg_max_drawdown": avg_max_drawdown,
                "latest_mapping": latest_mapping,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing regime performance: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent strategy decisions.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of recent decisions
        """
        if not self.log_file.exists():
            return []

        decisions = []
        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        decision = json.loads(line.strip())
                        decisions.append(decision)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.error(f"Error reading strategy decisions: {e}")

        return decisions[::-1]  # Reverse to get most recent first

    def get_strategy_performance(
        self, strategy_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific strategy.

        Args:
            strategy_name: Name of the strategy
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary with strategy performance metrics
        """
        decisions = self.get_recent_decisions(limit=1000)

        # Filter by strategy name
        strategy_decisions = [d for d in decisions if d.get("strategy") == strategy_name]

        # Filter by date range if provided
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            strategy_decisions = [
                d
                for d in strategy_decisions
                if datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")) >= start_dt
            ]

        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            strategy_decisions = [
                d for d in strategy_decisions if datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")) <= end_dt
            ]

        # Calculate metrics
        total_decisions = len(strategy_decisions)
        if total_decisions == 0:
            return {
                "strategy": strategy_name,
                "total_decisions": 0,
                "accuracy": 0.0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "period": {"start": start_date, "end": end_date},
            }

        avg_confidence = sum(d.get("confidence", 0) for d in strategy_decisions) / total_decisions

        # Simple win rate calculation (assuming 'buy' decisions are wins)
        buy_decisions = [d for d in strategy_decisions if d.get("decision") == "buy"]
        win_rate = len(buy_decisions) / total_decisions if total_decisions > 0 else 0.0

        return {
            "strategy": strategy_name,
            "total_decisions": total_decisions,
            "accuracy": avg_confidence,  # Using confidence as accuracy proxy
            "win_rate": win_rate,
            "avg_confidence": avg_confidence,
            "period": {"start": start_date, "end": end_date},
        }

    def clear_logs(self) -> None:
        """Clear all strategy logs."""
        if self.log_file.exists():
            self.log_file.unlink()
        self.logger.info("Strategy logs cleared")

    def analyze_strategy(self, strategy_name: str, analysis_type: str = "performance") -> Dict[str, Any]:
        """Analyze strategy performance and behavior.

        Args:
            strategy_name: Name of the strategy to analyze
            analysis_type: Type of analysis ("performance", "behavior", "risk")

        Returns:
            Dictionary with analysis results
        """
        try:
            decisions = self.get_recent_decisions(limit=1000)
            strategy_decisions = [d for d in decisions if d.get("strategy") == strategy_name]

            if not strategy_decisions:
                return {
                    "strategy": strategy_name,
                    "analysis_type": analysis_type,
                    "status": "no_data",
                    "message": "No decisions found for this strategy",
                }

            if analysis_type == "performance":
                return self._analyze_performance(strategy_decisions)
            elif analysis_type == "behavior":
                return self._analyze_behavior(strategy_decisions)
            elif analysis_type == "risk":
                return self._analyze_risk(strategy_decisions)
            else:
                return {
                    "strategy": strategy_name,
                    "analysis_type": analysis_type,
                    "status": "invalid_type",
                    "message": f"Unknown analysis type: {analysis_type}",
                }

        except Exception as e:
            self.logger.error(f"Error analyzing strategy {strategy_name}: {e}")
            return {"strategy": strategy_name, "analysis_type": analysis_type, "status": "error", "message": str(e)}

    def _analyze_performance(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze strategy performance metrics."""
        total_decisions = len(decisions)
        buy_decisions = [d for d in decisions if d.get("decision") == "buy"]
        sell_decisions = [d for d in decisions if d.get("decision") == "sell"]
        hold_decisions = [d for d in decisions if d.get("decision") == "hold"]

        avg_confidence = sum(d.get("confidence", 0) for d in decisions) / total_decisions

        # Calculate decision distribution
        decision_distribution = {
            "buy": len(buy_decisions) / total_decisions,
            "sell": len(sell_decisions) / total_decisions,
            "hold": len(hold_decisions) / total_decisions,
        }

        # Calculate confidence trends
        recent_decisions = decisions[-10:] if len(decisions) >= 10 else decisions
        recent_confidence = sum(d.get("confidence", 0) for d in recent_decisions) / len(recent_decisions)
        confidence_trend = "improving" if recent_confidence > avg_confidence else "declining"

        return {
            "analysis_type": "performance",
            "status": "success",
            "total_decisions": total_decisions,
            "avg_confidence": round(avg_confidence, 3),
            "recent_confidence": round(recent_confidence, 3),
            "confidence_trend": confidence_trend,
            "decision_distribution": decision_distribution,
            "buy_ratio": decision_distribution["buy"],
            "sell_ratio": decision_distribution["sell"],
            "hold_ratio": decision_distribution["hold"],
        }

    def _analyze_behavior(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze strategy behavior patterns."""
        if len(decisions) < 2:
            return {
                "analysis_type": "behavior",
                "status": "insufficient_data",
                "message": "Need at least 2 decisions for behavior analysis",
            }

        # Analyze decision frequency
        timestamps = [datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")) for d in decisions]
        time_diffs = [(timestamps[i] - timestamps[i - 1]).total_seconds() / 3600 for i in range(1, len(timestamps))]
        avg_decision_interval = sum(time_diffs) / len(time_diffs)

        # Analyze confidence patterns
        confidences = [d.get("confidence", 0) for d in decisions]
        confidence_volatility = sum(abs(confidences[i] - confidences[i - 1]) for i in range(1, len(confidences))) / len(
            confidences
        )

        # Analyze decision consistency
        decisions_list = [d.get("decision") for d in decisions]
        decision_changes = sum(1 for i in range(1, len(decisions_list)) if decisions_list[i] != decisions_list[i - 1])
        consistency_ratio = 1 - (decision_changes / (len(decisions_list) - 1)) if len(decisions_list) > 1 else 1.0

        return {
            "analysis_type": "behavior",
            "status": "success",
            "avg_decision_interval_hours": round(avg_decision_interval, 2),
            "confidence_volatility": round(confidence_volatility, 3),
            "decision_consistency": round(consistency_ratio, 3),
            "behavior_pattern": "consistent" if consistency_ratio > 0.7 else "volatile",
            "decision_frequency": "high" if avg_decision_interval < 24 else "low",
        }

    def _analyze_risk(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze strategy risk metrics."""
        confidences = [d.get("confidence", 0) for d in decisions]

        # Calculate risk metrics
        avg_confidence = sum(confidences) / len(confidences)
        confidence_std = (sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)) ** 0.5

        # Risk assessment
        risk_level = "low"
        if avg_confidence < 0.6:
            risk_level = "high"
        elif avg_confidence < 0.8:
            risk_level = "medium"

        # Volatility assessment
        volatility_level = "low"
        if confidence_std > 0.2:
            volatility_level = "high"
        elif confidence_std > 0.1:
            volatility_level = "medium"

        return {
            "analysis_type": "risk",
            "status": "success",
            "avg_confidence": round(avg_confidence, 3),
            "confidence_std": round(confidence_std, 3),
            "risk_level": risk_level,
            "volatility_level": volatility_level,
            "risk_score": round((1 - avg_confidence) * 100, 1),
            "recommendation": self._get_risk_recommendation(risk_level, volatility_level),
        }

    def _get_risk_recommendation(self, risk_level: str, volatility_level: str) -> str:
        """Get risk management recommendation."""
        if risk_level == "high" and volatility_level == "high":
            return "Consider reducing position sizes and implementing stricter stop-losses"
        elif risk_level == "high":
            return "Review strategy parameters and consider additional validation"
        elif volatility_level == "high":
            return "Monitor closely and consider smoothing techniques"
        else:
            return "Strategy appears stable, continue monitoring"


def log_strategy_decision(
    strategy_name: str, decision: str, confidence: float, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log a strategy decision.

    Args:
        strategy_name: Name of the strategy
        decision: Decision made (buy/sell/hold)
        confidence: Confidence level (0-1)
        metadata: Additional metadata
    """
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy_name,
        "decision": decision,
        "confidence": confidence,
        "metadata": metadata or {},
    }

    logger.info(f"Strategy Decision: {json.dumps(log_data)}")


def get_strategy_analysis(
    strategy_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Get strategy analysis for a given period.

    Args:
        strategy_name: Name of the strategy
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        Dictionary with strategy analysis
    """
    try:
        # Get recent decisions
        decisions = get_recent_decisions(1000)  # Get last 1000 decisions

        # Filter by date range if provided
        if start_date or end_date:
            filtered_decisions = []
            for decision in decisions:
                if decision.get("strategy") == strategy_name:
                    decision_date = datetime.fromisoformat(decision["timestamp"].replace("Z", "+00:00"))

                    if start_date:
                        start_dt = datetime.fromisoformat(start_date)
                        if decision_date < start_dt:
                            continue

                    if end_date:
                        end_dt = datetime.fromisoformat(end_date)
                        if decision_date > end_dt:
                            continue

                    filtered_decisions.append(decision)
            decisions = filtered_decisions
        else:
            # Filter by strategy name only
            decisions = [d for d in decisions if d.get("strategy") == strategy_name]

        if not decisions:
            return {
                "strategy": strategy_name,
                "total_decisions": 0,
                "accuracy": 0.0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "period": {"start": start_date, "end": end_date},
                "status": "no_data",
            }

        # Calculate metrics
        total_decisions = len(decisions)
        avg_confidence = sum(d.get("confidence", 0) for d in decisions) / total_decisions

        # Calculate win rate (assuming positive outcomes for buy decisions with high confidence)
        buy_decisions = [d for d in decisions if d.get("decision") == "buy" and d.get("confidence", 0) > 0.7]
        win_rate = len(buy_decisions) / total_decisions if total_decisions > 0 else 0.0

        # Estimate accuracy based on confidence levels
        high_confidence_decisions = [d for d in decisions if d.get("confidence", 0) > 0.8]
        accuracy = len(high_confidence_decisions) / total_decisions if total_decisions > 0 else 0.0

        return {
            "strategy": strategy_name,
            "total_decisions": total_decisions,
            "accuracy": round(accuracy, 3),
            "win_rate": round(win_rate, 3),
            "avg_confidence": round(avg_confidence, 3),
            "period": {"start": start_date, "end": end_date},
            "status": "success",
            "last_decision": decisions[-1] if decisions else None,
            "decision_distribution": {
                "buy": len([d for d in decisions if d.get("decision") == "buy"]),
                "sell": len([d for d in decisions if d.get("decision") == "sell"]),
                "hold": len([d for d in decisions if d.get("decision") == "hold"]),
            },
        }

    except Exception as e:
        logger.error(f"Error analyzing strategy {strategy_name}: {e}")
        return {
            "strategy": strategy_name,
            "total_decisions": 0,
            "accuracy": 0.0,
            "win_rate": 0.0,
            "avg_confidence": 0.0,
            "period": {"start": start_date, "end": end_date},
            "status": "error",
            "error": str(e),
        }
