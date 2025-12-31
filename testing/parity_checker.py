"""
Parity Checker for Backtest vs Live Execution

Ensures same inputs produce same outputs between backtest and live execution.
Detects discrepancies in signals, features, and decisions.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ParityChecker:
    """Checks parity between backtest and live execution."""

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize parity checker.

        Args:
            tolerance: Tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.backtest_log: List[Dict[str, Any]] = []
        self.live_log: List[Dict[str, Any]] = []
        self.parity_results: List[Dict[str, Any]] = []

    def log_backtest_decision(
        self,
        timestamp: datetime,
        symbol: str,
        signal: Dict[str, Any],
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a backtest decision.

        Args:
            timestamp: Decision timestamp
            symbol: Trading symbol
            signal: Trading signal (action, quantity, etc.)
            features: Feature vector used for decision
            context: Additional context (strategy, model, etc.)
        """
        try:
            decision = {
                "timestamp": timestamp,
                "symbol": symbol,
                "signal": signal.copy() if isinstance(signal, dict) else signal,
                "features": features.copy() if isinstance(features, dict) else features,
                "context": context.copy() if context else {},
                "log_time": datetime.now(),
            }
            self.backtest_log.append(decision)
            logger.debug(f"Logged backtest decision: {symbol} at {timestamp}")

        except Exception as e:
            logger.error(f"Error logging backtest decision: {e}")

    def log_live_decision(
        self,
        timestamp: datetime,
        symbol: str,
        signal: Dict[str, Any],
        features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a live execution decision.

        Args:
            timestamp: Decision timestamp
            symbol: Trading symbol
            signal: Trading signal (action, quantity, etc.)
            features: Feature vector used for decision
            context: Additional context (strategy, model, etc.)
        """
        try:
            decision = {
                "timestamp": timestamp,
                "symbol": symbol,
                "signal": signal.copy() if isinstance(signal, dict) else signal,
                "features": features.copy() if isinstance(features, dict) else features,
                "context": context.copy() if context else {},
                "log_time": datetime.now(),
            }
            self.live_log.append(decision)
            logger.debug(f"Logged live decision: {symbol} at {timestamp}")

        except Exception as e:
            logger.error(f"Error logging live decision: {e}")

    def check_parity(
        self,
        time_window_seconds: int = 60,
        check_features: bool = True,
    ) -> Dict[str, Any]:
        """
        Check if backtest and live produce same results.

        Args:
            time_window_seconds: Time window for matching decisions (default 60s)
            check_features: Whether to check feature parity

        Returns:
            Dictionary with parity results
        """
        try:
            if len(self.backtest_log) == 0 or len(self.live_log) == 0:
                return {
                    "parity": False,
                    "reason": "No data to compare",
                    "backtest_count": len(self.backtest_log),
                    "live_count": len(self.live_log),
                    "matches": 0,
                    "mismatches": 0,
                }

            matches = 0
            mismatches = []
            unmatched_backtest = []
            unmatched_live = []

            # Match decisions by timestamp and symbol
            for bt_decision in self.backtest_log:
                # Find corresponding live decision within time window
                live_decision = self._find_matching_decision(
                    bt_decision, self.live_log, time_window_seconds
                )

                if live_decision is None:
                    unmatched_backtest.append({
                        "timestamp": bt_decision["timestamp"],
                        "symbol": bt_decision["symbol"],
                        "reason": "No matching live decision",
                    })
                    continue

                # Compare signals
                signal_match, signal_diff = self._compare_signals(
                    bt_decision["signal"], live_decision["signal"]
                )

                if not signal_match:
                    mismatches.append({
                        "timestamp": bt_decision["timestamp"],
                        "symbol": bt_decision["symbol"],
                        "backtest_signal": bt_decision["signal"],
                        "live_signal": live_decision["signal"],
                        "reason": "Signal mismatch",
                        "differences": signal_diff,
                    })
                    continue

                # Compare features if requested
                if check_features:
                    feature_match, feature_diff = self._compare_features(
                        bt_decision["features"], live_decision["features"]
                    )

                    if not feature_match:
                        mismatches.append({
                            "timestamp": bt_decision["timestamp"],
                            "symbol": bt_decision["symbol"],
                            "reason": "Feature mismatch",
                            "differences": feature_diff,
                        })
                        continue

                matches += 1

            # Find unmatched live decisions
            for live_decision in self.live_log:
                bt_decision = self._find_matching_decision(
                    live_decision, self.backtest_log, time_window_seconds
                )
                if bt_decision is None:
                    unmatched_live.append({
                        "timestamp": live_decision["timestamp"],
                        "symbol": live_decision["symbol"],
                        "reason": "No matching backtest decision",
                    })

            parity = len(mismatches) == 0 and len(unmatched_backtest) == 0 and len(unmatched_live) == 0

            result = {
                "parity": parity,
                "matches": matches,
                "mismatches": len(mismatches),
                "unmatched_backtest": len(unmatched_backtest),
                "unmatched_live": len(unmatched_live),
                "backtest_count": len(self.backtest_log),
                "live_count": len(self.live_log),
                "mismatch_details": mismatches[:10],  # Limit to first 10
                "unmatched_backtest_details": unmatched_backtest[:10],
                "unmatched_live_details": unmatched_live[:10],
                "check_time": datetime.now().isoformat(),
            }

            self.parity_results.append(result)
            return result

        except Exception as e:
            logger.error(f"Error checking parity: {e}")
            return {
                "parity": False,
                "reason": f"Parity check error: {str(e)}",
                "backtest_count": len(self.backtest_log),
                "live_count": len(self.live_log),
            }

    def _find_matching_decision(
        self,
        target_decision: Dict[str, Any],
        decision_list: List[Dict[str, Any]],
        time_window_seconds: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Find live decision matching backtest decision.

        Args:
            target_decision: Decision to match
            decision_list: List of decisions to search
            time_window_seconds: Time window for matching

        Returns:
            Matching decision or None
        """
        target_timestamp = target_decision["timestamp"]
        target_symbol = target_decision["symbol"]

        for decision in decision_list:
            if decision["symbol"] != target_symbol:
                continue

            # Check if timestamps are within window
            time_diff = abs((decision["timestamp"] - target_timestamp).total_seconds())
            if time_diff <= time_window_seconds:
                return decision

        return None

    def _compare_signals(
        self, bt_signal: Dict[str, Any], live_signal: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Compare signal dictionaries.

        Args:
            bt_signal: Backtest signal
            live_signal: Live signal

        Returns:
            Tuple of (match, differences)
        """
        differences = []

        # Normalize signal format
        bt_action = self._normalize_action(bt_signal)
        live_action = self._normalize_action(live_signal)

        if bt_action != live_action:
            differences.append(f"Action: backtest={bt_action}, live={live_action}")

        # Compare quantity
        bt_quantity = bt_signal.get("quantity") or bt_signal.get("size", 0)
        live_quantity = live_signal.get("quantity") or live_signal.get("size", 0)

        if abs(bt_quantity - live_quantity) > self.tolerance:
            differences.append(
                f"Quantity: backtest={bt_quantity}, live={live_quantity}"
            )

        # Compare price if present
        bt_price = bt_signal.get("price") or bt_signal.get("entry_price")
        live_price = live_signal.get("price") or live_signal.get("entry_price")

        if bt_price is not None and live_price is not None:
            if abs(bt_price - live_price) > self.tolerance:
                differences.append(f"Price: backtest={bt_price}, live={live_price}")

        return len(differences) == 0, differences

    def _normalize_action(self, signal: Dict[str, Any]) -> str:
        """Normalize action to standard format."""
        action = signal.get("action") or signal.get("type") or signal.get("direction", "")

        if isinstance(action, str):
            action_lower = action.lower()
            # Normalize to buy/sell/hold
            if action_lower in ["buy", "long", "1"]:
                return "buy"
            elif action_lower in ["sell", "short", "-1"]:
                return "sell"
            elif action_lower in ["hold", "0", "none"]:
                return "hold"
            else:
                return action_lower
        else:
            # Handle numeric actions
            if action == 1 or action == 1.0:
                return "buy"
            elif action == -1 or action == -1.0:
                return "sell"
            else:
                return "hold"

    def _compare_features(
        self, bt_features: Dict[str, Any], live_features: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Compare feature vectors.

        Args:
            bt_features: Backtest features
            live_features: Live features

        Returns:
            Tuple of (match, differences)
        """
        differences = []

        # Check all backtest features exist in live
        for key, bt_value in bt_features.items():
            if key not in live_features:
                differences.append(f"Missing in live: {key}")
                continue

            live_value = live_features[key]

            # Compare values (with tolerance for floating point)
            if isinstance(bt_value, (int, float)) and isinstance(live_value, (int, float)):
                if abs(bt_value - live_value) > self.tolerance:
                    differences.append(
                        f"{key}: backtest={bt_value}, live={live_value}, diff={abs(bt_value - live_value)}"
                    )
            elif isinstance(bt_value, (list, np.ndarray)) and isinstance(
                live_value, (list, np.ndarray)
            ):
                # Compare arrays
                bt_arr = np.array(bt_value)
                live_arr = np.array(live_value)
                if not np.allclose(bt_arr, live_arr, atol=self.tolerance):
                    differences.append(
                        f"{key}: arrays differ (max_diff={np.max(np.abs(bt_arr - live_arr))})"
                    )
            else:
                if bt_value != live_value:
                    differences.append(
                        f"{key}: backtest={bt_value}, live={live_value}"
                    )

        # Check for extra features in live
        for key in live_features:
            if key not in bt_features:
                differences.append(f"Extra in live: {key}")

        return len(differences) == 0, differences

    def get_parity_summary(self) -> Dict[str, Any]:
        """Get summary of parity checks."""
        if not self.parity_results:
            return {"message": "No parity checks performed"}

        latest = self.parity_results[-1]
        return {
            "latest_check": latest,
            "total_checks": len(self.parity_results),
            "parity_rate": sum(1 for r in self.parity_results if r["parity"]) / len(self.parity_results),
            "average_matches": np.mean([r["matches"] for r in self.parity_results]),
            "average_mismatches": np.mean([r["mismatches"] for r in self.parity_results]),
        }

    def clear_logs(self) -> None:
        """Clear all logs."""
        self.backtest_log.clear()
        self.live_log.clear()
        self.parity_results.clear()
        logger.info("Parity checker logs cleared")

    def export_logs(self, filepath: str) -> None:
        """Export logs to file."""
        try:
            import json

            export_data = {
                "backtest_log": [
                    {
                        **log,
                        "timestamp": log["timestamp"].isoformat(),
                        "log_time": log["log_time"].isoformat(),
                    }
                    for log in self.backtest_log
                ],
                "live_log": [
                    {
                        **log,
                        "timestamp": log["timestamp"].isoformat(),
                        "log_time": log["log_time"].isoformat(),
                    }
                    for log in self.live_log
                ],
                "parity_results": self.parity_results,
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported parity logs to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting logs: {e}")


# Global parity checker instance
_parity_checker = None


def get_parity_checker() -> ParityChecker:
    """Get the global parity checker instance."""
    global _parity_checker
    if _parity_checker is None:
        _parity_checker = ParityChecker()
    return _parity_checker

