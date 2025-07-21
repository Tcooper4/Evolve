"""
Backtest Compiler

Enhanced with Batch 10 features: automatic deduplication of signals from overlapping strategies
within the same timeframe.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestCompiler:
    """Enhanced backtest compiler with signal deduplication."""

    def __init__(self, deduplication_enabled: bool = True):
        """Initialize the backtest compiler.

        Args:
            deduplication_enabled: Whether to enable signal deduplication
        """
        self.deduplication_enabled = deduplication_enabled
        self.compilation_history = []
        self.deduplication_stats = {}

        logger.info(
            f"BacktestCompiler initialized with deduplication={deduplication_enabled}"
        )

    def compile_signals(
        self,
        strategy_signals: Dict[str, pd.DataFrame],
        timeframe: str = "1D",
        deduplication_method: str = "priority",
    ) -> pd.DataFrame:
        """Compile signals from multiple strategies with deduplication.

        Args:
            strategy_signals: Dictionary of strategy names to signal DataFrames
            timeframe: Timeframe for signal alignment
            deduplication_method: Method for deduplication ('priority', 'confidence', 'timestamp')

        Returns:
            Compiled DataFrame with deduplicated signals
        """
        try:
            if not strategy_signals:
                logger.warning("No strategy signals provided")
                return pd.DataFrame()

            # Prepare signals for compilation
            prepared_signals = self._prepare_signals(strategy_signals, timeframe)

            # Compile signals
            compiled_df = self._compile_raw_signals(prepared_signals)

            # Apply deduplication if enabled
            if self.deduplication_enabled:
                compiled_df, stats = self._deduplicate_signals(
                    compiled_df, deduplication_method
                )
                self.deduplication_stats = stats
                logger.info(
                    f"Signal deduplication applied: {stats['duplicates_removed']} duplicates removed"
                )

            # Store compilation record
            self._store_compilation_record(strategy_signals, compiled_df, timeframe)

            logger.info(
                f"Signals compiled successfully: {len(compiled_df)} signals from {len(strategy_signals)} strategies"
            )
            return compiled_df

        except Exception as e:
            logger.error(f"Error compiling signals: {e}")
            return pd.DataFrame()

    def _prepare_signals(
        self, strategy_signals: Dict[str, pd.DataFrame], timeframe: str
    ) -> Dict[str, pd.DataFrame]:
        """Prepare signals for compilation."""
        prepared_signals = {}

        for strategy_name, signals_df in strategy_signals.items():
            try:
                # Ensure required columns exist
                required_columns = ["timestamp", "symbol", "signal"]
                missing_columns = [
                    col for col in required_columns if col not in signals_df.columns
                ]

                if missing_columns:
                    logger.warning(
                        f"Strategy {strategy_name} missing columns: {missing_columns}"
                    )
                    continue

                # Clean and prepare DataFrame
                prepared_df = signals_df.copy()

                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(prepared_df["timestamp"]):
                    prepared_df["timestamp"] = pd.to_datetime(prepared_df["timestamp"])

                # Add strategy name column
                prepared_df["strategy"] = strategy_name

                # Resample to timeframe if needed
                prepared_df = self._resample_to_timeframe(prepared_df, timeframe)

                # Sort by timestamp
                prepared_df = prepared_df.sort_values("timestamp")

                prepared_signals[strategy_name] = prepared_df

            except Exception as e:
                logger.error(
                    f"Error preparing signals for strategy {strategy_name}: {e}"
                )
                continue

        return prepared_signals

    def _resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample signals to specified timeframe."""
        try:
            # Set timestamp as index for resampling
            df_resampled = df.set_index("timestamp")

            # Resample based on timeframe
            if timeframe == "1D":
                df_resampled = df_resampled.resample("D").last()
            elif timeframe == "1H":
                df_resampled = df_resampled.resample("H").last()
            elif timeframe == "15T":
                df_resampled = df_resampled.resample("15T").last()
            elif timeframe == "5T":
                df_resampled = df_resampled.resample("5T").last()
            else:
                # Custom timeframe
                df_resampled = df_resampled.resample(timeframe).last()

            # Reset index
            df_resampled = df_resampled.reset_index()

            # Remove rows with all NaN values
            df_resampled = df_resampled.dropna(subset=["signal"])

            return df_resampled

        except Exception as e:
            logger.error(f"Error resampling to timeframe {timeframe}: {e}")
            return df

    def _compile_raw_signals(
        self, prepared_signals: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compile raw signals from all strategies."""
        try:
            # Concatenate all signals
            all_signals = []

            for strategy_name, signals_df in prepared_signals.items():
                all_signals.append(signals_df)

            if not all_signals:
                return pd.DataFrame()

            compiled_df = pd.concat(all_signals, ignore_index=True)

            # Sort by timestamp and symbol
            compiled_df = compiled_df.sort_values(["timestamp", "symbol", "strategy"])

            return compiled_df

        except Exception as e:
            logger.error(f"Error compiling raw signals: {e}")
            return pd.DataFrame()

    def _deduplicate_signals(
        self, compiled_df: pd.DataFrame, method: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Deduplicate signals using specified method."""
        try:
            initial_count = len(compiled_df)

            if method == "priority":
                deduplicated_df = self._deduplicate_by_priority(compiled_df)
            elif method == "confidence":
                deduplicated_df = self._deduplicate_by_confidence(compiled_df)
            elif method == "timestamp":
                deduplicated_df = self._deduplicate_by_timestamp(compiled_df)
            else:
                logger.warning(
                    f"Unknown deduplication method: {method}, using priority"
                )
                deduplicated_df = self._deduplicate_by_priority(compiled_df)

            final_count = len(deduplicated_df)
            duplicates_removed = initial_count - final_count

            stats = {
                "method": method,
                "initial_signals": initial_count,
                "final_signals": final_count,
                "duplicates_removed": duplicates_removed,
                "deduplication_rate": (
                    duplicates_removed / initial_count if initial_count > 0 else 0
                ),
            }

            return deduplicated_df, stats

        except Exception as e:
            logger.error(f"Error during signal deduplication: {e}")
            return compiled_df, {"error": str(e)}

    def _deduplicate_by_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate signals by strategy priority."""
        try:
            # Define strategy priorities (higher number = higher priority)
            strategy_priorities = {
                "momentum": 3,
                "mean_reversion": 2,
                "breakout": 4,
                "trend_following": 3,
                "arbitrage": 5,
                "default": 1,
            }

            # Add priority column
            df["priority"] = (
                df["strategy"]
                .map(strategy_priorities)
                .fillna(strategy_priorities["default"])
            )

            # Group by timestamp and symbol, keep highest priority signal
            deduplicated = df.loc[
                df.groupby(["timestamp", "symbol"])["priority"].idxmax()
            ]

            # Remove priority column
            deduplicated = deduplicated.drop("priority", axis=1)

            return deduplicated

        except Exception as e:
            logger.error(f"Error in priority-based deduplication: {e}")
            return df

    def _deduplicate_by_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate signals by confidence score."""
        try:
            # Add confidence column if not present
            if "confidence" not in df.columns:
                # Generate confidence based on signal strength or other factors
                df["confidence"] = self._calculate_confidence(df)

            # Group by timestamp and symbol, keep highest confidence signal
            deduplicated = df.loc[
                df.groupby(["timestamp", "symbol"])["confidence"].idxmax()
            ]

            return deduplicated

        except Exception as e:
            logger.error(f"Error in confidence-based deduplication: {e}")
            return df

    def _deduplicate_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate signals by keeping the most recent signal."""
        try:
            # Group by timestamp and symbol, keep the last signal (most recent)
            deduplicated = (
                df.groupby(["timestamp", "symbol"]).tail(1).reset_index(drop=True)
            )

            return deduplicated

        except Exception as e:
            logger.error(f"Error in timestamp-based deduplication: {e}")
            return df

    def _calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence scores for signals."""
        try:
            confidence_scores = pd.Series(index=df.index, dtype=float)

            for idx, row in df.iterrows():
                # Base confidence
                confidence = 0.5

                # Adjust based on signal type
                signal = str(row.get("signal", "")).lower()
                if "strong" in signal or "high" in signal:
                    confidence += 0.3
                elif "weak" in signal or "low" in signal:
                    confidence -= 0.2

                # Adjust based on strategy
                strategy = str(row.get("strategy", "")).lower()
                if "momentum" in strategy:
                    confidence += 0.1
                elif "arbitrage" in strategy:
                    confidence += 0.2

                # Add some randomness for demonstration
                confidence += np.random.uniform(-0.1, 0.1)

                # Clamp to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                confidence_scores.iloc[idx] = confidence

            return confidence_scores

        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return pd.Series(0.5, index=df.index)

    def compile_with_overlap_detection(
        self, strategy_signals: Dict[str, pd.DataFrame], overlap_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Compile signals with overlap detection and analysis.

        Args:
            strategy_signals: Dictionary of strategy signals
            overlap_threshold: Threshold for considering signals overlapping

        Returns:
            Dictionary with compiled signals and overlap analysis
        """
        try:
            # Compile signals normally
            compiled_df = self.compile_signals(strategy_signals)

            if compiled_df.empty:
                return {"compiled_signals": compiled_df, "overlap_analysis": {}}

            # Analyze overlaps
            overlap_analysis = self._analyze_overlaps(compiled_df, overlap_threshold)

            return {
                "compiled_signals": compiled_df,
                "overlap_analysis": overlap_analysis,
                "deduplication_stats": self.deduplication_stats,
            }

        except Exception as e:
            logger.error(f"Error in overlap detection compilation: {e}")
            return {"error": str(e)}

    def _analyze_overlaps(self, df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Analyze signal overlaps between strategies."""
        try:
            analysis = {
                "total_signals": len(df),
                "overlapping_periods": 0,
                "overlap_details": [],
                "strategy_overlaps": {},
            }

            # Group by timestamp and symbol
            grouped = df.groupby(["timestamp", "symbol"])

            for (timestamp, symbol), group in grouped:
                if len(group) > 1:
                    analysis["overlapping_periods"] += 1

                    # Get strategies involved
                    strategies = group["strategy"].tolist()
                    signals = group["signal"].tolist()

                    overlap_detail = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "strategies": strategies,
                        "signals": signals,
                        "overlap_count": len(strategies),
                    }
                    analysis["overlap_details"].append(overlap_detail)

                    # Track strategy pair overlaps
                    for i, strategy1 in enumerate(strategies):
                        for j, strategy2 in enumerate(strategies[i + 1:], i + 1):
                            pair = tuple(sorted([strategy1, strategy2]))
                            analysis["strategy_overlaps"][pair] = (
                                analysis["strategy_overlaps"].get(pair, 0) + 1
                            )

            # Calculate overlap percentage
            analysis["overlap_percentage"] = (
                analysis["overlapping_periods"] / len(grouped) * 100
                if len(grouped) > 0
                else 0
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing overlaps: {e}")
            return {"error": str(e)}

    def _store_compilation_record(
        self,
        strategy_signals: Dict[str, pd.DataFrame],
        compiled_df: pd.DataFrame,
        timeframe: str,
    ):
        """Store compilation record for tracking."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "strategies_count": len(strategy_signals),
            "strategy_names": list(strategy_signals.keys()),
            "timeframe": timeframe,
            "initial_signals": sum(len(df) for df in strategy_signals.values()),
            "final_signals": len(compiled_df),
            "deduplication_enabled": self.deduplication_enabled,
            "deduplication_stats": self.deduplication_stats.copy(),
        }
        self.compilation_history.append(record)

    def get_compilation_summary(self) -> Dict[str, Any]:
        """Get summary of all compilations."""
        if not self.compilation_history:
            return {"total_compilations": 0}

        total_compilations = len(self.compilation_history)
        total_duplicates_removed = sum(
            record.get("deduplication_stats", {}).get("duplicates_removed", 0)
            for record in self.compilation_history
        )

        return {
            "total_compilations": total_compilations,
            "total_duplicates_removed": total_duplicates_removed,
            "average_deduplication_rate": (
                total_duplicates_removed / total_compilations
                if total_compilations > 0
                else 0
            ),
            "recent_compilations": (
                self.compilation_history[-5:]
                if len(self.compilation_history) > 5
                else self.compilation_history
            ),
        }

    def export_compilation_report(self, filename: str) -> bool:
        """Export compilation report to file."""
        try:
            import json

            report = {
                "compilation_summary": self.get_compilation_summary(),
                "compilation_history": self.compilation_history,
                "deduplication_stats": self.deduplication_stats,
            }

            with open(filename, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Compilation report exported to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error exporting compilation report: {e}")
            return False
