# -*- coding: utf-8 -*-
"""
Strategy switcher for the trading system.

This module handles strategy switching based on performance metrics and drift detection,
with support for multi-agent environments and robust logging.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from filelock import FileLock

# Local imports
from trading.config.settings import (
    STRATEGY_SWITCH_API_ENDPOINT,
    STRATEGY_SWITCH_BACKEND,
    STRATEGY_SWITCH_LOCK_TIMEOUT,
    STRATEGY_SWITCH_LOG_PATH,
)
from trading.models.model_registry import get_available_models

# Default values for missing constants
STRATEGY_SWITCH_THRESHOLD = 0.1  # 10% performance decline threshold
STRATEGY_PERFORMANCE_WINDOW = 30  # 30 days performance window
STRATEGY_MIN_PERFORMANCE = 0.5  # Minimum acceptable performance
STRATEGY_MAX_DRAWDOWN = 0.2  # Maximum acceptable drawdown
STRATEGY_SWITCH_COOLDOWN = 24  # Hours between strategy switches

logger = logging.getLogger(__name__)


class StrategySwitchBackend(str, Enum):
    """Available strategy switch logging backends."""

    FILE = "file"
    SQLITE = "sqlite"
    API = "api"


@dataclass
class StrategySwitch:
    """Strategy switch event data."""

    timestamp: datetime
    from_strategy: str
    to_strategy: str
    reason: str
    metrics: Dict[str, float]
    confidence: float
    agent_id: Optional[str] = None


class StrategySwitcher:
    """Handles strategy switching with robust multi-agent support."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy switcher.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.log_path = Path(self.config.get("log_path", STRATEGY_SWITCH_LOG_PATH))
        self.lock_path = self.log_path.with_suffix(".lock")
        self.lock_timeout = self.config.get("lock_timeout", STRATEGY_SWITCH_LOCK_TIMEOUT)
        self.backend = StrategySwitchBackend(self.config.get("backend", STRATEGY_SWITCH_BACKEND))
        self.api_endpoint = self.config.get("api_endpoint", STRATEGY_SWITCH_API_ENDPOINT)

        # Strategy switching parameters
        self.switch_threshold = self.config.get("switch_threshold", STRATEGY_SWITCH_THRESHOLD)
        self.performance_window = self.config.get("performance_window", STRATEGY_PERFORMANCE_WINDOW)
        self.min_performance = self.config.get("min_performance", STRATEGY_MIN_PERFORMANCE)
        self.max_drawdown = self.config.get("max_drawdown", STRATEGY_MAX_DRAWDOWN)
        self.switch_cooldown = self.config.get("switch_cooldown", STRATEGY_SWITCH_COOLDOWN)

        # Initialize backend
        self._init_backend()

        # Load last known working strategy
        self.last_working_strategy = self._load_last_working_strategy()

        # Strategy mapping for persistence
        self.strategy_map = {}

    def _init_backend(self):
        """Initialize the selected logging backend."""
        if self.backend == StrategySwitchBackend.SQLITE:
            self._init_sqlite()
        elif self.backend == StrategySwitchBackend.FILE:
            self._init_file()
        elif self.backend == StrategySwitchBackend.API:
            self._init_api()

    def _init_sqlite(self):
        """Initialize SQLite backend."""
        db_path = self.log_path.with_suffix(".db")
        self.conn = sqlite3.connect(str(db_path))
        self.cursor = self.conn.cursor()

        # Create tables
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_switches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                from_strategy TEXT NOT NULL,
                to_strategy TEXT NOT NULL,
                reason TEXT NOT NULL,
                metrics TEXT NOT NULL,
                confidence REAL NOT NULL,
                agent_id TEXT
            )
        """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS working_strategies (
                strategy TEXT PRIMARY KEY,
                last_used TEXT NOT NULL,
                success_count INTEGER DEFAULT 0
            )
        """
        )

        self.conn.commit()

    def _init_file(self):
        """Initialize file backend."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self._save_switches([])

    def _init_api(self):
        """Initialize API backend."""
        # Test connection
        try:
            response = requests.get(f"{self.api_endpoint}/health")
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to connect to API backend: {e}")
            raise

    def _get_file_lock(self) -> FileLock:
        """Get a file lock for thread-safe operations.

        Returns:
            FileLock instance
        """
        return FileLock(self.lock_path, timeout=self.lock_timeout)

    def _load_switches(self) -> List[Dict[str, Any]]:
        """Load strategy switches from the backend.

        Returns:
            List of strategy switch events
        """
        try:
            if self.backend == StrategySwitchBackend.SQLITE:
                self.cursor.execute("SELECT * FROM strategy_switches ORDER BY timestamp DESC")
                rows = self.cursor.fetchall()
                return [
                    {
                        "timestamp": row[1],
                        "from_strategy": row[2],
                        "to_strategy": row[3],
                        "reason": row[4],
                        "metrics": json.loads(row[5]),
                        "confidence": row[6],
                        "agent_id": row[7],
                    }
                    for row in rows
                ]

            elif self.backend == StrategySwitchBackend.FILE:
                with self._get_file_lock():
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        return json.load(f)

            elif self.backend == StrategySwitchBackend.API:
                response = requests.get(f"{self.api_endpoint}/switches")
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Error loading strategy switches: {e}")
            return []

    def _save_switches(self, switches: List[Dict[str, Any]]):
        """Save strategy switches to the backend.

        Args:
            switches: List of strategy switch events
        """
        try:
            if self.backend == StrategySwitchBackend.SQLITE:
                self.cursor.execute("DELETE FROM strategy_switches")
                for switch in switches:
                    self.cursor.execute(
                        """
                        INSERT INTO strategy_switches
                        (timestamp, from_strategy, to_strategy, reason, metrics, confidence, agent_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            switch["timestamp"],
                            switch["from_strategy"],
                            switch["to_strategy"],
                            switch["reason"],
                            json.dumps(switch["metrics"]),
                            switch["confidence"],
                            switch.get("agent_id"),
                        ),
                    )
                self.conn.commit()

            elif self.backend == StrategySwitchBackend.FILE:
                with self._get_file_lock():
                    with open(self.log_path, "w", encoding="utf-8") as f:
                        json.dump(switches, f, indent=2)

            elif self.backend == StrategySwitchBackend.API:
                response = requests.post(f"{self.api_endpoint}/switches", json=switches)
                response.raise_for_status()

        except Exception as e:
            logger.error(f"Error saving strategy switches: {e}")
            raise

    def _load_last_working_strategy(self) -> Optional[str]:
        """Load the last known working strategy.

        Returns:
            Strategy name or None if not found
        """
        try:
            if self.backend == StrategySwitchBackend.SQLITE:
                self.cursor.execute(
                    """
                    SELECT strategy FROM working_strategies
                    ORDER BY success_count DESC, last_used DESC
                    LIMIT 1
                """
                )
                row = self.cursor.fetchone()
                return row[0] if row else None

            elif self.backend == StrategySwitchBackend.FILE:
                with self._get_file_lock():
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        switches = json.load(f)
                        if switches:
                            return switches[0]["to_strategy"]

            elif self.backend == StrategySwitchBackend.API:
                response = requests.get(f"{self.api_endpoint}/last-working")
                response.raise_for_status()
                data = response.json()
                return data.get("strategy")

        except Exception as e:
            logger.error(f"Error loading last working strategy: {e}")
            return None

    def _update_working_strategy(self, strategy: str, success: bool = True):
        """Update the working strategy record.

        Args:
            strategy: Strategy name
            success: Whether the strategy was successful
        """
        try:
            if self.backend == StrategySwitchBackend.SQLITE:
                self.cursor.execute(
                    """
                    INSERT INTO working_strategies (strategy, last_used, success_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(strategy) DO UPDATE SET
                        last_used = excluded.last_used,
                        success_count = success_count + ?
                """,
                    (strategy, datetime.now().isoformat(), 1 if success else 0, 1 if success else 0),
                )
                self.conn.commit()

            elif self.backend == StrategySwitchBackend.FILE:
                # File backend doesn't track working strategies
                pass

            elif self.backend == StrategySwitchBackend.API:
                response = requests.post(
                    f"{self.api_endpoint}/working-strategy", json={"strategy": strategy, "success": success}
                )
                response.raise_for_status()

        except Exception as e:
            logger.error(f"Error updating working strategy: {e}")

    def switch_strategy_if_needed(
        self, current_strategy: str, metrics: Dict[str, float], agent_id: Optional[str] = None
    ) -> Optional[str]:
        """Check if strategy switch is needed and perform it.

        Args:
            current_strategy: Current strategy name
            metrics: Performance metrics
            agent_id: Optional agent identifier

        Returns:
            New strategy name if switched, None otherwise
        """
        try:
            # Get available models
            available_models = get_available_models()

            # Check for drift
            if self._detect_drift(metrics):
                # Get best alternative strategy
                new_strategy = self._get_best_strategy(metrics, available_models)

                if new_strategy and new_strategy != current_strategy:
                    # Log the switch
                    switch = StrategySwitch(
                        timestamp=datetime.now(),
                        from_strategy=current_strategy,
                        to_strategy=new_strategy,
                        reason="Drift detected",
                        metrics=metrics,
                        confidence=self._calculate_confidence(metrics),
                        agent_id=agent_id,
                    )

                    self._log_switch(switch)
                    self._update_working_strategy(new_strategy)
                    return new_strategy

            # If drift detection fails or no better strategy found,
            # fall back to last known working strategy
            if self.last_working_strategy and self.last_working_strategy != current_strategy:
                logger.info(f"Falling back to last known working strategy: {self.last_working_strategy}")
                return self.last_working_strategy

            return None

        except Exception as e:
            logger.error(f"Error in strategy switch: {e}")
            return None

    def _detect_drift(self, metrics: Dict[str, float]) -> bool:
        """Detect if strategy drift has occurred.

        Args:
            metrics: Performance metrics

        Returns:
            True if drift detected
        """
        # Implement drift detection logic here
        # This is a placeholder implementation
        return metrics.get("drift_score", 0) > 0.5

    def _get_best_strategy(self, metrics: Dict[str, float], available_models: List[str]) -> Optional[str]:
        """Get the best strategy based on metrics.

        Args:
            metrics: Performance metrics
            available_models: List of available model names

        Returns:
            Best strategy name or None
        """
        # Implement strategy selection logic here
        # This is a placeholder implementation
        return available_models[0] if available_models else None

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score for strategy switch.

        Args:
            metrics: Performance metrics

        Returns:
            Confidence score between 0 and 1
        """
        # Implement confidence calculation logic here
        # This is a placeholder implementation
        return 0.8

    def _log_switch(self, switch: StrategySwitch):
        """Log a strategy switch event.

        Args:
            switch: Strategy switch event to log
        """
        try:
            # Load existing switches
            switches = self._load_switches()

            # Add new switch
            switches.insert(
                0,
                {
                    "timestamp": switch.timestamp.isoformat(),
                    "from_strategy": switch.from_strategy,
                    "to_strategy": switch.to_strategy,
                    "reason": switch.reason,
                    "metrics": switch.metrics,
                    "confidence": switch.confidence,
                    "agent_id": switch.agent_id,
                },
            )

            # Save updated switches
            self._save_switches(switches)

            # Persist selected strategy and regime mapping to config
            self.strategy_map[switch.to_strategy] = {
                "regime": switch.metrics.get("market_regime", "unknown"),
                "timestamp": switch.timestamp.isoformat(),
                "confidence": switch.confidence,
            }
            self._save_strategy_map()

            logger.info(
                f"Strategy switch logged: {switch.from_strategy} -> {switch.to_strategy} "
                f"(confidence: {switch.confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Error logging strategy switch: {e}")

    def _save_strategy_map(self):
        """Save strategy regime mapping to config file."""
        try:
            config_path = Path("configs/strategy_regime_map.json")
            config_path.parent.mkdir(exist_ok=True)

            with open(config_path, "w") as f:
                json.dump(self.strategy_map, f, indent=2)

            logger.info(f"Strategy regime mapping saved to {config_path}")

        except Exception as e:
            logger.error(f"Error saving strategy regime mapping: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "conn"):
            self.conn.close()
