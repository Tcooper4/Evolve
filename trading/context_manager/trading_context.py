"""
Trading Context Manager - Batch 19
Enhanced context management with session expiration and strategy limits
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Trading session status."""

    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    COMPLETED = "completed"
    ERROR = "error"


class StrategyType(Enum):
    """Strategy types for tracking."""

    FORECAST = "forecast"
    SIGNAL = "signal"
    EXECUTION = "execution"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"


@dataclass
class TradingSession:
    """Trading session with metadata."""

    session_id: str
    user_id: str
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    strategies: List[str] = field(default_factory=list)
    context_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_age: Optional[timedelta] = None


@dataclass
class StrategyContext:
    """Context for individual strategy."""

    strategy_id: str
    strategy_type: StrategyType
    session_id: str
    created_at: datetime
    last_used: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "active"


class TradingContextManager:
    """
    Enhanced trading context manager with session management.

    Features:
    - Auto-expiring sessions with max_age parameter
    - Concurrent strategy limits
    - Session cleanup and resource management
    - Context data persistence
    """

    def __init__(
        self,
        max_active_strategies: int = 10,
        default_session_max_age: timedelta = timedelta(hours=24),
        cleanup_interval: int = 300,  # 5 minutes
        enable_persistence: bool = True,
    ):
        """
        Initialize trading context manager.

        Args:
            max_active_strategies: Maximum concurrent active strategies
            default_session_max_age: Default session expiration time
            cleanup_interval: Session cleanup interval in seconds
            enable_persistence: Enable context data persistence
        """
        self.max_active_strategies = max_active_strategies
        self.default_session_max_age = default_session_max_age
        self.cleanup_interval = cleanup_interval
        self.enable_persistence = enable_persistence

        # Session storage
        self.sessions: Dict[str, TradingSession] = {}
        self.strategy_contexts: Dict[str, StrategyContext] = {}

        # Active strategy tracking
        self.active_strategies: Set[str] = set()
        self.strategy_session_map: Dict[str, str] = {}  # strategy_id -> session_id

        # Threading
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False

        # Statistics
        self.stats = {
            "total_sessions": 0,
            "expired_sessions": 0,
            "active_strategies": 0,
            "max_strategies_reached": 0,
        }

        logger.info(
            f"TradingContextManager initialized with max strategies: {max_active_strategies}"
        )

        # Start cleanup thread
        self._start_cleanup_thread()

    def create_session(
        self,
        user_id: str,
        max_age: Optional[timedelta] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new trading session.

        Args:
            user_id: User identifier
            max_age: Session expiration time (uses default if None)
            initial_context: Initial context data

        Returns:
            Session ID
        """
        with self.lock:
            session_id = str(uuid.uuid4())
            now = datetime.now()

            session = TradingSession(
                session_id=session_id,
                user_id=user_id,
                status=SessionStatus.ACTIVE,
                created_at=now,
                last_activity=now,
                context_data=initial_context or {},
                max_age=max_age or self.default_session_max_age,
            )

            self.sessions[session_id] = session
            self.stats["total_sessions"] += 1

            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id

    def get_session(self, session_id: str) -> Optional[TradingSession]:
        """Get session by ID."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session and session.status != SessionStatus.EXPIRED:
                session.last_activity = datetime.now()
                return session
            return None

    def update_session_context(
        self, session_id: str, context_data: Dict[str, Any]
    ) -> bool:
        """
        Update session context data.

        Args:
            session_id: Session identifier
            context_data: New context data

        Returns:
            Success status
        """
        with self.lock:
            session = self.sessions.get(session_id)
            if not session or session.status == SessionStatus.EXPIRED:
                return False

            session.context_data.update(context_data)
            session.last_activity = datetime.now()

            logger.debug(f"Updated context for session {session_id}")
            return True

    def register_strategy(
        self,
        session_id: str,
        strategy_id: str,
        strategy_type: StrategyType,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a strategy in a session.

        Args:
            session_id: Session identifier
            strategy_id: Strategy identifier
            strategy_type: Type of strategy
            parameters: Strategy parameters

        Returns:
            Success status
        """
        with self.lock:
            # Check session validity
            session = self.sessions.get(session_id)
            if not session or session.status == SessionStatus.EXPIRED:
                logger.warning(
                    f"Cannot register strategy: session {session_id} not found or expired"
                )
                return False

            # Check strategy limit
            if len(self.active_strategies) >= self.max_active_strategies:
                logger.warning(
                    f"Maximum active strategies ({self.max_active_strategies}) reached"
                )
                self.stats["max_strategies_reached"] += 1
                return False

            # Create strategy context
            now = datetime.now()
            strategy_context = StrategyContext(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                session_id=session_id,
                created_at=now,
                last_used=now,
                parameters=parameters or {},
            )

            # Register strategy
            self.strategy_contexts[strategy_id] = strategy_context
            self.active_strategies.add(strategy_id)
            self.strategy_session_map[strategy_id] = session_id
            session.strategies.append(strategy_id)

            self.stats["active_strategies"] = len(self.active_strategies)

            logger.info(f"Registered strategy {strategy_id} in session {session_id}")
            return True

    def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Success status
        """
        with self.lock:
            if strategy_id not in self.active_strategies:
                return False

            # Remove from active strategies
            self.active_strategies.remove(strategy_id)

            # Remove from session
            session_id = self.strategy_session_map.pop(strategy_id, None)
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                if strategy_id in session.strategies:
                    session.strategies.remove(strategy_id)

            # Remove context
            self.strategy_contexts.pop(strategy_id, None)

            self.stats["active_strategies"] = len(self.active_strategies)

            logger.info(f"Unregistered strategy {strategy_id}")
            return True

    def get_strategy_context(self, strategy_id: str) -> Optional[StrategyContext]:
        """Get strategy context."""
        with self.lock:
            return self.strategy_contexts.get(strategy_id)

    def update_strategy_metrics(
        self, strategy_id: str, metrics: Dict[str, float]
    ) -> bool:
        """
        Update strategy performance metrics.

        Args:
            strategy_id: Strategy identifier
            metrics: Performance metrics

        Returns:
            Success status
        """
        with self.lock:
            context = self.strategy_contexts.get(strategy_id)
            if not context:
                return False

            context.performance_metrics.update(metrics)
            context.last_used = datetime.now()

            # Update session activity
            session = self.sessions.get(context.session_id)
            if session:
                session.last_activity = datetime.now()

            return True

    def pause_session(self, session_id: str) -> bool:
        """Pause a session."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            session.status = SessionStatus.PAUSED
            logger.info(f"Paused session {session_id}")
            return True

    def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session or session.status != SessionStatus.PAUSED:
                return False

            session.status = SessionStatus.ACTIVE
            session.last_activity = datetime.now()
            logger.info(f"Resumed session {session_id}")
            return True

    def complete_session(self, session_id: str) -> bool:
        """Mark session as completed."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            session.status = SessionStatus.COMPLETED

            # Unregister all strategies in this session
            strategies_to_remove = [s for s in session.strategies]
            for strategy_id in strategies_to_remove:
                self.unregister_strategy(strategy_id)

            logger.info(f"Completed session {session_id}")
            return True

    def _check_session_expiration(self, session: TradingSession) -> bool:
        """Check if session has expired."""
        if not session.max_age:
            return False

        expiration_time = session.last_activity + session.max_age
        return datetime.now() > expiration_time

    def _expire_session(self, session_id: str) -> bool:
        """Expire a session."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False

            session.status = SessionStatus.EXPIRED

            # Unregister all strategies in this session
            strategies_to_remove = [s for s in session.strategies]
            for strategy_id in strategies_to_remove:
                self.unregister_strategy(strategy_id)

            self.stats["expired_sessions"] += 1

            logger.info(f"Expired session {session_id}")
            return True

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        expired_count = 0

        with self.lock:
            sessions_to_expire = []

            for session_id, session in self.sessions.items():
                if (
                    session.status == SessionStatus.ACTIVE
                    and self._check_session_expiration(session)
                ):
                    sessions_to_expire.append(session_id)

            for session_id in sessions_to_expire:
                if self._expire_session(session_id):
                    expired_count += 1

        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")

        return expired_count

    def _cleanup_worker(self):
        """Background cleanup worker."""
        while self.running:
            try:
                self.cleanup_expired_sessions()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(60)  # Wait 1 minute on error

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self.cleanup_thread is None:
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker, daemon=True
            )
            self.cleanup_thread.start()
            logger.info("Started cleanup thread")

    def stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
            self.cleanup_thread = None
            logger.info("Stopped cleanup thread")

    def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return {}

            session_strategies = [
                self.strategy_contexts.get(sid)
                for sid in session.strategies
                if sid in self.strategy_contexts
            ]

            stats = {
                "session_id": session_id,
                "user_id": session.user_id,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "age_seconds": (datetime.now() - session.created_at).total_seconds(),
                "inactive_seconds": (
                    datetime.now() - session.last_activity
                ).total_seconds(),
                "strategy_count": len(session.strategies),
                "max_age_seconds": (
                    session.max_age.total_seconds() if session.max_age else None
                ),
                "is_expired": self._check_session_expiration(session),
                "strategies": [
                    {
                        "strategy_id": ctx.strategy_id,
                        "type": ctx.strategy_type.value,
                        "created_at": ctx.created_at.isoformat(),
                        "last_used": ctx.last_used.isoformat(),
                        "performance_metrics": ctx.performance_metrics,
                    }
                    for ctx in session_strategies
                ],
            }

            return stats

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics."""
        with self.lock:
            active_sessions = sum(
                1 for s in self.sessions.values() if s.status == SessionStatus.ACTIVE
            )
            paused_sessions = sum(
                1 for s in self.sessions.values() if s.status == SessionStatus.PAUSED
            )

            strategy_type_counts = {}
            for ctx in self.strategy_contexts.values():
                strategy_type = ctx.strategy_type.value
                strategy_type_counts[strategy_type] = (
                    strategy_type_counts.get(strategy_type, 0) + 1
                )

            stats = {
                "total_sessions": len(self.sessions),
                "active_sessions": active_sessions,
                "paused_sessions": paused_sessions,
                "expired_sessions": self.stats["expired_sessions"],
                "active_strategies": len(self.active_strategies),
                "max_active_strategies": self.max_active_strategies,
                "strategy_type_distribution": strategy_type_counts,
                "max_strategies_reached_count": self.stats["max_strategies_reached"],
                "cleanup_interval_seconds": self.cleanup_interval,
            }

            return stats

    def save_context_data(self, filepath: str) -> bool:
        """Save context data to file."""
        if not self.enable_persistence:
            return False

        try:
            with self.lock:
                data = {
                    "sessions": {
                        sid: {
                            "session_id": session.session_id,
                            "user_id": session.user_id,
                            "status": session.status.value,
                            "created_at": session.created_at.isoformat(),
                            "last_activity": session.last_activity.isoformat(),
                            "strategies": session.strategies,
                            "context_data": session.context_data,
                            "metadata": session.metadata,
                            "max_age_seconds": (
                                session.max_age.total_seconds()
                                if session.max_age
                                else None
                            ),
                        }
                        for sid, session in self.sessions.items()
                    },
                    "strategy_contexts": {
                        sid: {
                            "strategy_id": ctx.strategy_id,
                            "strategy_type": ctx.strategy_type.value,
                            "session_id": ctx.session_id,
                            "created_at": ctx.created_at.isoformat(),
                            "last_used": ctx.last_used.isoformat(),
                            "parameters": ctx.parameters,
                            "performance_metrics": ctx.performance_metrics,
                            "status": ctx.status,
                        }
                        for sid, ctx in self.strategy_contexts.items()
                    },
                    "stats": self.stats,
                }

                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2, default=str)

                logger.info(f"Saved context data to {filepath}")
                return True

        except Exception as e:
            logger.error(f"Failed to save context data: {e}")
            return False

    def load_context_data(self, filepath: str) -> bool:
        """Load context data from file."""
        if not self.enable_persistence:
            return False

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            with self.lock:
                # Clear existing data
                self.sessions.clear()
                self.strategy_contexts.clear()
                self.active_strategies.clear()
                self.strategy_session_map.clear()

                # Load sessions
                for sid, session_data in data.get("sessions", {}).items():
                    session = TradingSession(
                        session_id=session_data["session_id"],
                        user_id=session_data["user_id"],
                        status=SessionStatus(session_data["status"]),
                        created_at=datetime.fromisoformat(session_data["created_at"]),
                        last_activity=datetime.fromisoformat(
                            session_data["last_activity"]
                        ),
                        strategies=session_data["strategies"],
                        context_data=session_data["context_data"],
                        metadata=session_data["metadata"],
                        max_age=(
                            timedelta(seconds=session_data["max_age_seconds"])
                            if session_data.get("max_age_seconds")
                            else None
                        ),
                    )
                    self.sessions[sid] = session

                # Load strategy contexts
                for sid, ctx_data in data.get("strategy_contexts", {}).items():
                    context = StrategyContext(
                        strategy_id=ctx_data["strategy_id"],
                        strategy_type=StrategyType(ctx_data["strategy_type"]),
                        session_id=ctx_data["session_id"],
                        created_at=datetime.fromisoformat(ctx_data["created_at"]),
                        last_used=datetime.fromisoformat(ctx_data["last_used"]),
                        parameters=ctx_data["parameters"],
                        performance_metrics=ctx_data["performance_metrics"],
                        status=ctx_data["status"],
                    )
                    self.strategy_contexts[sid] = context

                    # Rebuild active strategies set
                    if context.status == "active":
                        self.active_strategies.add(context.strategy_id)
                        self.strategy_session_map[
                            context.strategy_id
                        ] = context.session_id

                # Load stats
                self.stats.update(data.get("stats", {}))

                logger.info(f"Loaded context data from {filepath}")
                return True

        except Exception as e:
            logger.error(f"Failed to load context data: {e}")
            return False


def create_trading_context_manager(
    max_active_strategies: int = 10,
) -> TradingContextManager:
    """Factory function to create trading context manager."""
    return TradingContextManager(max_active_strategies=max_active_strategies)
