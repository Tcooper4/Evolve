"""
Centralized persistent MemoryStore (SQLite) for the Evolve agent layer.

AGENT_MEMORY_LAYER:
- Provides shared context across all agents (not request-scoped only).
- Uses SQLAlchemy + SQLite with NullPool (P2.1) to avoid concurrent write pool issues.
- Designed to be safe on shutdown (P4.2 / P3P4_FIXES.md pattern via atexit hooks in main/app).

Memory types supported:
- short_term: current session context (cleared by session or explicitly)
- long_term: trading history, backtests, outcomes, model performance
- preference: user preferences (risk tolerance, favorite strategies, style)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Index, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

Base = declarative_base()


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PREFERENCE = "preference"


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id = Column(String(36), primary_key=True)
    memory_type = Column(String(32), nullable=False, index=True)
    # namespace groups memory per agent/system area (e.g. "CommentaryAgent", "ExecutionAgent", "global")
    namespace = Column(String(128), nullable=False, index=True, default="global")
    # session_id scopes short-term memory to a single run/session
    session_id = Column(String(64), nullable=True, index=True)
    # key allows stable upserts (especially for preferences)
    key = Column(String(256), nullable=True, index=True)
    category = Column(String(64), nullable=True, index=True)
    value_json = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


Index("idx_memory_type_namespace", MemoryEntry.memory_type, MemoryEntry.namespace)


def _default_db_path() -> Path:
    # Keep separate from trading.db so we don't interfere with existing DB lifecycle.
    path = os.getenv("EVOLVE_MEMORY_DB_PATH", "data/memory_store.db")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _create_memory_engine(db_path: Path) -> Engine:
    database_url = f"sqlite:///{db_path.as_posix()}"
    connect_args = {"check_same_thread": False}
    # P2 fix: SQLite uses NullPool; QueuePool can cause unexpected behavior with concurrent writes.
    engine = create_engine(
        database_url,
        connect_args=connect_args,
        poolclass=NullPool,
        pool_pre_ping=True,
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


@dataclass
class MemoryRecord:
    id: str
    memory_type: str
    namespace: str
    session_id: Optional[str]
    key: Optional[str]
    category: Optional[str]
    value: Any
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class MemoryStore:
    """Central shared context store for all agents."""

    def __init__(self, db_path: Optional[Path] = None, session_id: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        self.engine = _create_memory_engine(self.db_path)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables on first init
        Base.metadata.create_all(bind=self.engine)

        self.session_id = session_id or os.getenv("EVOLVE_SESSION_ID") or datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )
        logger.info(f"MemoryStore initialized at {self.db_path} (session_id={self.session_id})")

    def close(self) -> None:
        """Dispose engine; safe to call multiple times."""
        try:
            self.engine.dispose()
        except Exception as e:
            logger.warning(f"MemoryStore dispose failed: {e}")

    def _serialize(self, value: Any) -> str:
        return json.dumps(value, default=str)

    def _deserialize(self, payload: str) -> Any:
        try:
            return json.loads(payload)
        except Exception:
            return payload

    def _to_record(self, entry: MemoryEntry) -> MemoryRecord:
        return MemoryRecord(
            id=entry.id,
            memory_type=entry.memory_type,
            namespace=entry.namespace,
            session_id=entry.session_id,
            key=entry.key,
            category=entry.category,
            value=self._deserialize(entry.value_json),
            metadata=self._deserialize(entry.metadata_json) if entry.metadata_json else {},
            created_at=entry.created_at.isoformat() if entry.created_at else "",
            updated_at=entry.updated_at.isoformat() if entry.updated_at else "",
        )

    def upsert(
        self,
        memory_type: MemoryType,
        namespace: str,
        key: Optional[str],
        value: Any,
        *,
        category: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert or update an entry by (memory_type, namespace, session_id, key). Returns entry id."""
        sid = session_id if session_id is not None else (self.session_id if memory_type == MemoryType.SHORT_TERM else None)
        meta = metadata or {}
        now = datetime.utcnow()
        entry_id = str(uuid.uuid4())

        with self.SessionLocal() as db:  # type: Session
            try:
                q = db.query(MemoryEntry).filter(
                    MemoryEntry.memory_type == memory_type.value,
                    MemoryEntry.namespace == namespace,
                )
                if sid is None:
                    q = q.filter(MemoryEntry.session_id.is_(None))
                else:
                    q = q.filter(MemoryEntry.session_id == sid)
                if key is None:
                    existing = None
                else:
                    existing = q.filter(MemoryEntry.key == key).one_or_none()

                if existing:
                    existing.value_json = self._serialize(value)
                    existing.metadata_json = self._serialize(meta) if meta else None
                    existing.category = category
                    existing.updated_at = now
                    db.add(existing)
                    db.commit()
                    return existing.id

                entry = MemoryEntry(
                    id=entry_id,
                    memory_type=memory_type.value,
                    namespace=namespace,
                    session_id=sid,
                    key=key,
                    category=category,
                    value_json=self._serialize(value),
                    metadata_json=self._serialize(meta) if meta else None,
                    created_at=now,
                    updated_at=now,
                )
                db.add(entry)
                db.commit()
                return entry.id
            except Exception:
                db.rollback()
                raise

    def add(
        self,
        memory_type: MemoryType,
        namespace: str,
        value: Any,
        *,
        key: Optional[str] = None,
        category: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a new entry (no upsert). Returns entry id."""
        sid = session_id if session_id is not None else (self.session_id if memory_type == MemoryType.SHORT_TERM else None)
        meta = metadata or {}
        now = datetime.utcnow()
        entry_id = str(uuid.uuid4())

        with self.SessionLocal() as db:
            try:
                entry = MemoryEntry(
                    id=entry_id,
                    memory_type=memory_type.value,
                    namespace=namespace,
                    session_id=sid,
                    key=key,
                    category=category,
                    value_json=self._serialize(value),
                    metadata_json=self._serialize(meta) if meta else None,
                    created_at=now,
                    updated_at=now,
                )
                db.add(entry)
                db.commit()
                return entry.id
            except Exception:
                db.rollback()
                raise

    def list(
        self,
        memory_type: MemoryType,
        *,
        namespace: Optional[str] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
        newest_first: bool = True,
    ) -> List[MemoryRecord]:
        with self.SessionLocal() as db:
            q = db.query(MemoryEntry).filter(MemoryEntry.memory_type == memory_type.value)
            if namespace:
                q = q.filter(MemoryEntry.namespace == namespace)
            if category:
                q = q.filter(MemoryEntry.category == category)
            if memory_type == MemoryType.SHORT_TERM:
                sid = session_id if session_id is not None else self.session_id
                q = q.filter(MemoryEntry.session_id == sid)
            if newest_first:
                q = q.order_by(MemoryEntry.created_at.desc())
            else:
                q = q.order_by(MemoryEntry.created_at.asc())
            rows = q.offset(offset).limit(limit).all()
            return [self._to_record(r) for r in rows]

    def get(self, entry_id: str) -> Optional[MemoryRecord]:
        with self.SessionLocal() as db:
            row = db.query(MemoryEntry).filter(MemoryEntry.id == entry_id).one_or_none()
            return self._to_record(row) if row else None

    def update_entry(
        self,
        entry_id: str,
        *,
        value: Optional[Any] = None,
        key: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self.SessionLocal() as db:
            try:
                row = db.query(MemoryEntry).filter(MemoryEntry.id == entry_id).one_or_none()
                if not row:
                    return False
                if value is not None:
                    row.value_json = self._serialize(value)
                if key is not None:
                    row.key = key
                if category is not None:
                    row.category = category
                if metadata is not None:
                    row.metadata_json = self._serialize(metadata)
                row.updated_at = datetime.utcnow()
                db.add(row)
                db.commit()
                return True
            except Exception:
                db.rollback()
                raise

    def delete_entry(self, entry_id: str) -> bool:
        with self.SessionLocal() as db:
            try:
                row = db.query(MemoryEntry).filter(MemoryEntry.id == entry_id).one_or_none()
                if not row:
                    return False
                db.delete(row)
                db.commit()
                return True
            except Exception:
                db.rollback()
                raise

    def clear(self, memory_type: MemoryType, *, session_id: Optional[str] = None) -> int:
        """Clear all entries by memory type. For short_term, clears current session by default."""
        with self.SessionLocal() as db:
            try:
                q = db.query(MemoryEntry).filter(MemoryEntry.memory_type == memory_type.value)
                if memory_type == MemoryType.SHORT_TERM:
                    sid = session_id if session_id is not None else self.session_id
                    q = q.filter(MemoryEntry.session_id == sid)
                count = q.count()
                q.delete(synchronize_session=False)
                db.commit()
                return int(count)
            except Exception:
                db.rollback()
                raise

    # --- Preference helpers ---

    def get_preference(self, key: str) -> Optional[Any]:
        """Return the value for a preference key, or None if not set."""
        with self.SessionLocal() as db:
            row = (
                db.query(MemoryEntry)
                .filter(
                    MemoryEntry.memory_type == MemoryType.PREFERENCE.value,
                    MemoryEntry.namespace == "global",
                    MemoryEntry.key == key,
                )
                .one_or_none()
            )
            if not row:
                return None
            return self._deserialize(row.value_json)

    def upsert_preference(self, key: str, value: Any, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        return self.upsert(MemoryType.PREFERENCE, namespace="global", key=key, value=value, category="preference", metadata=metadata)

    def ingest_preference_text(self, text: str, *, source: str = "conversation") -> List[str]:
        """Extract a few common preferences from user text and upsert them. Returns keys updated."""
        updated: List[str] = []
        t = text.lower()

        def _set(k: str, v: Any):
            self.upsert_preference(k, v, metadata={"source": source, "text_preview": text[:200]})
            updated.append(k)

        # Very lightweight extraction; can be upgraded later to LLM-based extraction.
        if "risk averse" in t or "low risk" in t or "conservative" in t:
            _set("risk_tolerance", "low")
        if "high risk" in t or "aggressive" in t:
            _set("risk_tolerance", "high")
        if "medium risk" in t or "moderate risk" in t:
            _set("risk_tolerance", "medium")
        if "day trade" in t or "daytrading" in t:
            _set("trading_style", "day_trading")
        if "swing" in t:
            _set("trading_style", "swing")
        if "long term" in t or "long-term" in t:
            _set("trading_style", "long_term")
        for s in ["rsi", "bollinger", "macd", "moving average", "mean reversion", "momentum"]:
            if s in t:
                _set("favorite_strategy_hint", s)
                break

        return updated


_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Singleton access pattern (mirrors get_agent_manager / AGENT_UPGRADE.md)."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def close_memory_store() -> None:
    """Shutdown hook to dispose memory engine without impacting trading DB."""
    global _memory_store
    if _memory_store is not None:
        try:
            _memory_store.close()
        finally:
            _memory_store = None

