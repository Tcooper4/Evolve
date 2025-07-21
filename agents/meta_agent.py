"""
Meta Agent with Memory Store

This module provides a meta-agent that tracks and manages the performance
of upgraded models using a persistent memory store (JSON/SQLite).
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model upgrade status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class PerformanceMetric(Enum):
    """Performance metrics for models."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MAE = "mae"
    RMSE = "rmse"
    PROFIT_LOSS = "profit_loss"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class ModelPerformance:
    """Model performance record."""

    model_id: str
    model_name: str
    version: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ModelUpgrade:
    """Model upgrade record."""

    upgrade_id: str
    original_model_id: str
    new_model_id: str
    upgrade_type: str  # 'hyperparameter', 'architecture', 'data', 'ensemble'
    timestamp: datetime
    reason: str
    expected_improvement: Dict[str, float]
    actual_improvement: Optional[Dict[str, float]] = None
    status: ModelStatus = ModelStatus.TESTING
    rollback_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelMemory:
    """Model memory record."""

    model_id: str
    model_name: str
    current_version: str
    status: ModelStatus
    created_at: datetime
    last_updated: datetime
    total_upgrades: int = 0
    successful_upgrades: int = 0
    avg_performance: Dict[str, float] = field(default_factory=dict)
    best_performance: Dict[str, float] = field(default_factory=dict)
    worst_performance: Dict[str, float] = field(default_factory=dict)
    performance_history: List[ModelPerformance] = field(default_factory=list)
    upgrade_history: List[ModelUpgrade] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None


class MemoryStore(ABC):
    """Abstract base class for memory stores."""

    @abstractmethod
    def save_model_memory(self, memory: ModelMemory) -> bool:
        """Save model memory to store."""

    @abstractmethod
    def load_model_memory(self, model_id: str) -> Optional[ModelMemory]:
        """Load model memory from store."""

    @abstractmethod
    def list_model_ids(self) -> List[str]:
        """List all model IDs in store."""

    @abstractmethod
    def delete_model_memory(self, model_id: str) -> bool:
        """Delete model memory from store."""

    @abstractmethod
    def save_performance(self, performance: ModelPerformance) -> bool:
        """Save performance record."""

    @abstractmethod
    def get_performance_history(
        self, model_id: str, metric: Optional[str] = None
    ) -> List[ModelPerformance]:
        """Get performance history for a model."""


class JSONMemoryStore(MemoryStore):
    """JSON-based memory store."""

    def __init__(self, file_path: str = "data/meta_agent_memory.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure the JSON file exists with proper structure."""
        if not self.file_path.exists():
            initial_data = {
                "models": {},
                "performance_records": [],
                "upgrade_records": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0",
                },
            }
            with open(self.file_path, "w") as f:
                json.dump(initial_data, f, indent=2, default=str)

    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                # Convert datetime strings back to datetime objects
                for model_data in data.get("models", {}).values():
                    if "created_at" in model_data:
                        model_data["created_at"] = datetime.fromisoformat(
                            model_data["created_at"]
                        )
                    if "last_updated" in model_data:
                        model_data["last_updated"] = datetime.fromisoformat(
                            model_data["last_updated"]
                        )
                return data
        except Exception as e:
            logger.error(f"Error loading JSON memory store: {e}")
            return {
                "models": {},
                "performance_records": [],
                "upgrade_records": [],
                "metadata": {},
            }

    def _save_data(self, data: Dict[str, Any]):
        """Save data to JSON file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving JSON memory store: {e}")
            raise

    def save_model_memory(self, memory: ModelMemory) -> bool:
        """Save model memory to JSON store."""
        try:
            data = self._load_data()
            data["models"][memory.model_id] = asdict(memory)
            self._save_data(data)
            logger.info(f"Saved model memory for {memory.model_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving model memory: {e}")
            return False

    def load_model_memory(self, model_id: str) -> Optional[ModelMemory]:
        """Load model memory from JSON store."""
        try:
            data = self._load_data()
            model_data = data.get("models", {}).get(model_id)
            if model_data:
                # Convert back to ModelMemory object
                memory = ModelMemory(**model_data)
                return memory
            return None
        except Exception as e:
            logger.error(f"Error loading model memory: {e}")
            return None

    def list_model_ids(self) -> List[str]:
        """List all model IDs in JSON store."""
        try:
            data = self._load_data()
            return list(data.get("models", {}).keys())
        except Exception as e:
            logger.error(f"Error listing model IDs: {e}")
            return []

    def delete_model_memory(self, model_id: str) -> bool:
        """Delete model memory from JSON store."""
        try:
            data = self._load_data()
            if model_id in data.get("models", {}):
                del data["models"][model_id]
                self._save_data(data)
                logger.info(f"Deleted model memory for {model_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting model memory: {e}")
            return False

    def save_performance(self, performance: ModelPerformance) -> bool:
        """Save performance record to JSON store."""
        try:
            data = self._load_data()
            data["performance_records"].append(asdict(performance))
            self._save_data(data)
            logger.info(f"Saved performance record for {performance.model_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving performance record: {e}")
            return False

    def get_performance_history(
        self, model_id: str, metric: Optional[str] = None
    ) -> List[ModelPerformance]:
        """Get performance history for a model."""
        try:
            data = self._load_data()
            records = []
            for record_data in data.get("performance_records", []):
                if record_data["model_id"] == model_id:
                    if metric is None or record_data["metric_name"] == metric:
                        records.append(ModelPerformance(**record_data))
            return records
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []


class SQLiteMemoryStore(MemoryStore):
    """SQLite-based memory store."""

    def __init__(self, db_path: str = "data/meta_agent_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create models table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        current_version TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        total_upgrades INTEGER DEFAULT 0,
                        successful_upgrades INTEGER DEFAULT 0,
                        avg_performance TEXT,
                        best_performance TEXT,
                        worst_performance TEXT,
                        tags TEXT,
                        description TEXT
                    )
                """
                )

                # Create performance records table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        confidence_interval TEXT,
                        sample_size INTEGER,
                        context TEXT,
                        FOREIGN KEY (model_id) REFERENCES models (model_id)
                    )
                """
                )

                # Create upgrade records table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS upgrade_records (
                        upgrade_id TEXT PRIMARY KEY,
                        original_model_id TEXT NOT NULL,
                        new_model_id TEXT NOT NULL,
                        upgrade_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        expected_improvement TEXT,
                        actual_improvement TEXT,
                        status TEXT NOT NULL,
                        rollback_reason TEXT,
                        metadata TEXT,
                        FOREIGN KEY (original_model_id) REFERENCES models (model_id),
                        FOREIGN KEY (new_model_id) REFERENCES models (model_id)
                    )
                """
                )

                conn.commit()
                logger.info("SQLite memory store initialized")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise

    def save_model_memory(self, memory: ModelMemory) -> bool:
        """Save model memory to SQLite store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO models
                    (model_id, model_name, current_version, status, created_at, last_updated,
                     total_upgrades, successful_upgrades, avg_performance, best_performance,
                     worst_performance, tags, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        memory.model_id,
                        memory.model_name,
                        memory.current_version,
                        memory.status.value,
                        memory.created_at.isoformat(),
                        memory.last_updated.isoformat(),
                        memory.total_upgrades,
                        memory.successful_upgrades,
                        json.dumps(memory.avg_performance),
                        json.dumps(memory.best_performance),
                        json.dumps(memory.worst_performance),
                        json.dumps(memory.tags),
                        memory.description,
                    ),
                )

                conn.commit()
                logger.info(f"Saved model memory for {memory.model_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving model memory: {e}")
            return False

    def load_model_memory(self, model_id: str) -> Optional[ModelMemory]:
        """Load model memory from SQLite store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT * FROM models WHERE model_id = ?
                """,
                    (model_id,),
                )

                row = cursor.fetchone()
                if row:
                    return ModelMemory(
                        model_id=row[0],
                        model_name=row[1],
                        current_version=row[2],
                        status=ModelStatus(row[3]),
                        created_at=datetime.fromisoformat(row[4]),
                        last_updated=datetime.fromisoformat(row[5]),
                        total_upgrades=row[6],
                        successful_upgrades=row[7],
                        avg_performance=json.loads(row[8]) if row[8] else {},
                        best_performance=json.loads(row[9]) if row[9] else {},
                        worst_performance=json.loads(row[10]) if row[10] else {},
                        tags=json.loads(row[11]) if row[11] else [],
                        description=row[12],
                    )
                return None
        except Exception as e:
            logger.error(f"Error loading model memory: {e}")
            return None

    def list_model_ids(self) -> List[str]:
        """List all model IDs in SQLite store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT model_id FROM models")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing model IDs: {e}")
            return []

    def delete_model_memory(self, model_id: str) -> bool:
        """Delete model memory from SQLite store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                conn.commit()
                logger.info(f"Deleted model memory for {model_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting model memory: {e}")
            return False

    def save_performance(self, performance: ModelPerformance) -> bool:
        """Save performance record to SQLite store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO performance_records
                    (model_id, model_name, version, timestamp, metric_name, metric_value,
                     confidence_interval, sample_size, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        performance.model_id,
                        performance.model_name,
                        performance.version,
                        performance.timestamp.isoformat(),
                        performance.metric_name,
                        performance.metric_value,
                        (
                            json.dumps(performance.confidence_interval)
                            if performance.confidence_interval
                            else None
                        ),
                        performance.sample_size,
                        (
                            json.dumps(performance.context)
                            if performance.context
                            else None
                        ),
                    ),
                )

                conn.commit()
                logger.info(f"Saved performance record for {performance.model_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving performance record: {e}")
            return False

    def get_performance_history(
        self, model_id: str, metric: Optional[str] = None
    ) -> List[ModelPerformance]:
        """Get performance history for a model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if metric:
                    cursor.execute(
                        """
                        SELECT * FROM performance_records
                        WHERE model_id = ? AND metric_name = ?
                        ORDER BY timestamp DESC
                    """,
                        (model_id, metric),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM performance_records
                        WHERE model_id = ?
                        ORDER BY timestamp DESC
                    """,
                        (model_id,),
                    )

                records = []
                for row in cursor.fetchall():
                    records.append(
                        ModelPerformance(
                            model_id=row[1],
                            model_name=row[2],
                            version=row[3],
                            timestamp=datetime.fromisoformat(row[4]),
                            metric_name=row[5],
                            metric_value=row[6],
                            confidence_interval=json.loads(row[7]) if row[7] else None,
                            sample_size=row[8],
                            context=json.loads(row[9]) if row[9] else None,
                        )
                    )

                return records
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []


class MetaAgent:
    """
    Meta agent that tracks and manages model performance using a memory store.
    """

    def __init__(
        self, memory_store: Optional[MemoryStore] = None, store_type: str = "json"
    ):
        """
        Initialize meta agent.

        Args:
            memory_store: Custom memory store instance
            store_type: Type of memory store ('json' or 'sqlite')
        """
        if memory_store:
            self.memory_store = memory_store
        else:
            if store_type == "sqlite":
                self.memory_store = SQLiteMemoryStore()
            else:
                self.memory_store = JSONMemoryStore()

        logger.info(f"MetaAgent initialized with {store_type} memory store")

    def register_model(
        self,
        model_id: str,
        model_name: str,
        version: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Register a new model in the memory store.

        Args:
            model_id: Unique model identifier
            model_name: Human-readable model name
            version: Model version
            description: Model description
            tags: Model tags

        Returns:
            True if registration successful
        """
        try:
            memory = ModelMemory(
                model_id=model_id,
                model_name=model_name,
                current_version=version,
                status=ModelStatus.ACTIVE,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                description=description,
                tags=tags or [],
            )

            return self.memory_store.save_model_memory(memory)
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False

    def record_performance(
        self,
        model_id: str,
        model_name: str,
        version: str,
        metric_name: str,
        metric_value: float,
        confidence_interval: Optional[Tuple[float, float]] = None,
        sample_size: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Record model performance.

        Args:
            model_id: Model identifier
            model_name: Model name
            version: Model version
            metric_name: Performance metric name
            metric_value: Metric value
            confidence_interval: Confidence interval (lower, upper)
            sample_size: Sample size used for evaluation
            context: Additional context

        Returns:
            True if recording successful
        """
        try:
            performance = ModelPerformance(
                model_id=model_id,
                model_name=model_name,
                version=version,
                timestamp=datetime.now(),
                metric_name=metric_name,
                metric_value=metric_value,
                confidence_interval=confidence_interval,
                sample_size=sample_size,
                context=context,
            )

            success = self.memory_store.save_performance(performance)

            if success:
                # Update model memory with latest performance
                self._update_model_performance(model_id, metric_name, metric_value)

            return success
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
            return False

    def _update_model_performance(
        self, model_id: str, metric_name: str, metric_value: float
    ):
        """Update model memory with latest performance metrics."""
        try:
            memory = self.memory_store.load_model_memory(model_id)
            if not memory:
                return

            # Update average performance
            if metric_name not in memory.avg_performance:
                memory.avg_performance[metric_name] = metric_value
            else:
                # Simple moving average
                current_avg = memory.avg_performance[metric_name]
                memory.avg_performance[metric_name] = (current_avg + metric_value) / 2

            # Update best performance
            if (
                metric_name not in memory.best_performance
                or metric_value > memory.best_performance[metric_name]
            ):
                memory.best_performance[metric_name] = metric_value

            # Update worst performance
            if (
                metric_name not in memory.worst_performance
                or metric_value < memory.worst_performance[metric_name]
            ):
                memory.worst_performance[metric_name] = metric_value

            memory.last_updated = datetime.now()
            self.memory_store.save_model_memory(memory)

        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

    def get_model_performance(
        self, model_id: str, metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model performance summary.

        Args:
            model_id: Model identifier
            metric: Specific metric to retrieve

        Returns:
            Performance summary dictionary
        """
        try:
            memory = self.memory_store.load_model_memory(model_id)
            if not memory:
                return {}

            performance_history = self.memory_store.get_performance_history(
                model_id, metric
            )

            return {
                "model_info": {
                    "model_id": memory.model_id,
                    "model_name": memory.model_name,
                    "current_version": memory.current_version,
                    "status": memory.status.value,
                    "total_upgrades": memory.total_upgrades,
                    "successful_upgrades": memory.successful_upgrades,
                    "tags": memory.tags,
                    "description": memory.description,
                },
                "performance_summary": {
                    "avg_performance": memory.avg_performance,
                    "best_performance": memory.best_performance,
                    "worst_performance": memory.worst_performance,
                },
                "performance_history": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "metric_name": p.metric_name,
                        "metric_value": p.metric_value,
                        "version": p.version,
                    }
                    for p in performance_history
                ],
            }
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model summaries
        """
        try:
            model_ids = self.memory_store.list_model_ids()
            models = []

            for model_id in model_ids:
                memory = self.memory_store.load_model_memory(model_id)
                if memory:
                    models.append(
                        {
                            "model_id": memory.model_id,
                            "model_name": memory.model_name,
                            "current_version": memory.current_version,
                            "status": memory.status.value,
                            "total_upgrades": memory.total_upgrades,
                            "successful_upgrades": memory.successful_upgrades,
                            "last_updated": memory.last_updated.isoformat(),
                            "tags": memory.tags,
                        }
                    )

            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_performance_trends(
        self, model_id: str, metric: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get performance trends for a model.

        Args:
            model_id: Model identifier
            metric: Metric name
            days: Number of days to look back

        Returns:
            List of performance records
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            performance_history = self.memory_store.get_performance_history(
                model_id, metric
            )

            # Filter by date
            recent_performance = [
                p for p in performance_history if p.timestamp >= cutoff_date
            ]

            return [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "metric_value": p.metric_value,
                    "version": p.version,
                    "confidence_interval": p.confidence_interval,
                }
                for p in recent_performance
            ]
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return []


def create_meta_agent(
    store_type: str = "json", memory_store: Optional[MemoryStore] = None
) -> MetaAgent:
    """
    Create a meta agent instance.

    Args:
        store_type: Type of memory store ('json' or 'sqlite')
        memory_store: Custom memory store instance

    Returns:
        MetaAgent instance
    """
    return MetaAgent(memory_store=memory_store, store_type=store_type)
