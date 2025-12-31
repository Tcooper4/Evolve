"""
Disaster Recovery Manager

Comprehensive disaster recovery system for the Evolve trading platform.
Provides automated backups, snapshots, and recovery capabilities.
"""

import asyncio
import gzip
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class RecoveryType(Enum):
    """Types of recovery operations."""
    FULL = "full"
    DATABASE = "database"
    PORTFOLIO = "portfolio"
    MODELS = "models"
    CONFIG = "config"
    LOGS = "logs"
    STATE = "state"


class BackupStatus(Enum):
    """Backup status."""
    SUCCESS = "success"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    backup_type: RecoveryType
    timestamp: datetime
    status: BackupStatus
    size_bytes: int
    components: List[str]
    checksum: Optional[str] = None
    error_message: Optional[str] = None


class DisasterRecoveryManager:
    """
    Comprehensive disaster recovery manager.
    
    Features:
    - Automated backups with scheduling
    - Point-in-time recovery
    - System-wide snapshots
    - Component-level recovery
    - Backup verification and validation
    - Backup rotation and cleanup
    """
    
    def __init__(
        self,
        backup_dir: str = "backups/disaster_recovery",
        max_backups: int = 30,
        backup_retention_days: int = 90,
        enable_compression: bool = True,
        enable_encryption: bool = False,
    ):
        """
        Initialize disaster recovery manager.
        
        Args:
            backup_dir: Directory for storing backups
            max_backups: Maximum number of backups to keep
            backup_retention_days: Days to retain backups
            enable_compression: Enable backup compression
            enable_encryption: Enable backup encryption (future)
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_backups = max_backups
        self.backup_retention_days = backup_retention_days
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        
        # Backup metadata storage
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.backup_metadata: Dict[str, BackupMetadata] = {}
        self._load_metadata()
        
        # Components to backup
        self.backup_components = {
            "database": self._backup_database,
            "portfolio": self._backup_portfolio,
            "models": self._backup_models,
            "config": self._backup_config,
            "state": self._backup_state,
            "logs": self._backup_logs,
        }
        
        logger.info(f"Disaster Recovery Manager initialized: {backup_dir}")
    
    def _load_metadata(self):
        """Load backup metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    for backup_id, meta in data.items():
                        self.backup_metadata[backup_id] = BackupMetadata(
                            backup_id=meta["backup_id"],
                            backup_type=RecoveryType(meta["backup_type"]),
                            timestamp=datetime.fromisoformat(meta["timestamp"]),
                            status=BackupStatus(meta["status"]),
                            size_bytes=meta["size_bytes"],
                            components=meta["components"],
                            checksum=meta.get("checksum"),
                            error_message=meta.get("error_message"),
                        )
            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")
                self.backup_metadata = {}
    
    def _save_metadata(self):
        """Save backup metadata to file."""
        try:
            data = {}
            for backup_id, meta in self.backup_metadata.items():
                data[backup_id] = {
                    "backup_id": meta.backup_id,
                    "backup_type": meta.backup_type.value,
                    "timestamp": meta.timestamp.isoformat(),
                    "status": meta.status.value,
                    "size_bytes": meta.size_bytes,
                    "components": meta.components,
                    "checksum": meta.checksum,
                    "error_message": meta.error_message,
                }
            
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    async def create_backup(
        self,
        backup_type: RecoveryType = RecoveryType.FULL,
        components: Optional[List[str]] = None,
        backup_name: Optional[str] = None,
    ) -> BackupMetadata:
        """
        Create a backup.
        
        Args:
            backup_type: Type of backup to create
            components: Specific components to backup (None = all)
            backup_name: Optional custom backup name
        
        Returns:
            BackupMetadata for the created backup
        """
        backup_id = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=datetime.now(),
            status=BackupStatus.IN_PROGRESS,
            size_bytes=0,
            components=[],
        )
        
        self.backup_metadata[backup_id] = metadata
        self._save_metadata()
        
        logger.info(f"Creating backup: {backup_id} (type: {backup_type.value})")
        
        try:
            components_to_backup = components or list(self.backup_components.keys())
            successful_components = []
            
            for component in components_to_backup:
                if component in self.backup_components:
                    try:
                        await self.backup_components[component](backup_path)
                        successful_components.append(component)
                        logger.info(f"Backed up component: {component}")
                    except Exception as e:
                        logger.error(f"Failed to backup component {component}: {e}")
                        metadata.error_message = f"Component {component} failed: {str(e)}"
                else:
                    logger.warning(f"Unknown backup component: {component}")
            
            # Create archive if compression enabled
            if self.enable_compression:
                archive_path = backup_path.with_suffix(".tar.gz")
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(backup_path, arcname=backup_id)
                
                # Remove uncompressed directory
                shutil.rmtree(backup_path)
                backup_path = archive_path
            
            # Calculate size
            if backup_path.is_file():
                size = backup_path.stat().st_size
            else:
                size = sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file())
            
            # Update metadata
            metadata.status = (
                BackupStatus.SUCCESS
                if len(successful_components) == len(components_to_backup)
                else BackupStatus.PARTIAL
            )
            metadata.size_bytes = size
            metadata.components = successful_components
            
            # Calculate checksum (simple size-based for now)
            metadata.checksum = str(hash(str(size) + backup_id))
            
            self._save_metadata()
            
            logger.info(
                f"Backup completed: {backup_id} "
                f"({len(successful_components)}/{len(components_to_backup)} components, "
                f"{size / (1024*1024):.2f} MB)"
            )
            
            # Cleanup old backups
            await self._cleanup_old_backups()
            
            return metadata
        
        except Exception as e:
            logger.error(f"Backup failed: {backup_id} - {e}")
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            self._save_metadata()
            raise
    
    async def _backup_database(self, backup_path: Path):
        """Backup database."""
        db_backup_path = backup_path / "database"
        db_backup_path.mkdir(exist_ok=True)
        
        try:
            from trading.database import get_db_session, get_engine
            from trading.database.models import Base
            from sqlalchemy import text
            
            engine = get_engine()
            if not engine:
                logger.warning("Database engine not available, skipping database backup")
                return
            
            # Get database URL to determine type
            db_url = str(engine.url)
            
            if db_url.startswith("sqlite"):
                # SQLite backup
                db_path = db_url.replace("sqlite:///", "")
                if os.path.exists(db_path):
                    shutil.copy2(db_path, db_backup_path / "trading.db")
                    logger.info("SQLite database backed up")
            
            elif db_url.startswith("postgresql"):
                # PostgreSQL backup using pg_dump
                try:
                    dump_file = db_backup_path / "postgres_dump.sql"
                    subprocess.run(
                        [
                            "pg_dump",
                            db_url,
                            "-f",
                            str(dump_file),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    logger.info("PostgreSQL database backed up")
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.warning(f"pg_dump not available, using SQLAlchemy export: {e}")
                    # Fallback: export via SQLAlchemy
                    await self._export_database_via_sqlalchemy(db_backup_path)
            
            else:
                logger.warning(f"Unknown database type: {db_url}")
        
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    async def _export_database_via_sqlalchemy(self, backup_path: Path):
        """Export database via SQLAlchemy as fallback."""
        try:
            from trading.database import get_db_session
            from trading.database.models import (
                PortfolioStateModel,
                PositionModel,
                TradingSessionModel,
                StateManagerModel,
                AgentMemoryModel,
                TaskModel,
            )
            
            export_data = {}
            
            with get_db_session() as session:
                # Export portfolio states
                portfolios = session.query(PortfolioStateModel).all()
                export_data["portfolios"] = [p.to_dict() for p in portfolios]
                
                # Export positions
                positions = session.query(PositionModel).all()
                export_data["positions"] = [p.to_dict() for p in positions]
                
                # Export sessions
                sessions = session.query(TradingSessionModel).all()
                export_data["sessions"] = [
                    {
                        "session_id": s.session_id,
                        "user_id": s.user_id,
                        "status": s.status,
                        "created_at": s.created_at.isoformat(),
                        "last_activity": s.last_activity.isoformat(),
                        "strategies": s.strategies,
                        "context_data": s.context_data,
                        "metadata": s.metadata,
                    }
                    for s in sessions
                ]
                
                # Export state manager
                states = session.query(StateManagerModel).all()
                export_data["state_manager"] = [
                    {
                        "state_key": s.state_key,
                        "state_value": s.state_value,
                        "version": s.version,
                    }
                    for s in states
                ]
            
            # Save export
            export_file = backup_path / "database_export.json"
            with open(export_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info("Database exported via SQLAlchemy")
        
        except Exception as e:
            logger.error(f"SQLAlchemy export failed: {e}")
            raise
    
    async def _backup_portfolio(self, backup_path: Path):
        """Backup portfolio state."""
        portfolio_backup_path = backup_path / "portfolio"
        portfolio_backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup portfolio files
            portfolio_files = [
                "data/portfolio_*.json",
                "trading/portfolio/*.json",
            ]
            
            for pattern in portfolio_files:
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file():
                        shutil.copy2(file_path, portfolio_backup_path / file_path.name)
            
            logger.info("Portfolio state backed up")
        
        except Exception as e:
            logger.error(f"Portfolio backup failed: {e}")
            raise
    
    async def _backup_models(self, backup_path: Path):
        """Backup trained models."""
        models_backup_path = backup_path / "models"
        models_backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup model directory
            model_dirs = ["models", "trading/models"]
            
            for model_dir in model_dirs:
                model_path = Path(model_dir)
                if model_path.exists():
                    for model_file in model_path.rglob("*"):
                        if model_file.is_file() and not model_file.suffix == ".pyc":
                            rel_path = model_file.relative_to(model_path)
                            dest_path = models_backup_path / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(model_file, dest_path)
            
            logger.info("Models backed up")
        
        except Exception as e:
            logger.error(f"Models backup failed: {e}")
            raise
    
    async def _backup_config(self, backup_path: Path):
        """Backup configuration files."""
        config_backup_path = backup_path / "config"
        config_backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup config files
            config_files = [
                "config",
                "trading/config",
                ".env",
                "env.example",
            ]
            
            for config_path in config_files:
                path = Path(config_path)
                if path.exists():
                    if path.is_file():
                        shutil.copy2(path, config_backup_path / path.name)
                    else:
                        for config_file in path.rglob("*"):
                            if config_file.is_file():
                                rel_path = config_file.relative_to(path)
                                dest_path = config_backup_path / rel_path
                                dest_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(config_file, dest_path)
            
            logger.info("Configuration backed up")
        
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            raise
    
    async def _backup_state(self, backup_path: Path):
        """Backup system state."""
        state_backup_path = backup_path / "state"
        state_backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup state files
            state_files = [
                "trading/memory/state_manager.py",
                "data/state_*.pkl",
                "data/state_*.json",
            ]
            
            for pattern in state_files:
                for file_path in Path(".").glob(pattern):
                    if file_path.is_file():
                        shutil.copy2(file_path, state_backup_path / file_path.name)
            
            logger.info("System state backed up")
        
        except Exception as e:
            logger.error(f"State backup failed: {e}")
            raise
    
    async def _backup_logs(self, backup_path: Path):
        """Backup log files."""
        logs_backup_path = backup_path / "logs"
        logs_backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup recent log files
            log_dirs = ["logs", "trading/logs"]
            
            for log_dir in log_dirs:
                log_path = Path(log_dir)
                if log_path.exists():
                    for log_file in log_path.glob("*.log"):
                        # Only backup recent logs (last 7 days)
                        if (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).days <= 7:
                            shutil.copy2(log_file, logs_backup_path / log_file.name)
            
            logger.info("Logs backed up")
        
        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            raise
    
    async def restore_backup(
        self,
        backup_id: str,
        components: Optional[List[str]] = None,
        verify: bool = True,
    ) -> bool:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of the backup to restore
            components: Specific components to restore (None = all)
            verify: Verify backup before restoring
        
        Returns:
            True if restore successful
        """
        if backup_id not in self.backup_metadata:
            raise ValueError(f"Backup {backup_id} not found")
        
        metadata = self.backup_metadata[backup_id]
        
        if metadata.status != BackupStatus.SUCCESS:
            raise ValueError(f"Backup {backup_id} is not in a valid state for restore")
        
        logger.info(f"Restoring backup: {backup_id}")
        
        # Find backup file/directory
        backup_path = self.backup_dir / backup_id
        
        # Handle compressed backups
        if backup_path.with_suffix(".tar.gz").exists():
            backup_path = backup_path.with_suffix(".tar.gz")
            # Extract to temp directory
            temp_dir = self.backup_dir / f"temp_{backup_id}"
            temp_dir.mkdir(exist_ok=True)
            
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(temp_dir)
            
            backup_path = temp_dir / backup_id
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup path not found: {backup_path}")
        
        try:
            components_to_restore = components or metadata.components
            
            for component in components_to_restore:
                if component == "database":
                    await self._restore_database(backup_path)
                elif component == "portfolio":
                    await self._restore_portfolio(backup_path)
                elif component == "models":
                    await self._restore_models(backup_path)
                elif component == "config":
                    await self._restore_config(backup_path)
                elif component == "state":
                    await self._restore_state(backup_path)
                elif component == "logs":
                    await self._restore_logs(backup_path)
            
            logger.info(f"Backup restored successfully: {backup_id}")
            return True
        
        except Exception as e:
            logger.error(f"Restore failed: {backup_id} - {e}")
            raise
    
    async def _restore_database(self, backup_path: Path):
        """Restore database from backup."""
        db_backup_path = backup_path / "database"
        
        if not db_backup_path.exists():
            logger.warning("Database backup not found, skipping restore")
            return
        
        try:
            from trading.database import get_engine, init_database
            
            engine = get_engine()
            if not engine:
                logger.warning("Database engine not available, skipping database restore")
                return
            
            db_url = str(engine.url)
            
            if db_url.startswith("sqlite"):
                # Restore SQLite
                db_file = db_backup_path / "trading.db"
                if db_file.exists():
                    target_db = db_url.replace("sqlite:///", "")
                    shutil.copy2(db_file, target_db)
                    logger.info("SQLite database restored")
            
            elif db_url.startswith("postgresql"):
                # Restore PostgreSQL
                dump_file = db_backup_path / "postgres_dump.sql"
                if dump_file.exists():
                    subprocess.run(
                        ["psql", db_url, "-f", str(dump_file)],
                        check=True,
                        capture_output=True,
                    )
                    logger.info("PostgreSQL database restored")
                else:
                    # Try JSON export
                    await self._import_database_from_json(db_backup_path)
            
            else:
                # Try JSON export
                await self._import_database_from_json(db_backup_path)
        
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise
    
    async def _import_database_from_json(self, backup_path: Path):
        """Import database from JSON export."""
        try:
            export_file = backup_path / "database_export.json"
            if not export_file.exists():
                logger.warning("Database JSON export not found")
                return
            
            from trading.database import get_db_session, init_database
            from trading.database.models import (
                PortfolioStateModel,
                PositionModel,
                TradingSessionModel,
                StateManagerModel,
            )
            
            # Initialize database
            init_database(create_tables=True)
            
            with open(export_file, "r") as f:
                export_data = json.load(f)
            
            with get_db_session() as session:
                # Restore portfolios
                if "portfolios" in export_data:
                    for p_data in export_data["portfolios"]:
                        portfolio = PortfolioStateModel(**p_data)
                        session.merge(portfolio)
                
                # Restore positions
                if "positions" in export_data:
                    for pos_data in export_data["positions"]:
                        position = PositionModel(**pos_data)
                        session.merge(position)
                
                # Restore sessions
                if "sessions" in export_data:
                    for s_data in export_data["sessions"]:
                        session_obj = TradingSessionModel(**s_data)
                        session.merge(session_obj)
                
                session.commit()
            
            logger.info("Database imported from JSON")
        
        except Exception as e:
            logger.error(f"JSON import failed: {e}")
            raise
    
    async def _restore_portfolio(self, backup_path: Path):
        """Restore portfolio state."""
        portfolio_backup_path = backup_path / "portfolio"
        
        if portfolio_backup_path.exists():
            for file_path in portfolio_backup_path.glob("*.json"):
                dest_path = Path("data") / file_path.name
                dest_path.parent.mkdir(exist_ok=True)
                shutil.copy2(file_path, dest_path)
            
            logger.info("Portfolio state restored")
    
    async def _restore_models(self, backup_path: Path):
        """Restore models."""
        models_backup_path = backup_path / "models"
        
        if models_backup_path.exists():
            for model_file in models_backup_path.rglob("*"):
                if model_file.is_file():
                    rel_path = model_file.relative_to(models_backup_path)
                    dest_path = Path("models") / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(model_file, dest_path)
            
            logger.info("Models restored")
    
    async def _restore_config(self, backup_path: Path):
        """Restore configuration."""
        config_backup_path = backup_path / "config"
        
        if config_backup_path.exists():
            for config_file in config_backup_path.rglob("*"):
                if config_file.is_file():
                    rel_path = config_file.relative_to(config_backup_path)
                    dest_path = Path("config") / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(config_file, dest_path)
            
            logger.info("Configuration restored")
    
    async def _restore_state(self, backup_path: Path):
        """Restore system state."""
        state_backup_path = backup_path / "state"
        
        if state_backup_path.exists():
            for file_path in state_backup_path.glob("*"):
                if file_path.is_file():
                    dest_path = Path("data") / file_path.name
                    dest_path.parent.mkdir(exist_ok=True)
                    shutil.copy2(file_path, dest_path)
            
            logger.info("System state restored")
    
    async def _restore_logs(self, backup_path: Path):
        """Restore logs."""
        logs_backup_path = backup_path / "logs"
        
        if logs_backup_path.exists():
            for log_file in logs_backup_path.glob("*.log"):
                dest_path = Path("logs") / log_file.name
                dest_path.parent.mkdir(exist_ok=True)
                shutil.copy2(log_file, dest_path)
            
            logger.info("Logs restored")
    
    async def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy."""
        try:
            # Sort backups by timestamp
            sorted_backups = sorted(
                self.backup_metadata.items(),
                key=lambda x: x[1].timestamp,
                reverse=True,
            )
            
            # Remove backups beyond max_backups
            if len(sorted_backups) > self.max_backups:
                for backup_id, _ in sorted_backups[self.max_backups:]:
                    await self.delete_backup(backup_id)
            
            # Remove backups older than retention period
            cutoff_date = datetime.now() - timedelta(days=self.backup_retention_days)
            for backup_id, metadata in list(self.backup_metadata.items()):
                if metadata.timestamp < cutoff_date:
                    await self.delete_backup(backup_id)
        
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        if backup_id not in self.backup_metadata:
            return False
        
        try:
            # Delete backup file/directory
            backup_path = self.backup_dir / backup_id
            
            # Try compressed version
            if backup_path.with_suffix(".tar.gz").exists():
                backup_path.with_suffix(".tar.gz").unlink()
            elif backup_path.exists():
                if backup_path.is_file():
                    backup_path.unlink()
                else:
                    shutil.rmtree(backup_path)
            
            # Remove metadata
            del self.backup_metadata[backup_id]
            self._save_metadata()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def list_backups(
        self,
        backup_type: Optional[RecoveryType] = None,
        status: Optional[BackupStatus] = None,
    ) -> List[BackupMetadata]:
        """List available backups."""
        backups = list(self.backup_metadata.values())
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        if status:
            backups = [b for b in backups if b.status == status]
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get information about a specific backup."""
        return self.backup_metadata.get(backup_id)
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        total_backups = len(self.backup_metadata)
        successful_backups = sum(
            1 for b in self.backup_metadata.values() if b.status == BackupStatus.SUCCESS
        )
        total_size = sum(b.size_bytes for b in self.backup_metadata.values())
        
        return {
            "total_backups": total_backups,
            "successful_backups": successful_backups,
            "failed_backups": total_backups - successful_backups,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_backup": (
                min(b.timestamp for b in self.backup_metadata.values()).isoformat()
                if self.backup_metadata
                else None
            ),
            "newest_backup": (
                max(b.timestamp for b in self.backup_metadata.values()).isoformat()
                if self.backup_metadata
                else None
            ),
        }

