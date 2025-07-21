#!/usr/bin/env python3
"""
Backup management script.
Provides commands for creating, restoring, and managing application backups.

This script supports:
- Creating backups
- Restoring from backups
- Listing and managing backup files

Usage:
    python manage_backup.py <command> [options]

Commands:
    create      Create a backup
    restore     Restore from a backup
    list        List available backups

Examples:
    # Create a backup
    python manage_backup.py create --output backup_2023_01_01.zip

    # Restore from a backup
    python manage_backup.py restore --input backup_2023_01_01.zip

    # List available backups
    python manage_backup.py list
"""

import argparse
import asyncio
import json
import logging
import logging.config
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import croniter
import redis
import yaml

# Azure storage imports
try:
    from azure.storage.blob import BlobServiceClient

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    BlobServiceClient = None

# Google Cloud Storage imports
try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

from utils.launch_utils import setup_logging


class BackupManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the backup manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.schedule_dir = Path("schedules")
        self.schedule_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Set up logging for the service."""
        return setup_logging(service_name="service")

    def create_backup(self, backup_type: str, target: str = "local"):
        """Create backup of specified type."""
        self.logger.info(f"Creating {backup_type} backup...")

        # Create backup directory
        backup_path = (
            self.backup_dir
            / f"{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        backup_path.mkdir(parents=True, exist_ok=True)

        try:
            if backup_type == "data":
                self._backup_data(backup_path)
            elif backup_type == "config":
                self._backup_config(backup_path)
            elif backup_type == "logs":
                self._backup_logs(backup_path)
            elif backup_type == "full":
                self._backup_full(backup_path)
            else:
                self.logger.error(f"Unknown backup type: {backup_type}")
                return False

            self.logger.info(f"Backup created successfully: {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False

    async def restore_backup(self, backup_file: str, target: str = "local"):
        """Restore from backup."""
        self.logger.info(f"Restoring from backup: {backup_file}")

        try:
            # Download backup if not local
            if target != "local":
                backup_file = await self._download_backup(backup_file, target)

            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(self.backup_dir)

            # Load backup info
            info_file = next(self.backup_dir.glob("backup_info_*.json"))
            with open(info_file) as f:
                backup_info = json.load(f)

            # Restore components
            for component, paths in backup_info["components"].items():
                if component == "config":
                    await self._restore_config(paths)
                elif component == "data":
                    await self._restore_data(paths)
                elif component == "database":
                    await self._restore_database(paths)
                elif component == "logs":
                    await self._restore_logs(paths)
                elif component == "models":
                    await self._restore_models(paths)

            # Verify restoration
            if not await self._verify_restoration(backup_info):
                raise ValueError("Restoration verification failed")

            self.logger.info("Backup restored successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            raise

    def schedule_backup(self, backup_type: str, schedule: str, target: str = "local"):
        """Schedule automated backups."""
        self.logger.info(f"Scheduling {backup_type} backups to {target}")

        try:
            # Validate schedule
            if not croniter.is_valid(schedule):
                raise ValueError(f"Invalid schedule format: {schedule}")

            # Create schedule info
            schedule_info = {
                "type": backup_type,
                "schedule": schedule,
                "target": target,
                "last_run": None,
                "next_run": croniter.croniter(schedule, datetime.now())
                .get_next(datetime)
                .isoformat(),
            }

            # Save schedule
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            schedule_file = self.schedule_dir / f"backup_schedule_{timestamp}.json"

            with open(schedule_file, "w") as f:
                json.dump(schedule_info, f, indent=2)

            # Set up schedule
            def backup_job():
                asyncio.run(self.create_backup(backup_type, target))
                schedule_info["last_run"] = datetime.now().isoformat()
                schedule_info["next_run"] = (
                    croniter.croniter(schedule, datetime.now())
                    .get_next(datetime)
                    .isoformat()
                )

                with open(schedule_file, "w") as f:
                    json.dump(schedule_info, f, indent=2)

            return self.logger.info(f"Backup scheduled: {schedule_file}")
            return schedule_info
        except Exception as e:
            self.logger.error(f"Failed to schedule backup: {e}")
            raise

    def list_backups(self, backup_type: Optional[str] = None):
        """List available backups."""
        self.logger.info("Listing backups")

        try:
            backups = []

            # List local backups
            for info_file in self.backup_dir.glob("backup_info_*.json"):
                with open(info_file) as f:
                    backup_info = json.load(f)
                    if backup_type is None or backup_info["type"] == backup_type:
                        backups.append(backup_info)

            # List remote backups
            for target in ["s3", "gcs", "azure"]:
                if target in self.config["backup"]["targets"]:
                    remote_backups = self._list_remote_backups(target)
                    backups.extend(remote_backups)

            # Sort by timestamp
            backups.sort(key=lambda x: x["timestamp"], reverse=True)

            # Print backup list
            self._print_backup_list(backups)

            return backups
        except Exception as e:
            self.logger.error(f"Failed to list backups: {e}")
            raise

    async def _create_full_backup(self) -> Dict[str, List[str]]:
        """Create a full backup."""
        try:
            components = {}

            # Backup config
            components["config"] = await self._backup_config()

            # Backup data
            components["data"] = await self._backup_data()

            # Backup database
            components["database"] = await self._backup_database()

            # Backup logs
            components["logs"] = await self._backup_logs()

            # Backup models
            components["models"] = await self._backup_models()

            return components
        except Exception as e:
            self.logger.error(f"Failed to create full backup: {e}")
            raise

    async def _create_incremental_backup(self) -> Dict[str, List[str]]:
        """Create an incremental backup."""
        try:
            components = {}

            # Get last backup timestamp
            last_backup = self._get_last_backup_time()

            # Backup changed files since last backup
            components["config"] = await self._backup_changed_files(
                "config", last_backup
            )
            components["data"] = await self._backup_changed_files("data", last_backup)
            components["database"] = (
                await self._backup_database()
            )  # Always backup database
            components["logs"] = await self._backup_changed_files("logs", last_backup)
            components["models"] = await self._backup_changed_files(
                "models", last_backup
            )

            return components
        except Exception as e:
            self.logger.error(f"Failed to create incremental backup: {e}")
            raise

    async def _create_differential_backup(self) -> Dict[str, List[str]]:
        """Create a differential backup."""
        try:
            components = {}

            # Get last full backup timestamp
            last_full_backup = self._get_last_full_backup_time()

            # Backup all files changed since last full backup
            components["config"] = await self._backup_changed_files(
                "config", last_full_backup
            )
            components["data"] = await self._backup_changed_files(
                "data", last_full_backup
            )
            components["database"] = (
                await self._backup_database()
            )  # Always backup database
            components["logs"] = await self._backup_changed_files(
                "logs", last_full_backup
            )
            components["models"] = await self._backup_changed_files(
                "models", last_full_backup
            )

            return components
        except Exception as e:
            self.logger.error(f"Failed to create differential backup: {e}")
            raise

    async def _backup_config(self) -> List[str]:
        """Backup configuration files."""
        try:
            config_dir = Path("config")
            backup_paths = []

            for file in config_dir.glob("*.yaml"):
                backup_path = self.backup_dir / f"config_{file.name}"
                shutil.copy2(file, backup_path)
                backup_paths.append(str(backup_path))

            return backup_paths
        except Exception as e:
            self.logger.error(f"Failed to backup config: {e}")
            raise

    async def _backup_data(self) -> List[str]:
        """Backup data files."""
        try:
            data_dir = Path("data")
            backup_paths = []

            for file in data_dir.glob("**/*"):
                if file.is_file():
                    backup_path = self.backup_dir / f"data_{file.name}"
                    shutil.copy2(file, backup_path)
                    backup_paths.append(str(backup_path))

            return backup_paths
        except Exception as e:
            self.logger.error(f"Failed to backup data: {e}")
            raise

    async def _backup_database(self) -> List[str]:
        """Backup database."""
        try:
            # Connect to Redis
            redis_client = redis.Redis(
                host=self.config["database"]["host"],
                port=self.config["database"]["port"],
                db=self.config["database"]["db"],
                password=self.config["database"]["password"],
            )

            # Create backup
            backup_path = (
                self.backup_dir
                / f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rdb"
            )
            redis_client.save()
            shutil.copy2(Path(self.config["database"]["rdb_path"]), backup_path)

            return [str(backup_path)]
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            raise

    async def _backup_logs(self) -> List[str]:
        """Backup log files."""
        try:
            logs_dir = Path("logs")
            backup_paths = []

            for file in logs_dir.glob("*.log"):
                backup_path = self.backup_dir / f"logs_{file.name}"
                shutil.copy2(file, backup_path)
                backup_paths.append(str(backup_path))

            return backup_paths
        except Exception as e:
            self.logger.error(f"Failed to backup logs: {e}")
            raise

    async def _backup_models(self) -> List[str]:
        """Backup model files."""
        try:
            models_dir = Path("models")
            backup_paths = []

            for file in models_dir.glob("**/*"):
                if file.is_file():
                    backup_path = self.backup_dir / f"models_{file.name}"
                    shutil.copy2(file, backup_path)
                    backup_paths.append(str(backup_path))

            return backup_paths
        except Exception as e:
            self.logger.error(f"Failed to backup models: {e}")
            raise

    async def _backup_changed_files(self, component: str, since: datetime) -> List[str]:
        """Backup files changed since specified time."""
        try:
            component_dir = Path(component)
            backup_paths = []

            for file in component_dir.glob("**/*"):
                if file.is_file() and file.stat().st_mtime > since.timestamp():
                    backup_path = self.backup_dir / f"{component}_{file.name}"
                    shutil.copy2(file, backup_path)
                    backup_paths.append(str(backup_path))

            return backup_paths
        except Exception as e:
            self.logger.error(f"Failed to backup changed files: {e}")
            raise

    def _get_last_backup_time(self) -> datetime:
        """Get timestamp of last backup."""
        try:
            backup_files = list(self.backup_dir.glob("backup_info_*.json"))
            if not backup_files:
                return datetime.min

            latest_file = max(backup_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file) as f:
                backup_info = json.load(f)
                return datetime.fromisoformat(backup_info["timestamp"])
        except Exception as e:
            self.logger.error(f"Failed to get last backup time: {e}")
            raise

    def _get_last_full_backup_time(self) -> datetime:
        """Get timestamp of last full backup."""
        try:
            backup_files = list(self.backup_dir.glob("backup_info_*.json"))
            if not backup_files:
                return datetime.min

            full_backups = []
            for file in backup_files:
                with open(file) as f:
                    backup_info = json.load(f)
                    if backup_info["type"] == "full":
                        full_backups.append(
                            datetime.fromisoformat(backup_info["timestamp"])
                        )

            return max(full_backups) if full_backups else datetime.min
        except Exception as e:
            self.logger.error(f"Failed to get last full backup time: {e}")
            raise

    async def _calculate_backup_size(self, components: Dict[str, List[str]]) -> int:
        """Calculate total size of backup."""
        try:
            total_size = 0
            for paths in components.values():
                for path in paths:
                    total_size += Path(path).stat().st_size
            return total_size
        except Exception as e:
            self.logger.error(f"Failed to calculate backup size: {e}")
            raise

    async def _upload_backup(self, backup_file: Path, target: str):
        """Upload backup to target storage."""
        try:
            if target == "s3":
                s3 = boto3.client("s3")
                s3.upload_file(
                    str(backup_file),
                    self.config["backup"]["s3"]["bucket"],
                    f"backups/{backup_file.name}",
                )
            elif target == "gcs":
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.config["backup"]["gcs"]["bucket"])
                blob = bucket.blob(f"backups/{backup_file.name}")
                blob.upload_from_filename(str(backup_file))
            elif target == "azure":
                blob_service_client = BlobServiceClient.from_connection_string(
                    self.config["backup"]["azure"]["connection_string"]
                )
                container_client = blob_service_client.get_container_client(
                    self.config["backup"]["azure"]["container"]
                )
                with open(backup_file, "rb") as data:
                    container_client.upload_blob(f"backups/{backup_file.name}", data)
            else:
                raise ValueError(f"Unsupported target: {target}")
        except Exception as e:
            self.logger.error(f"Failed to upload backup: {e}")
            raise

    async def _download_backup(self, backup_file: str, target: str) -> str:
        """Download backup from target storage."""
        try:
            local_path = self.backup_dir / Path(backup_file).name

            if target == "s3":
                s3 = boto3.client("s3")
                s3.download_file(
                    self.config["backup"]["s3"]["bucket"],
                    f"backups/{backup_file}",
                    str(local_path),
                )
            elif target == "gcs":
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.config["backup"]["gcs"]["bucket"])
                blob = bucket.blob(f"backups/{backup_file}")
                blob.download_to_filename(str(local_path))
            elif target == "azure":
                blob_service_client = BlobServiceClient.from_connection_string(
                    self.config["backup"]["azure"]["connection_string"]
                )
                container_client = blob_service_client.get_container_client(
                    self.config["backup"]["azure"]["container"]
                )
                with open(local_path, "wb") as data:
                    data.write(
                        container_client.download_blob(
                            f"backups/{backup_file}"
                        ).readall()
                    )
            else:
                raise ValueError(f"Unsupported target: {target}")

            return str(local_path)
        except Exception as e:
            self.logger.error(f"Failed to download backup: {e}")
            raise

    def _list_remote_backups(self, target: str) -> List[Dict[str, Any]]:
        """List backups in remote storage."""
        try:
            backups = []

            if target == "s3":
                s3 = boto3.client("s3")
                response = s3.list_objects_v2(
                    Bucket=self.config["backup"]["s3"]["bucket"],
                    Prefix="backups/backup_info_",
                )
                for obj in response.get("Contents", []):
                    info = s3.get_object(
                        Bucket=self.config["backup"]["s3"]["bucket"], Key=obj["Key"]
                    )
                    backups.append(json.loads(info["Body"].read()))

            elif target == "gcs":
                storage_client = storage.Client()
                bucket = storage_client.bucket(self.config["backup"]["gcs"]["bucket"])
                blobs = bucket.list_blobs(prefix="backups/backup_info_")
                for blob in blobs:
                    info = blob.download_as_string()
                    backups.append(json.loads(info))

            elif target == "azure":
                blob_service_client = BlobServiceClient.from_connection_string(
                    self.config["backup"]["azure"]["connection_string"]
                )
                container_client = blob_service_client.get_container_client(
                    self.config["backup"]["azure"]["container"]
                )
                blobs = container_client.list_blobs(
                    name_starts_with="backups/backup_info_"
                )
                for blob in blobs:
                    info = container_client.download_blob(blob.name).readall()
                    backups.append(json.loads(info))

            return backups
        except Exception as e:
            self.logger.error(f"Failed to list remote backups: {e}")
            raise

    def _print_backup_list(self, backups: List[Dict[str, Any]]):
        """Print list of backups."""
        print("\nAvailable Backups:")

        for backup in backups:
            print(f"\nTimestamp: {backup['timestamp']}")
            print(f"Type: {backup['type']}")
            print(f"Target: {backup['target']}")
            print(f"Size: {backup['metadata']['size'] / 1024 / 1024:.2f} MB")
            print("Components:")
            for component, paths in backup["components"].items():
                print(f"  {component}: {len(paths)} files")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Backup Manager")
    parser.add_argument(
        "command",
        choices=["create", "restore", "schedule", "list"],
        help="Command to execute",
    )
    parser.add_argument(
        "--type", choices=["full", "incremental", "differential"], help="Backup type"
    )
    parser.add_argument(
        "--target",
        default="local",
        choices=["local", "s3", "gcs", "azure"],
        help="Backup target",
    )
    parser.add_argument("--schedule", help="Cron schedule for automated backups")
    parser.add_argument("--backup-file", help="Backup file to restore from")

    args = parser.parse_args()
    manager = BackupManager()

    commands = {
        "create": lambda: asyncio.run(manager.create_backup(args.type, args.target)),
        "restore": lambda: asyncio.run(
            manager.restore_backup(args.backup_file, args.target)
        ),
        "schedule": lambda: manager.schedule_backup(
            args.type, args.schedule, args.target
        ),
        "list": lambda: manager.list_backups(args.type),
    }

    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
