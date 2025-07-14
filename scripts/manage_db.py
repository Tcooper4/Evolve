#!/usr/bin/env python3
"""
Database management script.
Provides commands for managing the application's database, including migrations, backups, and restores.

This script supports:
- Running database migrations
- Creating and restoring database backups
- Managing database connections

Usage:
    python manage_db.py <command> [options]

Commands:
    migrate     Run database migrations
    backup      Create a database backup
    restore     Restore database from backup
    status      Show database status

Examples:
    # Run database migrations
    python manage_db.py migrate

    # Create a database backup
    python manage_db.py backup --output db_backup.sql

    # Restore database from backup
    python manage_db.py restore --input db_backup.sql

    # Show database status
    python manage_db.py status
"""

import argparse
import logging
import logging.config
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis
import yaml


class DatabaseManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the database manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.db_dir = Path("data/db")
        self.backup_dir = self.db_dir / "backups"

        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=self.config["database"]["host"],
            port=self.config["database"]["port"],
            db=self.config["database"]["db"],
            password=self.config["database"]["password"],
            ssl=self.config["database"]["ssl"],
        )

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)

        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)

        logging.config.dictConfig(log_config)

    def backup_database(self, backup_name: Optional[str] = None):
        """Backup database."""
        self.logger.info("Backing up database...")

        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate backup name if not provided
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.backup_dir / f"{backup_name}.rdb"

        try:
            # Save Redis database
            self.redis_client.save()

            # Copy RDB file
            rdb_path = Path(self.config["database"]["rdb_path"])
            if rdb_path.exists():
                shutil.copy2(rdb_path, backup_path)
                self.logger.info(f"Database backed up to {backup_path}")
                return True
            else:
                self.logger.error("RDB file not found")
                return False
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False

    def restore_database(self, backup_name: str):
        """Restore database from backup."""
        self.logger.info(f"Restoring database from {backup_name}...")

        backup_path = self.backup_dir / f"{backup_name}.rdb"
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_path}")
            return False

        try:
            # Stop Redis server
            subprocess.run(["redis-cli", "shutdown"], check=True)

            # Copy backup to RDB location
            rdb_path = Path(self.config["database"]["rdb_path"])
            shutil.copy2(backup_path, rdb_path)

            # Start Redis server
            subprocess.run(["redis-server"], check=True)

            self.logger.info("Database restored successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            return False

    def clear_database(self):
        """Clear all data from database."""
        self.logger.info("Clearing database...")

        try:
            # Flush all data
            self.redis_client.flushall()
            self.logger.info("Database cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear database: {e}")
            return False

    def get_database_info(self):
        """Get database information."""
        self.logger.info("Getting database information...")

        try:
            # Get Redis info
            info = self.redis_client.info()

            # Print information
            print("\nDatabase Information:")
            print(f"Redis Version: {info['redis_version']}")
            print(f"Connected Clients: {info['connected_clients']}")
            print(f"Used Memory: {info['used_memory_human']}")
            print(f"Total Keys: {info['db0']['keys']}")
            print(f"Last Save Time: {datetime.fromtimestamp(info['last_save_time'])}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to get database information: {e}")
            return False

    def optimize_database(self):
        """Optimize database."""
        self.logger.info("Optimizing database...")

        try:
            # Run Redis optimization commands
            self.redis_client.bgrewriteaof()
            self.redis_client.bgsave()

            self.logger.info("Database optimization completed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize database: {e}")
            return False

    def check_database_health(self):
        """Check database health."""
        self.logger.info("Checking database health...")

        try:
            # Check connection
            if not self.redis_client.ping():
                self.logger.error("Database connection failed")
                return False

            # Get memory usage
            info = self.redis_client.info()
            memory_usage = (
                info["used_memory"] / info["maxmemory"] * 100
                if info["maxmemory"] > 0
                else 0
            )

            # Check memory usage
            if memory_usage > 80:
                self.logger.warning(f"High memory usage: {memory_usage:.1f}%")

            # Check connected clients
            if info["connected_clients"] > 100:
                self.logger.warning(
                    f"High number of connected clients: {info['connected_clients']}"
                )

            self.logger.info("Database health check completed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to check database health: {e}")
            return False

    def monitor_database(self):
        """Monitor database in real-time."""
        self.logger.info("Monitoring database...")

        try:
            # Start Redis monitor
            pubsub = self.redis_client.pubsub()
            pubsub.psubscribe("__keyspace@*__:*")

            print("Monitoring database (press Ctrl+C to stop)...")
            for message in pubsub.listen():
                if message["type"] == "pmessage":
                    print(f"{datetime.now()}: {message['data']}")

            return True
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to monitor database: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Database Manager")
    parser.add_argument(
        "command",
        choices=["backup", "restore", "clear", "info", "optimize", "health", "monitor"],
        help="Command to execute",
    )
    parser.add_argument("--backup-name", help="Name of backup to create or restore")

    args = parser.parse_args()
    manager = DatabaseManager()

    commands = {
        "backup": lambda: manager.backup_database(args.backup_name),
        "restore": lambda: manager.restore_database(args.backup_name)
        if args.backup_name
        else False,
        "clear": manager.clear_database,
        "info": manager.get_database_info,
        "optimize": manager.optimize_database,
        "health": manager.check_database_health,
        "monitor": manager.monitor_database,
    }

    if args.command in commands:
        if args.command == "restore" and not args.backup_name:
            parser.error("restore requires --backup-name")

        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
