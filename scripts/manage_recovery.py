#!/usr/bin/env python3
"""
Disaster recovery management script.
Provides commands for managing system recovery and resilience.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import psutil
import boto3
import kubernetes
from kubernetes import client, config
import docker
import redis
import requests
import socket
import subprocess
import shutil
import hashlib
import tarfile
import zipfile

class RecoveryManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the recovery manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_dir = Path("recovery")
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

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

    async def create_recovery_point(self, components: List[str] = None):
        """Create a system recovery point."""
        self.logger.info("Creating system recovery point")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recovery_point = {
                "timestamp": timestamp,
                "components": {},
                "system_state": {},
                "checksums": {}
            }
            
            # Backup components
            if components is None:
                components = ["config", "data", "database", "logs", "models"]
            
            for component in components:
                if component == "config":
                    recovery_point["components"]["config"] = await self._backup_config()
                elif component == "data":
                    recovery_point["components"]["data"] = await self._backup_data()
                elif component == "database":
                    recovery_point["components"]["database"] = await self._backup_database()
                elif component == "logs":
                    recovery_point["components"]["logs"] = await self._backup_logs()
                elif component == "models":
                    recovery_point["components"]["models"] = await self._backup_models()
            
            # Capture system state
            recovery_point["system_state"] = await self._capture_system_state()
            
            # Calculate checksums
            recovery_point["checksums"] = await self._calculate_checksums(recovery_point)
            
            # Save recovery point
            recovery_file = self.recovery_dir / f"recovery_point_{timestamp}.json"
            with open(recovery_file, "w") as f:
                json.dump(recovery_point, f, indent=2)
            
            # Create backup archive
            archive_file = self.backup_dir / f"recovery_point_{timestamp}.tar.gz"
            with tarfile.open(archive_file, "w:gz") as tar:
                tar.add(self.recovery_dir / f"recovery_point_{timestamp}.json")
                for component, paths in recovery_point["components"].items():
                    for path in paths:
                        tar.add(path)
            
            self.logger.info(f"Recovery point created: {recovery_file}")
            self.logger.info(f"Backup archive created: {archive_file}")
            
            return recovery_point
        except Exception as e:
            self.logger.error(f"Failed to create recovery point: {e}")
            raise

    async def restore_from_point(self, recovery_point: str):
        """Restore system from recovery point."""
        self.logger.info(f"Restoring from recovery point: {recovery_point}")
        
        try:
            # Load recovery point
            with open(recovery_point) as f:
                point = json.load(f)
            
            # Verify checksums
            if not await self._verify_checksums(point):
                raise ValueError("Recovery point checksum verification failed")
            
            # Restore components
            for component, paths in point["components"].items():
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
            if not await self._verify_restoration(point):
                raise ValueError("Restoration verification failed")
            
            self.logger.info("System restored successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore from recovery point: {e}")
            raise

    async def check_system_health(self):
        """Check system health and resilience."""
        self.logger.info("Checking system health")
        
        try:
            health_check = {
                "timestamp": datetime.now().isoformat(),
                "components": {},
                "system_metrics": {},
                "recommendations": []
            }
            
            # Check application components
            health_check["components"] = await self._check_components()
            
            # Check system metrics
            health_check["system_metrics"] = await self._check_system_metrics()
            
            # Generate recommendations
            health_check["recommendations"] = await self._generate_recommendations(health_check)
            
            # Save health check
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            health_file = self.recovery_dir / f"health_check_{timestamp}.json"
            
            with open(health_file, "w") as f:
                json.dump(health_check, f, indent=2)
            
            # Print health check results
            self._print_health_check(health_check)
            
            return health_check
        except Exception as e:
            self.logger.error(f"Failed to check system health: {e}")
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
                password=self.config["database"]["password"]
            )
            
            # Create backup
            backup_path = self.backup_dir / f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rdb"
            redis_client.save()
            shutil.copy2(
                Path(self.config["database"]["rdb_path"]),
                backup_path
            )
            
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

    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture system state."""
        try:
            return {
                "cpu": psutil.cpu_percent(interval=1),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": dict(psutil.disk_usage("/")._asdict()),
                "network": dict(psutil.net_io_counters()._asdict()),
                "processes": len(psutil.pids()),
                "hostname": socket.gethostname(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to capture system state: {e}")
            raise

    async def _calculate_checksums(self, recovery_point: Dict[str, Any]) -> Dict[str, str]:
        """Calculate checksums for recovery point files."""
        try:
            checksums = {}
            
            # Calculate checksum for recovery point file
            recovery_file = self.recovery_dir / f"recovery_point_{recovery_point['timestamp']}.json"
            with open(recovery_file, "rb") as f:
                checksums["recovery_point"] = hashlib.sha256(f.read()).hexdigest()
            
            # Calculate checksums for component files
            for component, paths in recovery_point["components"].items():
                for path in paths:
                    with open(path, "rb") as f:
                        checksums[path] = hashlib.sha256(f.read()).hexdigest()
            
            return checksums
        except Exception as e:
            self.logger.error(f"Failed to calculate checksums: {e}")
            raise

    async def _verify_checksums(self, recovery_point: Dict[str, Any]) -> bool:
        """Verify checksums of recovery point files."""
        try:
            current_checksums = await self._calculate_checksums(recovery_point)
            return current_checksums == recovery_point["checksums"]
        except Exception as e:
            self.logger.error(f"Failed to verify checksums: {e}")
            raise

    async def _verify_restoration(self, recovery_point: Dict[str, Any]) -> bool:
        """Verify system restoration."""
        try:
            # Check if all components are restored
            for component, paths in recovery_point["components"].items():
                for path in paths:
                    if not Path(path).exists():
                        return False
            
            # Check system state
            current_state = await self._capture_system_state()
            if not self._compare_system_states(current_state, recovery_point["system_state"]):
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to verify restoration: {e}")
            raise

    def _compare_system_states(self, current: Dict[str, Any], original: Dict[str, Any]) -> bool:
        """Compare system states."""
        try:
            # Compare critical metrics
            metrics = ["cpu", "memory", "disk", "network"]
            for metric in metrics:
                if abs(current[metric] - original[metric]) > 10:  # 10% threshold
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to compare system states: {e}")
            raise

    async def _check_components(self) -> Dict[str, Any]:
        """Check health of system components."""
        try:
            components = {}
            
            # Check application
            try:
                response = requests.get("http://localhost:8000/health")
                components["application"] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
            except:
                components["application"] = {
                    "status": "unhealthy",
                    "error": "Application not responding"
                }
            
            # Check database
            try:
                redis_client = redis.Redis(
                    host=self.config["database"]["host"],
                    port=self.config["database"]["port"],
                    db=self.config["database"]["db"],
                    password=self.config["database"]["password"]
                )
                redis_client.ping()
                components["database"] = {
                    "status": "healthy",
                    "connected_clients": redis_client.client_list()
                }
            except:
                components["database"] = {
                    "status": "unhealthy",
                    "error": "Database not responding"
                }
            
            # Check file system
            try:
                disk_usage = psutil.disk_usage("/")
                components["filesystem"] = {
                    "status": "healthy" if disk_usage.percent < 90 else "warning",
                    "usage": disk_usage.percent
                }
            except:
                components["filesystem"] = {
                    "status": "unhealthy",
                    "error": "Failed to check filesystem"
                }
            
            return components
        except Exception as e:
            self.logger.error(f"Failed to check components: {e}")
            raise

    async def _check_system_metrics(self) -> Dict[str, Any]:
        """Check system metrics."""
        try:
            return {
                "cpu": {
                    "usage": psutil.cpu_percent(interval=1),
                    "load": psutil.getloadavg()
                },
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": dict(psutil.disk_usage("/")._asdict()),
                "network": dict(psutil.net_io_counters()._asdict()),
                "processes": {
                    "total": len(psutil.pids()),
                    "zombie": len([p for p in psutil.pids() if psutil.Process(p).status() == "zombie"])
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to check system metrics: {e}")
            raise

    async def _generate_recommendations(self, health_check: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on health check."""
        try:
            recommendations = []
            
            # Check CPU usage
            if health_check["system_metrics"]["cpu"]["usage"] > 80:
                recommendations.append({
                    "component": "cpu",
                    "issue": "High CPU usage",
                    "recommendation": "Consider scaling up or optimizing CPU-intensive operations"
                })
            
            # Check memory usage
            if health_check["system_metrics"]["memory"]["percent"] > 80:
                recommendations.append({
                    "component": "memory",
                    "issue": "High memory usage",
                    "recommendation": "Consider increasing memory or optimizing memory usage"
                })
            
            # Check disk usage
            if health_check["system_metrics"]["disk"]["percent"] > 80:
                recommendations.append({
                    "component": "disk",
                    "issue": "High disk usage",
                    "recommendation": "Consider cleaning up old files or increasing disk space"
                })
            
            # Check application response time
            if health_check["components"]["application"]["response_time"] > 1:
                recommendations.append({
                    "component": "application",
                    "issue": "Slow response time",
                    "recommendation": "Consider optimizing application performance"
                })
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            raise

    def _print_health_check(self, health_check: Dict[str, Any]):
        """Print health check results."""
        print("\nSystem Health Check Results:")
        print(f"\nTimestamp: {health_check['timestamp']}")
        
        print("\nComponents:")
        for component, status in health_check["components"].items():
            print(f"\n{component.title()}:")
            for key, value in status.items():
                print(f"  {key}: {value}")
        
        print("\nSystem Metrics:")
        for metric, value in health_check["system_metrics"].items():
            print(f"\n{metric.title()}:")
            if isinstance(value, dict):
                for key, val in value.items():
                    print(f"  {key}: {val}")
            else:
                print(f"  {value}")
        
        if health_check["recommendations"]:
            print("\nRecommendations:")
            for rec in health_check["recommendations"]:
                print(f"\n{rec['component'].title()}:")
                print(f"  Issue: {rec['issue']}")
                print(f"  Recommendation: {rec['recommendation']}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Recovery Manager")
    parser.add_argument(
        "command",
        choices=["create", "restore", "health"],
        help="Command to execute"
    )
    parser.add_argument(
        "--components",
        nargs="+",
        help="Components to include in recovery point"
    )
    parser.add_argument(
        "--recovery-point",
        help="Recovery point to restore from"
    )
    
    args = parser.parse_args()
    manager = RecoveryManager()
    
    commands = {
        "create": lambda: asyncio.run(
            manager.create_recovery_point(args.components)
        ),
        "restore": lambda: asyncio.run(
            manager.restore_from_point(args.recovery_point)
        ),
        "health": lambda: asyncio.run(
            manager.check_system_health()
        )
    }
    
    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 