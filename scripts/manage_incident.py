#!/usr/bin/env python3
"""
Incident management script.
Provides commands for managing incidents, reporting, and resolution tracking.

This script supports:
- Reporting incidents
- Viewing and updating incident status
- Exporting incident reports

Usage:
    python manage_incident.py <command> [options]

Commands:
    report      Report a new incident
    status      View or update incident status
    export      Export incident reports

Examples:
    # Report a new incident
    python manage_incident.py report --description "API outage"

    # View incident status
    python manage_incident.py status --incident-id 123

    # Export incident reports
    python manage_incident.py export --output incidents.json
"""

import argparse
import asyncio
import json
import logging
import logging.config
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psutil
import redis
import requests
import yaml


class IncidentManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the incident manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.incidents_dir = Path("incidents")
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir = Path("responses")
        self.responses_dir.mkdir(parents=True, exist_ok=True)

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

    async def monitor_incidents(self, duration: int = 300):
        """Monitor for incidents."""
        self.logger.info(f"Monitoring for incidents for {duration} seconds")

        try:
            incidents = []
            start_time = time.time()

            while time.time() - start_time < duration:
                # Check system health
                system_health = await self._check_system_health()
                if not system_health["healthy"]:
                    incident = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "system_health",
                        "severity": "high",
                        "details": system_health["issues"],
                    }
                    incidents.append(incident)
                    await self._handle_incident(incident)

                # Check application health
                app_health = await self._check_application_health()
                if not app_health["healthy"]:
                    incident = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "application_health",
                        "severity": "high",
                        "details": app_health["issues"],
                    }
                    incidents.append(incident)
                    await self._handle_incident(incident)

                # Check database health
                db_health = await self._check_database_health()
                if not db_health["healthy"]:
                    incident = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "database_health",
                        "severity": "critical",
                        "details": db_health["issues"],
                    }
                    incidents.append(incident)
                    await self._handle_incident(incident)

                # Check network health
                network_health = await self._check_network_health()
                if not network_health["healthy"]:
                    incident = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "network_health",
                        "severity": "high",
                        "details": network_health["issues"],
                    }
                    incidents.append(incident)
                    await self._handle_incident(incident)

                # Check security
                security_health = await self._check_security_health()
                if not security_health["healthy"]:
                    incident = {
                        "timestamp": datetime.now().isoformat(),
                        "type": "security_health",
                        "severity": "critical",
                        "details": security_health["issues"],
                    }
                    incidents.append(incident)
                    await self._handle_incident(incident)

                await asyncio.sleep(1)

            # Save incidents
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            incidents_file = self.incidents_dir / f"incidents_{timestamp}.json"

            with open(incidents_file, "w") as f:
                json.dump(incidents, f, indent=2)

            self.logger.info(f"Incidents saved to {incidents_file}")
            return incidents
        except Exception as e:
            self.logger.error(f"Failed to monitor incidents: {e}")
            raise

    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            issues = []

            # Check CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")

            # Check disk usage
            disk = psutil.disk_usage("/")
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")

            return {"healthy": len(issues) == 0, "issues": issues}
        except Exception as e:
            self.logger.error(f"Failed to check system health: {e}")
            raise

    async def _check_application_health(self) -> Dict[str, Any]:
        """Check application health."""
        try:
            issues = []

            # Check application processes
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                try:
                    if proc.info["name"] in self.config["app"]["processes"]:
                        if proc.info["cpu_percent"] > 90:
                            issues.append(
                                f"High CPU usage for {proc.info['name']}: {proc.info['cpu_percent']}%"
                            )
                        if proc.info["memory_percent"] > 90:
                            issues.append(
                                f"High memory usage for {proc.info['name']}: {proc.info['memory_percent']}%"
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    issues.append(f"Process {proc.info['name']} not accessible")

            # Check application logs
            log_dir = Path("logs")
            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_size > 100 * 1024 * 1024:  # 100 MB
                    issues.append(f"Large log file: {log_file}")

            return {"healthy": len(issues) == 0, "issues": issues}
        except Exception as e:
            self.logger.error(f"Failed to check application health: {e}")
            raise

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            issues = []

            # Connect to Redis
            redis_client = redis.Redis(
                host=self.config["database"]["host"],
                port=self.config["database"]["port"],
                db=self.config["database"]["db"],
                password=self.config["database"]["password"],
            )

            # Check Redis connection
            try:
                redis_client.ping()
            except redis.ConnectionError:
                issues.append("Redis connection failed")

            # Check Redis memory usage
            info = redis_client.info()
            if info["used_memory"] > info["maxmemory"] * 0.9:
                issues.append(
                    f"High Redis memory usage: {info['used_memory'] / info['maxmemory'] * 100}%"
                )

            return {"healthy": len(issues) == 0, "issues": issues}
        except Exception as e:
            self.logger.error(f"Failed to check database health: {e}")
            raise

    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network health."""
        try:
            issues = []

            # Check network interfaces
            net_io = psutil.net_io_counters()
            if net_io.errin > 0 or net_io.errout > 0:
                issues.append(f"Network errors: in={net_io.errin}, out={net_io.errout}")

            # Check API endpoints
            for endpoint in self.config["api"]["endpoints"]:
                try:
                    response = requests.get(endpoint["url"], timeout=5)
                    if response.status_code != 200:
                        issues.append(
                            f"API endpoint {endpoint['url']} returned {response.status_code}"
                        )
                except requests.RequestException as e:
                    issues.append(f"API endpoint {endpoint['url']} failed: {e}")

            return {"healthy": len(issues) == 0, "issues": issues}
        except Exception as e:
            self.logger.error(f"Failed to check network health: {e}")
            raise

    async def _check_security_health(self) -> Dict[str, Any]:
        """Check security health."""
        try:
            issues = []

            # Check file permissions
            for file in Path(".").rglob("*"):
                if file.is_file():
                    if file.stat().st_mode & 0o777 == 0o777:
                        issues.append(f"Insecure file permissions: {file}")

            # Check for sensitive data
            sensitive_patterns = ["password", "secret", "key", "token"]
            for file in Path(".").rglob("*"):
                if file.is_file():
                    try:
                        content = file.read_text()
                        for pattern in sensitive_patterns:
                            if pattern in content.lower():
                                issues.append(f"Sensitive data found in {file}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not read file {file}: {e}")
                        continue

            return {"healthy": len(issues) == 0, "issues": issues}
        except Exception as e:
            self.logger.error(f"Failed to check security health: {e}")
            raise

    async def _handle_incident(self, incident: Dict[str, Any]):
        """Handle an incident."""
        self.logger.info(f"Handling incident: {incident}")

        try:
            # Create incident response
            response = {
                "timestamp": datetime.now().isoformat(),
                "incident": incident,
                "actions": [],
            }

            # Handle based on incident type
            if incident["type"] == "system_health":
                response["actions"].extend(await self._handle_system_incident(incident))
            elif incident["type"] == "application_health":
                response["actions"].extend(
                    await self._handle_application_incident(incident)
                )
            elif incident["type"] == "database_health":
                response["actions"].extend(
                    await self._handle_database_incident(incident)
                )
            elif incident["type"] == "network_health":
                response["actions"].extend(
                    await self._handle_network_incident(incident)
                )
            elif incident["type"] == "security_health":
                response["actions"].extend(
                    await self._handle_security_incident(incident)
                )

            # Save response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_file = self.responses_dir / f"response_{timestamp}.json"

            with open(response_file, "w") as f:
                json.dump(response, f, indent=2)

            self.logger.info(f"Response saved to {response_file}")
            return response
        except Exception as e:
            self.logger.error(f"Failed to handle incident: {e}")
            raise

    async def _handle_system_incident(
        self, incident: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle system health incident."""
        actions = []

        # Scale up resources
        if "High CPU usage" in incident["details"]:
            actions.append(
                {"action": "scale_cpu", "details": "Scaling up CPU resources"}
            )

        if "High memory usage" in incident["details"]:
            actions.append(
                {"action": "scale_memory", "details": "Scaling up memory resources"}
            )

        if "High disk usage" in incident["details"]:
            actions.append(
                {"action": "cleanup_disk", "details": "Cleaning up disk space"}
            )

        return actions

    async def _handle_application_incident(
        self, incident: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle application health incident."""
        actions = []

        # Restart processes
        for issue in incident["details"]:
            if "High CPU usage" in issue or "High memory usage" in issue:
                process_name = issue.split(" for ")[1].split(":")[0]
                actions.append(
                    {
                        "action": "restart_process",
                        "details": f"Restarting process {process_name}",
                    }
                )

        # Rotate logs
        if "Large log file" in incident["details"]:
            actions.append({"action": "rotate_logs", "details": "Rotating log files"})

        return actions

    async def _handle_database_incident(
        self, incident: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle database health incident."""
        actions = []

        # Restart Redis
        if "Redis connection failed" in incident["details"]:
            actions.append(
                {"action": "restart_redis", "details": "Restarting Redis server"}
            )

        # Clear Redis cache
        if "High Redis memory usage" in incident["details"]:
            actions.append(
                {"action": "clear_redis_cache", "details": "Clearing Redis cache"}
            )

        return actions

    async def _handle_network_incident(
        self, incident: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle network health incident."""
        actions = []

        # Restart network interfaces
        if "Network errors" in incident["details"]:
            actions.append(
                {
                    "action": "restart_network",
                    "details": "Restarting network interfaces",
                }
            )

        # Restart API services
        for issue in incident["details"]:
            if "API endpoint" in issue:
                actions.append(
                    {"action": "restart_api", "details": "Restarting API services"}
                )

        return actions

    async def _handle_security_incident(
        self, incident: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle security health incident."""
        actions = []

        # Fix file permissions
        for issue in incident["details"]:
            if "Insecure file permissions" in issue:
                file_path = issue.split(": ")[1]
                actions.append(
                    {
                        "action": "fix_permissions",
                        "details": f"Fixing permissions for {file_path}",
                    }
                )

        # Remove sensitive data
        for issue in incident["details"]:
            if "Sensitive data found" in issue:
                file_path = issue.split(" in ")[1]
                actions.append(
                    {
                        "action": "remove_sensitive_data",
                        "details": f"Removing sensitive data from {file_path}",
                    }
                )

        return actions

    def analyze_incidents(self, incidents_file: str):
        """Analyze incidents."""
        self.logger.info(f"Analyzing incidents from {incidents_file}")

        try:
            # Load incidents
            with open(incidents_file) as f:
                incidents = json.load(f)

            # Calculate statistics
            stats = {
                "total_incidents": len(incidents),
                "incidents_by_type": {},
                "incidents_by_severity": {},
                "incidents_by_time": {},
            }

            for incident in incidents:
                # Count by type
                incident_type = incident["type"]
                stats["incidents_by_type"][incident_type] = (
                    stats["incidents_by_type"].get(incident_type, 0) + 1
                )

                # Count by severity
                severity = incident["severity"]
                stats["incidents_by_severity"][severity] = (
                    stats["incidents_by_severity"].get(severity, 0) + 1
                )

                # Count by time
                hour = datetime.fromisoformat(incident["timestamp"]).hour
                stats["incidents_by_time"][hour] = (
                    stats["incidents_by_time"].get(hour, 0) + 1
                )

            # Generate recommendations
            recommendations = []

            if stats["incidents_by_type"].get("system_health", 0) > 0:
                recommendations.append(
                    "System health incidents detected. Consider scaling up resources."
                )

            if stats["incidents_by_type"].get("application_health", 0) > 0:
                recommendations.append(
                    "Application health incidents detected. Consider optimizing application performance."
                )

            if stats["incidents_by_type"].get("database_health", 0) > 0:
                recommendations.append(
                    "Database health incidents detected. Consider optimizing database performance."
                )

            if stats["incidents_by_type"].get("network_health", 0) > 0:
                recommendations.append(
                    "Network health incidents detected. Consider improving network reliability."
                )

            if stats["incidents_by_type"].get("security_health", 0) > 0:
                recommendations.append(
                    "Security health incidents detected. Consider improving security measures."
                )

            # Save analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.incidents_dir / f"analysis_{timestamp}.json"

            analysis = {
                "timestamp": timestamp,
                "statistics": stats,
                "recommendations": recommendations,
            }

            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)

            self.logger.info(f"Analysis saved to {analysis_file}")
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze incidents: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Incident Manager")
    parser.add_argument(
        "command", choices=["monitor", "analyze"], help="Command to execute"
    )
    parser.add_argument(
        "--duration", type=int, default=300, help="Duration for monitoring in seconds"
    )
    parser.add_argument("--incidents-file", help="Incidents file to use")

    args = parser.parse_args()
    manager = IncidentManager()

    commands = {
        "monitor": lambda: asyncio.run(manager.monitor_incidents(args.duration)),
        "analyze": lambda: manager.analyze_incidents(args.incidents_file),
    }

    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
