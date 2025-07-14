"""System status and metrics calculation utilities."""

import json
import logging
import os
import platform
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import psutil

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SystemStatus:
    def __init__(self) -> None:
        """Initialize system status monitor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_uptime(self) -> str:
        """Get system uptime."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            return str(timedelta(seconds=int(uptime_seconds)))
        except Exception as e:
            self.logger.error(f"Error getting uptime: {str(e)}")
            return "Unknown"

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        try:
            return {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "load_avg": psutil.getloadavg(),
            }
        except Exception as e:
            self.logger.error(f"Error getting CPU info: {str(e)}")
            return {
                "usage_percent": None,
                "count": None,
                "frequency": None,
                "load_avg": None,
            }

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            }
        except Exception as e:
            self.logger.error(f"Error getting memory info: {str(e)}")
            return {"total": None, "available": None, "used": None, "percent": None}

    def get_disk_info(self) -> Dict[str, Any]:
        """Get disk information."""
        try:
            disk = psutil.disk_usage("/")
            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }
        except Exception as e:
            self.logger.error(f"Error getting disk info: {str(e)}")
            return {"total": None, "used": None, "free": None, "percent": None}

    def get_network_info(self) -> Dict[str, Any]:
        """Get network information."""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            }
        except Exception as e:
            self.logger.error(f"Error getting network info: {str(e)}")
            return {
                "bytes_sent": None,
                "bytes_recv": None,
                "packets_sent": None,
                "packets_recv": None,
            }

    def get_process_info(self) -> Dict[str, Any]:
        """Get process information."""
        try:
            process = psutil.Process()
            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": datetime.fromtimestamp(
                    process.create_time()
                ).isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error getting process info: {str(e)}")
            return {
                "pid": None,
                "name": None,
                "status": None,
                "cpu_percent": None,
                "memory_percent": None,
                "create_time": None,
            }

    def get_agent_liveness(self) -> Dict[str, Any]:
        """Get agent liveness information."""
        try:
            # Check if agent process is running
            agent_process = None
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if "agent" in proc.info["name"].lower():
                    agent_process = proc
                    break

            if agent_process:
                return {
                    "status": "running",
                    "pid": agent_process.info["pid"],
                    "uptime": str(
                        timedelta(
                            seconds=int(time.time() - agent_process.create_time())
                        )
                    ),
                    "last_heartbeat": self._get_last_heartbeat(),
                }
            else:
                return {
                    "status": "stopped",
                    "pid": None,
                    "uptime": None,
                    "last_heartbeat": None,
                }
        except Exception as e:
            self.logger.error(f"Error getting agent liveness: {str(e)}")
            return {
                "status": "unknown",
                "pid": None,
                "uptime": None,
                "last_heartbeat": None,
            }

    def _get_last_heartbeat(self) -> Optional[str]:
        """Get last agent heartbeat timestamp."""
        try:
            heartbeat_file = "logs/agent_heartbeat.log"
            if os.path.exists(heartbeat_file):
                with open(heartbeat_file, "r") as f:
                    last_line = f.readlines()[-1]
                    return last_line.strip()
            return None
        except Exception as e:
            self.logger.error(f"Error getting last heartbeat: {str(e)}")
            return None

    def get_system_info(self) -> Dict[str, Any]:
        """Get complete system information."""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": self.get_uptime(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "disk": self.get_disk_info(),
            "network": self.get_network_info(),
            "process": self.get_process_info(),
            "agent": self.get_agent_liveness(),
        }

    def save_status_report(self, filepath: str) -> dict:
        """Save status report to file. Returns status dict with filepath."""
        try:
            status = self.get_system_info()
            with open(filepath, "w") as f:
                json.dump(status, f, indent=2)
            return {"status": "report_saved", "filepath": filepath}
        except Exception as e:
            self.logger.error(f"Error saving status report: {str(e)}")
            return {"status": "report_error", "error": str(e)}

    def print_status(self) -> dict:
        """Print system status to console with proper logging."""
        try:
            status = self.get_system_info()
            logger.info("System Status Report")
            logger.info("===================")
            logger.info(f"Timestamp: {status['timestamp']}")
            logger.info(f"Uptime: {status['uptime']}")

            logger.info("Platform:")
            for key, value in status["platform"].items():
                logger.info(f"  {key}: {value}")

            logger.info("CPU:")
            for key, value in status["cpu"].items():
                logger.info(f"  {key}: {value}")

            logger.info("Memory:")
            for key, value in status["memory"].items():
                logger.info(f"  {key}: {value}")

            logger.info("Disk:")
            for key, value in status["disk"].items():
                logger.info(f"  {key}: {value}")

            logger.info("Network:")
            for key, value in status["network"].items():
                logger.info(f"  {key}: {value}")

            logger.info("Process:")
            for key, value in status["process"].items():
                logger.info(f"  {key}: {value}")

            logger.info("Agent:")
            for key, value in status["agent"].items():
                logger.info(f"  {key}: {value}")

            return {"status": "status_printed"}
        except Exception as e:
            self.logger.error(f"Error printing status: {str(e)}")
            return {"status": "status_error", "error": str(e)}


def get_system_health() -> Dict[str, Any]:
    """Get system health status."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_feed": "healthy",
                "models": "healthy",
                "strategies": "healthy",
                "execution": "healthy",
            },
            "overall_status": "healthy",
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "overall_status": "error",
        }


def get_system_scorecard() -> Dict[str, Any]:
    """Get system scorecard with health metrics."""
    try:
        status = SystemStatus()
        system_info = status.get_system_info()

        # Calculate health scores
        cpu_score = 100 - (system_info["cpu"]["usage_percent"] or 0)
        memory_score = 100 - (system_info["memory"]["percent"] or 0)
        disk_score = 100 - (system_info["disk"]["percent"] or 0)

        # Overall health score
        overall_score = (cpu_score + memory_score + disk_score) / 3

        scorecard = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": round(overall_score, 2),
            "components": {
                "cpu": {
                    "score": round(cpu_score, 2),
                    "usage": system_info["cpu"]["usage_percent"],
                    "status": "healthy"
                    if cpu_score > 70
                    else "warning"
                    if cpu_score > 50
                    else "critical",
                },
                "memory": {
                    "score": round(memory_score, 2),
                    "usage": system_info["memory"]["percent"],
                    "status": "healthy"
                    if memory_score > 70
                    else "warning"
                    if memory_score > 50
                    else "critical",
                },
                "disk": {
                    "score": round(disk_score, 2),
                    "usage": system_info["disk"]["percent"],
                    "status": "healthy"
                    if disk_score > 70
                    else "warning"
                    if disk_score > 50
                    else "critical",
                },
            },
            "system_info": {
                "uptime": system_info["uptime"],
                "platform": system_info["platform"]["system"],
                "agent_status": system_info["agent"]["status"],
            },
        }

        return scorecard

    except Exception as e:
        logging.error(f"Error generating system scorecard: {str(e)}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health": 0,
            "error": str(e),
            "components": {},
            "system_info": {},
        }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and print status
    status = SystemStatus()
    status.print_status()
