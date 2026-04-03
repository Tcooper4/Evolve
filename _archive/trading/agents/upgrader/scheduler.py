"""
Scheduler module for the Upgrader Agent.

This module handles the scheduling of periodic upgrade checks and maintenance tasks.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import schedule

logger = logging.getLogger("UpgraderScheduler")


class UpgradeScheduler:
    """
    Scheduler for managing periodic upgrade checks and maintenance tasks.
    """

    def __init__(self, check_interval: int = 24):
        """
        Initialize the upgrade scheduler.

        Args:
            check_interval: Hours between upgrade checks
        """
        self.check_interval = check_interval
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        self.last_check: Optional[datetime] = None
        self.init_status = {
            "success": True,
            "message": "UpgradeScheduler initialized successfully",
            "timestamp": datetime.now().isoformat(),
        }

    def start(self, check_callback: Callable) -> dict:
        """
        Start the scheduler with the given callback function.

        Args:
            check_callback: Function to call for upgrade checks
        """
        if self.running:
            logger.warning("Scheduler is already running")
            return {
                "success": False,
                "error": "Scheduler is already running",
                "timestamp": datetime.now().isoformat(),
            }
        self.running = True
        # Schedule regular upgrade checks
        schedule.every(self.check_interval).hours.do(check_callback)

        # Start the scheduler in a separate thread
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)

        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info(f"Scheduler started with {self.check_interval} hour interval")
        return {
            "success": True,
            "message": f"Scheduler started with {self.check_interval} hour interval",
            "timestamp": datetime.now().isoformat(),
        }

    def stop(self) -> dict:
        """Stop the scheduler."""
        if not self.running:
            logger.warning("Scheduler is not running")
            return {
                "success": False,
                "error": "Scheduler is not running",
                "timestamp": datetime.now().isoformat(),
            }
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")
        return {
            "success": True,
            "message": "Scheduler stopped",
            "timestamp": datetime.now().isoformat(),
        }

    def run_check(self, check_callback: Callable) -> dict:
        """
        Run an immediate upgrade check.

        Args:
            check_callback: Function to call for upgrade checks
        """
        try:
            self.last_check = datetime.now()
            check_callback()
            logger.info("Manual upgrade check completed")
            return {
                "success": True,
                "message": "Manual upgrade check completed",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error during manual upgrade check: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_status(self) -> dict:
        """
        Get the current status of the scheduler.

        Returns:
            dict: Scheduler status information
        """
        return {
            "success": True,
            "result": {
                "running": self.running,
                "check_interval": self.check_interval,
                "last_check": self.last_check.isoformat() if self.last_check else None,
                "next_check": (
                    schedule.next_run().isoformat() if schedule.jobs else None
                ),
            },
            "message": "Scheduler status retrieved",
            "timestamp": datetime.now().isoformat(),
        }
