"""
Scheduler module for managing periodic tasks and updates.
"""

import time
import threading
import logging
from typing import Callable, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class UpdateScheduler:
    """Scheduler for managing periodic model updates."""
    
    def __init__(self, check_interval: int = 6):
        """Initialize the scheduler.
        
        Args:
            check_interval: Interval in hours between checks
        """
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.callback = None
        
    def start(self, callback: Callable):
        """Start the scheduler.
        
        Args:
            callback: Function to call periodically
        """
        if self.running:
            logger.warning("Scheduler is already running")
            return
            
        self.callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Scheduler started with {self.check_interval} hour interval")
        
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Scheduler stopped")
        
    def _run(self):
        """Main scheduler loop."""
        while self.running:
            try:
                if self.callback:
                    self.callback()
                time.sleep(self.check_interval * 3600)  # Convert hours to seconds
            except Exception as e:
                logger.error(f"Error in scheduler callback: {e}")
                time.sleep(60)  # Wait 1 minute before retrying 