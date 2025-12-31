"""
Graceful shutdown handler for EVOLVE trading system

Handles system shutdown gracefully, ensuring all resources are cleaned up
and all operations complete before termination.
"""

import signal
import sys
import logging
import threading
from typing import Callable, List

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """Handles graceful shutdown"""
    
    def __init__(self):
        self.shutdown_callbacks: List[Callable] = []
        self.shutdown_in_progress = False
        self.shutdown_lock = threading.Lock()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("Shutdown handler initialized")
    
    def register_callback(self, callback: Callable) -> None:
        """
        Register a callback to be called during shutdown.
        
        Args:
            callback: Callable to execute during shutdown
        """
        with self.shutdown_lock:
            if callback not in self.shutdown_callbacks:
                self.shutdown_callbacks.append(callback)
                logger.debug(f"Registered shutdown callback: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable) -> None:
        """Unregister a shutdown callback"""
        with self.shutdown_lock:
            if callback in self.shutdown_callbacks:
                self.shutdown_callbacks.remove(callback)
                logger.debug(f"Unregistered shutdown callback: {callback.__name__}")
    
    def _signal_handler(self, signum, frame) -> None:
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum}), shutting down gracefully...")
        
        with self.shutdown_lock:
            if self.shutdown_in_progress:
                logger.warning("Shutdown already in progress, forcing exit")
                sys.exit(1)
            
            self.shutdown_in_progress = True
        
        # Execute all shutdown callbacks
        for callback in self.shutdown_callbacks:
            try:
                logger.debug(f"Executing shutdown callback: {callback.__name__}")
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")
        
        logger.info("Graceful shutdown complete")
        sys.exit(0)
    
    def shutdown(self) -> None:
        """Manually trigger graceful shutdown"""
        logger.info("Manual shutdown triggered")
        self._signal_handler(signal.SIGTERM, None)
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress"""
        with self.shutdown_lock:
            return self.shutdown_in_progress


# Global shutdown handler instance
_shutdown_handler = ShutdownHandler()


def get_shutdown_handler() -> ShutdownHandler:
    """Get the global shutdown handler instance"""
    return _shutdown_handler

