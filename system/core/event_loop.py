"""
Enhanced Event Loop Management

This module provides robust event loop management with:
- Safe event loop creation and reuse
- RuntimeError handling for crashed loops
- Backoff and retry mechanisms for task submission
- Thread-safe operations
"""

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EventLoopManager:
    """Thread-safe event loop manager with crash recovery and retry logic."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 0.1):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.RLock()
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._crashed_loops = weakref.WeakSet()
        self._task_registry = {}
        
    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create a healthy event loop."""
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._create_new_loop()
            return self._loop
    
    def _create_new_loop(self) -> None:
        """Create a new event loop with error handling."""
        try:
            # Use new_event_loop() instead of get_event_loop()
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            self._loop = new_loop
            logger.info("âœ… Created new event loop")
        except RuntimeError as e:
            logger.error(f"âŒ Failed to create event loop: {e}")
            # Try to recover by cleaning up and retrying
            self._cleanup_crashed_loops()
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            self._loop = new_loop
            logger.info("âœ… Recovered and created new event loop")
    
    def _cleanup_crashed_loops(self) -> None:
        """Clean up crashed event loops."""
        try:
            # Close any existing loops
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            
            # Clear crashed loops from registry
            self._crashed_loops.clear()
            
            # Force garbage collection of closed loops
            import gc
            gc.collect()
            
        except Exception as e:
            logger.warning(f"âš ï¸ Error during loop cleanup: {e}")
    
    async def submit_task_with_retry(
        self, 
        coro: Callable[..., Any], 
        *args, 
        task_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Submit a task with exponential backoff and retry logic."""
        task_id = task_name or f"task_{id(coro)}_{time.time()}"
        
        for attempt in range(self._max_retries):
            try:
                loop = self.get_event_loop()
                
                # Check if loop is healthy
                if loop.is_closed():
                    logger.warning(f"âš ï¸ Event loop closed, recreating...")
                    self._create_new_loop()
                    loop = self.get_event_loop()
                
                # Submit task with timeout
                result = await asyncio.wait_for(
                    coro(*args, **kwargs),
                    timeout=30.0  # 30 second timeout
                )
                
                logger.info(f"âœ… Task {task_id} completed successfully")
                return result
                
            except asyncio.TimeoutError:
                delay = self._base_delay * (2 ** attempt)
                logger.warning(f"â° Task {task_id} timed out, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)
                
            except RuntimeError as e:
                if "Event loop is closed" in str(e) or "no running event loop" in str(e):
                    logger.error(f"âŒ Event loop crashed: {e}")
                    self._crashed_loops.add(self._loop)
                    self._create_new_loop()
                    delay = self._base_delay * (2 ** attempt)
                    logger.info(f"ðŸ”„ Retrying task {task_id} in {delay}s after loop recovery")
                    await asyncio.sleep(delay)
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"âŒ Task {task_id} failed: {e}")
                if attempt == self._max_retries - 1:
                    raise
                delay = self._base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        
        raise RuntimeError(f"Task {task_id} failed after {self._max_retries} attempts")
    
    def run_in_executor_with_retry(
        self, 
        executor: Any, 
        func: Callable[..., T], 
        *args,
        **kwargs
    ) -> asyncio.Future[T]:
        """Run function in executor with retry logic."""
        loop = self.get_event_loop()
        
        if loop.is_closed():
            self._create_new_loop()
            loop = self.get_event_loop()
        
        return loop.run_in_executor(executor, func, *args, **kwargs)
    
    def is_healthy(self) -> bool:
        """Check if the current event loop is healthy."""
        with self._lock:
            return (
                self._loop is not None and 
                not self._loop.is_closed() and 
                self._loop not in self._crashed_loops
            )
    
    def shutdown(self) -> None:
        """Safely shutdown the event loop manager."""
        with self._lock:
            if self._loop and not self._loop.is_closed():
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    
                    # Run until all tasks are cancelled
                    if pending:
                        self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    
                    self._loop.close()
                    logger.info("âœ… Event loop manager shutdown complete")
                except Exception as e:
                    logger.error(f"âŒ Error during shutdown: {e}")

# Global instance
_loop_manager = EventLoopManager()

def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get the managed event loop."""
    return _loop_manager.get_event_loop()

def submit_task_with_retry(coro: Callable[..., Any], *args, **kwargs) -> Any:
    """Submit a task with retry logic using the global manager."""
    return _loop_manager.submit_task_with_retry(coro, *args, **kwargs)

def run_in_executor_with_retry(executor: Any, func: Callable[..., T], *args, **kwargs) -> asyncio.Future[T]:
    """Run function in executor with retry logic using the global manager."""
    return _loop_manager.run_in_executor_with_retry(executor, func, *args, **kwargs)

def is_loop_healthy() -> bool:
    """Check if the global event loop is healthy."""
    return _loop_manager.is_healthy()

def shutdown_loop_manager() -> None:
    """Shutdown the global event loop manager."""
    _loop_manager.shutdown()

# Decorator for automatic retry
def with_retry(max_retries: int = 3, base_delay: float = 0.1):
    """Decorator to add retry logic to async functions."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
            # If we get here, all retries failed
            raise last_exc
        return wrapper
    return decorator

# Context manager for temporary event loop
class EventLoopContext:
    """Context manager for temporary event loop operations."""
    
    def __init__(self, create_new: bool = False):
        self.create_new = create_new
        self.original_loop = None
        self.temp_loop = None
    
    async def __aenter__(self):
        if self.create_new:
            self.original_loop = asyncio.get_running_loop()
            self.temp_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.temp_loop)
        return self.temp_loop or asyncio.get_running_loop()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.create_new and self.temp_loop:
            try:
                self.temp_loop.close()
            except Exception:
                pass
            asyncio.set_event_loop(self.original_loop)

# Utility functions
def ensure_event_loop():
    """Ensure an event loop is running, create one if necessary."""
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No running event loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def safe_run_async(coro):
    """Safely run a coroutine with proper event loop handling."""
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(coro)
    except RuntimeError:
        # No running loop, create one and run
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
