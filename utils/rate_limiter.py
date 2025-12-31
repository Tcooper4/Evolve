"""
Rate limiting for EVOLVE trading system

Provides rate limiting to prevent API abuse and ensure fair resource usage.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self, calls_per_minute: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls allowed per minute
            window_seconds: Time window in seconds (default: 60)
        """
        self.calls_per_minute = calls_per_minute
        self.window_seconds = window_seconds
        self.calls: Dict[str, List[float]] = defaultdict(list)
        self.blocked_keys: Dict[str, float] = {}  # key -> unblock_time
    
    def check_rate_limit(self, key: str) -> bool:
        """
        Check if rate limit allows call.
        
        Args:
            key: Unique identifier for rate limiting (e.g., user_id, IP, API endpoint)
            
        Returns:
            True if call is allowed, False if rate limited
        """
        now = time.time()
        
        # Check if key is blocked
        if key in self.blocked_keys:
            if now < self.blocked_keys[key]:
                return False
            else:
                # Unblock
                del self.blocked_keys[key]
        
        # Remove old calls outside window
        window_start = now - self.window_seconds
        self.calls[key] = [t for t in self.calls[key] if t > window_start]
        
        # Check limit
        if len(self.calls[key]) >= self.calls_per_minute:
            # Block for remainder of window
            self.blocked_keys[key] = now + (self.window_seconds - (now - self.calls[key][0]))
            logger.warning(f"Rate limit exceeded for key: {key}")
            return False
        
        # Record call
        self.calls[key].append(now)
        return True
    
    def wait_if_needed(self, key: str, max_wait_seconds: float = 60.0) -> bool:
        """
        Wait if rate limited.
        
        Args:
            key: Unique identifier for rate limiting
            max_wait_seconds: Maximum time to wait
            
        Returns:
            True if call is now allowed, False if still rate limited after wait
        """
        start_time = time.time()
        
        while not self.check_rate_limit(key):
            if time.time() - start_time > max_wait_seconds:
                logger.warning(f"Rate limit wait timeout for key: {key}")
                return False
            
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        
        return True
    
    def get_remaining_calls(self, key: str) -> int:
        """
        Get remaining calls for key in current window.
        
        Args:
            key: Unique identifier
            
        Returns:
            Number of remaining calls allowed
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove old calls
        self.calls[key] = [t for t in self.calls[key] if t > window_start]
        
        remaining = self.calls_per_minute - len(self.calls[key])
        return max(0, remaining)
    
    def get_reset_time(self, key: str) -> Optional[datetime]:
        """
        Get time when rate limit resets for key.
        
        Args:
            key: Unique identifier
            
        Returns:
            Datetime when rate limit resets, or None if not rate limited
        """
        if not self.calls[key]:
            return None
        
        now = time.time()
        oldest_call = min(self.calls[key])
        reset_time = oldest_call + self.window_seconds
        
        if reset_time > now:
            return datetime.fromtimestamp(reset_time)
        
        return None
    
    def reset_key(self, key: str) -> None:
        """
        Reset rate limit for a key.
        
        Args:
            key: Unique identifier to reset
        """
        if key in self.calls:
            del self.calls[key]
        if key in self.blocked_keys:
            del self.blocked_keys[key]
        logger.debug(f"Rate limit reset for key: {key}")
    
    def clear_all(self) -> None:
        """Clear all rate limit data"""
        self.calls.clear()
        self.blocked_keys.clear()
        logger.info("Rate limiter cleared")


# Global rate limiter instance
_rate_limiter = RateLimiter()


def get_rate_limiter(calls_per_minute: int = 60) -> RateLimiter:
    """
    Get the global rate limiter instance.
    
    Args:
        calls_per_minute: Maximum calls per minute (only used on first call)
        
    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter.calls_per_minute != calls_per_minute:
        _rate_limiter = RateLimiter(calls_per_minute)
    return _rate_limiter

