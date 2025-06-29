"""
Centralized capability router for managing optional features and fallbacks.

This module provides a centralized way to check for system capabilities,
handle fallbacks gracefully, and log when fallbacks are triggered.
"""

import logging
import importlib
from typing import Dict, Any, Optional, Callable, Union
from functools import wraps
import streamlit as st

logger = logging.getLogger(__name__)


class CapabilityRouter:
    """Centralized router for managing system capabilities and fallbacks."""
    
    def __init__(self):
        """Initialize the capability router."""
        self._capabilities = {}
        self._fallbacks = {}
        self._capability_cache = {}
        
        # Register default capabilities
        self._register_default_capabilities()
    
    def _register_default_capabilities(self):
        """Register default system capabilities."""
        # LLM capabilities
        self.register_capability(
            'openai_api',
            self._check_openai_api,
            fallback=self._fallback_no_llm
        )
        
        self.register_capability(
            'huggingface_models',
            self._check_huggingface,
            fallback=self._fallback_no_hf
        )
        
        # Database capabilities
        self.register_capability(
            'redis_connection',
            self._check_redis,
            fallback=self._fallback_no_redis
        )
        
        self.register_capability(
            'postgres_connection',
            self._check_postgres,
            fallback=self._fallback_no_postgres
        )
        
        # External API capabilities
        self.register_capability(
            'alpha_vantage_api',
            self._check_alpha_vantage,
            fallback=self._fallback_no_alpha_vantage
        )
        
        self.register_capability(
            'yfinance_api',
            self._check_yfinance,
            fallback=self._fallback_no_yfinance
        )
        
        # Model capabilities
        self.register_capability(
            'torch_models',
            self._check_torch,
            fallback=self._fallback_no_torch
        )
    
    def register_capability(
        self, 
        name: str, 
        check_func: Callable[[], bool], 
        fallback: Optional[Callable] = None
    ):
        """Register a capability with its check function and optional fallback.
        
        Args:
            name: Name of the capability
            check_func: Function that returns True if capability is available
            fallback: Optional fallback function to call if capability is not available
        """
        self._capabilities[name] = check_func
        if fallback:
            self._fallbacks[name] = fallback
        logger.debug(f"Registered capability: {name}")
    
    def check_capability(self, name: str, use_cache: bool = True) -> bool:
        """Check if a capability is available.
        
        Args:
            name: Name of the capability to check
            use_cache: Whether to use cached results
            
        Returns:
            True if capability is available, False otherwise
        """
        if name not in self._capabilities:
            logger.warning(f"Unknown capability: {name}")
            return False
        
        # Use cache if requested and available
        if use_cache and name in self._capability_cache:
            return self._capability_cache[name]
        
        try:
            result = self._capabilities[name]()
            self._capability_cache[name] = result
            
            if not result:
                logger.warning(f"Capability '{name}' is not available")
                if name in self._fallbacks:
                    logger.info(f"Using fallback for capability '{name}'")
                    self._fallbacks[name]()
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking capability '{name}': {e}")
            self._capability_cache[name] = False
            return False
    
    def get_capability_status(self) -> Dict[str, bool]:
        """Get status of all registered capabilities.
        
        Returns:
            Dictionary mapping capability names to their availability status
        """
        status = {}
        for name in self._capabilities:
            status[name] = self.check_capability(name)
        return status
    
    def with_fallback(self, capability_name: str, fallback_value: Any = None):
        """Decorator to provide fallback behavior for functions that require capabilities.
        
        Args:
            capability_name: Name of the capability required
            fallback_value: Value to return if capability is not available
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.check_capability(capability_name):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in function {func.__name__}: {e}")
                        return fallback_value
                else:
                    logger.warning(f"Capability '{capability_name}' not available for {func.__name__}, using fallback")
                    return fallback_value
            return wrapper
        return decorator
    
    def safe_call(self, capability_name: str, func: Callable, *args, fallback_value: Any = None, **kwargs):
        """Safely call a function that requires a capability.
        
        Args:
            capability_name: Name of the capability required
            func: Function to call
            *args: Arguments to pass to the function
            fallback_value: Value to return if capability is not available
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of function call or fallback value
        """
        if self.check_capability(capability_name):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error calling {func.__name__}: {e}")
                return fallback_value
        else:
            logger.warning(f"Capability '{capability_name}' not available for {func.__name__}, using fallback")
            return fallback_value
    
    # Capability check functions
    def _check_openai_api(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            import openai
            # Check if API key is set
            api_key = st.secrets.get("openai_api_key") or st.session_state.get("openai_api_key")
            return bool(api_key)
        except ImportError:
            return False
    
    def _check_huggingface(self) -> bool:
        """Check if HuggingFace models are available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def _check_redis(self) -> bool:
        """Check if Redis connection is available."""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
            r.ping()
            return True
        except Exception:
            return False
    
    def _check_postgres(self) -> bool:
        """Check if PostgreSQL connection is available."""
        try:
            import psycopg2
            # This is a basic check - in practice you'd check actual connection
            return True
        except ImportError:
            return False
    
    def _check_alpha_vantage(self) -> bool:
        """Check if Alpha Vantage API is available."""
        try:
            from alpha_vantage.timeseries import TimeSeries
            api_key = st.secrets.get("alpha_vantage_api_key") or st.session_state.get("alpha_vantage_api_key")
            return bool(api_key)
        except ImportError:
            return False
    
    def _check_yfinance(self) -> bool:
        """Check if yfinance is available."""
        try:
            import yfinance
            return True
        except ImportError:
            return False
    
    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    # Fallback functions
    def _fallback_no_llm(self):
        """Fallback when LLM capabilities are not available."""
        logger.warning("LLM capabilities not available - using rule-based responses")
        st.warning("⚠️ AI features temporarily unavailable - using basic functionality")
    
    def _fallback_no_hf(self):
        """Fallback when HuggingFace models are not available."""
        logger.warning("HuggingFace models not available - using alternative models")
    
    def _fallback_no_redis(self):
        """Fallback when Redis is not available."""
        logger.warning("Redis not available - using in-memory storage")
        st.warning("⚠️ Using in-memory storage (data will not persist)")
    
    def _fallback_no_postgres(self):
        """Fallback when PostgreSQL is not available."""
        logger.warning("PostgreSQL not available - using file-based storage")
    
    def _fallback_no_alpha_vantage(self):
        """Fallback when Alpha Vantage API is not available."""
        logger.warning("Alpha Vantage API not available - using alternative data sources")
    
    def _fallback_no_yfinance(self):
        """Fallback when yfinance is not available."""
        logger.warning("yfinance not available - using alternative data sources")
    
    def _fallback_no_torch(self):
        """Fallback when PyTorch is not available."""
        logger.warning("PyTorch not available - using alternative models")


# Global capability router instance
capability_router = CapabilityRouter()


def check_capability(name: str) -> bool:
    """Check if a capability is available.
    
    Args:
        name: Name of the capability to check
        
    Returns:
        True if capability is available, False otherwise
    """
    return capability_router.check_capability(name)


def safe_call(capability_name: str, func: Callable, *args, fallback_value: Any = None, **kwargs):
    """Safely call a function that requires a capability.
    
    Args:
        capability_name: Name of the capability required
        func: Function to call
        *args: Arguments to pass to the function
        fallback_value: Value to return if capability is not available
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of function call or fallback value
    """
    return capability_router.safe_call(capability_name, func, *args, fallback_value=fallback_value, **kwargs)


def with_fallback(capability_name: str, fallback_value: Any = None):
    """Decorator to provide fallback behavior for functions that require capabilities.
    
    Args:
        capability_name: Name of the capability required
        fallback_value: Value to return if capability is not available
        
    Returns:
        Decorator function
    """
    return capability_router.with_fallback(capability_name, fallback_value)


def get_capability_status() -> Dict[str, bool]:
    """Get status of all registered capabilities.
    
    Returns:
        Dictionary mapping capability names to their availability status
    """
    return capability_router.get_capability_status()


def register_capability(name: str, check_func: Callable[[], bool], fallback: Optional[Callable] = None):
    """Register a capability with its check function and optional fallback.
    
    Args:
        name: Name of the capability
        check_func: Function that returns True if capability is available
        fallback: Optional fallback function to call if capability is not available
    """
    capability_router.register_capability(name, check_func, fallback) 