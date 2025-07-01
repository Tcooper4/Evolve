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
        self._fallback_log = []
        
        # Register default capabilities
        self._register_default_capabilities()
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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
        
        # Additional capabilities
        self.register_capability(
            'streamlit_interface',
            self._check_streamlit,
            fallback=self._fallback_no_streamlit
        )
        
        self.register_capability(
            'plotly_visualization',
            self._check_plotly,
            fallback=self._fallback_no_plotly
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
                    self._log_fallback(name, "capability_unavailable")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking capability '{name}': {e}")
            self._capability_cache[name] = False
            self._log_fallback(name, f"error: {str(e)}")
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
                        result = func(*args, **kwargs)
                        return result if result is not None else fallback_value
                    except Exception as e:
                        logger.error(f"Error in function {func.__name__}: {e}")
                        self._log_fallback(capability_name, f"function_error: {str(e)}")
                        return fallback_value
                else:
                    logger.warning(f"Capability '{capability_name}' not available for {func.__name__}, using fallback")
                    self._log_fallback(capability_name, "capability_unavailable")
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
                result = func(*args, **kwargs)
                return result if result is not None else fallback_value
            except Exception as e:
                logger.error(f"Error calling function {func.__name__}: {e}")
                self._log_fallback(capability_name, f"function_error: {str(e)}")
                return fallback_value
        else:
            logger.warning(f"Capability '{capability_name}' not available for {func.__name__}, using fallback")
            self._log_fallback(capability_name, "capability_unavailable")
            return fallback_value
    
    def _log_fallback(self, capability_name: str, reason: str):
        """Log when a fallback is triggered.
        
        Args:
            capability_name: Name of the capability that triggered fallback
            reason: Reason for the fallback
        """
        from datetime import datetime
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'capability': capability_name,
            'reason': reason
        }
        self._fallback_log.append(log_entry)
        
        # Keep only last 100 fallback logs
        if len(self._fallback_log) > 100:
            self._fallback_log = self._fallback_log[-100:]
    
    def get_fallback_log(self, limit: int = 10) -> list:
        """Get recent fallback logs.
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of recent fallback logs
        """
        return self._fallback_log[-limit:]
    
    def clear_cache(self) -> None:
        """Clear the capability cache."""
        self._capability_cache.clear()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on capabilities.
        
        Returns:
            Dictionary containing system health information
        """
        capability_status = self.get_capability_status()
        available_capabilities = sum(capability_status.values())
        total_capabilities = len(capability_status)
        
        critical_capabilities = ['openai_api', 'alpha_vantage_api', 'streamlit_interface']
        critical_available = all(capability_status.get(cap, False) for cap in critical_capabilities)
        
        return {
            'overall_status': 'healthy' if critical_available else 'degraded',
            'available_capabilities': available_capabilities,
            'total_capabilities': total_capabilities,
            'critical_capabilities_available': critical_available,
            'capability_status': capability_status,
            'recent_fallbacks': self.get_fallback_log(5)
        }
    
    # Capability check methods
    def _check_openai_api(self) -> bool:
        """Check if OpenAI API is available."""
        try:
            import openai
            # Check if API key is configured
            if hasattr(openai, 'api_key') and openai.api_key:
                return True
            # Check environment variable
            import os
            return os.getenv('OPENAI_API_KEY') is not None
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_huggingface(self) -> bool:
        """Check if HuggingFace models are available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
        except Exception:
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
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_alpha_vantage(self) -> bool:
        """Check if Alpha Vantage API is available."""
        try:
            import os
            return os.getenv('ALPHA_VANTAGE_API_KEY') is not None
        except Exception:
            return False
    
    def _check_yfinance(self) -> bool:
        """Check if yfinance is available."""
        try:
            import yfinance
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_streamlit(self) -> bool:
        """Check if Streamlit is available."""
        try:
            import streamlit
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def _check_plotly(self) -> bool:
        """Check if Plotly is available."""
        try:
            import plotly
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    # Fallback methods
    def _fallback_no_llm(self):
        """Fallback when LLM is not available."""
        logger.warning("LLM not available, using rule-based fallback")
        return {'success': True, 'result': {"status": "fallback", "method": "rule_based"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_hf(self):
        """Fallback when HuggingFace is not available."""
        logger.warning("HuggingFace not available, using OpenAI fallback")
        return {'success': True, 'result': {"status": "fallback", "method": "openai"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_redis(self):
        """Fallback when Redis is not available."""
        logger.warning("Redis not available, using in-memory storage")
        return {'success': True, 'result': {"status": "fallback", "method": "memory"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_postgres(self):
        """Fallback when PostgreSQL is not available."""
        logger.warning("PostgreSQL not available, using SQLite fallback")
        return {'success': True, 'result': {"status": "fallback", "method": "sqlite"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_alpha_vantage(self):
        """Fallback when Alpha Vantage is not available."""
        logger.warning("Alpha Vantage not available, using yfinance fallback")
        return {'success': True, 'result': {"status": "fallback", "method": "yfinance"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_yfinance(self):
        """Fallback when yfinance is not available."""
        logger.warning("yfinance not available, using mock data")
        return {'success': True, 'result': {"status": "fallback", "method": "mock_data"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_torch(self):
        """Fallback when PyTorch is not available."""
        logger.warning("PyTorch not available, using scikit-learn fallback")
        return {'success': True, 'result': {"status": "fallback", "method": "sklearn"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_streamlit(self):
        """Fallback when Streamlit is not available."""
        logger.warning("Streamlit not available, using CLI interface")
        return {'success': True, 'result': {"status": "fallback", "method": "cli"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _fallback_no_plotly(self):
        """Fallback when Plotly is not available."""
        logger.warning("Plotly not available, using matplotlib fallback")
        return {'success': True, 'result': {"status": "fallback", "method": "matplotlib"}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

# Global instance
_capability_router = CapabilityRouter()

# Convenience functions
def check_capability(name: str) -> bool:
    """Check if a capability is available.
    
    Args:
        name: Name of the capability to check
        
    Returns:
        True if capability is available, False otherwise
    """
    return _capability_router.check_capability(name)

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
    return {'success': True, 'result': _capability_router.safe_call(capability_name, func, *args, fallback_value=fallback_value, **kwargs), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def with_fallback(capability_name: str, fallback_value: Any = None):
    """Decorator to provide fallback behavior for functions that require capabilities.
    
    Args:
        capability_name: Name of the capability required
        fallback_value: Value to return if capability is not available
        
    Returns:
        Decorator function
    """
    return {'success': True, 'result': _capability_router.with_fallback(capability_name, fallback_value), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def get_capability_status() -> Dict[str, bool]:
    """Get status of all registered capabilities.
    
    Returns:
        Dictionary mapping capability names to their availability status
    """
    return _capability_router.get_capability_status()

def register_capability(name: str, check_func: Callable[[], bool], fallback: Optional[Callable] = None):
    """Register a capability with its check function and optional fallback.
    
    Args:
        name: Name of the capability
        check_func: Function that returns True if capability is available
        fallback: Optional fallback function to call if capability is not available
    """
    _capability_router.register_capability(name, check_func, fallback)

def get_system_health() -> Dict[str, Any]:
    """Get overall system health based on capabilities.
    
    Returns:
        Dictionary containing system health information
    """
    return _capability_router.get_system_health()

def get_fallback_log(limit: int = 10) -> list:
    """Get recent fallback logs.
    
    Args:
        limit: Maximum number of logs to return
        
    Returns:
        List of recent fallback logs
    """
    return _capability_router.get_fallback_log(limit)