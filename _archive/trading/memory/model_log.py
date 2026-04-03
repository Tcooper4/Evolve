"""
Model Log Bridge Module

This module provides a bridge to the memory.model_log module
for backward compatibility and cleaner import paths.
"""

try:
    # Try importing from memory.model_log (top-level module)
    from memory.model_log import log_model_performance
    from memory.model_log import get_best_models, get_model_performance_history
    
    # Create a ModelLog class wrapper for compatibility
    import logging
    from typing import Any, Dict, Optional
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    class ModelLog:
        """Model logging class that wraps memory.model_log functions."""
        
        def __init__(self, log_dir: Optional[str] = None):
            """Initialize model log."""
            self.log_dir = log_dir or "memory/logs"
            self.logger = logging.getLogger(__name__)
            logger.info("ModelLog initialized")
        
        def log(self, model_name: str, ticker: str, **kwargs) -> bool:
            """Log model performance data."""
            try:
                log_model_performance(
                    model_name=model_name,
                    ticker=ticker,
                    **kwargs
                )
                return True
            except Exception as e:
                self.logger.error(f"Error logging model data: {e}")
                return False
        
        def get_logs(self, model_name: Optional[str] = None, ticker: Optional[str] = None) -> list:
            """Get model logs."""
            try:
                if ticker:
                    # Return best models for ticker as dict, convert to list format
                    best_models = get_best_models(ticker)
                    if best_models:
                        return [best_models]  # Return as list of dicts
                    # Fallback to performance history if no best models
                    history_df = get_model_performance_history(ticker=ticker, model_name=model_name)
                    return history_df.to_dict('records') if not history_df.empty else []
                else:
                    # Return performance history as DataFrame, convert to list of dicts
                    history_df = get_model_performance_history(model_name=model_name)
                    return history_df.to_dict('records') if not history_df.empty else []
            except Exception as e:
                self.logger.error(f"Error getting model logs: {e}")
                return []
    
    __all__ = ["ModelLog", "log_model_performance", "get_best_models", "get_model_performance_history"]
except ImportError:
    # If memory.model_log doesn't exist, create a simple ModelLog class
    import logging
    from typing import Any, Dict, Optional
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    class ModelLog:
        """Simple model logging class."""
        
        def __init__(self, log_dir: Optional[str] = None):
            """Initialize model log."""
            self.log_dir = log_dir or "logs/model_logs"
            self.logger = logging.getLogger(__name__)
            logger.info("ModelLog initialized")
        
        def log(self, model_name: str, ticker: str, **kwargs) -> bool:
            """Log model data."""
            try:
                # Simple logging implementation
                self.logger.info(f"Model log: {model_name} ({ticker}) - {kwargs}")
                return True
            except Exception as e:
                self.logger.error(f"Error logging model data: {e}")
                return False
        
        def get_logs(self, model_name: Optional[str] = None, ticker: Optional[str] = None) -> list:
            """Get model logs."""
            return []
    
    __all__ = ["ModelLog"]

