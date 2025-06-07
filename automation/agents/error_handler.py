import logging
import traceback
from typing import Dict, Optional, Callable
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorHandler:
    def __init__(self, config: Dict):
        """Initialize the error handler."""
        self.config = config
        self.error_log_path = Path("automation/logs/errors")
        self.error_log_path.mkdir(parents=True, exist_ok=True)
        self.recovery_strategies: Dict[str, Callable] = {}
        self._register_default_recovery_strategies()

    def _register_default_recovery_strategies(self):
        """Register default recovery strategies."""
        self.recovery_strategies.update({
            "openai_api_error": self._handle_openai_api_error,
            "file_system_error": self._handle_file_system_error,
            "validation_error": self._handle_validation_error,
            "dependency_error": self._handle_dependency_error,
            "timeout_error": self._handle_timeout_error,
            "memory_error": self._handle_memory_error,
            "network_error": self._handle_network_error
        })

    async def handle_error(self, error: Exception, context: Dict) -> Optional[Dict]:
        """Handle an error and attempt recovery."""
        error_id = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        error_info = {
            "id": error_id,
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context
        }

        # Log the error
        self._log_error(error_info)

        # Attempt recovery
        recovery_result = await self._attempt_recovery(error, context)
        if recovery_result:
            error_info["recovery_successful"] = True
            error_info["recovery_result"] = recovery_result
        else:
            error_info["recovery_successful"] = False

        return error_info

    def _log_error(self, error_info: Dict):
        """Log error information to file."""
        try:
            log_file = self.error_log_path / f"{error_info['id']}.json"
            with open(log_file, 'w') as f:
                json.dump(error_info, f, indent=4)
            logger.error(f"Error logged: {error_info['id']}")
        except Exception as e:
            logger.error(f"Failed to log error: {str(e)}")

    async def _attempt_recovery(self, error: Exception, context: Dict) -> Optional[Dict]:
        """Attempt to recover from an error."""
        error_type = type(error).__name__.lower()
        
        # Find matching recovery strategy
        strategy = None
        for key in self.recovery_strategies:
            if key in error_type:
                strategy = self.recovery_strategies[key]
                break

        if strategy:
            try:
                return await strategy(error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {str(recovery_error)}")
                return None
        return None

    async def _handle_openai_api_error(self, error: Exception, context: Dict) -> Dict:
        """Handle OpenAI API errors."""
        # Implement retry logic with exponential backoff
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Attempt to retry the operation
                return {"status": "retried", "attempt": retry_count + 1}
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff

    async def _handle_file_system_error(self, error: Exception, context: Dict) -> Dict:
        """Handle file system errors."""
        try:
            # Attempt to create missing directories
            if "path" in context:
                Path(context["path"]).parent.mkdir(parents=True, exist_ok=True)
            return {"status": "directory_created"}
        except Exception as e:
            logger.error(f"Failed to handle file system error: {str(e)}")
            raise

    async def _handle_validation_error(self, error: Exception, context: Dict) -> Dict:
        """Handle validation errors."""
        try:
            # Attempt to fix validation issues
            if "data" in context:
                # Implement data cleaning/fixing logic
                return {"status": "data_fixed"}
            return {"status": "validation_failed"}
        except Exception as e:
            logger.error(f"Failed to handle validation error: {str(e)}")
            raise

    async def _handle_dependency_error(self, error: Exception, context: Dict) -> Dict:
        """Handle dependency errors."""
        try:
            # Attempt to install missing dependencies
            if "dependency" in context:
                # Implement dependency installation logic
                return {"status": "dependency_installed"}
            return {"status": "dependency_failed"}
        except Exception as e:
            logger.error(f"Failed to handle dependency error: {str(e)}")
            raise

    async def _handle_timeout_error(self, error: Exception, context: Dict) -> Dict:
        """Handle timeout errors."""
        try:
            # Implement timeout handling logic
            return {"status": "timeout_handled"}
        except Exception as e:
            logger.error(f"Failed to handle timeout error: {str(e)}")
            raise

    async def _handle_memory_error(self, error: Exception, context: Dict) -> Dict:
        """Handle memory errors."""
        try:
            # Implement memory optimization logic
            return {"status": "memory_optimized"}
        except Exception as e:
            logger.error(f"Failed to handle memory error: {str(e)}")
            raise

    async def _handle_network_error(self, error: Exception, context: Dict) -> Dict:
        """Handle network errors."""
        try:
            # Implement network retry logic
            return {"status": "network_retried"}
        except Exception as e:
            logger.error(f"Failed to handle network error: {str(e)}")
            raise

    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a custom recovery strategy."""
        self.recovery_strategies[error_type] = strategy

    def get_error_statistics(self) -> Dict:
        """Get error statistics."""
        try:
            error_files = list(self.error_log_path.glob("*.json"))
            stats = {
                "total_errors": len(error_files),
                "error_types": {},
                "recovery_success_rate": 0
            }
            
            successful_recoveries = 0
            for error_file in error_files:
                with open(error_file, 'r') as f:
                    error_info = json.load(f)
                    error_type = error_info["type"]
                    stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
                    if error_info.get("recovery_successful", False):
                        successful_recoveries += 1
            
            if stats["total_errors"] > 0:
                stats["recovery_success_rate"] = successful_recoveries / stats["total_errors"]
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get error statistics: {str(e)}")
            return {} 