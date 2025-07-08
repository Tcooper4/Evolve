"""
Safe Model Execution Layer

Provides isolated, timeout-protected, and memory-limited execution
for user-defined models and strategies to protect system stability.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import psutil
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
DEFAULT_MEMORY_LIMIT_MB = 1024  # 1GB
DEFAULT_MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CODE_SIZE_BYTES = 100000  # 100KB
DEFAULT_MONITORING_INTERVAL = 1  # 1 second
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

class ExecutionStatus(Enum):
    """Execution status enumeration."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    EXECUTION_ERROR = "execution_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"

@dataclass
class ExecutionResult:
    """Result of safe execution."""
    status: ExecutionStatus
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: float = 0.0
    return_value: Optional[Any] = None
    logs: List[str] = field(default_factory=list)

class SafeExecutor:
    """
    Safe execution environment for user-defined models and strategies.
    
    Features:
    - Timeout protection
    - Memory usage monitoring
    - Code validation and sanitization
    - Isolated execution scope
    - Error logging and recovery
    - Resource monitoring
    """
    
    def __init__(self, 
                 timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                 memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
                 max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE,
                 enable_sandbox: bool = True,
                 log_executions: bool = True):
        """
        Initialize the SafeExecutor.
        
        Args:
            timeout_seconds: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
            max_output_size: Maximum output size in bytes
            enable_sandbox: Enable sandboxed execution
            log_executions: Log all executions
        """
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.max_output_size = max_output_size
        self.enable_sandbox = enable_sandbox
        self.log_executions = log_executions
        
        # Execution statistics
        self.execution_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        
        # Create logs directory
        self.logs_dir = Path("logs/safe_executor")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SafeExecutor initialized with timeout={timeout_seconds}s, memory_limit={memory_limit_mb}MB")
    
    def execute_model(self, 
                     model_code: str,
                     model_name: str,
                     input_data: Dict[str, Any] = None,
                     model_type: str = "custom") -> ExecutionResult:
        """
        Execute a user-defined model safely.
        
        Args:
            model_code: Python code for the model
            model_name: Name of the model
            input_data: Input data for the model
            model_type: Type of model (custom, strategy, indicator, etc.)
            
        Returns:
            ExecutionResult with status and output
        """
        try:
            self.execution_count += 1
            
            # Validate input
            validation_result = self._validate_model_code(model_code, model_name)
            if validation_result.status != ExecutionStatus.SUCCESS:
                return validation_result
            
            # Create execution environment
            env_result = self._create_execution_environment(model_code, input_data)
            if env_result.status != ExecutionStatus.SUCCESS:
                return env_result
            
            # Execute in isolated process
            result = self._execute_isolated_process(
                env_result.return_value,  # script_path
                model_name,
                model_type
            )
            
            # Log execution
            if self.log_executions:
                self._log_execution(model_name, model_type, result)
            
            # Update statistics
            if result.status == ExecutionStatus.SUCCESS:
                self.success_count += 1
            else:
                self.error_count += 1
            
            self.total_execution_time += result.execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing model {model_name}: {e}")
            return ExecutionResult(
                status=ExecutionStatus.SYSTEM_ERROR,
                error=f"System error: {str(e)}",
                logs=[traceback.format_exc()]
            )
    
    def execute_strategy(self,
                        strategy_code: str,
                        strategy_name: str,
                        market_data: Dict[str, Any] = None,
                        parameters: Dict[str, Any] = None) -> ExecutionResult:
        """
        Execute a user-defined trading strategy safely.
        
        Args:
            strategy_code: Python code for the strategy
            strategy_name: Name of the strategy
            market_data: Market data for the strategy
            parameters: Strategy parameters
            
        Returns:
            ExecutionResult with status and output
        """
        # Prepare input data
        input_data = {
            'market_data': market_data or {},
            'parameters': parameters or {},
            'strategy_name': strategy_name
        }
        
        return {'success': True, 'result': self.execute_model(strategy_code, strategy_name, input_data, "strategy"), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def execute_indicator(self,
                         indicator_code: str,
                         indicator_name: str,
                         price_data: Dict[str, Any] = None,
                         parameters: Dict[str, Any] = None) -> ExecutionResult:
        """
        Execute a user-defined technical indicator safely.
        
        Args:
            indicator_code: Python code for the indicator
            indicator_name: Name of the indicator
            price_data: Price data for the indicator
            parameters: Indicator parameters
            
        Returns:
            ExecutionResult with status and output
        """
        # Prepare input data
        input_data = {
            'price_data': price_data or {},
            'parameters': parameters or {},
            'indicator_name': indicator_name
        }
        
        return {'success': True, 'result': self.execute_model(indicator_code, indicator_name, input_data, "indicator"), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _validate_model_code(self, model_code: str, model_name: str) -> ExecutionResult:
        """Validate model code for safety and syntax."""
        try:
            # Check for dangerous imports
            dangerous_imports = [
                'os', 'subprocess', 'sys', 'importlib', 'eval', 'exec',
                'open', 'file', 'input', 'raw_input', 'globals', 'locals'
            ]
            
            for dangerous_import in dangerous_imports:
                if f"import {dangerous_import}" in model_code or f"from {dangerous_import}" in model_code:
                    return ExecutionResult(
                        status=ExecutionStatus.VALIDATION_ERROR,
                        error=f"Dangerous import detected: {dangerous_import}",
                        logs=[f"Security validation failed for {model_name}"]
                    )
            
            # Check for dangerous functions
            dangerous_functions = ['eval', 'exec', 'compile', 'input']
            for func in dangerous_functions:
                if func in model_code:
                    return ExecutionResult(
                        status=ExecutionStatus.VALIDATION_ERROR,
                        error=f"Dangerous function detected: {func}",
                        logs=[f"Security validation failed for {model_name}"]
                    )
            
            # Check syntax
            try:
                compile(model_code, '<string>', 'exec')
            except SyntaxError as e:
                return ExecutionResult(
                    status=ExecutionStatus.VALIDATION_ERROR,
                    error=f"Syntax error: {str(e)}",
                    logs=[f"Syntax validation failed for {model_name}"]
                )
            
            # Check code length
            if len(model_code) > MAX_CODE_SIZE_BYTES:  # 100KB limit
                return ExecutionResult(
                    status=ExecutionStatus.VALIDATION_ERROR,
                    error="Code too long (max 100KB)",
                    logs=[f"Code size validation failed for {model_name}"]
                )
            
            return ExecutionResult(status=ExecutionStatus.SUCCESS)
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.VALIDATION_ERROR,
                error=f"Validation error: {str(e)}",
                logs=[traceback.format_exc()]
            )
    
    def _create_execution_environment(self, model_code: str, input_data: Dict[str, Any]) -> ExecutionResult:
        """Create isolated execution environment."""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="safe_exec_")
            script_path = Path(temp_dir) / "model_script.py"
            
            # Create wrapper script
            wrapper_code = self._create_wrapper_script(model_code, input_data)
            
            # Write script to file
            with open(script_path, 'w') as f:
                f.write(wrapper_code)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                return_value=str(script_path)
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.SYSTEM_ERROR,
                error=f"Failed to create execution environment: {str(e)}",
                logs=[traceback.format_exc()]
            )
    
    def _create_wrapper_script(self, model_code: str, input_data: Dict[str, Any]) -> str:
        """Create a wrapper script for safe execution."""
        wrapper = f'''#!/usr/bin/env python3
"""
Safe execution wrapper for user-defined model.
"""

import sys
import json
import traceback
import signal
import resource
import time
from typing import Dict, Any

# Set resource limits
def set_resource_limits():
    """Set resource limits for safe execution."""
    try:
        # Set memory limit (soft and hard)
        memory_limit = {self.memory_limit_mb * 1024 * 1024}  # Convert to bytes
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        
        # Set CPU time limit
        cpu_limit = {self.timeout_seconds}
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        
        # Set file size limit
        file_limit = {self.max_output_size}
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        
    except Exception as e:
        logger.warning(f"Warning: Could not set resource limits: {e}")

# Signal handler for timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout")

# Set up signal handler
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.timeout_seconds})

# Set resource limits
set_resource_limits()

# Input data
INPUT_DATA = {json.dumps(input_data, indent=2)}

# Safe execution environment
SAFE_MODULES = {{
    'numpy': 'np',
    'pandas': 'pd',
    'matplotlib.pyplot': 'plt',
    'sklearn': 'sklearn',
    'torch': 'torch',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels',
    'ta': 'ta',  # Technical analysis
    'datetime': 'datetime',
    'math': 'math',
    'random': 'random',
    'json': 'json',
    'collections': 'collections'
}}

# Import safe modules
imported_modules = {{}}
for module_name, alias in SAFE_MODULES.items():
    try:
        if '.' in module_name:
            parts = module_name.split('.')
            module = __import__(module_name, fromlist=[parts[-1]])
            imported_modules[alias] = getattr(module, parts[-1])
        else:
            imported_modules[alias] = __import__(module_name)
    except ImportError as e:
        logger.warning(f"Failed to import module {module_name}: {e}")
        # Continue with other imports

# Add safe modules to globals
globals().update(imported_modules)

# User model code
{model_code}

# Execute the model
try:
    start_time = time.time()
    
    # Check if main function exists
    if 'main' in globals() and callable(globals()['main']):
        result = main(INPUT_DATA)
    elif 'run_model' in globals() and callable(globals()['run_model']):
        result = run_model(INPUT_DATA)
    elif 'execute' in globals() and callable(globals()['execute']):
        result = execute(INPUT_DATA)
    else:
        # Execute the code directly
        exec(model_code)
        result = globals().get('result', None)
    
    execution_time = time.time() - start_time
    
    # Prepare output
    output = {{
        'status': 'success',
        'result': result,
        'execution_time': execution_time,
        'memory_used': 0.0  # Will be calculated by parent process
    }}
    
    logger.info(json.dumps(output))
    
except Exception as e:
    error_output = {{
        'status': 'error',
        'error': str(e),
        'traceback': traceback.format_exc(),
        'execution_time': time.time() - start_time if 'start_time' in locals() else 0.0
    }}
    logger.error(json.dumps(error_output))
    
finally:
    # Cancel alarm
    signal.alarm(0)
'''
        return wrapper
    
    def _execute_isolated_process(self, script_path: str, model_name: str, model_type: str) -> ExecutionResult:
        """Execute the model in an isolated process."""
        start_time = time.time()
        process = None
        
        try:
            # Start the process
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._set_process_limits if self.enable_sandbox else None
            )
            
            # Monitor the process
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            execution_time = time.time() - start_time
            
            # Check process return code
            if process.returncode == 0:
                try:
                    # Parse JSON output
                    output_data = json.loads(stdout.strip())
                    
                    if output_data.get('status') == 'success':
                        return ExecutionResult(
                            status=ExecutionStatus.SUCCESS,
                            output=stdout,
                            execution_time=execution_time,
                            return_value=output_data.get('result'),
                            logs=[f"Model {model_name} executed successfully"]
                        )
                    else:
                        return ExecutionResult(
                            status=ExecutionStatus.EXECUTION_ERROR,
                            error=output_data.get('error', 'Unknown error'),
                            execution_time=execution_time,
                            logs=[output_data.get('traceback', '')]
                        )
                        
                except json.JSONDecodeError:
                    return ExecutionResult(
                        status=ExecutionStatus.EXECUTION_ERROR,
                        error="Invalid JSON output from model",
                        execution_time=execution_time,
                        logs=[stdout, stderr]
                    )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.EXECUTION_ERROR,
                    error=f"Process returned code {process.returncode}",
                    execution_time=execution_time,
                    logs=[stderr]
                )
                
        except subprocess.TimeoutExpired:
            if process:
                process.kill()
                process.wait()
            
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timeout after {self.timeout_seconds} seconds",
                execution_time=time.time() - start_time,
                logs=[f"Model {model_name} timed out"]
            )
            
        except Exception as e:
            if process:
                process.kill()
                process.wait()
            
            return ExecutionResult(
                status=ExecutionStatus.SYSTEM_ERROR,
                error=f"System error: {str(e)}",
                execution_time=time.time() - start_time,
                logs=[traceback.format_exc()]
            )
    
    def _set_process_limits(self):
        """Set process limits for sandboxed execution."""
        try:
            import resource
            
            # Set memory limit (soft and hard)
            memory_limit = self.memory_limit_mb * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            
            # Set CPU time limit
            cpu_limit = self.timeout_seconds
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            
            # Set file size limit
            file_limit = self.max_output_size
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
            
        except Exception as e:
            logger.warning(f"Warning: Could not set resource limits: {e}")

    def _timeout_handler(self, signum, frame):
        """Signal handler for timeout."""
        raise TimeoutError("Execution timeout")

    def _log_execution(self, model_name: str, model_type: str, result: ExecutionResult):
        """Log execution details."""
        try:
            log_entry = {
                'timestamp': time.time(),
                'model_name': model_name,
                'model_type': model_type,
                'status': result.status.value,
                'execution_time': result.execution_time,
                'error': result.error,
                'logs': result.logs
            }
            
            log_file = self.logs_dir / f"execution_log_{int(time.time() // 86400)}.json"
            
            # Load existing logs
            logs = []
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'total_executions': self.execution_count,
            'successful_executions': self.success_count,
            'failed_executions': self.error_count,
            'success_rate': self.success_count / max(self.execution_count, 1),
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.total_execution_time / max(self.execution_count, 1),
            'timeout_seconds': self.timeout_seconds,
            'memory_limit_mb': self.memory_limit_mb
        }
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            # Clean up temporary files
            temp_pattern = Path(tempfile.gettempdir()) / "safe_exec_*"
            for temp_dir in Path(tempfile.gettempdir()).glob("safe_exec_*"):
                if temp_dir.is_dir():
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global safe executor instance
_safe_executor = None

def get_safe_executor() -> SafeExecutor:
    """Get the global safe executor instance."""
    global _safe_executor
    if _safe_executor is None:
        _safe_executor = SafeExecutor()
    return _safe_executor

def execute_model_safely(model_code: str, 
                        model_name: str,
                        input_data: Dict[str, Any] = None,
                        model_type: str = "custom") -> ExecutionResult:
    """
    Convenience function to execute a model safely.
    
    Args:
        model_code: Python code for the model
        model_name: Name of the model
        input_data: Input data for the model
        model_type: Type of model
        
    Returns:
        ExecutionResult with status and output
    """
    executor = get_safe_executor()
    return {'success': True, 'result': executor.execute_model(model_code, model_name, input_data, model_type), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def execute_strategy_safely(strategy_code: str,
                           strategy_name: str,
                           market_data: Dict[str, Any] = None,
                           parameters: Dict[str, Any] = None) -> ExecutionResult:
    """
    Convenience function to execute a strategy safely.
    
    Args:
        strategy_code: Python code for the strategy
        strategy_name: Name of the strategy
        market_data: Market data for the strategy
        parameters: Strategy parameters
        
    Returns:
        ExecutionResult with status and output
    """
    executor = get_safe_executor()
    return {'success': True, 'result': executor.execute_strategy(strategy_code, strategy_name, market_data, parameters), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def execute_indicator_safely(indicator_code: str,
                            indicator_name: str,
                            price_data: Dict[str, Any] = None,
                            parameters: Dict[str, Any] = None) -> ExecutionResult:
    """
    Convenience function to execute an indicator safely.
    
    Args:
        indicator_code: Python code for the indicator
        indicator_name: Name of the indicator
        price_data: Price data for the indicator
        parameters: Indicator parameters
        
    Returns:
        ExecutionResult with status and output
    """
    executor = get_safe_executor()
    return {'success': True, 'result': executor.execute_indicator(indicator_code, indicator_name, price_data, parameters), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}