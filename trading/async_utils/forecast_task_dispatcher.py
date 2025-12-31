"""
Async Forecast Task Dispatcher

Handles concurrent forecast model execution with proper error handling,
timeouts, and async-safe result reporting.
"""

import asyncio
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Configure async-safe logging
logger = logging.getLogger(__name__)


@dataclass
class ForecastTask:
    """Represents a forecast task with metadata"""

    task_id: str
    model_name: str
    symbol: str
    horizon: int
    created_at: datetime
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ForecastResult:
    """Represents a forecast result"""

    task_id: str
    model_name: str
    symbol: str
    forecast: pd.DataFrame
    confidence: float
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AsyncResultReporter:
    """Thread-safe result reporter for async operations"""

    def __init__(self, max_queue_size: int = 1000):
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._results: Dict[str, ForecastResult] = {}

    def add_result(self, result: ForecastResult) -> None:
        """Add a result to the queue and storage"""
        try:
            self.result_queue.put_nowait(result)
            with self._lock:
                self._results[result.task_id] = result
        except queue.Full:
            logger.warning(
                f"Result queue full, dropping result for task {result.task_id}"
            )

    def get_result(self, task_id: str) -> Optional[ForecastResult]:
        """Get a specific result by task ID"""
        with self._lock:
            return self._results.get(task_id)

    def get_all_results(self) -> List[ForecastResult]:
        """Get all results"""
        with self._lock:
            return list(self._results.values())

    def clear_old_results(self, max_age_hours: int = 24) -> None:
        """Clear results older than specified age"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        with self._lock:
            old_task_ids = [
                task_id
                for task_id, result in self._results.items()
                if result.created_at < cutoff_time
            ]
            for task_id in old_task_ids:
                del self._results[task_id]
            logger.info(f"Cleared {len(old_task_ids)} old results")


def async_safe_logging(func):
    """Decorator to ensure logging is async-safe"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Use asyncio.create_task to avoid blocking
            asyncio.create_task(
                asyncio.to_thread(logger.error, f"Error in {func.__name__}: {str(e)}")
            )
            raise

    return wrapper


class ForecastTaskDispatcher:
    """
    Async dispatcher for forecast tasks with proper error handling and timeouts
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        default_timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Task management
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_reporter = AsyncResultReporter()

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)

        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_timeout": 0,
            "total_execution_time": 0.0,
        }

    async def start(self) -> None:
        """Start the dispatcher"""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start worker tasks
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_concurrent_tasks)
        ]

        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            f"ForecastTaskDispatcher started with {self.max_concurrent_tasks} workers"
        )

        try:
            await asyncio.gather(*workers, cleanup_task, return_exceptions=True)
        except Exception as e:
            logger.error(f"Dispatcher error: {e}")
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the dispatcher gracefully"""
        if not self._running:
            return

        logger.info("Stopping ForecastTaskDispatcher...")
        self._shutdown_event.set()

        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        # Shutdown executor
        self.executor.shutdown(wait=True)
        self._running = False
        logger.info("ForecastTaskDispatcher stopped")

    async def submit_forecast_task(
        self,
        model_name: str,
        symbol: str,
        horizon: int,
        data: pd.DataFrame,
        priority: int = 0,
        timeout: Optional[int] = None,
    ) -> str:
        """Submit a forecast task for execution"""
        task_id = f"{model_name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        task = ForecastTask(
            task_id=task_id,
            model_name=model_name,
            symbol=symbol,
            horizon=horizon,
            created_at=datetime.now(),
            priority=priority,
        )

        await self.task_queue.put((priority, task, data))
        self.stats["tasks_submitted"] += 1

        logger.info(f"Submitted forecast task {task_id} for {model_name} on {symbol}")
        return task_id

    async def get_forecast_result(
        self, task_id: str, timeout: float = 30.0
    ) -> Optional[ForecastResult]:
        """Get forecast result by task ID with timeout"""
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            result = self.result_reporter.get_result(task_id)
            if result is not None:
                return result
            await asyncio.sleep(0.1)

        logger.warning(f"Timeout waiting for result for task {task_id}")
        return None

    async def get_all_results(self) -> List[ForecastResult]:
        """Get all completed results"""
        return self.result_reporter.get_all_results()

    @async_safe_logging
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine that processes forecast tasks"""
        logger.info(f"Worker {worker_name} started")

        while not self._shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    priority, task, data = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process task with timeout
                result = await self._process_forecast_task(task, data)
                self.result_reporter.add_result(result)

                # Update statistics
                if result.success:
                    self.stats["tasks_completed"] += 1
                else:
                    self.stats["tasks_failed"] += 1

                self.stats["total_execution_time"] += result.execution_time

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1.0)

        logger.info(f"Worker {worker_name} stopped")

    async def _process_forecast_task(
        self, task: ForecastTask, data: pd.DataFrame
    ) -> ForecastResult:
        """Process a single forecast task with retry logic"""
        start_time = datetime.now()

        for attempt in range(task.max_retries + 1):
            try:
                # Execute forecast with timeout
                forecast_result = await asyncio.wait_for(
                    self._execute_forecast_model(task, data),
                    timeout=self.default_timeout,
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                return ForecastResult(
                    task_id=task.task_id,
                    model_name=task.model_name,
                    symbol=task.symbol,
                    forecast=forecast_result["forecast"],
                    confidence=forecast_result.get("confidence", 0.5),
                    execution_time=execution_time,
                    success=True,
                    metadata=forecast_result.get("metadata", {}),
                )

            except asyncio.TimeoutError:
                logger.warning(f"Task {task.task_id} timed out (attempt {attempt + 1})")
                self.stats["tasks_timeout"] += 1

                if attempt < task.max_retries:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                else:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    return ForecastResult(
                        task_id=task.task_id,
                        model_name=task.model_name,
                        symbol=task.symbol,
                        forecast=pd.DataFrame(),
                        confidence=0.0,
                        execution_time=execution_time,
                        success=False,
                        error_message="Task timed out after all retries",
                    )

            except Exception as e:
                logger.error(f"Task {task.task_id} failed (attempt {attempt + 1}): {e}")

                if attempt < task.max_retries:
                    await asyncio.sleep(self.retry_delay * (2**attempt))
                    continue
                else:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    return ForecastResult(
                        task_id=task.task_id,
                        model_name=task.model_name,
                        symbol=task.symbol,
                        forecast=pd.DataFrame(),
                        confidence=0.0,
                        execution_time=execution_time,
                        success=False,
                        error_message=str(e),
                    )

    async def _execute_forecast_model(
        self, task: ForecastTask, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute the actual forecast model (to be implemented by subclasses)"""
        # This is a placeholder - actual implementation would call specific models
        # For now, return a mock forecast

        # Simulate model execution time
        await asyncio.sleep(0.1)

        # Generate real forecast using forecast engine
        try:
            from models.forecast_engine import ForecastEngine, ModelType
            
            # Initialize forecast engine
            forecast_engine = ForecastEngine()
            
            # Prepare data for forecast engine
            # Ensure we have required columns
            prepared_data = data.copy()
            if "Close" not in prepared_data.columns and "close" not in prepared_data.columns:
                # Use first numeric column
                price_col = prepared_data.select_dtypes(include=[np.number]).columns[0]
                prepared_data = prepared_data.rename(columns={price_col: "Close"})
            
            if "close" in prepared_data.columns:
                prepared_data = prepared_data.rename(columns={"close": "Close"})
            
            # Add timestamp if missing
            if "timestamp" not in prepared_data.columns:
                if isinstance(prepared_data.index, pd.DatetimeIndex):
                    prepared_data = prepared_data.reset_index()
                    if "Date" in prepared_data.columns:
                        prepared_data = prepared_data.rename(columns={"Date": "timestamp"})
                    else:
                        prepared_data["timestamp"] = prepared_data.index
                else:
                    from datetime import datetime
                    prepared_data["timestamp"] = pd.date_range(
                        end=datetime.now(), periods=len(prepared_data), freq="D"
                    )
            
            # Determine model type from task
            model_type = ModelType.LSTM  # Default
            if hasattr(task, 'model_name') and task.model_name:
                model_type_map = {
                    "lstm": ModelType.LSTM,
                    "arima": ModelType.ARIMA,
                    "xgboost": ModelType.XGBOOST,
                    "prophet": ModelType.PROPHET,
                    "transformer": ModelType.TRANSFORMER,
                    "ensemble": ModelType.ENSEMBLE,
                }
                model_type = model_type_map.get(task.model_name.lower(), ModelType.LSTM)
            
            # Generate forecast using real model
            forecast_result = forecast_engine.generate_forecast(
                data=prepared_data,
                model_type=model_type,
                forecast_days=task.horizon
            )
            
            # Extract forecast values and dates
            forecast_values = forecast_result.forecast_values
            forecast_dates = forecast_result.forecast_dates
            
            # Calculate confidence intervals
            confidence = forecast_result.confidence
            forecast_std = np.std(forecast_values) if len(forecast_values) > 1 else forecast_values[0] * 0.05
            z_score = 1.96  # 95% confidence
            
            lower_bound = np.array(forecast_values) - z_score * forecast_std * (1 - confidence)
            upper_bound = np.array(forecast_values) + z_score * forecast_std * (1 - confidence)
            
            forecast_df = pd.DataFrame(
                {
                    "forecast": forecast_values,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                },
                index=forecast_dates,
            )
            
            return {
                "forecast": forecast_df,
                "confidence": confidence,
                "metadata": {
                    "model_version": forecast_result.model_info.get("version", "1.0"),
                    "model_type": forecast_result.model_type.value,
                    "training_samples": len(data),
                    "last_training_date": prepared_data["timestamp"].max().isoformat() if "timestamp" in prepared_data.columns else data.index[-1].isoformat(),
                    "processing_time": forecast_result.processing_time,
                },
            }
            
        except Exception as e:
            logger.error(f"Error generating real forecast: {e}")
            # Fallback to simple forecast if model fails
            future_dates = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1), periods=task.horizon, freq="D"
            )
            
            # Use last price as baseline for fallback
            last_price = data.iloc[-1, 0] if len(data.columns) > 0 else 100.0
            forecast_values = np.full(task.horizon, last_price)  # Flat forecast as fallback
            
            forecast_df = pd.DataFrame(
                {
                    "forecast": forecast_values,
                    "lower_bound": forecast_values * 0.95,
                    "upper_bound": forecast_values * 1.05,
                },
                index=future_dates,
            )
            
            return {
                "forecast": forecast_df,
                "confidence": 0.5,  # Low confidence for fallback
                "metadata": {
                    "model_version": "fallback",
                    "training_samples": len(data),
                    "last_training_date": data.index[-1].isoformat(),
                    "error": str(e),
                },
            }

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old results and statistics"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                self.result_reporter.clear_old_results()

                # Log statistics
                logger.info(f"Dispatcher stats: {self.stats}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current dispatcher statistics"""
        return {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize(),
            "running": self._running,
        }


# Example usage and integration
class ModelForecastDispatcher(ForecastTaskDispatcher):
    """Specialized dispatcher for model forecasts"""

    def __init__(self, models: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.models = models

    async def _execute_forecast_model(
        self, task: ForecastTask, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute forecast using the specified model"""
        model = self.models.get(task.model_name)
        if not model:
            raise ValueError(f"Model {task.model_name} not found")

        # Execute model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._run_model_sync, model, data, task.horizon
        )

        return result

    def _run_model_sync(
        self, model: Any, data: pd.DataFrame, horizon: int
    ) -> Dict[str, Any]:
        """Run model synchronously in thread pool"""
        try:
            # Call the actual model's forecast method
            if hasattr(model, "forecast"):
                # Model has forecast method
                forecast_result = model.forecast(data, horizon)
                
                # Handle different return types
                if isinstance(forecast_result, dict):
                    forecast_values = forecast_result.get("forecast", forecast_result.get("predictions", []))
                    if isinstance(forecast_values, pd.DataFrame):
                        forecast_values = forecast_values.values.flatten()
                    
                    # Generate dates if not provided
                    if "forecast_dates" in forecast_result:
                        future_dates = forecast_result["forecast_dates"]
                    else:
                        from datetime import datetime
                        future_dates = pd.date_range(
                            start=data.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
                        )
                    
                    # Calculate confidence intervals
                    confidence = forecast_result.get("confidence", 0.8)
                    forecast_std = np.std(forecast_values) if len(forecast_values) > 1 else forecast_values[0] * 0.05
                    z_score = 1.96
                    lower_bound = np.array(forecast_values) - z_score * forecast_std * (1 - confidence)
                    upper_bound = np.array(forecast_values) + z_score * forecast_std * (1 - confidence)
                    
                    forecast_df = pd.DataFrame(
                        {
                            "forecast": forecast_values,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                        },
                        index=future_dates,
                    )
                    
                    return {
                        "forecast": forecast_df,
                        "confidence": confidence,
                        "metadata": {
                            "model_type": type(model).__name__,
                            "training_samples": len(data),
                        },
                    }
                else:
                    # Forecast result is array-like
                    forecast_values = np.array(forecast_result).flatten()
                    future_dates = pd.date_range(
                        start=data.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
                    )
                    
                    forecast_std = np.std(forecast_values) if len(forecast_values) > 1 else forecast_values[0] * 0.05
                    z_score = 1.96
                    lower_bound = forecast_values - z_score * forecast_std
                    upper_bound = forecast_values + z_score * forecast_std
                    
                    forecast_df = pd.DataFrame(
                        {
                            "forecast": forecast_values,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                        },
                        index=future_dates,
                    )
                    
                    return {
                        "forecast": forecast_df,
                        "confidence": 0.8,
                        "metadata": {
                            "model_type": type(model).__name__,
                            "training_samples": len(data),
                        },
                    }
            elif hasattr(model, "predict"):
                # Model has predict method
                predictions = model.predict(data)
                forecast_values = np.array(predictions).flatten()[:horizon]
                
                # If we need more values, extend with trend
                if len(forecast_values) < horizon:
                    last_value = forecast_values[-1] if len(forecast_values) > 0 else data.iloc[-1, 0]
                    trend = (forecast_values[-1] - forecast_values[0]) / len(forecast_values) if len(forecast_values) > 1 else 0
                    additional_values = [last_value + trend * i for i in range(1, horizon - len(forecast_values) + 1)]
                    forecast_values = np.concatenate([forecast_values, additional_values])
                
                future_dates = pd.date_range(
                    start=data.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
                )
                
                forecast_std = np.std(forecast_values) if len(forecast_values) > 1 else forecast_values[0] * 0.05
                z_score = 1.96
                lower_bound = forecast_values - z_score * forecast_std
                upper_bound = forecast_values + z_score * forecast_std
                
                forecast_df = pd.DataFrame(
                    {
                        "forecast": forecast_values,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    },
                    index=future_dates,
                )
                
                return {
                    "forecast": forecast_df,
                    "confidence": 0.8,
                    "metadata": {
                        "model_type": type(model).__name__,
                        "training_samples": len(data),
                    },
                }
            else:
                raise ValueError(f"Model {type(model).__name__} does not have forecast or predict method")
                
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise RuntimeError(f"Model execution failed: {e}")
