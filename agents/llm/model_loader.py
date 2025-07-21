"""
Enhanced Model Loader with Batch 12 Features

Dynamic model loader for various LLM providers with enhanced asyncio support.

Enhanced with Batch 12 features:
- Asyncio loading for deep models (LSTM, XGBoost, Transformers)
- Prefetch on startup for critical models
- Parallel model loading with resource management
- Enhanced caching and memory optimization
- Background model warming and health checks
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import openai
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Try to import additional model types
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Enhanced configuration for a model."""

    name: str
    provider: str  # "openai", "huggingface", "claude", "mistral", "lstm", "xgboost"
    model_type: str  # "causal", "sequence", "chat", "lstm", "xgboost", "transformer"
    api_key: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = None
    fallback_models: List[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    preload: bool = False  # Whether to preload on startup
    memory_limit: Optional[int] = None  # Memory limit in MB
    async_loading: bool = True  # Whether to load asynchronously
    metadata: Optional[Dict[str, Any]] = None  # Additional configuration metadata

    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelLoadStatus:
    """Status of model loading."""

    model_name: str
    status: str  # "loading", "loaded", "failed", "unloading"
    progress: float = 0.0  # 0.0 to 1.0
    error_message: Optional[str] = None
    load_time: Optional[float] = None
    memory_usage: Optional[float] = None
    last_used: Optional[float] = None


class AsyncModelLoader:
    """Enhanced model loader with asyncio support and deep model handling."""

    def __init__(self, config_path: Optional[str] = None, max_workers: int = 4):
        """Initialize the enhanced model loader.

        Args:
            config_path: Path to model configuration file
            max_workers: Maximum number of parallel model loading workers
        """
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.load_status: Dict[str, ModelLoadStatus] = {}
        self.active_model: Optional[str] = None

        # Async loading infrastructure
        self.loading_queue: asyncio.Queue = asyncio.Queue()
        self.loading_tasks: Set[asyncio.Task] = set()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Resource management
        self.memory_usage: Dict[str, float] = {}
        self.total_memory_limit = 8192  # 8GB default
        self.model_locks: Dict[str, asyncio.Lock] = {}

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.running = False

        # Load configuration
        self._load_config(config_path)

        # Initialize model locks
        for model_name in self.configs:
            self.model_locks[model_name] = asyncio.Lock()

    async def start(self) -> None:
        """Start the async model loader."""
        if self.running:
            return

        self.running = True

        # Start background tasks
        self.background_tasks.add(asyncio.create_task(self._loading_worker()))
        self.background_tasks.add(asyncio.create_task(self._memory_monitor()))
        self.background_tasks.add(asyncio.create_task(self._health_checker()))

        # Preload high-priority models
        await self._preload_priority_models()

        logger.info("Async model loader started")

    async def stop(self) -> None:
        """Stop the async model loader."""
        if not self.running:
            return

        self.running = False

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Async model loader stopped")

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load model configurations from file.

        Args:
            config_path: Path to model configuration file. If None, uses default config.
        """
        # Try to load from provided config path or default location
        config_locations = [
            config_path,
            "config/model_registry.yaml",
            "agents/llm/model_registry.yaml",
            "trading/config/model_registry.yaml",
        ]

        config_loaded = False
        for location in config_locations:
            if location and Path(location).exists():
                try:
                    with open(location, "r") as f:
                        config_data = yaml.safe_load(f)

                    # Load global configuration
                    global_config = config_data.get("global_config", {})
                    self.total_memory_limit = global_config.get(
                        "total_memory_limit", 8192
                    )
                    self.max_workers = global_config.get("max_concurrent_models", 4)

                    # Load model configurations
                    models_config = config_data.get("models", {})
                    for model_name, model_config in models_config.items():
                        # Create ModelConfig from registry data
                        config = ModelConfig(
                            name=model_name,
                            provider=model_config.get("provider", "unknown"),
                            model_type=model_config.get("model_type", "unknown"),
                            api_key=os.getenv(model_config.get("api_key_env", "")),
                            max_tokens=model_config.get("max_tokens", 500),
                            temperature=model_config.get("temperature", 0.7),
                            top_p=model_config.get("top_p", 0.9),
                            device=model_config.get("device", "auto"),
                            cache_dir=model_config.get("cache_dir"),
                            fallback_models=model_config.get("fallback_models", []),
                            priority=model_config.get("priority", 3),
                            preload=model_config.get("preload", False),
                            memory_limit=model_config.get("memory_limit"),
                            async_loading=model_config.get("async_loading", True),
                        )

                        # Store additional metadata
                        config.metadata = {
                            "class_path": model_config.get("class_path"),
                            "model_path": model_config.get("model_path"),
                            "tokenizer_path": model_config.get("tokenizer_path"),
                            "provider_config": model_config,
                        }

                        self.configs[model_name] = config

                    logger.info(f"Loaded {len(self.configs)} models from {location}")
                    config_loaded = True
                    break

                except Exception as e:
                    logger.warning(f"Failed to load config from {location}: {e}")
                    continue

        # Fallback to default configurations if no config file found
        if not config_loaded:
            logger.warning("No config file found, using default configurations")
            self._load_default_configs()

    def _load_default_configs(self) -> None:
        """Load default model configurations as fallback."""
        self.configs = {
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                model_type="chat",
                priority=1,
                preload=True,
                metadata={"class_path": "openai.ChatCompletion"},
            ),
            "gpt2": ModelConfig(
                name="gpt2",
                provider="huggingface",
                model_type="causal",
                priority=2,
                preload=True,
                metadata={
                    "class_path": "transformers.AutoModelForCausalLM",
                    "model_path": "gpt2",
                },
            ),
            "lstm-forecast": ModelConfig(
                name="lstm-forecast",
                provider="tensorflow",
                model_type="lstm",
                priority=1,
                preload=True,
                async_loading=True,
                metadata={"class_path": "trading.models.lstm_model.LSTMForecaster"},
            ),
            "xgboost-classifier": ModelConfig(
                name="xgboost-classifier",
                provider="xgboost",
                model_type="xgboost",
                priority=2,
                preload=True,
                async_loading=True,
                metadata={"class_path": "trading.models.xgboost_model.XGBoostModel"},
            ),
        }

    async def _preload_priority_models(self) -> None:
        """Preload high-priority models on startup."""
        priority_models = [
            config
            for config in self.configs.values()
            if config.preload and config.priority <= 2
        ]

        # Sort by priority
        priority_models.sort(key=lambda x: x.priority)

        logger.info(f"Preloading {len(priority_models)} priority models")

        # Load models in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.max_workers)

        async def load_with_semaphore(config):
            async with semaphore:
                return await self.load_model(config.name)

        tasks = [load_with_semaphore(config) for config in priority_models]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Preloaded {successful}/{len(priority_models)} priority models")

    async def load_model(self, model_name: str, api_key: Optional[str] = None) -> bool:
        """Load a model asynchronously with enhanced error handling.

        Args:
            model_name: Name of the model to load
            api_key: Optional API key for the model

        Returns:
            bool: True if model loaded successfully
        """
        if model_name not in self.configs:
            logger.error(f"Unknown model: {model_name}")
            return False

        config = self.configs[model_name]
        if api_key:
            config.api_key = api_key

        # Check if already loading or loaded
        if model_name in self.load_status:
            status = self.load_status[model_name]
            if status.status == "loading":
                logger.info(f"Model {model_name} is already loading")
                return True
            elif status.status == "loaded":
                logger.info(f"Model {model_name} is already loaded")
                return True

        # Initialize load status
        self.load_status[model_name] = ModelLoadStatus(
            model_name=model_name, status="loading", progress=0.0
        )

        try:
            # Acquire lock for this model
            async with self.model_locks[model_name]:
                # Verify model before loading
                if not await self._verify_model_async(model_name):
                    logger.warning(f"Model verification failed for {model_name}")
                    if config.fallback_models:
                        for fallback in config.fallback_models:
                            if await self._verify_model_async(fallback):
                                logger.info(
                                    f"Using verified fallback model: {fallback}"
                                )
                                return await self.load_model(fallback, api_key)
                    raise ValueError(
                        "Model verification failed and no valid fallbacks available"
                    )

                # Load model based on provider
                start_time = time.time()

                if config.provider == "openai":
                    await self._load_openai_model_async(config)
                elif config.provider == "huggingface":
                    await self._load_huggingface_model_async(config)
                elif config.provider == "tensorflow":
                    await self._load_tensorflow_model_async(config)
                elif config.provider == "xgboost":
                    await self._load_xgboost_model_async(config)
                elif config.provider == "claude":
                    await self._load_claude_model_async(config)
                elif config.provider == "mistral":
                    await self._load_mistral_model_async(config)
                else:
                    raise ValueError(f"Unsupported provider: {config.provider}")

                # Update status
                load_time = time.time() - start_time
                self.load_status[model_name] = ModelLoadStatus(
                    model_name=model_name,
                    status="loaded",
                    progress=1.0,
                    load_time=load_time,
                    last_used=time.time(),
                )

                logger.info(
                    f"Successfully loaded model: {model_name} in {load_time:.2f}s"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            self.load_status[model_name] = ModelLoadStatus(
                model_name=model_name, status="failed", error_message=str(e)
            )
            return False

    async def _verify_model_async(self, model_name: str) -> bool:
        """Verify model asynchronously.

        Args:
            model_name: Name of the model to verify

        Returns:
            bool: True if model is valid
        """
        try:
            if model_name not in self.configs:
                return False

            self.configs[model_name]

            # Run verification in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, self._verify_model_sync, model_name
            )

        except Exception as e:
            logger.error(f"Error verifying model {model_name}: {e}")
            return False

    def _verify_model_sync(self, model_name: str) -> bool:
        """Synchronous model verification.

        Args:
            model_name: Name of the model to verify

        Returns:
            bool: True if model is valid
        """
        try:
            config = self.configs[model_name]

            if config.provider == "openai":
                if not config.api_key and not os.getenv("OPENAI_API_KEY"):
                    return False

            elif config.provider == "huggingface":
                try:
                    AutoTokenizer.from_pretrained(
                        model_name, cache_dir=config.cache_dir
                    )
                except Exception:
                    return False

            elif config.provider == "tensorflow":
                if not TENSORFLOW_AVAILABLE:
                    return False

            elif config.provider == "xgboost":
                if not XGBOOST_AVAILABLE:
                    return False

            elif config.provider in ["claude", "mistral"]:
                if not config.api_key and not os.getenv(
                    f"{config.provider.upper()}_API_KEY"
                ):
                    return False

            return True

        except Exception:
            return False

    async def _load_huggingface_model_async(self, config: ModelConfig) -> None:
        """Load a HuggingFace model asynchronously."""
        try:
            # Load tokenizer and model in parallel
            loop = asyncio.get_event_loop()

            # Load tokenizer
            tokenizer_task = loop.run_in_executor(
                self.executor,
                AutoTokenizer.from_pretrained,
                config.name,
                cache_dir=config.cache_dir,
            )

            # Load model based on type
            if config.model_type == "causal":
                model_task = loop.run_in_executor(
                    self.executor,
                    AutoModelForCausalLM.from_pretrained,
                    config.name,
                    cache_dir=config.cache_dir,
                    device_map=config.device,
                )
            elif config.model_type == "sequence":
                model_task = loop.run_in_executor(
                    self.executor,
                    AutoModelForSequenceClassification.from_pretrained,
                    config.name,
                    cache_dir=config.cache_dir,
                    device_map=config.device,
                )
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")

            # Wait for both to complete
            tokenizer, model = await asyncio.gather(tokenizer_task, model_task)

            self.models[config.name] = {
                "provider": "huggingface",
                "config": config,
                "tokenizer": tokenizer,
                "model": model,
            }

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {config.name}: {e}")
            raise

    async def _load_tensorflow_model_async(self, config: ModelConfig) -> None:
        """Load a TensorFlow model asynchronously."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")

        try:
            loop = asyncio.get_event_loop()

            # Load model in thread pool
            model = await loop.run_in_executor(
                self.executor, keras.models.load_model, config.name
            )

            self.models[config.name] = {
                "provider": "tensorflow",
                "config": config,
                "model": model,
            }

        except Exception as e:
            logger.error(f"Failed to load TensorFlow model {config.name}: {e}")
            raise

    async def _load_xgboost_model_async(self, config: ModelConfig) -> None:
        """Load an XGBoost model asynchronously."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")

        try:
            loop = asyncio.get_event_loop()

            # Load model in thread pool
            model = await loop.run_in_executor(
                self.executor, xgb.Booster, model_file=config.name
            )

            self.models[config.name] = {
                "provider": "xgboost",
                "config": config,
                "model": model,
            }

        except Exception as e:
            logger.error(f"Failed to load XGBoost model {config.name}: {e}")
            raise

    async def _load_openai_model_async(self, config: ModelConfig) -> None:
        """Load an OpenAI model asynchronously."""
        if not config.api_key:
            raise ValueError("OpenAI API key is required")

        openai.api_key = config.api_key
        self.models[config.name] = {"provider": "openai", "config": config}

    async def _load_claude_model_async(self, config: ModelConfig) -> None:
        """Load a Claude model asynchronously."""
        if not config.api_key:
            raise ValueError("Claude API key is required")

        self.models[config.name] = {"provider": "claude", "config": config}

    async def _load_mistral_model_async(self, config: ModelConfig) -> None:
        """Load a Mistral model asynchronously."""
        if not config.api_key:
            raise ValueError("Mistral API key is required")

        self.models[config.name] = {"provider": "mistral", "config": config}

    async def _loading_worker(self) -> None:
        """Background worker for processing loading queue."""
        while self.running:
            try:
                # Process loading queue
                while not self.loading_queue.empty():
                    task = await self.loading_queue.get()
                    self.loading_tasks.add(task)

                    try:
                        await task
                    except Exception as e:
                        logger.error(f"Loading task failed: {e}")
                    finally:
                        self.loading_tasks.discard(task)
                        self.loading_queue.task_done()

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in loading worker: {e}")
                await asyncio.sleep(1)

    async def _memory_monitor(self) -> None:
        """Monitor memory usage and unload models if needed."""
        while self.running:
            try:
                # Calculate total memory usage
                total_memory = sum(self.memory_usage.values())

                if total_memory > self.total_memory_limit:
                    # Unload least recently used models
                    await self._unload_lru_models()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                await asyncio.sleep(10)

    async def _health_checker(self) -> None:
        """Periodic health check of loaded models."""
        while self.running:
            try:
                for model_name, status in self.load_status.items():
                    if status.status == "loaded":
                        # Check if model is still accessible
                        if not await self._check_model_health(model_name):
                            logger.warning(f"Model {model_name} health check failed")
                            # Mark for reload
                            status.status = "failed"

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(10)

    async def _check_model_health(self, model_name: str) -> bool:
        """Check if a loaded model is healthy.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model is healthy
        """
        try:
            if model_name not in self.models:
                return False

            model_info = self.models[model_name]

            # Provider-specific health checks
            if model_info["provider"] == "huggingface":
                # Check if model and tokenizer are accessible
                model = model_info["model"]
                tokenizer = model_info["tokenizer"]

                # Try a simple forward pass
                if hasattr(model, "forward"):
                    test_input = tokenizer("test", return_tensors="pt")
                    with torch.no_grad():
                        _ = model(**test_input)

            elif model_info["provider"] in ["tensorflow", "xgboost"]:
                # Check if model is accessible
                model = model_info["model"]
                if hasattr(model, "predict"):
                    # Try a simple prediction
                    pass  # Add specific health check logic

            return True

        except Exception:
            return False

    async def _unload_lru_models(self) -> None:
        """Unload least recently used models to free memory."""
        # Sort models by last used time
        lru_models = sorted(self.load_status.items(), key=lambda x: x[1].last_used or 0)

        for model_name, status in lru_models:
            if status.status == "loaded":
                await self.unload_model(model_name)
                break

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model asynchronously.

        Args:
            model_name: Name of the model to unload

        Returns:
            bool: True if unloaded successfully
        """
        if model_name not in self.models:
            return True

        try:
            async with self.model_locks[model_name]:
                # Clear model from memory
                del self.models[model_name]

                # Update status
                self.load_status[model_name] = ModelLoadStatus(
                    model_name=model_name, status="unloading"
                )

                # Clear memory usage
                if model_name in self.memory_usage:
                    del self.memory_usage[model_name]

                logger.info(f"Unloaded model: {model_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False

    def get_model(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a loaded model.

        Args:
            model_name: Name of the model to get

        Returns:
            Model info or None if not loaded
        """
        if model_name is None:
            model_name = self.active_model

        if model_name in self.models:
            # Update last used time
            if model_name in self.load_status:
                self.load_status[model_name].last_used = time.time()
            return self.models[model_name]

        return None

    def get_load_status(self, model_name: str) -> Optional[ModelLoadStatus]:
        """Get loading status of a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelLoadStatus or None if not found
        """
        return self.load_status.get(model_name)

    def list_loaded_models(self) -> List[str]:
        """List all loaded models.

        Returns:
            List of loaded model names
        """
        return list(self.models.keys())

    def list_available_models(self) -> List[str]:
        """List all available models.

        Returns:
            List of available model names
        """
        return list(self.configs.keys())

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage for all models.

        Returns:
            Dictionary mapping model names to memory usage in MB
        """
        return self.memory_usage.copy()

    def get_loading_queue_size(self) -> int:
        """Get current loading queue size.

        Returns:
            Number of models in loading queue
        """
        return self.loading_queue.qsize()

    def get_background_task_count(self) -> int:
        """Get number of background tasks.

        Returns:
            Number of background tasks
        """
        return len(self.background_tasks) + len(self.loading_tasks)


# Backward compatibility
ModelLoader = AsyncModelLoader
