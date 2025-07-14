"""Dynamic model loader for various LLM providers."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""

    name: str
    provider: str  # "openai", "huggingface", "claude", "mistral"
    model_type: str  # "causal", "sequence", "chat"
    api_key: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = None
    fallback_models: List[str] = None


class ModelLoader:
    """Dynamic model loader with support for multiple providers."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model loader.

        Args:
            config_path: Path to model configuration file
        """
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self._load_config(config_path)

    def initialize(self) -> Dict[str, Any]:
        """Initialize the model loader."""
        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def _load_config(self, config_path: Optional[str]) -> None:
        """Load model configurations from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                configs = yaml.safe_load(f)
                for name, config in configs.items():
                    self.configs[name] = ModelConfig(name=name, **config)
        else:
            # Default configurations
            self.configs = {
                "gpt-3.5-turbo": ModelConfig(
                    name="gpt-3.5-turbo", provider="openai", model_type="chat"
                ),
                "gpt2": ModelConfig(
                    name="gpt2", provider="huggingface", model_type="causal"
                ),
            }

    async def load_model(self, model_name: str, api_key: Optional[str] = None) -> None:
        """Load a model asynchronously with verification and fallback.

        Args:
            model_name: Name of the model to load
            api_key: Optional API key for the model
        """
        if model_name not in self.configs:
            raise ValueError(f"Unknown model: {model_name}")

        config = self.configs[model_name]
        if api_key:
            config.api_key = api_key

        # Verify model before loading
        if not self.verify_model(model_name):
            logger.warning(f"Model verification failed for {model_name}, attempting fallback")
            if config.fallback_models:
                for fallback in config.fallback_models:
                    if self.verify_model(fallback):
                        logger.info(f"Using verified fallback model: {fallback}")
                        await self.load_model(fallback, api_key)
                        return
            raise ValueError(f"Model verification failed and no valid fallbacks available")

        try:
            if config.provider == "openai":
                await self._load_openai_model(config)
            elif config.provider == "huggingface":
                await self._load_huggingface_model(config)
            elif config.provider == "claude":
                await self._load_claude_model(config)
            elif config.provider == "mistral":
                await self._load_mistral_model(config)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")

            self.active_model = model_name
            logger.info(f"Successfully loaded model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            if config.fallback_models:
                for fallback in config.fallback_models:
                    try:
                        await self.load_model(fallback, api_key)
                        logger.info(f"Successfully loaded fallback model: {fallback}")
                        return
                    except Exception as fallback_error:
                        logger.error(
                            f"Failed to load fallback model {fallback}: {str(fallback_error)}"
                        )
            raise

    def verify_model(self, model_name: str) -> bool:
        """
        Verify if a model path/type is valid.
        
        Args:
            model_name: Name of the model to verify
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        try:
            if model_name not in self.configs:
                logger.warning(f"Model {model_name} not found in configurations")
                return False
            
            config = self.configs[model_name]
            
            # Verify provider-specific requirements
            if config.provider == "openai":
                # Check if API key is available
                if not config.api_key and not os.getenv("OPENAI_API_KEY"):
                    logger.warning(f"OpenAI API key not available for model {model_name}")
                    return False
                    
            elif config.provider == "huggingface":
                # Check if model exists on HuggingFace Hub
                try:
                    from transformers import AutoTokenizer
                    AutoTokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
                    logger.info(f"Model {model_name} verified on HuggingFace Hub")
                except Exception as e:
                    logger.warning(f"Model {model_name} not found on HuggingFace Hub: {e}")
                    return False
                    
            elif config.provider == "claude":
                # Check if API key is available
                if not config.api_key and not os.getenv("ANTHROPIC_API_KEY"):
                    logger.warning(f"Claude API key not available for model {model_name}")
                    return False
                    
            elif config.provider == "mistral":
                # Check if API key is available
                if not config.api_key and not os.getenv("MISTRAL_API_KEY"):
                    logger.warning(f"Mistral API key not available for model {model_name}")
                    return False
            
            logger.info(f"Model {model_name} verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying model {model_name}: {e}")
            return False

    async def _load_openai_model(self, config: ModelConfig) -> None:
        """Load an OpenAI model."""
        if not config.api_key:
            raise ValueError("OpenAI API key is required")

        openai.api_key = config.api_key
        self.models[config.name] = {"provider": "openai", "config": config}

    async def _load_huggingface_model(self, config: ModelConfig) -> None:
        """Load a HuggingFace model with fallback to OpenAI."""
        try:
            # Wrap from_pretrained in try/except and fallback to OpenAI
            tokenizer = self._safe_from_pretrained(
                AutoTokenizer.from_pretrained, config.name, cache_dir=config.cache_dir
            )

            if config.model_type == "causal":
                model = self._safe_from_pretrained(
                    AutoModelForCausalLM.from_pretrained, config.name, cache_dir=config.cache_dir
                )
            else:
                model = self._safe_from_pretrained(
                    AutoModelForSequenceClassification.from_pretrained, config.name, cache_dir=config.cache_dir
                )

            model.to(config.device)

            self.models[config.name] = {
                "provider": "huggingface",
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
            }

        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            # Fallback to OpenAI
            await self._fallback_to_openai(config)

    def _safe_from_pretrained(self, loader_func, model_name: str, **kwargs):
        """
        Safely load a model using from_pretrained with fallback.
        
        Args:
            loader_func: The from_pretrained function to use
            model_name: Name of the model to load
            **kwargs: Additional arguments for the loader
            
        Returns:
            Loaded model or tokenizer
            
        Raises:
            Exception: If loading fails and no fallback is available
        """
        try:
            return loader_func(model_name, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to load {model_name} with {loader_func.__name__}: {e}")
            
            # Try with different model variants
            fallback_models = [
                "distilgpt2",
                "gpt2",
                "bert-base-uncased",
                "roberta-base"
            ]
            
            for fallback in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback}")
                    return loader_func(fallback, **kwargs)
                except Exception as fallback_error:
                    logger.warning(f"Fallback {fallback} also failed: {fallback_error}")
                    continue
            
            raise Exception(f"All fallback models failed for {model_name}")

    async def _fallback_to_openai(self, config: ModelConfig) -> None:
        """
        Fallback to OpenAI when HuggingFace model loading fails.
        
        Args:
            config: Model configuration
        """
        try:
            # Check if OpenAI is available
            if not config.api_key and not os.getenv("OPENAI_API_KEY"):
                raise Exception("No OpenAI API key available for fallback")
            
            logger.info("Falling back to OpenAI model")
            
            # Use a default OpenAI model
            fallback_config = ModelConfig(
                name="gpt-3.5-turbo",
                provider="openai",
                model_type="chat",
                api_key=config.api_key or os.getenv("OPENAI_API_KEY")
            )
            
            await self._load_openai_model(fallback_config)
            
            # Update the original config to use the fallback
            self.models[config.name] = self.models["gpt-3.5-turbo"]
            self.models[config.name]["config"] = config
            
            logger.info(f"Successfully fell back to OpenAI for {config.name}")
            
        except Exception as e:
            logger.error(f"OpenAI fallback also failed: {e}")
            raise

    async def _load_claude_model(self, config: ModelConfig) -> None:
        """Load a Claude model."""
        if not config.api_key:
            raise ValueError("Claude API key is required")

        # Placeholder for Claude model loading
        self.models[config.name] = {"provider": "claude", "config": config}

    async def _load_mistral_model(self, config: ModelConfig) -> None:
        """Load a Mistral model."""
        if not config.api_key:
            raise ValueError("Mistral API key is required")

        # Placeholder for Mistral model loading
        self.models[config.name] = {"provider": "mistral", "config": config}

    def get_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get a loaded model.

        Args:
            model_name: Name of the model to get, defaults to active model

        Returns:
            Dictionary containing model information
        """
        model_name = model_name or self.active_model
        if not model_name or model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        return self.models[model_name]

    def unload_model(self, model_name: str) -> None:
        """Unload a model to free memory.

        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            if self.models[model_name]["provider"] == "huggingface":
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del self.models[model_name]
            if self.active_model == model_name:
                self.active_model = None
            logger.info(f"Unloaded model: {model_name}")

    def list_available_models(self) -> List[str]:
        """List all available model configurations."""
        return list(self.configs.keys())

    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models."""
        return list(self.models.keys())
