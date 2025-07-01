"""Dynamic model loader for various LLM providers."""

from typing import Dict, Any, Optional, Union, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import openai
from huggingface_hub import HfApi
import logging
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass
import yaml

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
        
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def _load_config(self, config_path: Optional[str]) -> None:
        """Load model configurations from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                configs = yaml.safe_load(f)
                for name, config in configs.items():
                    self.configs[name] = ModelConfig(name=name, **config)
        else:
            # Default configurations
            self.configs = {
                "gpt-3.5-turbo": ModelConfig(
                    name="gpt-3.5-turbo",
                    provider="openai",
                    model_type="chat"
                ),
                "gpt2": ModelConfig(
                    name="gpt2",
                    provider="huggingface",
                    model_type="causal"
                )
            }

    async def load_model(self, model_name: str, api_key: Optional[str] = None) -> None:
        """Load a model asynchronously.
        
        Args:
            model_name: Name of the model to load
            api_key: Optional API key for the model
        """
        if model_name not in self.configs:
            raise ValueError(f"Unknown model: {model_name}")
            
        config = self.configs[model_name]
        if api_key:
            config.api_key = api_key
            
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
                        logger.error(f"Failed to load fallback model {fallback}: {str(fallback_error)}")
            raise
    
    async def _load_openai_model(self, config: ModelConfig) -> None:
        """Load an OpenAI model."""
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
            
        openai.api_key = config.api_key
        self.models[config.name] = {
            "provider": "openai",
            "config": config
        }
    
    async def _load_huggingface_model(self, config: ModelConfig) -> None:
        """Load a HuggingFace model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.name,
                cache_dir=config.cache_dir
            )
            
            if config.model_type == "causal":
                model = AutoModelForCausalLM.from_pretrained(
                    config.name,
                    cache_dir=config.cache_dir
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    config.name,
                    cache_dir=config.cache_dir
                )
            
            model.to(config.device)
            
            self.models[config.name] = {
                "provider": "huggingface",
                "model": model,
                "tokenizer": tokenizer,
                "config": config
            }
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            raise
    
    async def _load_claude_model(self, config: ModelConfig) -> None:
        """Load a Claude model."""
        if not config.api_key:
            raise ValueError("Claude API key is required")
            
        # Placeholder for Claude model loading
        self.models[config.name] = {
            "provider": "claude",
            "config": config
        }
    
    async def _load_mistral_model(self, config: ModelConfig) -> None:
        """Load a Mistral model."""
        if not config.api_key:
            raise ValueError("Mistral API key is required")
            
        # Placeholder for Mistral model loading
        self.models[config.name] = {
            "provider": "mistral",
            "config": config
        }
    
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