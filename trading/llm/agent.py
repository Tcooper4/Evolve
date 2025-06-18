"""LLM Agent implementation with advanced prompting and tool usage."""

from typing import Dict, List, Optional, Any, Union, Callable
import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime
import yaml
from dataclasses import dataclass, field
import openai
from .model_loader import ModelLoader, ModelConfig
from .memory import MemoryManager
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for an LLM agent."""
    name: str
    role: str  # "educator", "analyst", "optimizer", etc.
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.9
    tools_enabled: bool = True
    memory_enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    confidence_threshold: float = 0.7

@dataclass
class AgentMetrics:
    """Metrics for agent performance tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    tool_calls: int = 0
    memory_hits: int = 0
    retries: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

class LLMAgent:
    """Advanced LLM agent with tool usage and memory capabilities."""
    
    def __init__(
        self,
        config: AgentConfig,
        model_loader: ModelLoader,
        memory_manager: Optional[MemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        """Initialize the LLM agent.
        
        Args:
            config: Agent configuration
            model_loader: Model loader instance
            memory_manager: Optional memory manager
            tool_registry: Optional tool registry
        """
        self.config = config
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        self.tool_registry = tool_registry
        self.metrics = AgentMetrics()
        self._load_templates()
        
    def _load_templates(self) -> None:
        """Load prompt templates for the agent's role."""
        template_path = Path(__file__).parent / "templates" / f"{self.config.role}.yaml"
        if template_path.exists():
            with open(template_path, 'r') as f:
                self.templates = yaml.safe_load(f)
        else:
            logger.warning(f"No templates found for role {self.config.role}")
            self.templates = {}
    
    async def process_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process a prompt with the agent.
        
        Args:
            prompt: Input prompt
            context: Optional context information
            tools: Optional list of tool names to enable
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()
        self.metrics.total_requests += 1
        
        try:
            # Check memory for similar prompts
            if self.config.memory_enabled and self.memory_manager:
                memory_result = await self.memory_manager.recall(prompt)
                if memory_result:
                    self.metrics.memory_hits += 1
                    return memory_result
            
            # Prepare prompt with context
            prepared_prompt = self._prepare_prompt(prompt, context)
            
            # Get model response with retries
            response = await self._get_model_response(prepared_prompt)
            
            # Process tool calls if enabled
            if self.config.tools_enabled and self.tool_registry:
                response = await self._process_tool_calls(response, tools)
            
            # Store in memory if enabled
            if self.config.memory_enabled and self.memory_manager:
                await self.memory_manager.store(prompt, response)
            
            # Update metrics
            self.metrics.successful_requests += 1
            self.metrics.total_time += (datetime.now() - start_time).total_seconds()
            
            return {
                "response": response,
                "metadata": {
                    "role": self.config.role,
                    "model": self.config.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": asdict(self.metrics)
                }
            }
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.errors.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "prompt": prompt
            })
            logger.error(f"Error processing prompt: {str(e)}")
            raise
    
    async def _get_model_response(self, prompt: str) -> str:
        """Get response from the model with retries.
        
        Args:
            prompt: Prepared prompt
            
        Returns:
            Model response
        """
        for attempt in range(self.config.max_retries):
            try:
                model = self.model_loader.get_model(self.config.model_name)
                if model["provider"] == "openai":
                    response = await self._get_openai_response(prompt, model)
                elif model["provider"] == "huggingface":
                    response = await self._get_huggingface_response(prompt, model)
                else:
                    raise ValueError(f"Unsupported provider: {model['provider']}")
                
                return response
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                self.metrics.retries += 1
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
    
    async def _get_openai_response(self, prompt: str, model: Dict[str, Any]) -> str:
        """Get response from OpenAI model."""
        messages = [
            {"role": "system", "content": self.templates.get("system", "")},
            {"role": "user", "content": prompt}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model=model["config"].name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p
        )
        
        return response.choices[0].message.content
    
    async def _get_huggingface_response(self, prompt: str, model: Dict[str, Any]) -> str:
        """Get response from HuggingFace model."""
        inputs = model["tokenizer"](
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model["config"].device)
        
        outputs = model["model"].generate(
            **inputs,
            max_length=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True
        )
        
        return model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    
    async def _process_tool_calls(
        self,
        response: str,
        enabled_tools: Optional[List[str]] = None
    ) -> str:
        """Process tool calls in the response.
        
        Args:
            response: Model response
            enabled_tools: List of enabled tool names
            
        Returns:
            Updated response with tool results
        """
        if not self.tool_registry:
            return response
            
        # Extract tool calls from response
        tool_calls = self._extract_tool_calls(response)
        if not tool_calls:
            return response
            
        results = []
        for tool_call in tool_calls:
            if enabled_tools and tool_call["name"] not in enabled_tools:
                continue
                
            try:
                result = await self.tool_registry.execute_tool(
                    tool_call["name"],
                    tool_call["args"]
                )
                results.append({
                    "tool": tool_call["name"],
                    "result": result
                })
                self.metrics.tool_calls += 1
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                results.append({
                    "tool": tool_call["name"],
                    "error": str(e)
                })
        
        # Update response with tool results
        return self._update_response_with_tool_results(response, results)
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from response text."""
        # Implement tool call extraction logic
        # This is a placeholder - actual implementation would depend on
        # the specific format used for tool calls
        return []
    
    def _update_response_with_tool_results(
        self,
        response: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """Update response with tool execution results."""
        # Implement response update logic
        # This is a placeholder - actual implementation would depend on
        # the specific format used for tool results
        return response
    
    def _prepare_prompt(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Prepare prompt with context and templates."""
        template = self.templates.get("prompt", "{prompt}")
        context_str = ""
        
        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            
        return template.format(
            prompt=prompt,
            context=context_str,
            role=self.config.role
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        return asdict(self.metrics)
    
    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self.metrics = AgentMetrics() 