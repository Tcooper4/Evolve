"""Advanced LLM interface for trading system with robust prompt processing and context management."""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import yaml

from ..logs.logger import get_logger
from ..logs.audit_logger import audit_logger
from trading.model_loader import ModelLoader, ModelConfig
from trading.agents.llm.agent import LLMAgent, AgentConfig
from trading.memory import MemoryManager
from trading.tools import ToolRegistry, tool

# Get logger
logger = get_logger(__name__)

class LLMInterface:
    """Advanced LLM interface with agent-based processing and tool support."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        tools_dir: Optional[Union[str, Path]] = None,
        memory_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the LLM interface.
        
        Args:
            config_path: Path to configuration file
            model_name: Name of the model to use
            api_key: Optional API key for the model
            tools_dir: Optional directory containing tool definitions
            memory_dir: Optional directory for memory storage
        """
        logger.info(f"Initializing LLM interface with model: {model_name}")
        
        # Initialize components
        self.model_loader = ModelLoader(config_path)
        self.tool_registry = ToolRegistry(tools_dir)
        self.memory_manager = MemoryManager(memory_dir) if memory_dir else None
        
        # Create default agent
        self.agent = LLMAgent(
            config=AgentConfig(
                name="default",
                role="analyst",
                model_name=model_name
            ),
            model_loader=self.model_loader,
            memory_manager=self.memory_manager,
            tool_registry=self.tool_registry
        )
        
        # Load model
        asyncio.create_task(self.model_loader.load_model(model_name, api_key))
        
        # Initialize metrics
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "tool_calls": 0,
            "memory_hits": 0,
            "errors": []
        }
        
        logger.info("LLM interface initialized successfully")
    
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
        self.metrics["total_requests"] += 1
        
        logger.info(f"Processing prompt: {prompt[:100]}...")
        
        try:
            # Process with agent
            result = await self.agent.process_prompt(prompt, context, tools)
            
            # Update metrics
            self.metrics["successful_requests"] += 1
            self.metrics["total_time"] += (datetime.now() - start_time).total_seconds()
            self.metrics["total_tokens"] += result.get("metadata", {}).get("tokens", 0)
            self.metrics["tool_calls"] += result.get("metadata", {}).get("tool_calls", 0)
            self.metrics["memory_hits"] += result.get("metadata", {}).get("memory_hits", 0)
            
            # Log to audit trail
            audit_logger.log_prompt(
                prompt=prompt,
                response=result.get("content", ""),
                agent_id=self.agent.config.name,
                module=__name__,
                metadata={
                    "context": context,
                    "tools": tools,
                    "metrics": {
                        "tokens": result.get("metadata", {}).get("tokens", 0),
                        "tool_calls": result.get("metadata", {}).get("tool_calls", 0),
                        "memory_hits": result.get("metadata", {}).get("memory_hits", 0)
                    }
                }
            )
            
            logger.info("Prompt processed successfully")
            return result
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.metrics["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "prompt": prompt
            })
            
            # Log error to audit trail
            audit_logger.log_error(
                error=str(e),
                agent_id=self.agent.config.name,
                module=__name__,
                metadata={
                    "prompt": prompt,
                    "context": context,
                    "tools": tools
                }
            )
            
            logger.error(f"Error processing prompt: {str(e)}")
            raise
    
    def create_agent(
        self,
        name: str,
        role: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> LLMAgent:
        """Create a new agent with specific configuration.
        
        Args:
            name: Name of the agent
            role: Role of the agent
            model_name: Optional model name to use
            **kwargs: Additional configuration parameters
            
        Returns:
            Newly created LLMAgent instance
        """
        logger.info(f"Creating new agent: {name} with role: {role}")
        
        config = AgentConfig(
            name=name,
            role=role,
            model_name=model_name or self.agent.config.model_name,
            **kwargs
        )
        
        agent = LLMAgent(
            config=config,
            model_loader=self.model_loader,
            memory_manager=self.memory_manager,
            tool_registry=self.tool_registry
        )
        
        # Log agent creation to audit trail
        audit_logger.log_llm(
            model_name=config.model_name,
            action="agent_created",
            agent_id=name,
            module=__name__,
            metadata={
                "role": role,
                "config": kwargs
            }
        )
        
        return agent
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "agent_metrics": self.agent.get_metrics(),
            "memory_stats": self.memory_manager.get_memory_stats() if self.memory_manager else None,
            "tools": self.tool_registry.list_tools()
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        logger.info("Resetting metrics")
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "tool_calls": 0,
            "memory_hits": 0,
            "errors": []
        }
        self.agent.reset_metrics()
    
    def export_metrics(self, path: Union[str, Path]) -> None:
        """Export metrics to a file.
        
        Args:
            path: Path to save metrics
        """
        logger.info(f"Exporting metrics to: {path}")
        with open(path, 'w') as f:
            json.dump(self.get_metrics(), f, indent=2)
    
    def __del__(self):
        """Cleanup when the interface is destroyed."""
        logger.info("Cleaning up LLM interface")
        if self.memory_manager:
            self.memory_manager.clear_memories() 