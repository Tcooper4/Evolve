"""
Demo: Pluggable Agents System

This script demonstrates how to use the new pluggable agent system
with dynamic enable/disable functionality and configuration management.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Local imports
from trading.agents.agent_manager import get_agent_manager, register_agent, execute_agent
from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.agents.model_builder_agent import ModelBuilderAgent, ModelBuildRequest
from trading.agents.performance_critic_agent import PerformanceCriticAgent, ModelEvaluationRequest
from trading.agents.updater_agent import UpdaterAgent


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoAgent(BaseAgent):
    """A simple demo agent to show custom agent registration."""
    
    # Agent metadata
    version = "1.0.0"
    description = "A simple demo agent for testing the pluggable system"
    author = "Demo User"
    tags = ["demo", "test"]
    capabilities = ["demo_execution"]
    dependencies = []
    
    def __init__(self, config):
        """Initialize the demo agent."""
        super().__init__(config)
        self.execution_count = 0
    
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the demo agent logic."""
        self.execution_count += 1
        message = kwargs.get('message', 'Hello from demo agent!')
        
        return AgentResult(
            success=True,
            data={
                "message": message,
                "execution_count": self.execution_count,
                "timestamp": self.timestamp.isoformat()
            }
        )


async def demo_basic_usage():
    """Demonstrate basic agent manager usage."""
    logger.info("=== Demo: Basic Agent Manager Usage ===")
    
    # Get the agent manager
    manager = get_agent_manager()
    
    # List all registered agents
    agents = manager.list_agents()
    logger.info(f"Registered agents: {[agent['name'] for agent in agents]}")
    
    # Get agent statuses
    statuses = manager.get_all_agent_statuses()
    for name, status in statuses.items():
        logger.info(f"Agent {name}: enabled={status.enabled}, running={status.is_running}")
    
    # Enable/disable agents
    logger.info("\n--- Enabling/Disabling Agents ---")
    manager.enable_agent("model_builder")
    manager.disable_agent("performance_critic")
    
    # Check status again
    statuses = manager.get_all_agent_statuses()
    for name, status in statuses.items():
        logger.info(f"Agent {name}: enabled={status.enabled}")


async def demo_agent_execution():
    """Demonstrate agent execution."""
    logger.info("\n=== Demo: Agent Execution ===")
    
    # Create a model build request
    build_request = ModelBuildRequest(
        model_type="lstm",
        data_path="data/sample_data.csv",
        target_column="close",
        hyperparameters={"epochs": 10, "batch_size": 32}
    )
    
    # Execute the model builder agent
    logger.info("Executing model builder agent...")
    result = await execute_agent("model_builder", request=build_request)
    
    if result.success:
        logger.info(f"Model built successfully: {result.data}")
    else:
        logger.error(f"Model building failed: {result.error_message}")


async def demo_custom_agent():
    """Demonstrate custom agent registration."""
    logger.info("\n=== Demo: Custom Agent Registration ===")
    
    # Create custom agent configuration
    custom_config = AgentConfig(
        name="demo_agent",
        enabled=True,
        priority=1,
        max_concurrent_runs=1,
        timeout_seconds=60,
        retry_attempts=2,
        custom_config={"demo_param": "demo_value"}
    )
    
    # Register custom agent
    register_agent("demo_agent", DemoAgent, custom_config)
    
    # Execute custom agent
    logger.info("Executing custom demo agent...")
    result = await execute_agent("demo_agent", message="Custom message from demo!")
    
    if result.success:
        logger.info(f"Demo agent executed successfully: {result.data}")
    else:
        logger.error(f"Demo agent failed: {result.error_message}")


async def demo_configuration_management():
    """Demonstrate configuration management."""
    logger.info("\n=== Demo: Configuration Management ===")
    
    manager = get_agent_manager()
    
    # Update agent configuration
    new_config = {
        "max_models": 15,
        "model_types": ["lstm", "xgboost", "ensemble", "transformer"]
    }
    
    success = manager.update_agent_config("model_builder", new_config)
    if success:
        logger.info("Updated model builder configuration")
    
    # Save configuration to file
    manager.save_config()
    logger.info("Saved configuration to file")
    
    # Get execution metrics
    metrics = manager.get_execution_metrics()
    logger.info(f"Execution metrics: {json.dumps(metrics, indent=2)}")


async def demo_agent_workflow():
    """Demonstrate a complete agent workflow."""
    logger.info("\n=== Demo: Complete Agent Workflow ===")
    
    # Step 1: Build a model
    logger.info("Step 1: Building model...")
    build_request = ModelBuildRequest(
        model_type="xgboost",
        data_path="data/sample_data.csv",
        target_column="close"
    )
    
    build_result = await execute_agent("model_builder", request=build_request)
    
    if not build_result.success:
        logger.error("Model building failed, stopping workflow")
        return
    
    model_id = build_result.data["model_id"]
    model_path = build_result.data["model_path"]
    
    # Step 2: Evaluate the model
    logger.info("Step 2: Evaluating model...")
    eval_request = ModelEvaluationRequest(
        model_id=model_id,
        model_path=model_path,
        model_type="xgboost",
        test_data_path="data/test_data.csv"
    )
    
    eval_result = await execute_agent("performance_critic", request=eval_request)
    
    if not eval_result.success:
        logger.error("Model evaluation failed, stopping workflow")
        return
    
    # Step 3: Update model if needed
    logger.info("Step 3: Processing evaluation for potential updates...")
    update_result = await execute_agent("updater", evaluation_result=eval_result.data)
    
    if update_result.success:
        logger.info(f"Workflow completed: {update_result.data}")
    else:
        logger.info("No updates needed or update failed")


async def demo_agent_swapping():
    """Demonstrate agent swapping and replacement."""
    logger.info("\n=== Demo: Agent Swapping ===")
    
    manager = get_agent_manager()
    
    # Show current agents
    agents = manager.list_agents()
    logger.info(f"Current agents: {[agent['name'] for agent in agents]}")
    
    # Unregister an agent
    manager.unregister_agent("performance_critic")
    logger.info("Unregistered performance_critic agent")
    
    # Register a different version or replacement
    # (In a real scenario, you might register a different implementation)
    logger.info("Agent swapping demonstration completed")


def main():
    """Main demo function."""
    logger.info("Starting Pluggable Agents Demo")
    
    # Run all demos
    asyncio.run(demo_basic_usage())
    asyncio.run(demo_agent_execution())
    asyncio.run(demo_custom_agent())
    asyncio.run(demo_configuration_management())
    asyncio.run(demo_agent_workflow())
    asyncio.run(demo_agent_swapping())
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main() 