"""
Task Agent Module

This module provides a TaskAgent class that implements recursive task execution
with performance monitoring and automatic retry logic. The agent can handle
various task types and will recursively attempt to improve performance until
success is achieved or maximum depth is reached.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import uuid

from trading.memory.agent_logger import AgentLogger, AgentType, AgentAction, LogLevel
from memory.prompt_log import get_prompt_memory, log_prompt


class TaskType(Enum):
    """Types of tasks that can be executed."""
    MODEL_BUILD = "model_build"
    MODEL_EVALUATE = "model_evaluate"
    MODEL_UPDATE = "model_update"
    STRATEGY_OPTIMIZE = "strategy_optimize"
    DATA_ANALYSIS = "data_analysis"
    FORECAST_GENERATE = "forecast_generate"
    TRADE_EXECUTE = "trade_execute"
    RISK_ASSESS = "risk_assess"
    GENERAL = "general"


class ActionType(Enum):
    """Types of actions that can be taken."""
    RUN_MODEL = "run_model"
    SCORE_PERFORMANCE = "score_performance"
    UPDATE_PARAMETERS = "update_parameters"
    RETRY_WITH_DIFFERENT_APPROACH = "retry_different_approach"
    FALLBACK_TO_BASELINE = "fallback_baseline"
    STOP_AND_REPORT = "stop_and_report"


@dataclass
class TaskContext:
    """Context information for a task execution."""
    task_id: str
    task_type: TaskType
    prompt: str
    parameters: Dict[str, Any]
    depth: int = 0
    max_depth: int = 5
    performance_threshold: float = 0.7
    parent_task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    performance_score: float
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    error_details: Optional[str] = None
    next_action: Optional[ActionType] = None
    should_continue: bool = True


class ActionStrategy(ABC):
    """Abstract base class for action strategies."""
    
    @abstractmethod
    async def execute(self, context: TaskContext, **kwargs) -> TaskResult:
        """Execute the action strategy."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this action strategy."""
        pass


class RunModelStrategy(ActionStrategy):
    """Strategy for running a model."""
    
    def get_name(self) -> str:
        return "run_model"
    
    async def execute(self, context: TaskContext, **kwargs) -> TaskResult:
        """Execute model running logic."""
        try:
            # Simulate model execution
            model_type = context.parameters.get("model_type", "default")
            symbol = context.parameters.get("symbol", "AAPL")
            
            # Mock performance score (in real implementation, this would be actual model performance)
            performance_score = 0.6 + (context.depth * 0.1)  # Improves with depth
            
            return TaskResult(
                success=performance_score > 0.5,
                performance_score=performance_score,
                data={"model_type": model_type, "symbol": symbol, "prediction": "mock_result"},
                message=f"Model {model_type} executed for {symbol}",
                next_action=ActionType.SCORE_PERFORMANCE if performance_score < context.performance_threshold else ActionType.STOP_AND_REPORT,
                should_continue=performance_score < context.performance_threshold
            )
        except Exception as e:
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=str(e),
                message="Model execution failed",
                next_action=ActionType.RETRY_WITH_DIFFERENT_APPROACH,
                should_continue=True
            )


class ScorePerformanceStrategy(ActionStrategy):
    """Strategy for scoring performance."""
    
    def get_name(self) -> str:
        return "score_performance"
    
    async def execute(self, context: TaskContext, **kwargs) -> TaskResult:
        """Execute performance scoring logic."""
        try:
            # Analyze current performance and determine next action
            current_score = context.performance_history[-1]["score"] if context.performance_history else 0.0
            
            if current_score >= context.performance_threshold:
                return TaskResult(
                    success=True,
                    performance_score=current_score,
                    message="Performance threshold achieved",
                    next_action=ActionType.STOP_AND_REPORT,
                    should_continue=False
                )
            elif context.depth >= context.max_depth:
                return TaskResult(
                    success=False,
                    performance_score=current_score,
                    message="Maximum depth reached",
                    next_action=ActionType.STOP_AND_REPORT,
                    should_continue=False
                )
            else:
                # Determine next action based on performance
                if current_score < 0.3:
                    next_action = ActionType.FALLBACK_TO_BASELINE
                elif current_score < 0.6:
                    next_action = ActionType.UPDATE_PARAMETERS
                else:
                    next_action = ActionType.RETRY_WITH_DIFFERENT_APPROACH
                
                return TaskResult(
                    success=False,
                    performance_score=current_score,
                    message=f"Performance below threshold ({current_score:.2f})",
                    next_action=next_action,
                    should_continue=True
                )
        except Exception as e:
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=str(e),
                message="Performance scoring failed",
                next_action=ActionType.RETRY_WITH_DIFFERENT_APPROACH,
                should_continue=True
            )


class UpdateParametersStrategy(ActionStrategy):
    """Strategy for updating parameters."""
    
    def get_name(self) -> str:
        return "update_parameters"
    
    async def execute(self, context: TaskContext, **kwargs) -> TaskResult:
        """Execute parameter update logic."""
        try:
            # Update parameters based on performance history
            current_params = context.parameters.copy()
            
            # Simple parameter adjustment logic
            if context.performance_history:
                last_score = context.performance_history[-1]["score"]
                if last_score < 0.4:
                    current_params["learning_rate"] = current_params.get("learning_rate", 0.01) * 1.5
                    current_params["epochs"] = current_params.get("epochs", 100) + 50
                elif last_score < 0.6:
                    current_params["batch_size"] = max(16, current_params.get("batch_size", 32) // 2)
            
            return TaskResult(
                success=True,
                performance_score=0.0,  # Will be updated by next action
                data={"updated_parameters": current_params},
                message="Parameters updated successfully",
                next_action=ActionType.RUN_MODEL,
                should_continue=True
            )
        except Exception as e:
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=str(e),
                message="Parameter update failed",
                next_action=ActionType.RETRY_WITH_DIFFERENT_APPROACH,
                should_continue=True
            )


class RetryDifferentApproachStrategy(ActionStrategy):
    """Strategy for retrying with a different approach."""
    
    def get_name(self) -> str:
        return "retry_different_approach"
    
    async def execute(self, context: TaskContext, **kwargs) -> TaskResult:
        """Execute retry with different approach logic."""
        try:
            # Try a different model or strategy
            current_params = context.parameters.copy()
            
            # Switch to different approach based on task type
            if context.task_type == TaskType.MODEL_BUILD:
                models = ["lstm", "xgboost", "ensemble", "transformer"]
                current_model = current_params.get("model_type", "lstm")
                current_index = models.index(current_model) if current_model in models else 0
                next_model = models[(current_index + 1) % len(models)]
                current_params["model_type"] = next_model
            
            return TaskResult(
                success=True,
                performance_score=0.0,  # Will be updated by next action
                data={"new_approach": current_params},
                message=f"Switched to different approach: {current_params.get('model_type', 'unknown')}",
                next_action=ActionType.RUN_MODEL,
                should_continue=True
            )
        except Exception as e:
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=str(e),
                message="Approach change failed",
                next_action=ActionType.FALLBACK_TO_BASELINE,
                should_continue=True
            )


class FallbackBaselineStrategy(ActionStrategy):
    """Strategy for falling back to baseline approach."""
    
    def get_name(self) -> str:
        return "fallback_baseline"
    
    async def execute(self, context: TaskContext, **kwargs) -> TaskResult:
        """Execute fallback to baseline logic."""
        try:
            # Use baseline/simple approach
            baseline_params = {
                "model_type": "simple_linear",
                "learning_rate": 0.01,
                "epochs": 50,
                "batch_size": 32
            }
            
            return TaskResult(
                success=True,
                performance_score=0.0,  # Will be updated by next action
                data={"baseline_parameters": baseline_params},
                message="Falling back to baseline approach",
                next_action=ActionType.RUN_MODEL,
                should_continue=True
            )
        except Exception as e:
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=str(e),
                message="Baseline fallback failed",
                next_action=ActionType.STOP_AND_REPORT,
                should_continue=False
            )


class StopAndReportStrategy(ActionStrategy):
    """Strategy for stopping and reporting results."""
    
    def get_name(self) -> str:
        return "stop_and_report"
    
    async def execute(self, context: TaskContext, **kwargs) -> TaskResult:
        """Execute stop and report logic."""
        try:
            # Compile final results
            final_score = context.performance_history[-1]["score"] if context.performance_history else 0.0
            success = final_score >= context.performance_threshold
            
            return TaskResult(
                success=success,
                performance_score=final_score,
                data={
                    "final_performance": final_score,
                    "total_actions": len(context.action_history),
                    "depth_reached": context.depth,
                    "performance_history": context.performance_history
                },
                message=f"Task completed with final score: {final_score:.3f}",
                should_continue=False
            )
        except Exception as e:
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=str(e),
                message="Final reporting failed",
                should_continue=False
            )


class TaskAgent:
    """
    Task Agent that implements recursive task execution with performance monitoring.
    
    This agent receives prompts and task types, logs them to memory, chooses actions
    based on performance, and recursively calls itself until success is achieved
    or maximum depth is reached.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the TaskAgent."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.agent_logger = AgentLogger()
        self.prompt_memory = get_prompt_memory()
        
        # Initialize action strategies
        self.action_strategies: Dict[ActionType, ActionStrategy] = {
            ActionType.RUN_MODEL: RunModelStrategy(),
            ActionType.SCORE_PERFORMANCE: ScorePerformanceStrategy(),
            ActionType.UPDATE_PARAMETERS: UpdateParametersStrategy(),
            ActionType.RETRY_WITH_DIFFERENT_APPROACH: RetryDifferentApproachStrategy(),
            ActionType.FALLBACK_TO_BASELINE: FallbackBaselineStrategy(),
            ActionType.STOP_AND_REPORT: StopAndReportStrategy(),
        }
        
        # Task registry for tracking
        self.task_registry: Dict[str, TaskContext] = {}
        
        self.logger.info("TaskAgent initialized successfully with prompt memory")
    
    async def execute_task(
        self,
        prompt: str,
        task_type: TaskType,
        parameters: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
        performance_threshold: float = 0.7,
        parent_task_id: Optional[str] = None
    ) -> TaskResult:
        """
        Execute a task with recursive improvement logic.
        
        Args:
            prompt: The task prompt
            task_type: Type of task to execute
            parameters: Task parameters
            max_depth: Maximum recursion depth
            performance_threshold: Performance threshold for success
            parent_task_id: ID of parent task (for nested tasks)
            
        Returns:
            TaskResult: Final result of the task execution
        """
        start_time = datetime.now()
        
        # Create task context
        task_id = str(uuid.uuid4())
        context = TaskContext(
            task_id=task_id,
            task_type=task_type,
            prompt=prompt,
            parameters=parameters or {},
            max_depth=max_depth,
            performance_threshold=performance_threshold,
            parent_task_id=parent_task_id
        )
        
        # Register task
        self.task_registry[task_id] = context
        
        # Log task start
        self.agent_logger.log_action(
            agent_name="TaskAgent",
            action=AgentAction.MODEL_SYNTHESIS,
            message=f"Starting task: {task_type.value}",
            data={
                "task_id": task_id,
                "task_type": task_type.value,
                "prompt": prompt,
                "parameters": parameters,
                "depth": context.depth
            },
            level=LogLevel.INFO
        )
        
        # Log initial task to prompt memory
        await self.prompt_memory.log_prompt(
            prompt=prompt,
            result={
                "task_id": task_id,
                "task_type": task_type.value,
                "status": "started",
                "parameters": parameters or {},
                "max_depth": max_depth,
                "performance_threshold": performance_threshold
            },
            session_id=task_id,
            user_id="task_agent",
            agent_type="TaskAgent",
            execution_time=0.0,
            success=True,
            metadata={
                "task_type": task_type.value,
                "parent_task_id": parent_task_id,
                "initial_depth": context.depth
            }
        )
        
        # Execute task recursively
        result = await self._execute_task_recursive(context)
        
        # Calculate total execution time
        total_execution_time = (datetime.now() - start_time).total_seconds()
        
        # Log task completion
        self.agent_logger.log_action(
            agent_name="TaskAgent",
            action=AgentAction.MODEL_SYNTHESIS,
            message=f"Task completed: {task_type.value}",
            data={
                "task_id": task_id,
                "final_success": result.success,
                "final_score": result.performance_score,
                "total_actions": len(context.action_history),
                "depth_reached": context.depth,
                "total_execution_time": total_execution_time
            },
            level=LogLevel.INFO if result.success else LogLevel.WARNING
        )
        
        # Log final task result to prompt memory
        await self.prompt_memory.log_prompt(
            prompt=f"Task completed: {prompt}",
            result={
                "task_id": task_id,
                "task_type": task_type.value,
                "status": "completed",
                "final_success": result.success,
                "final_performance_score": result.performance_score,
                "total_actions": len(context.action_history),
                "depth_reached": context.depth,
                "total_execution_time": total_execution_time,
                "action_history": context.action_history,
                "performance_history": context.performance_history
            },
            session_id=task_id,
            user_id="task_agent",
            agent_type="TaskAgent",
            execution_time=total_execution_time,
            success=result.success,
            metadata={
                "task_type": task_type.value,
                "final_depth": context.depth,
                "total_actions": len(context.action_history)
            }
        )
        
        return result
    
    async def _execute_task_recursive(self, context: TaskContext) -> TaskResult:
        """
        Recursively execute a task until success or max depth.
        
        Args:
            context: Task context
            
        Returns:
            TaskResult: Result of the task execution
        """
        # Check if we've reached max depth
        if context.depth >= context.max_depth:
            return TaskResult(
                success=False,
                performance_score=0.0,
                message=f"Maximum depth ({context.max_depth}) reached",
                should_continue=False
            )
        
        # Determine initial action
        if context.depth == 0:
            # First iteration - start with running the model
            action_type = ActionType.RUN_MODEL
        else:
            # Use the next action from previous result
            action_type = context.action_history[-1].get("next_action", ActionType.STOP_AND_REPORT)
        
        # Execute the action
        result = await self._execute_action(context, action_type)
        
        # Log the action
        self._log_action(context, action_type, result)
        
        # Update context
        context.action_history.append({
            "action": action_type.value,
            "timestamp": datetime.now().isoformat(),
            "result": {
                "success": result.success,
                "performance_score": result.performance_score,
                "message": result.message
            },
            "next_action": result.next_action.value if result.next_action else None
        })
        
        if result.data:
            context.parameters.update(result.data)
        
        # Check if we should continue
        if not result.should_continue:
            return result
        
        # Recursively continue if needed
        context.depth += 1
        return await self._execute_task_recursive(context)
    
    async def _execute_action(self, context: TaskContext, action_type: ActionType) -> TaskResult:
        """
        Execute a specific action.
        
        Args:
            context: Task context
            action_type: Type of action to execute
            
        Returns:
            TaskResult: Result of the action
        """
        strategy = self.action_strategies.get(action_type)
        if not strategy:
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=f"Unknown action type: {action_type}",
                message="Unknown action type",
                next_action=ActionType.STOP_AND_REPORT,
                should_continue=False
            )
        
        start_time = datetime.now()
        
        try:
            result = await strategy.execute(context)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance history if we have a score
            if result.performance_score > 0:
                context.performance_history.append({
                    "score": result.performance_score,
                    "timestamp": datetime.now().isoformat(),
                    "action": action_type.value
                })
            
            # Log to prompt memory
            await self.prompt_memory.log_prompt(
                prompt=f"Task: {context.task_type.value} - Action: {action_type.value}",
                result={
                    "task_id": context.task_id,
                    "action_type": action_type.value,
                    "success": result.success,
                    "performance_score": result.performance_score,
                    "message": result.message,
                    "next_action": result.next_action.value if result.next_action else None,
                    "depth": context.depth,
                    "parameters": context.parameters
                },
                session_id=context.task_id,
                user_id="task_agent",
                agent_type="TaskAgent",
                execution_time=execution_time,
                success=result.success,
                metadata={
                    "task_type": context.task_type.value,
                    "action_type": action_type.value,
                    "depth": context.depth,
                    "strategy": strategy.get_name()
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error executing action {action_type}: {e}")
            
            # Log error to prompt memory
            await self.prompt_memory.log_prompt(
                prompt=f"Task: {context.task_type.value} - Action: {action_type.value}",
                result={
                    "task_id": context.task_id,
                    "action_type": action_type.value,
                    "success": False,
                    "error": str(e),
                    "depth": context.depth
                },
                session_id=context.task_id,
                user_id="task_agent",
                agent_type="TaskAgent",
                execution_time=execution_time,
                success=False,
                metadata={
                    "task_type": context.task_type.value,
                    "action_type": action_type.value,
                    "depth": context.depth,
                    "error_type": type(e).__name__
                }
            )
            
            return TaskResult(
                success=False,
                performance_score=0.0,
                error_details=str(e),
                message=f"Action {action_type.value} failed",
                next_action=ActionType.RETRY_WITH_DIFFERENT_APPROACH,
                should_continue=True
            )
    
    def _log_action(self, context: TaskContext, action_type: ActionType, result: TaskResult):
        """Log an action execution."""
        self.agent_logger.log_action(
            agent_name="TaskAgent",
            action=AgentAction.MODEL_SYNTHESIS,
            message=f"Action executed: {action_type.value}",
            data={
                "task_id": context.task_id,
                "action": action_type.value,
                "depth": context.depth,
                "success": result.success,
                "performance_score": result.performance_score,
                "message": result.message
            },
            level=LogLevel.INFO if result.success else LogLevel.WARNING
        )
    
    def get_task_history(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the history of a specific task."""
        context = self.task_registry.get(task_id)
        if not context:
            return None
        
        return {
            "task_id": context.task_id,
            "task_type": context.task_type.value,
            "prompt": context.prompt,
            "depth": context.depth,
            "max_depth": context.max_depth,
            "performance_threshold": context.performance_threshold,
            "created_at": context.created_at.isoformat(),
            "action_history": context.action_history,
            "performance_history": context.performance_history
        }
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all task histories."""
        return [self.get_task_history(task_id) for task_id in self.task_registry.keys()]
    
    def clear_task_history(self):
        """Clear all task history."""
        self.task_registry.clear()


# Global instance
_task_agent: Optional[TaskAgent] = None


def get_task_agent() -> TaskAgent:
    """Get the global TaskAgent instance."""
    global _task_agent
    if _task_agent is None:
        _task_agent = TaskAgent()
    return _task_agent


async def execute_task(
    prompt: str,
    task_type: TaskType,
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> TaskResult:
    """Convenience function to execute a task."""
    agent = get_task_agent()
    return await agent.execute_task(prompt, task_type, parameters, **kwargs) 