"""
Task Delegation Agent for orchestrating multiple agents with specific roles.
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from trading.agents.base_agent_interface import BaseAgent, AgentConfig, AgentResult
from trading.memory.agent_memory import AgentMemory
from trading.memory.agent_thoughts_logger import log_agent_thought

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskDelegationRequest:
    """Task delegation request."""
    action: str  # 'delegate_task', 'delegate_workflow', 'get_task_status', 'cancel_task', 'get_agent_status', 'register_agent'
    task_description: Optional[str] = None
    workflow: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    agent_name: Optional[str] = None
    roles: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    priority: Optional[str] = None
    timeout: Optional[int] = None

@dataclass
class TaskDelegationResult:
    """Task delegation result."""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class Task:
    """Represents a task to be delegated."""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    assigned_agent: Optional[str] = None
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentRole(Enum):
    """Agent roles for task delegation."""
    FORECASTER = "forecaster"
    OPTIMIZER = "optimizer"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    COORDINATOR = "coordinator"

class TaskDelegationAgent(BaseAgent):
    """
    Agent responsible for delegating tasks across multiple agents with specific roles.
    
    This agent orchestrates complex workflows by breaking them down into tasks
    and assigning them to appropriate agents based on their roles and capabilities.
    """
    
    def __init__(self, name: str = "task_delegator", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Task Delegation Agent.
        
        Args:
            name: Agent name
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # Initialize components
        self.memory = AgentMemory()
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        
        # Agent registry with roles
        self.agent_roles: Dict[str, List[AgentRole]] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Delegation settings
        self.max_concurrent_tasks = config.get('max_concurrent_tasks', 5)
        self.task_timeout = config.get('task_timeout', 3600)  # 1 hour
        self.retry_attempts = config.get('retry_attempts', 3)
        
        # Initialize default agent roles
        self._initialize_default_roles()
        
        logger.info(f"Initialized TaskDelegationAgent with {len(self.agent_roles)} registered agents")
    
    def _initialize_default_roles(self) -> None:
        """Initialize default agent roles and capabilities."""
        try:
            # Define default agent roles
            self.agent_roles = {
                "model_builder": [AgentRole.FORECASTER, AgentRole.OPTIMIZER],
                "strategy_selector": [AgentRole.ANALYZER, AgentRole.REVIEWER],
                "execution_agent": [AgentRole.EXECUTOR],
                "risk_manager": [AgentRole.REVIEWER, AgentRole.ANALYZER],
                "performance_analyzer": [AgentRole.ANALYZER, AgentRole.REVIEWER],
                "model_improver": [AgentRole.OPTIMIZER],
                "strategy_improver": [AgentRole.OPTIMIZER],
                "goal_planner": [AgentRole.COORDINATOR],
                "meta_learning_feedback": [AgentRole.ANALYZER, AgentRole.OPTIMIZER]
            }
            
            # Define agent capabilities
            self.agent_capabilities = {
                "model_builder": ["build_models", "train_models", "evaluate_models"],
                "strategy_selector": ["select_strategies", "analyze_performance", "compare_strategies"],
                "execution_agent": ["execute_trades", "manage_positions", "risk_control"],
                "risk_manager": ["assess_risk", "set_limits", "monitor_exposure"],
                "performance_analyzer": ["analyze_performance", "generate_reports", "identify_issues"],
                "model_improver": ["tune_hyperparameters", "optimize_models", "improve_accuracy"],
                "strategy_improver": ["adjust_parameters", "optimize_strategies", "improve_returns"],
                "goal_planner": ["plan_goals", "coordinate_tasks", "track_progress"],
                "meta_learning_feedback": ["learn_from_feedback", "adapt_strategies", "improve_decisions"]
            }
            
        except Exception as e:
            logger.error(f"Error initializing default roles: {str(e)}")
    
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the task delegation logic.
        
        Args:
            **kwargs: action, task_description, workflow, etc.
            
        Returns:
            AgentResult: Result of the delegation process
        """
        try:
            action = kwargs.get('action', 'delegate_task')
            
            if action == 'delegate_task':
                task_description = kwargs.get('task_description')
                if not task_description:
                    return AgentResult(success=False, error_message="Missing task_description")
                return await self._delegate_single_task(task_description, **kwargs)
            
            elif action == 'delegate_workflow':
                workflow = kwargs.get('workflow')
                if not workflow:
                    return AgentResult(success=False, error_message="Missing workflow")
                return await self._delegate_workflow(workflow, **kwargs)
            
            elif action == 'get_task_status':
                task_id = kwargs.get('task_id')
                if not task_id:
                    return AgentResult(success=False, error_message="Missing task_id")
                return self._get_task_status(task_id)
            
            elif action == 'cancel_task':
                task_id = kwargs.get('task_id')
                if not task_id:
                    return AgentResult(success=False, error_message="Missing task_id")
                return self._cancel_task(task_id)
            
            elif action == 'get_agent_status':
                return self._get_agent_status()
            
            elif action == 'register_agent':
                agent_name = kwargs.get('agent_name')
                roles = kwargs.get('roles', [])
                capabilities = kwargs.get('capabilities', [])
                if not agent_name:
                    return AgentResult(success=False, error_message="Missing agent_name")
                return self._register_agent(agent_name, roles, capabilities)
            
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            return self.handle_error(e)
    
    async def _delegate_single_task(self, task_description: str, **kwargs) -> AgentResult:
        """Delegate a single task to an appropriate agent."""
        try:
            # Create task
            task = self._create_task(task_description, **kwargs)
            
            # Find appropriate agent
            assigned_agent = self._find_best_agent(task)
            
            if not assigned_agent:
                return AgentResult(
                    success=False,
                    error_message=f"No suitable agent found for task: {task.title}"
                )
            
            # Assign task
            task.assigned_agent = assigned_agent
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now().isoformat()
            
            # Store task
            self.tasks[task.task_id] = task
            self.task_queue.append(task.task_id)
            
            # Log delegation
            log_agent_thought(
                agent_name=self.name,
                context=f"Delegating task: {task.title}",
                decision=f"Assigned to {assigned_agent}",
                rationale=f"Agent {assigned_agent} has roles {self.agent_roles.get(assigned_agent, [])} and capabilities {self.agent_capabilities.get(assigned_agent, [])}",
                confidence=0.8
            )
            
            # Execute task (simulated for now)
            task_result = await self._execute_task(task)
            
            return AgentResult(
                success=True,
                data={
                    'task_id': task.task_id,
                    'assigned_agent': assigned_agent,
                    'task_status': task.status.value,
                    'result': task_result
                }
            )
            
        except Exception as e:
            logger.error(f"Error delegating task: {str(e)}")
            return AgentResult(success=False, error_message=str(e))
    
    async def _delegate_workflow(self, workflow: Dict[str, Any], **kwargs) -> AgentResult:
        """Delegate a complex workflow with multiple tasks."""
        try:
            workflow_id = str(uuid.uuid4())
            workflow_tasks = workflow.get('tasks', [])
            dependencies = workflow.get('dependencies', {})
            
            if not workflow_tasks:
                return AgentResult(success=False, error_message="No tasks in workflow")
            
            # Create tasks for workflow
            created_tasks = []
            for task_def in workflow_tasks:
                task = self._create_task(
                    task_def.get('description', ''),
                    priority=TaskPriority(task_def.get('priority', 'medium')),
                    metadata={
                        'workflow_id': workflow_id,
                        'dependencies': task_def.get('dependencies', [])
                    }
                )
                created_tasks.append(task)
                self.tasks[task.task_id] = task
            
            # Execute workflow
            workflow_result = await self._execute_workflow(created_tasks, dependencies)
            
            return AgentResult(
                success=True,
                data={
                    'workflow_id': workflow_id,
                    'tasks_created': len(created_tasks),
                    'workflow_result': workflow_result
                }
            )
            
        except Exception as e:
            logger.error(f"Error delegating workflow: {str(e)}")
            return AgentResult(success=False, error_message=str(e))
    
    def _create_task(self, description: str, **kwargs) -> Task:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        
        return Task(
            task_id=task_id,
            title=kwargs.get('title', description[:50]),
            description=description,
            priority=TaskPriority(kwargs.get('priority', 'medium')),
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
            metadata=kwargs.get('metadata', {})
        )
    
    def _find_best_agent(self, task: Task) -> Optional[str]:
        """Find the best agent for a given task."""
        try:
            # Analyze task requirements
            task_requirements = self._analyze_task_requirements(task)
            
            best_agent = None
            best_score = 0.0
            
            for agent_name, roles in self.agent_roles.items():
                # Check if agent is available
                if not self._is_agent_available(agent_name):
                    continue
                
                # Calculate suitability score
                score = self._calculate_agent_suitability(agent_name, roles, task_requirements)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_name
            
            return best_agent
            
        except Exception as e:
            logger.error(f"Error finding best agent: {str(e)}")
            return None
    
    def _analyze_task_requirements(self, task: Task) -> Dict[str, Any]:
        """Analyze task requirements to determine needed roles and capabilities."""
        try:
            requirements = {
                'roles': [],
                'capabilities': [],
                'complexity': 'medium'
            }
            
            description_lower = task.description.lower()
            
            # Determine required roles based on task description
            if any(word in description_lower for word in ['forecast', 'predict', 'model']):
                requirements['roles'].append(AgentRole.FORECASTER)
            
            if any(word in description_lower for word in ['optimize', 'tune', 'improve']):
                requirements['roles'].append(AgentRole.OPTIMIZER)
            
            if any(word in description_lower for word in ['analyze', 'evaluate', 'assess']):
                requirements['roles'].append(AgentRole.ANALYZER)
            
            if any(word in description_lower for word in ['execute', 'trade', 'order']):
                requirements['roles'].append(AgentRole.EXECUTOR)
            
            if any(word in description_lower for word in ['review', 'validate', 'check']):
                requirements['roles'].append(AgentRole.REVIEWER)
            
            if any(word in description_lower for word in ['coordinate', 'plan', 'manage']):
                requirements['roles'].append(AgentRole.COORDINATOR)
            
            # Determine complexity
            if task.priority == TaskPriority.CRITICAL:
                requirements['complexity'] = 'high'
            elif task.priority == TaskPriority.LOW:
                requirements['complexity'] = 'low'
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error analyzing task requirements: {str(e)}")
            return {'roles': [], 'capabilities': [], 'complexity': 'medium'}
    
    def _is_agent_available(self, agent_name: str) -> bool:
        """Check if an agent is available for new tasks."""
        try:
            # Count current tasks for this agent
            current_tasks = sum(
                1 for task in self.tasks.values()
                if task.assigned_agent == agent_name and 
                task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
            )
            
            return current_tasks < self.max_concurrent_tasks
            
        except Exception as e:
            logger.error(f"Error checking agent availability: {str(e)}")
            return False
    
    def _calculate_agent_suitability(self, 
                                   agent_name: str, 
                                   agent_roles: List[AgentRole],
                                   task_requirements: Dict[str, Any]) -> float:
        """Calculate how suitable an agent is for a task."""
        try:
            score = 0.0
            
            # Role matching
            required_roles = task_requirements.get('roles', [])
            matching_roles = set(agent_roles) & set(required_roles)
            
            if matching_roles:
                score += len(matching_roles) / len(required_roles) * 0.6
            
            # Capability matching
            agent_capabilities = self.agent_capabilities.get(agent_name, [])
            required_capabilities = task_requirements.get('capabilities', [])
            
            if required_capabilities:
                matching_capabilities = set(agent_capabilities) & set(required_capabilities)
                score += len(matching_capabilities) / len(required_capabilities) * 0.3
            
            # Agent performance history (simplified)
            score += 0.1  # Base score for all agents
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating agent suitability: {str(e)}")
            return 0.0
    
    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task (simulated for now)."""
        try:
            # Simulate task execution
            await asyncio.sleep(1)  # Simulate work
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = {
                'message': f'Task {task.title} completed successfully',
                'agent': task.assigned_agent,
                'execution_time': '1 second'
            }
            
            # Log completion
            log_agent_thought(
                agent_name=task.assigned_agent or "unknown",
                context=f"Executing task: {task.title}",
                decision="Task completed successfully",
                rationale=f"Task was executed by {task.assigned_agent}",
                confidence=0.9
            )
            
            return task.result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            logger.error(f"Error executing task {task.task_id}: {str(e)}")
            return {'error': str(e)}
    
    async def _execute_workflow(self, tasks: List[Task], dependencies: Dict[str, List[str]]) -> Dict[str, Any]:
        """Execute a workflow with dependencies."""
        try:
            # Simple workflow execution (no dependency resolution for now)
            results = []
            
            for task in tasks:
                result = await self._execute_task(task)
                results.append({
                    'task_id': task.task_id,
                    'result': result
                })
            
            return {
                'workflow_completed': True,
                'tasks_executed': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            return {'error': str(e)}
    
    def _get_task_status(self, task_id: str) -> AgentResult:
        """Get the status of a specific task."""
        try:
            if task_id not in self.tasks:
                return AgentResult(success=False, error_message=f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            
            return AgentResult(
                success=True,
                data={
                    'task_id': task_id,
                    'status': task.status.value,
                    'assigned_agent': task.assigned_agent,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'result': task.result,
                    'error_message': task.error_message
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            return AgentResult(success=False, error_message=str(e))
    
    def _cancel_task(self, task_id: str) -> AgentResult:
        """Cancel a task."""
        try:
            if task_id not in self.tasks:
                return AgentResult(success=False, error_message=f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return AgentResult(success=False, error_message=f"Task {task_id} cannot be cancelled")
            
            task.status = TaskStatus.CANCELLED
            
            return AgentResult(
                success=True,
                data={'message': f'Task {task_id} cancelled successfully'}
            )
            
        except Exception as e:
            logger.error(f"Error cancelling task: {str(e)}")
            return AgentResult(success=False, error_message=str(e))
    
    def _get_agent_status(self) -> AgentResult:
        """Get status of all agents."""
        try:
            agent_status = {}
            
            for agent_name in self.agent_roles.keys():
                current_tasks = sum(
                    1 for task in self.tasks.values()
                    if task.assigned_agent == agent_name and 
                    task.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
                )
                
                agent_status[agent_name] = {
                    'roles': [role.value for role in self.agent_roles.get(agent_name, [])],
                    'capabilities': self.agent_capabilities.get(agent_name, []),
                    'current_tasks': current_tasks,
                    'available': current_tasks < self.max_concurrent_tasks
                }
            
            return AgentResult(
                success=True,
                data={
                    'agent_status': agent_status,
                    'total_tasks': len(self.tasks),
                    'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                    'active_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting agent status: {str(e)}")
            return AgentResult(success=False, error_message=str(e))
    
    def _register_agent(self, 
                       agent_name: str, 
                       roles: List[str], 
                       capabilities: List[str]) -> AgentResult:
        """Register a new agent with roles and capabilities."""
        try:
            # Convert role strings to AgentRole enums
            agent_roles = []
            for role_str in roles:
                try:
                    agent_roles.append(AgentRole(role_str))
                except ValueError:
                    logger.warning(f"Unknown role: {role_str}")
            
            # Register agent
            self.agent_roles[agent_name] = agent_roles
            self.agent_capabilities[agent_name] = capabilities
            
            logger.info(f"Registered agent {agent_name} with roles {roles} and capabilities {capabilities}")
            
            return AgentResult(
                success=True,
                data={
                    'agent_name': agent_name,
                    'roles': roles,
                    'capabilities': capabilities,
                    'message': f'Agent {agent_name} registered successfully'
                }
            )
            
        except Exception as e:
            logger.error(f"Error registering agent: {str(e)}")
            return AgentResult(success=False, error_message=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        base_status = super().get_status()
        base_status.update({
            'total_tasks': len(self.tasks),
            'registered_agents': len(self.agent_roles),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'task_timeout': self.task_timeout,
            'retry_attempts': self.retry_attempts
        })
        return base_status 