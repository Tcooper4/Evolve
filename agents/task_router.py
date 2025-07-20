"""
Task Router with Batch 12 Features

This module handles intelligent task routing for multi-intent prompts,
sequentially routing to forecast â†’ strategy â†’ report based on prompt analysis.

Features:
- Multi-intent prompt parsing and decomposition
- Sequential routing with dependency management
- Priority-based execution ordering
- Context preservation across task chains
- Result aggregation and synthesis
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses_json import dataclass_json

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be routed."""
    
    FORECAST = "forecast"
    STRATEGY = "strategy"
    REPORT = "report"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    GENERAL = "general"


class TaskPriority(Enum):
    """Task priority levels."""
    
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskStatus(Enum):
    """Task execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEPENDENCY_WAIT = "dependency_wait"


@dataclass_json
@dataclass
class TaskContext:
    """Context information for task execution."""
    
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    previous_tasks: List[str] = field(default_factory=list)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class TaskDefinition:
    """Definition of a task to be executed."""
    
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 30.0  # seconds
    max_retries: int = 3
    timeout: float = 300.0  # seconds
    context: Optional[TaskContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class TaskResult:
    """Result of task execution."""
    
    task_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class TaskExecution:
    """Task execution instance."""
    
    task: TaskDefinition
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    error_history: List[str] = field(default_factory=list)


@dataclass_json
@dataclass
class MultiIntentPrompt:
    """Parsed multi-intent prompt."""
    
    original_prompt: str
    intents: List[TaskType] = field(default_factory=list)
    confidence: float = 0.0
    task_chain: List[TaskDefinition] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    context: TaskContext = field(default_factory=TaskContext)


class TaskRouter:
    """Intelligent task router for multi-intent prompts."""
    
    def __init__(self):
        """Initialize the task router."""
        self.logger = logging.getLogger(__name__)
        
        # Task execution tracking
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.completed_tasks: Dict[str, TaskExecution] = {}
        self.failed_tasks: Dict[str, TaskExecution] = {}
        
        # Task routing patterns
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Task dependencies and execution order
        self.task_dependencies = self._initialize_task_dependencies()
        
        # Performance tracking
        self.execution_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time": 0.0,
            "task_type_stats": {}
        }
        
        # Background task management
        self.running = False
        self.background_tasks: Set[asyncio.Task] = set()
    
    def _initialize_intent_patterns(self) -> Dict[TaskType, List[str]]:
        """Initialize patterns for intent detection."""
        return {
            TaskType.FORECAST: [
                r"\b(forecast|predict|price|stock|market)\b",
                r"\b(next|future|upcoming)\s+\d+\s+(day|week|month)",
                r"\b(what|how)\s+will\s+\w+\s+(perform|move|go)",
                r"\b(price|value)\s+(prediction|forecast|estimate)"
            ],
            TaskType.STRATEGY: [
                r"\b(strategy|strategy|trade|signal|position)\b",
                r"\b(buy|sell|hold|entry|exit)\b",
                r"\b(trading|investment)\s+(strategy|plan|approach)",
                r"\b(risk|reward|profit|loss)\s+(management|strategy)"
            ],
            TaskType.REPORT: [
                r"\b(report|summary|analysis|review)\b",
                r"\b(performance|results|outcomes)\s+(report|summary)",
                r"\b(what|how)\s+(did|has)\s+\w+\s+(perform|do)",
                r"\b(generate|create|make)\s+(report|summary|analysis)"
            ],
            TaskType.ANALYSIS: [
                r"\b(analyze|analysis|examine|study)\b",
                r"\b(market|trend|pattern|indicator)\s+(analysis)",
                r"\b(technical|fundamental)\s+(analysis)",
                r"\b(what|why|how)\s+(is|are)\s+\w+\s+(performing|moving)"
            ],
            TaskType.OPTIMIZATION: [
                r"\b(optimize|optimization|improve|enhance)\b",
                r"\b(best|optimal|maximum|minimum)\s+(performance|return)",
                r"\b(tune|adjust|fine-tune)\s+(parameters|settings)",
                r"\b(portfolio|allocation)\s+(optimization)"
            ],
            TaskType.PORTFOLIO: [
                r"\b(portfolio|position|allocation|diversification)\b",
                r"\b(manage|rebalance|adjust)\s+(portfolio|positions)",
                r"\b(risk|exposure)\s+(management|control)",
                r"\b(asset|investment)\s+(allocation|distribution)"
            ],
            TaskType.SYSTEM: [
                r"\b(system|status|health|monitor)\b",
                r"\b(performance|metrics|statistics)\s+(system)",
                r"\b(what|how)\s+(is|are)\s+(system|platform)\s+(performing)",
                r"\b(check|verify|test)\s+(system|connection)"
            ]
        }
    
    def _initialize_task_dependencies(self) -> Dict[TaskType, List[TaskType]]:
        """Initialize task dependencies."""
        return {
            TaskType.STRATEGY: [TaskType.FORECAST, TaskType.ANALYSIS],
            TaskType.REPORT: [TaskType.FORECAST, TaskType.STRATEGY, TaskType.ANALYSIS],
            TaskType.OPTIMIZATION: [TaskType.ANALYSIS, TaskType.PORTFOLIO],
            TaskType.PORTFOLIO: [TaskType.ANALYSIS],
            TaskType.GENERAL: []
        }
    
    def parse_multi_intent_prompt(self, prompt: str, context: Optional[TaskContext] = None) -> MultiIntentPrompt:
        """Parse a multi-intent prompt and extract task definitions.
        
        Args:
            prompt: User prompt
            context: Optional context information
            
        Returns:
            MultiIntentPrompt: Parsed multi-intent prompt
        """
        if context is None:
            context = TaskContext()
        
        # Detect intents
        detected_intents = self._detect_intents(prompt)
        
        # Create task definitions
        task_chain = []
        for intent in detected_intents:
            task_def = self._create_task_definition(prompt, intent, context)
            task_chain.append(task_def)
        
        # Determine execution order
        execution_order = self._determine_execution_order(task_chain)
        
        # Build dependencies
        dependencies = self._build_dependencies(task_chain)
        
        # Calculate confidence
        confidence = self._calculate_confidence(detected_intents, prompt)
        
        return MultiIntentPrompt(
            original_prompt=prompt,
            intents=detected_intents,
            confidence=confidence,
            task_chain=task_chain,
            execution_order=execution_order,
            dependencies=dependencies,
            context=context
        )
    
    def _detect_intents(self, prompt: str) -> List[TaskType]:
        """Detect intents in the prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            List of detected task types
        """
        detected_intents = []
        prompt_lower = prompt.lower()
        
        for task_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    detected_intents.append(task_type)
                    break
        
        # If no intents detected, default to general
        if not detected_intents:
            detected_intents.append(TaskType.GENERAL)
        
        # Remove duplicates while preserving order
        unique_intents = []
        for intent in detected_intents:
            if intent not in unique_intents:
                unique_intents.append(intent)
        
        return unique_intents
    
    def _create_task_definition(
        self, 
        prompt: str, 
        task_type: TaskType, 
        context: TaskContext
    ) -> TaskDefinition:
        """Create a task definition from intent.
        
        Args:
            prompt: Original prompt
            task_type: Detected task type
            context: Task context
            
        Returns:
            TaskDefinition: Created task definition
        """
        task_id = f"{task_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Extract parameters based on task type
        parameters = self._extract_parameters(prompt, task_type)
        
        # Determine priority
        priority = self._determine_priority(task_type, context)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(task_type, parameters)
        
        return TaskDefinition(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            prompt=prompt,
            parameters=parameters,
            estimated_duration=estimated_duration,
            context=context
        )
    
    def _extract_parameters(self, prompt: str, task_type: TaskType) -> Dict[str, Any]:
        """Extract parameters from prompt based on task type.
        
        Args:
            prompt: User prompt
            task_type: Task type
            
        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        
        # Extract common parameters
        # Stock symbols
        symbol_pattern = r'\b[A-Z]{1,5}\b'
        symbols = re.findall(symbol_pattern, prompt.upper())
        if symbols:
            parameters["symbols"] = symbols
        
        # Timeframes
        timeframe_pattern = r'\b(\d+)\s*(day|week|month|year)s?\b'
        timeframes = re.findall(timeframe_pattern, prompt.lower())
        if timeframes:
            parameters["timeframe"] = f"{timeframes[0][0]} {timeframes[0][1]}"
        
        # Amounts
        amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        amounts = re.findall(amount_pattern, prompt)
        if amounts:
            parameters["amount"] = float(amounts[0].replace(",", ""))
        
        # Task-specific parameters
        if task_type == TaskType.FORECAST:
            # Extract forecast-specific parameters
            if "price" in prompt.lower():
                parameters["forecast_type"] = "price"
            if "trend" in prompt.lower():
                parameters["forecast_type"] = "trend"
        
        elif task_type == TaskType.STRATEGY:
            # Extract strategy-specific parameters
            if "risk" in prompt.lower():
                parameters["risk_level"] = "medium"  # Default
            if "conservative" in prompt.lower():
                parameters["risk_level"] = "low"
            elif "aggressive" in prompt.lower():
                parameters["risk_level"] = "high"
        
        elif task_type == TaskType.REPORT:
            # Extract report-specific parameters
            if "detailed" in prompt.lower():
                parameters["report_type"] = "detailed"
            elif "summary" in prompt.lower():
                parameters["report_type"] = "summary"
        
        return parameters
    
    def _determine_priority(self, task_type: TaskType, context: TaskContext) -> TaskPriority:
        """Determine task priority.
        
        Args:
            task_type: Task type
            context: Task context
            
        Returns:
            TaskPriority: Determined priority
        """
        # System tasks are critical
        if task_type == TaskType.SYSTEM:
            return TaskPriority.CRITICAL
        
        # Analysis and forecast are high priority
        if task_type in [TaskType.ANALYSIS, TaskType.FORECAST]:
            return TaskPriority.HIGH
        
        # Strategy and optimization are medium priority
        if task_type in [TaskType.STRATEGY, TaskType.OPTIMIZATION]:
            return TaskPriority.MEDIUM
        
        # Reports and portfolio are lower priority
        if task_type in [TaskType.REPORT, TaskType.PORTFOLIO]:
            return TaskPriority.LOW
        
        # General tasks are lowest priority
        return TaskPriority.LOW
    
    def _estimate_duration(self, task_type: TaskType, parameters: Dict[str, Any]) -> float:
        """Estimate task execution duration.
        
        Args:
            task_type: Task type
            parameters: Task parameters
            
        Returns:
            Estimated duration in seconds
        """
        base_durations = {
            TaskType.FORECAST: 30.0,
            TaskType.STRATEGY: 45.0,
            TaskType.REPORT: 60.0,
            TaskType.ANALYSIS: 40.0,
            TaskType.OPTIMIZATION: 120.0,
            TaskType.PORTFOLIO: 90.0,
            TaskType.SYSTEM: 10.0,
            TaskType.GENERAL: 20.0
        }
        
        base_duration = base_durations.get(task_type, 30.0)
        
        # Adjust based on parameters
        if "symbols" in parameters and len(parameters["symbols"]) > 1:
            base_duration *= len(parameters["symbols"]) * 0.5
        
        if "timeframe" in parameters:
            if "month" in parameters["timeframe"] or "year" in parameters["timeframe"]:
                base_duration *= 1.5
        
        return base_duration
    
    def _determine_execution_order(self, task_chain: List[TaskDefinition]) -> List[str]:
        """Determine the execution order for tasks.
        
        Args:
            task_chain: List of task definitions
            
        Returns:
            List of task IDs in execution order
        """
        # Create dependency graph
        task_map = {task.task_id: task for task in task_chain}
        dependencies = {}
        
        for task in task_chain:
            dependencies[task.task_id] = []
            # Check if this task depends on other tasks in the chain
            for other_task in task_chain:
                if other_task.task_id != task.task_id:
                    if other_task.task_type in self.task_dependencies.get(task.task_type, []):
                        dependencies[task.task_id].append(other_task.task_id)
        
        # Topological sort
        execution_order = []
        visited = set()
        temp_visited = set()
        
        def visit(task_id):
            if task_id in temp_visited:
                raise ValueError("Circular dependency detected")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            for dep_id in dependencies[task_id]:
                if dep_id in task_map:  # Only visit dependencies in our chain
                    visit(dep_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            execution_order.append(task_id)
        
        # Visit all tasks
        for task_id in task_map:
            if task_id not in visited:
                visit(task_id)
        
        return execution_order
    
    def _build_dependencies(self, task_chain: List[TaskDefinition]) -> Dict[str, List[str]]:
        """Build dependency map for tasks.
        
        Args:
            task_chain: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to their dependencies
        """
        dependencies = {}
        
        for task in task_chain:
            dependencies[task.task_id] = []
            for other_task in task_chain:
                if other_task.task_id != task.task_id:
                    if other_task.task_type in self.task_dependencies.get(task.task_type, []):
                        dependencies[task.task_id].append(other_task.task_id)
        
        return dependencies
    
    def _calculate_confidence(self, intents: List[TaskType], prompt: str) -> float:
        """Calculate confidence in intent detection.
        
        Args:
            intents: Detected intents
            prompt: Original prompt
            
        Returns:
            Confidence score between 0 and 1
        """
        if not intents:
            return 0.0
        
        # Base confidence on number of pattern matches
        total_matches = 0
        for task_type in intents:
            patterns = self.intent_patterns.get(task_type, [])
            for pattern in patterns:
                if re.search(pattern, prompt.lower(), re.IGNORECASE):
                    total_matches += 1
        
        # Normalize by prompt length and number of intents
        confidence = min(1.0, total_matches / (len(prompt.split()) * 0.1))
        
        # Boost confidence for single, clear intent
        if len(intents) == 1:
            confidence *= 1.2
        
        return min(1.0, confidence)
    
    async def execute_task_chain(self, multi_intent: MultiIntentPrompt) -> Dict[str, Any]:
        """Execute a chain of tasks in the correct order.
        
        Args:
            multi_intent: Parsed multi-intent prompt
            
        Returns:
            Dictionary with execution results
        """
        self.logger.info(f"Executing task chain with {len(multi_intent.task_chain)} tasks")
        
        # Create task executions
        task_executions = {}
        for task in multi_intent.task_chain:
            execution = TaskExecution(task=task)
            task_executions[task.task_id] = execution
            self.active_tasks[task.task_id] = execution
        
        # Execute tasks in order
        results = {}
        shared_context = multi_intent.context
        
        for task_id in multi_intent.execution_order:
            execution = task_executions[task_id]
            task = execution.task
            
            # Check dependencies
            if not self._check_dependencies(task_id, multi_intent.dependencies, results):
                execution.status = TaskStatus.DEPENDENCY_WAIT
                continue
            
            # Execute task
            try:
                execution.status = TaskStatus.RUNNING
                execution.start_time = datetime.now()
                
                result = await self._execute_single_task(task, shared_context)
                execution.result = result
                execution.status = TaskStatus.COMPLETED
                execution.end_time = datetime.now()
                
                results[task_id] = result
                
                # Update shared context with result
                if result.success:
                    shared_context.shared_data[task_id] = result.data
                
                self.logger.info(f"Task {task_id} completed successfully")
                
            except Exception as e:
                execution.status = TaskStatus.FAILED
                execution.end_time = datetime.now()
                execution.error_history.append(str(e))
                
                result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error_message=str(e)
                )
                execution.result = result
                results[task_id] = result
                
                self.logger.error(f"Task {task_id} failed: {e}")
        
        # Move completed tasks
        for task_id, execution in task_executions.items():
            if execution.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                
                if execution.status == TaskStatus.COMPLETED:
                    self.completed_tasks[task_id] = execution
                else:
                    self.failed_tasks[task_id] = execution
        
        # Aggregate results
        aggregated_result = self._aggregate_results(results, multi_intent)
        
        # Update metrics
        self._update_execution_metrics(results)
        
        return aggregated_result
    
    def _check_dependencies(
        self, 
        task_id: str, 
        dependencies: Dict[str, List[str]], 
        completed_results: Dict[str, TaskResult]
    ) -> bool:
        """Check if task dependencies are satisfied.
        
        Args:
            task_id: Task ID to check
            dependencies: Dependency map
            completed_results: Results of completed tasks
            
        Returns:
            True if dependencies are satisfied
        """
        task_deps = dependencies.get(task_id, [])
        
        for dep_id in task_deps:
            if dep_id not in completed_results:
                return False
            
            dep_result = completed_results[dep_id]
            if not dep_result.success:
                return False
        
        return True
    
    async def _execute_single_task(self, task: TaskDefinition, context: TaskContext) -> TaskResult:
        """Execute a single task.
        
        Args:
            task: Task definition
            context: Shared context
            
        Returns:
            TaskResult: Execution result
        """
        start_time = datetime.now()
        
        try:
            # Route to appropriate handler based on task type
            if task.task_type == TaskType.FORECAST:
                data = await self._execute_forecast_task(task, context)
            elif task.task_type == TaskType.STRATEGY:
                data = await self._execute_strategy_task(task, context)
            elif task.task_type == TaskType.REPORT:
                data = await self._execute_report_task(task, context)
            elif task.task_type == TaskType.ANALYSIS:
                data = await self._execute_analysis_task(task, context)
            elif task.task_type == TaskType.OPTIMIZATION:
                data = await self._execute_optimization_task(task, context)
            elif task.task_type == TaskType.PORTFOLIO:
                data = await self._execute_portfolio_task(task, context)
            elif task.task_type == TaskType.SYSTEM:
                data = await self._execute_system_task(task, context)
            else:
                data = await self._execute_general_task(task, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                data=data,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _execute_forecast_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute forecast task."""
        # Placeholder implementation
        return {
            "forecast_type": task.parameters.get("forecast_type", "price"),
            "symbols": task.parameters.get("symbols", []),
            "timeframe": task.parameters.get("timeframe", "7 days"),
            "predictions": {
                "AAPL": {"price": 150.0, "confidence": 0.8},
                "GOOGL": {"price": 2800.0, "confidence": 0.75}
            }
        }
    
    async def _execute_strategy_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute strategy task."""
        # Use forecast results if available
        forecast_data = context.shared_data.get("forecast", {})
        
        return {
            "strategy_type": "momentum",
            "risk_level": task.parameters.get("risk_level", "medium"),
            "recommendations": [
                {"symbol": "AAPL", "action": "buy", "confidence": 0.8},
                {"symbol": "GOOGL", "action": "hold", "confidence": 0.6}
            ],
            "based_on_forecast": bool(forecast_data)
        }
    
    async def _execute_report_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute report task."""
        # Aggregate data from previous tasks
        report_data = {}
        
        for task_id, data in context.shared_data.items():
            if "forecast" in task_id:
                report_data["forecast"] = data
            elif "strategy" in task_id:
                report_data["strategy"] = data
            elif "analysis" in task_id:
                report_data["analysis"] = data
        
        return {
            "report_type": task.parameters.get("report_type", "summary"),
            "sections": list(report_data.keys()),
            "summary": "Comprehensive analysis completed",
            "data": report_data
        }
    
    async def _execute_analysis_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute analysis task."""
        return {
            "analysis_type": "technical",
            "symbols": task.parameters.get("symbols", []),
            "indicators": ["RSI", "MACD", "Bollinger Bands"],
            "findings": "Market shows bullish momentum"
        }
    
    async def _execute_optimization_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute optimization task."""
        return {
            "optimization_type": "portfolio",
            "objective": "maximize_sharpe_ratio",
            "constraints": ["risk_budget", "sector_limits"],
            "optimal_weights": {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3}
        }
    
    async def _execute_portfolio_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute portfolio task."""
        return {
            "portfolio_action": "rebalance",
            "current_allocation": {"AAPL": 0.25, "GOOGL": 0.35, "MSFT": 0.4},
            "target_allocation": {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3},
            "trades_needed": [{"symbol": "AAPL", "action": "buy", "amount": 1000}]
        }
    
    async def _execute_system_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute system task."""
        return {
            "system_status": "healthy",
            "uptime": "99.9%",
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks)
        }
    
    async def _execute_general_task(self, task: TaskDefinition, context: TaskContext) -> Dict[str, Any]:
        """Execute general task."""
        return {
            "response": f"Processed general request: {task.prompt}",
            "task_type": "general",
            "parameters": task.parameters
        }
    
    def _aggregate_results(
        self, 
        results: Dict[str, TaskResult], 
        multi_intent: MultiIntentPrompt
    ) -> Dict[str, Any]:
        """Aggregate results from multiple tasks.
        
        Args:
            results: Task results
            multi_intent: Original multi-intent prompt
            
        Returns:
            Aggregated result
        """
        successful_results = {
            task_id: result.data 
            for task_id, result in results.items() 
            if result.success
        }
        
        failed_results = {
            task_id: result.error_message
            for task_id, result in results.items()
            if not result.success
        }
        
        # Create summary
        summary = {
            "original_prompt": multi_intent.original_prompt,
            "detected_intents": [intent.value for intent in multi_intent.intents],
            "confidence": multi_intent.confidence,
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "results": successful_results,
            "errors": failed_results,
            "execution_order": multi_intent.execution_order
        }
        
        # Add synthesized response based on results
        if successful_results:
            summary["synthesized_response"] = self._synthesize_response(successful_results, multi_intent)
        
        return summary
    
    def _synthesize_response(
        self, 
        results: Dict[str, Any], 
        multi_intent: MultiIntentPrompt
    ) -> str:
        """Synthesize a natural language response from results.
        
        Args:
            results: Successful task results
            multi_intent: Original multi-intent prompt
            
        Returns:
            Synthesized response
        """
        response_parts = []
        
        if "forecast" in results:
            forecast = results["forecast"]
            symbols = forecast.get("symbols", [])
            if symbols:
                response_parts.append(f"Forecast for {', '.join(symbols)}: ")
                for symbol, pred in forecast.get("predictions", {}).items():
                    response_parts.append(f"{symbol} expected at ${pred['price']:.2f}")
        
        if "strategy" in results:
            strategy = results["strategy"]
            recommendations = strategy.get("recommendations", [])
            if recommendations:
                response_parts.append("Trading recommendations: ")
                for rec in recommendations:
                    response_parts.append(f"{rec['action'].upper()} {rec['symbol']}")
        
        if "report" in results:
            report = results["report"]
            response_parts.append(f"Report summary: {report.get('summary', 'Analysis completed')}")
        
        return " ".join(response_parts) if response_parts else "Task execution completed successfully."
    
    def _update_execution_metrics(self, results: Dict[str, TaskResult]) -> None:
        """Update execution metrics.
        
        Args:
            results: Task results
        """
        self.execution_metrics["total_tasks"] += len(results)
        
        successful_count = sum(1 for r in results.values() if r.success)
        self.execution_metrics["successful_tasks"] += successful_count
        self.execution_metrics["failed_tasks"] += len(results) - successful_count
        
        # Update task type statistics
        for result in results.values():
            task_type = result.task_id.split("_")[0]
            if task_type not in self.execution_metrics["task_type_stats"]:
                self.execution_metrics["task_type_stats"][task_type] = {
                    "total": 0,
                    "successful": 0,
                    "avg_time": 0.0
                }
            
            stats = self.execution_metrics["task_type_stats"][task_type]
            stats["total"] += 1
            if result.success:
                stats["successful"] += 1
            
            # Update average time
            if result.execution_time > 0:
                stats["avg_time"] = (
                    (stats["avg_time"] * (stats["total"] - 1) + result.execution_time) 
                    / stats["total"]
                )
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics.
        
        Returns:
            Dictionary with execution metrics
        """
        return self.execution_metrics.copy()
    
    def get_active_tasks(self) -> Dict[str, TaskExecution]:
        """Get currently active tasks.
        
        Returns:
            Dictionary of active task executions
        """
        return self.active_tasks.copy()
    
    def get_completed_tasks(self) -> Dict[str, TaskExecution]:
        """Get completed tasks.
        
        Returns:
            Dictionary of completed task executions
        """
        return self.completed_tasks.copy()


# Global task router instance
_task_router = None

def get_task_router() -> TaskRouter:
    """Get the global task router instance.
    
    Returns:
        TaskRouter: Global task router instance
    """
    global _task_router
    if _task_router is None:
        _task_router = TaskRouter()
    return _task_router

async def route_and_execute(prompt: str, context: Optional[TaskContext] = None) -> Dict[str, Any]:
    """Route and execute a prompt.
    
    Args:
        prompt: User prompt
        context: Optional context
        
    Returns:
        Execution results
    """
    router = get_task_router()
    multi_intent = router.parse_multi_intent_prompt(prompt, context)
    return await router.execute_task_chain(multi_intent)
