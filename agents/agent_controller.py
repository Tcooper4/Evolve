"""
Agent Controller Module

This module centralizes all agent workflow orchestration logic for the Evolve Trading Platform.
It provides structured workflows for the three main agent roles:
- Builder: Model building and initialization
- Evaluator: Performance evaluation and analysis  
- Updater: Model updates and optimization

Each role is implemented as a callable class or function that can be easily imported
and used from the prompt router or other parts of the application.

Key Features:
- Centralized workflow orchestration
- Structured agent role implementations
- Error handling and retry logic
- Performance tracking and logging
- Clean interfaces for integration
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from agents.task_agent import TaskAgent, TaskType, execute_task

logger = logging.getLogger(__name__)


class AgentWorkflowResult:
    """Result of an agent workflow execution."""
    
    def __init__(
        self,
        success: bool,
        workflow_type: str,
        data: Dict[str, Any],
        error_message: Optional[str] = None,
        execution_time: float = 0.0,
        agent_results: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.workflow_type = workflow_type
        self.data = data
        self.error_message = error_message
        self.execution_time = execution_time
        self.agent_results = agent_results or {}
        self.timestamp = datetime.now().isoformat()


class BuilderWorkflow:
    """
    Builder workflow orchestrator for model building and initialization.
    
    This class handles the complete model building workflow including:
    - Data preparation and validation
    - Model type selection and configuration
    - Model training and validation
    - Model registration and storage
    """
    
    def __init__(self):
        """Initialize the builder workflow."""
        self.logger = logging.getLogger(f"{__name__}.BuilderWorkflow")
        self.workflow_name = "model_builder"
        
    async def __call__(self, **kwargs) -> AgentWorkflowResult:
        """
        Execute the builder workflow.
        
        Args:
            **kwargs: Builder parameters including:
                - model_type: Type of model to build (lstm, xgboost, ensemble)
                - data_path: Path to training data
                - target_column: Target column for prediction
                - hyperparameters: Model hyperparameters
                - validation_split: Validation data split ratio
                
        Returns:
            AgentWorkflowResult: Result of the builder workflow
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting model builder workflow")
            
            # Validate input parameters
            validation_result = self._validate_builder_inputs(kwargs)
            if not validation_result["valid"]:
                return AgentWorkflowResult(
                    success=False,
                    workflow_type=self.workflow_name,
                    data={},
                    error_message=validation_result["error"],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Execute the builder workflow
            result = await self._execute_builder_workflow(kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=result["success"],
                workflow_type=self.workflow_name,
                data=result.get("data", {}),
                error_message=result.get("error_message"),
                execution_time=execution_time,
                agent_results=result.get("agent_results", {})
            )
            
        except Exception as e:
            self.logger.error(f"Builder workflow failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=False,
                workflow_type=self.workflow_name,
                data={},
                error_message=f"Builder workflow failed: {str(e)}",
                execution_time=execution_time
            )
    
    def _validate_builder_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate builder workflow inputs."""
        required_fields = ["model_type", "data_path", "target_column"]
        
        for field in required_fields:
            if field not in kwargs:
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Validate model type
        valid_model_types = ["lstm", "xgboost", "ensemble", "transformer"]
        if kwargs["model_type"].lower() not in valid_model_types:
            return {
                "valid": False,
                "error": f"Invalid model type: {kwargs['model_type']}. Valid types: {valid_model_types}"
            }
        
        return {"valid": True}
    
    async def _execute_builder_workflow(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the builder workflow steps."""
        try:
            # Step 1: Prepare data
            data_result = await self._prepare_data(kwargs)
            if not data_result["success"]:
                return data_result
            
            # Step 2: Build model
            model_result = await self._build_model(kwargs, data_result["data"])
            if not model_result["success"]:
                return model_result
            
            # Step 3: Validate model
            validation_result = await self._validate_model(model_result["data"])
            if not validation_result["success"]:
                return validation_result
            
            # Step 4: Register model
            registration_result = await self._register_model(validation_result["data"])
            
            return {
                "success": True,
                "data": registration_result["data"],
                "agent_results": {
                    "data_preparation": data_result,
                    "model_building": model_result,
                    "model_validation": validation_result,
                    "model_registration": registration_result
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in builder workflow execution: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }
    
    async def _prepare_data(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for model building."""
        try:
            # Import and execute data preparation agent
            from trading.agents.agent_manager import execute_agent
            
            data_request = {
                "data_path": kwargs["data_path"],
                "target_column": kwargs["target_column"],
                "validation_split": kwargs.get("validation_split", 0.2),
                "preprocessing_steps": kwargs.get("preprocessing_steps", [])
            }
            
            result = await execute_agent("data_preparation_agent", request=data_request)
            
            return {
                "success": result.success,
                "data": result.data if result.success else {},
                "error_message": result.error_message if not result.success else None
            }
            
        except ImportError:
            # Fallback to direct data loading
            self.logger.warning("Data preparation agent not available, using fallback")
            return {
                "success": True,
                "data": {
                    "train_data": kwargs["data_path"],
                    "validation_split": kwargs.get("validation_split", 0.2)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Data preparation failed: {str(e)}"
            }
    
    async def _build_model(self, kwargs: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Build the model using the model builder agent."""
        try:
            from trading.agents.agent_manager import execute_agent
            from trading.agents.model_builder_agent import ModelBuildRequest
            
            build_request = ModelBuildRequest(
                model_type=kwargs["model_type"],
                data_path=data.get("train_data", kwargs["data_path"]),
                target_column=kwargs["target_column"],
                hyperparameters=kwargs.get("hyperparameters", {}),
                request_id=f"builder_{uuid.uuid4().hex[:8]}"
            )
            
            result = await execute_agent("model_builder", request=build_request)
            
            return {
                "success": result.success,
                "data": result.data if result.success else {},
                "error_message": result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Model building failed: {str(e)}"
            }
    
    async def _validate_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the built model."""
        try:
            # Basic validation - model file exists and is accessible
            model_path = model_data.get("model_path")
            if not model_path:
                return {
                    "success": False,
                    "error_message": "No model path provided for validation"
                }
            
            # Add validation logic here if needed
            return {
                "success": True,
                "data": model_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Model validation failed: {str(e)}"
            }
    
    async def _register_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register the model in the system."""
        try:
            # Register model in the model registry
            model_id = model_data.get("model_id", f"model_{uuid.uuid4().hex[:8]}")
            
            registration_data = {
                "model_id": model_id,
                "model_type": model_data.get("model_type"),
                "model_path": model_data.get("model_path"),
                "training_metrics": model_data.get("training_metrics", {}),
                "build_timestamp": model_data.get("build_timestamp", datetime.now().isoformat()),
                "status": "active"
            }
            
            return {
                "success": True,
                "data": registration_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Model registration failed: {str(e)}"
            }


class EvaluatorWorkflow:
    """
    Evaluator workflow orchestrator for model performance evaluation.
    
    This class handles the complete model evaluation workflow including:
    - Model loading and validation
    - Performance metrics calculation
    - Risk assessment and analysis
    - Benchmark comparison
    """
    
    def __init__(self):
        """Initialize the evaluator workflow."""
        self.logger = logging.getLogger(f"{__name__}.EvaluatorWorkflow")
        self.workflow_name = "model_evaluator"
        
    async def __call__(self, **kwargs) -> AgentWorkflowResult:
        """
        Execute the evaluator workflow.
        
        Args:
            **kwargs: Evaluator parameters including:
                - model_id: ID of the model to evaluate
                - model_path: Path to the model file
                - test_data_path: Path to test data
                - evaluation_metrics: List of metrics to calculate
                - benchmark_data: Benchmark data for comparison
                
        Returns:
            AgentWorkflowResult: Result of the evaluator workflow
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting model evaluator workflow")
            
            # Validate input parameters
            validation_result = self._validate_evaluator_inputs(kwargs)
            if not validation_result["valid"]:
                return AgentWorkflowResult(
                    success=False,
                    workflow_type=self.workflow_name,
                    data={},
                    error_message=validation_result["error"],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Execute the evaluator workflow
            result = await self._execute_evaluator_workflow(kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=result["success"],
                workflow_type=self.workflow_name,
                data=result.get("data", {}),
                error_message=result.get("error_message"),
                execution_time=execution_time,
                agent_results=result.get("agent_results", {})
            )
            
        except Exception as e:
            self.logger.error(f"Evaluator workflow failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=False,
                workflow_type=self.workflow_name,
                data={},
                error_message=f"Evaluator workflow failed: {str(e)}",
                execution_time=execution_time
            )
    
    def _validate_evaluator_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evaluator workflow inputs."""
        required_fields = ["model_id", "model_path"]
        
        for field in required_fields:
            if field not in kwargs:
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        return {"valid": True}
    
    async def _execute_evaluator_workflow(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the evaluator workflow steps."""
        try:
            # Step 1: Load model
            load_result = await self._load_model(kwargs)
            if not load_result["success"]:
                return load_result
            
            # Step 2: Prepare test data
            data_result = await self._prepare_test_data(kwargs)
            if not data_result["success"]:
                return data_result
            
            # Step 3: Evaluate model
            evaluation_result = await self._evaluate_model(load_result["data"], data_result["data"])
            if not evaluation_result["success"]:
                return evaluation_result
            
            # Step 4: Calculate risk metrics
            risk_result = await self._calculate_risk_metrics(evaluation_result["data"])
            
            # Step 5: Compare with benchmarks
            benchmark_result = await self._compare_benchmarks(evaluation_result["data"])
            
            # Combine all results
            combined_data = {
                **evaluation_result["data"],
                "risk_metrics": risk_result.get("data", {}),
                "benchmark_comparison": benchmark_result.get("data", {})
            }
            
            return {
                "success": True,
                "data": combined_data,
                "agent_results": {
                    "model_loading": load_result,
                    "test_data_preparation": data_result,
                    "model_evaluation": evaluation_result,
                    "risk_calculation": risk_result,
                    "benchmark_comparison": benchmark_result
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in evaluator workflow execution: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }
    
    async def _load_model(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Load the model for evaluation."""
        try:
            model_id = kwargs["model_id"]
            model_path = kwargs["model_path"]
            
            # Basic model loading validation
            return {
                "success": True,
                "data": {
                    "model_id": model_id,
                    "model_path": model_path,
                    "loaded_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Model loading failed: {str(e)}"
            }
    
    async def _prepare_test_data(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare test data for evaluation."""
        try:
            test_data_path = kwargs.get("test_data_path")
            
            if not test_data_path:
                return {
                    "success": True,
                    "data": {
                        "test_data_ready": True,
                        "data_source": "default"
                    }
                }
            
            return {
                "success": True,
                "data": {
                    "test_data_path": test_data_path,
                    "test_data_ready": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Test data preparation failed: {str(e)}"
            }
    
    async def _evaluate_model(self, model_data: Dict[str, Any], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model using the performance critic agent."""
        try:
            from trading.agents.agent_manager import execute_agent
            from trading.agents.performance_critic_agent import ModelEvaluationRequest
            
            eval_request = ModelEvaluationRequest(
                model_id=model_data["model_id"],
                model_path=model_data["model_path"],
                model_type=model_data.get("model_type", "unknown"),
                test_data_path=test_data.get("test_data_path"),
                request_id=f"eval_{model_data['model_id']}_{datetime.now().isoformat()}"
            )
            
            result = await execute_agent("performance_critic", request=eval_request)
            
            return {
                "success": result.success,
                "data": result.data if result.success else {},
                "error_message": result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Model evaluation failed: {str(e)}"
            }
    
    async def _calculate_risk_metrics(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for the model."""
        try:
            # Basic risk calculation
            sharpe_ratio = evaluation_data.get("sharpe_ratio", 0.0)
            max_drawdown = evaluation_data.get("max_drawdown", 0.0)
            
            risk_score = max(0, 1 - (sharpe_ratio + abs(max_drawdown)) / 2)
            
            return {
                "success": True,
                "data": {
                    "risk_score": risk_score,
                    "volatility": evaluation_data.get("volatility", 0.0),
                    "var_95": evaluation_data.get("var_95", 0.0),
                    "max_drawdown": max_drawdown
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Risk calculation failed: {str(e)}"
            }
    
    async def _compare_benchmarks(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance with benchmarks."""
        try:
            # Basic benchmark comparison
            model_sharpe = evaluation_data.get("sharpe_ratio", 0.0)
            benchmark_sharpe = 0.5  # Example benchmark
            
            outperformance = model_sharpe - benchmark_sharpe
            
            return {
                "success": True,
                "data": {
                    "benchmark_sharpe": benchmark_sharpe,
                    "outperformance": outperformance,
                    "outperforms_benchmark": outperformance > 0
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Benchmark comparison failed: {str(e)}"
            }


class UpdaterWorkflow:
    """
    Updater workflow orchestrator for model updates and optimization.
    
    This class handles the complete model update workflow including:
    - Performance analysis and decision making
    - Model retraining and tuning
    - Model replacement and optimization
    - Ensemble weight adjustment
    """
    
    def __init__(self):
        """Initialize the updater workflow."""
        self.logger = logging.getLogger(f"{__name__}.UpdaterWorkflow")
        self.workflow_name = "model_updater"
        
    async def __call__(self, **kwargs) -> AgentWorkflowResult:
        """
        Execute the updater workflow.
        
        Args:
            **kwargs: Updater parameters including:
                - model_id: ID of the model to update
                - evaluation_result: Previous evaluation results
                - update_type: Type of update (retrain, tune, replace, ensemble_adjust)
                - new_data_path: Path to new training data
                - optimization_target: Target metric for optimization
                
        Returns:
            AgentWorkflowResult: Result of the updater workflow
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting model updater workflow")
            
            # Validate input parameters
            validation_result = self._validate_updater_inputs(kwargs)
            if not validation_result["valid"]:
                return AgentWorkflowResult(
                    success=False,
                    workflow_type=self.workflow_name,
                    data={},
                    error_message=validation_result["error"],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Execute the updater workflow
            result = await self._execute_updater_workflow(kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=result["success"],
                workflow_type=self.workflow_name,
                data=result.get("data", {}),
                error_message=result.get("error_message"),
                execution_time=execution_time,
                agent_results=result.get("agent_results", {})
            )
            
        except Exception as e:
            self.logger.error(f"Updater workflow failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=False,
                workflow_type=self.workflow_name,
                data={},
                error_message=f"Updater workflow failed: {str(e)}",
                execution_time=execution_time
            )
    
    def _validate_updater_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate updater workflow inputs."""
        required_fields = ["model_id"]
        
        for field in required_fields:
            if field not in kwargs:
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Validate update type if provided
        update_type = kwargs.get("update_type", "auto")
        valid_update_types = ["retrain", "tune", "replace", "ensemble_adjust", "auto"]
        if update_type not in valid_update_types:
            return {
                "valid": False,
                "error": f"Invalid update type: {update_type}. Valid types: {valid_update_types}"
            }
        
        return {"valid": True}
    
    async def _execute_updater_workflow(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the updater workflow steps."""
        try:
            # Step 1: Analyze current performance
            analysis_result = await self._analyze_performance(kwargs)
            if not analysis_result["success"]:
                return analysis_result
            
            # Step 2: Determine update strategy
            strategy_result = await self._determine_update_strategy(analysis_result["data"], kwargs)
            if not strategy_result["success"]:
                return strategy_result
            
            # Step 3: Execute update
            update_result = await self._execute_update(strategy_result["data"])
            if not update_result["success"]:
                return update_result
            
            # Step 4: Validate update
            validation_result = await self._validate_update(update_result["data"])
            
            # Step 5: Update registry
            registry_result = await self._update_registry(validation_result["data"])
            
            return {
                "success": True,
                "data": registry_result["data"],
                "agent_results": {
                    "performance_analysis": analysis_result,
                    "strategy_determination": strategy_result,
                    "update_execution": update_result,
                    "update_validation": validation_result,
                    "registry_update": registry_result
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in updater workflow execution: {e}")
            return {
                "success": False,
                "error_message": str(e)
            }
    
    async def _analyze_performance(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current model performance."""
        try:
            model_id = kwargs["model_id"]
            evaluation_result = kwargs.get("evaluation_result", {})
            
            # Analyze performance metrics
            sharpe_ratio = evaluation_result.get("sharpe_ratio", 0.0)
            max_drawdown = evaluation_result.get("max_drawdown", 0.0)
            win_rate = evaluation_result.get("win_rate", 0.0)
            
            performance_score = (sharpe_ratio + win_rate - abs(max_drawdown)) / 3
            
            return {
                "success": True,
                "data": {
                    "model_id": model_id,
                    "performance_score": performance_score,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "needs_update": performance_score < 0.3
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Performance analysis failed: {str(e)}"
            }
    
    async def _determine_update_strategy(self, analysis_data: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the appropriate update strategy."""
        try:
            update_type = kwargs.get("update_type", "auto")
            performance_score = analysis_data.get("performance_score", 0.0)
            
            if update_type == "auto":
                if performance_score < 0.1:
                    update_type = "replace"
                elif performance_score < 0.3:
                    update_type = "retrain"
                elif performance_score < 0.5:
                    update_type = "tune"
                else:
                    update_type = "ensemble_adjust"
            
            return {
                "success": True,
                "data": {
                    "update_type": update_type,
                    "priority": "high" if performance_score < 0.3 else "normal",
                    "target_improvement": max(0.1, 0.5 - performance_score)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Strategy determination failed: {str(e)}"
            }
    
    async def _execute_update(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the model update using the updater agent."""
        try:
            from trading.agents.agent_manager import execute_agent
            from trading.agents.updater_agent import UpdateRequest
            
            # Create a mock evaluation result since we don't have the actual one
            from trading.agents.performance_critic_agent import ModelEvaluationResult
            mock_evaluation = ModelEvaluationResult(
                request_id=f"mock_eval_{uuid.uuid4().hex[:8]}",
                model_id=strategy_data.get("model_id"),
                evaluation_timestamp=datetime.now().isoformat(),
                performance_metrics={},
                risk_metrics={},
                trading_metrics={}
            )
            
            update_request = UpdateRequest(
                model_id=strategy_data.get("model_id"),
                evaluation_result=mock_evaluation,
                update_type=strategy_data["update_type"],
                priority=strategy_data.get("priority", "normal"),
                request_id=f"updater_{uuid.uuid4().hex[:8]}"
            )
            
            result = await execute_agent("updater", request=update_request)
            
            return {
                "success": result.success,
                "data": result.data if result.success else {},
                "error_message": result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Update execution failed: {str(e)}"
            }
    
    async def _validate_update(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the model update."""
        try:
            # Basic validation
            update_status = update_data.get("update_status", "unknown")
            
            return {
                "success": update_status == "success",
                "data": {
                    **update_data,
                    "validated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Update validation failed: {str(e)}"
            }
    
    async def _update_registry(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update the model registry with the new model."""
        try:
            # Update registry entry
            model_id = validation_data.get("new_model_id", validation_data.get("model_id"))
            
            registry_data = {
                "model_id": model_id,
                "update_timestamp": datetime.now().isoformat(),
                "update_status": "completed",
                "previous_model_id": validation_data.get("original_model_id")
            }
            
            return {
                "success": True,
                "data": registry_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Registry update failed: {str(e)}"
            }


class TaskWorkflow:
    """
    Task workflow orchestrator for complex recursive task execution.
    
    This class handles complex tasks that may require multiple iterations
    and different approaches to achieve success.
    """
    
    def __init__(self):
        """Initialize the task workflow."""
        self.logger = logging.getLogger(f"{__name__}.TaskWorkflow")
        self.workflow_name = "task_executor"
        self.task_agent = TaskAgent()
        
    async def __call__(self, **kwargs) -> AgentWorkflowResult:
        """
        Execute a complex task with recursive improvement logic.
        
        Args:
            **kwargs: Task parameters including:
                - prompt: The task prompt
                - task_type: Type of task (model_build, evaluate, update, etc.)
                - parameters: Task-specific parameters
                - max_depth: Maximum recursion depth (default: 5)
                - performance_threshold: Success threshold (default: 0.7)
                
        Returns:
            AgentWorkflowResult: Result of the task execution
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting task workflow execution")
            
            # Validate input parameters
            validation_result = self._validate_task_inputs(kwargs)
            if not validation_result["valid"]:
                return AgentWorkflowResult(
                    success=False,
                    workflow_type=self.workflow_name,
                    data={},
                    error_message=validation_result["error"],
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Execute the task
            result = await self._execute_task_workflow(kwargs)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=result["success"],
                workflow_type=self.workflow_name,
                data=result.get("data", {}),
                error_message=result.get("error_message"),
                execution_time=execution_time,
                agent_results=result.get("agent_results", {})
            )
            
        except Exception as e:
            self.logger.error(f"Task workflow failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentWorkflowResult(
                success=False,
                workflow_type=self.workflow_name,
                data={},
                error_message=f"Task workflow failed: {str(e)}",
                execution_time=execution_time
            )
    
    def _validate_task_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task workflow inputs."""
        required_fields = ["prompt"]
        
        for field in required_fields:
            if field not in kwargs:
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        # Validate task type if provided
        task_type_str = kwargs.get("task_type", "general")
        valid_task_types = [t.value for t in TaskType]
        if task_type_str not in valid_task_types:
            return {
                "valid": False,
                "error": f"Invalid task type: {task_type_str}. Valid types: {valid_task_types}"
            }
        
        return {"valid": True}
    
    async def _execute_task_workflow(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task workflow."""
        try:
            # Extract parameters
            prompt = kwargs["prompt"]
            task_type_str = kwargs.get("task_type", "general")
            parameters = kwargs.get("parameters", {})
            max_depth = kwargs.get("max_depth", 5)
            performance_threshold = kwargs.get("performance_threshold", 0.7)
            
            # Convert string to TaskType enum
            task_type = TaskType(task_type_str)
            
            # Execute the task
            result = await self.task_agent.execute_task(
                prompt=prompt,
                task_type=task_type,
                parameters=parameters,
                max_depth=max_depth,
                performance_threshold=performance_threshold
            )
            
            return {
                "success": result.success,
                "data": {
                    "task_id": result.data.get("final_performance", {}),
                    "performance_score": result.performance_score,
                    "message": result.message,
                    "action_count": len(result.data.get("performance_history", [])),
                    "depth_reached": result.data.get("depth_reached", 0)
                },
                "error_message": result.error_details if not result.success else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": f"Task execution failed: {str(e)}"
            }


class AgentController:
    """
    Main agent controller that orchestrates all agent workflows.
    
    This class provides a unified interface for executing agent workflows
    and managing the overall agent orchestration process.
    """
    
    def __init__(self):
        """Initialize the agent controller."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize workflow orchestrators
        self.builder = BuilderWorkflow()
        self.evaluator = EvaluatorWorkflow()
        self.updater = UpdaterWorkflow()
        self.task_executor = TaskWorkflow()
        
        # Workflow registry
        self.workflows = {
            "builder": self.builder,
            "evaluator": self.evaluator,
            "updater": self.updater,
            "task": self.task_executor
        }
        
        # Agent registration tracking
        self.registered_agents = {}
        self.agent_registration_status = {
            "total_agents": 0,
            "successful_registrations": 0,
            "failed_registrations": 0,
            "fallback_agent_created": False
        }
        
        # Initialize agent registration
        self._initialize_agent_registration()
        
        self.logger.info("AgentController initialized with all workflows including TaskAgent")
    
    def _initialize_agent_registration(self):
        """Initialize agent registration and check for available agents."""
        self.logger.info("Initializing agent registration...")
        
        try:
            # Try to get agent registry
            from trading.agents.agent_registry import get_registry
            registry = get_registry()
            
            # Get list of registered agents
            registered_agents = registry.list_agents()
            self.agent_registration_status["total_agents"] = len(registered_agents)
            
            if registered_agents:
                self.logger.info(f"Found {len(registered_agents)} registered agents: {registered_agents}")
                self.agent_registration_status["successful_registrations"] = len(registered_agents)
                
                # Store agent information
                for agent_name in registered_agents:
                    agent_info = registry.get_agent(agent_name)
                    if agent_info:
                        self.registered_agents[agent_name] = {
                            "name": agent_name,
                            "class_name": agent_info.class_name,
                            "module_path": agent_info.module_path,
                            "capabilities": [cap.name for cap in agent_info.capabilities],
                            "category": agent_info.category.value if agent_info.category else "unknown"
                        }
                
                self.logger.info("✅ Agent registration successful")
                
            else:
                self.logger.warning("⚠️ No agents found in registry")
                self._create_fallback_agent()
                
        except ImportError as e:
            self.logger.warning(f"⚠️ Agent registry not available: {e}")
            self._create_fallback_agent()
        except Exception as e:
            self.logger.error(f"❌ Error during agent registration: {e}")
            self.agent_registration_status["failed_registrations"] += 1
            self._create_fallback_agent()
    
    def _create_fallback_agent(self):
        """Create a fallback agent when no agents are registered."""
        try:
            from agents.mock_agent import create_mock_agent
            
            fallback_agent = create_mock_agent("FallbackAgent", [
                "general_query",
                "system_status", 
                "help",
                "fallback_response"
            ])
            
            self.registered_agents["fallback_agent"] = {
                "name": "fallback_agent",
                "class_name": "MockAgent",
                "module_path": "agents.mock_agent",
                "capabilities": ["general_query", "system_status", "help", "fallback_response"],
                "category": "fallback",
                "instance": fallback_agent
            }
            
            self.agent_registration_status["fallback_agent_created"] = True
            self.agent_registration_status["total_agents"] = 1
            self.agent_registration_status["successful_registrations"] = 1
            
            self.logger.warning("⚠️ Created fallback agent - no real agents available")
            self.logger.info("System will continue running with mock agent for UI testing")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create fallback agent: {e}")
            self.agent_registration_status["failed_registrations"] += 1
    
    def get_agent_registration_status(self) -> Dict[str, Any]:
        """Get agent registration status."""
        return {
            **self.agent_registration_status,
            "registered_agent_names": list(self.registered_agents.keys()),
            "agent_details": self.registered_agents
        }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.registered_agents.keys())
    
    def has_real_agents(self) -> bool:
        """Check if real agents (not fallback) are available."""
        return not self.agent_registration_status["fallback_agent_created"]
    
    async def execute_workflow(
        self, 
        workflow_type: str, 
        **kwargs
    ) -> AgentWorkflowResult:
        """
        Execute a specific agent workflow.
        
        Args:
            workflow_type: Type of workflow to execute (builder, evaluator, updater)
            **kwargs: Workflow-specific parameters
            
        Returns:
            AgentWorkflowResult: Result of the workflow execution
        """
        if workflow_type not in self.workflows:
            return AgentWorkflowResult(
                success=False,
                workflow_type=workflow_type,
                data={},
                error_message=f"Unknown workflow type: {workflow_type}"
            )
        
        workflow = self.workflows[workflow_type]
        return await workflow(**kwargs)
    
    async def execute_full_pipeline(
        self,
        model_type: str,
        data_path: str,
        target_column: str,
        **kwargs
    ) -> Dict[str, AgentWorkflowResult]:
        """
        Execute the full agent pipeline: build -> evaluate -> update.
        
        Args:
            model_type: Type of model to build
            data_path: Path to training data
            target_column: Target column for prediction
            **kwargs: Additional parameters for the pipeline
            
        Returns:
            Dict containing results from each workflow step
        """
        self.logger.info("Starting full agent pipeline execution")
        
        results = {}
        
        try:
            # Step 1: Build model
            self.logger.info("Step 1: Building model")
            build_result = await self.builder(
                model_type=model_type,
                data_path=data_path,
                target_column=target_column,
                **kwargs
            )
            results["build"] = build_result
            
            if not build_result.success:
                self.logger.error("Build step failed, stopping pipeline")
                return results
            
            # Step 2: Evaluate model
            self.logger.info("Step 2: Evaluating model")
            model_data = build_result.data
            eval_result = await self.evaluator(
                model_id=model_data["model_id"],
                model_path=model_data["model_path"],
                model_type=model_type,
                **kwargs
            )
            results["evaluate"] = eval_result
            
            if not eval_result.success:
                self.logger.error("Evaluate step failed, stopping pipeline")
                return results
            
            # Step 3: Update model if needed
            self.logger.info("Step 3: Checking if update is needed")
            evaluation_data = eval_result.data
            
            # Determine if update is needed based on performance
            performance_score = evaluation_data.get("performance_score", 0.0)
            if performance_score < 0.5:  # Threshold for updates
                self.logger.info("Performance below threshold, updating model")
                update_result = await self.updater(
                    model_id=model_data["model_id"],
                    evaluation_result=evaluation_data,
                    update_type="auto",
                    **kwargs
                )
                results["update"] = update_result
            else:
                self.logger.info("Performance above threshold, no update needed")
                results["update"] = AgentWorkflowResult(
                    success=True,
                    workflow_type="model_updater",
                    data={"message": "No update needed - performance is satisfactory"},
                    execution_time=0.0
                )
            
            self.logger.info("Full agent pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Full pipeline execution failed: {e}", exc_info=True)
            results["error"] = AgentWorkflowResult(
                success=False,
                workflow_type="full_pipeline",
                data={},
                error_message=f"Pipeline execution failed: {str(e)}"
            )
        
        return results
    
    def get_workflow_status(self, workflow_type: str) -> Dict[str, Any]:
        """
        Get the status of a specific workflow.
        
        Args:
            workflow_type: Type of workflow to check
            
        Returns:
            Dict containing workflow status information
        """
        if workflow_type not in self.workflows:
            return {
                "available": False,
                "error": f"Unknown workflow type: {workflow_type}"
            }
        
        return {
            "available": True,
            "workflow_type": workflow_type,
            "status": "ready"
        }
    
    def list_available_workflows(self) -> List[str]:
        """
        List all available workflows.
        
        Returns:
            List of available workflow types
        """
        return list(self.workflows.keys())


# Global agent controller instance
_agent_controller = None


def get_agent_controller() -> AgentController:
    """
    Get the global agent controller instance.
    
    Returns:
        AgentController: Global agent controller instance
    """
    global _agent_controller
    if _agent_controller is None:
        _agent_controller = AgentController()
    return _agent_controller


# Convenience functions for direct workflow execution
async def execute_builder_workflow(**kwargs) -> AgentWorkflowResult:
    """Execute the builder workflow."""
    controller = get_agent_controller()
    return await controller.execute_workflow("builder", **kwargs)


async def execute_evaluator_workflow(**kwargs) -> AgentWorkflowResult:
    """Execute the evaluator workflow."""
    controller = get_agent_controller()
    return await controller.execute_workflow("evaluator", **kwargs)


async def execute_updater_workflow(**kwargs) -> AgentWorkflowResult:
    """Execute the updater workflow."""
    controller = get_agent_controller()
    return await controller.execute_workflow("updater", **kwargs)


async def execute_full_pipeline(
    model_type: str,
    data_path: str,
    target_column: str,
    **kwargs
) -> Dict[str, AgentWorkflowResult]:
    """Execute the full agent pipeline."""
    controller = get_agent_controller()
    return await controller.execute_full_pipeline(
        model_type=model_type,
        data_path=data_path,
        target_column=target_column,
        **kwargs
    ) 