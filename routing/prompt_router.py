"""
Prompt Router Module

This module handles all prompt processing and routing logic for the Evolve Trading Platform.
It consolidates natural language prompt interpretation, model/strategy/backtest decision logic,
and result forwarding into a clean, modular interface.

Key Features:
- Natural language prompt analysis and classification
- Intelligent routing to appropriate agents and services
- Model/strategy selection and backtest decision logic
- Result formatting and forwarding
- Error handling and fallback mechanisms
- TaskAgent integration for recursive task execution
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.task_agent import TaskAgent, TaskType

logger = logging.getLogger(__name__)


class PromptRouter:
    """
    Main prompt router that handles all prompt processing and routing logic.
    
    This class consolidates the prompt handling logic from app.py and provides
    a clean interface for processing user prompts and routing them to appropriate
    handlers.
    """
    
    def __init__(self):
        """Initialize the prompt router."""
        self.logger = logging.getLogger(__name__)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components with error handling."""
        # Initialize TaskAgent
        try:
            self.task_agent = TaskAgent()
            self.logger.info("✅ TaskAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TaskAgent: {e}")
            self.task_agent = None
        
        # Initialize prompt agent
        try:
            from agents.llm.agent import PromptAgent
            self.prompt_agent = PromptAgent()
            self.logger.info("✅ Prompt agent initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Prompt agent not available: {e}")
            self.prompt_agent = None
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize prompt agent: {e}")
            self.prompt_agent = None
        
        # Initialize agent logger
        try:
            from trading.memory.agent_logger import get_agent_logger
            self.agent_logger = get_agent_logger()
            self.logger.info("✅ Agent logger initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Agent logger not available: {e}")
            self.agent_logger = None
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agent logger: {e}")
            self.agent_logger = None
        
        # Initialize agent controller
        try:
            from agents.agent_controller import get_agent_controller
            self.agent_controller = get_agent_controller()
            self.logger.info("✅ Agent controller initialized successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Agent controller not available: {e}")
            self.agent_controller = None
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agent controller: {e}")
            self.agent_controller = None
    
    async def route_prompt(self, prompt: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Route a prompt to the appropriate agent or workflow.
        
        Args:
            prompt: User prompt
            user_id: User identifier
            
        Returns:
            Dict containing routing result and response
        """
        try:
            # Log the prompt
            if self.agent_logger:
                from trading.memory.agent_logger import AgentAction, LogLevel
                self.agent_logger.log_action(
                    agent_name="PromptRouter",
                    action=AgentAction.MODEL_SYNTHESIS,
                    message=f"Processing prompt: {prompt[:100]}...",
                    data={"prompt": prompt, "user_id": user_id},
                    level=LogLevel.INFO
                )
            
            # Check if this is a complex task that needs recursive execution
            if self._is_complex_task(prompt):
                return await self._handle_complex_task(prompt, user_id)
            
            # Check if this is a workflow-specific prompt
            if self._is_workflow_prompt(prompt):
                return await self._handle_workflow_prompt(prompt, user_id)
            
            # Default to prompt agent
            return await self._handle_general_prompt(prompt, user_id)
            
        except Exception as e:
            self.logger.error(f"Error routing prompt: {e}")
            return {
                "success": False,
                "error": str(e),
                "routing_type": "error"
            }
    
    def _is_complex_task(self, prompt: str) -> bool:
        """
        Determine if a prompt requires complex recursive task execution.
        
        Args:
            prompt: User prompt
            
        Returns:
            bool: True if this is a complex task
        """
        prompt_lower = prompt.lower()
        
        # Keywords that indicate complex tasks requiring multiple iterations
        complex_keywords = [
            "optimize until", "improve performance", "iteratively", "recursively",
            "keep trying", "multiple attempts", "until success", "achieve target",
            "meet threshold", "converge", "refine", "tune", "calibrate",
            "build and test", "evaluate and improve", "learn from mistakes"
        ]
        
        # Task patterns that suggest complexity
        task_patterns = [
            "build a model that", "create a strategy that", "develop an approach",
            "find the best", "determine optimal", "figure out the right"
        ]
        
        # Check for complex keywords
        for keyword in complex_keywords:
            if keyword in prompt_lower:
                return True
        
        # Check for task patterns
        for pattern in task_patterns:
            if pattern in prompt_lower:
                return True
        
        return False
    
    def _is_workflow_prompt(self, prompt: str) -> bool:
        """
        Determine if a prompt is for a specific workflow.
        
        Args:
            prompt: User prompt
            
        Returns:
            bool: True if this is a workflow prompt
        """
        prompt_lower = prompt.lower()
        
        # Workflow-specific keywords
        workflow_keywords = {
            "builder": ["build model", "create model", "train model", "new model"],
            "evaluator": ["evaluate", "assess", "test model", "performance", "score"],
            "updater": ["update", "improve", "modify", "change", "adjust"]
        }
        
        for workflow, keywords in workflow_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    return True
        
        return False
    
    async def _handle_complex_task(self, prompt: str, user_id: str) -> Dict[str, Any]:
        """
        Handle complex tasks that require recursive execution using TaskAgent.
        
        Args:
            prompt: User prompt
            user_id: User identifier
            
        Returns:
            Dict containing task execution result
        """
        try:
            if not self.task_agent:
                raise Exception("TaskAgent not available")
            
            # Determine task type from prompt
            task_type = self._determine_task_type(prompt)
            
            # Extract parameters from prompt
            parameters = self._extract_task_parameters(prompt)
            
            # Add user_id to parameters
            parameters["user_id"] = user_id
            
            # Execute the task using TaskAgent
            result = await self.task_agent.execute_task(
                prompt=prompt,
                task_type=task_type,
                parameters=parameters,
                max_depth=5,
                performance_threshold=0.7
            )
            
            return {
                "success": result.success,
                "response": result.data,
                "routing_type": "complex_task",
                "task_type": task_type.value,
                "performance_score": result.performance_score,
                "message": result.message,
                "task_id": result.data.get("task_id") if result.data else None,
                "depth_reached": result.data.get("depth_reached") if result.data else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error handling complex task: {e}")
            return {
                "success": False,
                "error": str(e),
                "routing_type": "complex_task_error"
            }
    
    def _determine_task_type(self, prompt: str) -> TaskType:
        """
        Determine the task type from the prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            TaskType: Task type enum
        """
        prompt_lower = prompt.lower()
        
        # Check for primary task types first
        if any(word in prompt_lower for word in ["forecast", "predict", "future", "prediction"]):
            return TaskType.FORECAST
        elif any(word in prompt_lower for word in ["strategy", "approach", "method", "trading strategy"]):
            return TaskType.STRATEGY
        elif any(word in prompt_lower for word in ["backtest", "back test", "historical test", "validate strategy"]):
            return TaskType.BACKTEST
        elif any(word in prompt_lower for word in ["build", "create", "train", "new model"]):
            return TaskType.MODEL_BUILD
        elif any(word in prompt_lower for word in ["evaluate", "assess", "test", "performance"]):
            return TaskType.MODEL_EVALUATE
        elif any(word in prompt_lower for word in ["update", "improve", "modify", "optimize"]):
            return TaskType.MODEL_UPDATE
        elif any(word in prompt_lower for word in ["strategy optimize", "optimize strategy"]):
            return TaskType.STRATEGY_OPTIMIZE
        elif any(word in prompt_lower for word in ["analyze", "data", "insights"]):
            return TaskType.DATA_ANALYSIS
        elif any(word in prompt_lower for word in ["forecast generate", "generate forecast"]):
            return TaskType.FORECAST_GENERATE
        elif any(word in prompt_lower for word in ["trade", "buy", "sell", "execute"]):
            return TaskType.TRADE_EXECUTE
        elif any(word in prompt_lower for word in ["risk", "danger", "safety"]):
            return TaskType.RISK_ASSESS
        else:
            return TaskType.GENERAL
    
    def _extract_task_parameters(self, prompt: str) -> Dict[str, Any]:
        """
        Extract task parameters from the prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            Dict: Task parameters
        """
        parameters = {}
        
        # Extract symbol
        import re
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', prompt.upper())
        if symbol_match:
            parameters["symbol"] = symbol_match.group(1)
        
        # Extract model type
        model_types = ["lstm", "xgboost", "ensemble", "transformer", "linear", "neural"]
        for model_type in model_types:
            if model_type in prompt.lower():
                parameters["model_type"] = model_type
                break
        
        # Extract timeframe
        timeframe_match = re.search(r'(\d+)\s*(day|week|month|hour)s?', prompt.lower())
        if timeframe_match:
            value = int(timeframe_match.group(1))
            unit = timeframe_match.group(2)
            if unit == "day":
                parameters["timeframe"] = f"{value}d"
            elif unit == "week":
                parameters["timeframe"] = f"{value * 7}d"
            elif unit == "month":
                parameters["timeframe"] = f"{value * 30}d"
            elif unit == "hour":
                parameters["timeframe"] = f"{value}h"
        
        return parameters
    
    async def _handle_workflow_prompt(self, prompt: str, user_id: str) -> Dict[str, Any]:
        """
        Handle workflow-specific prompts using TaskAgent.
        
        Args:
            prompt: User prompt
            user_id: User identifier
            
        Returns:
            Dict containing workflow result
        """
        try:
            if not self.task_agent:
                raise Exception("TaskAgent not available")
            
            # Determine workflow type and map to task type
            workflow_type = self._determine_workflow_type(prompt)
            task_type = self._map_workflow_to_task_type(workflow_type)
            
            # Extract parameters from prompt
            parameters = self._extract_task_parameters(prompt)
            parameters["user_id"] = user_id
            parameters["workflow_type"] = workflow_type
            
            # Execute the workflow using TaskAgent
            result = await self.task_agent.execute_task(
                prompt=prompt,
                task_type=task_type,
                parameters=parameters,
                max_depth=3,
                performance_threshold=0.7
            )
            
            return {
                "success": result.success,
                "response": result.data,
                "routing_type": "workflow",
                "workflow_type": workflow_type,
                "task_type": task_type.value,
                "message": result.message,
                "performance_score": result.performance_score
            }
            
        except Exception as e:
            self.logger.error(f"Error handling workflow prompt: {e}")
            return {
                "success": False,
                "error": str(e),
                "routing_type": "workflow_error"
            }
    
    def _determine_workflow_type(self, prompt: str) -> str:
        """
        Determine the workflow type from the prompt.
        
        Args:
            prompt: User prompt
            
        Returns:
            str: Workflow type
        """
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["build", "create", "train", "new model"]):
            return "builder"
        elif any(word in prompt_lower for word in ["evaluate", "assess", "test", "performance"]):
            return "evaluator"
        elif any(word in prompt_lower for word in ["update", "improve", "modify", "optimize"]):
            return "updater"
        else:
            return "builder"  # Default to builder
    
    def _map_workflow_to_task_type(self, workflow_type: str) -> TaskType:
        """
        Map workflow type to TaskType enum.
        
        Args:
            workflow_type: Workflow type string
            
        Returns:
            TaskType: Corresponding task type
        """
        workflow_mapping = {
            "builder": TaskType.MODEL_BUILD,
            "evaluator": TaskType.MODEL_EVALUATE,
            "updater": TaskType.MODEL_UPDATE
        }
        
        return workflow_mapping.get(workflow_type, TaskType.GENERAL)
    
    async def _process_with_router_agent(self, prompt: str, llm_type: str) -> Dict[str, Any]:
        """
        Process the prompt using the prompt router agent.
        
        Args:
            prompt: User's input prompt
            llm_type: Type of LLM to use
            
        Returns:
            Dict containing the processing result
        """
        try:
            # Check if this is an agent workflow request
            workflow_result = await self._check_for_agent_workflow(prompt)
            if workflow_result:
                return workflow_result
            
            # Try to get the prompt router agent from registry
            from agents.registry import get_prompt_router_agent
            
            router_agent = get_prompt_router_agent()
            if router_agent is None:
                # Fallback to direct prompt agent if router not available
                if self.prompt_agent:
                    return await self._process_with_prompt_agent(prompt, llm_type)
                else:
                    return self._create_fallback_response(prompt)
            
            # Use the router agent
            result = router_agent.handle_prompt(prompt)
            
            # Convert to standard format if needed
            if hasattr(result, 'message'):
                return {
                    "success": True,
                    "message": result.message,
                    "data": getattr(result, 'data', {}),
                    "strategy_name": getattr(result, 'strategy_name', None),
                    "model_used": getattr(result, 'model_used', None),
                    "confidence": getattr(result, 'confidence', None),
                    "signal": getattr(result, 'signal', None),
                }
            else:
                return result
                
        except ImportError as e:
            self.logger.warning(f"Prompt router agent not available: {e}")
            return await self._process_with_prompt_agent(prompt, llm_type)
        except Exception as e:
            self.logger.error(f"Error with router agent: {e}")
            return self._create_fallback_response(prompt)
    
    async def _process_with_prompt_agent(self, prompt: str, llm_type: str) -> Dict[str, Any]:
        """
        Process the prompt using the direct prompt agent.
        
        Args:
            prompt: User's input prompt
            llm_type: Type of LLM to use
            
        Returns:
            Dict containing the processing result
        """
        if not self.prompt_agent:
            return self._create_fallback_response(prompt)
        
        try:
            # Try to use the process_prompt method first
            if hasattr(self.prompt_agent, 'process_prompt'):
                result = self.prompt_agent.process_prompt(prompt)
                if hasattr(result, 'message'):
                    return {
                        "success": True,
                        "message": result.message,
                        "data": getattr(result, 'data', {}),
                        "strategy_name": getattr(result, 'strategy_name', None),
                        "model_used": getattr(result, 'model_used', None),
                        "confidence": getattr(result, 'confidence', None),
                        "signal": getattr(result, 'signal', None),
                    }
            
            # Fallback to handle_prompt method
            if hasattr(self.prompt_agent, 'handle_prompt'):
                result = self.prompt_agent.handle_prompt(prompt)
                return {
                    "success": True,
                    "message": result.get("message", "Request processed successfully"),
                    "data": result.get("data", {}),
                    "strategy_name": result.get("strategy_name"),
                    "model_used": result.get("model_used"),
                    "confidence": result.get("confidence"),
                    "signal": result.get("signal"),
                }
            
            # If neither method exists, create a simple response
            return {
                "success": True,
                "message": f"Processed your request: {prompt}. The system is analyzing your query.",
                "data": {"prompt": prompt},
                "strategy_name": None,
                "model_used": None,
                "confidence": 0.5,
                "signal": None,
            }
            
        except Exception as e:
            self.logger.error(f"Error with prompt agent: {e}")
            return self._create_fallback_response(prompt)
    
    async def _handle_general_prompt(self, prompt: str, user_id: str) -> Dict[str, Any]:
        """
        Handle general prompts using the prompt agent.
        
        Args:
            prompt: User prompt
            user_id: User identifier
            
        Returns:
            Dict containing prompt agent result
        """
        try:
            # Use the prompt agent for general prompts
            if self.prompt_agent:
                result = await self.prompt_agent.process_prompt(prompt)
                return {
                    "success": result.success,
                    "response": result.data,
                    "routing_type": "general_prompt",
                    "message": result.message
                }
            else:
                # Fallback to basic response
                return {
                    "success": True,
                    "response": {"message": "Prompt received and processed"},
                    "routing_type": "general_prompt",
                    "message": "General prompt processed"
                }
                
        except Exception as e:
            self.logger.error(f"Error handling general prompt: {e}")
            return {
                "success": False,
                "error": str(e),
                "routing_type": "general_prompt_error"
            }
    
    def _extract_navigation_info(self, result: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract navigation information from the result.
        
        Args:
            result: Processing result
            
        Returns:
            Dict containing navigation information
        """
        navigation_info = {}
        
        if not result.get("success"):
            return navigation_info
        
        message = result.get("message", "").lower()
        
        # Determine navigation based on message content
        if any(keyword in message for keyword in ["forecast", "prediction"]):
            navigation_info["main_nav"] = "Forecasting"
        elif any(keyword in message for keyword in ["strategy", "signal"]):
            navigation_info["main_nav"] = "Strategy Lab"
        elif any(keyword in message for keyword in ["report", "export", "analysis"]):
            navigation_info["main_nav"] = "Reports"
        elif any(keyword in message for keyword in ["tune", "optimize", "model"]):
            navigation_info["main_nav"] = "Model Lab"
        elif any(keyword in message for keyword in ["setting", "config"]):
            navigation_info["main_nav"] = "Settings"
        
        return navigation_info
    
    def _format_response(self, result: Dict[str, Any], navigation_info: Dict[str, str], start_time: datetime) -> Dict[str, Any]:
        """
        Format the final response.
        
        Args:
            result: Processing result
            navigation_info: Navigation information
            start_time: Start time of processing
            
        Returns:
            Dict containing the formatted response
        """
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "success": result.get("success", False),
            "message": result.get("message", "No response available"),
            "data": result.get("data", {}),
            "navigation_info": navigation_info,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add optional fields if available
        if result.get("strategy_name"):
            response["strategy_name"] = result["strategy_name"]
        if result.get("model_used"):
            response["model_used"] = result["model_used"]
        if result.get("confidence"):
            response["confidence"] = result["confidence"]
        if result.get("signal"):
            response["signal"] = result["signal"]
        
        return response
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create an error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Dict containing error response
        """
        return {
            "success": False,
            "message": error_message,
            "data": {},
            "navigation_info": {},
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error": True,
        }
    
    def _create_fallback_response(self, prompt: str) -> Dict[str, Any]:
        """
        Create a fallback response when no agents are available.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Dict containing fallback response
        """
        return {
            "success": True,
            "message": f"I understand you're asking about: '{prompt}'. I'm currently setting up the specialized agents to help you with this request. Please try again in a moment.",
            "data": {"fallback_used": True},
            "navigation_info": {},
            "processing_time": 0.0,
            "timestamp": datetime.now().isoformat(),
            "fallback": True,
        }
    
    async def _check_for_agent_workflow(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Check if the prompt is requesting an agent workflow and execute it.
        
        Args:
            prompt: User's input prompt
            
        Returns:
            Dict containing workflow result if applicable, None otherwise
        """
        if not self.agent_controller:
            return None
        
        try:
            # Simple keyword-based detection for now
            prompt_lower = prompt.lower()
            
            # Check for builder workflow
            if any(keyword in prompt_lower for keyword in ["build model", "create model", "train model", "new model"]):
                return await self._handle_builder_workflow(prompt)
            
            # Check for evaluator workflow
            if any(keyword in prompt_lower for keyword in ["evaluate model", "assess model", "test model", "model performance"]):
                return await self._handle_evaluator_workflow(prompt)
            
            # Check for updater workflow
            if any(keyword in prompt_lower for keyword in ["update model", "improve model", "retrain model", "optimize model"]):
                return await self._handle_updater_workflow(prompt)
            
            # Check for full pipeline
            if any(keyword in prompt_lower for keyword in ["full pipeline", "complete workflow", "build evaluate update"]):
                return await self._handle_full_pipeline_workflow(prompt)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking for agent workflow: {e}")
            return None
    
    async def _handle_builder_workflow(self, prompt: str) -> Dict[str, Any]:
        """Handle builder workflow requests using TaskAgent."""
        try:
            if not self.task_agent:
                raise Exception("TaskAgent not available")
            
            # Extract parameters from prompt
            params = self._extract_task_parameters(prompt)
            
            # Set default parameters for builder workflow
            params.setdefault("model_type", "lstm")
            params.setdefault("data_path", "data/sample_data.csv")
            params.setdefault("target_column", "close")
            
            # Try to extract model type from prompt
            if "lstm" in prompt.lower():
                params["model_type"] = "lstm"
            elif "xgboost" in prompt.lower():
                params["model_type"] = "xgboost"
            elif "ensemble" in prompt.lower():
                params["model_type"] = "ensemble"
            
            # Execute builder workflow using TaskAgent
            result = await self.task_agent.execute_task(
                prompt=prompt,
                task_type=TaskType.MODEL_BUILD,
                parameters=params,
                max_depth=3,
                performance_threshold=0.7
            )
            
            return {
                "success": result.success,
                "message": f"Model building workflow completed. {result.message}",
                "data": result.data,
                "workflow_type": "builder",
                "performance_score": result.performance_score,
                "task_id": result.data.get("task_id") if result.data else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Builder workflow failed: {str(e)}",
                "data": {},
                "workflow_type": "builder"
            }
    
    async def _handle_evaluator_workflow(self, prompt: str) -> Dict[str, Any]:
        """Handle evaluator workflow requests using TaskAgent."""
        try:
            if not self.task_agent:
                raise Exception("TaskAgent not available")
            
            # Extract parameters from prompt
            params = self._extract_task_parameters(prompt)
            
            # Set default parameters for evaluator workflow
            params.setdefault("model_id", "latest")
            params.setdefault("model_path", "models/latest.pkl")
            params.setdefault("test_data_path", "data/test_data.csv")
            
            # Execute evaluator workflow using TaskAgent
            result = await self.task_agent.execute_task(
                prompt=prompt,
                task_type=TaskType.MODEL_EVALUATE,
                parameters=params,
                max_depth=2,
                performance_threshold=0.6
            )
            
            return {
                "success": result.success,
                "message": f"Model evaluation workflow completed. {result.message}",
                "data": result.data,
                "workflow_type": "evaluator",
                "performance_score": result.performance_score,
                "task_id": result.data.get("task_id") if result.data else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Evaluator workflow failed: {str(e)}",
                "data": {},
                "workflow_type": "evaluator"
            }
    
    async def _handle_updater_workflow(self, prompt: str) -> Dict[str, Any]:
        """Handle updater workflow requests using TaskAgent."""
        try:
            if not self.task_agent:
                raise Exception("TaskAgent not available")
            
            # Extract parameters from prompt
            params = self._extract_task_parameters(prompt)
            
            # Set default parameters for updater workflow
            params.setdefault("model_id", "latest")
            params.setdefault("update_type", "auto")
            params.setdefault("evaluation_result", {})
            
            # Execute updater workflow using TaskAgent
            result = await self.task_agent.execute_task(
                prompt=prompt,
                task_type=TaskType.MODEL_UPDATE,
                parameters=params,
                max_depth=3,
                performance_threshold=0.7
            )
            
            return {
                "success": result.success,
                "message": f"Model update workflow completed. {result.message}",
                "data": result.data,
                "workflow_type": "updater",
                "performance_score": result.performance_score,
                "task_id": result.data.get("task_id") if result.data else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Updater workflow failed: {str(e)}",
                "data": {},
                "workflow_type": "updater"
            }
    
    async def _handle_full_pipeline_workflow(self, prompt: str) -> Dict[str, Any]:
        """Handle full pipeline workflow requests using TaskAgent."""
        try:
            if not self.task_agent:
                raise Exception("TaskAgent not available")
            
            # Extract parameters from prompt
            params = self._extract_task_parameters(prompt)
            
            # Set default parameters for full pipeline
            params.setdefault("model_type", "lstm")
            params.setdefault("data_path", "data/sample_data.csv")
            params.setdefault("target_column", "close")
            
            # Execute full pipeline using TaskAgent with higher depth for complete workflow
            result = await self.task_agent.execute_task(
                prompt=prompt,
                task_type=TaskType.MODEL_BUILD,  # Start with model build, TaskAgent will handle the full chain
                parameters=params,
                max_depth=5,  # Higher depth for full pipeline
                performance_threshold=0.8  # Higher threshold for full pipeline
            )
            
            return {
                "success": result.success,
                "message": f"Full pipeline completed. {result.message}",
                "data": result.data,
                "workflow_type": "full_pipeline",
                "performance_score": result.performance_score,
                "task_id": result.data.get("task_id") if result.data else None,
                "depth_reached": result.data.get("depth_reached") if result.data else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Full pipeline workflow failed: {str(e)}",
                "data": {},
                "workflow_type": "full_pipeline"
            }

    def _log_interaction(self, prompt: str, result: Dict[str, Any], start_time: datetime):
        """
        Log the interaction for debugging and monitoring.
        
        Args:
            prompt: Original prompt
            result: Processing result
            start_time: Start time of processing
        """
        try:
            if self.agent_logger:
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Log the interaction
                from trading.memory.agent_logger import AgentAction, LogLevel
                
                self.agent_logger.log_action(
                    agent_name="PromptRouter",
                    action=AgentAction.DATA_ANALYSIS,
                    message=f"Processed prompt: {prompt[:100]}...",
                    data={
                        "prompt": prompt,
                        "response": result.get("message", ""),
                        "success": result.get("success", False),
                        "processing_time": processing_time,
                        "strategy_name": result.get("strategy_name"),
                        "model_used": result.get("model_used"),
                        "confidence": result.get("confidence"),
                        "signal": result.get("signal"),
                    },
                    level=LogLevel.INFO if result.get("success") else LogLevel.ERROR
                )
        except Exception as e:
            self.logger.warning(f"Failed to log interaction: {e}")


# Global instance for easy access
_prompt_router = None


def get_prompt_router() -> PromptRouter:
    """
    Get a singleton instance of the prompt router.
    
    Returns:
        PromptRouter instance
    """
    global _prompt_router
    if _prompt_router is None:
        _prompt_router = PromptRouter()
    return _prompt_router


async def route_prompt(prompt: str, llm_type: str = "default") -> Dict[str, Any]:
    """
    Convenience function to route a prompt.
    
    Args:
        prompt: User's input prompt
        llm_type: Type of LLM to use
        
    Returns:
        Dict containing the routing result
    """
    router = get_prompt_router()
    return await router.route_prompt(prompt, llm_type) 