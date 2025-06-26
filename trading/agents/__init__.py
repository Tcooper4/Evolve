"""
Trading Agents Module

This module provides autonomous agents for model management:
- ModelBuilderAgent: Builds ML models from scratch
- PerformanceCriticAgent: Evaluates model performance
- UpdaterAgent: Updates models based on evaluation results
- AgentLoopManager: Orchestrates the autonomous 3-agent system
"""

from .model_builder_agent import ModelBuilderAgent, ModelBuildRequest, ModelBuildResult
from .performance_critic_agent import PerformanceCriticAgent, ModelEvaluationRequest, ModelEvaluationResult
from .updater_agent import UpdaterAgent, UpdateRequest, UpdateResult
from .agent_loop_manager import AgentLoopManager, AgentLoopState, AgentCommunication

__all__ = [
    # Model Builder Agent
    'ModelBuilderAgent',
    'ModelBuildRequest', 
    'ModelBuildResult',
    
    # Performance Critic Agent
    'PerformanceCriticAgent',
    'ModelEvaluationRequest',
    'ModelEvaluationResult',
    
    # Updater Agent
    'UpdaterAgent',
    'UpdateRequest',
    'UpdateResult',
    
    # Agent Loop Manager
    'AgentLoopManager',
    'AgentLoopState',
    'AgentCommunication'
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Autonomous 3-Agent Model Management System"
