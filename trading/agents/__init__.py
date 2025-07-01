"""
Trading Agents Module

This module provides autonomous agents for model management and centralized prompt templates:
- ModelBuilderAgent: Builds ML models from scratch
- PerformanceCriticAgent: Evaluates model performance
- UpdaterAgent: Updates models based on evaluation results
- AgentLoopManager: Orchestrates the autonomous 3-agent system
- PromptRouterAgent: Routes prompts to appropriate agents
- Prompt Templates: Centralized source of truth for all prompt templates
"""

from .model_builder_agent import ModelBuilderAgent, ModelBuildRequest, ModelBuildResult
from .performance_critic_agent import PerformanceCriticAgent, ModelEvaluationRequest, ModelEvaluationResult
from .updater_agent import UpdaterAgent, UpdateRequest, UpdateResult
from .agent_loop_manager import AgentLoopManager, AgentLoopState, AgentCommunication
from .prompt_router_agent import PromptRouterAgent, ParsedIntent
from .prompt_templates import (
    PROMPT_TEMPLATES,
    get_template,
    format_template,
    get_templates_by_category,
    list_templates,
    list_categories,
    TEMPLATE_CATEGORIES
)

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
    'AgentCommunication',
    
    # Prompt Router Agent
    'PromptRouterAgent',
    'ParsedIntent',
    
    # Prompt Templates
    'PROMPT_TEMPLATES',
    'get_template',
    'format_template',
    'get_templates_by_category',
    'list_templates',
    'list_categories',
    'TEMPLATE_CATEGORIES'
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Autonomous 3-Agent Model Management System"
