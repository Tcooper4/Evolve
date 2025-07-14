"""
Trading Agents Module

This module provides autonomous agents for model management and centralized prompt templates:
- BaseAgent: Base interface for all agents
- ModelBuilderAgent: Builds ML models from scratch
- PerformanceCriticAgent: Evaluates model performance
- UpdaterAgent: Updates models based on evaluation results
- AgentLoopManager: Orchestrates the autonomous 3-agent system
- PromptRouterAgent: Routes prompts to appropriate agents
- MarketRegimeAgent: Analyzes and adapts to market conditions
- StrategySelectorAgent: Selects optimal trading strategies
- ExecutionAgent: Executes trades and manages positions
- And many more specialized agents...
- Prompt Templates: Centralized source of truth for all prompt templates
"""

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Base Agent Interface - Import directly as it's lightweight

# Lazy loading functions to avoid circular imports


def _lazy_import(module_name: str, class_name: str) -> Any:
    """Lazy import a class from a module."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import {class_name} from {module_name}: {e}")
        return None


# Model Management Agents - Lazy loading


def get_model_builder_agent():
    """Get ModelBuilderAgent with lazy loading."""
    return _lazy_import(".model_builder_agent", "ModelBuilderAgent")


def get_model_selector_agent():
    """Get ModelSelectorAgent with lazy loading."""
    return _lazy_import(".model_selector_agent", "ModelSelectorAgent")


def get_model_optimizer_agent():
    """Get ModelOptimizerAgent with lazy loading."""
    return _lazy_import(".model_optimizer_agent", "ModelOptimizerAgent")


def get_model_evaluator_agent():
    """Get ModelEvaluatorAgent with lazy loading."""
    return _lazy_import(".model_evaluator_agent", "ModelEvaluatorAgent")


def get_model_improver_agent():
    """Get ModelImproverAgent with lazy loading."""
    return _lazy_import(".model_improver_agent", "ModelImproverAgent")


def get_model_synthesizer_agent():
    """Get ModelSynthesizerAgent with lazy loading."""
    return _lazy_import(".model_synthesizer_agent", "ModelSynthesizerAgent")


def get_performance_critic_agent():
    """Get PerformanceCriticAgent with lazy loading."""
    return _lazy_import(".performance_critic_agent", "PerformanceCriticAgent")


# Strategy Agents - Lazy loading


def get_strategy_selector_agent():
    """Get StrategySelectorAgent with lazy loading."""
    return _lazy_import(".strategy_selector_agent", "StrategySelectorAgent")


def get_strategy_improver_agent():
    """Get StrategyImproverAgent with lazy loading."""
    return _lazy_import(".strategy_improver_agent", "StrategyImproverAgent")


def get_meta_strategy_agent():
    """Get MetaStrategyAgent with lazy loading."""
    return _lazy_import(".meta_strategy_agent", "MetaStrategyAgent")


# Market Analysis Agents - Lazy loading


def get_market_regime_agent():
    """Get MarketRegimeAgent with lazy loading."""
    return _lazy_import(".market_regime_agent", "MarketRegimeAgent")


def get_data_quality_agent():
    """Get DataQualityAgent with lazy loading."""
    return _lazy_import(".data_quality_agent", "DataQualityAgent")


# Execution and Risk Agents - Lazy loading


def get_execution_agent():
    """Get ExecutionAgent with lazy loading."""
    return _lazy_import(".execution_agent", "ExecutionAgent")


def get_execution_risk_agent():
    """Get ExecutionRiskAgent with lazy loading."""
    return _lazy_import(".execution_risk_agent", "ExecutionRiskAgent")


def get_execution_risk_control_agent():
    """Get ExecutionRiskControlAgent with lazy loading."""
    return _lazy_import(".execution_risk_control_agent", "ExecutionRiskControlAgent")


# Optimization Agents - Lazy loading


def get_optimizer_agent():
    """Get OptimizerAgent with lazy loading."""
    return _lazy_import(".optimizer_agent", "OptimizerAgent")


def get_self_tuning_optimizer_agent():
    """Get SelfTuningOptimizerAgent with lazy loading."""
    return _lazy_import(".self_tuning_optimizer_agent", "SelfTuningOptimizerAgent")


def get_meta_tuner_agent():
    """Get MetaTunerAgent with lazy loading."""
    return _lazy_import(".meta_tuner_agent", "MetaTunerAgent")


# Learning and Research Agents - Lazy loading


def get_meta_learner():
    """Get MetaLearner with lazy loading."""
    return _lazy_import(".meta_learner", "MetaLearner")


def get_meta_research_agent():
    """Get MetaResearchAgent with lazy loading."""
    return _lazy_import(".meta_research_agent", "MetaResearchAgent")


def get_meta_learning_feedback_agent():
    """Get MetaLearningFeedbackAgent with lazy loading."""
    return _lazy_import(".meta_learning_feedback_agent", "MetaLearningFeedbackAgent")


def get_research_agent():
    """Get ResearchAgent with lazy loading."""
    return _lazy_import(".research_agent", "ResearchAgent")


# Management and Coordination Agents - Lazy loading


def get_agent_manager():
    """Get AgentManager with lazy loading."""
    return _lazy_import(".agent_manager", "AgentManager")


def get_agent_loop_manager():
    """Get AgentLoopManager with lazy loading."""
    return _lazy_import(".agent_loop_manager", "AgentLoopManager")


def get_task_delegation_agent():
    """Get TaskDelegationAgent with lazy loading."""
    return _lazy_import(".task_delegation_agent", "TaskDelegationAgent")


def get_agent_registry():
    """Get AgentRegistry with lazy loading."""
    return _lazy_import(".agent_registry", "AgentRegistry")


def get_agent_leaderboard():
    """Get AgentLeaderboard with lazy loading."""
    return _lazy_import(".agent_leaderboard", "AgentLeaderboard")


# Specialized Agents - Lazy loading


def get_prompt_router_agent():
    """Get PromptRouterAgent with lazy loading."""
    return _lazy_import(".prompt_router_agent", "PromptRouterAgent")


def get_nlp_agent():
    """Get NLPAgent with lazy loading."""
    return _lazy_import(".nlp_agent", "NLPAgent")


def get_multimodal_agent():
    """Get MultimodalAgent with lazy loading."""
    return _lazy_import(".multimodal_agent", "MultimodalAgent")


def get_rolling_retraining_agent():
    """Get RollingRetrainingAgent with lazy loading."""
    return _lazy_import(".rolling_retraining_agent", "RollingRetrainingAgent")


def get_walk_forward_agent():
    """Get WalkForwardAgent with lazy loading."""
    return _lazy_import(".walk_forward_agent", "WalkForwardAgent")


def get_updater_agent():
    """Get UpdaterAgent with lazy loading."""
    return _lazy_import(".updater_agent", "UpdaterAgent")


def get_self_improving_agent():
    """Get SelfImprovingAgent with lazy loading."""
    return _lazy_import(".self_improving_agent", "SelfImprovingAgent")


def get_intent_detector():
    """Get IntentDetector with lazy loading."""
    return _lazy_import(".intent_detector", "IntentDetector")


def get_regime_detection_agent():
    """Get RegimeDetectionAgent with lazy loading."""
    return _lazy_import(".regime_detection_agent", "RegimeDetectionAgent")


def get_commentary_agent():
    """Get CommentaryAgent with lazy loading."""
    return _lazy_import(".commentary_agent", "CommentaryAgent")


# Prompt Templates - Import directly as it's lightweight
try:
    from .prompt_templates import (
        PROMPT_TEMPLATES,
        get_template as get_prompt_template,
        list_templates as list_prompt_templates,
    )
except ImportError as e:
    logger.warning(f"Could not import prompt templates: {e}")
    PROMPT_TEMPLATES = {}
    get_prompt_template = lambda x: None
    list_prompt_templates = lambda: []

# Wildcard imports for core agents to improve usability during testing


def _import_core_agents():
    """Import core agents for wildcard access during testing."""
    core_agents = {}

    # Core agent classes to import
    core_agent_classes = [
        "BaseAgent",
        "AgentConfig",
        "AgentStatus",
        "AgentResult",
        "ModelBuilderAgent",
        "PerformanceCriticAgent",
        "ExecutionAgent",
        "AgentManager",
        "AgentLoopManager",
        "PromptRouterAgent",
        "MarketRegimeAgent",
        "StrategySelectorAgent",
        "CommentaryAgent",
        "AgentLeaderboard",
        "AgentRegistry",
    ]

    for class_name in core_agent_classes:
        try:
            if class_name in ["BaseAgent", "AgentConfig", "AgentStatus", "AgentResult"]:
                # These are already imported
                if class_name in globals():
                    core_agents[class_name] = globals()[class_name]
            else:
                # Try to import from specific modules
                module_mapping = {
                    "ModelBuilderAgent": ".model_builder_agent",
                    "PerformanceCriticAgent": ".performance_critic_agent",
                    "ExecutionAgent": ".execution_agent",
                    "AgentManager": ".agent_manager",
                    "AgentLoopManager": ".agent_loop_manager",
                    "PromptRouterAgent": ".prompt_router_agent",
                    "MarketRegimeAgent": ".market_regime_agent",
                    "StrategySelectorAgent": ".strategy_selector_agent",
                    "CommentaryAgent": ".commentary_agent",
                    "AgentLeaderboard": ".agent_leaderboard",
                    "AgentRegistry": ".agent_registry",
                }

                if class_name in module_mapping:
                    module_name = module_mapping[class_name]
                    agent_class = _lazy_import(module_name, class_name)
                    if agent_class:
                        core_agents[class_name] = agent_class

        except Exception as e:
            logger.debug(f"Could not import {class_name}: {e}")

    return core_agents


# Import core agents for wildcard access
CORE_AGENTS = _import_core_agents()


def get_core_agent(agent_name: str):
    """Get a core agent by name for easy access during testing.

    Args:
        agent_name: Name of the core agent

    Returns:
        Agent class or None if not found
    """
    return CORE_AGENTS.get(agent_name)


def list_core_agents() -> list:
    """List all available core agents.

    Returns:
        List of core agent names
    """
    return list(CORE_AGENTS.keys())


def import_all_agents():
    """Import all available agents for comprehensive testing.

    Returns:
        Dictionary of all available agents
    """
    all_agents = {}

    # Add core agents
    all_agents.update(CORE_AGENTS)

    # Add all lazy-loaded agents
    agent_functions = [
        get_model_builder_agent,
        get_model_selector_agent,
        get_model_optimizer_agent,
        get_model_evaluator_agent,
        get_model_improver_agent,
        get_model_synthesizer_agent,
        get_performance_critic_agent,
        get_strategy_selector_agent,
        get_strategy_improver_agent,
        get_meta_strategy_agent,
        get_market_regime_agent,
        get_data_quality_agent,
        get_execution_agent,
        get_execution_risk_agent,
        get_execution_risk_control_agent,
        get_optimizer_agent,
        get_self_tuning_optimizer_agent,
        get_meta_tuner_agent,
        get_meta_learner,
        get_meta_research_agent,
        get_meta_learning_feedback_agent,
        get_research_agent,
        get_agent_manager,
        get_agent_loop_manager,
        get_task_delegation_agent,
        get_agent_registry,
        get_agent_leaderboard,
        get_prompt_router_agent,
        get_nlp_agent,
        get_multimodal_agent,
        get_rolling_retraining_agent,
        get_walk_forward_agent,
        get_updater_agent,
        get_self_improving_agent,
        get_intent_detector,
        get_regime_detection_agent,
        get_commentary_agent,
    ]

    for func in agent_functions:
        try:
            agent_class = func()
            if agent_class:
                # Extract class name from function name
                class_name = (
                    func.__name__.replace("get_", "")
                    .replace("_agent", "Agent")
                    .replace("_", "")
                )
                all_agents[class_name] = agent_class
        except Exception as e:
            logger.debug(f"Could not import agent from {func.__name__}: {e}")

    return all_agents


# Convenience function for testing


def get_test_agents():
    """Get a subset of agents commonly used in testing.

    Returns:
        Dictionary of test agents
    """
    test_agents = {}

    # Essential agents for testing
    essential_agents = [
        "BaseAgent",
        "ModelBuilderAgent",
        "PerformanceCriticAgent",
        "ExecutionAgent",
        "AgentManager",
        "AgentLoopManager",
    ]

    for agent_name in essential_agents:
        agent_class = get_core_agent(agent_name)
        if agent_class:
            test_agents[agent_name] = agent_class

    return test_agents


__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Comprehensive Autonomous Agent Management System"
