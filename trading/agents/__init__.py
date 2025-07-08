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

# Base Agent Interface
from .base_agent_interface import (
    BaseAgent,
    AgentConfig,
    AgentStatus,
    AgentResult
)

# Model Management Agents
from .model_builder_agent import ModelBuilderAgent, ModelBuildRequest, ModelBuildResult
from .model_selector_agent import ModelSelectorAgent, ModelPerformance, ModelCapability, ForecastingHorizon, MarketRegime, ModelType
from .model_optimizer_agent import ModelOptimizerAgent, OptimizationResult, OptimizationConfig, OptimizationType, OptimizationStatus
from .model_evaluator_agent import ModelEvaluatorAgent, ModelEvaluationRequest, ModelEvaluationResult
from .model_improver_agent import ModelImproverAgent, ModelImprovementRequest, ModelImprovementResult
from .performance_critic_agent import PerformanceCriticAgent, ModelEvaluationRequest, ModelEvaluationResult

# Strategy Agents
from .strategy_selector_agent import StrategySelectorAgent, StrategyPerformance, StrategyRecommendation, StrategyType
from .strategy_improver_agent import StrategyImproverAgent, StrategyImprovementRequest, StrategyImprovementResult
from .meta_strategy_agent import MetaStrategyAgent, MetaStrategyRequest, MetaStrategyResult

# Market Analysis Agents
from .market_regime_agent import MarketRegimeAgent, MarketRegimeRequest, MarketRegimeResult
from .data_quality_agent import DataQualityAgent, DataQualityRequest, DataQualityResult

# Execution and Risk Agents
from .execution_agent import ExecutionAgent, ExecutionRequest, ExecutionResult
from .execution_risk_agent import ExecutionRiskAgent, RiskAssessmentRequest, RiskAssessmentResult
from .execution_risk_control_agent import ExecutionRiskControlAgent, RiskControlRequest, RiskControlResult

# Optimization Agents
from .optimizer_agent import OptimizerAgent, OptimizationRequest, OptimizationResult
from .self_tuning_optimizer_agent import SelfTuningOptimizerAgent, SelfTuningRequest, SelfTuningResult
from .meta_tuner_agent import MetaTunerAgent, MetaTuningRequest, MetaTuningResult

# Learning and Research Agents
from .meta_learner import MetaLearner, MetaLearningRequest, MetaLearningResult
from .meta_research_agent import MetaResearchAgent, ResearchRequest, ResearchResult
from .meta_learning_feedback_agent import MetaLearningFeedbackAgent, FeedbackRequest, FeedbackResult
from .research_agent import ResearchAgent, ResearchRequest, ResearchResult

# Management and Coordination Agents
from .agent_manager import AgentManager, AgentManagementRequest, AgentManagementResult
from .agent_loop_manager import AgentLoopManager, AgentLoopState, AgentCommunication
from .task_delegation_agent import TaskDelegationAgent, TaskDelegationRequest, TaskDelegationResult
from .agent_registry import AgentRegistry, AgentRegistrationRequest, AgentRegistrationResult
from .agent_leaderboard import AgentLeaderboard, LeaderboardRequest, LeaderboardResult

# Specialized Agents
from .prompt_router_agent import PromptRouterAgent, ParsedIntent, create_prompt_router
from .nlp_agent import NLPAgent, NLPRequest, NLPResult
from .multimodal_agent import MultimodalAgent, MultimodalRequest, MultimodalResult
from .rolling_retraining_agent import RollingRetrainingAgent, RetrainingRequest, RetrainingResult
from .walk_forward_agent import WalkForwardAgent, WalkForwardRequest, WalkForwardResult
from .updater_agent import UpdaterAgent, UpdateRequest, UpdateResult
from .self_improving_agent import SelfImprovingAgent, SelfImprovementRequest, SelfImprovementResult
from .intent_detector import IntentDetector, IntentDetectionRequest, IntentDetectionResult
from .market_analyzer_agent import MarketAnalyzerAgent
from .regime_detection_agent import RegimeDetectionAgent, create_regime_detection_agent
from .quant_gpt_agent import QuantGPTAgent
from .commentary_agent import CommentaryAgent, create_commentary_agent

# Prompt Templates
from .prompt_templates import (
    PROMPT_TEMPLATES,
    get_template,
    format_template,
    get_templates_by_category,
    list_templates,
    list_categories,
    TEMPLATE_CATEGORIES
)

# Commentary Engine
from trading.commentary import CommentaryEngine, create_commentary_engine

__all__ = [
    # Base Agent Interface
    'BaseAgent',
    'AgentConfig',
    'AgentStatus',
    'AgentResult',
    
    # Model Management Agents
    'ModelBuilderAgent',
    'ModelBuildRequest', 
    'ModelBuildResult',
    'ModelSelectorAgent',
    'ModelPerformance',
    'ModelCapability',
    'ForecastingHorizon',
    'MarketRegime',
    'ModelType',
    'ModelOptimizerAgent',
    'OptimizationResult',
    'OptimizationConfig',
    'OptimizationType',
    'OptimizationStatus',
    'ModelEvaluatorAgent',
    'ModelEvaluationRequest',
    'ModelEvaluationResult',
    'ModelImproverAgent',
    'ModelImprovementRequest',
    'ModelImprovementResult',
    'PerformanceCriticAgent',
    
    # Strategy Agents
    'StrategySelectorAgent',
    'StrategyPerformance',
    'StrategyRecommendation',
    'StrategyType',
    'StrategyImproverAgent',
    'StrategyImprovementRequest',
    'StrategyImprovementResult',
    'MetaStrategyAgent',
    'MetaStrategyRequest',
    'MetaStrategyResult',
    
    # Market Analysis Agents
    'MarketRegimeAgent',
    'MarketRegimeRequest',
    'MarketRegimeResult',
    'DataQualityAgent',
    'DataQualityRequest',
    'DataQualityResult',
    
    # Execution and Risk Agents
    'ExecutionAgent',
    'ExecutionRequest',
    'ExecutionResult',
    'ExecutionRiskAgent',
    'RiskAssessmentRequest',
    'RiskAssessmentResult',
    'ExecutionRiskControlAgent',
    'RiskControlRequest',
    'RiskControlResult',
    
    # Optimization Agents
    'OptimizerAgent',
    'OptimizationRequest',
    'OptimizationResult',
    'SelfTuningOptimizerAgent',
    'SelfTuningRequest',
    'SelfTuningResult',
    'MetaTunerAgent',
    'MetaTuningRequest',
    'MetaTuningResult',
    
    # Learning and Research Agents
    'MetaLearner',
    'MetaLearningRequest',
    'MetaLearningResult',
    'MetaResearchAgent',
    'ResearchRequest',
    'ResearchResult',
    'MetaLearningFeedbackAgent',
    'FeedbackRequest',
    'FeedbackResult',
    'ResearchAgent',
    
    # Management and Coordination Agents
    'AgentManager',
    'AgentManagementRequest',
    'AgentManagementResult',
    'AgentLoopManager',
    'AgentLoopState',
    'AgentCommunication',
    'TaskDelegationAgent',
    'TaskDelegationRequest',
    'TaskDelegationResult',
    'AgentRegistry',
    'AgentRegistrationRequest',
    'AgentRegistrationResult',
    'AgentLeaderboard',
    'LeaderboardRequest',
    'LeaderboardResult',
    
    # Specialized Agents
    'PromptRouterAgent',
    'ParsedIntent',
    'NLPAgent',
    'NLPRequest',
    'NLPResult',
    'MultimodalAgent',
    'MultimodalRequest',
    'MultimodalResult',
    'RollingRetrainingAgent',
    'RetrainingRequest',
    'RetrainingResult',
    'WalkForwardAgent',
    'WalkForwardRequest',
    'WalkForwardResult',
    'UpdaterAgent',
    'UpdateRequest',
    'UpdateResult',
    'SelfImprovingAgent',
    'SelfImprovementRequest',
    'SelfImprovementResult',
    'IntentDetector',
    'IntentDetectionRequest',
    'IntentDetectionResult',
    'MarketAnalyzerAgent',
    'RegimeDetectionAgent',
    'create_regime_detection_agent',
    'QuantGPTAgent',
    'CommentaryAgent',
    'create_commentary_agent',
    
    # Prompt Templates
    'PROMPT_TEMPLATES',
    'get_template',
    'format_template',
    'get_templates_by_category',
    'list_templates',
    'list_categories',
    'TEMPLATE_CATEGORIES',
    'create_prompt_router',
    
    # Commentary Engine
    'CommentaryEngine',
    'create_commentary_engine'
]

__version__ = "1.0.0"
__author__ = "Evolve Trading System"
__description__ = "Comprehensive Autonomous Agent Management System"
