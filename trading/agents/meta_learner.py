"""
Meta-Learning Agent for Trading System

Learns from past experiences and improves decision-making over time.
Implements meta-learning algorithms to adapt to changing market conditions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np
from dataclasses import dataclass
from .base_agent_interface import BaseAgent, AgentConfig, AgentResult

logger = logging.getLogger(__name__)

@dataclass
class MetaLearningRequest:
    """Meta-learning request."""
    action: str  # 'store_experience', 'learn_from_experiences', 'get_recommendation', 'get_summary', 'run'
    context: Optional[Dict[str, Any]] = None
    action_taken: Optional[str] = None
    outcome: Optional[Dict[str, Any]] = None
    performance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetaLearningResult:
    """Meta-learning result."""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class MetaLearningExperience:
    """Represents a learning experience for meta-learning."""
    timestamp: datetime
    context: Dict[str, Any]
    action: str
    outcome: Dict[str, Any]
    performance: float
    metadata: Dict[str, Any]

class MetaLearnerAgent(BaseAgent):
    """Meta-learning agent that improves decision-making over time."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="MetaLearnerAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        
        # Extract config from custom_config or use defaults
        custom_config = config.custom_config or {}
        self.memory_size = custom_config.get('memory_size', 1000)
        self.learning_rate = custom_config.get('learning_rate', 0.01)
        
        self.experiences: List[MetaLearningExperience] = []
        self.meta_models: Dict[str, Any] = {}
        self.performance_history: List[float] = []
        
        logger.info("Meta-learning agent initialized")

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the meta-learning logic.
        Args:
            **kwargs: context, action, outcome, performance, metadata, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'store_experience')
            
            if action == 'store_experience':
                context = kwargs.get('context')
                action_taken = kwargs.get('action_taken')
                outcome = kwargs.get('outcome')
                performance = kwargs.get('performance')
                metadata = kwargs.get('metadata')
                
                if context is None or action_taken is None or outcome is None or performance is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameters: context, action_taken, outcome, performance"
                    )
                
                self.store_experience(context, action_taken, outcome, performance, metadata)
                return AgentResult(success=True, data={
                    "message": "Experience stored successfully",
                    "total_experiences": len(self.experiences)
                })
                
            elif action == 'learn_from_experiences':
                insights = self.learn_from_experiences()
                return AgentResult(success=True, data={
                    "learning_insights": insights,
                    "meta_models_count": len(self.meta_models)
                })
                
            elif action == 'get_recommendation':
                context = kwargs.get('context')
                
                if context is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: context"
                    )
                
                recommendation = self.get_recommendation(context)
                return AgentResult(success=True, data={
                    "recommendation": recommendation
                })
                
            elif action == 'get_learning_summary':
                summary = self.get_learning_summary()
                return AgentResult(success=True, data={
                    "learning_summary": summary
                })
                
            elif action == 'run':
                context = kwargs.get('context', {})
                result = self.run(context)
                return AgentResult(success=True, data={
                    "meta_learning_result": result
                })
                
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            return self.handle_error(e)
    
    def store_experience(self, context: Dict[str, Any], action: str, 
                        outcome: Dict[str, Any], performance: float,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a learning experience.
        
        Args:
            context: Market context and conditions
            action: Action taken
            outcome: Result of the action
            performance: Performance metric
            metadata: Additional metadata
        """
        experience = MetaLearningExperience(
            timestamp=datetime.now(),
            context=context,
            action=action,
            outcome=outcome,
            performance=performance,
            metadata=metadata or {}
        )
        
        self.experiences.append(experience)
        
        # Maintain memory size
        if len(self.experiences) > self.memory_size:
            self.experiences.pop(0)
        
        # Update performance history
        self.performance_history.append(performance)
        if len(self.performance_history) > self.memory_size:
            self.performance_history.pop(0)
        
        logger.info(f"Stored experience: {action} with performance {performance:.3f}")
    
    def learn_from_experiences(self) -> Dict[str, Any]:
        """
        Learn from stored experiences.
        
        Returns:
            Dictionary with learning insights
        """
        if not self.experiences:
            return {}
        
        # Analyze performance trends
        recent_performance = self.performance_history[-100:] if len(self.performance_history) >= 100 else self.performance_history
        performance_trend = np.mean(recent_performance) - np.mean(self.performance_history[:-100]) if len(self.performance_history) >= 100 else 0
        
        # Identify successful patterns
        successful_experiences = [exp for exp in self.experiences if exp.performance > 0.7]
        unsuccessful_experiences = [exp for exp in self.experiences if exp.performance < 0.3]
        
        # Extract common patterns
        successful_patterns = self._extract_patterns(successful_experiences)
        unsuccessful_patterns = self._extract_patterns(unsuccessful_experiences)
        
        # Update meta-models
        self._update_meta_models(successful_patterns, unsuccessful_patterns)
        
        insights = {
            'total_experiences': len(self.experiences),
            'performance_trend': performance_trend,
            'successful_patterns': len(successful_patterns),
            'unsuccessful_patterns': len(unsuccessful_patterns),
            'average_performance': np.mean(self.performance_history),
            'learning_progress': self._calculate_learning_progress()
        }
        
        logger.info(f"Meta-learning completed: {insights}")
        return insights
    
    def _extract_patterns(self, experiences: List[MetaLearningExperience]) -> Dict[str, Any]:
        """Extract common patterns from experiences."""
        if not experiences:
            return {}
        
        patterns = {
            'common_contexts': {},
            'common_actions': {},
            'performance_correlations': {}
        }
        
        # Analyze contexts
        for exp in experiences:
            for key, value in exp.context.items():
                if key not in patterns['common_contexts']:
                    patterns['common_contexts'][key] = []
                patterns['common_contexts'][key].append(value)
        
        # Analyze actions
        for exp in experiences:
            if exp.action not in patterns['common_actions']:
                patterns['common_actions'][exp.action] = 0
            patterns['common_actions'][exp.action] += 1
        
        # Analyze performance correlations
        for exp in experiences:
            for key, value in exp.context.items():
                if key not in patterns['performance_correlations']:
                    patterns['performance_correlations'][key] = []
                patterns['performance_correlations'][key].append((value, exp.performance))
        
        return patterns
    
    def _update_meta_models(self, successful_patterns: Dict[str, Any], 
                           unsuccessful_patterns: Dict[str, Any]) -> None:
        """Update meta-models based on patterns."""
        # Update action success rates
        if 'action_success_rates' not in self.meta_models:
            self.meta_models['action_success_rates'] = {}
        
        all_actions = set(successful_patterns.get('common_actions', {}).keys()) | \
                     set(unsuccessful_patterns.get('common_actions', {}).keys())
        
        for action in all_actions:
            successful_count = successful_patterns.get('common_actions', {}).get(action, 0)
            unsuccessful_count = unsuccessful_patterns.get('common_actions', {}).get(action, 0)
            total_count = successful_count + unsuccessful_count
            
            if total_count > 0:
                success_rate = successful_count / total_count
                self.meta_models['action_success_rates'][action] = success_rate
        
        # Update context performance models
        if 'context_performance' not in self.meta_models:
            self.meta_models['context_performance'] = {}
        
        for context_key, correlations in successful_patterns.get('performance_correlations', {}).items():
            if correlations:
                avg_performance = np.mean([corr[1] for corr in correlations])
                self.meta_models['context_performance'][context_key] = avg_performance
    
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress based on performance improvement."""
        if len(self.performance_history) < 20:
            return 0.0
        
        recent_performance = np.mean(self.performance_history[-20:])
        early_performance = np.mean(self.performance_history[:20])
        
        if early_performance == 0:
            return 0.0
        
        progress = (recent_performance - early_performance) / early_performance
        return max(0.0, min(1.0, progress))
    
    def get_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendation based on learned patterns.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary with recommendation and confidence
        """
        if not self.meta_models:
            return {
                'recommendation': 'no_data',
                'confidence': 0.0,
                'reasoning': 'No learning data available'
            }
        
        # Score actions based on context
        action_scores = {}
        for action, success_rate in self.meta_models.get('action_success_rates', {}).items():
            score = success_rate
            
            # Adjust score based on context performance
            for context_key, context_value in context.items():
                if context_key in self.meta_models.get('context_performance', {}):
                    context_performance = self.meta_models['context_performance'][context_key]
                    score += context_performance * 0.1  # Small adjustment
            
            action_scores[action] = score
        
        if not action_scores:
            return {
                'recommendation': 'no_data',
                'confidence': 0.0,
                'reasoning': 'No action data available'
            }
        
        # Get best action
        best_action = max(action_scores, key=action_scores.get)
        confidence = action_scores[best_action]
        
        return {
            'recommendation': best_action,
            'confidence': confidence,
            'reasoning': f'Based on {len(self.experiences)} experiences',
            'action_scores': action_scores
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""
        return {
            'total_experiences': len(self.experiences),
            'average_performance': np.mean(self.performance_history) if self.performance_history else 0,
            'learning_progress': self._calculate_learning_progress(),
            'meta_models': {
                'action_success_rates': len(self.meta_models.get('action_success_rates', {})),
                'context_performance': len(self.meta_models.get('context_performance', {}))
            },
            'recent_performance_trend': self._get_recent_trend()
        }
    
    def _get_recent_trend(self) -> str:
        """Get recent performance trend."""
        if len(self.performance_history) < 10:
            return 'insufficient_data'
        
        recent = np.mean(self.performance_history[-10:])
        previous = np.mean(self.performance_history[-20:-10]) if len(self.performance_history) >= 20 else recent
        
        if recent > previous * 1.1:
            return 'improving'
        elif recent < previous * 0.9:
            return 'declining'
        else:
            return 'stable'
    
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the meta-learning agent.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary with meta-learning results
        """
        try:
            # Learn from experiences
            learning_results = self.learn_from_experiences()
            
            # Get recommendation if context provided
            recommendation = None
            if context:
                recommendation = self.get_recommendation(context)
            
            # Get summary
            summary = self.get_learning_summary()
            
            return {
                'success': True,
                'learning_results': learning_results,
                'recommendation': recommendation,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in meta-learning agent: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Alias for backward compatibility
MetaLearner = MetaLearnerAgent 