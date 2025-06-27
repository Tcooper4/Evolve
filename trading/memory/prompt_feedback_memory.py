"""
Prompt Feedback Memory System

Stores and learns from user interactions and prompt feedback.
Implements a memory loop for continuous improvement.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PromptInteraction:
    """Represents a user prompt interaction."""
    timestamp: datetime
    prompt: str
    response: Dict[str, Any]
    user_feedback: Optional[float] = None  # 0.0 to 1.0
    user_comment: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class PromptPattern:
    """Represents a learned prompt pattern."""
    pattern: str
    success_rate: float
    usage_count: int
    last_used: datetime
    context: Dict[str, Any]

class PromptFeedbackMemory:
    """Memory system for storing and learning from prompt interactions."""
    
    def __init__(self, memory_size: int = 1000, feedback_threshold: float = 0.7):
        """
        Initialize the prompt feedback memory system.
        
        Args:
            memory_size: Maximum number of interactions to remember
            feedback_threshold: Threshold for considering feedback positive
        """
        self.memory_size = memory_size
        self.feedback_threshold = feedback_threshold
        self.interactions: List[PromptInteraction] = []
        self.patterns: Dict[str, PromptPattern] = {}
        self.feedback_stats: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("Prompt feedback memory system initialized")
    
    def store_interaction(self, prompt: str, response: Dict[str, Any], 
                         user_feedback: Optional[float] = None,
                         user_comment: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a prompt interaction.
        
        Args:
            prompt: User prompt
            response: System response
            user_feedback: User feedback score (0.0 to 1.0)
            user_comment: User comment
            metadata: Additional metadata
        """
        interaction = PromptInteraction(
            timestamp=datetime.now(),
            prompt=prompt,
            response=response,
            user_feedback=user_feedback,
            user_comment=user_comment,
            metadata=metadata or {}
        )
        
        self.interactions.append(interaction)
        
        # Maintain memory size
        if len(self.interactions) > self.memory_size:
            self.interactions.pop(0)
        
        # Update feedback statistics
        if user_feedback is not None:
            self.feedback_stats['overall'].append(user_feedback)
            
            # Categorize by prompt type
            prompt_type = self._categorize_prompt(prompt)
            self.feedback_stats[prompt_type].append(user_feedback)
        
        logger.info(f"Stored interaction: {prompt[:50]}... with feedback {user_feedback}")
    
    def _categorize_prompt(self, prompt: str) -> str:
        """Categorize prompt by type."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['forecast', 'predict', 'price']):
            return 'forecasting'
        elif any(word in prompt_lower for word in ['tune', 'optimize', 'parameter']):
            return 'optimization'
        elif any(word in prompt_lower for word in ['strategy', 'backtest', 'trade']):
            return 'strategy'
        elif any(word in prompt_lower for word in ['portfolio', 'position', 'holdings']):
            return 'portfolio'
        elif any(word in prompt_lower for word in ['risk', 'volatility', 'drawdown']):
            return 'risk'
        else:
            return 'general'
    
    def learn_patterns(self) -> Dict[str, Any]:
        """
        Learn patterns from stored interactions.
        
        Returns:
            Dictionary with learning insights
        """
        if not self.interactions:
            return {'message': 'No interactions to learn from'}
        
        # Analyze successful patterns
        successful_interactions = [
            interaction for interaction in self.interactions
            if interaction.user_feedback and interaction.user_feedback >= self.feedback_threshold
        ]
        
        unsuccessful_interactions = [
            interaction for interaction in self.interactions
            if interaction.user_feedback and interaction.user_feedback < self.feedback_threshold
        ]
        
        # Extract patterns
        successful_patterns = self._extract_patterns(successful_interactions)
        unsuccessful_patterns = self._extract_patterns(unsuccessful_interactions)
        
        # Update pattern database
        self._update_patterns(successful_patterns, unsuccessful_patterns)
        
        insights = {
            'total_interactions': len(self.interactions),
            'successful_interactions': len(successful_interactions),
            'unsuccessful_interactions': len(unsuccessful_interactions),
            'success_rate': len(successful_interactions) / len(self.interactions) if self.interactions else 0,
            'patterns_learned': len(self.patterns),
            'feedback_by_category': self._get_feedback_by_category()
        }
        
        logger.info(f"Pattern learning completed: {insights}")
        return insights
    
    def _extract_patterns(self, interactions: List[PromptInteraction]) -> Dict[str, Any]:
        """Extract patterns from interactions."""
        if not interactions:
            return {}
        
        patterns = {
            'common_keywords': defaultdict(int),
            'response_types': defaultdict(int),
            'context_patterns': defaultdict(list)
        }
        
        for interaction in interactions:
            # Extract keywords
            words = interaction.prompt.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    patterns['common_keywords'][word] += 1
            
            # Categorize response types
            response_type = self._categorize_response(interaction.response)
            patterns['response_types'][response_type] += 1
            
            # Extract context patterns
            if interaction.metadata:
                for key, value in interaction.metadata.items():
                    patterns['context_patterns'][key].append(value)
        
        return patterns
    
    def _categorize_response(self, response: Dict[str, Any]) -> str:
        """Categorize response type."""
        if 'error' in response:
            return 'error'
        elif 'forecast' in response.get('type', ''):
            return 'forecast'
        elif 'tuning' in response.get('type', ''):
            return 'tuning'
        elif 'strategy' in response.get('type', ''):
            return 'strategy'
        elif 'portfolio' in response.get('type', ''):
            return 'portfolio'
        else:
            return 'general'
    
    def _update_patterns(self, successful_patterns: Dict[str, Any], 
                        unsuccessful_patterns: Dict[str, Any]) -> None:
        """Update pattern database."""
        # Update keyword patterns
        all_keywords = set(successful_patterns.get('common_keywords', {}).keys()) | \
                      set(unsuccessful_patterns.get('common_keywords', {}).keys())
        
        for keyword in all_keywords:
            successful_count = successful_patterns.get('common_keywords', {}).get(keyword, 0)
            unsuccessful_count = unsuccessful_patterns.get('common_keywords', {}).get(keyword, 0)
            total_count = successful_count + unsuccessful_count
            
            if total_count >= 3:  # Minimum threshold
                success_rate = successful_count / total_count if total_count > 0 else 0
                
                pattern_key = f"keyword_{keyword}"
                self.patterns[pattern_key] = PromptPattern(
                    pattern=keyword,
                    success_rate=success_rate,
                    usage_count=total_count,
                    last_used=datetime.now(),
                    context={'type': 'keyword'}
                )
    
    def _get_feedback_by_category(self) -> Dict[str, float]:
        """Get average feedback by category."""
        feedback_by_category = {}
        
        for category, feedbacks in self.feedback_stats.items():
            if feedbacks:
                feedback_by_category[category] = np.mean(feedbacks)
        
        return feedback_by_category
    
    def get_prompt_suggestions(self, partial_prompt: str) -> List[Dict[str, Any]]:
        """
        Get prompt suggestions based on learned patterns.
        
        Args:
            partial_prompt: Partial user prompt
            
        Returns:
            List of prompt suggestions with confidence scores
        """
        suggestions = []
        
        # Find similar successful patterns
        for pattern_key, pattern in self.patterns.items():
            if pattern.success_rate > 0.7 and pattern.usage_count >= 5:
                # Check if pattern is relevant to partial prompt
                if self._is_pattern_relevant(pattern.pattern, partial_prompt):
                    suggestions.append({
                        'suggestion': f"{partial_prompt} {pattern.pattern}",
                        'confidence': pattern.success_rate,
                        'usage_count': pattern.usage_count,
                        'pattern': pattern.pattern
                    })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _is_pattern_relevant(self, pattern: str, partial_prompt: str) -> bool:
        """Check if pattern is relevant to partial prompt."""
        partial_lower = partial_prompt.lower()
        pattern_lower = pattern.lower()
        
        # Check for semantic similarity
        if pattern_lower in partial_lower or partial_lower in pattern_lower:
            return True
        
        # Check for category similarity
        partial_category = self._categorize_prompt(partial_prompt)
        pattern_category = self._categorize_prompt(pattern)
        
        return partial_category == pattern_category
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory system."""
        return {
            'total_interactions': len(self.interactions),
            'total_patterns': len(self.patterns),
            'average_feedback': np.mean(self.feedback_stats['overall']) if self.feedback_stats['overall'] else 0,
            'feedback_by_category': self._get_feedback_by_category(),
            'recent_interactions': len([i for i in self.interactions if i.timestamp > datetime.now() - timedelta(days=7)]),
            'top_patterns': self._get_top_patterns()
        }
    
    def _get_top_patterns(self) -> List[Dict[str, Any]]:
        """Get top performing patterns."""
        top_patterns = []
        
        for pattern_key, pattern in self.patterns.items():
            if pattern.success_rate > 0.6 and pattern.usage_count >= 3:
                top_patterns.append({
                    'pattern': pattern.pattern,
                    'success_rate': pattern.success_rate,
                    'usage_count': pattern.usage_count,
                    'last_used': pattern.last_used.isoformat()
                })
        
        # Sort by success rate
        top_patterns.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return top_patterns[:10]  # Return top 10 patterns
    
    def run_memory_loop(self) -> Dict[str, Any]:
        """
        Run the memory learning loop.
        
        Returns:
            Dictionary with memory loop results
        """
        try:
            # Learn patterns
            learning_results = self.learn_patterns()
            
            # Get summary
            summary = self.get_memory_summary()
            
            return {
                'success': True,
                'learning_results': learning_results,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in memory loop: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 