"""
Hybrid Agent Prompt Memory System

This module provides a hybrid agent prompt memory system that adapts to
user intent trends over time, learning from user interactions and
optimizing prompt processing based on historical patterns.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict, deque
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """Categories of user intents."""
    FORECAST = "forecast"
    STRATEGY = "strategy"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    GENERAL = "general"
    INVESTMENT = "investment"
    TRADING = "trading"
    RESEARCH = "research"


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"  # Recent interactions
    LONG_TERM = "long_term"    # Historical patterns
    SEMANTIC = "semantic"      # Meaning-based associations
    CONTEXTUAL = "contextual"  # Context-dependent patterns


@dataclass
class UserIntent:
    """User intent record."""
    intent_id: str
    user_id: str
    session_id: str
    prompt: str
    intent_category: IntentCategory
    confidence: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentPattern:
    """Pattern of user intents."""
    pattern_id: str
    user_id: str
    intent_sequence: List[IntentCategory]
    frequency: int
    avg_confidence: float
    first_seen: datetime
    last_seen: datetime
    context_patterns: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptMemory:
    """Memory record for a prompt."""
    prompt_hash: str
    original_prompt: str
    intent_category: IntentCategory
    confidence: float
    timestamp: datetime
    user_id: str
    session_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    response_quality: float = 0.0
    user_feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User profile with intent preferences."""
    user_id: str
    created_at: datetime
    last_active: datetime
    total_interactions: int
    intent_preferences: Dict[IntentCategory, float] = field(default_factory=dict)
    common_patterns: List[str] = field(default_factory=list)
    preferred_contexts: Dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for memory system."""
    max_short_term_memory: int = 1000
    max_long_term_memory: int = 10000
    pattern_window_size: int = 10
    decay_factor: float = 0.95
    learning_rate: float = 0.1
    min_pattern_frequency: int = 3
    max_pattern_age_days: int = 30
    enable_semantic_memory: bool = True
    enable_contextual_memory: bool = True


class IntentTrendAnalyzer:
    """Analyzes trends in user intents."""
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize intent trend analyzer.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self.intent_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.pattern_window_size))
        self.pattern_cache: Dict[str, IntentPattern] = {}
        
    def add_intent(self, user_id: str, intent: UserIntent):
        """
        Add intent to trend analysis.
        
        Args:
            user_id: User identifier
            intent: User intent record
        """
        self.intent_history[user_id].append(intent)
        self._update_patterns(user_id)
    
    def _update_patterns(self, user_id: str):
        """Update patterns for a user."""
        intents = list(self.intent_history[user_id])
        if len(intents) < 2:
            return
        
        # Generate pattern sequences
        for window_size in range(2, min(len(intents) + 1, 6)):
            for i in range(len(intents) - window_size + 1):
                sequence = [intent.intent_category for intent in intents[i:i + window_size]]
                pattern_key = self._generate_pattern_key(sequence)
                
                if pattern_key in self.pattern_cache:
                    pattern = self.pattern_cache[pattern_key]
                    pattern.frequency += 1
                    pattern.last_seen = intents[i + window_size - 1].timestamp
                    
                    # Update average confidence
                    confidences = [intent.confidence for intent in intents[i:i + window_size]]
                    pattern.avg_confidence = np.mean(confidences)
                else:
                    pattern = IntentPattern(
                        pattern_id=pattern_key,
                        user_id=user_id,
                        intent_sequence=sequence,
                        frequency=1,
                        avg_confidence=np.mean([intent.confidence for intent in intents[i:i + window_size]]),
                        first_seen=intents[i].timestamp,
                        last_seen=intents[i + window_size - 1].timestamp
                    )
                    self.pattern_cache[pattern_key] = pattern
    
    def _generate_pattern_key(self, sequence: List[IntentCategory]) -> str:
        """Generate unique key for intent sequence."""
        return hashlib.md5(str(sequence).encode()).hexdigest()[:8]
    
    def get_user_patterns(self, user_id: str, min_frequency: int = None) -> List[IntentPattern]:
        """
        Get patterns for a user.
        
        Args:
            user_id: User identifier
            min_frequency: Minimum frequency threshold
            
        Returns:
            List of intent patterns
        """
        if min_frequency is None:
            min_frequency = self.config.min_pattern_frequency
        
        user_patterns = [
            pattern for pattern in self.pattern_cache.values()
            if pattern.user_id == user_id and pattern.frequency >= min_frequency
        ]
        
        # Sort by frequency and recency
        user_patterns.sort(key=lambda p: (p.frequency, p.last_seen), reverse=True)
        return user_patterns
    
    def predict_next_intent(self, user_id: str, current_intent: IntentCategory) -> Optional[IntentCategory]:
        """
        Predict next intent based on patterns.
        
        Args:
            user_id: User identifier
            current_intent: Current intent
            
        Returns:
            Predicted next intent or None
        """
        patterns = self.get_user_patterns(user_id)
        
        for pattern in patterns:
            if len(pattern.intent_sequence) > 1:
                # Find patterns that start with current intent
                for i in range(len(pattern.intent_sequence) - 1):
                    if pattern.intent_sequence[i] == current_intent:
                        return pattern.intent_sequence[i + 1]
        
        return None


class SemanticMemory:
    """Semantic memory for meaning-based associations."""
    
    def __init__(self, config: MemoryConfig):
        """
        Initialize semantic memory.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self.semantic_vectors: Dict[str, np.ndarray] = {}
        self.intent_embeddings: Dict[IntentCategory, np.ndarray] = {}
        self.similarity_cache: Dict[str, float] = {}
        
    def add_semantic_memory(self, prompt: str, intent: IntentCategory, embedding: np.ndarray):
        """
        Add semantic memory.
        
        Args:
            prompt: User prompt
            intent: Intent category
            embedding: Semantic embedding
        """
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self.semantic_vectors[prompt_hash] = embedding
        
        # Update intent embeddings
        if intent not in self.intent_embeddings:
            self.intent_embeddings[intent] = embedding
        else:
            # Update with moving average
            self.intent_embeddings[intent] = (
                self.intent_embeddings[intent] * 0.9 + embedding * 0.1
            )
    
    def find_similar_prompts(self, prompt: str, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar prompts.
        
        Args:
            prompt: Query prompt
            embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of (prompt_hash, similarity) tuples
        """
        similarities = []
        
        for prompt_hash, stored_embedding in self.semantic_vectors.items():
            similarity = self._cosine_similarity(embedding, stored_embedding)
            similarities.append((prompt_hash, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class HybridPromptMemory:
    """
    Hybrid prompt memory system that adapts to user intent trends.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None, storage_path: str = "data/hybrid_memory"):
        """
        Initialize hybrid prompt memory system.
        
        Args:
            config: Memory configuration
            storage_path: Path for persistent storage
        """
        self.config = config or MemoryConfig()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.trend_analyzer = IntentTrendAnalyzer(self.config)
        self.semantic_memory = SemanticMemory(self.config) if self.config.enable_semantic_memory else None
        
        # Memory storage
        self.short_term_memory: Dict[str, PromptMemory] = {}
        self.long_term_memory: Dict[str, PromptMemory] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Load existing data
        self._load_memory()
        
        logger.info("HybridPromptMemory initialized")
    
    def store_prompt(self, prompt: str, intent_category: IntentCategory, confidence: float,
                    user_id: str, session_id: str, context: Optional[Dict[str, Any]] = None,
                    embedding: Optional[np.ndarray] = None) -> str:
        """
        Store a prompt in memory.
        
        Args:
            prompt: User prompt
            intent_category: Detected intent category
            confidence: Confidence score
            user_id: User identifier
            session_id: Session identifier
            context: Additional context
            embedding: Semantic embedding (optional)
            
        Returns:
            Memory record ID
        """
        try:
            # Create memory record
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            timestamp = datetime.now()
            
            memory_record = PromptMemory(
                prompt_hash=prompt_hash,
                original_prompt=prompt,
                intent_category=intent_category,
                confidence=confidence,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                context=context or {}
            )
            
            # Store in short-term memory
            self.short_term_memory[prompt_hash] = memory_record
            
            # Add to trend analysis
            intent_record = UserIntent(
                intent_id=f"{user_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}",
                user_id=user_id,
                session_id=session_id,
                prompt=prompt,
                intent_category=intent_category,
                confidence=confidence,
                timestamp=timestamp,
                context=context or {}
            )
            
            self.trend_analyzer.add_intent(user_id, intent_record)
            
            # Add to semantic memory if available
            if self.semantic_memory and embedding is not None:
                self.semantic_memory.add_semantic_memory(prompt, intent_category, embedding)
            
            # Update user profile
            self._update_user_profile(user_id, intent_category, confidence, context)
            
            # Manage memory size
            self._manage_memory_size()
            
            logger.debug(f"Stored prompt in memory: {prompt_hash}")
            return prompt_hash
            
        except Exception as e:
            logger.error(f"Error storing prompt: {e}")
            return ""
    
    def retrieve_similar_prompts(self, prompt: str, user_id: str, top_k: int = 5,
                                embedding: Optional[np.ndarray] = None) -> List[PromptMemory]:
        """
        Retrieve similar prompts from memory.
        
        Args:
            prompt: Query prompt
            user_id: User identifier
            top_k: Number of results to return
            embedding: Semantic embedding (optional)
            
        Returns:
            List of similar prompt memories
        """
        try:
            similar_prompts = []
            
            # Search in short-term memory first
            for memory_record in self.short_term_memory.values():
                if memory_record.user_id == user_id:
                    similarity = self._calculate_similarity(prompt, memory_record.original_prompt, embedding)
                    similar_prompts.append((memory_record, similarity))
            
            # Search in long-term memory
            for memory_record in self.long_term_memory.values():
                if memory_record.user_id == user_id:
                    similarity = self._calculate_similarity(prompt, memory_record.original_prompt, embedding)
                    similar_prompts.append((memory_record, similarity))
            
            # Sort by similarity and return top_k
            similar_prompts.sort(key=lambda x: x[1], reverse=True)
            return [record for record, _ in similar_prompts[:top_k]]
            
        except Exception as e:
            logger.error(f"Error retrieving similar prompts: {e}")
            return []
    
    def _calculate_similarity(self, prompt1: str, prompt2: str, embedding: Optional[np.ndarray] = None) -> float:
        """Calculate similarity between prompts."""
        if embedding is not None and self.semantic_memory:
            # Use semantic similarity
            prompt2_hash = hashlib.md5(prompt2.encode()).hexdigest()
            if prompt2_hash in self.semantic_memory.semantic_vectors:
                return self.semantic_memory._cosine_similarity(
                    embedding, self.semantic_memory.semantic_vectors[prompt2_hash]
                )
        
        # Fallback to simple text similarity
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def predict_user_intent(self, user_id: str, current_intent: IntentCategory) -> Optional[IntentCategory]:
        """
        Predict user's next intent based on patterns.
        
        Args:
            user_id: User identifier
            current_intent: Current intent
            
        Returns:
            Predicted next intent or None
        """
        return self.trend_analyzer.predict_next_intent(user_id, current_intent)
    
    def get_user_patterns(self, user_id: str) -> List[IntentPattern]:
        """
        Get patterns for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of intent patterns
        """
        return self.trend_analyzer.get_user_patterns(user_id)
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile or None
        """
        return self.user_profiles.get(user_id)
    
    def update_response_quality(self, prompt_hash: str, quality_score: float, user_feedback: Optional[str] = None):
        """
        Update response quality for a prompt.
        
        Args:
            prompt_hash: Prompt hash
            quality_score: Quality score (0.0 to 1.0)
            user_feedback: User feedback (optional)
        """
        try:
            # Update in short-term memory
            if prompt_hash in self.short_term_memory:
                self.short_term_memory[prompt_hash].response_quality = quality_score
                self.short_term_memory[prompt_hash].user_feedback = user_feedback
            
            # Update in long-term memory
            if prompt_hash in self.long_term_memory:
                self.long_term_memory[prompt_hash].response_quality = quality_score
                self.long_term_memory[prompt_hash].user_feedback = user_feedback
            
        except Exception as e:
            logger.error(f"Error updating response quality: {e}")
    
    def _update_user_profile(self, user_id: str, intent_category: IntentCategory,
                           confidence: float, context: Optional[Dict[str, Any]] = None):
        """Update user profile with new intent."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                created_at=datetime.now(),
                last_active=datetime.now(),
                total_interactions=0
            )
        
        profile = self.user_profiles[user_id]
        profile.last_active = datetime.now()
        profile.total_interactions += 1
        
        # Update intent preferences
        if intent_category not in profile.intent_preferences:
            profile.intent_preferences[intent_category] = confidence
        else:
            # Update with moving average
            profile.intent_preferences[intent_category] = (
                profile.intent_preferences[intent_category] * (1 - self.config.learning_rate) +
                confidence * self.config.learning_rate
            )
        
        # Update context preferences
        if context:
            for key, value in context.items():
                if key not in profile.preferred_contexts:
                    profile.preferred_contexts[key] = value
                else:
                    # Update context with learning rate
                    if isinstance(value, (int, float)) and isinstance(profile.preferred_contexts[key], (int, float)):
                        profile.preferred_contexts[key] = (
                            profile.preferred_contexts[key] * (1 - self.config.learning_rate) +
                            value * self.config.learning_rate
                        )
    
    def _manage_memory_size(self):
        """Manage memory size by moving old records to long-term memory."""
        current_time = datetime.now()
        
        # Move old records from short-term to long-term memory
        to_move = []
        for prompt_hash, memory_record in self.short_term_memory.items():
            age_hours = (current_time - memory_record.timestamp).total_seconds() / 3600
            
            if age_hours > 24:  # Move after 24 hours
                to_move.append(prompt_hash)
        
        for prompt_hash in to_move:
            if prompt_hash in self.short_term_memory:
                self.long_term_memory[prompt_hash] = self.short_term_memory.pop(prompt_hash)
        
        # Clean up old long-term memory
        cutoff_time = current_time - timedelta(days=self.config.max_pattern_age_days)
        to_remove = []
        
        for prompt_hash, memory_record in self.long_term_memory.items():
            if memory_record.timestamp < cutoff_time:
                to_remove.append(prompt_hash)
        
        for prompt_hash in to_remove:
            self.long_term_memory.pop(prompt_hash, None)
        
        # Limit memory sizes
        if len(self.short_term_memory) > self.config.max_short_term_memory:
            # Remove oldest records
            sorted_records = sorted(
                self.short_term_memory.items(),
                key=lambda x: x[1].timestamp
            )
            excess = len(self.short_term_memory) - self.config.max_short_term_memory
            
            for i in range(excess):
                prompt_hash, _ = sorted_records[i]
                self.short_term_memory.pop(prompt_hash, None)
        
        if len(self.long_term_memory) > self.config.max_long_term_memory:
            # Remove oldest records
            sorted_records = sorted(
                self.long_term_memory.items(),
                key=lambda x: x[1].timestamp
            )
            excess = len(self.long_term_memory) - self.config.max_long_term_memory
            
            for i in range(excess):
                prompt_hash, _ = sorted_records[i]
                self.long_term_memory.pop(prompt_hash, None)
    
    def _load_memory(self):
        """Load memory from persistent storage."""
        try:
            # Load short-term memory
            short_term_path = self.storage_path / "short_term_memory.pkl"
            if short_term_path.exists():
                with open(short_term_path, 'rb') as f:
                    self.short_term_memory = pickle.load(f)
            
            # Load long-term memory
            long_term_path = self.storage_path / "long_term_memory.pkl"
            if long_term_path.exists():
                with open(long_term_path, 'rb') as f:
                    self.long_term_memory = pickle.load(f)
            
            # Load user profiles
            profiles_path = self.storage_path / "user_profiles.pkl"
            if profiles_path.exists():
                with open(profiles_path, 'rb') as f:
                    self.user_profiles = pickle.load(f)
            
            logger.info("Memory loaded from persistent storage")
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def save_memory(self):
        """Save memory to persistent storage."""
        try:
            # Save short-term memory
            short_term_path = self.storage_path / "short_term_memory.pkl"
            with open(short_term_path, 'wb') as f:
                pickle.dump(self.short_term_memory, f)
            
            # Save long-term memory
            long_term_path = self.storage_path / "long_term_memory.pkl"
            with open(long_term_path, 'wb') as f:
                pickle.dump(self.long_term_memory, f)
            
            # Save user profiles
            profiles_path = self.storage_path / "user_profiles.pkl"
            with open(profiles_path, 'wb') as f:
                pickle.dump(self.user_profiles, f)
            
            logger.info("Memory saved to persistent storage")
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "short_term_memory_size": len(self.short_term_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "user_profiles_count": len(self.user_profiles),
            "patterns_count": len(self.trend_analyzer.pattern_cache),
            "semantic_memory_enabled": self.semantic_memory is not None,
            "total_interactions": sum(profile.total_interactions for profile in self.user_profiles.values())
        }


def create_hybrid_prompt_memory(config: Optional[MemoryConfig] = None,
                               storage_path: str = "data/hybrid_memory") -> HybridPromptMemory:
    """
    Create a hybrid prompt memory instance.
    
    Args:
        config: Memory configuration
        storage_path: Path for persistent storage
        
    Returns:
        HybridPromptMemory instance
    """
    return HybridPromptMemory(config, storage_path) 