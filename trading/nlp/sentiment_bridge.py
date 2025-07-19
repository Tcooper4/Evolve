"""
Sentiment Bridge - Batch 16
Refactored lexicon-based scoring with soft-matching support using cosine similarity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print("⚠️ sentence_transformers not available. Disabling soft-matching features.")
    print(f"   Missing: {e}")
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("⚠️ scikit-learn not available. Disabling cosine similarity calculations.")
    print(f"   Missing: {e}")
    cosine_similarity = None
    SKLEARN_AVAILABLE = False

# Overall embeddings availability
EMBEDDINGS_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE and SKLEARN_AVAILABLE
if not EMBEDDINGS_AVAILABLE:
    logger.warning("Embedding dependencies not available, soft-matching disabled")

@dataclass
class SoftMatchResult:
    """Result of soft-matching operation."""
    word: str
    matched_word: str
    similarity_score: float
    sentiment_score: float
    lexicon_name: str

@dataclass
class LexiconScore:
    """Lexicon scoring result."""
    total_score: float
    word_count: int
    matched_words: List[str]
    soft_matches: List[SoftMatchResult]
    confidence: float

class SentimentBridge:
    """
    Enhanced sentiment bridge with soft-matching capabilities.
    
    Features:
    - Traditional lexicon-based scoring
    - Soft-matching using cosine similarity for rare words
    - Configurable similarity thresholds
    - Multiple lexicon support
    """
    
    def __init__(self, 
                 enable_soft_matching: bool = True,
                 similarity_threshold: float = 0.7,
                 max_soft_matches: int = 5):
        """
        Initialize sentiment bridge.
        
        Args:
            enable_soft_matching: Enable soft-matching with embeddings
            similarity_threshold: Minimum similarity for soft-matching
            max_soft_matches: Maximum number of soft matches per word
        """
        self.enable_soft_matching = enable_soft_matching and EMBEDDINGS_AVAILABLE
        self.similarity_threshold = similarity_threshold
        self.max_soft_matches = max_soft_matches
        
        # Initialize embedding model if available
        self.embedding_model = None
        if self.enable_soft_matching:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded for soft-matching")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.enable_soft_matching = False
        
        # Lexicon embeddings cache
        self.lexicon_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Standard sentiment lexicons
        self.sentiment_lexicons = {
            "basic": {
                "bullish": 1.0, "bearish": -1.0, "positive": 0.8, "negative": -0.8,
                "strong": 0.6, "weak": -0.6, "growth": 0.7, "decline": -0.7,
                "profit": 0.9, "loss": -0.9, "gain": 0.8, "drop": -0.8,
                "rise": 0.7, "fall": -0.7, "up": 0.5, "down": -0.5,
                "good": 0.6, "bad": -0.6, "excellent": 0.9, "terrible": -0.9
            },
            "financial": {
                "earnings": 0.7, "revenue": 0.6, "profit": 0.9, "loss": -0.9,
                "dividend": 0.5, "buyback": 0.6, "acquisition": 0.4,
                "bankruptcy": -0.9, "liquidation": -0.8, "restructuring": -0.3,
                "growth": 0.7, "expansion": 0.6, "contraction": -0.6,
                "bull": 0.8, "bear": -0.8, "rally": 0.7, "crash": -0.9
            },
            "market": {
                "volatile": -0.3, "stable": 0.4, "trending": 0.5, "sideways": 0.0,
                "momentum": 0.6, "breakout": 0.7, "breakdown": -0.7,
                "support": 0.3, "resistance": -0.2, "consolidation": 0.1,
                "overbought": -0.4, "oversold": 0.4, "correction": -0.5
            }
        }
        
        # Pre-compute embeddings for lexicons if soft-matching is enabled
        if self.enable_soft_matching:
            self._precompute_lexicon_embeddings()
    
    def _precompute_lexicon_embeddings(self):
        """Pre-compute embeddings for all lexicon words."""
        try:
            for lexicon_name, lexicon in self.sentiment_lexicons.items():
                words = list(lexicon.keys())
                if words:
                    embeddings = self.embedding_model.encode(words)
                    self.lexicon_embeddings[lexicon_name] = {
                        word: embedding for word, embedding in zip(words, embeddings)
                    }
                    logger.info(f"Pre-computed embeddings for {len(words)} words in {lexicon_name} lexicon")
        except Exception as e:
            logger.error(f"Failed to pre-compute lexicon embeddings: {e}")
            self.enable_soft_matching = False
    
    def score_text(self, 
                  text: str, 
                  lexicon_name: str = "basic",
                  use_soft_matching: bool = None) -> LexiconScore:
        """
        Score text using lexicon-based approach with optional soft-matching.
        
        Args:
            text: Text to score
            lexicon_name: Name of lexicon to use
            use_soft_matching: Override global soft-matching setting
            
        Returns:
            LexiconScore object with scoring results
        """
        if use_soft_matching is None:
            use_soft_matching = self.enable_soft_matching
        
        if lexicon_name not in self.sentiment_lexicons:
            logger.warning(f"Lexicon '{lexicon_name}' not found, using 'basic'")
            lexicon_name = "basic"
        
        lexicon = self.sentiment_lexicons[lexicon_name]
        words = self._tokenize_text(text)
        
        # Direct matches
        matched_words = []
        total_score = 0.0
        
        for word in words:
            if word.lower() in lexicon:
                score = lexicon[word.lower()]
                matched_words.append(word)
                total_score += score
        
        # Soft matches
        soft_matches = []
        if use_soft_matching and self.embedding_model:
            soft_matches = self._find_soft_matches(words, lexicon_name)
            for match in soft_matches:
                total_score += match.sentiment_score * 0.7  # Weight soft matches less
        
        # Calculate confidence
        word_count = len(words)
        confidence = min(1.0, (len(matched_words) + len(soft_matches)) / max(word_count, 1))
        
        return LexiconScore(
            total_score=total_score,
            word_count=word_count,
            matched_words=matched_words,
            soft_matches=soft_matches,
            confidence=confidence
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        import re
        # Simple tokenization - split on whitespace and remove punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _find_soft_matches(self, 
                          words: List[str], 
                          lexicon_name: str) -> List[SoftMatchResult]:
        """
        Find soft matches for words using cosine similarity.
        
        Args:
            words: List of words to match
            lexicon_name: Name of lexicon to search in
            
        Returns:
            List of SoftMatchResult objects
        """
        if lexicon_name not in self.lexicon_embeddings:
            return []
        
        soft_matches = []
        lexicon = self.sentiment_lexicons[lexicon_name]
        lexicon_embeddings = self.lexicon_embeddings[lexicon_name]
        
        for word in words:
            if word in lexicon:  # Skip if direct match exists
                continue
            
            try:
                # Get embedding for the word
                word_embedding = self.embedding_model.encode([word])[0].reshape(1, -1)
                
                # Compare with lexicon embeddings
                best_matches = []
                for lexicon_word, lexicon_embedding in lexicon_embeddings.items():
                    similarity = cosine_similarity(
                        word_embedding, 
                        lexicon_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity >= self.similarity_threshold:
                        best_matches.append((lexicon_word, similarity))
                
                # Sort by similarity and take top matches
                best_matches.sort(key=lambda x: x[1], reverse=True)
                best_matches = best_matches[:self.max_soft_matches]
                
                for matched_word, similarity in best_matches:
                    soft_matches.append(SoftMatchResult(
                        word=word,
                        matched_word=matched_word,
                        similarity_score=similarity,
                        sentiment_score=lexicon[matched_word],
                        lexicon_name=lexicon_name
                    ))
                    
            except Exception as e:
                logger.warning(f"Failed to find soft matches for word '{word}': {e}")
                continue
        
        return soft_matches
    
    def score_batch(self, 
                   texts: List[str], 
                   lexicon_name: str = "basic") -> List[LexiconScore]:
        """
        Score multiple texts in batch.
        
        Args:
            texts: List of texts to score
            lexicon_name: Name of lexicon to use
            
        Returns:
            List of LexiconScore objects
        """
        return [self.score_text(text, lexicon_name) for text in texts]
    
    def get_lexicon_stats(self, lexicon_name: str = "basic") -> Dict[str, any]:
        """
        Get statistics about a lexicon.
        
        Args:
            lexicon_name: Name of lexicon
            
        Returns:
            Dictionary with lexicon statistics
        """
        if lexicon_name not in self.sentiment_lexicons:
            return {}
        
        lexicon = self.sentiment_lexicons[lexicon_name]
        scores = list(lexicon.values())
        
        return {
            "word_count": len(lexicon),
            "positive_words": len([s for s in scores if s > 0]),
            "negative_words": len([s for s in scores if s < 0]),
            "neutral_words": len([s for s in scores if s == 0]),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_score": np.mean(scores),
            "has_embeddings": lexicon_name in self.lexicon_embeddings
        }
    
    def add_custom_lexicon(self, 
                          name: str, 
                          lexicon: Dict[str, float],
                          compute_embeddings: bool = True):
        """
        Add a custom lexicon.
        
        Args:
            name: Name of the lexicon
            lexicon: Dictionary mapping words to sentiment scores
            compute_embeddings: Whether to compute embeddings for soft-matching
        """
        self.sentiment_lexicons[name] = lexicon
        
        if compute_embeddings and self.enable_soft_matching:
            try:
                words = list(lexicon.keys())
                if words:
                    embeddings = self.embedding_model.encode(words)
                    self.lexicon_embeddings[name] = {
                        word: embedding for word, embedding in zip(words, embeddings)
                    }
                    logger.info(f"Added custom lexicon '{name}' with {len(words)} words")
            except Exception as e:
                logger.error(f"Failed to compute embeddings for custom lexicon '{name}': {e}")
    
    def clear_embeddings_cache(self):
        """Clear the embeddings cache to free memory."""
        self.lexicon_embeddings.clear()
        logger.info("Embeddings cache cleared")

def create_sentiment_bridge(enable_soft_matching: bool = True) -> SentimentBridge:
    """Factory function to create a sentiment bridge."""
    return SentimentBridge(enable_soft_matching=enable_soft_matching) 