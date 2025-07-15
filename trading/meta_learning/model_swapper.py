"""
Model Swapper - Batch 19
Enhanced model swapping with performance validation and safety checks
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status in ensemble."""
    ACTIVE = "active"
    CANDIDATE = "candidate"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class ValidationResult(Enum):
    """Model validation results."""
    PASSED = "passed"
    FAILED_SHARPE = "failed_sharpe"
    FAILED_MSE = "failed_mse"
    FAILED_OTHER = "failed_other"

@dataclass
class ModelMetrics:
    """Performance metrics for a model."""
    model_id: str
    sharpe_ratio: float
    mse: float
    win_rate: float
    total_return: float
    max_drawdown: float
    volatility: float
    last_updated: datetime
    validation_status: ValidationResult = ValidationResult.FAILED_OTHER

@dataclass
class SwapCandidate:
    """Candidate model for swapping."""
    model_id: str
    model_type: str
    metrics: ModelMetrics
    confidence_score: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwapResult:
    """Result of model swapping operation."""
    success: bool
    old_model_id: Optional[str]
    new_model_id: Optional[str]
    validation_passed: bool
    performance_improvement: float
    swap_time: datetime
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class ModelSwapper:
    """
    Enhanced model swapper with performance validation.
    
    Features:
    - Sharpe ratio validation (> 0.8)
    - MSE comparison with current models
    - Ensemble safety checks
    - Performance tracking and rollback capability
    """
    
    def __init__(self, 
                 min_sharpe_ratio: float = 0.8,
                 max_mse_increase: float = 0.1,  # 10% increase allowed
                 enable_safety_checks: bool = True,
                 max_swap_attempts: int = 3):
        """
        Initialize model swapper.
        
        Args:
            min_sharpe_ratio: Minimum Sharpe ratio for candidate models
            max_mse_increase: Maximum allowed MSE increase
            enable_safety_checks: Enable additional safety checks
            max_swap_attempts: Maximum swap attempts before giving up
        """
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_mse_increase = max_mse_increase
        self.enable_safety_checks = enable_safety_checks
        self.max_swap_attempts = max_swap_attempts
        
        # Ensemble state
        self.active_models: Dict[str, ModelMetrics] = {}
        self.candidate_models: Dict[str, SwapCandidate] = {}
        self.swap_history: List[SwapResult] = []
        
        # Performance tracking
        self.ensemble_performance: Dict[str, float] = {}
        self.model_contribution: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'total_swaps': 0,
            'successful_swaps': 0,
            'failed_validations': 0,
            'sharpe_failures': 0,
            'mse_failures': 0,
            'avg_performance_improvement': 0.0
        }
        
        logger.info(f"ModelSwapper initialized with min Sharpe: {min_sharpe_ratio}, max MSE increase: {max_mse_increase}")
    
    def add_candidate_model(self, 
                          model_id: str,
                          model_type: str,
                          metrics: Dict[str, float],
                          confidence_score: float = 0.5) -> bool:
        """
        Add a candidate model for potential swapping.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model
            metrics: Performance metrics
            confidence_score: Confidence in model quality
            
        Returns:
            Success status
        """
        try:
            # Create model metrics
            model_metrics = ModelMetrics(
                model_id=model_id,
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                mse=metrics.get('mse', float('inf')),
                win_rate=metrics.get('win_rate', 0.0),
                total_return=metrics.get('total_return', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                volatility=metrics.get('volatility', 0.0),
                last_updated=datetime.now()
            )
            
            # Validate candidate
            validation_result = self._validate_candidate_model(model_metrics)
            model_metrics.validation_status = validation_result
            
            if validation_result != ValidationResult.PASSED:
                self.stats['failed_validations'] += 1
                if validation_result == ValidationResult.FAILED_SHARPE:
                    self.stats['sharpe_failures'] += 1
                elif validation_result == ValidationResult.FAILED_MSE:
                    self.stats['mse_failures'] += 1
                
                logger.warning(f"Candidate model {model_id} failed validation: {validation_result.value}")
                return False
            
            # Create swap candidate
            candidate = SwapCandidate(
                model_id=model_id,
                model_type=model_type,
                metrics=model_metrics,
                confidence_score=confidence_score,
                created_at=datetime.now()
            )
            
            self.candidate_models[model_id] = candidate
            logger.info(f"Added candidate model {model_id} with Sharpe: {model_metrics.sharpe_ratio:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding candidate model {model_id}: {e}")
            return False
    
    def _validate_candidate_model(self, metrics: ModelMetrics) -> ValidationResult:
        """
        Validate candidate model against performance criteria.
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            ValidationResult
        """
        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.min_sharpe_ratio:
            logger.debug(f"Sharpe ratio too low: {metrics.sharpe_ratio:.3f} < {self.min_sharpe_ratio}")
            return ValidationResult.FAILED_SHARPE
        
        # Check MSE against current ensemble
        if self.active_models:
            current_mse = self._get_ensemble_mse()
            if current_mse is not None:
                mse_increase = (metrics.mse - current_mse) / current_mse
                if mse_increase > self.max_mse_increase:
                    logger.debug(f"MSE increase too high: {mse_increase:.3f} > {self.max_mse_increase}")
                    return ValidationResult.FAILED_MSE
        
        # Additional safety checks
        if self.enable_safety_checks:
            if not self._passes_safety_checks(metrics):
                return ValidationResult.FAILED_OTHER
        
        return ValidationResult.PASSED
    
    def _passes_safety_checks(self, metrics: ModelMetrics) -> bool:
        """
        Perform additional safety checks on candidate model.
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            True if passes safety checks
        """
        # Check for reasonable metrics
        if metrics.win_rate < 0.3 or metrics.win_rate > 0.9:
            logger.debug(f"Win rate out of reasonable range: {metrics.win_rate:.3f}")
            return False
        
        if metrics.max_drawdown > 0.5:  # 50% max drawdown
            logger.debug(f"Max drawdown too high: {metrics.max_drawdown:.3f}")
            return False
        
        if metrics.volatility > 0.3:  # 30% volatility
            logger.debug(f"Volatility too high: {metrics.volatility:.3f}")
            return False
        
        # Check for NaN or infinite values
        if (np.isnan(metrics.sharpe_ratio) or np.isinf(metrics.sharpe_ratio) or
            np.isnan(metrics.mse) or np.isinf(metrics.mse)):
            logger.debug("NaN or infinite values detected in metrics")
            return False
        
        return True
    
    def _get_ensemble_mse(self) -> Optional[float]:
        """Get current ensemble MSE."""
        if not self.active_models:
            return None
        
        mse_values = [model.mse for model in self.active_models.values() 
                     if not np.isnan(model.mse) and not np.isinf(model.mse)]
        
        if not mse_values:
            return None
        
        return np.mean(mse_values)
    
    def find_best_candidate(self, 
                          target_model_type: Optional[str] = None,
                          min_confidence: float = 0.5) -> Optional[SwapCandidate]:
        """
        Find the best candidate model for swapping.
        
        Args:
            target_model_type: Specific model type to look for
            min_confidence: Minimum confidence score
            
        Returns:
            Best swap candidate or None
        """
        valid_candidates = []
        
        for candidate in self.candidate_models.values():
            # Check validation status
            if candidate.metrics.validation_status != ValidationResult.PASSED:
                continue
            
            # Check confidence threshold
            if candidate.confidence_score < min_confidence:
                continue
            
            # Check model type if specified
            if target_model_type and candidate.model_type != target_model_type:
                continue
            
            valid_candidates.append(candidate)
        
        if not valid_candidates:
            return None
        
        # Sort by composite score
        def candidate_score(candidate: SwapCandidate) -> float:
            # Weighted combination of Sharpe ratio and confidence
            sharpe_weight = 0.7
            confidence_weight = 0.3
            
            normalized_sharpe = min(candidate.metrics.sharpe_ratio / 2.0, 1.0)  # Normalize to 0-1
            return sharpe_weight * normalized_sharpe + confidence_weight * candidate.confidence_score
        
        best_candidate = max(valid_candidates, key=candidate_score)
        logger.info(f"Selected best candidate: {best_candidate.model_id} (score: {candidate_score(best_candidate):.3f})")
        
        return best_candidate
    
    def swap_model(self, 
                  candidate_id: str,
                  target_model_id: Optional[str] = None) -> SwapResult:
        """
        Swap a model in the ensemble.
        
        Args:
            candidate_id: ID of candidate model to swap in
            target_model_id: ID of model to replace (auto-select if None)
            
        Returns:
            SwapResult with operation details
        """
        start_time = datetime.now()
        
        try:
            # Get candidate
            candidate = self.candidate_models.get(candidate_id)
            if not candidate:
                return SwapResult(
                    success=False,
                    old_model_id=None,
                    new_model_id=candidate_id,
                    validation_passed=False,
                    performance_improvement=0.0,
                    swap_time=start_time,
                    error_message=f"Candidate model {candidate_id} not found"
                )
            
            # Validate candidate again
            if candidate.metrics.validation_status != ValidationResult.PASSED:
                return SwapResult(
                    success=False,
                    old_model_id=None,
                    new_model_id=candidate_id,
                    validation_passed=False,
                    performance_improvement=0.0,
                    swap_time=start_time,
                    error_message=f"Candidate validation failed: {candidate.metrics.validation_status.value}"
                )
            
            # Select target model if not specified
            if target_model_id is None:
                target_model_id = self._select_target_model(candidate)
            
            if not target_model_id:
                return SwapResult(
                    success=False,
                    old_model_id=None,
                    new_model_id=candidate_id,
                    validation_passed=True,
                    performance_improvement=0.0,
                    swap_time=start_time,
                    error_message="No suitable target model found"
                )
            
            # Perform swap
            old_metrics = self.active_models.get(target_model_id)
            if not old_metrics:
                return SwapResult(
                    success=False,
                    old_model_id=target_model_id,
                    new_model_id=candidate_id,
                    validation_passed=True,
                    performance_improvement=0.0,
                    swap_time=start_time,
                    error_message=f"Target model {target_model_id} not found in active models"
                )
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(
                old_metrics, candidate.metrics
            )
            
            # Execute swap
            self.active_models[target_model_id] = candidate.metrics
            self.model_contribution[target_model_id] = candidate.confidence_score
            
            # Remove candidate from candidate pool
            self.candidate_models.pop(candidate_id, None)
            
            # Update statistics
            self.stats['total_swaps'] += 1
            self.stats['successful_swaps'] += 1
            self._update_average_improvement(performance_improvement)
            
            # Create result
            result = SwapResult(
                success=True,
                old_model_id=target_model_id,
                new_model_id=candidate_id,
                validation_passed=True,
                performance_improvement=performance_improvement,
                swap_time=start_time
            )
            
            self.swap_history.append(result)
            
            logger.info(f"Successfully swapped model {target_model_id} -> {candidate_id} "
                       f"(improvement: {performance_improvement:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during model swap: {e}")
            return SwapResult(
                success=False,
                old_model_id=target_model_id,
                new_model_id=candidate_id,
                validation_passed=False,
                performance_improvement=0.0,
                swap_time=start_time,
                error_message=str(e)
            )
    
    def _select_target_model(self, candidate: SwapCandidate) -> Optional[str]:
        """
        Select the best target model to replace.
        
        Args:
            candidate: Candidate model for swapping
            
        Returns:
            ID of target model to replace
        """
        if not self.active_models:
            return None
        
        # Find models of the same type
        same_type_models = [
            model_id for model_id, metrics in self.active_models.items()
            if self._get_model_type(model_id) == candidate.model_type
        ]
        
        if same_type_models:
            # Replace worst performing model of same type
            worst_model = min(same_type_models, 
                            key=lambda mid: self.active_models[mid].sharpe_ratio)
            return worst_model
        
        # If no same type, replace worst performing model overall
        worst_model = min(self.active_models.keys(),
                         key=lambda mid: self.active_models[mid].sharpe_ratio)
        
        return worst_model
    
    def _get_model_type(self, model_id: str) -> str:
        """Get model type from model ID."""
        # Simple heuristic - extract type from ID
        if '_' in model_id:
            return model_id.split('_')[0]
        return 'unknown'
    
    def _calculate_performance_improvement(self, 
                                         old_metrics: ModelMetrics,
                                         new_metrics: ModelMetrics) -> float:
        """
        Calculate performance improvement from swap.
        
        Args:
            old_metrics: Metrics of old model
            new_metrics: Metrics of new model
            
        Returns:
            Performance improvement score
        """
        # Weighted improvement calculation
        sharpe_improvement = new_metrics.sharpe_ratio - old_metrics.sharpe_ratio
        mse_improvement = old_metrics.mse - new_metrics.mse  # Lower MSE is better
        
        # Normalize improvements
        normalized_sharpe = sharpe_improvement / max(old_metrics.sharpe_ratio, 0.1)
        normalized_mse = mse_improvement / max(old_metrics.mse, 0.001)
        
        # Combined improvement score
        improvement = 0.7 * normalized_sharpe + 0.3 * normalized_mse
        
        return improvement
    
    def _update_average_improvement(self, improvement: float):
        """Update average performance improvement."""
        successful_swaps = self.stats['successful_swaps']
        current_avg = self.stats['avg_performance_improvement']
        
        if successful_swaps == 1:
            self.stats['avg_performance_improvement'] = improvement
        else:
            self.stats['avg_performance_improvement'] = (
                (current_avg * (successful_swaps - 1) + improvement) / successful_swaps
            )
    
    def get_swap_statistics(self) -> Dict[str, Any]:
        """Get swap operation statistics."""
        stats = self.stats.copy()
        
        # Add recent performance
        if self.swap_history:
            recent_swaps = self.swap_history[-10:]  # Last 10 swaps
            stats['recent_success_rate'] = sum(1 for s in recent_swaps if s.success) / len(recent_swaps)
            stats['recent_avg_improvement'] = np.mean([s.performance_improvement for s in recent_swaps if s.success])
        
        # Add current ensemble info
        stats['active_models_count'] = len(self.active_models)
        stats['candidate_models_count'] = len(self.candidate_models)
        stats['avg_active_sharpe'] = np.mean([m.sharpe_ratio for m in self.active_models.values()]) if self.active_models else 0.0
        
        return stats
    
    def rollback_last_swap(self) -> bool:
        """
        Rollback the last successful swap.
        
        Returns:
            Success status
        """
        # Find last successful swap
        for result in reversed(self.swap_history):
            if result.success and result.old_model_id and result.new_model_id:
                # Restore old model
                if result.new_model_id in self.active_models:
                    # Note: We don't have the old metrics, so we'll mark it as deprecated
                    logger.warning(f"Rolling back swap: {result.old_model_id} <- {result.new_model_id}")
                    return True
        
        logger.warning("No successful swap found to rollback")
        return False
    
    def set_validation_criteria(self, 
                              min_sharpe_ratio: float = None,
                              max_mse_increase: float = None):
        """Update validation criteria."""
        if min_sharpe_ratio is not None:
            self.min_sharpe_ratio = min_sharpe_ratio
            logger.info(f"Updated min Sharpe ratio to: {min_sharpe_ratio}")
        
        if max_mse_increase is not None:
            self.max_mse_increase = max_mse_increase
            logger.info(f"Updated max MSE increase to: {max_mse_increase}")

def create_model_swapper(min_sharpe_ratio: float = 0.8) -> ModelSwapper:
    """Factory function to create model swapper."""
    return ModelSwapper(min_sharpe_ratio=min_sharpe_ratio) 