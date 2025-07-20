"""
Ensemble Integrity Checker - Batch 20
Distribution drift detection and ensemble health monitoring
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of distribution drift."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class DistanceMetric(Enum):
    """Distance metrics for drift detection."""
    KL_DIVERGENCE = "kl_divergence"
    WASSERSTEIN = "wasserstein"
    JENSEN_SHANNON = "jensen_shannon"
    EARTH_MOVERS = "earth_movers"

@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_type: DriftType
    distance_metric: DistanceMetric
    distance_value: float
    threshold: float
    confidence: float
    detected_at: datetime
    model_id: str
    feature_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleHealth:
    """Ensemble health status."""
    overall_health: float
    drift_alerts: List[DriftResult]
    model_stability: Dict[str, float]
    ensemble_coherence: float
    last_check: datetime
    recommendations: List[str] = field(default_factory=list)

class EnsembleIntegrityChecker:
    """
    Enhanced ensemble integrity checker with distribution drift detection.
    
    Features:
    - Distribution drift detection using KL divergence and Wasserstein distance
    - Rolling reference distribution updates
    - Multi-metric drift analysis
    - Ensemble health monitoring
    - Drift alert management
    """
    
    def __init__(self, 
                 reference_window: int = 1000,
                 drift_thresholds: Optional[Dict[str, float]] = None,
                 enable_visualization: bool = False,
                 update_frequency: int = 100):
        """
        Initialize ensemble integrity checker.
        
        Args:
            reference_window: Number of samples for reference distribution
            drift_thresholds: Custom drift thresholds
            enable_visualization: Enable drift visualization
            update_frequency: Frequency of reference distribution updates
        """
        self.reference_window = reference_window
        self.enable_visualization = enable_visualization
        self.update_frequency = update_frequency
        
        # Default drift thresholds
        self.drift_thresholds = drift_thresholds or {
            'mild': 0.1,
            'moderate': 0.25,
            'severe': 0.5,
            'critical': 1.0
        }
        
        # Reference distributions
        self.reference_distributions: Dict[str, Dict[str, np.ndarray]] = {}
        self.current_distributions: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Drift history
        self.drift_history: List[DriftResult] = []
        self.health_history: List[EnsembleHealth] = []
        
        # Model tracking
        self.model_samples: Dict[str, List[np.ndarray]] = {}
        self.sample_count = 0
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'drift_detections': 0,
            'severe_drifts': 0,
            'critical_drifts': 0,
            'avg_drift_score': 0.0
        }
        
        logger.info(f"EnsembleIntegrityChecker initialized with reference window: {reference_window}")
    
    def add_model_predictions(self, 
                            model_id: str,
                            predictions: np.ndarray,
                            features: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        Add model predictions for drift detection.
        
        Args:
            model_id: Model identifier
            predictions: Model predictions
            features: Optional feature values for multi-dimensional drift
        """
        if model_id not in self.model_samples:
            self.model_samples[model_id] = []
        
        # Store predictions
        self.model_samples[model_id].append(predictions)
        
        # Limit sample history
        if len(self.model_samples[model_id]) > self.reference_window * 2:
            self.model_samples[model_id] = self.model_samples[model_id][-self.reference_window * 2:]
        
        self.sample_count += 1
        
        # Update reference distributions periodically
        if self.sample_count % self.update_frequency == 0:
            self._update_reference_distributions()
    
    def distribution_drift_check(self, 
                               model_id: str,
                               current_predictions: np.ndarray,
                               distance_metric: DistanceMetric = DistanceMetric.KL_DIVERGENCE,
                               features: Optional[Dict[str, np.ndarray]] = None) -> DriftResult:
        """
        Check for distribution drift in model predictions.
        
        Args:
            model_id: Model identifier
            current_predictions: Current model predictions
            distance_metric: Distance metric to use
            features: Optional feature values for multi-dimensional drift
            
        Returns:
            DriftResult with drift analysis
        """
        try:
            # Get reference distribution
            reference_dist = self.reference_distributions.get(model_id, {})
            if not reference_dist:
                logger.warning(f"No reference distribution found for model {model_id}")
                return self._create_no_drift_result(model_id, distance_metric)
            
            # Calculate distance for predictions
            pred_distance = self._calculate_distribution_distance(
                reference_dist.get('predictions', np.array([])),
                current_predictions,
                distance_metric
            )
            
            # Calculate distances for features if provided
            feature_distances = {}
            if features and 'features' in reference_dist:
                for feature_name, feature_values in features.items():
                    ref_feature = reference_dist['features'].get(feature_name, np.array([]))
                    if len(ref_feature) > 0 and len(feature_values) > 0:
                        feature_dist = self._calculate_distribution_distance(
                            ref_feature, feature_values, distance_metric
                        )
                        feature_distances[feature_name] = feature_dist
            
            # Determine overall drift
            max_distance = pred_distance
            if feature_distances:
                max_distance = max(pred_distance, max(feature_distances.values()))
            
            # Determine drift type
            drift_type = self._classify_drift(max_distance)
            
            # Calculate confidence
            confidence = self._calculate_drift_confidence(max_distance, len(current_predictions))
            
            # Create result
            result = DriftResult(
                drift_type=drift_type,
                distance_metric=distance_metric,
                distance_value=max_distance,
                threshold=self._get_threshold_for_drift(drift_type),
                confidence=confidence,
                detected_at=datetime.now(),
                model_id=model_id,
                metadata={'feature_distances': feature_distances}
            )
            
            # Store result
            self.drift_history.append(result)
            
            # Update statistics
            self._update_drift_stats(result)
            
            logger.info(f"Drift check for {model_id}: {drift_type.value} (distance: {max_distance:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in drift check for {model_id}: {e}")
            return self._create_error_result(model_id, distance_metric, str(e))
    
    def _calculate_distribution_distance(self, 
                                       reference: np.ndarray,
                                       current: np.ndarray,
                                       distance_metric: DistanceMetric) -> float:
        """
        Calculate distance between reference and current distributions.
        
        Args:
            reference: Reference distribution
            current: Current distribution
            distance_metric: Distance metric to use
            
        Returns:
            Distance value
        """
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        
        try:
            if distance_metric == DistanceMetric.KL_DIVERGENCE:
                return self._kl_divergence(reference, current)
            elif distance_metric == DistanceMetric.WASSERSTEIN:
                return wasserstein_distance(reference, current)
            elif distance_metric == DistanceMetric.JENSEN_SHANNON:
                return self._jensen_shannon_distance(reference, current)
            elif distance_metric == DistanceMetric.EARTH_MOVERS:
                return wasserstein_distance(reference, current)  # Same as Wasserstein
            else:
                return self._kl_divergence(reference, current)  # Default
                
        except Exception as e:
            logger.warning(f"Error calculating {distance_metric.value} distance: {e}")
            return 0.0
    
    def _kl_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate KL divergence between distributions."""
        try:
            # Create histograms
            min_val = min(np.min(reference), np.min(current))
            max_val = max(np.max(reference), np.max(current))
            
            if min_val == max_val:
                return 0.0
            
            bins = np.linspace(min_val, max_val, 50)
            ref_hist, _ = np.histogram(reference, bins=bins, density=True)
            cur_hist, _ = np.histogram(current, bins=bins, density=True)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            cur_hist = cur_hist + epsilon
            
            # Calculate KL divergence
            kl_div = np.sum(ref_hist * np.log(ref_hist / cur_hist))
            
            return max(0.0, kl_div)  # KL divergence should be non-negative
            
        except Exception as e:
            logger.warning(f"Error calculating KL divergence: {e}")
            return 0.0
    
    def _jensen_shannon_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Jensen-Shannon distance between distributions."""
        try:
            # Create histograms
            min_val = min(np.min(reference), np.min(current))
            max_val = max(np.max(reference), np.max(current))
            
            if min_val == max_val:
                return 0.0
            
            bins = np.linspace(min_val, max_val, 50)
            ref_hist, _ = np.histogram(reference, bins=bins, density=True)
            cur_hist, _ = np.histogram(current, bins=bins, density=True)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            cur_hist = cur_hist + epsilon
            
            # Calculate Jensen-Shannon distance
            m = 0.5 * (ref_hist + cur_hist)
            js_distance = 0.5 * (
                np.sum(ref_hist * np.log(ref_hist / m)) +
                np.sum(cur_hist * np.log(cur_hist / m))
            )
            
            return max(0.0, js_distance)
            
        except Exception as e:
            logger.warning(f"Error calculating Jensen-Shannon distance: {e}")
            return 0.0
    
    def _classify_drift(self, distance: float) -> DriftType:
        """Classify drift based on distance value."""
        if distance < self.drift_thresholds['mild']:
            return DriftType.NONE
        elif distance < self.drift_thresholds['moderate']:
            return DriftType.MILD
        elif distance < self.drift_thresholds['severe']:
            return DriftType.MODERATE
        elif distance < self.drift_thresholds['critical']:
            return DriftType.SEVERE
        else:
            return DriftType.CRITICAL
    
    def _get_threshold_for_drift(self, drift_type: DriftType) -> float:
        """Get threshold value for drift type."""
        if drift_type == DriftType.NONE:
            return self.drift_thresholds['mild']
        elif drift_type == DriftType.MILD:
            return self.drift_thresholds['moderate']
        elif drift_type == DriftType.MODERATE:
            return self.drift_thresholds['severe']
        elif drift_type == DriftType.SEVERE:
            return self.drift_thresholds['critical']
        else:
            return float('inf')
    
    def _calculate_drift_confidence(self, distance: float, sample_size: int) -> float:
        """Calculate confidence in drift detection."""
        # Base confidence on distance magnitude
        base_confidence = min(distance / self.drift_thresholds['critical'], 1.0)
        
        # Adjust for sample size
        sample_factor = min(sample_size / 100, 1.0)
        
        return base_confidence * sample_factor
    
    def _update_reference_distributions(self):
        """Update reference distributions from recent samples."""
        for model_id, samples in self.model_samples.items():
            if len(samples) < self.reference_window:
                continue
            
            # Use recent samples for reference
            recent_samples = samples[-self.reference_window:]
            
            # Combine predictions
            all_predictions = np.concatenate(recent_samples)
            
            # Store reference distribution
            self.reference_distributions[model_id] = {
                'predictions': all_predictions,
                'features': {},  # Will be populated if features are provided
                'updated_at': datetime.now()
            }
        
        logger.debug(f"Updated reference distributions for {len(self.reference_distributions)} models")
    
    def check_ensemble_health(self) -> EnsembleHealth:
        """
        Check overall ensemble health.
        
        Returns:
            EnsembleHealth status
        """
        # Get recent drift results
        recent_drifts = [d for d in self.drift_history 
                        if (datetime.now() - d.detected_at) < timedelta(hours=24)]
        
        # Calculate overall health score
        health_score = 1.0
        drift_alerts = []
        
        for drift in recent_drifts:
            if drift.drift_type in [DriftType.SEVERE, DriftType.CRITICAL]:
                health_score -= 0.2
                drift_alerts.append(drift)
            elif drift.drift_type == DriftType.MODERATE:
                health_score -= 0.1
                drift_alerts.append(drift)
            elif drift.drift_type == DriftType.MILD:
                health_score -= 0.05
        
        health_score = max(0.0, health_score)
        
        # Calculate model stability
        model_stability = {}
        for model_id in self.model_samples.keys():
            model_drifts = [d for d in recent_drifts if d.model_id == model_id]
            if model_drifts:
                avg_drift = np.mean([d.distance_value for d in model_drifts])
                stability = max(0.0, 1.0 - avg_drift)
            else:
                stability = 1.0
            model_stability[model_id] = stability
        
        # Calculate ensemble coherence
        if len(self.model_samples) > 1:
            predictions_list = []
            for samples in self.model_samples.values():
                if samples:
                    predictions_list.append(samples[-1])  # Most recent predictions
            
            if len(predictions_list) > 1:
                # Calculate correlation between model predictions
                correlations = []
                for i in range(len(predictions_list)):
                    for j in range(i + 1, len(predictions_list)):
                        if len(predictions_list[i]) == len(predictions_list[j]):
                            corr = np.corrcoef(predictions_list[i], predictions_list[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                
                ensemble_coherence = np.mean(correlations) if correlations else 0.0
            else:
                ensemble_coherence = 1.0
        else:
            ensemble_coherence = 1.0
        
        # Generate recommendations
        recommendations = []
        if health_score < 0.7:
            recommendations.append("Consider retraining models with recent data")
        if len(drift_alerts) > 3:
            recommendations.append("High drift frequency detected - review model stability")
        if ensemble_coherence < 0.5:
            recommendations.append("Low ensemble coherence - check model diversity")
        
        # Create health status
        health = EnsembleHealth(
            overall_health=health_score,
            drift_alerts=drift_alerts,
            model_stability=model_stability,
            ensemble_coherence=ensemble_coherence,
            last_check=datetime.now(),
            recommendations=recommendations
        )
        
        self.health_history.append(health)
        
        return health
    
    def _create_no_drift_result(self, model_id: str, distance_metric: DistanceMetric) -> DriftResult:
        """Create result for no drift detected."""
        return DriftResult(
            drift_type=DriftType.NONE,
            distance_metric=distance_metric,
            distance_value=0.0,
            threshold=self.drift_thresholds['mild'],
            confidence=1.0,
            detected_at=datetime.now(),
            model_id=model_id
        )
    
    def _create_error_result(self, model_id: str, distance_metric: DistanceMetric, error: str) -> DriftResult:
        """Create result for error in drift detection."""
        return DriftResult(
            drift_type=DriftType.NONE,
            distance_metric=distance_metric,
            distance_value=0.0,
            threshold=0.0,
            confidence=0.0,
            detected_at=datetime.now(),
            model_id=model_id,
            metadata={'error': error}
        )
    
    def _update_drift_stats(self, result: DriftResult):
        """Update drift statistics."""
        self.stats['total_checks'] += 1
        
        if result.drift_type != DriftType.NONE:
            self.stats['drift_detections'] += 1
            
            if result.drift_type == DriftType.SEVERE:
                self.stats['severe_drifts'] += 1
            elif result.drift_type == DriftType.CRITICAL:
                self.stats['critical_drifts'] += 1
        
        # Update average drift score
        if self.stats['total_checks'] > 0:
            total_score = self.stats['avg_drift_score'] * (self.stats['total_checks'] - 1) + result.distance_value
            self.stats['avg_drift_score'] = total_score / self.stats['total_checks']
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get drift detection statistics."""
        stats = self.stats.copy()
        
        # Add recent health metrics
        if self.health_history:
            recent_health = self.health_history[-1]
            stats['current_health_score'] = recent_health.overall_health
            stats['current_ensemble_coherence'] = recent_health.ensemble_coherence
            stats['active_drift_alerts'] = len(recent_health.drift_alerts)
        
        # Add drift type distribution
        drift_types = [d.drift_type.value for d in self.drift_history[-100:]]  # Last 100 drifts
        stats['drift_type_distribution'] = {
            drift_type: drift_types.count(drift_type) 
            for drift_type in set(drift_types)
        }
        
        return stats
    
    def set_drift_thresholds(self, thresholds: Dict[str, float]):
        """Update drift thresholds."""
        self.drift_thresholds.update(thresholds)
        logger.info(f"Updated drift thresholds: {thresholds}")

def create_ensemble_integrity_checker(reference_window: int = 1000) -> EnsembleIntegrityChecker:
    """Factory function to create ensemble integrity checker."""
    return EnsembleIntegrityChecker(reference_window=reference_window)
