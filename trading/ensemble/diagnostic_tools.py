"""
Ensemble Diagnostic Tools - Batch 18
Enhanced diagnostic tools with confidence score filtering
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class DiagnosticLevel(Enum):
    """Diagnostic logging levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    DEBUG = "debug"

@dataclass
class ModelDiagnostic:
    """Diagnostic information for a model."""
    model_name: str
    model_type: str
    confidence_score: float
    performance_metrics: Dict[str, float]
    prediction_quality: float
    last_updated: datetime
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleDiagnostic:
    """Diagnostic information for ensemble."""
    ensemble_id: str
    models: List[ModelDiagnostic]
    ensemble_score: float
    diversity_score: float
    correlation_matrix: Optional[np.ndarray] = None
    generated_at: datetime = field(default_factory=datetime.now)

class EnsembleDiagnosticTools:
    """
    Enhanced ensemble diagnostic tools with confidence filtering.
    
    Features:
    - Confidence score filtering for log reports
    - Multiple diagnostic levels
    - Model performance tracking
    - Ensemble diversity analysis
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.2,
                 enable_debug_mode: bool = False,
                 log_level: DiagnosticLevel = DiagnosticLevel.BASIC):
        """
        Initialize diagnostic tools.
        
        Args:
            confidence_threshold: Minimum confidence score for logging
            enable_debug_mode: Enable debug mode for detailed logging
            log_level: Diagnostic logging level
        """
        self.confidence_threshold = confidence_threshold
        self.enable_debug_mode = enable_debug_mode
        self.log_level = log_level
        
        # Diagnostic history
        self.diagnostic_history: List[EnsembleDiagnostic] = []
        self.model_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"EnsembleDiagnosticTools initialized with confidence threshold: {confidence_threshold}")
        logger.info(f"Debug mode: {enable_debug_mode}, Log level: {log_level.value}")
    
    def analyze_ensemble(self, 
                        ensemble_data: Dict[str, Any],
                        include_low_confidence: bool = None) -> EnsembleDiagnostic:
        """
        Analyze ensemble performance with confidence filtering.
        
        Args:
            ensemble_data: Ensemble data containing models and predictions
            include_low_confidence: Override debug mode for this analysis
            
        Returns:
            EnsembleDiagnostic object
        """
        if include_low_confidence is None:
            include_low_confidence = self.enable_debug_mode
        
        models = ensemble_data.get('models', [])
        ensemble_id = ensemble_data.get('ensemble_id', f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Filter models based on confidence threshold
        filtered_models = []
        low_confidence_models = []
        
        for model_data in models:
            confidence_score = model_data.get('confidence_score', 0.0)
            
            if confidence_score >= self.confidence_threshold or include_low_confidence:
                model_diagnostic = self._create_model_diagnostic(model_data)
                filtered_models.append(model_diagnostic)
            else:
                low_confidence_models.append(model_data.get('model_name', 'unknown'))
        
        # Log warning about filtered models
        if low_confidence_models and not include_low_confidence:
            logger.warning(
                f"Filtered out {len(low_confidence_models)} models with confidence < {self.confidence_threshold}: "
                f"{low_confidence_models}"
            )
        
        # Calculate ensemble metrics
        ensemble_score = self._calculate_ensemble_score(filtered_models)
        diversity_score = self._calculate_diversity_score(filtered_models)
        correlation_matrix = self._calculate_correlation_matrix(filtered_models)
        
        diagnostic = EnsembleDiagnostic(
            ensemble_id=ensemble_id,
            models=filtered_models,
            ensemble_score=ensemble_score,
            diversity_score=diversity_score,
            correlation_matrix=correlation_matrix
        )
        
        # Store in history
        self.diagnostic_history.append(diagnostic)
        
        # Update model performance history
        self._update_model_performance_history(filtered_models)
        
        return diagnostic
    
    def _create_model_diagnostic(self, model_data: Dict[str, Any]) -> ModelDiagnostic:
        """
        Create ModelDiagnostic from model data.
        
        Args:
            model_data: Raw model data
            
        Returns:
            ModelDiagnostic object
        """
        return ModelDiagnostic(
            model_name=model_data.get('model_name', 'unknown'),
            model_type=model_data.get('model_type', 'unknown'),
            confidence_score=model_data.get('confidence_score', 0.0),
            performance_metrics=model_data.get('performance_metrics', {}),
            prediction_quality=model_data.get('prediction_quality', 0.0),
            last_updated=datetime.now(),
            status=model_data.get('status', 'active'),
            metadata=model_data.get('metadata', {})
        )
    
    def _calculate_ensemble_score(self, models: List[ModelDiagnostic]) -> float:
        """
        Calculate ensemble score.
        
        Args:
            models: List of model diagnostics
            
        Returns:
            Ensemble score
        """
        if not models:
            return 0.0
        
        # Weighted average based on confidence scores
        total_weight = 0.0
        weighted_sum = 0.0
        
        for model in models:
            weight = model.confidence_score
            score = model.prediction_quality
            
            total_weight += weight
            weighted_sum += weight * score
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_diversity_score(self, models: List[ModelDiagnostic]) -> float:
        """
        Calculate ensemble diversity score.
        
        Args:
            models: List of model diagnostics
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(models) < 2:
            return 0.0
        
        # Calculate diversity based on model types and performance
        model_types = [model.model_type for model in models]
        unique_types = len(set(model_types))
        
        # Type diversity (0-1)
        type_diversity = unique_types / len(models)
        
        # Performance diversity (standard deviation of prediction quality)
        qualities = [model.prediction_quality for model in models]
        performance_diversity = np.std(qualities) if len(qualities) > 1 else 0.0
        
        # Combined diversity score
        diversity_score = 0.7 * type_diversity + 0.3 * min(performance_diversity, 1.0)
        
        return min(1.0, diversity_score)
    
    def _calculate_correlation_matrix(self, models: List[ModelDiagnostic]) -> Optional[np.ndarray]:
        """
        Calculate correlation matrix between models.
        
        Args:
            models: List of model diagnostics
            
        Returns:
            Correlation matrix or None if insufficient data
        """
        if len(models) < 2:
            return None
        
        try:
            # Extract predictions from metadata if available
            predictions = []
            for model in models:
                if 'predictions' in model.metadata:
                    pred = model.metadata['predictions']
                    if isinstance(pred, (list, np.ndarray)):
                        predictions.append(pred)
            
            if len(predictions) < 2:
                return None
            
            # Convert to numpy array and calculate correlation
            pred_array = np.array(predictions)
            correlation_matrix = np.corrcoef(pred_array)
            
            return correlation_matrix
            
        except Exception as e:
            logger.warning(f"Failed to calculate correlation matrix: {e}")
            return None
    
    def _update_model_performance_history(self, models: List[ModelDiagnostic]):
        """Update model performance history."""
        for model in models:
            model_name = model.model_name
            
            if model_name not in self.model_performance_history:
                self.model_performance_history[model_name] = []
            
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'confidence_score': model.confidence_score,
                'prediction_quality': model.prediction_quality,
                'ensemble_score': model.performance_metrics.get('ensemble_score', 0.0),
                'status': model.status
            }
            
            self.model_performance_history[model_name].append(history_entry)
            
            # Keep only recent history
            if len(self.model_performance_history[model_name]) > 100:
                self.model_performance_history[model_name] = self.model_performance_history[model_name][-100:]
    
    def generate_diagnostic_report(self, 
                                 diagnostic: EnsembleDiagnostic,
                                 include_low_confidence: bool = None) -> Dict[str, Any]:
        """
        Generate diagnostic report with confidence filtering.
        
        Args:
            diagnostic: EnsembleDiagnostic object
            include_low_confidence: Override debug mode for this report
            
        Returns:
            Diagnostic report dictionary
        """
        if include_low_confidence is None:
            include_low_confidence = self.enable_debug_mode
        
        # Filter models based on confidence
        if include_low_confidence:
            report_models = diagnostic.models
        else:
            report_models = [
                model for model in diagnostic.models
                if model.confidence_score >= self.confidence_threshold
            ]
        
        # Generate report
        report = {
            'ensemble_id': diagnostic.ensemble_id,
            'generated_at': diagnostic.generated_at.isoformat(),
            'ensemble_score': diagnostic.ensemble_score,
            'diversity_score': diagnostic.diversity_score,
            'total_models': len(diagnostic.models),
            'reported_models': len(report_models),
            'filtered_models': len(diagnostic.models) - len(report_models),
            'confidence_threshold': self.confidence_threshold,
            'models': []
        }
        
        # Add model details
        for model in report_models:
            model_info = {
                'name': model.model_name,
                'type': model.model_type,
                'confidence_score': model.confidence_score,
                'prediction_quality': model.prediction_quality,
                'status': model.status,
                'performance_metrics': model.performance_metrics
            }
            report['models'].append(model_info)
        
        # Add correlation matrix if available
        if diagnostic.correlation_matrix is not None:
            report['correlation_matrix'] = diagnostic.correlation_matrix.tolist()
        
        return report
    
    def log_diagnostic_summary(self, 
                             diagnostic: EnsembleDiagnostic,
                             include_low_confidence: bool = None):
        """
        Log diagnostic summary with confidence filtering.
        
        Args:
            diagnostic: EnsembleDiagnostic object
            include_low_confidence: Override debug mode for logging
        """
        if include_low_confidence is None:
            include_low_confidence = self.enable_debug_mode
        
        # Filter models for logging
        if include_low_confidence:
            log_models = diagnostic.models
        else:
            log_models = [
                model for model in diagnostic.models
                if model.confidence_score >= self.confidence_threshold
            ]
        
        # Log summary
        logger.info(f"Ensemble Diagnostic Summary - {diagnostic.ensemble_id}")
        logger.info(f"  Ensemble Score: {diagnostic.ensemble_score:.4f}")
        logger.info(f"  Diversity Score: {diagnostic.diversity_score:.4f}")
        logger.info(f"  Total Models: {len(diagnostic.models)}")
        logger.info(f"  Reported Models: {len(log_models)}")
        
        if len(diagnostic.models) > len(log_models):
            logger.info(f"  Filtered Models: {len(diagnostic.models) - len(log_models)} (confidence < {self.confidence_threshold})")
        
        # Log model details based on log level
        if self.log_level in [DiagnosticLevel.DETAILED, DiagnosticLevel.DEBUG]:
            for model in log_models:
                logger.info(f"    {model.model_name} ({model.model_type}): "
                          f"confidence={model.confidence_score:.3f}, "
                          f"quality={model.prediction_quality:.3f}")
        
        # Log low confidence models in debug mode
        if self.log_level == DiagnosticLevel.DEBUG and not include_low_confidence:
            low_confidence_models = [
                model for model in diagnostic.models
                if model.confidence_score < self.confidence_threshold
            ]
            if low_confidence_models:
                logger.debug("Low confidence models (filtered):")
                for model in low_confidence_models:
                    logger.debug(f"    {model.model_name}: confidence={model.confidence_score:.3f}")
    
    def get_model_performance_trends(self, model_name: str, window: int = 10) -> Dict[str, Any]:
        """
        Get performance trends for a specific model.
        
        Args:
            model_name: Name of the model
            window: Number of recent entries to analyze
            
        Returns:
            Performance trends dictionary
        """
        if model_name not in self.model_performance_history:
            return {}
        
        history = self.model_performance_history[model_name][-window:]
        
        if not history:
            return {}
        
        confidence_scores = [entry['confidence_score'] for entry in history]
        prediction_qualities = [entry['prediction_quality'] for entry in history]
        
        trends = {
            'model_name': model_name,
            'data_points': len(history),
            'avg_confidence': np.mean(confidence_scores),
            'avg_quality': np.mean(prediction_qualities),
            'confidence_trend': np.polyfit(range(len(confidence_scores)), confidence_scores, 1)[0],
            'quality_trend': np.polyfit(range(len(prediction_qualities)), prediction_qualities, 1)[0],
            'confidence_std': np.std(confidence_scores),
            'quality_std': np.std(prediction_qualities)
        }
        
        return trends
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to: {threshold}")
    
    def enable_debug_mode(self, enable: bool = True):
        """Enable or disable debug mode."""
        self.enable_debug_mode = enable
        logger.info(f"Debug mode {'enabled' if enable else 'disabled'}")
    
    def set_log_level(self, level: DiagnosticLevel):
        """Set diagnostic log level."""
        self.log_level = level
        logger.info(f"Diagnostic log level set to: {level.value}")

def create_ensemble_diagnostic_tools(confidence_threshold: float = 0.2) -> EnsembleDiagnosticTools:
    """Factory function to create ensemble diagnostic tools."""
    return EnsembleDiagnosticTools(confidence_threshold=confidence_threshold) 