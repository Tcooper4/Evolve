"""
Long-Term Performance Tracker

Tracks and analyzes system performance over extended periods.
Provides insights into performance trends and degradation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    timestamp: datetime
    metric_name: str
    value: float
    context: Dict[str, Any]
    metadata: Dict[str, Any] = None

@dataclass
class PerformanceTrend:
    """Represents a performance trend."""
    metric_name: str
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0.0 to 1.0
    period_days: int
    average_value: float
    volatility: float

class LongTermPerformanceTracker:
    """Tracks long-term performance metrics and trends."""
    
    def __init__(self, retention_days: int = 365):
        """
        Initialize the performance tracker.
        
        Args:
            retention_days: Number of days to retain performance data
        """
        self.retention_days = retention_days
        self.metrics: List[PerformanceMetric] = []
        self.trends: Dict[str, PerformanceTrend] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info("Long-term performance tracker initialized")
    
    def record_metric(self, metric_name: str, value: float, 
                     context: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Context information
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            context=context or {},
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Clean old metrics
        self._clean_old_metrics()
        
        # Check for alerts
        self._check_alerts(metric)
        
        logger.info(f"Recorded metric: {metric_name} = {value}")
    
    def _clean_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_date]
    
    def _check_alerts(self, metric: PerformanceMetric) -> None:
        """Check for performance alerts."""
        # Get recent metrics for this metric type
        recent_metrics = [
            m for m in self.metrics[-100:]  # Last 100 metrics
            if m.metric_name == metric.metric_name
        ]
        
        if len(recent_metrics) < 10:
            return
        
        # Calculate statistics
        values = [m.value for m in recent_metrics]
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        # Check for significant deviations
        if abs(metric.value - mean_value) > 2 * std_value:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'metric_name': metric.metric_name,
                'value': metric.value,
                'expected_range': f"{mean_value - 2*std_value:.2f} to {mean_value + 2*std_value:.2f}",
                'severity': 'high' if abs(metric.value - mean_value) > 3 * std_value else 'medium',
                'context': metric.context
            }
            
            self.alerts.append(alert)
            logger.warning(f"Performance alert: {alert}")
    
    def analyze_trends(self, metric_name: Optional[str] = None, 
                      period_days: int = 30) -> Dict[str, Any]:
        """
        Analyze performance trends.
        
        Args:
            metric_name: Specific metric to analyze (None for all)
            period_days: Analysis period in days
            
        Returns:
            Dictionary with trend analysis
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Filter metrics by period and name
        recent_metrics = [
            m for m in self.metrics
            if m.timestamp > cutoff_date and (metric_name is None or m.metric_name == metric_name)
        ]
        
        if not recent_metrics:
            return {'message': 'No metrics available for analysis'}
        
        # Group by metric name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.metric_name].append(metric)
        
        trends = {}
        for name, metrics in metrics_by_name.items():
            if len(metrics) < 5:  # Need at least 5 data points
                continue
            
            # Calculate trend
            values = [m.value for m in metrics]
            timestamps = [m.timestamp for m in metrics]
            
            # Simple linear trend
            x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
            y = np.array(values)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trend_strength = abs(slope) / (np.std(y) + 1e-8)
                
                # Determine trend direction
                if slope > 0.01:
                    direction = 'improving'
                elif slope < -0.01:
                    direction = 'declining'
                else:
                    direction = 'stable'
                
                trend = PerformanceTrend(
                    metric_name=name,
                    trend_direction=direction,
                    trend_strength=min(1.0, trend_strength),
                    period_days=period_days,
                    average_value=np.mean(values),
                    volatility=np.std(values)
                )
                
                trends[name] = trend
                self.trends[name] = trend
        
        return {
            'period_days': period_days,
            'total_metrics': len(recent_metrics),
            'trends': {name: asdict(trend) for name, trend in trends.items()}
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics:
            return {'message': 'No performance data available'}
        
        # Overall statistics
        all_values = [m.value for m in self.metrics]
        
        # Recent vs historical comparison
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_date]
        historical_metrics = [m for m in self.metrics if m.timestamp <= cutoff_date]
        
        recent_values = [m.value for m in recent_metrics] if recent_metrics else []
        historical_values = [m.value for m in historical_metrics] if historical_metrics else []
        
        summary = {
            'total_metrics': len(self.metrics),
            'metrics_by_name': self._get_metrics_by_name(),
            'overall_stats': {
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'min': np.min(all_values),
                'max': np.max(all_values)
            },
            'recent_vs_historical': {
                'recent_mean': np.mean(recent_values) if recent_values else 0,
                'historical_mean': np.mean(historical_values) if historical_values else 0,
                'change_percent': self._calculate_change_percent(recent_values, historical_values)
            },
            'active_alerts': len([a for a in self.alerts if a['severity'] == 'high']),
            'trends': {name: asdict(trend) for name, trend in self.trends.items()}
        }
        
        return summary
    
    def _get_metrics_by_name(self) -> Dict[str, Dict[str, float]]:
        """Get statistics by metric name."""
        metrics_by_name = defaultdict(list)
        for metric in self.metrics:
            metrics_by_name[metric.metric_name].append(metric.value)
        
        stats_by_name = {}
        for name, values in metrics_by_name.items():
            stats_by_name[name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return stats_by_name
    
    def _calculate_change_percent(self, recent_values: List[float], 
                                historical_values: List[float]) -> float:
        """Calculate percentage change between recent and historical values."""
        if not recent_values or not historical_values:
            return 0.0
        
        recent_mean = np.mean(recent_values)
        historical_mean = np.mean(historical_values)
        
        if historical_mean == 0:
            return 0.0
        
        return ((recent_mean - historical_mean) / historical_mean) * 100
    
    def get_performance_forecast(self, metric_name: str, 
                               forecast_days: int = 7) -> Dict[str, Any]:
        """
        Generate performance forecast.
        
        Args:
            metric_name: Metric to forecast
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Get recent metrics for the specified metric
        recent_metrics = [
            m for m in self.metrics[-50:]  # Last 50 metrics
            if m.metric_name == metric_name
        ]
        
        if len(recent_metrics) < 10:
            return {'error': 'Insufficient data for forecasting'}
        
        # Simple linear forecast
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        y = np.array(values)
        
        # Fit linear model
        coeffs = np.polyfit(x, y, 1)
        
        # Generate forecast
        last_timestamp = timestamps[-1]
        forecast_values = []
        forecast_dates = []
        
        for i in range(1, forecast_days + 1):
            future_timestamp = last_timestamp + timedelta(days=i)
            future_seconds = (future_timestamp - timestamps[0]).total_seconds()
            forecast_value = coeffs[0] * future_seconds + coeffs[1]
            
            forecast_values.append(forecast_value)
            forecast_dates.append(future_timestamp.isoformat())
        
        return {
            'metric_name': metric_name,
            'forecast_days': forecast_days,
            'forecast_values': forecast_values,
            'forecast_dates': forecast_dates,
            'trend_slope': coeffs[0],
            'confidence': self._calculate_forecast_confidence(values)
        }
    
    def _calculate_forecast_confidence(self, values: List[float]) -> float:
        """Calculate forecast confidence based on data stability."""
        if len(values) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if mean_value == 0:
            return 0.0
        
        cv = std_value / abs(mean_value)
        
        # Convert to confidence (lower CV = higher confidence)
        confidence = max(0.0, min(1.0, 1.0 - cv))
        
        return confidence
    
    def run(self) -> Dict[str, Any]:
        """
        Run the performance tracker.
        
        Returns:
            Dictionary with tracker results
        """
        try:
            # Analyze trends
            trend_analysis = self.analyze_trends()
            
            # Get summary
            summary = self.get_performance_summary()
            
            # Get recent alerts
            recent_alerts = [
                alert for alert in self.alerts[-10:]  # Last 10 alerts
                if alert['severity'] == 'high'
            ]
            
            return {
                'success': True,
                'trend_analysis': trend_analysis,
                'summary': summary,
                'recent_alerts': recent_alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in performance tracker: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            } 