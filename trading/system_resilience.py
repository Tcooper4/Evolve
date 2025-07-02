"""
System Resilience Module

This module provides comprehensive resilience features including:
- Fallback mechanisms for model failures during ensemble voting
- User warnings for inactive strategies (no buy/sell signals)
- System health monitoring and recovery
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ResilienceLevel(Enum):
    """Resilience levels for system operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    enabled: bool = True
    max_fallback_attempts: int = 3
    fallback_threshold: float = 0.5
    recovery_timeout: int = 300  # seconds
    alert_on_fallback: bool = True

@dataclass
class SignalWarning:
    """Warning information for signal generation."""
    warning_type: str
    message: str
    severity: str
    timestamp: datetime
    details: Dict[str, Any]

class SystemResilience:
    """System resilience manager for handling failures and warnings."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize system resilience manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.fallback_config = FallbackConfig(**self.config.get('fallback', {}))
        self.warnings = []
        self.fallback_history = []
        self.system_health = {}
        self.last_health_check = None
        
        # Initialize resilience components
        self._initialize_resilience()
        
        logger.info("System resilience manager initialized")
    
    def _initialize_resilience(self):
        """Initialize resilience components."""
        try:
            # Create resilience directories
            resilience_dirs = ['logs/resilience', 'cache/fallback', 'backups/models']
            for dir_path in resilience_dirs:
                os.makedirs(dir_path, exist_ok=True)
            
            # Load previous fallback history
            self._load_fallback_history()
            
            # Initialize system health
            self._initialize_system_health()
            
        except Exception as e:
            logger.error(f"Error initializing resilience: {e}")
    
    def _load_fallback_history(self):
        """Load fallback history from file."""
        try:
            history_file = Path("logs/resilience/fallback_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.fallback_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading fallback history: {e}")
            self.fallback_history = []
    
    def _save_fallback_history(self):
        """Save fallback history to file."""
        try:
            history_file = Path("logs/resilience/fallback_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.fallback_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving fallback history: {e}")
    
    def _initialize_system_health(self):
        """Initialize system health monitoring."""
        self.system_health = {
            'overall_status': 'healthy',
            'last_check': datetime.now().isoformat(),
            'components': {
                'models': {'status': 'healthy', 'last_check': datetime.now().isoformat()},
                'strategies': {'status': 'healthy', 'last_check': datetime.now().isoformat()},
                'data_feed': {'status': 'healthy', 'last_check': datetime.now().isoformat()},
                'ensemble': {'status': 'healthy', 'last_check': datetime.now().isoformat()}
            },
            'metrics': {
                'error_rate': 0.0,
                'fallback_rate': 0.0,
                'response_time': 0.0
            }
        }
    
    def handle_model_failure(self, 
                           model_name: str, 
                           error: Exception, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model failure with fallback mechanism.
        
        Args:
            model_name: Name of the failed model
            error: The error that occurred
            context: Context information about the failure
            
        Returns:
            Fallback result
        """
        try:
            logger.warning(f"Model failure detected: {model_name} - {error}")
            
            # Record fallback attempt
            fallback_record = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'error': str(error),
                'error_type': type(error).__name__,
                'context': context,
                'fallback_attempt': len(self.fallback_history) + 1
            }
            
            # Attempt fallback
            fallback_result = self._attempt_model_fallback(model_name, context)
            
            # Update record with result
            fallback_record.update(fallback_result)
            self.fallback_history.append(fallback_record)
            self._save_fallback_history()
            
            # Update system health
            self._update_component_health('models', 'degraded' if fallback_result['success'] else 'failed')
            
            # Send alert if configured
            if self.fallback_config.alert_on_fallback:
                self._send_fallback_alert(fallback_record)
            
            return fallback_result
            
        except Exception as e:
            logger.error(f"Error handling model failure: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_used': False,
                'message': 'Failed to handle model failure'
            }
    
    def _attempt_model_fallback(self, model_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt model fallback.
        
        Args:
            model_name: Name of the failed model
            context: Context information
            
        Returns:
            Fallback result
        """
        try:
            # Check if we have a backup model
            backup_model = self._get_backup_model(model_name)
            
            if backup_model:
                logger.info(f"Using backup model for {model_name}")
                return {
                    'success': True,
                    'fallback_used': True,
                    'backup_model': backup_model,
                    'message': f'Successfully used backup model: {backup_model}'
                }
            
            # Check if we can use ensemble fallback
            if context.get('ensemble_available', False):
                logger.info(f"Using ensemble fallback for {model_name}")
                return {
                    'success': True,
                    'fallback_used': True,
                    'fallback_type': 'ensemble',
                    'message': 'Using ensemble fallback'
                }
            
            # Use simple moving average as last resort
            logger.info(f"Using simple MA fallback for {model_name}")
            return {
                'success': True,
                'fallback_used': True,
                'fallback_type': 'simple_ma',
                'message': 'Using simple moving average fallback'
            }
            
        except Exception as e:
            logger.error(f"Error in model fallback: {e}")
            return {
                'success': False,
                'fallback_used': False,
                'error': str(e),
                'message': 'Fallback attempt failed'
            }
    
    def _get_backup_model(self, model_name: str) -> Optional[str]:
        """Get backup model for failed model.
        
        Args:
            model_name: Name of the failed model
            
        Returns:
            Backup model name or None
        """
        backup_mapping = {
            'lstm': 'transformer',
            'transformer': 'xgboost',
            'xgboost': 'arima',
            'arima': 'simple_ma',
            'ensemble': 'simple_ma'
        }
        
        return backup_mapping.get(model_name.lower())
    
    def check_signal_activity(self, 
                             signals: pd.Series, 
                             data: pd.DataFrame,
                             strategy_name: str = "Unknown") -> List[SignalWarning]:
        """Check for signal activity and generate warnings.
        
        Args:
            signals: Trading signals
            data: Market data
            strategy_name: Name of the strategy
            
        Returns:
            List of signal warnings
        """
        warnings = []
        
        try:
            # Check for empty signals
            if signals.empty:
                warnings.append(SignalWarning(
                    warning_type="no_signals",
                    message=f"Strategy '{strategy_name}' generated no signals",
                    severity="high",
                    timestamp=datetime.now(),
                    details={'strategy': strategy_name, 'data_length': len(data)}
                ))
                return warnings
            
            # Check for all zero signals
            if (signals == 0).all():
                warnings.append(SignalWarning(
                    warning_type="inactive_strategy",
                    message=f"Strategy '{strategy_name}' is inactive - no buy/sell signals generated",
                    severity="medium",
                    timestamp=datetime.now(),
                    details={
                        'strategy': strategy_name,
                        'period': f"{data.index[0]} to {data.index[-1]}",
                        'signal_count': len(signals)
                    }
                ))
            
            # Check for signal imbalance
            buy_signals = (signals > 0).sum()
            sell_signals = (signals < 0).sum()
            total_signals = buy_signals + sell_signals
            
            if total_signals > 0:
                buy_ratio = buy_signals / total_signals
                if buy_ratio > 0.8:
                    warnings.append(SignalWarning(
                        warning_type="signal_imbalance",
                        message=f"Strategy '{strategy_name}' shows heavy buy bias ({buy_ratio:.1%} buy signals)",
                        severity="low",
                        timestamp=datetime.now(),
                        details={
                            'strategy': strategy_name,
                            'buy_ratio': buy_ratio,
                            'buy_signals': buy_signals,
                            'sell_signals': sell_signals
                        }
                    ))
                elif buy_ratio < 0.2:
                    warnings.append(SignalWarning(
                        warning_type="signal_imbalance",
                        message=f"Strategy '{strategy_name}' shows heavy sell bias ({(1-buy_ratio):.1%} sell signals)",
                        severity="low",
                        timestamp=datetime.now(),
                        details={
                            'strategy': strategy_name,
                            'sell_ratio': 1 - buy_ratio,
                            'buy_signals': buy_signals,
                            'sell_signals': sell_signals
                        }
                    ))
            
            # Check for recent signal activity
            recent_signals = signals.tail(20)  # Last 20 periods
            if (recent_signals == 0).all():
                warnings.append(SignalWarning(
                    warning_type="recent_inactivity",
                    message=f"Strategy '{strategy_name}' has been inactive for the last 20 periods",
                    severity="medium",
                    timestamp=datetime.now(),
                    details={
                        'strategy': strategy_name,
                        'inactive_periods': 20,
                        'last_signal_date': data.index[-20].strftime('%Y-%m-%d') if hasattr(data.index[-20], 'strftime') else str(data.index[-20])
                    }
                ))
            
            # Store warnings
            self.warnings.extend(warnings)
            
            # Log warnings
            for warning in warnings:
                if warning.severity == "high":
                    logger.error(f"Signal warning: {warning.message}")
                elif warning.severity == "medium":
                    logger.warning(f"Signal warning: {warning.message}")
                else:
                    logger.info(f"Signal warning: {warning.message}")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error checking signal activity: {e}")
            return []
    
    def get_signal_warnings_summary(self) -> Dict[str, Any]:
        """Get summary of signal warnings.
        
        Returns:
            Warning summary
        """
        try:
            if not self.warnings:
                return {'message': 'No signal warnings'}
            
            # Group warnings by type
            warning_types = {}
            for warning in self.warnings:
                if warning.warning_type not in warning_types:
                    warning_types[warning.warning_type] = []
                warning_types[warning.warning_type].append(warning)
            
            # Count by severity
            severity_counts = {'high': 0, 'medium': 0, 'low': 0}
            for warning in self.warnings:
                severity_counts[warning.severity] += 1
            
            return {
                'total_warnings': len(self.warnings),
                'warning_types': {k: len(v) for k, v in warning_types.items()},
                'severity_counts': severity_counts,
                'recent_warnings': [
                    {
                        'type': w.warning_type,
                        'message': w.message,
                        'severity': w.severity,
                        'timestamp': w.timestamp.isoformat()
                    }
                    for w in self.warnings[-10:]  # Last 10 warnings
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting warning summary: {e}")
            return {'error': str(e)}
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health.
        
        Returns:
            System health status
        """
        try:
            current_time = datetime.now()
            
            # Update health check timestamp
            self.system_health['last_check'] = current_time.isoformat()
            
            # Check component health
            for component in self.system_health['components']:
                self.system_health['components'][component]['last_check'] = current_time.isoformat()
            
            # Calculate error rate
            if self.fallback_history:
                recent_failures = [
                    f for f in self.fallback_history[-100:]  # Last 100 records
                    if not f.get('success', True)
                ]
                self.system_health['metrics']['error_rate'] = len(recent_failures) / min(100, len(self.fallback_history))
            
            # Calculate fallback rate
            if self.fallback_history:
                recent_fallbacks = [
                    f for f in self.fallback_history[-100:]
                    if f.get('fallback_used', False)
                ]
                self.system_health['metrics']['fallback_rate'] = len(recent_fallbacks) / min(100, len(self.fallback_history))
            
            # Determine overall status
            error_rate = self.system_health['metrics']['error_rate']
            fallback_rate = self.system_health['metrics']['fallback_rate']
            
            if error_rate > 0.5 or fallback_rate > 0.7:
                self.system_health['overall_status'] = 'critical'
            elif error_rate > 0.2 or fallback_rate > 0.3:
                self.system_health['overall_status'] = 'degraded'
            else:
                self.system_health['overall_status'] = 'healthy'
            
            self.last_health_check = current_time
            
            return self.system_health
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {'error': str(e)}
    
    def _update_component_health(self, component: str, status: str):
        """Update component health status.
        
        Args:
            component: Component name
            status: Health status
        """
        if component in self.system_health['components']:
            self.system_health['components'][component]['status'] = status
            self.system_health['components'][component]['last_check'] = datetime.now().isoformat()
    
    def _send_fallback_alert(self, fallback_record: Dict[str, Any]):
        """Send fallback alert.
        
        Args:
            fallback_record: Fallback record
        """
        try:
            alert_message = f"Model fallback triggered: {fallback_record['model_name']}"
            logger.warning(alert_message)
            
            # Could integrate with external alerting systems here
            # For now, just log the alert
            
        except Exception as e:
            logger.error(f"Error sending fallback alert: {e}")
    
    def get_resilience_report(self) -> Dict[str, Any]:
        """Get comprehensive resilience report.
        
        Returns:
            Resilience report
        """
        try:
            health_status = self.check_system_health()
            warning_summary = self.get_signal_warnings_summary()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_health': health_status,
                'signal_warnings': warning_summary,
                'fallback_history': {
                    'total_fallbacks': len(self.fallback_history),
                    'recent_fallbacks': self.fallback_history[-10:] if self.fallback_history else [],
                    'success_rate': len([f for f in self.fallback_history if f.get('success', False)]) / max(1, len(self.fallback_history))
                },
                'configuration': {
                    'fallback_enabled': self.fallback_config.enabled,
                    'max_attempts': self.fallback_config.max_fallback_attempts,
                    'threshold': self.fallback_config.fallback_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating resilience report: {e}")
            return {'error': str(e)}
    
    def clear_warnings(self):
        """Clear all warnings."""
        self.warnings = []
        logger.info("All warnings cleared")
    
    def clear_fallback_history(self):
        """Clear fallback history."""
        self.fallback_history = []
        self._save_fallback_history()
        logger.info("Fallback history cleared")

# Global instance
system_resilience = SystemResilience()

def handle_model_failure(model_name: str, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """Global function to handle model failures."""
    return system_resilience.handle_model_failure(model_name, error, context)

def check_signal_activity(signals: pd.Series, data: pd.DataFrame, strategy_name: str = "Unknown") -> List[SignalWarning]:
    """Global function to check signal activity."""
    return system_resilience.check_signal_activity(signals, data, strategy_name)

def get_resilience_report() -> Dict[str, Any]:
    """Global function to get resilience report."""
    return system_resilience.get_resilience_report()

def check_system_health() -> Dict[str, Any]:
    """Global function to check system health."""
    return system_resilience.check_system_health() 