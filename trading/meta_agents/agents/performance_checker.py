"""
Performance Checker for Meta-Agent Loop

This module provides performance monitoring and improvement suggestions
for strategies and models in the Evolve trading system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PerformanceChecker:
    """Performance checker for continuous improvement of strategies and models."""
    
    def __init__(self):
        """Initialize the performance checker."""
        self.performance_history = {}
        self.improvement_suggestions = {}
        self.retirement_thresholds = {
            'sharpe_ratio': 0.3,      # Minimum Sharpe ratio
            'win_rate': 0.4,          # Minimum win rate
            'max_drawdown': 0.25,     # Maximum drawdown
            'total_return': 0.05,     # Minimum total return
            'consecutive_losses': 10   # Maximum consecutive losses
        }
        self.tuning_thresholds = {
            'sharpe_decay': 0.2,      # Sharpe ratio decay threshold
            'accuracy_drop': 0.1,     # Accuracy drop threshold
            'performance_trend': -0.1  # Performance trend threshold
        }
        
        logger.info("Performance Checker initialized")
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def check_strategy_performance(self, strategy_name: str, 
                                 performance: Dict[str, float]) -> Dict[str, Any]:
        """Check strategy performance and determine if action is needed.
        
        Args:
            strategy_name: Name of the strategy
            performance: Current performance metrics
            
        Returns:
            Dictionary with recommendations
        """
        try:
            # Store performance in history
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []
            
            performance_record = {
                'timestamp': datetime.now(),
                'performance': performance.copy()
            }
            self.performance_history[strategy_name].append(performance_record)
            
            # Keep only last 50 records
            if len(self.performance_history[strategy_name]) > 50:
                self.performance_history[strategy_name] = self.performance_history[strategy_name][-50:]
            
            # Check retirement conditions
            should_retire = self._check_retirement_conditions(strategy_name, performance)
            
            # Check tuning conditions
            should_tune = self._check_tuning_conditions(strategy_name, performance)
            
            # Calculate confidence in recommendation
            confidence = self._calculate_confidence(strategy_name, performance)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(strategy_name, performance, should_retire, should_tune)
            
            return {
                'should_retire': should_retire,
                'should_tune': should_tune,
                'confidence': confidence,
                'recommendations': recommendations,
                'performance_trend': self._calculate_performance_trend(strategy_name),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking strategy performance: {e}")
            return {
                'should_retire': False,
                'should_tune': False,
                'confidence': 0.0,
                'recommendations': ['Error occurred during performance check'],
                'error': str(e)
            }
    
    def check_model_performance(self, model_name: str, 
                              performance: Dict[str, float]) -> Dict[str, Any]:
        """Check model performance and determine if action is needed.
        
        Args:
            model_name: Name of the model
            performance: Current performance metrics
            
        Returns:
            Dictionary with recommendations
        """
        try:
            # Store performance in history
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
            
            performance_record = {
                'timestamp': datetime.now(),
                'performance': performance.copy()
            }
            self.performance_history[model_name].append(performance_record)
            
            # Keep only last 50 records
            if len(self.performance_history[model_name]) > 50:
                self.performance_history[model_name] = self.performance_history[model_name][-50:]
            
            # Check retirement conditions (adapted for models)
            should_retire = self._check_model_retirement_conditions(model_name, performance)
            
            # Check tuning conditions
            should_tune = self._check_model_tuning_conditions(model_name, performance)
            
            # Calculate confidence in recommendation
            confidence = self._calculate_model_confidence(model_name, performance)
            
            # Generate recommendations
            recommendations = self._generate_model_recommendations(model_name, performance, should_retire, should_tune)
            
            return {
                'should_retire': should_retire,
                'should_tune': should_tune,
                'confidence': confidence,
                'recommendations': recommendations,
                'performance_trend': self._calculate_performance_trend(model_name),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return {
                'should_retire': False,
                'should_tune': False,
                'confidence': 0.0,
                'recommendations': ['Error occurred during performance check'],
                'error': str(e)
            }
    
    def suggest_improvements(self, strategy_name: str, 
                           performance: Dict[str, float]) -> List[str]:
        """Suggest improvements for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance: Current performance metrics
            
        Returns:
            List of improvement suggestions
        """
        try:
            suggestions = []
            
            # Check specific performance areas
            if performance.get('sharpe_ratio', 0) < 0.8:
                suggestions.append("Consider reducing position sizes to improve Sharpe ratio")
            
            if performance.get('win_rate', 0) < 0.5:
                suggestions.append("Review entry/exit criteria to improve win rate")
            
            if performance.get('max_drawdown', 0) > 0.15:
                suggestions.append("Implement stricter stop-loss mechanisms")
            
            if performance.get('volatility', 0) > 0.25:
                suggestions.append("Consider adding volatility filters")
            
            # Check for parameter tuning opportunities
            if self._should_tune_parameters(strategy_name, performance):
                suggestions.append("Consider hyperparameter optimization")
            
            # Check for regime adaptation
            if self._should_adapt_to_regime(strategy_name, performance):
                suggestions.append("Consider regime-specific parameter adjustments")
            
            # If no specific suggestions, provide general ones
            if not suggestions:
                suggestions.append("Monitor performance closely and consider parameter fine-tuning")
                suggestions.append("Review market conditions and adjust strategy accordingly")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return ["Monitor performance and consider parameter adjustments"]
    
    def _check_retirement_conditions(self, strategy_name: str, 
                                   performance: Dict[str, float]) -> bool:
        """Check if strategy should be retired."""
        try:
            # Check individual thresholds
            if performance.get('sharpe_ratio', 0) < self.retirement_thresholds['sharpe_ratio']:
                return True
            
            if performance.get('win_rate', 0) < self.retirement_thresholds['win_rate']:
                return True
            
            if abs(performance.get('max_drawdown', 0)) > self.retirement_thresholds['max_drawdown']:
                return True
            
            if performance.get('total_return', 0) < self.retirement_thresholds['total_return']:
                return True
            
            # Check performance trend
            if strategy_name in self.performance_history and len(self.performance_history[strategy_name]) >= 10:
                recent_performance = self.performance_history[strategy_name][-10:]
                older_performance = self.performance_history[strategy_name][-20:-10]
                
                if len(recent_performance) >= 5 and len(older_performance) >= 5:
                    recent_sharpe = np.mean([p['performance'].get('sharpe_ratio', 0) for p in recent_performance])
                    older_sharpe = np.mean([p['performance'].get('sharpe_ratio', 0) for p in older_performance])
                    
                    if recent_sharpe < older_sharpe * 0.5:  # 50% decay
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retirement conditions: {e}")
            return False
    
    def _check_model_retirement_conditions(self, model_name: str, 
                                         performance: Dict[str, float]) -> bool:
        """Check if model should be retired."""
        try:
            # Check model-specific thresholds
            if performance.get('accuracy', 0) < 0.4:
                return True
            
            if performance.get('mse', float('inf')) > 0.1:
                return True
            
            if performance.get('r2', 0) < 0.3:
                return True
            
            # Check performance trend
            if model_name in self.performance_history and len(self.performance_history[model_name]) >= 10:
                recent_performance = self.performance_history[model_name][-10:]
                older_performance = self.performance_history[model_name][-20:-10]
                
                if len(recent_performance) >= 5 and len(older_performance) >= 5:
                    recent_accuracy = np.mean([p['performance'].get('accuracy', 0) for p in recent_performance])
                    older_accuracy = np.mean([p['performance'].get('accuracy', 0) for p in older_performance])
                    
                    if recent_accuracy < older_accuracy * 0.7:  # 30% decay
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking model retirement conditions: {e}")
            return False
    
    def _check_tuning_conditions(self, strategy_name: str, 
                               performance: Dict[str, float]) -> bool:
        """Check if strategy should be tuned."""
        try:
            # Check if performance is below optimal but above retirement threshold
            if (self.retirement_thresholds['sharpe_ratio'] < performance.get('sharpe_ratio', 0) < 0.8):
                return True
            
            if (self.retirement_thresholds['win_rate'] < performance.get('win_rate', 0) < 0.6):
                return True
            
            # Check performance trend
            if strategy_name in self.performance_history and len(self.performance_history[strategy_name]) >= 10:
                recent_performance = self.performance_history[strategy_name][-5:]
                older_performance = self.performance_history[strategy_name][-10:-5]
                
                if len(recent_performance) >= 3 and len(older_performance) >= 3:
                    recent_sharpe = np.mean([p['performance'].get('sharpe_ratio', 0) for p in recent_performance])
                    older_sharpe = np.mean([p['performance'].get('sharpe_ratio', 0) for p in older_performance])
                    
                    if recent_sharpe < older_sharpe * (1 + self.tuning_thresholds['sharpe_decay']):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking tuning conditions: {e}")
            return False
    
    def _check_model_tuning_conditions(self, model_name: str, 
                                     performance: Dict[str, float]) -> bool:
        """Check if model should be tuned."""
        try:
            # Check if performance is below optimal but above retirement threshold
            if (0.4 < performance.get('accuracy', 0) < 0.7):
                return True
            
            if (0.3 < performance.get('r2', 0) < 0.6):
                return True
            
            # Check performance trend
            if model_name in self.performance_history and len(self.performance_history[model_name]) >= 10:
                recent_performance = self.performance_history[model_name][-5:]
                older_performance = self.performance_history[model_name][-10:-5]
                
                if len(recent_performance) >= 3 and len(older_performance) >= 3:
                    recent_accuracy = np.mean([p['performance'].get('accuracy', 0) for p in recent_performance])
                    older_accuracy = np.mean([p['performance'].get('accuracy', 0) for p in older_performance])
                    
                    if recent_accuracy < older_accuracy * (1 + self.tuning_thresholds['accuracy_drop']):
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking model tuning conditions: {e}")
            return False
    
    def _calculate_confidence(self, strategy_name: str, 
                            performance: Dict[str, float]) -> float:
        """Calculate confidence in the performance assessment."""
        try:
            # Base confidence on performance stability
            if strategy_name in self.performance_history and len(self.performance_history[strategy_name]) >= 5:
                recent_performances = self.performance_history[strategy_name][-5:]
                sharpe_values = [p['performance'].get('sharpe_ratio', 0) for p in recent_performances]
                
                # Higher confidence for more stable performance
                stability = 1.0 - min(1.0, np.std(sharpe_values) / 0.5)
                return max(0.3, min(1.0, stability))
            else:
                return 0.5  # Default confidence for new strategies
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_model_confidence(self, model_name: str, 
                                  performance: Dict[str, float]) -> float:
        """Calculate confidence in the model performance assessment."""
        try:
            # Base confidence on model performance stability
            if model_name in self.performance_history and len(self.performance_history[model_name]) >= 5:
                recent_performances = self.performance_history[model_name][-5:]
                accuracy_values = [p['performance'].get('accuracy', 0) for p in recent_performances]
                
                # Higher confidence for more stable performance
                stability = 1.0 - min(1.0, np.std(accuracy_values) / 0.3)
                return max(0.3, min(1.0, stability))
            else:
                return 0.5  # Default confidence for new models
                
        except Exception as e:
            logger.error(f"Error calculating model confidence: {e}")
            return 0.5
    
    def _calculate_performance_trend(self, name: str) -> float:
        """Calculate performance trend over time."""
        try:
            if name not in self.performance_history or len(self.performance_history[name]) < 10:
                return 0.0
            
            recent_performances = self.performance_history[name][-10:]
            
            # Calculate trend based on key metrics
            if 'sharpe_ratio' in recent_performances[0]['performance']:
                # Strategy trend
                sharpe_values = [p['performance'].get('sharpe_ratio', 0) for p in recent_performances]
            else:
                # Model trend
                sharpe_values = [p['performance'].get('accuracy', 0) for p in recent_performances]
            
            if len(sharpe_values) >= 2:
                # Simple linear trend
                x = np.arange(len(sharpe_values))
                slope = np.polyfit(x, sharpe_values, 1)[0]
                return slope
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating performance trend: {e}")
            return 0.0
    
    def _generate_recommendations(self, strategy_name: str, 
                                performance: Dict[str, float],
                                should_retire: bool, 
                                should_tune: bool) -> List[str]:
        """Generate specific recommendations."""
        recommendations = []
        
        if should_retire:
            recommendations.append(f"Consider retiring {strategy_name} due to poor performance")
            recommendations.append("Evaluate alternative strategies for the current market conditions")
        elif should_tune:
            recommendations.append(f"Consider tuning parameters for {strategy_name}")
            recommendations.append("Review recent market conditions and adjust strategy accordingly")
        else:
            recommendations.append(f"Continue monitoring {strategy_name} performance")
            recommendations.append("Consider incremental improvements based on market analysis")
        
        return recommendations
    
    def _generate_model_recommendations(self, model_name: str, 
                                      performance: Dict[str, float],
                                      should_retire: bool, 
                                      should_tune: bool) -> List[str]:
        """Generate specific model recommendations."""
        recommendations = []
        
        if should_retire:
            recommendations.append(f"Consider retiring {model_name} due to poor performance")
            recommendations.append("Evaluate alternative models for the current data characteristics")
        elif should_tune:
            recommendations.append(f"Consider tuning hyperparameters for {model_name}")
            recommendations.append("Review feature engineering and data preprocessing")
        else:
            recommendations.append(f"Continue monitoring {model_name} performance")
            recommendations.append("Consider ensemble methods to improve robustness")
        
        return recommendations
    
    def _should_tune_parameters(self, strategy_name: str, 
                              performance: Dict[str, float]) -> bool:
        """Check if parameters should be tuned."""
        return {'success': True, 'result': performance.get('sharpe_ratio', 0) < 1.0 and performance.get('win_rate', 0) < 0.6, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def _should_adapt_to_regime(self, strategy_name: str, 
                              performance: Dict[str, float]) -> bool:
        """Check if strategy should adapt to market regime."""
        return {'success': True, 'result': performance.get('volatility', 0) > 0.2 or performance.get('max_drawdown', 0) > 0.1, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

# Global performance checker instance
performance_checker = PerformanceChecker()

def get_performance_checker() -> PerformanceChecker:
    """Get the global performance checker instance."""
    return performance_checker