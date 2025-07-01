import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator."""
        self.metrics = {}
        self.predictions = {}
        self.actuals = {}
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Evaluate model performance."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Calculate directional accuracy
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        metrics['directional_accuracy'] = np.mean(direction_true == direction_pred)
        
        # Store metrics and predictions
        self.metrics[model_name] = metrics
        self.predictions[model_name] = y_pred
        self.actuals[model_name] = y_true
        
        return metrics
    
    def plot_predictions(self, model_name: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot actual vs predicted values.
        
        Returns:
            Dictionary with plot status and details
        """
        try:
            if model_name not in self.predictions:
                return {
                    'success': False,
                    'message': f'No predictions found for model: {model_name}',
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.actuals[model_name], label='Actual', color='blue')
            plt.plot(self.predictions[model_name], label='Predicted', color='red')
            plt.title(f'Actual vs Predicted Values - {model_name}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return {
                    'success': True,
                    'message': f'Predictions plot saved to {save_path}',
                    'model_name': model_name,
                    'save_path': save_path,
                    'data_points': len(self.actuals[model_name]),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                plt.show()
                plt.close()
                return {
                    'success': True,
                    'message': f'Predictions plot displayed for {model_name}',
                    'model_name': model_name,
                    'data_points': len(self.actuals[model_name]),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error plotting predictions: {str(e)}',
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def plot_residuals(self, model_name: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot residuals.
        
        Returns:
            Dictionary with plot status and details
        """
        try:
            if model_name not in self.predictions:
                return {
                    'success': False,
                    'message': f'No predictions found for model: {model_name}',
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
            
            residuals = self.actuals[model_name] - self.predictions[model_name]
            
            plt.figure(figsize=(12, 6))
            plt.scatter(self.predictions[model_name], residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'Residuals Plot - {model_name}')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return {
                    'success': True,
                    'message': f'Residuals plot saved to {save_path}',
                    'model_name': model_name,
                    'save_path': save_path,
                    'residuals_mean': np.mean(residuals),
                    'residuals_std': np.std(residuals),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                plt.show()
                plt.close()
                return {
                    'success': True,
                    'message': f'Residuals plot displayed for {model_name}',
                    'model_name': model_name,
                    'residuals_mean': np.mean(residuals),
                    'residuals_std': np.std(residuals),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error plotting residuals: {str(e)}',
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, model_name: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Plot feature importance.
        
        Returns:
            Dictionary with plot status and details
        """
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return {
                    'success': True,
                    'message': f'Feature importance plot saved to {save_path}',
                    'model_name': model_name,
                    'save_path': save_path,
                    'num_features': len(feature_importance),
                    'max_importance': feature_importance['importance'].max(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                plt.show()
                plt.close()
                return {
                    'success': True,
                    'message': f'Feature importance plot displayed for {model_name}',
                    'model_name': model_name,
                    'num_features': len(feature_importance),
                    'max_importance': feature_importance['importance'].max(),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error plotting feature importance: {str(e)}',
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_report(self, model_name: str) -> Dict:
        """Generate evaluation report."""
        if model_name not in self.metrics:
            return {}
        
        report = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics[model_name],
            'summary': self._generate_summary(model_name)
        }
        
        return report
    
    def _generate_summary(self, model_name: str) -> str:
        """Generate summary of model performance."""
        if model_name not in self.metrics:
            return "No metrics available"
        
        metrics = self.metrics[model_name]
        summary = f"""
        Model Performance Summary:
        - MSE: {metrics['mse']:.4f}
        - RMSE: {metrics['rmse']:.4f}
        - MAE: {metrics['mae']:.4f}
        - RÂ²: {metrics['r2']:.4f}
        - Directional Accuracy: {metrics['directional_accuracy']:.2%}
        """
        
        return summary
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare multiple models."""
        comparison = pd.DataFrame()
        
        for name in model_names:
            if name in self.metrics:
                metrics = self.metrics[name]
                comparison[name] = pd.Series(metrics)
        
        return comparison
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics.
        
        Returns:
            Dictionary with evaluation metrics and status
        """
        try:
            return {
                'success': True,
                'num_models_evaluated': len(self.metrics),
                'num_metrics_per_model': len(next(iter(self.metrics.values()))) if self.metrics else 0,
                'model_names': list(self.metrics.keys()),
                'total_predictions': sum(len(pred) for pred in self.predictions.values()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error getting evaluation metrics: {str(e)}',
                'timestamp': datetime.now().isoformat()
            } 