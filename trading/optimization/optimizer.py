"""
DEPRECATED: This file is redundant or for development purposes only.
Please use strategy_optimizer.py for optimization functionality.
Last updated: 2025-06-18 13:06:26
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import os
from trading.strategy_optimizer import StrategyOptimizer

class OptimizationError(Exception):
    """Custom exception for optimization errors."""
    pass

class Optimizer:
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict] = None):
        """Initialize the optimizer with state and action dimensions.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary containing:
                - learning_rate: Learning rate for optimizer (default: 0.001)
                - hidden_dims: List of hidden layer dimensions (default: [64, 32])
                - dropout_rate: Dropout rate (default: 0.1)
                - batch_norm: Whether to use batch normalization (default: True)
                - log_level: Logging level (default: INFO)
                - results_dir: Directory for saving results (default: optimization_results)
        """
        # Load configuration from environment variables with defaults
        self.config = {
            'learning_rate': float(os.getenv('OPTIMIZER_LEARNING_RATE', 0.001)),
            'hidden_dims': [int(x) for x in os.getenv('OPTIMIZER_HIDDEN_DIMS', '64,32').split(',')],
            'dropout_rate': float(os.getenv('OPTIMIZER_DROPOUT_RATE', 0.1)),
            'batch_norm': os.getenv('OPTIMIZER_BATCH_NORM', 'true').lower() == 'true',
            'log_level': os.getenv('OPTIMIZER_LOG_LEVEL', 'INFO'),
            'results_dir': os.getenv('OPTIMIZER_RESULTS_DIR', 'optimization_results')
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config['log_level']))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Validate dimensions
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise OptimizationError("state_dim must be a positive integer")
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise OptimizationError("action_dim must be a positive integer")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize neural network
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'loss': [],
            'validation_loss': [],
            'optimization_time': []
        }
    
    def _build_model(self) -> nn.Module:
        """Build neural network model.
        
        Returns:
            Neural network model
        """
        layers = []
        prev_dim = self.state_dim
        
        # Add hidden layers
        for dim in self.config['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim) if self.config['batch_norm'] else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(self.config['dropout_rate'])
            ])
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, self.action_dim))
        
        return nn.Sequential(*layers).to(self.device)
    
    def optimize_portfolio(self, state: np.ndarray, target: np.ndarray,
                         validation_split: float = 0.2) -> Tuple[np.ndarray, float]:
        """Optimize portfolio weights given current state and target.
        
        Args:
            state: State array
            target: Target array
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (optimized weights, loss)
            
        Raises:
            OptimizationError: If optimization fails
        """
        try:
            # Validate inputs
            if not isinstance(state, np.ndarray) or not isinstance(target, np.ndarray):
                raise OptimizationError("state and target must be numpy arrays")
            if state.shape[0] != target.shape[0]:
                raise OptimizationError("state and target must have same number of samples")
            
            # Split data
            split_idx = int(len(state) * (1 - validation_split))
            train_state = state[:split_idx]
            train_target = target[:split_idx]
            val_state = state[split_idx:]
            val_target = target[split_idx:]
            
            # Convert to tensors
            train_state_tensor = torch.FloatTensor(train_state).to(self.device)
            train_target_tensor = torch.FloatTensor(train_target).to(self.device)
            val_state_tensor = torch.FloatTensor(val_state).to(self.device)
            val_target_tensor = torch.FloatTensor(val_target).to(self.device)
            
            # Training
            start_time = datetime.now()
            self.optimizer.zero_grad()
            train_output = self.model(train_state_tensor)
            train_loss = self.criterion(train_output, train_target_tensor)
            train_loss.backward()
            self.optimizer.step()
            
            # Validation
            with torch.no_grad():
                val_output = self.model(val_state_tensor)
                val_loss = self.criterion(val_output, val_target_tensor)
            
            # Update metrics
            optimization_time = (datetime.now() - start_time).total_seconds()
            self.metrics_history['loss'].append(train_loss.item())
            self.metrics_history['validation_loss'].append(val_loss.item())
            self.metrics_history['optimization_time'].append(optimization_time)
            
            self.logger.info(f"Optimization completed - Loss: {train_loss.item():.4f}, "
                           f"Validation Loss: {val_loss.item():.4f}, "
                           f"Time: {optimization_time:.2f}s")
            
            return train_output.detach().cpu().numpy(), train_loss.item()
            
        except Exception as e:
            raise OptimizationError(f"Failed to optimize portfolio: {str(e)}")
    
    def get_optimal_weights(self, state: np.ndarray) -> np.ndarray:
        """Get optimal portfolio weights for given state.
        
        Args:
            state: State array
            
        Returns:
            Optimal weights array
            
        Raises:
            OptimizationError: If weight calculation fails
        """
        try:
            if not isinstance(state, np.ndarray):
                raise OptimizationError("state must be a numpy array")
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                weights = self.model(state_tensor)
                return weights.cpu().numpy()
                
        except Exception as e:
            raise OptimizationError(f"Failed to get optimal weights: {str(e)}")
    
    def update_model(self, state: np.ndarray, target: np.ndarray) -> float:
        """Update model with new data.
        
        Args:
            state: State array
            target: Target array
            
        Returns:
            Loss value
            
        Raises:
            OptimizationError: If model update fails
        """
        try:
            if not isinstance(state, np.ndarray) or not isinstance(target, np.ndarray):
                raise OptimizationError("state and target must be numpy arrays")
            
            state_tensor = torch.FloatTensor(state).to(self.device)
            target_tensor = torch.FloatTensor(target).to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()
            
            self.logger.info(f"Model updated - Loss: {loss.item():.4f}")
            return loss.item()
            
        except Exception as e:
            raise OptimizationError(f"Failed to update model: {str(e)}")
    
    def save_model(self, path: str):
        """Save model to file.
        
        Args:
            path: Path to save model
            
        Raises:
            OptimizationError: If model saving fails
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'metrics_history': self.metrics_history
            }, save_path)
            
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            raise OptimizationError(f"Failed to save model: {str(e)}")
    
    def load_model(self, path: str):
        """Load model from file.
        
        Args:
            path: Path to load model from
            
        Raises:
            OptimizationError: If model loading fails
        """
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.metrics_history = checkpoint['metrics_history']
            
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            raise OptimizationError(f"Failed to load model: {str(e)}")
    
    def get_optimization_metrics(self) -> Dict[str, float]:
        """Get optimization metrics.
        
        Returns:
            Dictionary of optimization metrics
        """
        metrics = {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'avg_loss': np.mean(self.metrics_history['loss']) if self.metrics_history['loss'] else 0,
            'avg_validation_loss': np.mean(self.metrics_history['validation_loss']) if self.metrics_history['validation_loss'] else 0,
            'avg_optimization_time': np.mean(self.metrics_history['optimization_time']) if self.metrics_history['optimization_time'] else 0
        }
        
        return metrics
    
    def save_metrics(self, path: Optional[str] = None):
        """Save optimization metrics to file.
        
        Args:
            path: Optional path to save metrics
            
        Raises:
            OptimizationError: If metrics saving fails
        """
        try:
            if path is None:
                path = self.results_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            metrics = {
                'optimization_metrics': self.get_optimization_metrics(),
                'metrics_history': self.metrics_history
            }
            
            with open(path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            self.logger.info(f"Metrics saved to {path}")
            
        except Exception as e:
            raise OptimizationError(f"Failed to save metrics: {str(e)}")

# Re-export StrategyOptimizer as Optimizer for backward compatibility
Optimizer = StrategyOptimizer

__all__ = ['Optimizer'] 