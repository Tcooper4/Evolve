import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from collections import deque
import random
from trading.models.base_model import BaseModel
import torch.optim as optim
import pandas as pd
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime
import json
import os
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class OptimizationResult:
    """Results from strategy optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_scores: List[float]
    all_params: List[Dict[str, Any]]
    optimization_time: float
    n_iterations: int
    convergence_history: List[float]

class OptimizationMethod(ABC):
    """Base class for optimization methods."""
    
    @abstractmethod
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run optimization.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        pass

class GridSearch(OptimizationMethod):
    """Grid search optimization method."""
    
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run grid search optimization.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        start_time = datetime.now()
        
        # Generate parameter grid
        param_grid = self._generate_grid(param_space)
        n_iterations = len(param_grid)
        
        # Run grid search
        scores = []
        best_score = float('inf')
        best_params = None
        convergence_history = []
        
        for params in param_grid:
            score = objective(params, data)
            scores.append(score)
            convergence_history.append(score)
            
            if score < best_score:
                best_score = score
                best_params = params
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=scores,
            all_params=param_grid,
            optimization_time=optimization_time,
            n_iterations=n_iterations,
            convergence_history=convergence_history
        )
    
    def _generate_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter grid from parameter space.
        
        Args:
            param_space: Parameter space definition
            
        Returns:
            List of parameter combinations
        """
        import itertools
        
        # Convert parameter space to lists
        param_lists = {}
        for param, space in param_space.items():
            if isinstance(space, (list, tuple)):
                param_lists[param] = space
            elif isinstance(space, dict):
                if 'start' in space and 'end' in space and 'step' in space:
                    param_lists[param] = np.arange(
                        space['start'], space['end'], space['step']
                    )
                elif 'start' in space and 'end' in space and 'num' in space:
                    param_lists[param] = np.linspace(
                        space['start'], space['end'], space['num']
                    )
        
        # Generate all combinations
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())
        combinations = list(itertools.product(*param_values))
        
        return [dict(zip(param_names, combo)) for combo in combinations]

class BayesianOptimization(OptimizationMethod):
    """Bayesian optimization method."""
    
    def optimize(self, objective: Callable, param_space: Dict[str, Any], 
                data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Run Bayesian optimization.
        
        Args:
            objective: Objective function to minimize
            param_space: Parameter space to search
            data: Market data
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        
        start_time = datetime.now()
        
        # Convert parameter space to skopt format
        dimensions = []
        param_names = []
        for param, space in param_space.items():
            param_names.append(param)
            if isinstance(space, (list, tuple)):
                dimensions.append(Categorical(space, name=param))
            elif isinstance(space, dict):
                if 'start' in space and 'end' in space:
                    if isinstance(space['start'], int):
                        dimensions.append(Integer(space['start'], space['end'], name=param))
                    else:
                        dimensions.append(Real(space['start'], space['end'], name=param))
        
        # Define objective function
        def objective_wrapper(params):
            param_dict = dict(zip(param_names, params))
            return objective(param_dict, data)
        
        # Run optimization
        n_calls = kwargs.get('n_calls', 100)
        result = gp_minimize(
            objective_wrapper,
            dimensions,
            n_calls=n_calls,
            random_state=42,
            n_jobs=-1
        )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=dict(zip(param_names, result.x)),
            best_score=result.fun,
            all_scores=result.func_vals,
            all_params=[dict(zip(param_names, x)) for x in result.x_iters],
            optimization_time=optimization_time,
            n_iterations=n_calls,
            convergence_history=result.func_vals
        )

class StrategyOptimizer:
    """Optimizes trading strategy parameters."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize strategy optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create results directory
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize optimization methods
        self.methods = {
            'grid': GridSearch(),
            'bayesian': BayesianOptimization()
        }
    
    def optimize(self, strategy: Any, data: pd.DataFrame, param_space: Dict[str, Any],
                method: str = 'bayesian', n_splits: int = 5, **kwargs) -> OptimizationResult:
        """Optimize strategy parameters.
        
        Args:
            strategy: Strategy to optimize
            data: Market data
            param_space: Parameter space to search
            method: Optimization method to use
            n_splits: Number of time series splits for cross-validation
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        if method not in self.methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Define objective function
        def objective(params: Dict[str, Any], data: pd.DataFrame) -> float:
            # Set strategy parameters
            strategy.set_params(**params)
            
            # Perform time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            
            for train_idx, test_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Train strategy
                strategy.fit(train_data)
                
                # Evaluate on test set
                score = strategy.evaluate(test_data)
                scores.append(score)
            
            # Return average score
            return -np.mean(scores)  # Negative because we want to maximize
        
        # Run optimization
        self.logger.info(f"Starting optimization using {method} method")
        result = self.methods[method].optimize(objective, param_space, data, **kwargs)
        
        # Save results
        self.save_results(result, f"{strategy.__class__.__name__}_{method}")
        
        self.logger.info(f"Optimization completed in {result.optimization_time:.2f} seconds")
        self.logger.info(f"Best parameters: {result.best_params}")
        self.logger.info(f"Best score: {result.best_score:.4f}")
        
        return result
    
    def parallel_optimize(self, strategies: List[Any], data: pd.DataFrame,
                         param_spaces: List[Dict[str, Any]], method: str = 'bayesian',
                         n_splits: int = 5, **kwargs) -> Dict[str, OptimizationResult]:
        """Optimize multiple strategies in parallel.
        
        Args:
            strategies: List of strategies to optimize
            data: Market data
            param_spaces: List of parameter spaces for each strategy
            method: Optimization method to use
            n_splits: Number of time series splits for cross-validation
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary mapping strategy names to optimization results
        """
        results = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_strategy = {
                executor.submit(
                    self.optimize,
                    strategy,
                    data,
                    param_space,
                    method,
                    n_splits,
                    **kwargs
                ): strategy
                for strategy, param_space in zip(strategies, param_spaces)
            }
            
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results[strategy.__class__.__name__] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing {strategy.__class__.__name__}: {str(e)}")
        
        return results
    
    def save_results(self, result: OptimizationResult, filename: str) -> None:
        """Save optimization results to disk.
        
        Args:
            result: Optimization results
            filename: Output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"{filename}_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_result = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'all_scores': result.all_scores,
            'all_params': result.all_params,
            'optimization_time': result.optimization_time,
            'n_iterations': result.n_iterations,
            'convergence_history': result.convergence_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=4)
        self.logger.info(f"Saved results to {filepath}")
    
    def load_results(self, filename: str) -> OptimizationResult:
        """Load optimization results from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            OptimizationResult object
        """
        filepath = self.results_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return OptimizationResult(
            best_params=data['best_params'],
            best_score=data['best_score'],
            all_scores=data['all_scores'],
            all_params=data['all_params'],
            optimization_time=data['optimization_time'],
            n_iterations=data['n_iterations'],
            convergence_history=data['convergence_history']
        )

class BaseOptimizer(ABC):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Create results directory
        self.results_dir = Path("optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'sharpe_ratio': [],
            'returns': [],
            'volatility': []
        }
    
    @abstractmethod
    def optimize(self, data: pd.DataFrame, strategy: Any) -> Dict:
        """Optimize the strategy parameters."""
        pass
    
    def save_results(self, results: Dict, strategy_name: str):
        """Save optimization results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{strategy_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Saved optimization results to {filename}")
    
    def load_results(self, filename: str) -> Dict:
        """Load optimization results from disk."""
        with open(self.results_dir / filename, 'r') as f:
            results = json.load(f)
        self.logger.info(f"Loaded optimization results from {filename}")
        return results
    
    def plot_metrics(self):
        """Plot optimization metrics history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot losses
            axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.metrics_history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            
            # Plot Sharpe ratio
            axes[0, 1].plot(self.metrics_history['sharpe_ratio'])
            axes[0, 1].set_title('Sharpe Ratio')
            
            # Plot returns
            axes[1, 0].plot(self.metrics_history['returns'])
            axes[1, 0].set_title('Returns')
            
            # Plot volatility
            axes[1, 1].plot(self.metrics_history['volatility'])
            axes[1, 1].set_title('Volatility')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'optimization_metrics.png')
            plt.close()
            
        except ImportError:
            self.logger.warning("Matplotlib not installed. Skipping metrics plotting.")

class DQNStrategyOptimizer(BaseOptimizer):
    """Deep Q-Network for strategy optimization."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[Dict] = None):
        super().__init__(config)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = self._create_network()
        self.target_net = self._create_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                  lr=self.config.get('learning_rate', 0.001))
        
        # Initialize replay memory
        self.memory = ReplayMemory(
            capacity=self.config.get('memory_size', 10000),
            device=self.device
        )
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 64)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.target_update = self.config.get('target_update', 10)
        self.warmup_steps = self.config.get('warmup_steps', 1000)
    
    def _create_network(self) -> nn.Module:
        """Create the neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.action_dim)
        ).to(self.device)
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training."""
        # Convert data to tensors
        states = torch.FloatTensor(data.values).to(self.device)
        actions = torch.LongTensor(data['action'].values).to(self.device)
        rewards = torch.FloatTensor(data['reward'].values).to(self.device)
        next_states = torch.FloatTensor(data.shift(-1).values).to(self.device)
        
        # Create dataset
        dataset = TensorDataset(states, actions, rewards, next_states)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def _setup_model(self):
        """Setup the model for training."""
        self.policy_net.train()
        self.target_net.eval()
    
    def optimize(self, data: pd.DataFrame, strategy: Any) -> Dict:
        """Optimize the strategy using DQN."""
        self.logger.info("Starting DQN optimization")
        
        # Prepare data
        dataloader = self._prepare_data(data)
        self._setup_model()
        
        # Training loop
        episode_rewards = []
        best_reward = float('-inf')
        best_params = None
        
        for episode in range(self.config.get('episodes', 1000)):
            episode_reward = 0
            for states, actions, rewards, next_states in dataloader:
                # Get current Q values
                current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Get next Q values
                with torch.no_grad():
                    next_q_values = self.target_net(next_states).max(1)[0]
                
                # Compute target Q values
                target_q_values = rewards + (self.gamma * next_q_values)
                
                # Compute loss
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                
                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
                
                # Update epsilon
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                
                episode_reward += rewards.sum().item()
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Log progress
            episode_rewards.append(episode_reward)
            self.logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.3f}")
            
            # Save best parameters
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = self.policy_net.state_dict()
        
        # Save results
        results = {
            'best_reward': best_reward,
            'final_epsilon': self.epsilon,
            'episode_rewards': episode_rewards,
            'best_params': best_params
        }
        
        self.save_results(results, strategy.__class__.__name__)
        return results
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        self.logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.logger.info(f"Loaded model from {path}")

class ReplayMemory:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.memory = []
        self.position = 0
    
    def push(self, state: torch.Tensor, action: torch.Tensor, 
             reward: torch.Tensor, next_state: torch.Tensor):
        """Push a transition to memory."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions."""
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (torch.stack(state), torch.stack(action), torch.stack(reward), torch.stack(next_state))
    
    def __len__(self) -> int:
        return len(self.memory) 