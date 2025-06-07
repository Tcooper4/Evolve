from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from collections import deque
import random

class StrategyOptimizer(ABC):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize the strategy optimizer.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (Dict[str, Any]): Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.history = []

    @abstractmethod
    def train(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool) -> float:
        """Train the model on a single step.
        
        Args:
            state (torch.Tensor): Current state
            action (torch.Tensor): Action taken
            reward (float): Reward received
            next_state (torch.Tensor): Next state
            done (bool): Whether the episode is done
            
        Returns:
            float: Loss value
        """
        pass

    @abstractmethod
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Predict the best action for a given state.
        
        Args:
            state (torch.Tensor): Current state
            
        Returns:
            torch.Tensor: Predicted action
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        pass

class DQNStrategyOptimizer(StrategyOptimizer):
    """Deep Q-Network based strategy optimizer."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize the DQN strategy optimizer.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (Dict[str, Any]): Configuration dictionary
        """
        super().__init__(state_dim, action_dim, config)
        
        # Initialize networks
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        # Initialize memory
        self.memory = deque(maxlen=config.get('memory_size', 10000))
        
        # Initialize other parameters
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 10)
        self.steps_done = 0
        
    def _build_network(self) -> nn.Module:
        """Build the neural network.
        
        Returns:
            nn.Module: Neural network model
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        ).to(self.device)
    
    def _prepare_state(self, state: np.ndarray) -> torch.Tensor:
        """Prepare state for the network.
        
        Args:
            state (np.ndarray): Raw state array
            
        Returns:
            torch.Tensor: Prepared state tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        return state.to(self.device)
    
    def fit(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """Train the model on a single step.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode is done
            
        Returns:
            float: Loss value
        """
        # Store transition in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Perform one step of optimization
        if len(self.memory) >= self.batch_size:
            loss = self._train_step()
            
            # Update target network
            if self.steps_done % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.steps_done += 1
            return loss
        
        return 0.0
    
    def _train_step(self) -> float:
        """Perform one training step.
        
        Returns:
            float: Loss value
        """
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def predict(self, state: np.ndarray) -> int:
        """Predict the best action for a given state.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Predicted action
        """
        state = self._prepare_state(state)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'config': self.config
        }, save_path)
    
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.config = checkpoint['config'] 