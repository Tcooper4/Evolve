import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import random
from ...models.base_model import BaseModel
import torch.optim as optim

class DQNStrategyOptimizer(BaseModel):
    """Deep Q-Network for strategy optimization."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000,
                 batch_size: int = 32, target_update: int = 10):
        """Initialize DQN optimizer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay
            memory_size: Size of replay memory
            batch_size: Batch size for training
            target_update: Frequency of target network updates
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        ).to(self.device)
        
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Memory
        self.memory = deque(maxlen=memory_size)
        
        # Training state
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self._train_step_counter = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.policy_net(x)
        
    def train(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> float:
        """Train the DQN on a single experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Loss value
        """
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Sample batch for training
        if len(self.memory) < self.batch_size:
            return 0.0
            
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
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self._train_step_counter += 1
        if self._train_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
    def predict(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
            
    def save(self, path: str) -> None:
        """Save model state.
        
        Args:
            path: Path to save model state
        """
        state = {
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),
            '_train_step_counter': self._train_step_counter
        }
        torch.save(state, path)
        
    def load(self, path: str) -> None:
        """Load model state.
        
        Args:
            path: Path to load model state from
        """
        state = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state['policy_net_state'])
        self.target_net.load_state_dict(state['target_net_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.epsilon = state['epsilon']
        self.memory = deque(state['memory'], maxlen=self.memory_size)
        self._train_step_counter = state['_train_step_counter'] 