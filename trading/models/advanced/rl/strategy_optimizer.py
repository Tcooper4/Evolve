import logging
import random
from abc import ABC
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ PyTorch not available. Disabling RL strategy optimizers.")
    print(f"   Missing: {e}")
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class StrategyOptimizer(ABC):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """Initialize the strategy optimizer.

        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config (Dict[str, Any]): Configuration dictionary
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create StrategyOptimizer.")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.history = []

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
            nn.Linear(64, self.action_dim),
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

    def fit(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
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
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "epsilon": self.epsilon,
                "config": self.config,
            },
            save_path,
        )

    def load(self, path: str) -> None:
        """Load the model from disk.

        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        self.epsilon = checkpoint["epsilon"]
        self.config = checkpoint["config"]

    def plot_results(self, *args, **kwargs):
        """Plot training results and performance metrics.

        Args:
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        try:
            import matplotlib.pyplot as plt

            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("DQN Strategy Optimizer Results", fontsize=16)

            # Plot 1: Training loss over time
            if hasattr(self, "loss_history"):
                axes[0, 0].plot(self.loss_history)
                axes[0, 0].set_title("Training Loss")
                axes[0, 0].set_xlabel("Training Steps")
                axes[0, 0].set_ylabel("Loss")
                axes[0, 0].grid(True)
            else:
                axes[0, 0].text(
                    0.5,
                    0.5,
                    "No loss history available",
                    ha="center",
                    va="center",
                    transform=axes[0, 0].transAxes,
                )
                axes[0, 0].set_title("Training Loss")

            # Plot 2: Epsilon decay
            if hasattr(self, "epsilon_history"):
                axes[0, 1].plot(self.epsilon_history)
                axes[0, 1].set_title("Epsilon Decay")
                axes[0, 1].set_xlabel("Training Steps")
                axes[0, 1].set_ylabel("Epsilon")
                axes[0, 1].grid(True)
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    f"Current Epsilon: {self.epsilon:.3f}",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Epsilon Decay")

            # Plot 3: Q-value distribution
            if hasattr(self, "q_values_history"):
                axes[1, 0].hist(self.q_values_history, bins=20, alpha=0.7)
                axes[1, 0].set_title("Q-Value Distribution")
                axes[1, 0].set_xlabel("Q-Value")
                axes[1, 0].set_ylabel("Frequency")
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "No Q-value history available",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("Q-Value Distribution")

            # Plot 4: Action distribution
            if hasattr(self, "action_history"):
                action_counts = pd.Series(self.action_history).value_counts()
                axes[1, 1].bar(action_counts.index, action_counts.values)
                axes[1, 1].set_title("Action Distribution")
                axes[1, 1].set_xlabel("Action")
                axes[1, 1].set_ylabel("Count")
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No action history available",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Action Distribution")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"Error plotting DQN results: {e}")
            logger.error(f"Could not plot results: {e}")


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
            self.policy_net.parameters(), lr=config.get("learning_rate", 0.001)
        )

        # Initialize memory
        self.memory = deque(maxlen=config.get("memory_size", 10000))

        # Initialize other parameters
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.target_update = config.get("target_update", 10)
        self.steps_done = 0
