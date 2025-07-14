"""Reinforcement Learning Strategy Trainer for Evolve Trading Platform.

This module provides RL capabilities for training trading strategies
using various algorithms and environments.
"""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import RL libraries with fallbacks
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None

try:
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    PPO = DQN = A2C = DummyVecEnv = BaseCallback = None

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """Custom trading environment for reinforcement learning."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        transaction_fee: float = 0.001,
        max_position: float = 0.2,
    ):
        """Initialize trading environment.

        Args:
            data: Market data with OHLCV
            initial_balance: Initial account balance
            transaction_fee: Transaction fee percentage
            max_position: Maximum position size
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position

        self.reset()

        # Define action and observation spaces
        if GYMNASIUM_AVAILABLE:
            self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
            )
        else:
            self.action_space = None
            self.observation_space = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed

        Returns:
            Initial observation and info
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.total_value = self.initial_balance
        self.returns = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action in environment.

        Args:
            action: Action to take (0=Buy, 1=Sell, 2=Hold)

        Returns:
            Observation, reward, terminated, truncated, info
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, self._get_info()

        # Execute action
        reward = self._execute_action(action)

        # Move to next step
        self.current_step += 1

        # Update portfolio value
        self._update_portfolio_value()

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, done, False, info

    def _execute_action(self, action: int) -> float:
        """Execute trading action.

        Args:
            action: Action to execute

        Returns:
            Reward for the action
        """
        current_price = self.data.iloc[self.current_step]["Close"]
        next_price = self.data.iloc[self.current_step + 1]["Close"]

        reward = 0

        if action == 0:  # Buy
            if self.balance > 0:
                # Calculate position size
                position_value = self.balance * self.max_position
                shares_to_buy = position_value / current_price

                # Execute trade
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.shares += shares_to_buy
                    self.balance -= cost

                    # Calculate reward based on price movement
                    price_change = (next_price - current_price) / current_price
                    reward = price_change * shares_to_buy * current_price

        elif action == 1:  # Sell
            if self.shares > 0:
                # Sell all shares
                proceeds = self.shares * current_price * (1 - self.transaction_fee)
                self.balance += proceeds

                # Calculate reward based on price movement
                price_change = (current_price - next_price) / current_price
                reward = price_change * self.shares * current_price

                self.shares = 0

        else:  # Hold
            # Small penalty for holding to encourage action
            reward = -0.0001

        return reward

    def _update_portfolio_value(self):
        """Update total portfolio value."""
        current_price = self.data.iloc[self.current_step]["Close"]
        self.total_value = self.balance + self.shares * current_price

        # Calculate return
        if self.current_step > 0:
            prev_value = (
                self.total_value - self.returns[-1]
                if self.returns
                else self.initial_balance
            )
            return_val = (self.total_value - prev_value) / prev_value
            self.returns.append(return_val)
        else:
            self.returns.append(0)

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation array
        """
        if self.current_step >= len(self.data):
            return np.zeros(10, dtype=np.float32)

        row = self.data.iloc[self.current_step]

        # Create feature vector
        features = [
            row["Open"] / row["Close"] - 1,  # Open-close ratio
            row["High"] / row["Close"] - 1,  # High-close ratio
            row["Low"] / row["Close"] - 1,  # Low-close ratio
            row["Volume"] / self.data["Volume"].mean() - 1,  # Volume ratio
            self.balance / self.initial_balance - 1,  # Balance ratio
            self.shares * row["Close"] / self.initial_balance,  # Position size
            self.total_value / self.initial_balance - 1,  # Portfolio return
            len(self.returns) > 0 and self.returns[-1] or 0,  # Last return
            self.current_step / len(self.data),  # Progress
            1.0 if self.shares > 0 else 0.0,  # Has position
        ]

        return np.array(features, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get environment info.

        Returns:
            Environment information
        """
        return {
            "balance": self.balance,
            "shares": self.shares,
            "total_value": self.total_value,
            "step": self.current_step,
            "returns": self.returns.copy() if self.returns else [],
        }


class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called after each step."""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            if any(dones):
                # Episode ended
                pass


def create_rl_environment(
    data: pd.DataFrame, config: Optional[Dict[str, Any]] = None
) -> TradingEnvironment:
    """Create RL trading environment.

    Args:
        data: Market data
        config: Environment configuration

    Returns:
        TradingEnvironment instance
    """
    if not GYMNASIUM_AVAILABLE:
        raise ImportError("Gymnasium not available. Please install gymnasium.")

    config = config or {}
    initial_balance = config.get("initial_balance", 100000)
    transaction_fee = config.get("transaction_fee", 0.001)
    max_position = config.get("max_position", 0.2)

    return TradingEnvironment(
        data=data,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        max_position=max_position,
    )


def train_rl_strategy(
    env: TradingEnvironment,
    algorithm: str = "PPO",
    total_timesteps: int = 10000,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Train RL strategy.

    Args:
        env: Trading environment
        algorithm: RL algorithm to use
        total_timesteps: Number of training timesteps
        config: Training configuration

    Returns:
        Trained model
    """
    if not STABLE_BASELINES3_AVAILABLE:
        logger.error("Stable-baselines3 not available")
        return None

    if not GYMNASIUM_AVAILABLE:
        logger.error("Gymnasium not available")
        return None

    try:
        # Wrap environment
        vec_env = DummyVecEnv([lambda: env])

        # Create model
        if algorithm == "PPO":
            model = PPO("MlpPolicy", vec_env, verbose=1)
        elif algorithm == "DQN":
            model = DQN("MlpPolicy", vec_env, verbose=1)
        elif algorithm == "A2C":
            model = A2C("MlpPolicy", vec_env, verbose=1)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Train model
        model.learn(total_timesteps=total_timesteps)

        logger.info(f"Trained {algorithm} model for {total_timesteps} timesteps")
        return model

    except Exception as e:
        logger.error(f"Error training RL strategy: {e}")
        return None


def evaluate_rl_strategy(
    model: Any, env: TradingEnvironment, num_episodes: int = 10
) -> Dict[str, float]:
    """Evaluate trained RL strategy.

    Args:
        model: Trained model
        env: Trading environment
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    if model is None:
        return {}

    try:
        total_rewards = []
        total_returns = []
        final_values = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

                if done or truncated:
                    break

            total_rewards.append(episode_reward)
            total_returns.append(info["total_value"] / env.initial_balance - 1)
            final_values.append(info["total_value"])

        metrics = {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_return": np.mean(total_returns),
            "std_return": np.std(total_returns),
            "mean_final_value": np.mean(final_values),
            "sharpe_ratio": np.mean(total_returns) / np.std(total_returns)
            if np.std(total_returns) > 0
            else 0,
        }

        logger.info(f"RL strategy evaluation: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Error evaluating RL strategy: {e}")
        return {}


def create_rl_strategy_trainer(config: Optional[Dict[str, Any]] = None):
    """Create RL strategy trainer.

    Args:
        config: Configuration dictionary

    Returns:
        RL trainer functions
    """
    return {
        "create_environment": create_rl_environment,
        "train_strategy": train_rl_strategy,
        "evaluate_strategy": evaluate_rl_strategy,
        "available": GYMNASIUM_AVAILABLE and STABLE_BASELINES3_AVAILABLE,
    }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    data = pd.DataFrame(
        {
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(105, 115, 100),
            "Low": np.random.uniform(95, 105, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.uniform(1000000, 5000000, 100),
        },
        index=dates,
    )

    # Create environment
    env = create_rl_environment(data)

    # Train strategy
    model = train_rl_strategy(env, "PPO", total_timesteps=1000)

    # Evaluate strategy
    if model:
        metrics = evaluate_rl_strategy(model, env)
        print("Evaluation metrics:", metrics)
