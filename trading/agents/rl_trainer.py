"""
RL Trainer Module

Reinforcement learning agent training using Gymnasium and Stable-Baselines3.
Creates custom trading environments and trains PPO/A2C agents on price+macro+sentiment data.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Import RL libraries with fallback handling
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    logging.warning("Gymnasium not available. Install with: pip install gymnasium")

try:
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    logging.warning(
        "Stable-Baselines3 not available. Install with: pip install stable-baselines3"
    )

from trading.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        max_position: float = 1.0,
    ):
        """Initialize the trading environment.

        Args:
            data: Price data with features
            initial_balance: Initial account balance
            transaction_fee: Transaction fee as percentage
            max_position: Maximum position size
        """
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position

        # Reset environment state
        self.reset()

        # Define action space (buy, sell, hold)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        # Features: price, volume, technical indicators, macro features, sentiment
        n_features = len(data.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        logger.info(
            f"Trading environment initialized with {len(data)} timesteps and {n_features} features"
        )

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_price = (
            self.data.iloc[self.current_step]["close"]
            if "close" in self.data.columns
            else self.data.iloc[self.current_step, 0]
        )

        return self._get_observation(), {}

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: 0=hold, 1=buy, 2=sell

        Returns:
            observation, reward, done, truncated, info
        """
        # Get current price
        current_price = (
            self.data.iloc[self.current_step]["close"]
            if "close" in self.data.columns
            else self.data.iloc[self.current_step, 0]
        )

        # Execute action
        reward = 0
        if action == 1:  # Buy
            if self.balance > 0:
                shares_to_buy = self.balance / current_price * self.max_position
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.shares_held += shares_to_buy
                    self.balance -= cost
                    reward = 0  # No immediate reward for buying
        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held * self.max_position
                revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.shares_held -= shares_to_sell
                self.balance += revenue
                self.total_shares_sold += shares_to_sell
                self.total_sales_value += revenue

                # Calculate reward based on profit/loss
                reward = revenue - (shares_to_sell * self.current_price)

        # Move to next step
        self.current_step += 1
        self.current_price = current_price

        # Check if episode is done
        done = self.current_step >= len(self.data) - 1

        # Calculate total portfolio value
        portfolio_value = self.balance + (self.shares_held * current_price)

        # Get observation
        observation = self._get_observation()

        # Additional info
        info = {
            "balance": self.balance,
            "shares_held": self.shares_held,
            "portfolio_value": portfolio_value,
            "current_price": current_price,
            "total_return": (portfolio_value - self.initial_balance)
            / self.initial_balance,
        }

        return observation, reward, done, False, info

    def _get_observation(self):
        """Get current observation."""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])

        # Get features for current timestep
        features = self.data.iloc[self.current_step].values.astype(np.float32)

        # Add portfolio state features
        portfolio_features = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                self.shares_held,  # Shares held
                (
                    self.current_price / self.data.iloc[0]["close"]
                    if "close" in self.data.columns
                    else 1.0
                ),  # Price ratio
            ],
            dtype=np.float32,
        )

        # Combine features
        observation = np.concatenate([features, portfolio_features])

        return observation


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []

    def _on_step(self) -> bool:
        """Called after each step."""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    # Get episode info
                    if "episode" in self.locals["infos"][i]:
                        episode_info = self.locals["infos"][i]["episode"]
                        self.episode_rewards.append(episode_info["r"])
                        self.episode_lengths.append(episode_info["l"])

                    # Get portfolio value
                    if "portfolio_value" in self.locals["infos"][i]:
                        self.portfolio_values.append(
                            self.locals["infos"][i]["portfolio_value"]
                        )


class RLTrainer:
    """Reinforcement learning trainer for trading agents."""

    def __init__(
        self, model_dir: str = "models/rl_agents", log_dir: str = "logs/rl_training"
    ):
        """Initialize the RL trainer.

        Args:
            model_dir: Directory to save trained models
            log_dir: Directory for training logs
        """
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not GYMNASIUM_AVAILABLE:
            logger.warning("Gymnasium not available. RL training will not work.")
        if not STABLE_BASELINES3_AVAILABLE:
            logger.warning(
                "Stable-Baselines3 not available. RL training will not work."
            )

        logger.info("RL trainer initialized")

    def create_trading_environment(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        max_position: float = 1.0,
    ) -> TradingEnvironment:
        """Create a trading environment.

        Args:
            data: Price data with features
            initial_balance: Initial account balance
            transaction_fee: Transaction fee as percentage
            max_position: Maximum position size

        Returns:
            Trading environment instance
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("Gymnasium is required to create trading environments")

        return TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            max_position=max_position,
        )

    def train_ppo_agent(
        self,
        env: TradingEnvironment,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train a PPO agent.

        Args:
            env: Trading environment
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            verbose: Verbosity level

        Returns:
            Dictionary with training results
        """
        if not STABLE_BASELINES3_AVAILABLE:
            raise ImportError("Stable-Baselines3 is required for PPO training")

        try:
            # Create vectorized environment
            vec_env = DummyVecEnv([lambda: env])

            # Create callback
            callback = TrainingCallback()

            # Create PPO model
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                verbose=verbose,
            )

            # Train the model
            logger.info(f"Starting PPO training for {total_timesteps} timesteps")
            model.learn(
                total_timesteps=total_timesteps, callback=callback, progress_bar=True
            )

            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"ppo_model_{timestamp}.zip"
            model.save(str(model_path))

            # Compile training results
            results = {
                "model_type": "PPO",
                "model_path": str(model_path),
                "total_timesteps": total_timesteps,
                "training_params": {
                    "learning_rate": learning_rate,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "clip_range": clip_range,
                },
                "training_metrics": {
                    "episode_rewards": callback.episode_rewards,
                    "episode_lengths": callback.episode_lengths,
                    "portfolio_values": callback.portfolio_values,
                    "final_portfolio_value": (
                        callback.portfolio_values[-1]
                        if callback.portfolio_values
                        else 0
                    ),
                    "total_return": (
                        (callback.portfolio_values[-1] - 10000) / 10000
                        if callback.portfolio_values
                        else 0
                    ),
                },
                "timestamp": timestamp,
            }

            # Save training results
            results_path = self.model_dir / f"ppo_training_results_{timestamp}.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"PPO training completed. Model saved to {model_path}")

            # Add learning curve plot after training complete
            if hasattr(callback, "episode_rewards") and callback.episode_rewards:
                self.plot_rewards(callback.episode_rewards)

            return results

        except Exception as e:
            logger.error(f"Error training PPO agent: {e}")
            return {"error": str(e)}

    def train_a2c_agent(
        self,
        env: TradingEnvironment,
        total_timesteps: int = 100000,
        learning_rate: float = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Train an A2C agent.

        Args:
            env: Trading environment
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Number of steps per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            verbose: Verbosity level

        Returns:
            Dictionary with training results
        """
        if not STABLE_BASELINES3_AVAILABLE:
            raise ImportError("Stable-Baselines3 is required for A2C training")

        try:
            # Create vectorized environment
            vec_env = DummyVecEnv([lambda: env])

            # Create callback
            callback = TrainingCallback()

            # Create A2C model
            model = A2C(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                verbose=verbose,
                tensorboard_log=str(self.log_dir),
            )

            # Train the model
            logger.info(f"Starting A2C training for {total_timesteps} timesteps")
            model.learn(
                total_timesteps=total_timesteps, callback=callback, progress_bar=True
            )

            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.model_dir / f"a2c_model_{timestamp}.zip"
            model.save(str(model_path))

            # Compile training results
            results = {
                "model_type": "A2C",
                "model_path": str(model_path),
                "total_timesteps": total_timesteps,
                "training_params": {
                    "learning_rate": learning_rate,
                    "n_steps": n_steps,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "ent_coef": ent_coef,
                    "vf_coef": vf_coef,
                    "max_grad_norm": max_grad_norm,
                },
                "training_metrics": {
                    "episode_rewards": callback.episode_rewards,
                    "episode_lengths": callback.episode_lengths,
                    "portfolio_values": callback.portfolio_values,
                    "final_portfolio_value": (
                        callback.portfolio_values[-1]
                        if callback.portfolio_values
                        else 0
                    ),
                    "total_return": (
                        (callback.portfolio_values[-1] - 10000) / 10000
                        if callback.portfolio_values
                        else 0
                    ),
                },
                "timestamp": timestamp,
            }

            # Save training results
            results_path = self.model_dir / f"a2c_training_results_{timestamp}.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"A2C training completed. Model saved to {model_path}")

            # Add learning curve plot after training complete
            if hasattr(callback, "episode_rewards") and callback.episode_rewards:
                self.plot_rewards(callback.episode_rewards)

            return results

        except Exception as e:
            logger.error(f"Error training A2C agent: {e}")
            return {"error": str(e)}

    def evaluate_agent(
        self, model_path: str, env: TradingEnvironment, n_episodes: int = 10
    ) -> Dict[str, Any]:
        """Evaluate a trained agent.

        Args:
            model_path: Path to trained model
            env: Trading environment
            n_episodes: Number of evaluation episodes

        Returns:
            Dictionary with evaluation results
        """
        if not STABLE_BASELINES3_AVAILABLE:
            raise ImportError("Stable-Baselines3 is required for agent evaluation")

        try:
            # Load the model
            if "ppo" in model_path.lower():
                model = PPO.load(model_path)
            elif "a2c" in model_path.lower():
                model = A2C.load(model_path)
            else:
                raise ValueError("Unknown model type")

            # Evaluation metrics
            episode_rewards = []
            episode_returns = []
            episode_lengths = []
            final_portfolio_values = []

            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0

                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)

                    episode_reward += reward
                    episode_length += 1

                    if done or truncated:
                        break

                episode_rewards.append(episode_reward)
                episode_returns.append(info["total_return"])
                episode_lengths.append(episode_length)
                final_portfolio_values.append(info["portfolio_value"])

            # Compile evaluation results
            results = {
                "model_path": model_path,
                "n_episodes": n_episodes,
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_return": np.mean(episode_returns),
                "std_return": np.std(episode_returns),
                "mean_length": np.mean(episode_lengths),
                "mean_final_portfolio": np.mean(final_portfolio_values),
                "episode_rewards": episode_rewards,
                "episode_returns": episode_returns,
                "episode_lengths": episode_lengths,
                "final_portfolio_values": final_portfolio_values,
            }

            logger.info(
                f"Agent evaluation completed. Mean return: {results['mean_return']:.4f}"
            )
            return results
        except Exception as e:
            logger.error(f"Error evaluating agent: {e}")
            return {"error": str(e)}

    def plot_rewards(self, reward_log: List[float]):
        """Plot learning curve from reward log."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(reward_log)
            plt.title("Learning Curve - Episode Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)

            # Save plot
            plot_path = (
                self.log_dir
                / f"learning_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Learning curve plot saved to {plot_path}")

        except ImportError:
            logger.warning("Matplotlib not available for plotting learning curve")
        except Exception as e:
            logger.error(f"Error plotting learning curve: {e}")

    def compare_agents(
        self, model_paths: List[str], env: TradingEnvironment, n_episodes: int = 10
    ) -> pd.DataFrame:
        """Compare multiple trained agents.

        Args:
            model_paths: List of model paths
            env: Trading environment
            n_episodes: Number of evaluation episodes per agent

        Returns:
            DataFrame with comparison results
        """
        try:
            results = []

            for model_path in model_paths:
                evaluation = self.evaluate_agent(model_path, env, n_episodes)
                if "error" not in evaluation:
                    results.append(
                        {
                            "model": os.path.basename(model_path),
                            "mean_reward": evaluation["mean_reward"],
                            "std_reward": evaluation["std_reward"],
                            "mean_return": evaluation["mean_return"],
                            "std_return": evaluation["std_return"],
                            "mean_final_portfolio": evaluation["mean_final_portfolio"],
                        }
                    )

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"Error comparing agents: {e}")
            return pd.DataFrame()


# Global RL trainer instance
_rl_trainer = None


def get_rl_trainer() -> RLTrainer:
    """Get the global RL trainer instance."""
    global _rl_trainer
    if _rl_trainer is None:
        _rl_trainer = RLTrainer()
    return _rl_trainer
