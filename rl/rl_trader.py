"""Reinforcement Learning Trader for Evolve Trading Platform.

This module implements a PPO-based RL trader using stable-baselines3
with a custom Gym-compatible trading environment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
warnings.filterwarnings('ignore')

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
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    PPO = DQN = A2C = DummyVecEnv = BaseCallback = None

logger = logging.getLogger(__name__)

class TradingEnvironment:
    """Custom trading environment for reinforcement learning."""
    
    def __init__(self, data: pd.DataFrame, 
                 initial_balance: float = 100000,
                 transaction_fee: float = 0.001,
                 max_position: float = 0.2,
                 reward_function: str = 'sharpe_ratio'):
        """Initialize trading environment."""
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.max_position = max_position
        self.reward_function = reward_function
        
        # Define action and observation spaces
        if GYMNASIUM_AVAILABLE:
            self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(15,), dtype=np.float32
            )
        else:
            self.action_space = None
            self.observation_space = None
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.total_value = self.initial_balance
        self.returns = []
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action in environment."""
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
        """Execute trading action."""
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        
        reward = 0
        
        if action == 0:  # Buy
            if self.balance > 0:
                position_value = self.balance * self.max_position
                shares_to_buy = position_value / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                if cost <= self.balance:
                    self.shares += shares_to_buy
                    self.balance -= cost
                    
                    # Record trade
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })
                    
                    # Calculate reward
                    price_change = (next_price - current_price) / current_price
                    reward = price_change * shares_to_buy * current_price
        
        elif action == 1:  # Sell
            if self.shares > 0:
                proceeds = self.shares * current_price * (1 - self.transaction_fee)
                self.balance += proceeds
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': self.shares,
                    'price': current_price,
                    'proceeds': proceeds
                })
                
                # Calculate reward
                price_change = (current_price - next_price) / current_price
                reward = price_change * self.shares * current_price
                self.shares = 0
        
        else:  # Hold
            reward = -0.0001  # Small penalty for holding
        
        return reward
    
    def _update_portfolio_value(self):
        """Update total portfolio value."""
        current_price = self.data.iloc[self.current_step]['Close']
        self.total_value = self.balance + self.shares * current_price
        self.portfolio_values.append(self.total_value)
        
        if self.current_step > 0:
            prev_value = self.portfolio_values[-2]
            return_val = (self.total_value - prev_value) / prev_value
            self.returns.append(return_val)
        else:
            self.returns.append(0)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self.data):
            return np.zeros(15, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        current_price = row['Close']
        
        # Price features
        price_features = [
            row['Open'] / current_price - 1,
            row['High'] / current_price - 1,
            row['Low'] / current_price - 1,
            row['Volume'] / self.data['Volume'].mean() - 1,
        ]
        
        # Technical indicators
        if self.current_step >= 20:
            # Moving averages
            ma_5 = self.data.iloc[self.current_step-5:self.current_step]['Close'].mean()
            ma_20 = self.data.iloc[self.current_step-20:self.current_step]['Close'].mean()
            ma_features = [
                current_price / ma_5 - 1,
                current_price / ma_20 - 1,
                ma_5 / ma_20 - 1
            ]
        else:
            ma_features = [0, 0, 0]
        
        # Portfolio features
        portfolio_features = [
            self.balance / self.initial_balance - 1,
            self.shares * current_price / self.initial_balance,
            self.total_value / self.initial_balance - 1,
            len(self.returns) > 0 and self.returns[-1] or 0,
            self.current_step / len(self.data),
            1.0 if self.shares > 0 else 0.0
        ]
        
        # Risk features
        if len(self.returns) >= 10:
            recent_returns = self.returns[-10:]
            risk_features = [
                np.std(recent_returns),
                np.mean(recent_returns),
                min(recent_returns),
                max(recent_returns)
            ]
        else:
            risk_features = [0, 0, 0, 0]
        
        features = price_features + ma_features + portfolio_features + risk_features
        return np.array(features, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get environment info."""
        return {
            'balance': self.balance,
            'shares': self.shares,
            'total_value': self.total_value,
            'step': self.current_step,
            'returns': self.returns.copy() if self.returns else [],
            'trades': self.trades.copy(),
            'portfolio_values': self.portfolio_values.copy()
        }
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.returns:
            return {}
        
        returns_series = pd.Series(self.returns)
        
        metrics = {
            'total_return': (self.total_value / self.initial_balance) - 1,
            'sharpe_ratio': returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0,
            'volatility': returns_series.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(),
            'win_rate': len([r for r in self.returns if r > 0]) / len(self.returns) if self.returns else 0,
            'num_trades': len(self.trades)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.portfolio_values:
            return 0
        
        peak = self.portfolio_values[0]
        max_dd = 0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd

class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
    
    def _on_step(self) -> bool:
        """Called after each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            if any(dones):
                # Calculate metrics for completed episodes
                pass

class RLTrader:
    """Main RL trader class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RL trader."""
        self.config = config or {}
        self.model = None
        self.env = None
        self.training_history = []
        self.last_training_metrics = {}
        
        if not GYMNASIUM_AVAILABLE:
            logger.error("Gymnasium not available")
        if not STABLE_BASELINES3_AVAILABLE:
            logger.error("Stable-baselines3 not available")
    
    def create_environment(self, data: pd.DataFrame) -> TradingEnvironment:
        """Create trading environment."""
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("Gymnasium not available")
        
        initial_balance = self.config.get('initial_balance', 100000)
        transaction_fee = self.config.get('transaction_fee', 0.001)
        max_position = self.config.get('max_position', 0.2)
        reward_function = self.config.get('reward_function', 'sharpe_ratio')
        
        self.env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            max_position=max_position,
            reward_function=reward_function
        )
        
        return self.env
    
    def train_model(self, data: pd.DataFrame, 
                   algorithm: str = 'PPO',
                   total_timesteps: int = 10000,
                   model_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Train RL model."""
        if not STABLE_BASELINES3_AVAILABLE:
            logger.error("Stable-baselines3 not available")
            return {
                'success': False,
                'error': 'Stable-baselines3 not available',
                'metrics': {}
            }
        
        try:
            # Create environment
            env = self.create_environment(data)
            vec_env = DummyVecEnv([lambda: env])
            
            # Model parameters
            model_params = model_params or {
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01
            }
            
            # Create model
            if algorithm == 'PPO':
                self.model = PPO('MlpPolicy', vec_env, verbose=1, **model_params)
            elif algorithm == 'DQN':
                self.model = DQN('MlpPolicy', vec_env, verbose=1)
            elif algorithm == 'A2C':
                self.model = A2C('MlpPolicy', vec_env, verbose=1)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Train model
            callback = TrainingCallback()
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
            
            # Calculate training metrics
            training_metrics = self._calculate_training_metrics(env)
            self.last_training_metrics = training_metrics
            
            logger.info(f"Trained {algorithm} model for {total_timesteps} timesteps")
            return {
                'success': True,
                'algorithm': algorithm,
                'timesteps': total_timesteps,
                'metrics': training_metrics
            }
            
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def _calculate_training_metrics(self, env: TradingEnvironment) -> Dict[str, float]:
        """Calculate training metrics."""
        try:
            metrics = env.calculate_metrics()
            return {
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'volatility': metrics.get('volatility', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'num_trades': metrics.get('num_trades', 0)
            }
        except Exception as e:
            logger.error(f"Error calculating training metrics: {e}")
            return {}
    
    def predict_action(self, observation: np.ndarray) -> Tuple[int, Dict]:
        """Predict action for given observation."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return action, {'confidence': 0.8, 'model_available': True}
        except Exception as e:
            logger.error(f"Error predicting action: {e}")
            return 2, {'confidence': 0.0, 'model_available': False, 'error': str(e)}  # Default to hold
    
    def evaluate_model(self, data: pd.DataFrame, 
                      num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained model."""
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not trained',
                'metrics': {}
            }
        
        try:
            env = self.create_environment(data)
            total_rewards = []
            total_returns = []
            final_values = []
            all_metrics = []
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0
                
                while True:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if done or truncated:
                        break
                
                total_rewards.append(episode_reward)
                total_returns.append(info['total_value'] / env.initial_balance - 1)
                final_values.append(info['total_value'])
                
                # Calculate episode metrics
                metrics = env.calculate_metrics()
                all_metrics.append(metrics)
            
            # Aggregate metrics
            avg_metrics = {}
            if all_metrics:
                for key in all_metrics[0].keys():
                    values = [m[key] for m in all_metrics if key in m]
                    if values:
                        avg_metrics[f'avg_{key}'] = np.mean(values)
                        avg_metrics[f'std_{key}'] = np.std(values)
            
            evaluation_metrics = {
                'mean_reward': np.mean(total_rewards),
                'std_reward': np.std(total_rewards),
                'mean_return': np.mean(total_returns),
                'std_return': np.std(total_returns),
                'mean_final_value': np.mean(final_values),
                'sharpe_ratio': np.mean(total_returns) / np.std(total_returns) if np.std(total_returns) > 0 else 0
            }
            
            evaluation_metrics.update(avg_metrics)
            
            logger.info(f"RL model evaluation: {evaluation_metrics}")
            return {
                'success': True,
                'episodes': num_episodes,
                'metrics': evaluation_metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating RL model: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def save_model(self, filepath: str) -> Dict[str, Any]:
        """Save trained model."""
        if self.model is None:
            logger.error("No model to save")
            return {
                'success': False,
                'error': 'No model to save'
            }
        
        try:
            self.model.save(filepath)
            logger.info(f"Saved model to {filepath}")
            return {
                'success': True,
                'filepath': filepath,
                'model_type': type(self.model).__name__
            }
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_model(self, filepath: str, algorithm: str = 'PPO') -> Dict[str, Any]:
        """Load trained model."""
        if not STABLE_BASELINES3_AVAILABLE:
            logger.error("Stable-baselines3 not available")
            return {
                'success': False,
                'error': 'Stable-baselines3 not available'
            }
        
        try:
            if algorithm == 'PPO':
                self.model = PPO.load(filepath)
            elif algorithm == 'DQN':
                self.model = DQN.load(filepath)
            elif algorithm == 'A2C':
                self.model = A2C.load(filepath)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            logger.info(f"Loaded model from {filepath}")
            return {
                'success': True,
                'filepath': filepath,
                'algorithm': algorithm,
                'model_type': type(self.model).__name__
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            'model_available': self.model is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'last_training_metrics': self.last_training_metrics,
            'gymnasium_available': GYMNASIUM_AVAILABLE,
            'stable_baselines3_available': STABLE_BASELINES3_AVAILABLE
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        return {
            'overall_status': 'healthy' if (self.model is not None and GYMNASIUM_AVAILABLE and STABLE_BASELINES3_AVAILABLE) else 'degraded',
            'model_available': self.model is not None,
            'gymnasium_available': GYMNASIUM_AVAILABLE,
            'stable_baselines3_available': STABLE_BASELINES3_AVAILABLE,
            'last_training_success': self.last_training_metrics.get('total_return', 0) > 0 if self.last_training_metrics else False
        }

# Global RL trader instance
rl_trader = RLTrader()

def get_rl_trader() -> RLTrader:
    """Get the global RL trader instance."""
    return rl_trader
