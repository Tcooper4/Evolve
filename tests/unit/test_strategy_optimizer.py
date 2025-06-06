import pytest
import pandas as pd
import numpy as np
import torch
from trading.models.advanced.rl.strategy_optimizer import DQNStrategyOptimizer
from tests.unit.base_test import BaseModelTest
from pathlib import Path

def make_sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    return data

def test_strategy_optimizer_initialization():
    """Test strategy optimizer initialization."""
    config = {
        'state_dim': 10,
        'action_dim': 3,
        'learning_rate': 0.001
    }
    optimizer = DQNStrategyOptimizer(config)
    assert optimizer is not None
    assert optimizer.model is not None

def test_strategy_optimizer_training():
    """Test strategy optimizer training."""
    config = {
        'state_dim': 10,
        'action_dim': 3,
        'learning_rate': 0.001
    }
    optimizer = DQNStrategyOptimizer(config)
    data = make_sample_data()
    optimizer.train(data)
    assert len(optimizer.history) > 0

def test_strategy_optimizer_prediction():
    """Test strategy optimizer prediction."""
    config = {
        'state_dim': 10,
        'action_dim': 3,
        'learning_rate': 0.001
    }
    optimizer = DQNStrategyOptimizer(config)
    data = make_sample_data()
    optimizer.train(data)
    predictions = optimizer.predict(data)
    assert 'actions' in predictions
    assert len(predictions['actions']) > 0

def test_strategy_optimizer_save_load():
    """Test strategy optimizer saving and loading."""
    import tempfile
    import os
    
    config = {
        'state_dim': 10,
        'action_dim': 3,
        'learning_rate': 0.001
    }
    
    # Create and train optimizer
    optimizer = DQNStrategyOptimizer(config)
    data = make_sample_data()
    optimizer.train(data)
    
    # Save optimizer
    save_path = tempfile.mkdtemp()
    optimizer_path = os.path.join(save_path, 'optimizer')
    optimizer.save(optimizer_path)
    
    # Create new optimizer and load
    loaded_optimizer = DQNStrategyOptimizer(config)
    loaded_optimizer.load(optimizer_path)
    
    # Compare predictions
    original_pred = optimizer.predict(data)
    loaded_pred = loaded_optimizer.predict(data)
    
    np.testing.assert_array_equal(
        original_pred['actions'],
        loaded_pred['actions']
    )
    
    # Cleanup
    import shutil
    shutil.rmtree(save_path)

class TestStrategyOptimizer(BaseModelTest):
    """Test suite for Strategy Optimizer."""
    
    @pytest.fixture
    def model_class(self):
        return DQNStrategyOptimizer
    
    @pytest.fixture
    def config(self):
        return {
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 1000,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'target_update': 10
        }
    
    @pytest.fixture
    def state_dim(self):
        return 10
    
    @pytest.fixture
    def action_dim(self):
        return 3
    
    @pytest.fixture
    def optimizer(self, state_dim, action_dim, config):
        return DQNStrategyOptimizer(state_dim, action_dim, config)
    
    def test_model_instantiation(self, optimizer, state_dim, action_dim):
        """Test model instantiation."""
        assert optimizer is not None
        assert optimizer.state_dim == state_dim
        assert optimizer.action_dim == action_dim
        assert optimizer.policy_net is not None
        assert optimizer.target_net is not None
        assert optimizer.optimizer is not None
    
    def test_model_fit(self, optimizer):
        """Test model fitting."""
        # Create sample data
        state = np.random.randn(optimizer.state_dim)
        action = np.random.randint(0, optimizer.action_dim)
        reward = 1.0
        next_state = np.random.randn(optimizer.state_dim)
        done = False
        
        # Fit the model
        loss = optimizer.fit(state, action, reward, next_state, done)
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_model_predict(self, optimizer):
        """Test model prediction."""
        # Create sample state
        state = np.random.randn(optimizer.state_dim)
        
        # Get prediction
        action = optimizer.predict(state)
        assert isinstance(action, int)
        assert 0 <= action < optimizer.action_dim
    
    def test_model_save_load(self, optimizer, tmp_path):
        """Test model save and load."""
        # Save model
        save_path = tmp_path / "model.pt"
        optimizer.save(str(save_path))
        assert save_path.exists()
        
        # Create new optimizer
        new_optimizer = DQNStrategyOptimizer(
            optimizer.state_dim,
            optimizer.action_dim,
            optimizer.config
        )
        
        # Load model
        new_optimizer.load(str(save_path))
        
        # Compare states
        assert torch.allclose(
            optimizer.policy_net.state_dict()['0.weight'],
            new_optimizer.policy_net.state_dict()['0.weight']
        )
    
    def test_model_invalid_input(self, optimizer):
        """Test model with invalid input."""
        # Test with invalid state dimension
        with pytest.raises(RuntimeError):
            invalid_state = np.random.randn(optimizer.state_dim + 1)
            optimizer.predict(invalid_state)
        
        # Test with invalid action
        with pytest.raises(ValueError):
            invalid_action = optimizer.action_dim + 1
            optimizer.fit(
                np.random.randn(optimizer.state_dim),
                invalid_action,
                1.0,
                np.random.randn(optimizer.state_dim),
                False
            )
    
    def test_model_memory_management(self, optimizer):
        """Test memory management."""
        # Fill memory
        for _ in range(optimizer.memory.maxlen + 1):
            optimizer.fit(
                np.random.randn(optimizer.state_dim),
                np.random.randint(0, optimizer.action_dim),
                1.0,
                np.random.randn(optimizer.state_dim),
                False
            )
        
        # Check memory size
        assert len(optimizer.memory) == optimizer.memory.maxlen
    
    def test_model_learning_rate_scheduler(self, optimizer):
        """Test learning rate scheduler."""
        # Get initial learning rate
        initial_lr = optimizer.optimizer.param_groups[0]['lr']
        
        # Train for a few steps
        for _ in range(10):
            optimizer.fit(
                np.random.randn(optimizer.state_dim),
                np.random.randint(0, optimizer.action_dim),
                1.0,
                np.random.randn(optimizer.state_dim),
                False
            )
        
        # Check if learning rate changed
        assert optimizer.optimizer.param_groups[0]['lr'] == initial_lr
    
    def test_strategy_optimizer_specific_features(self, optimizer):
        """Test strategy optimizer specific features."""
        # Test epsilon decay
        initial_epsilon = optimizer.epsilon
        
        # Train for a few steps
        for _ in range(10):
            optimizer.fit(
                np.random.randn(optimizer.state_dim),
                np.random.randint(0, optimizer.action_dim),
                1.0,
                np.random.randn(optimizer.state_dim),
                False
            )
        
        # Check if epsilon decreased
        assert optimizer.epsilon < initial_epsilon
    
    def test_strategy_optimizer_action_selection(self, optimizer):
        """Test action selection."""
        # Test random action selection
        state = np.random.randn(optimizer.state_dim)
        actions = set()
        
        # Get multiple predictions
        for _ in range(100):
            action = optimizer.predict(state)
            actions.add(action)
        
        # Check if we get different actions (due to epsilon-greedy)
        assert len(actions) > 1
    
    def test_strategy_optimizer_memory_management(self, optimizer):
        """Test memory management."""
        # Test state preparation
        state = np.random.randn(optimizer.state_dim)
        prepared_state = optimizer._prepare_state(state)
        
        assert isinstance(prepared_state, torch.Tensor)
        assert prepared_state.shape == (optimizer.state_dim,)
    
    def test_strategy_optimizer_training_step(self, optimizer):
        """Test training step."""
        # Fill memory
        for _ in range(optimizer.batch_size):
            optimizer.fit(
                np.random.randn(optimizer.state_dim),
                np.random.randint(0, optimizer.action_dim),
                1.0,
                np.random.randn(optimizer.state_dim),
                False
            )
        
        # Perform training step
        loss = optimizer._train_step()
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_strategy_optimizer_target_update(self, optimizer):
        """Test target network update."""
        # Get initial target network weights
        initial_weights = optimizer.target_net.state_dict()['0.weight'].clone()
        
        # Train for target_update steps
        for _ in range(optimizer.target_update):
            optimizer.fit(
                np.random.randn(optimizer.state_dim),
                np.random.randint(0, optimizer.action_dim),
                1.0,
                np.random.randn(optimizer.state_dim),
                False
            )
        
        # Check if target network was updated
        assert not torch.allclose(
            initial_weights,
            optimizer.target_net.state_dict()['0.weight']
        )
    
    def test_strategy_optimizer_reward_shaping(self, optimizer):
        """Test reward shaping."""
        # Test with different rewards
        rewards = [-1.0, 0.0, 1.0]
        losses = []
        
        for reward in rewards:
            loss = optimizer.fit(
                np.random.randn(optimizer.state_dim),
                np.random.randint(0, optimizer.action_dim),
                reward,
                np.random.randn(optimizer.state_dim),
                False
            )
            losses.append(loss)
        
        # Check if losses are different for different rewards
        assert len(set(losses)) > 1 