import pytest
import os
import json
import yaml
from pathlib import Path
from trading.config.configuration import (
    ConfigManager,
    ModelConfig,
    DataConfig,
    TrainingConfig
)

class TestConfiguration:
    """Test suite for configuration utilities."""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager()
    
    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            model_type='transformer',
            d_model=64,
            nhead=8,
            num_layers=4,
            dropout=0.1,
            batch_size=32,
            learning_rate=0.001
        )
    
    @pytest.fixture
    def data_config(self):
        return DataConfig(
            data_source='yfinance',
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2024-01-01',
            end_date='2024-12-31',
            features=['Open', 'High', 'Low', 'Close', 'Volume']
        )
    
    @pytest.fixture
    def training_config(self):
        return TrainingConfig(
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            early_stopping_patience=10,
            learning_rate_scheduler='reduce_on_plateau'
        )
    
    def test_config_manager_initialization(self, config_manager):
        """Test config manager initialization."""
        assert config_manager is not None
        assert config_manager.config_dir is not None
    
    def test_config_manager_save_load_json(self, config_manager, model_config, tmp_path):
        """Test saving and loading JSON config."""
        # Save config
        config_path = tmp_path / 'model_config.json'
        config_manager.save_config(model_config, str(config_path))
        
        assert config_path.exists()
        
        # Load config
        loaded_config = config_manager.load_config(str(config_path))
        assert isinstance(loaded_config, ModelConfig)
        assert loaded_config.model_type == model_config.model_type
        assert loaded_config.d_model == model_config.d_model
    
    def test_config_manager_save_load_yaml(self, config_manager, data_config, tmp_path):
        """Test saving and loading YAML config."""
        # Save config
        config_path = tmp_path / 'data_config.yaml'
        config_manager.save_config(data_config, str(config_path))
        
        assert config_path.exists()
        
        # Load config
        loaded_config = config_manager.load_config(str(config_path))
        assert isinstance(loaded_config, DataConfig)
        assert loaded_config.data_source == data_config.data_source
        assert loaded_config.symbols == data_config.symbols
    
    def test_model_config_validation(self, model_config):
        """Test model config validation."""
        # Test valid config
        assert model_config.validate()
        
        # Test invalid config
        invalid_config = ModelConfig(
            model_type='invalid_type',
            d_model=-64,  # Invalid value
            nhead=0,  # Invalid value
            num_layers=-4,  # Invalid value
            dropout=1.5,  # Invalid value
            batch_size=0,  # Invalid value
            learning_rate=-0.001  # Invalid value
        )
        
        with pytest.raises(ValueError):
            invalid_config.validate()
    
    def test_data_config_validation(self, data_config):
        """Test data config validation."""
        # Test valid config
        assert data_config.validate()
        
        # Test invalid config
        invalid_config = DataConfig(
            data_source='invalid_source',
            symbols=[],  # Empty list
            start_date='2024-12-31',  # Invalid date order
            end_date='2024-01-01',
            features=[]  # Empty list
        )
        
        with pytest.raises(ValueError):
            invalid_config.validate()
    
    def test_training_config_validation(self, training_config):
        """Test training config validation."""
        # Test valid config
        assert training_config.validate()
        
        # Test invalid config
        invalid_config = TrainingConfig(
            epochs=0,  # Invalid value
            batch_size=0,  # Invalid value
            validation_split=1.5,  # Invalid value
            early_stopping_patience=-10,  # Invalid value
            learning_rate_scheduler='invalid_scheduler'  # Invalid value
        )
        
        with pytest.raises(ValueError):
            invalid_config.validate()
    
    def test_config_serialization(self, model_config):
        """Test config serialization."""
        # Test to_dict
        config_dict = model_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['model_type'] == model_config.model_type
        
        # Test from_dict
        new_config = ModelConfig.from_dict(config_dict)
        assert isinstance(new_config, ModelConfig)
        assert new_config.model_type == model_config.model_type
    
    def test_config_environment_variables(self, config_manager, tmp_path):
        """Test config environment variables."""
        # Set environment variables
        os.environ['MODEL_TYPE'] = 'transformer'
        os.environ['D_MODEL'] = '64'
        os.environ['NHEAD'] = '8'
        
        # Create config from environment
        config = config_manager.create_config_from_env()
        assert config['model_type'] == 'transformer'
        assert config['d_model'] == 64
        assert config['nhead'] == 8
    
    def test_config_merging(self, model_config, data_config):
        """Test config merging."""
        # Merge configs
        merged_config = model_config.merge(data_config)
        assert isinstance(merged_config, dict)
        assert 'model_type' in merged_config
        assert 'data_source' in merged_config
    
    def test_config_defaults(self):
        """Test config defaults."""
        # Test model config defaults
        default_model_config = ModelConfig()
        assert default_model_config.model_type == 'transformer'
        assert default_model_config.d_model == 512
        assert default_model_config.nhead == 8
        
        # Test data config defaults
        default_data_config = DataConfig()
        assert default_data_config.data_source == 'yfinance'
        assert default_data_config.features == ['Close']
        
        # Test training config defaults
        default_training_config = TrainingConfig()
        assert default_training_config.epochs == 100
        assert default_training_config.batch_size == 32
    
    def test_config_file_operations(self, config_manager, model_config, tmp_path):
        """Test config file operations."""
        # Test file creation
        config_path = tmp_path / 'test_config.json'
        config_manager.save_config(model_config, str(config_path))
        
        # Test file reading
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        assert config_data['model_type'] == model_config.model_type
        
        # Test file updating
        model_config.model_type = 'lstm'
        config_manager.save_config(model_config, str(config_path))
        
        with open(config_path, 'r') as f:
            updated_data = json.load(f)
        assert updated_data['model_type'] == 'lstm'
        
        # Test file deletion
        config_manager.delete_config(str(config_path))
        assert not config_path.exists() 