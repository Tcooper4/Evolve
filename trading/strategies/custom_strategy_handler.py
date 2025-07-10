"""
Custom Strategy Handler

This module provides functionality for users to define and run custom strategies
through code injection or modular config YAML files.
"""

import yaml
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import os
from pathlib import Path
import importlib.util
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CustomStrategy:
    """Custom strategy definition."""
    name: str
    description: str
    code: str
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

class CustomStrategyHandler:
    """Handler for custom strategy creation and execution."""
    
    def __init__(self, config_dir: str = "config/custom_strategies"):
        """Initialize custom strategy handler.
        
        Args:
            config_dir: Directory for custom strategy configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.strategies: Dict[str, CustomStrategy] = {}
        self.load_strategies()
    
    def load_strategies(self):
        """Load all custom strategies from configuration files."""
        try:
            # Load YAML configurations
            for yaml_file in self.config_dir.glob("*.yaml"):
                self._load_yaml_strategy(yaml_file)
            
            # Load JSON configurations
            for json_file in self.config_dir.glob("*.json"):
                self._load_json_strategy(json_file)
            
            logger.info(f"Loaded {len(self.strategies)} custom strategies")
            
        except Exception as e:
            logger.error(f"Error loading custom strategies: {e}")
    
    def _load_yaml_strategy(self, file_path: Path):
        """Load strategy from YAML file."""
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            strategy = CustomStrategy(
                name=config.get('name', file_path.stem),
                description=config.get('description', ''),
                code=config.get('code', ''),
                config=config.get('parameters', {}),
                created_at=datetime.fromisoformat(config.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(config.get('updated_at', datetime.now().isoformat())),
                is_active=config.get('is_active', True)
            )
            
            self.strategies[strategy.name] = strategy
            
        except Exception as e:
            logger.error(f"Error loading YAML strategy from {file_path}: {e}")
    
    def _load_json_strategy(self, file_path: Path):
        """Load strategy from JSON file."""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            strategy = CustomStrategy(
                name=config.get('name', file_path.stem),
                description=config.get('description', ''),
                code=config.get('code', ''),
                config=config.get('parameters', {}),
                created_at=datetime.fromisoformat(config.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(config.get('updated_at', datetime.now().isoformat())),
                is_active=config.get('is_active', True)
            )
            
            self.strategies[strategy.name] = strategy
            
        except Exception as e:
            logger.error(f"Error loading JSON strategy from {file_path}: {e}")
    
    def create_strategy_from_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Create a custom strategy from YAML content.
        
        Args:
            yaml_content: YAML string containing strategy definition
            
        Returns:
            Dictionary with creation result
        """
        try:
            config = yaml.safe_load(yaml_content)
            
            if not config.get('name'):
                return {'success': False, 'error': 'Strategy name is required'}
            
            strategy = CustomStrategy(
                name=config['name'],
                description=config.get('description', ''),
                code=config.get('code', ''),
                config=config.get('parameters', {}),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_active=config.get('is_active', True)
            )
            
            # Validate strategy
            validation_result = self._validate_strategy(strategy)
            if not validation_result['success']:
                return validation_result
            
            # Save strategy
            self.strategies[strategy.name] = strategy
            self._save_strategy(strategy)
            
            return {
                'success': True,
                'message': f'Strategy "{strategy.name}" created successfully',
                'strategy_name': strategy.name
            }
            
        except Exception as e:
            logger.error(f"Error creating strategy from YAML: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_strategy_from_code(self, name: str, code: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a custom strategy from Python code.
        
        Args:
            name: Strategy name
            code: Python code for the strategy
            parameters: Strategy parameters
            
        Returns:
            Dictionary with creation result
        """
        try:
            strategy = CustomStrategy(
                name=name,
                description=f'Custom strategy: {name}',
                code=code,
                config=parameters or {},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_active=True
            )
            
            # Validate strategy
            validation_result = self._validate_strategy(strategy)
            if not validation_result['success']:
                return validation_result
            
            # Save strategy
            self.strategies[strategy.name] = strategy
            self._save_strategy(strategy)
            
            return {
                'success': True,
                'message': f'Strategy "{strategy.name}" created successfully',
                'strategy_name': strategy.name
            }
            
        except Exception as e:
            logger.error(f"Error creating strategy from code: {e}")
            return {'success': False, 'error': str(e)}
    
    def _validate_strategy(self, strategy: CustomStrategy) -> Dict[str, Any]:
        """Validate a custom strategy.
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            Validation result
        """
        try:
            # Check if strategy name already exists
            if strategy.name in self.strategies:
                return {'success': False, 'error': f'Strategy "{strategy.name}" already exists'}
            
            # Validate code if provided
            if strategy.code:
                # Try to compile the code
                compile(strategy.code, '<string>', 'exec')
                
                # Check for required functions
                if 'generate_signals' not in strategy.code:
                    return {'success': False, 'error': 'Strategy code must contain generate_signals function'}
            
            return {'success': True, 'message': 'Strategy validation passed'}
            
        except SyntaxError as e:
            return {'success': False, 'error': f'Syntax error in strategy code: {e}'}
        except Exception as e:
            return {'success': False, 'error': f'Validation error: {e}'}
    
    def _save_strategy(self, strategy: CustomStrategy):
        """Save strategy to file."""
        try:
            config = {
                'name': strategy.name,
                'description': strategy.description,
                'code': strategy.code,
                'parameters': strategy.config,
                'created_at': strategy.created_at.isoformat(),
                'updated_at': strategy.updated_at.isoformat(),
                'is_active': strategy.is_active
            }
            
            file_path = self.config_dir / f"{strategy.name}.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Strategy {strategy.name} saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving strategy {strategy.name}: {e}")
    
    def execute_strategy(self, strategy_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute a custom strategy.
        
        Args:
            strategy_name: Name of the strategy to execute
            data: Market data
            
        Returns:
            Strategy execution result
        """
        try:
            if strategy_name not in self.strategies:
                return {'success': False, 'error': f'Strategy "{strategy_name}" not found'}
            
            strategy = self.strategies[strategy_name]
            if not strategy.is_active:
                return {'success': False, 'error': f'Strategy "{strategy_name}" is not active'}
            
            # Execute strategy code
            if strategy.code:
                result = self._execute_strategy_code(strategy, data)
            else:
                result = self._execute_strategy_config(strategy, data)
            
            return {
                'success': True,
                'strategy_name': strategy_name,
                'result': result,
                'execution_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_strategy_code(self, strategy: CustomStrategy, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute strategy using Python code."""
        try:
            # Create a new module for the strategy
            spec = importlib.util.spec_from_loader(strategy.name, loader=None)
            module = importlib.util.module_from_spec(spec)
            
            # Execute the code in the module
            exec(strategy.code, module.__dict__)
            
            # Call the generate_signals function
            if hasattr(module, 'generate_signals'):
                signals = module.generate_signals(data, **strategy.config)
                return {
                    'signals': signals,
                    'execution_method': 'code',
                    'parameters_used': strategy.config
                }
            else:
                raise ValueError("Strategy code must contain generate_signals function")
                
        except Exception as e:
            logger.error(f"Error executing strategy code: {e}")
            raise
    
    def _execute_strategy_config(self, strategy: CustomStrategy, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute strategy using configuration parameters."""
        try:
            # Default strategy implementation based on parameters
            config = strategy.config
            
            # Extract parameters
            indicator_type = config.get('indicator_type', 'sma')
            period = config.get('period', 20)
            threshold = config.get('threshold', 0.5)
            
            # Generate signals based on indicator type
            if indicator_type == 'sma':
                signals = self._generate_sma_signals(data, period, threshold)
            elif indicator_type == 'rsi':
                signals = self._generate_rsi_signals(data, period, threshold)
            elif indicator_type == 'bollinger':
                signals = self._generate_bollinger_signals(data, period, threshold)
            else:
                signals = pd.Series(0, index=data.index)
            
            return {
                'signals': signals,
                'execution_method': 'config',
                'parameters_used': config
            }
            
        except Exception as e:
            logger.error(f"Error executing strategy config: {e}")
            raise
    
    def _generate_sma_signals(self, data: pd.DataFrame, period: int, threshold: float) -> pd.Series:
        """Generate SMA-based signals."""
        # Add check to ensure df contains 'close' and no NaNs before calculations
        if 'close' not in data.columns or data['close'].isna().all():
            logger.warning("SMA signals: Missing 'close' column or all NaN values")
            return pd.DataFrame(index=data.index)
        
        # Handle NaN values in close column
        if data['close'].isna().any():
            logger.warning("SMA signals: NaN values found in close column, filling with forward fill")
            data = data.copy()
            data['close'] = data['close'].fillna(method='ffill').fillna(method='bfill')
        
        sma = data['close'].rolling(period).mean()
        signals = pd.Series(0, index=data.index)
        signals[data['close'] > sma * (1 + threshold)] = 1
        signals[data['close'] < sma * (1 - threshold)] = -1
        return signals
    
    def _generate_rsi_signals(self, data: pd.DataFrame, period: int, threshold: float) -> pd.Series:
        """Generate RSI-based signals."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < 30] = 1  # Oversold
        signals[rsi > 70] = -1  # Overbought
        return signals
    
    def _generate_bollinger_signals(self, data: pd.DataFrame, period: int, threshold: float) -> pd.Series:
        """Generate Bollinger Bands signals."""
        sma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        signals = pd.Series(0, index=data.index)
        signals[data['close'] < lower_band] = 1  # Buy signal
        signals[data['close'] > upper_band] = -1  # Sell signal
        return signals
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all custom strategies."""
        return [
            {
                'name': strategy.name,
                'description': strategy.description,
                'is_active': strategy.is_active,
                'created_at': strategy.created_at.isoformat(),
                'updated_at': strategy.updated_at.isoformat()
            }
            for strategy in self.strategies.values()
        ]
    
    def delete_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Delete a custom strategy."""
        try:
            if strategy_name not in self.strategies:
                return {'success': False, 'error': f'Strategy "{strategy_name}" not found'}
            
            # Remove from memory
            del self.strategies[strategy_name]
            
            # Remove file
            file_path = self.config_dir / f"{strategy_name}.yaml"
            if file_path.exists():
                file_path.unlink()
            
            return {
                'success': True,
                'message': f'Strategy "{strategy_name}" deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error deleting strategy {strategy_name}: {e}")
            return {'success': False, 'error': str(e)}

# Global instance
custom_strategy_handler = CustomStrategyHandler()

def get_custom_strategy_handler() -> CustomStrategyHandler:
    """Get the global custom strategy handler instance."""
    return custom_strategy_handler 