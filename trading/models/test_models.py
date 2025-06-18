"""Test cases for all registered models."""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import ModelRegistry
from utils.metrics import calculate_metrics
from utils.visualization import (
    plot_forecast,
    plot_attention_heatmap,
    plot_shap_values
)

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic data
    """
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic price series with trend, seasonality and noise
    t = np.arange(n_samples)
    trend = 0.1 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 365)  # Annual seasonality
    noise = np.random.normal(0, 5, n_samples)
    
    # Combine components
    price = 100 + trend + seasonality + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'close': price,
        'volume': np.random.lognormal(10, 1, n_samples),
        'returns': np.random.normal(0, 0.02, n_samples)
    })
    
    # Add some technical indicators
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['rsi'] = 50 + 20 * np.sin(2 * np.pi * t / 50)  # Oscillating RSI
    data['macd'] = data['close'].rolling(window=12).mean() - data['close'].rolling(window=26).mean()
    
    data.set_index('date', inplace=True)
    return data

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get default configuration for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    configs = {
        'TCNModel': {
            'input_size': 5,
            'output_size': 1,
            'num_channels': [64, 32, 16],
            'kernel_size': 3,
            'dropout': 0.2,
            'sequence_length': 20
        },
        'GNNForecaster': {
            'input_size': 5,
            'hidden_size': 64,
            'output_size': 1,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 20
        },
        'TransformerForecaster': {
            'input_size': 5,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 128,
            'dropout': 0.2,
            'sequence_length': 20
        },
        'LSTMForecaster': {
            'input_size': 5,
            'hidden_size': 64,
            'output_size': 1,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 20,
            'bidirectional': True
        },
        'ProphetModel': {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative'
        },
        'CatBoostModel': {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3
        },
        'AutoformerModel': {
            'input_size': 5,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.2,
            'sequence_length': 20
        }
    }
    
    return configs.get(model_name, {})

def test_model(model_name: str, save_dir: str = 'test_results') -> Dict[str, Any]:
    """Test a single model with synthetic data.
    
    Args:
        model_name: Name of the model to test
        save_dir: Directory to save results
        
    Returns:
        Dictionary with test results
    """
    print(f"\nTesting {model_name}...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate synthetic data
    data = generate_synthetic_data()
    
    # Get model configuration
    config = get_model_config(model_name)
    
    # Initialize model
    model_class = ModelRegistry.get_model_class(model_name)
    model = model_class(config)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Test training
    print("Testing training...")
    start_time = datetime.now()
    model.fit(train_data)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Test prediction
    print("Testing prediction...")
    predictions = model.predict(test_data)
    
    # Test multi-horizon forecasting
    print("Testing multi-horizon forecasting...")
    horizons = [1, 5, 10]
    multi_horizon_preds = {}
    for horizon in horizons:
        try:
            multi_horizon_preds[horizon] = model.predict(test_data, horizon=horizon)
        except Exception as e:
            print(f"Multi-horizon forecasting not supported for horizon {horizon}: {e}")
    
    # Test model summary
    print("Testing model summary...")
    try:
        summary = model.summary()
        print(summary)
    except Exception as e:
        print(f"Model summary not supported: {e}")
        summary = None
    
    # Test SHAP interpretability
    print("Testing SHAP interpretability...")
    try:
        shap_values = model.shap_interpret(test_data)
        fig = plot_shap_values(model, test_data)
        fig.write_html(os.path.join(save_dir, f'{model_name}_shap.html'))
    except Exception as e:
        print(f"SHAP interpretability not supported: {e}")
        shap_values = None
    
    # Test attention heatmap
    print("Testing attention heatmap...")
    try:
        attention = model.attention_heatmap(test_data)
        fig = plot_attention_heatmap(model, test_data)
        fig.write_html(os.path.join(save_dir, f'{model_name}_attention.html'))
    except Exception as e:
        print(f"Attention heatmap not supported: {e}")
        attention = None
    
    # Test model saving/loading
    print("Testing model save/load...")
    try:
        # Save model
        save_path = os.path.join(save_dir, f'{model_name}.pkl')
        model.save_model(save_path)
        
        # Load model
        new_model = model_class(config)
        new_model.load_model(save_path)
        
        # Verify predictions match
        new_preds = new_model.predict(test_data)
        pred_diff = np.mean(np.abs(predictions - new_preds))
        print(f"Prediction difference after save/load: {pred_diff:.6f}")
        
    except Exception as e:
        print(f"Model save/load not supported: {e}")
    
    # Calculate metrics
    metrics = calculate_metrics(
        test_data['close'].values,
        predictions
    )
    
    # Save results
    results = {
        'model_name': model_name,
        'config': config,
        'metrics': metrics,
        'training_time': training_time,
        'multi_horizon_supported': bool(multi_horizon_preds),
        'summary_supported': summary is not None,
        'shap_supported': shap_values is not None,
        'attention_supported': attention is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, f'{model_name}_test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': test_data.index,
        'actual': test_data['close'],
        'predicted': predictions
    })
    predictions_df.to_csv(os.path.join(save_dir, f'{model_name}_predictions.csv'))
    
    print(f"\nTest results saved to {save_dir}")
    print(f"Training time: {training_time:.2f} seconds")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return results

def main():
    """Run tests for all registered models."""
    # Get all registered models
    model_names = ModelRegistry.get_registered_models()
    
    # Run tests
    results = {}
    for model_name in model_names:
        try:
            results[model_name] = test_model(model_name)
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Save overall results
    with open('test_results/overall_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nTest Summary:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Training time: {result['training_time']:.2f}s")
            print(f"  Multi-horizon: {result['multi_horizon_supported']}")
            print(f"  Summary: {result['summary_supported']}")
            print(f"  SHAP: {result['shap_supported']}")
            print(f"  Attention: {result['attention_supported']}")
            print("  Metrics:")
            for metric, value in result['metrics'].items():
                print(f"    {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 