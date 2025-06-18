"""Sandbox environment for testing models with synthetic or real data."""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import ModelRegistry
from utils.metrics import calculate_metrics
from utils.visualization import (
    plot_forecast,
    plot_attention_heatmap,
    plot_shap_values,
    plot_model_components,
    plot_model_comparison,
    plot_performance_over_time,
    plot_backtest_results
)

def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load data from file or generate synthetic data.
    
    Args:
        data_path: Path to data file (optional)
        
    Returns:
        DataFrame with data
    """
    if data_path and os.path.exists(data_path):
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    return generate_synthetic_data()

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

def run_sandbox(model_name: str, data_path: Optional[str] = None, save_dir: str = 'sandbox_results',
                show_plots: bool = True, save_plots: bool = True) -> Dict[str, Any]:
    """Run sandbox environment for a model.
    
    Args:
        model_name: Name of the model to test
        data_path: Path to data file (optional)
        save_dir: Directory to save results
        show_plots: Whether to show plots
        save_plots: Whether to save plots
        
    Returns:
        Dictionary with results
    """
    print(f"\nRunning sandbox for {model_name}...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    data = load_data(data_path)
    
    # Get model configuration
    config = get_model_config(model_name)
    
    # Initialize model
    model_class = ModelRegistry.get_model_class(model_name)
    model = model_class(config)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Train model
    print("Training model...")
    start_time = datetime.now()
    model.fit(train_data)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Make predictions
    print("Making predictions...")
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
    
    # Calculate metrics
    metrics = calculate_metrics(
        test_data['close'].values,
        predictions
    )
    
    # Generate visualizations
    if show_plots or save_plots:
        print("Generating visualizations...")
        
        # Forecast plot
        forecast_fig = plot_forecast(
            test_data,
            predictions,
            show_confidence=True
        )
        if show_plots:
            forecast_fig.show()
        if save_plots:
            forecast_fig.write_html(os.path.join(save_dir, f'{model_name}_forecast.html'))
        
        # Model components
        try:
            components_fig = plot_model_components(model, test_data)
            if show_plots:
                components_fig.show()
            if save_plots:
                components_fig.write_html(os.path.join(save_dir, f'{model_name}_components.html'))
        except Exception as e:
            print(f"Model components not supported: {e}")
        
        # Attention heatmap
        try:
            attention_fig = plot_attention_heatmap(model, test_data)
            if show_plots:
                attention_fig.show()
            if save_plots:
                attention_fig.write_html(os.path.join(save_dir, f'{model_name}_attention.html'))
        except Exception as e:
            print(f"Attention heatmap not supported: {e}")
        
        # SHAP values
        try:
            shap_fig = plot_shap_values(model, test_data)
            if show_plots:
                shap_fig.show()
            if save_plots:
                shap_fig.write_html(os.path.join(save_dir, f'{model_name}_shap.html'))
        except Exception as e:
            print(f"SHAP values not supported: {e}")
        
        # Multi-horizon predictions
        if multi_horizon_preds:
            horizon_fig = make_subplots(rows=len(horizons), cols=1,
                                      subplot_titles=[f'Horizon {h}' for h in horizons])
            for i, (horizon, preds) in enumerate(multi_horizon_preds.items(), 1):
                horizon_fig.add_trace(
                    go.Scatter(x=test_data.index, y=preds, name=f'Horizon {horizon}'),
                    row=i, col=1
                )
            horizon_fig.update_layout(height=300 * len(horizons), showlegend=True)
            if show_plots:
                horizon_fig.show()
            if save_plots:
                horizon_fig.write_html(os.path.join(save_dir, f'{model_name}_horizons.html'))
    
    # Save results
    results = {
        'model_name': model_name,
        'config': config,
        'metrics': metrics,
        'training_time': training_time,
        'multi_horizon_supported': bool(multi_horizon_preds),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, f'{model_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': test_data.index,
        'actual': test_data['close'],
        'predicted': predictions
    })
    predictions_df.to_csv(os.path.join(save_dir, f'{model_name}_predictions.csv'))
    
    print(f"\nResults saved to {save_dir}")
    print(f"Training time: {training_time:.2f} seconds")
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return results

def main():
    """Run sandbox environment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model sandbox environment')
    parser.add_argument('--model', type=str, required=True,
                      help='Name of the model to test')
    parser.add_argument('--data', type=str, default=None,
                      help='Path to data file (optional)')
    parser.add_argument('--save-dir', type=str, default='sandbox_results',
                      help='Directory to save results')
    parser.add_argument('--no-plots', action='store_true',
                      help='Disable plot display')
    parser.add_argument('--no-save', action='store_true',
                      help='Disable plot saving')
    
    args = parser.parse_args()
    
    # Validate model name
    if args.model not in ModelRegistry.get_registered_models():
        print(f"Error: Model '{args.model}' not found in registry")
        print("Available models:", ModelRegistry.get_registered_models())
        sys.exit(1)
    
    # Run sandbox
    run_sandbox(
        args.model,
        data_path=args.data,
        save_dir=args.save_dir,
        show_plots=not args.no_plots,
        save_plots=not args.no_save
    )

if __name__ == "__main__":
    main() 