"""
Sample run script for all registered models.
- Loads data (real or synthetic)
- Lets user select any registered model
- Trains and predicts
- Shows forecast plot, model summary, interpretability
- Saves predictions to CSV
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.models.base_model import ModelRegistry
from utils.visualization import (
    plot_forecast,
    plot_attention_heatmap,
    plot_shap_values,
    plot_model_comparison
)
from utils.metrics import calculate_metrics

# 1. Load data (synthetic for demo)
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

data = generate_synthetic_data()

# 2. User selects model
def select_model():
    print("\nAvailable models:")
    for i, name in enumerate(ModelRegistry._models.keys()):
        print(f"{i+1}. {name}")
    idx = int(input("Select model by number: ")) - 1
    model_name = list(ModelRegistry._models.keys())[idx]
    return model_name

model_name = select_model()
model_cls = ModelRegistry._models[model_name]

# 3. Prepare config for each model type
def get_config(model_name):
    if model_name == 'Prophet':
        return {
            'date_column': 'date',
            'target_column': 'close',
            'prophet_params': {}
        }
    elif model_name == 'CatBoost':
        return {
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'catboost_params': {'iterations': 50, 'verbose': False}
        }
    elif model_name == 'Autoformer':
        return {
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'sequence_length': 24,
            'pred_length': 1,
            'autoformer_params': {}
        }
    elif model_name == 'EnsembleModel':
        # Example: ensemble of CatBoost and Prophet
        return {
            'models': [
                {'type': 'CatBoost', 'feature_columns': ['close', 'volume'], 'target_column': 'close', 'catboost_params': {'iterations': 50, 'verbose': False}},
                {'type': 'Prophet', 'date_column': 'date', 'target_column': 'close', 'prophet_params': {}}
            ],
            'weight_method': 'mse',
            'target_column': 'close'
        }
    else:
        # Default for other models
        return {
            'feature_columns': ['close', 'volume'],
            'target_column': 'close',
            'sequence_length': 24
        }

config = get_config(model_name)
model = model_cls(config)

# 4. Train/test split
train = data.iloc[:150]
test = data.iloc[150:]

# 5. Train
print("\nTraining...")
model.fit(train)

# 6. Predict
print("\nPredicting...")
preds = model.predict(test)

# 7. Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(test['date'], test['close'], label='Actual')
plt.plot(test['date'], preds, label='Forecast')
plt.title(f"Forecast with {model_name}")
plt.legend()
plt.tight_layout()
plt.show()

# 8. Model summary
print("\nModel Summary:")
model.summary()

# 9. Interpretability
print("\nInterpretability:")
if hasattr(model, 'shap_interpret'):
    try:
        if model_name == 'CatBoost':
            model.shap_interpret(test[config['feature_columns']])
        elif model_name == 'Prophet':
            model.shap_interpret(None)
        elif model_name == 'Autoformer':
            model.shap_interpret(test[config['feature_columns']].values.astype(np.float32))
        else:
            model.shap_interpret(test[config['feature_columns']])
    except Exception as e:
        print(f"Interpretability failed: {e}")
else:
    print("No interpretability method available.")

# 10. Save predictions
out_path = 'sample_predictions.csv'
pd.DataFrame({'date': test['date'], 'actual': test['close'], 'forecast': preds}).to_csv(out_path, index=False)
print(f"\nPredictions saved to {out_path}")

def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get default configurations for all registered models.
    
    Returns:
        Dictionary of model configurations
    """
    return {
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

def main():
    """Run sample demonstration of all registered models."""
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data()
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # Store results for comparison
    results = {}
    
    # Test each registered model
    for model_name in ModelRegistry.get_registered_models():
        print(f"\nTesting {model_name}...")
        
        try:
            # Get model class and configuration
            model_class = ModelRegistry.get_model_class(model_name)
            config = model_configs.get(model_name, {})
            
            # Initialize model
            model = model_class(config)
            
            # Split data into train and test
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Train model
            print("Training model...")
            model.fit(train_data)
            
            # Make predictions
            print("Making predictions...")
            predictions = model.predict(test_data)
            
            # Calculate metrics
            metrics = calculate_metrics(
                test_data['close'].values,
                predictions
            )
            results[model_name] = metrics
            
            # Plot forecast
            fig = plot_forecast(test_data, predictions)
            fig.write_html(f"forecast_{model_name}.html")
            
            # Plot attention heatmap if supported
            if hasattr(model, 'attention_heatmap'):
                try:
                    fig = plot_attention_heatmap(model, test_data)
                    fig.write_html(f"attention_{model_name}.html")
                except Exception as e:
                    print(f"Could not generate attention heatmap: {e}")
            
            # Plot SHAP values if supported
            if hasattr(model, 'shap_interpret'):
                try:
                    fig = plot_shap_values(model, test_data)
                    fig.write_html(f"shap_{model_name}.html")
                except Exception as e:
                    print(f"Could not generate SHAP values: {e}")
            
            print(f"Results saved for {model_name}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    # Compare all models
    if results:
        metrics_df = pd.DataFrame(results).T
        fig = plot_model_comparison(metrics_df)
        fig.write_html("model_comparison.html")
        print("\nModel comparison saved to model_comparison.html")
        
        # Save metrics to CSV
        metrics_df.to_csv("model_metrics.csv")
        print("Metrics saved to model_metrics.csv")

if __name__ == "__main__":
    main() 