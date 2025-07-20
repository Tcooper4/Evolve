"""
Optuna Tuner Example

This example demonstrates how to use the enhanced Optuna tuner for optimizing
trading models with Sharpe ratio as the objective function.

Features demonstrated:
- LSTM optimization with num_layers, dropout, learning_rate, lookback
- XGBoost optimization with max_depth, learning_rate, n_estimators
- Transformer optimization with d_model, num_heads, ff_dim, dropout
- Integration with forecasting pipeline
- Performance evaluation and model selection
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample financial data for demonstration.
    
    Args:
        n_samples: Number of data points to generate
        
    Returns:
        DataFrame with sample financial data
    """
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate price data with trend and noise
    trend = np.linspace(100, 150, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    prices = trend + noise
    
    # Generate volume data
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    # Generate technical indicators
    sma_20 = pd.Series(prices).rolling(20).mean().values
    rsi = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 5, n_samples)
    rsi = np.clip(rsi, 0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volume,
        'sma_20': sma_20,
        'rsi': rsi,
        'returns': np.diff(prices, prepend=prices[0]) / prices,
        'volatility': pd.Series(prices).pct_change().rolling(20).std().values
    })
    
    # Remove NaN values
    data = data.dropna()
    
    logger.info(f"Generated sample data with shape: {data.shape}")
    return data

def demonstrate_lstm_optimization():
    """Demonstrate LSTM hyperparameter optimization."""
    logger.info("=== LSTM Optimization Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Import and create tuner
    from trading.optimization.optuna_tuner import SharpeOptunaTuner
    
    tuner = SharpeOptunaTuner(
        study_name="lstm_demo",
        n_trials=20,  # Reduced for demo
        timeout=300,  # 5 minutes
        validation_split=0.2
    )
    
    # Optimize LSTM
    logger.info("Starting LSTM optimization...")
    result = tuner.optimize_lstm(
        data=data,
        target_column='close',
        feature_columns=['volume', 'sma_20', 'rsi', 'returns', 'volatility']
    )
    
    logger.info(f"LSTM optimization completed!")
    logger.info(f"Best parameters: {result['best_params']}")
    logger.info(f"Best Sharpe ratio: {result['best_score']:.4f}")
    
    return result

def demonstrate_xgboost_optimization():
    """Demonstrate XGBoost hyperparameter optimization."""
    logger.info("=== XGBoost Optimization Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Import and create tuner
    from trading.optimization.optuna_tuner import SharpeOptunaTuner
    
    tuner = SharpeOptunaTuner(
        study_name="xgboost_demo",
        n_trials=20,  # Reduced for demo
        timeout=300,  # 5 minutes
        validation_split=0.2
    )
    
    # Optimize XGBoost
    logger.info("Starting XGBoost optimization...")
    result = tuner.optimize_xgboost(
        data=data,
        target_column='close',
        feature_columns=['volume', 'sma_20', 'rsi', 'returns', 'volatility']
    )
    
    logger.info(f"XGBoost optimization completed!")
    logger.info(f"Best parameters: {result['best_params']}")
    logger.info(f"Best Sharpe ratio: {result['best_score']:.4f}")
    
    return result

def demonstrate_transformer_optimization():
    """Demonstrate Transformer hyperparameter optimization."""
    logger.info("=== Transformer Optimization Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Import and create tuner
    from trading.optimization.optuna_tuner import SharpeOptunaTuner
    
    tuner = SharpeOptunaTuner(
        study_name="transformer_demo",
        n_trials=20,  # Reduced for demo
        timeout=300,  # 5 minutes
        validation_split=0.2
    )
    
    # Optimize Transformer
    logger.info("Starting Transformer optimization...")
    result = tuner.optimize_transformer(
        data=data,
        target_column='close',
        feature_columns=['volume', 'sma_20', 'rsi', 'returns', 'volatility']
    )
    
    logger.info(f"Transformer optimization completed!")
    logger.info(f"Best parameters: {result['best_params']}")
    logger.info(f"Best Sharpe ratio: {result['best_score']:.4f}")
    
    return result

def demonstrate_all_models_optimization():
    """Demonstrate optimization of all model types and selection."""
    logger.info("=== All Models Optimization Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Import and create tuner
    from trading.optimization.optuna_tuner import SharpeOptunaTuner
    
    tuner = SharpeOptunaTuner(
        study_name="all_models_demo",
        n_trials=15,  # Reduced for demo
        timeout=600,  # 10 minutes
        validation_split=0.2
    )
    
    # Optimize all models
    logger.info("Starting optimization of all model types...")
    result = tuner.optimize_all_models(
        data=data,
        target_column='close',
        model_types=['lstm', 'xgboost', 'transformer']
    )
    
    logger.info(f"All models optimization completed!")
    logger.info(f"Best model: {result['best_model']}")
    logger.info(f"Best Sharpe ratio: {result['best_score']:.4f}")
    logger.info(f"All results: {result['all_results']}")
    
    return result

def demonstrate_forecasting_integration():
    """Demonstrate integration with forecasting pipeline."""
    logger.info("=== Forecasting Integration Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Import forecasting optimizer
    from trading.optimization.forecasting_integration import ForecastingOptimizer
    
    # Create optimizer
    optimizer = ForecastingOptimizer(
        optimization_config={
            'n_trials': 10,  # Reduced for demo
            'timeout': 300,  # 5 minutes
            'model_types': ['lstm', 'xgboost'],
            'min_sharpe_threshold': 0.05
        }
    )
    
    # Optimize for forecasting
    logger.info("Starting forecasting optimization...")
    result = optimizer.optimize_for_forecasting(
        data=data,
        target_column='close',
        forecast_horizon=30
    )
    
    if result['success']:
        logger.info(f"Forecasting optimization completed!")
        logger.info(f"Recommended model: {result['recommendation']['recommended_model']}")
        logger.info(f"Expected Sharpe ratio: {result['recommendation']['expected_sharpe']:.4f}")
        
        # Get optimized model
        model, params = optimizer.get_optimized_model(
            model_type=result['recommendation']['recommended_model'],
            data=data,
            target_column='close'
        )
        
        logger.info(f"Optimized model created with parameters: {params}")
        
        # Evaluate performance
        performance = optimizer.evaluate_model_performance(
            model=model,
            data=data.tail(100),  # Use last 100 points for evaluation
            target_column='close',
            model_type=result['recommendation']['recommended_model']
        )
        
        logger.info(f"Model performance: {performance}")
        
    else:
        logger.error(f"Forecasting optimization failed: {result.get('error')}")
    
    return result

def demonstrate_pipeline_integration():
    """Demonstrate integration with existing forecasting pipeline."""
    logger.info("=== Pipeline Integration Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Import integration function
    from trading.optimization.forecasting_integration import integrate_with_forecasting_pipeline
    
    # Integrate with pipeline
    logger.info("Integrating optimization with forecasting pipeline...")
    result = integrate_with_forecasting_pipeline(
        data=data,
        target_column='close',
        forecast_horizon=30,
        auto_optimize=True
    )
    
    if result['success']:
        logger.info(f"Pipeline integration successful!")
        logger.info(f"Selected model: {result['model_type']}")
        logger.info(f"Model parameters: {result['parameters']}")
        
        if result['optimization_result']:
            logger.info(f"Optimization Sharpe ratio: {result['optimization_result']['recommendation']['expected_sharpe']:.4f}")
    else:
        logger.error(f"Pipeline integration failed: {result.get('error')}")
    
    return result

def demonstrate_results_analysis():
    """Demonstrate analysis of optimization results."""
    logger.info("=== Results Analysis Demo ===")
    
    # Generate sample data
    data = generate_sample_data(500)
    
    # Import and create tuner
    from trading.optimization.optuna_tuner import SharpeOptunaTuner
    
    tuner = SharpeOptunaTuner(
        study_name="analysis_demo",
        n_trials=10,  # Reduced for demo
        timeout=300,  # 5 minutes
        validation_split=0.2
    )
    
    # Run optimization
    result = tuner.optimize_all_models(
        data=data,
        target_column='close',
        model_types=['lstm', 'xgboost']
    )
    
    # Save results
    results_file = tuner.save_results()
    logger.info(f"Results saved to: {results_file}")
    
    # Load results
    loaded_results = tuner.load_results(results_file)
    logger.info(f"Results loaded successfully")
    
    # Get best parameters for each model
    for model_type in ['lstm', 'xgboost']:
        best_params = tuner.get_best_params(model_type)
        best_score = tuner.get_best_score(model_type)
        
        if best_params and best_score:
            logger.info(f"{model_type.upper()} - Best Sharpe: {best_score:.4f}")
            logger.info(f"{model_type.upper()} - Best params: {best_params}")
    
    # Get model recommendation
    recommendation = tuner.get_model_recommendation(data, 'close')
    logger.info(f"Model recommendation: {recommendation}")
    
    return {
        'results_file': results_file,
        'loaded_results': loaded_results,
        'recommendation': recommendation
    }

def main():
    """Run all demonstration functions."""
    logger.info("Starting Optuna Tuner Examples")
    logger.info("=" * 50)
    
    try:
        # Run individual model optimizations
        lstm_result = demonstrate_lstm_optimization()
        logger.info("-" * 30)
        
        xgboost_result = demonstrate_xgboost_optimization()
        logger.info("-" * 30)
        
        transformer_result = demonstrate_transformer_optimization()
        logger.info("-" * 30)
        
        # Run all models optimization
        all_models_result = demonstrate_all_models_optimization()
        logger.info("-" * 30)
        
        # Run forecasting integration
        forecasting_result = demonstrate_forecasting_integration()
        logger.info("-" * 30)
        
        # Run pipeline integration
        pipeline_result = demonstrate_pipeline_integration()
        logger.info("-" * 30)
        
        # Run results analysis
        analysis_result = demonstrate_results_analysis()
        logger.info("-" * 30)
        
        # Summary
        logger.info("=== SUMMARY ===")
        logger.info(f"LSTM best Sharpe: {lstm_result['best_score']:.4f}")
        logger.info(f"XGBoost best Sharpe: {xgboost_result['best_score']:.4f}")
        logger.info(f"Transformer best Sharpe: {transformer_result['best_score']:.4f}")
        logger.info(f"Overall best model: {all_models_result['best_model']}")
        logger.info(f"Overall best Sharpe: {all_models_result['best_score']:.4f}")
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    main()
