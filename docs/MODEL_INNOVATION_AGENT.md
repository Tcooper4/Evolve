# Model Innovation Agent

## Overview

The `ModelInnovationAgent` is an autonomous agent that automatically discovers, evaluates, and integrates new forecasting models into the Evolve trading platform's ensemble. It uses AutoML techniques to search for better model architectures and automatically updates the model registry when superior models are found.

## Key Features

- **Automated Model Discovery**: Uses FLAML and Optuna for efficient hyperparameter optimization
- **Multi-Model Support**: Supports linear, tree-based, and neural network models
- **Performance Comparison**: Automatically compares candidates against existing ensemble
- **Intelligent Integration**: Updates model registry and optimizes ensemble weights
- **Comprehensive Evaluation**: Uses multiple metrics (MSE, Sharpe ratio, R², drawdown)
- **Caching**: Integrates with the platform's caching system for performance

## Architecture

```
Data Input → Model Discovery → Evaluation → Comparison → Integration → Weight Optimization
     ↓              ↓            ↓           ↓           ↓              ↓
  Preprocessing  AutoML Search  Metrics    Ensemble   Registry     Ensemble
                 (FLAML/Optuna)  Calculation  Comparison  Update      Optimization
```

## Installation

### Dependencies

The agent requires several optional dependencies for full functionality:

```bash
# Core AutoML libraries
pip install flaml optuna

# Machine learning libraries
pip install scikit-learn

# Deep learning (optional)
pip install torch

# Trading platform dependencies
pip install pandas numpy
```

### Basic Usage

```python
from agents.model_innovation_agent import create_model_innovation_agent, InnovationConfig

# Create agent with default configuration
agent = create_model_innovation_agent()

# Or with custom configuration
config = InnovationConfig(
    automl_time_budget=300,  # 5 minutes
    max_models_per_search=10,
    min_improvement_threshold=0.05,  # 5% improvement required
    enable_neural_models=True,
)

agent = create_model_innovation_agent(config)
```

## Configuration

### InnovationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `automl_time_budget` | int | 300 | Time budget for AutoML search (seconds) |
| `automl_metric` | str | "mse" | Metric to optimize (mse, mae, r2) |
| `automl_task` | str | "regression" | ML task type |
| `max_models_per_search` | int | 10 | Maximum models to discover per cycle |
| `min_improvement_threshold` | float | 0.05 | Minimum improvement required for integration |
| `evaluation_window_days` | int | 30 | Days of data for evaluation |
| `enable_linear_models` | bool | True | Enable linear model search |
| `enable_tree_models` | bool | True | Enable tree-based model search |
| `enable_neural_models` | bool | True | Enable neural network search |
| `enable_ensemble_models` | bool | True | Enable ensemble model search |
| `cv_folds` | int | 5 | Cross-validation folds |
| `test_size` | float | 0.2 | Test set size |
| `random_state` | int | 42 | Random seed |
| `models_dir` | str | "models/innovated" | Directory for saved models |
| `cache_dir` | str | "cache/model_innovation" | Cache directory |
| `min_sharpe_ratio` | float | 0.5 | Minimum Sharpe ratio threshold |
| `max_drawdown_threshold` | float | 0.15 | Maximum drawdown threshold |
| `min_r2_score` | float | 0.3 | Minimum R² score threshold |

## Usage Examples

### Basic Innovation Cycle

```python
import pandas as pd
from agents.model_innovation_agent import create_model_innovation_agent

# Create agent
agent = create_model_innovation_agent()

# Prepare data
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 4, 6, 8, 10],
    'target': [1.5, 3.0, 4.5, 6.0, 7.5]
})

# Run innovation cycle
results = agent.run_innovation_cycle(data, target_col="target")

print(f"Models discovered: {results['candidates_discovered']}")
print(f"Models evaluated: {results['candidates_evaluated']}")
print(f"Models integrated: {results['models_integrated']}")
```

### Advanced Configuration

```python
from agents.model_innovation_agent import InnovationConfig

# Custom configuration for aggressive search
config = InnovationConfig(
    automl_time_budget=600,  # 10 minutes
    max_models_per_search=20,
    min_improvement_threshold=0.01,  # 1% improvement
    enable_linear_models=True,
    enable_tree_models=True,
    enable_neural_models=True,
    enable_ensemble_models=True,
    cv_folds=10,
    min_sharpe_ratio=0.3,
    max_drawdown_threshold=0.2,
)

agent = create_model_innovation_agent(config)
```

### Continuous Innovation

```python
# Run multiple innovation cycles
for cycle in range(5):
    print(f"Running innovation cycle {cycle + 1}")
    
    # Get fresh data (in real scenario, this would be new market data)
    data = get_latest_market_data()
    
    # Run innovation cycle
    results = agent.run_innovation_cycle(data, target_col="target")
    
    # Check results
    if results['models_integrated'] > 0:
        print(f"Integrated {results['models_integrated']} new models")
    
    # Wait before next cycle
    time.sleep(3600)  # 1 hour
```

### Individual Model Evaluation

```python
# Discover models
candidates = agent.discover_models(data, target_col="target")

# Evaluate each candidate
for candidate in candidates:
    evaluation = agent.evaluate_candidate(candidate, data, target_col="target")
    
    print(f"Model: {candidate.name}")
    print(f"  MSE: {evaluation.mse:.4f}")
    print(f"  Sharpe: {evaluation.sharpe_ratio:.4f}")
    print(f"  R²: {evaluation.r2_score:.4f}")
    
    # Compare with ensemble
    comparison = agent.compare_with_ensemble(evaluation)
    
    if comparison['improvement']:
        print("  ✅ Model improves ensemble")
        agent.integrate_model(candidate, evaluation)
    else:
        print("  ❌ Model does not improve ensemble")
```

## Model Discovery Methods

### FLAML AutoML

FLAML provides efficient AutoML with the following search space:

```python
# Linear models
search_space = {
    "linear": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
    }
}

# Tree models
search_space.update({
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
})

# Neural networks
search_space.update({
    "neural": {
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
    }
})
```

### Optuna Hyperparameter Optimization

Optuna provides more advanced optimization with custom objective functions:

```python
def objective(trial):
    # Sample model type
    model_type = trial.suggest_categorical("model_type", ["linear", "tree", "neural"])
    
    if model_type == "linear":
        model = Ridge(
            alpha=trial.suggest_float("alpha", 0.001, 10.0, log=True),
            fit_intercept=trial.suggest_categorical("fit_intercept", [True, False])
        )
    elif model_type == "tree":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            random_state=42
        )
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()
```

## Evaluation Metrics

The agent evaluates models using multiple metrics:

### Regression Metrics
- **MSE (Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Robust measure of prediction error
- **R² Score**: Proportion of variance explained by the model

### Trading Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Total Return**: Cumulative return over evaluation period
- **Volatility**: Standard deviation of returns

### Performance Metrics
- **Training Time**: Time required to train the model
- **Inference Time**: Time required for predictions
- **Model Size**: Size of the saved model in MB

## Integration Process

### 1. Model Discovery
```python
candidates = agent.discover_models(data, target_col="target")
```

### 2. Model Evaluation
```python
evaluation = agent.evaluate_candidate(candidate, data, target_col="target")
```

### 3. Ensemble Comparison
```python
comparison = agent.compare_with_ensemble(evaluation)
```

### 4. Model Integration
```python
if comparison['improvement']:
    success = agent.integrate_model(candidate, evaluation)
```

### 5. Weight Optimization
```python
# Automatically optimized during integration
optimized_weights = optimize_ensemble_weights(
    model_names=current_models,
    method="performance_weighted"
)
```

## Monitoring and Statistics

### Innovation Statistics

```python
stats = agent.get_innovation_statistics()

print(f"Total cycles: {stats['total_cycles']}")
print(f"Models integrated: {stats['total_models_integrated']}")
print(f"Total evaluations: {stats['total_evaluations']}")

# Model type distribution
for model_type, count in stats['model_type_distribution'].items():
    print(f"{model_type}: {count}")

# Recent improvements
for improvement in stats['performance_improvements'][-5:]:
    print(f"{improvement['timestamp']}: {improvement['model_name']}")
```

### Performance Tracking

```python
# Get cache statistics
from utils.cache_utils import get_cache_stats
cache_stats = get_cache_stats()

print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
print(f"Total requests: {cache_stats['total_requests']}")

# Get registry summary
from utils.weight_registry import get_registry_summary
registry_summary = get_registry_summary()

print(f"Total models: {registry_summary['total_models']}")
print(f"Performance records: {registry_summary['total_performance_records']}")
```

## Best Practices

### 1. Data Quality
- Ensure data is clean and preprocessed
- Handle missing values appropriately
- Use sufficient data for training (minimum 1000 samples)

### 2. Configuration Tuning
- Start with shorter time budgets for testing
- Adjust improvement thresholds based on your needs
- Enable only necessary model types to reduce search time

### 3. Monitoring
- Monitor innovation cycles regularly
- Track performance improvements over time
- Review integrated models periodically

### 4. Resource Management
- Set appropriate time budgets for your environment
- Monitor memory usage during model training
- Use caching to improve performance

## Troubleshooting

### Common Issues

1. **No models discovered**
   - Check if AutoML libraries are installed
   - Verify data quality and size
   - Reduce time budget constraints

2. **Models not integrating**
   - Check improvement threshold settings
   - Verify ensemble comparison logic
   - Review error logs for integration failures

3. **Performance issues**
   - Reduce time budget for faster cycles
   - Enable caching for repeated operations
   - Use smaller datasets for testing

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Create agent with debug logging
agent = create_model_innovation_agent()
agent.logger.setLevel(logging.DEBUG)
```

## Integration with Trading Platform

### Weight Registry Integration

The agent automatically integrates with the platform's weight registry:

```python
from utils.weight_registry import get_weight_registry

registry = get_weight_registry()

# Check current models
models = registry.registry["models"]

# Get optimized weights
from utils.weight_registry import optimize_ensemble_weights
weights = optimize_ensemble_weights(list(models.keys()))
```

### Caching Integration

The agent uses the platform's caching system:

```python
from utils.cache_utils import get_cache_stats, cleanup_cache

# Check cache performance
stats = get_cache_stats()

# Clean up old cache entries
cleanup_cache()
```

## API Reference

### ModelInnovationAgent

#### Methods

- `discover_models(data, target_col)`: Discover new model candidates
- `evaluate_candidate(candidate, data, target_col)`: Evaluate a model candidate
- `compare_with_ensemble(evaluation)`: Compare with existing ensemble
- `integrate_model(candidate, evaluation)`: Integrate successful model
- `run_innovation_cycle(data, target_col)`: Run complete innovation cycle
- `get_innovation_statistics()`: Get innovation statistics

#### Properties

- `discovered_models`: List of discovered model candidates
- `evaluations`: List of model evaluations
- `innovation_history`: History of innovation activities

### ModelCandidate

#### Attributes

- `name`: Model name
- `model_type`: Type of model (linear, tree, neural)
- `model`: Trained model object
- `hyperparameters`: Model hyperparameters
- `training_time`: Training time in seconds
- `metadata`: Additional metadata

### ModelEvaluation

#### Attributes

- `model_name`: Name of evaluated model
- `mse`: Mean squared error
- `mae`: Mean absolute error
- `r2_score`: R² score
- `sharpe_ratio`: Sharpe ratio
- `max_drawdown`: Maximum drawdown
- `total_return`: Total return
- `volatility`: Volatility
- `training_time`: Training time
- `inference_time`: Inference time
- `model_size_mb`: Model size in MB

## Future Enhancements

### Planned Features

1. **Advanced Neural Networks**: Support for LSTM, Transformer models
2. **Feature Engineering**: Automated feature selection and engineering
3. **Ensemble Methods**: Advanced ensemble techniques (stacking, blending)
4. **Real-time Integration**: Real-time model updates during trading
5. **A/B Testing**: Model performance comparison framework

### Extensibility

The agent is designed to be easily extensible:

```python
# Custom model discovery
class CustomDiscoveryMethod:
    def discover(self, data, target_col):
        # Custom discovery logic
        return candidates

# Custom evaluation
class CustomEvaluator:
    def evaluate(self, candidate, data, target_col):
        # Custom evaluation logic
        return evaluation
```

---

*Model Innovation Agent v1.0 - AutoML-Powered Model Discovery* 