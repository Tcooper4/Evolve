# Autonomous 3-Agent Model Management System

A sophisticated autonomous system for continuous model management, evaluation, and improvement using three specialized agents working in harmony.

## 🏗️ Architecture Overview

The system consists of three autonomous agents that work together in a continuous loop:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ModelBuilder   │───▶│ PerformanceCritic│───▶│    Updater      │
│     Agent       │    │      Agent       │    │     Agent       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                                              │
         │                                              │
         └──────────────────────────────────────────────┘
```

### Agent Responsibilities

1. **ModelBuilderAgent**: Creates LSTM, XGBoost, and ensemble models from scratch
2. **PerformanceCriticAgent**: Evaluates models using financial metrics (Sharpe ratio, drawdown, win rate)
3. **UpdaterAgent**: Tunes, retrains, or replaces models based on critic feedback

## 🚀 Quick Start

### Installation

```bash
# Navigate to the trading directory
cd trading/agents

# Run the agent loop
python run_agent_loop.py
```

### Configuration

The system uses `agent_config.json` for configuration. Key settings:

```json
{
  "agent_loop": {
    "cycle_interval": 3600,  // 1 hour cycles
    "max_models": 10,        // Maximum active models
    "evaluation_threshold": 0.5
  }
}
```

### Command Line Options

```bash
# Run with custom configuration
python run_agent_loop.py --config custom_config.json

# Override cycle interval (in seconds)
python run_agent_loop.py --cycle-interval 1800

# Set maximum models
python run_agent_loop.py --max-models 15

# Set log level
python run_agent_loop.py --log-level DEBUG
```

## 🤖 Agent Details

### ModelBuilderAgent

**Purpose**: Creates and initializes ML models from scratch

**Features**:
- Builds LSTM, XGBoost, and ensemble models
- Automatic hyperparameter selection
- Model persistence and versioning
- Feature engineering integration

**Example Usage**:
```python
from trading.agents import ModelBuilderAgent, ModelBuildRequest

builder = ModelBuilderAgent()

request = ModelBuildRequest(
    model_type='lstm',
    data_path='data/market_data.csv',
    target_column='close',
    hyperparameters={
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2
    }
)

result = builder.build_model(request)
print(f"Built model: {result.model_id}")
```

### PerformanceCriticAgent

**Purpose**: Evaluates model performance using financial metrics

**Metrics Evaluated**:
- **Performance**: Sharpe ratio, total return, annualized return, volatility
- **Risk**: Maximum drawdown, VaR, CVaR, Sortino ratio, Calmar ratio
- **Trading**: Win rate, profit factor, average trade, total trades

**Example Usage**:
```python
from trading.agents import PerformanceCriticAgent, ModelEvaluationRequest

critic = PerformanceCriticAgent()

request = ModelEvaluationRequest(
    model_id='model_123',
    model_path='models/model_123.pkl',
    model_type='lstm',
    test_data_path='data/test_data.csv'
)

result = critic.evaluate_model(request)
print(f"Sharpe Ratio: {result.performance_metrics['sharpe_ratio']}")
print(f"Max Drawdown: {result.risk_metrics['max_drawdown']}")
```

### UpdaterAgent

**Purpose**: Updates models based on performance feedback

**Update Types**:
- **Retrain**: Rebuild model with improved hyperparameters
- **Tune**: Optimize hyperparameters using Bayesian optimization
- **Replace**: Replace failing model with alternative type
- **Ensemble Adjust**: Optimize ensemble weights

**Example Usage**:
```python
from trading.agents import UpdaterAgent

updater = UpdaterAgent()

# Process evaluation and determine update
update_request = updater.process_evaluation(evaluation_result)

if update_request:
    # Execute the update
    update_result = updater.execute_update(update_request)
    print(f"Updated model: {update_result.new_model_id}")
```

## 🔄 Autonomous Loop

### Cycle Process

1. **Model Builder Phase**:
   - Checks if new models are needed
   - Determines optimal model type
   - Builds models with appropriate hyperparameters

2. **Performance Critic Phase**:
   - Identifies models requiring evaluation
   - Calculates comprehensive performance metrics
   - Generates recommendations

3. **Updater Phase**:
   - Processes critic recommendations
   - Executes appropriate updates (retrain/tune/replace)
   - Maintains model registry

### State Persistence

The system maintains state across cycles:
- Model registry and metadata
- Evaluation history
- Update history
- Communication logs

### Communication System

Agents communicate through a queue-based system:
- JSON message format
- Priority-based processing
- Persistent logging
- Error handling and retries

## 📊 Monitoring and Metrics

### System Metrics

- **Cycle Statistics**: Models built, evaluated, updated per cycle
- **Performance Trends**: Model performance over time
- **Failure Tracking**: Failed operations and error rates
- **Resource Usage**: Memory, CPU, queue sizes

### Health Checks

- Agent status monitoring
- Communication queue health
- Model registry integrity
- Data source availability

## 🔧 Configuration

### Agent Loop Settings

```json
{
  "agent_loop": {
    "cycle_interval": 3600,
    "max_models": 10,
    "evaluation_threshold": 0.5,
    "auto_start": true
  }
}
```

### Model Builder Settings

```json
{
  "model_builder": {
    "models_dir": "trading/models/built",
    "default_hyperparameters": {
      "lstm": {
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2
      }
    }
  }
}
```

### Performance Critic Settings

```json
{
  "performance_critic": {
    "thresholds": {
      "min_sharpe_ratio": 0.5,
      "max_drawdown": -0.15,
      "min_win_rate": 0.45
    }
  }
}
```

### Updater Settings

```json
{
  "updater": {
    "update_thresholds": {
      "critical_sharpe": 0.0,
      "retrain_sharpe": 0.3,
      "tune_sharpe": 0.5
    }
  }
}
```

## 🛠️ Development

### Adding New Model Types

1. Implement model class in `trading/models/`
2. Add hyperparameters to `agent_config.json`
3. Update `ModelBuilderAgent._build_*_model()` methods
4. Add evaluation logic in `PerformanceCriticAgent`

### Adding New Metrics

1. Implement metric calculation in `trading/evaluation/metrics.py`
2. Add metric to `PerformanceCriticAgent._calculate_*_metrics()`
3. Update configuration thresholds
4. Add to recommendations logic

### Custom Update Strategies

1. Implement strategy in `UpdaterAgent`
2. Add update type to `UpdateRequest`
3. Update `_determine_update_action()` logic
4. Add configuration options

## 📝 Logging

The system provides comprehensive logging:

- **Agent Loop**: Cycle execution and coordination
- **Model Builder**: Model creation and training
- **Performance Critic**: Evaluation results and recommendations
- **Updater**: Update decisions and execution
- **Communication**: Inter-agent messages

Log files are stored in `trading/agents/logs/` with rotation and compression.

## 🚨 Error Handling

The system includes robust error handling:

- **Graceful Degradation**: Continues operation despite individual failures
- **Retry Logic**: Automatic retry for transient failures
- **Failure Tracking**: Comprehensive logging of failures
- **Recovery Mechanisms**: Automatic cleanup and recovery

## 🔒 Security

- **Model Validation**: Input validation for all model operations
- **Safe File Operations**: Secure file handling and cleanup
- **Memory Management**: Proper resource cleanup and memory limits
- **Error Isolation**: Failures don't propagate across agents

## 📈 Performance Optimization

- **Asynchronous Operations**: Non-blocking agent communication
- **Resource Pooling**: Efficient model and data management
- **Caching**: Intelligent caching of frequently used data
- **Parallel Processing**: Concurrent model evaluation and updates

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive type hints and docstrings
3. Include unit tests for new functionality
4. Update documentation for new features
5. Ensure backward compatibility

## 📄 License

This project is part of the Evolve Trading System and follows the same licensing terms. 