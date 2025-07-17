# 🚀 Evolve AI Trading Platform

A production-ready, quant-grade trading intelligence platform powered by advanced AI and machine learning. Evolve provides a clean, ChatGPT-like interface for natural language trading operations with comprehensive forecasting, strategy optimization, and risk management capabilities.

## 🎯 What the App Does

Evolve is an **autonomous AI trading system** that transforms natural language requests into sophisticated trading operations. Think of it as a "ChatGPT for trading" that can:

### Core Capabilities
- **🤖 Agentic Model Selection**: AI agents automatically select and optimize the best forecasting models for your data
- **📊 Advanced Forecasting**: Generate predictions using 15+ ML/DL models with ensemble methods
- **📈 Comprehensive Backtesting**: Walk-forward validation, Monte Carlo simulation, and detailed performance analysis
- **🎯 Strategy Optimization**: Automated strategy creation, parameter tuning, and risk management
- **💼 Portfolio Management**: Multi-asset allocation with institutional-grade risk controls
- **🔄 Real-time Adaptation**: Automatic strategy switching based on market regimes

### Natural Language Interface
Ask questions like:
- "Forecast AAPL for the next 30 days using LSTM"
- "Create a momentum strategy for crypto with 5% stop loss"
- "Run a walk-forward backtest on my RSI strategy"
- "Optimize my portfolio for maximum Sharpe ratio"

### Key Features
- **Dynamic Model Creation**: AI-powered model synthesis with multiple frameworks
- **Strategy Auto-Tuning**: Automated strategy optimization and parameter tuning
- **Comprehensive Analytics**: RMSE, MAE, MAPE, Sharpe, Win Rate, Drawdown metrics
- **Real-time Monitoring**: Live system health and performance tracking
- **Multi-format Export**: PDF, Excel, HTML, JSON report generation
- **Cost Modeling**: Commission, slippage, and cash drag analysis

## 🧠 Supported Models and Strategies

### Forecasting Models

#### Traditional Machine Learning
- **Ridge/Lasso/ElasticNet**: Linear models with regularization
- **Random Forest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high accuracy
- **LightGBM**: Fast gradient boosting framework
- **CatBoost**: Categorical feature handling

#### Deep Learning Models
- **LSTM**: Long Short-Term Memory for sequence modeling
- **Transformer**: Attention-based architecture for complex patterns
- **Autoformer**: Auto-correlation mechanism for time series
- **Informer**: Efficient transformer variant
- **MLP**: Multi-layer perceptron for feature learning

#### Time Series Specialists
- **ARIMA**: Auto-regressive integrated moving average
- **Prophet**: Facebook's forecasting tool with seasonality
- **GARCH**: Generalized autoregressive conditional heteroskedasticity
- **Kalman Filter**: State-space modeling for noisy data

#### Ensemble Methods
- **Weighted Combinations**: Optimal model weighting (MSE, Sharpe, Drawdown)
- **Voting Methods**: Majority-based predictions
- **Stacking**: Meta-learning for model combination
- **Hybrid Models**: Risk-aware ensemble weighting

### Trading Strategies

#### Mean Reversion Strategies
- **RSI Strategy**: Relative Strength Index with overbought/oversold signals
- **Bollinger Bands**: Price channel breakout detection
- **Stochastic Oscillator**: Momentum-based reversal signals
- **Williams %R**: Overbought/oversold momentum indicator

#### Trend Following Strategies
- **MACD**: Moving Average Convergence Divergence
- **Moving Average Crossover**: Short/long-term MA signals
- **EMA Strategy**: Exponential Moving Average trends
- **ADX**: Average Directional Index for trend strength

#### Breakout Strategies
- **Donchian Channels**: Price channel breakouts
- **ATR Breakout**: Average True Range volatility breakouts
- **Volatility Breakout**: Standard deviation-based signals

#### Advanced Strategies
- **Multi-timeframe**: Combining signals across timeframes
- **Regime-aware**: Adaptive strategy switching
- **Ensemble Strategy**: Combining multiple strategies
- **Strategy Combinations**: Intersection, union, and weighted modes

### Strategy Customization

#### Parameter Tuning
```python
# Example strategy configuration
strategy_config = {
    'name': 'Custom_RSI_Strategy',
    'parameters': {
        'rsi_period': 14,
        'oversold_threshold': 30,
        'overbought_threshold': 70,
        'position_size': 0.1,
        'stop_loss': 0.05,
        'take_profit': 0.10
    },
    'risk_management': {
        'max_position_size': 0.2,
        'max_daily_loss': 0.02,
        'cooling_period': 300
    }
}
```

#### Custom Strategy Creation
```python
# Natural language strategy creation
"Create a momentum strategy that:
- Uses 20 and 50-day moving averages
- Buys when price crosses above both MAs
- Sells when price crosses below 20-day MA
- Includes 2% stop loss and 6% take profit
- Only trades during market hours"
```

#### Strategy Optimization
- **Grid Search**: Systematic parameter exploration
- **Bayesian Optimization**: Efficient hyperparameter tuning
- **Genetic Algorithm**: Evolutionary strategy optimization
- **Walk-Forward Validation**: Robust out-of-sample testing
- **Optuna Integration**: Advanced hyperparameter optimization

## 🔄 Prompt Handling Flow

The Evolve platform uses an intelligent **multi-layer routing system** that processes natural language requests and routes them to the most appropriate components:

### Routing Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Prompt   │───▶│   LLM Parser    │───▶│  Intent Router  │
│                 │    │                 │    │                 │
│"Forecast AAPL   │    │ • OpenAI GPT-4  │    │ • Request Type  │
│  using LSTM"    │    │ • Local Models  │    │ • Confidence    │
│                 │    │ • Regex Fallback│    │ • Priority      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model Router  │◀───│  Strategy Router│◀───│  Agent Router   │
│                 │    │                 │    │                 │
│ • LSTM Model    │    │ • RSI Strategy  │    │ • Forecast Agent│
│ • Prophet Model │    │ • MACD Strategy │    │ • Strategy Agent│
│ • XGBoost Model │    │ • Custom Logic  │    │ • Analysis Agent│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backtest      │    │   Optimization  │    │   Risk Engine   │
│   Engine        │    │   Engine        │    │                 │
│                 │    │                 │    │                 │
│ • Performance   │    │ • Hyperparameter│    │ • VaR/CVaR      │
│ • Metrics       │    │ • Walk-Forward  │    │ • Position Size │
│ • Visualization │    │ • Regime Detect │    │ • Stop Loss     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │   UI Dashboard  │
                        │                 │
                        │ • Results       │
                        │ • Charts        │
                        │ • Reports       │
                        │ • Controls      │
                        └─────────────────┘
```

### Routing Components

#### 1. **LLM Parser** (`agents/llm/agent.py`)
- **OpenAI GPT-4**: Primary parser for complex requests
- **Local Models**: HuggingFace models for privacy
- **Regex Fallback**: Fast pattern matching for simple requests

#### 2. **Intent Router** (`routing/prompt_router.py`)
- **Request Classification**: Forecast, Strategy, Analysis, Optimization
- **Confidence Scoring**: Determines routing reliability
- **Priority Assignment**: Handles urgent vs. background tasks

#### 3. **Strategy Router** (`trading/strategies/strategy_router.py`)
- **Keyword Matching**: Identifies strategy preferences
- **Relevance Scoring**: Calculates strategy fit
- **Fallback Logic**: Ensures always has a strategy

#### 4. **Model Router** (`trading/models/forecast_router.py`)
- **Data Analysis**: Analyzes time series characteristics
- **Model Selection**: Chooses best model for data
- **Performance History**: Uses past performance for selection

#### 5. **Agent Router** (`trading/agents/prompt_router_agent.py`)
- **Agent Registry**: Maps intents to available agents
- **Load Balancing**: Distributes requests across agents
- **Health Monitoring**: Routes away from failed agents

### Example Routing Flow

```
User: "Create a momentum strategy for TSLA with 5% stop loss"

1. LLM Parser: Extracts intent="strategy", symbol="TSLA", type="momentum", stop_loss=0.05
2. Intent Router: Classifies as STRATEGY request, confidence=0.95
3. Strategy Router: Matches momentum → MACD_Strategy, relevance=0.88
4. Model Router: Selects LSTM for TSLA (good for tech stocks)
5. Agent Router: Routes to StrategyAgent with StrategyOptimizationAgent as backup
6. Backtest Engine: Runs strategy with historical data
7. Risk Engine: Applies 5% stop loss and position sizing
8. UI Dashboard: Displays results with charts and metrics
```

## 🚀 How to Deploy

### Prerequisites
- **Python 3.9+** (3.10 recommended)
- **8GB+ RAM** (16GB+ for large datasets)
- **GPU support** (optional, for deep learning models)
- **Redis** (optional, for caching and coordination)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/evolve-trading.git
   cd evolve-trading
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Environment Variables

Create a `.env` file with the following variables:

```env
# Data Provider API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# Optional: OpenAI for advanced features
OPENAI_API_KEY=your_key_here

# System Configuration
EVOLVE_DEV_MODE=0
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
```

### Production Deployment

#### Docker Deployment
```bash
# Build the image
docker build -t evolve-trading .

# Run with environment variables
docker run -p 8501:8501 \
  -e ALPHA_VANTAGE_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  evolve-trading
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evolve-trading
spec:
  replicas: 3
  selector:
    matchLabels:
      app: evolve-trading
  template:
    metadata:
      labels:
        app: evolve-trading
    spec:
      containers:
      - name: evolve-trading
        image: evolve-trading:latest
        ports:
        - containerPort: 8501
        env:
        - name: ALPHA_VANTAGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: alpha-vantage
```

#### Cloud Deployment

**AWS (ECS/Fargate)**
```bash
# Deploy to ECS
aws ecs create-service \
  --cluster evolve-cluster \
  --service-name evolve-trading \
  --task-definition evolve-trading:1 \
  --desired-count 2
```

**Google Cloud (GKE)**
```bash
# Deploy to GKE
gcloud container clusters create evolve-cluster \
  --num-nodes=3 \
  --machine-type=e2-standard-4

kubectl apply -f k8s/deployment.yaml
```

### Monitoring & Logging

#### Health Checks
```bash
# Check system health
curl http://localhost:8501/health

# Check agent status
curl http://localhost:8501/api/agents/status
```

#### Logging Configuration
```python
# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evolve.log'),
        logging.StreamHandler()
    ]
)
```

## 📊 Sample Screenshots and Diagrams

### Main Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ 🚀 Evolve AI Trading Platform                    [Settings] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📈 Forecast Trade    📊 Strategy Health    🎯 Model Trust │
│  📉 Backtest Results  💰 Portfolio Opt      🔄 Monte Carlo │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Natural Language Interface                              │ │
│  │ "Forecast AAPL for next 30 days using LSTM"            │ │
│  │ [Generate Forecast]                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 📊 Recent Performance                                   │ │
│  │ Sharpe: 1.85 | Win Rate: 68% | Max DD: -12.3%          │ │
│  │ [View Details]                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Forecasting Results
```
┌─────────────────────────────────────────────────────────────┐
│ 📈 AAPL Forecast Results (LSTM Model)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Actual Price: $150.25                                     │
│  Predicted: $152.80 ± $3.45                                │
│  Confidence: 78%                                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Price Chart with Forecast Bands                         │ │
│  │    ┌─┐                                                  │ │
│  │    │ │  ┌─────────────────────────────────────────────┐ │ │
│  │    │ │  │ Forecast Range                              │ │ │
│  │    │ │  └─────────────────────────────────────────────┘ │ │
│  │    └─┘                                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  Model Performance:                                        │
│  • RMSE: 2.34 | MAE: 1.89 | MAPE: 1.26%                   │
│  • Sharpe: 1.45 | Win Rate: 72%                            │
└─────────────────────────────────────────────────────────────┘
```

### Strategy Backtesting
```
┌─────────────────────────────────────────────────────────────┐
│ 📉 RSI Strategy Backtest Results                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Period: 2023-01-01 to 2024-01-01                          │
│  Symbol: TSLA                                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Equity Curve with Monte Carlo Bands                     │ │
│  │    ┌─────────────────────────────────────────────────┐ │ │
│  │    │ 95th Percentile                                │ │ │
│  │    │ ┌─────────────────────────────────────────────┐ │ │ │
│  │    │ │ 50th Percentile (Median)                   │ │ │ │
│  │    │ └─────────────────────────────────────────────┘ │ │ │
│  │    │ 5th Percentile                                 │ │ │
│  │    └─────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  Performance Metrics:                                      │
│  • Total Return: 45.2% | Sharpe: 1.67 | Max DD: -8.9%     │
│  • Win Rate: 71% | Profit Factor: 2.34                     │
│  • Trades: 156 | Avg Hold: 3.2 days                       │
└─────────────────────────────────────────────────────────────┘
```

### Model Comparison Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ 🎯 Model Trust Dashboard                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │ Model       │ Sharpe      │ Win Rate    │ Confidence  │ │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤ │
│  │ LSTM        │ 1.85        │ 72%         │ 78%         │ │
│  │ XGBoost     │ 1.67        │ 68%         │ 82%         │ │
│  │ Transformer │ 1.92        │ 75%         │ 71%         │ │
│  │ Prophet     │ 1.45        │ 65%         │ 85%         │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Model Performance Over Time                             │ │
│  │    ┌─┐                                                  │ │
│  │    │ │  LSTM ──── XGBoost ──── Transformer             │ │
│  │    └─┘                                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Usage Examples

### Forecasting
```python
# Natural language forecasting
"Forecast AAPL for the next 30 days using LSTM"
"Create a neural network for cryptocurrency prediction"
"Generate a 30-day forecast using ensemble methods"
```

### Strategy Development
```python
# Strategy creation and optimization
"Switch to RSI strategy and optimize it"
"Backtest Bollinger Bands strategy on TSLA"
"Create a custom strategy for high volatility markets"
```

### Analysis & Reporting
```python
# Performance analysis
"Generate a performance report for my portfolio"
"Analyze risk metrics for the last 6 months"
"Compare strategy performance across different market regimes"
```

## 🔧 Configuration

### Model Configuration
```python
model_config = {
    'name': 'Custom_LSTM',
    'type': 'LSTM',
    'parameters': {
        'layers': 3,
        'units': 128,
        'dropout': 0.2,
        'learning_rate': 0.001
    },
    'training': {
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2
    }
}
```

### Strategy Configuration
```python
strategy_config = {
    'name': 'RSI_Mean_Reversion',
    'parameters': {
        'rsi_period': 14,
        'oversold_threshold': 30,
        'overbought_threshold': 70,
        'position_size': 0.1
    },
    'risk_management': {
        'stop_loss': 0.05,
        'take_profit': 0.10,
        'max_position_size': 0.2
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/evolve-trading/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/evolve-trading/discussions)
- **Email**: support@evolve-trading.com

---

**Evolve AI Trading Platform** - Transforming natural language into sophisticated trading intelligence. 🚀 