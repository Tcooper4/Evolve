# Evolve: Autonomous Financial Forecasting & Trading Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/Tcooper4/Evolve)

**Evolve** is an autonomous financial forecasting and trading strategy platform that leverages multiple machine learning models to predict stock price movements, generate technical trading signals, backtest strategies, and visualize performance through an interactive Streamlit dashboard. The platform combines traditional quantitative finance techniques with cutting-edge AI capabilities to provide comprehensive market analysis and automated trading insights.

## 🚀 Features

- **🤖 ML Forecasting**: LSTM, XGBoost, Prophet, ARIMA, and ensemble models
- **🧠 Agentic AI**: Intelligent prompt interpreter using GPT and Hugging Face models
- **📊 Strategy Engine**: RSI, MACD, Bollinger Bands, and custom technical indicators
- **📈 Backtesting**: Comprehensive analysis with Sharpe ratio, drawdown, and win rate metrics
- **🎨 Signal Visualization**: Interactive charts and equity curve rendering
- **⚖️ Auto-weighted Ensemble**: Intelligent model selection and combination
- **🖥️ Streamlit UI**: End-to-end interactive dashboard for seamless user experience
- **📊 Real-time Monitoring**: Live performance tracking and system health metrics

## 🔮 Unified Interface

Evolve provides a **Unified Interface** that gives you access to all features through multiple access methods:

### Access Methods
- **🌐 Streamlit Web Interface**: Interactive dashboards and forms
- **💻 Terminal Command Line**: Quick command execution
- **🤖 Natural Language**: Ask questions in plain English via QuantGPT
- **⚡ Direct Commands**: Execute specific actions with parameters

### Quick Start
```bash
# Launch Streamlit interface
streamlit run app.py

# Use terminal interface
python unified_interface.py --terminal

# Execute commands directly
python unified_interface.py --command "forecast AAPL 7d"

# Ask natural language questions
python unified_interface.py --command "What's the best model for TSLA?"
```

### Available Commands
- **Forecasting**: `forecast AAPL 30d`, `predict BTCUSDT 1h`
- **Tuning**: `tune model lstm AAPL`, `optimize strategy rsi`
- **Strategies**: `strategy list`, `strategy run bollinger AAPL`
- **Portfolio**: `portfolio status`, `portfolio rebalance`
- **Agents**: `agent list`, `agent status`
- **Reports**: `report generate AAPL`, `performance report`

### Natural Language Examples
- "What's the best model for NVDA over 90 days?"
- "Should I long TSLA this week?"
- "Analyze BTCUSDT market conditions"
- "What's the trading signal for AAPL?"

### Interface Features
- **Command Routing**: Automatically routes commands to appropriate services
- **Error Handling**: Graceful error handling and user feedback
- **Help System**: Comprehensive help and documentation
- **Status Monitoring**: Real-time system status and health checks
- **Integration**: Seamless integration with all existing services

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md) and [README_UNIFIED_INTERFACE.md](README_UNIFIED_INTERFACE.md).

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Streamlit** - Interactive web application framework
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Plotly** - Interactive data visualization

### Machine Learning & AI
- **TensorFlow & PyTorch** - Deep learning frameworks
- **scikit-learn** - Traditional ML algorithms
- **XGBoost** - Gradient boosting for time series
- **Prophet** - Facebook's forecasting tool
- **OpenAI GPT & Hugging Face** - Large language models for agentic behavior

### Financial Data & Analysis
- **yfinance** - Yahoo Finance data provider
- **Alpha Vantage** - Alternative market data source
- **pandas-ta** - Technical analysis indicators
- **backtrader** - Backtesting framework
- **empyrical** - Financial risk metrics

### Infrastructure & Monitoring
- **Docker** - Containerization
- **Prometheus & Grafana** - Monitoring and visualization
- **Redis & MongoDB** - Data storage and caching
- **Kubernetes** - Orchestration (optional)

## 🏗️ Architecture

```
Raw Market Data → Feature Engineering → Forecasting Models → Strategy Engine → Signal Visualizer → Backtest Reports → Streamlit Dashboard
```

### System Components
- **Data Layer**: Market data ingestion and preprocessing
- **ML Pipeline**: Model training, validation, and ensemble selection
- **Strategy Engine**: Technical indicator calculation and signal generation
- **Backtesting**: Historical performance analysis and risk metrics
- **Visualization**: Interactive charts and performance dashboards
- **API Layer**: RESTful services for external integrations

## 🚀 Quickstart

### Prerequisites
- Python 3.8 or higher
- Git
- (Optional) Docker

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tcooper4/Evolve
   cd Evolve
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp config.example.json config.json
   # Edit config.json with your API keys and preferences
   ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard**
   Open your browser and navigate to `http://localhost:8501`

### Live Demo Instructions

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Navigate to Forecast & Trade**
   - Click on "📈 Forecast & Trade" in the sidebar
   - This is the main interface for generating forecasts

3. **Try AI Agent Interface**
   - Enter a natural language request like "Forecast AAPL for the next 30 days"
   - Click "🚀 Process with AI Agent"
   - Watch the AI route your request and generate insights

4. **Use Traditional Interface**
   - Enter a ticker symbol (e.g., AAPL, GOOGL, TSLA)
   - Select date range for historical data
   - Choose a model (LSTM, Transformer, XGBoost, Ensemble)
   - Click "Generate Forecast"

5. **Explore Other Features**
   - **Portfolio Dashboard**: View portfolio performance and allocations
   - **Strategy History**: See historical strategy performance
   - **Risk Dashboard**: Analyze risk metrics and drawdowns
   - **System Scorecard**: Monitor system health and performance

### Quick Examples

**Generate a Forecast:**
```python
# Using the unified interface
python unified_interface.py --command "forecast AAPL 30d"

# Using the API
from trading.models.ensemble_model import EnsembleModel
model = EnsembleModel()
forecast = model.predict("AAPL", days=30)
```

**Run a Backtest:**
```python
# Using the unified interface
python unified_interface.py --command "backtest rsi AAPL 2023-01-01 2023-12-31"

# Using the API
from trading.strategies.rsi_strategy import RSIStrategy
from trading.backtesting.backtester import Backtester
strategy = RSIStrategy(period=14, overbought=70, oversold=30)
backtester = Backtester(strategy)
results = backtester.run("AAPL", start_date="2023-01-01", end_date="2023-12-31")
```

**Ask AI Questions:**
```python
# Using the unified interface
python unified_interface.py --command "What's the best model for TSLA?"

# Using natural language
python unified_interface.py --command "Should I long AAPL this week?"
```

### Docker Deployment

```bash
# Build the container
docker build -t evolve-platform .

# Run the application
docker run -p 8501:8501 evolve-platform
```

## 📊 Model & Strategy Overview

### Forecasting Models

**LSTM (Long Short-Term Memory)**
- Best for: Time series with long-term dependencies
- Use case: Stock price prediction with trend analysis
- Performance: High accuracy for volatile markets

**Transformer**
- Best for: Complex patterns and attention mechanisms
- Use case: Multi-timeframe analysis
- Performance: Excellent for capturing market regime changes

**XGBoost**
- Best for: Tabular data with feature interactions
- Use case: Technical indicator-based prediction
- Performance: Fast training and good interpretability

**Ensemble**
- Best for: Combining multiple model strengths
- Use case: Robust prediction with uncertainty quantification
- Performance: Most reliable across different market conditions

### Trading Strategies

**RSI (Relative Strength Index)**
- Signal: Buy when RSI < 30, Sell when RSI > 70
- Best for: Mean reversion trading
- Risk: Medium

**MACD (Moving Average Convergence Divergence)**
- Signal: Buy when MACD crosses above signal line
- Best for: Trend following
- Risk: Low to Medium

**Bollinger Bands**
- Signal: Buy when price touches lower band, Sell when price touches upper band
- Best for: Volatility-based trading
- Risk: Medium

**Custom Strategies**
- Signal: Based on custom technical indicators
- Best for: Specific market conditions
- Risk: Variable

## 📁 Project Structure

```
Evolve/
├── app.py                 # Main Streamlit application
├── trading/              # Core trading components
│   ├── models/          # ML forecasting models
│   ├── strategies/      # Trading strategies
│   ├── backtesting/     # Backtesting engine
│   ├── data/           # Data providers and preprocessing
│   └── optimization/   # Strategy optimization
├── pages/              # Streamlit page modules
├── core/               # Core AI and routing logic
├── system/             # Infrastructure and monitoring
├── tests/              # Comprehensive test suite
├── docs/               # Documentation and guides
└── config/             # Configuration files
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/strategies/
```

## 📈 Performance Metrics

The platform tracks and visualizes key performance indicators:

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit to gross loss ratio
- **Calmar Ratio**: Annual return to maximum drawdown

## 📊 Report Generation System

The Evolve trading system includes a comprehensive reporting system that automatically generates detailed reports after each forecast and strategy execution.

### Features

- **Trade Reports**: PnL analysis, win rate, average gains/losses, Sharpe ratio, drawdown
- **Model Reports**: MSE, MAE, RMSE, accuracy, precision, recall, F1 score  
- **Strategy Reasoning**: GPT-powered analysis of why actions were taken
- **Multiple Formats**: PDF, Markdown, and HTML output
- **Integrations**: Slack, Notion, and email notifications
- **Automated Service**: Redis pub/sub service for automatic report generation
- **Visualizations**: Equity curves, prediction vs actual charts, PnL distributions

### Quick Usage

```python
from trading.report.report_generator import generate_quick_report

# Generate comprehensive report
report_data = generate_quick_report(
    trade_data=trade_data,
    model_data=model_data, 
    strategy_data=strategy_data,
    symbol='AAPL',
    timeframe='1h',
    period='7d'
)

print(f"Report generated: {report_data['report_id']}")
print(f"Files: {report_data['files']}")
```

### Service Integration

```python
from trading.services.service_client import ServiceClient

client = ServiceClient()

# Trigger automated report generation
event_id = client.trigger_strategy_report(
    strategy_data=strategy_data,
    trade_data=trade_data,
    model_data=model_data,
    symbol='AAPL',
    timeframe='1h',
    period='7d'
)

# Generate report directly
report_data = client.generate_report(
    trade_data=trade_data,
    model_data=model_data,
    strategy_data=strategy_data,
    symbol='AAPL',
    timeframe='1h',
    period='7d'
)
```

### Report Service

Start the automated report service:

```bash
# Start report service
python trading/report/launch_report_service.py

# Or via service manager
python trading/services/service_manager.py start report
```

The report service automatically listens for:
- `forecast_completed` events
- `strategy_completed` events  
- `backtest_completed` events
- `model_evaluation_completed` events

### Configuration

Set environment variables for integrations:

```bash
# OpenAI for GPT reasoning
export OPENAI_API_KEY="your_openai_api_key"

# Integrations (optional)
export NOTION_TOKEN="your_notion_token"
export SLACK_WEBHOOK="your_slack_webhook_url"

# Email configuration (optional)
export EMAIL_SMTP_SERVER="smtp.gmail.com"
export EMAIL_SMTP_PORT="587"
export EMAIL_USERNAME="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
export EMAIL_FROM="your_email@gmail.com"
export EMAIL_TO="recipient@example.com"
```

### Demo

Run the demo to see the system in action:

```bash
python trading/report/demo_report_generation.py
```

### Testing

```bash
# Run report system tests
python trading/report/test_report_system.py
```

For detailed documentation, see [trading/report/README.md](trading/report/README.md).

## 🤖 Reasoning Logger System

A comprehensive system for recording, displaying, and analyzing agent decisions in plain language for transparency and explainability.

### Features

- **Decision Logging**: Record every decision agents make with detailed context
- **Plain Language Summaries**: Human-readable summaries of complex decisions
- **Chat-Style Explanations**: Conversational explanations of why actions were taken
- **Real-Time Updates**: Live monitoring of agent decisions via Redis
- **Multiple Display Formats**: Terminal and Streamlit interfaces
- **Search & Filter**: Find and analyze specific decisions
- **Statistics & Analytics**: Comprehensive decision analytics

### Quick Usage

```python
from trading.utils.reasoning_logger import ReasoningLogger, DecisionType, ConfidenceLevel

# Initialize logger
logger = ReasoningLogger()

# Log a decision
decision_id = logger.log_decision(
    agent_name='LSTMForecaster',
    decision_type=DecisionType.FORECAST,
    action_taken='Predicted AAPL will reach $185.50 in 7 days',
    context={
        'symbol': 'AAPL',
        'timeframe': '1h',
        'market_conditions': {'trend': 'bullish', 'rsi': 65},
        'available_data': ['price', 'volume', 'rsi', 'macd'],
        'constraints': {'max_forecast_days': 30},
        'user_preferences': {'risk_tolerance': 'medium'}
    },
    reasoning={
        'primary_reason': 'Strong technical indicators showing bullish momentum',
        'supporting_factors': [
            'RSI indicates bullish momentum (65)',
            'MACD shows positive crossover',
            'Volume is above average'
        ],
        'alternatives_considered': [
            'Conservative forecast of $180.00',
            'Aggressive forecast of $190.00'
        ],
        'risks_assessed': [
            'Market volatility could increase',
            'Earnings announcement next week'
        ],
        'confidence_explanation': 'High confidence due to strong technical signals',
        'expected_outcome': 'AAPL expected to continue bullish trend'
    },
    confidence_level=ConfidenceLevel.HIGH,
    metadata={'model_name': 'LSTM_v2', 'forecast_value': 185.50}
)

print(f"Decision logged: {decision_id}")
```

### Service Integration

```python
from trading.services.service_client import ServiceClient

client = ServiceClient()

# Log a reasoning decision
decision_id = client.log_reasoning_decision({
    'agent_name': 'LSTMForecaster',
    'decision_type': 'FORECAST',
    'action_taken': 'Predicted AAPL will reach $185.50',
    'context': {
        'symbol': 'AAPL',
        'timeframe': '1h',
        'market_conditions': {'trend': 'bullish'},
        'available_data': ['price', 'volume'],
        'constraints': {},
        'user_preferences': {}
    },
    'reasoning': {
        'primary_reason': 'Technical analysis shows bullish momentum',
        'supporting_factors': ['RSI oversold', 'MACD positive'],
        'alternatives_considered': ['Wait for confirmation'],
        'risks_assessed': ['Market volatility'],
        'confidence_explanation': 'High confidence due to clear signals',
        'expected_outcome': 'Expected 5% upside'
    },
    'confidence_level': 'HIGH',
    'metadata': {'forecast_value': 185.50}
})

# Get reasoning decisions
decisions = client.get_reasoning_decisions(agent_name='LSTMForecaster', limit=10)

# Get reasoning statistics
stats = client.get_reasoning_statistics()
```

### Reasoning Service

Start the automated reasoning service:

```bash
# Start reasoning service
python trading/utils/launch_reasoning_service.py

# Or via service manager
python trading/services/service_manager.py start reasoning
```

The reasoning service automatically listens for:
- `agent_decisions` events
- `forecast_completed` events
- `strategy_completed` events
- `model_evaluation_completed` events

And publishes `reasoning_updates` events with decision summaries.

### Display Components

```python
from trading.utils.reasoning_display import ReasoningDisplay

# Initialize display
display = ReasoningDisplay(logger)

# Display recent decisions in terminal
display.display_recent_decisions_terminal(limit=10)

# Display specific decision
decision = logger.get_decision(decision_id)
display.display_decision_terminal(decision)

# Display statistics
display.display_statistics_terminal()
```

### Streamlit Dashboard

```python
# Create complete reasoning page
from trading.utils.reasoning_display import create_reasoning_page_streamlit

# In your Streamlit app
create_reasoning_page_streamlit()
```

### Demo

Run the demo to see the system in action:

```bash
python trading/utils/demo_reasoning.py
```

### Testing

```bash
# Run reasoning system tests
python trading/utils/test_reasoning.py
```

For detailed documentation, see [trading/utils/README.md](trading/utils/README.md).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **GitHub**: [@Tcooper4](https://github.com/Tcooper4)
- **Project Link**: [https://github.com/Tcooper4/Evolve](https://github.com/Tcooper4/Evolve)

## 🙏 Acknowledgments

- Built with ❤️ using Python and Streamlit
- Inspired by modern quantitative finance practices
- Leverages state-of-the-art machine learning techniques

---

**⭐ Star this repository if you find it helpful!** 
**⭐ Star this repository if you find it helpful!** 