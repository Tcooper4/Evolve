# Advanced Financial Forecasting Platform

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Tcooper4/Evolve)](https://github.com/Tcooper4/Evolve/blob/main/LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Agentic%20Dashboard-red)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/Tcooper4/Evolve/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An intelligent, agentic trading platform built with Streamlit, featuring advanced forecasting models, strategy optimization, and natural language processing capabilities.

## Features

- 🤖 **Intelligent Agents**
  - Forecaster: Advanced time series prediction
  - Strategy: Dynamic trading strategy optimization
  - Commentary: Natural language market analysis
  - Backtester: Historical performance validation
  - Updater: Continuous model improvement

- 📊 **Advanced Analytics**
  - Multi-timeframe analysis
  - Technical indicators
  - Risk metrics
  - Performance visualization
  - Real-time updates

- 🧠 **LLM Integration**
  - OpenAI and HuggingFace support
  - Natural language query processing
  - Context-aware responses
  - Multi-step reasoning
  - Confidence scoring

- 🔄 **Auto-Repair System**
  - Package dependency management
  - DLL error handling
  - Environment validation
  - Automatic recovery
  - Health monitoring

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Tcooper4/Evolve.git
   cd Evolve
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   # Fix common install issues
   pip install --upgrade pip setuptools wheel
   
   # Install project dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

5. Launch the application:
   ```bash
   # Start the application
   streamlit run app.py
   
   # If you encounter any issues:
   # 1. Ensure all dependencies are installed
   # 2. Check that .env is properly configured
   # 3. Verify Python version (3.11+)
   # 4. Try clearing Streamlit cache:
   #    streamlit cache clear
   ```

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User Input    │────▶│  Prompt Agent   │────▶│  LLM Interface  │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                                 ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Performance    │◀────│  Strategy Agent │◀────│  Forecast Agent │
│    Memory       │     └─────────────────┘     └─────────────────┘
└─────────────────┘
```

### Core Components

1. **Prompt Agent**: Processes natural language requests
2. **Forecast Agent**: Generates price predictions
3. **Strategy Agent**: Optimizes trading strategies
4. **LLM Interface**: Manages language model interactions
5. **Performance Memory**: Tracks and stores metrics

## 🔧 Configuration

The platform uses a layered configuration system:

1. **Environment Variables** (`.env`):
   - API keys
   - Database credentials
   - Feature flags

2. **Application Config** (`config.yaml`):
   - Agent settings
   - Model parameters
   - System limits

3. **Development Config** (`config.dev.yaml`):
   - Debug settings
   - Test parameters
   - Local overrides

## 📂 Directory Structure

```
Evolve/
├── trading/              # Core trading logic
│   ├── agents/          # Agent implementations
│   ├── models/          # ML models
│   ├── utils/           # Utilities
│   └── meta_agents/     # Meta-agent orchestration
├── tests/               # Test suite
├── scripts/             # Utility scripts
├── config/              # Configuration files
├── data/                # Data storage
└── docs/                # Documentation
```

## 🧪 Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --e2e

# Run with coverage
python run_tests.py --coverage
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatters
black .
isort .
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://github.com/Tcooper4/Evolve/wiki)
- [Issue Tracker](https://github.com/Tcooper4/Evolve/issues)
- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the web framework
- [OpenAI](https://openai.com) for LLM capabilities
- [Polygon.io](https://polygon.io) for market data
- All our contributors and users

## Agent Roles

### Forecaster Agent
- Time series analysis and prediction
- Multiple model support (ARIMA, LSTM, Prophet)
- Confidence scoring and uncertainty estimation
- Real-time updates and adaptation

### Strategy Agent
- Technical indicator analysis
- Risk management optimization
- Position sizing and entry/exit rules
- Performance monitoring and adjustment

### Commentary Agent
- Market sentiment analysis
- News and event impact assessment
- Natural language report generation
- Trend and pattern identification

### Backtester Agent
- Historical data validation
- Strategy performance testing
- Risk metric calculation
- Optimization parameter tuning

### Updater Agent
- Model weight optimization
- Performance metric tracking
- Adaptive learning
- Memory management

## Prompt Examples

### Forecasting
```
"Forecast AAPL price for the next 7 days with high confidence"
"Predict SPY trend for the next month using technical indicators"
"Generate a 30-day forecast for BTC with uncertainty bounds"
```

### Strategy
```
"Optimize a momentum strategy for MSFT with 0.8 risk tolerance"
"Create a mean reversion strategy for ETH with stop loss"
"Backtest a hybrid strategy combining ML and technical indicators"
```

### Analysis
```
"Analyze the current market conditions for tech stocks"
"Explain the recent price movement in TSLA"
"Compare the performance of different sectors"
```

## Auto-Repair System

The platform includes a robust auto-repair system that handles common issues:

1. **Package Management**
   - Detects missing dependencies
   - Updates outdated packages
   - Resolves version conflicts
   - Handles DLL issues

2. **Environment Validation**
   - Checks Python version
   - Verifies GPU availability
   - Validates memory access
   - Tests data loading

3. **Error Recovery**
   - Automatic retry logic
   - Fallback mechanisms
   - Error logging
   - State recovery

4. **Health Monitoring**
   - System resource checks
   - Agent status monitoring
   - Performance metrics
   - Memory usage tracking

## Deployment

### Docker
```bash
docker build -t trading-platform .
docker run -p 8501:8501 trading-platform
```

### Heroku
```bash
heroku create your-app-name
git push heroku main
```
