# Advanced Financial Forecasting Platform

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

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-platform.git
   cd trading-platform
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
