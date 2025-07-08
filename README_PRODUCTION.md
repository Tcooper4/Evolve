# Evolve Trading Platform - Production Ready

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](https://github.com/Tcooper4/Evolve)

**Evolve** is a production-ready, autonomous financial forecasting and trading strategy platform that leverages advanced AI and machine learning to provide institutional-grade trading insights. This version has been completely refactored for production deployment with robust error handling, fallback mechanisms, and modular architecture.

## 🚀 Production Features

### Core Architecture
- **Modular Design**: Clean separation of concerns with independent components
- **Fallback System**: Comprehensive fallback mechanisms for all components
- **Error Handling**: Robust error handling with graceful degradation
- **Logging**: Comprehensive logging with multiple output formats
- **Health Monitoring**: Real-time system health monitoring and reporting

### AI & Machine Learning
- **Multi-Model Ensemble**: LSTM, Transformer, XGBoost, Prophet, ARIMA, GARCH
- **Intelligent Model Selection**: Automatic model selection based on market conditions
- **Market Regime Detection**: Real-time market regime classification
- **Strategy Optimization**: Hyperopt-based strategy parameter optimization
- **Explainable AI**: LLM-powered decision explanations and commentary

### Trading Capabilities
- **Strategy Engine**: RSI, MACD, Bollinger Bands, SMA, EMA, and custom strategies
- **Portfolio Management**: Position tracking, risk management, and rebalancing
- **Backtesting**: Comprehensive backtesting with multiple metrics
- **Real-time Data**: Live market data integration with multiple providers
- **Risk Management**: VaR, CVaR, position sizing, and drawdown protection

### User Interfaces
- **Streamlit Web Interface**: Interactive dashboard with real-time updates
- **Terminal Interface**: Command-line interface for quick operations
- **REST API**: Full REST API for programmatic access
- **Unified Interface**: Single entry point with multiple access methods

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interfaces                          │
├─────────────────────────────────────────────────────────────┤
│  Streamlit │ Terminal │ API │ Unified Interface            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core System                              │
├─────────────────────────────────────────────────────────────┤
│  Agent Hub │ Capability Router │ Fallback System           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Trading Engine                           │
├─────────────────────────────────────────────────────────────┤
│  Data Feed │ Models │ Strategies │ Portfolio │ Risk        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure                           │
├─────────────────────────────────────────────────────────────┤
│  Logging │ Monitoring │ Configuration │ Health Checks      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git
- Redis (optional, for caching)
- MongoDB (optional, for data storage)

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

3. **Install production dependencies**
   ```bash
   pip install -r requirements_production.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize the system**
   ```bash
   python main.py --health
   ```

### Usage

#### Launch Streamlit Interface
```bash
python main.py --streamlit
```
Open your browser to `http://localhost:8501`

#### Launch Terminal Interface
```bash
python main.py --terminal
```

#### Execute Commands
```bash
# Generate forecast
python main.py --command "forecast AAPL 7d"

# Check system health
python main.py --health

# Launch API server
python main.py --api
```

#### Docker Deployment
```bash
# Build the container
docker build -t evolve-platform .

# Run the application
docker run -p 8501:8501 -p 8000:8000 evolve-platform
```

## 📊 System Components

### Fallback System
The system includes comprehensive fallback mechanisms for all components:

- **AgentHub**: Basic agent routing when primary hub unavailable
- **DataFeed**: Mock data generation and yfinance fallback
- **PromptRouter**: Keyword-based routing when AI router fails
- **ModelMonitor**: Mock performance tracking
- **StrategyLogger**: Basic decision logging
- **PortfolioManager**: Mock portfolio operations
- **StrategySelector**: Basic strategy selection
- **MarketRegimeAgent**: Simple regime classification
- **HybridEngine**: Basic strategy execution
- **QuantGPT**: Template-based commentary generation
- **ReportExporter**: Basic report generation

### Core Utilities
- **Math Utils**: Financial calculations (Sharpe ratio, drawdown, etc.)
- **Logging Utils**: Enhanced logging and monitoring
- **Validation Utils**: Input validation and error checking
- **File Utils**: Safe file operations and configuration management
- **Time Utils**: Date/time manipulation and trading calendar

## 🔧 Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Database
REDIS_URL=redis://localhost:6379
MONGODB_URL=mongodb://localhost:27017

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/evolve.log

# System
SYSTEM_MODE=production
FALLBACK_MODE=automatic
```

### Configuration Files
- `config/system_config.yaml`: Main system configuration
- `config/trading_config.yaml`: Trading-specific settings
- `config/models_config.yaml`: Model configurations

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit to gross loss ratio
- **Calmar Ratio**: Annual return to maximum drawdown
- **Beta**: Market correlation
- **Alpha**: Excess return relative to market

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/fallback/
```

### Test Coverage
The system includes comprehensive tests for:
- Unit tests for all components
- Integration tests for component interactions
- Fallback system tests
- Performance and stress tests

## 📊 Monitoring

### Health Checks
```bash
# Check system health
python main.py --health

# Monitor specific components
curl http://localhost:8000/health
```

### Logging
- **File Logging**: Structured logs in `logs/` directory
- **Console Logging**: Real-time console output
- **Redis Logging**: Centralized logging via Redis (optional)
- **Sentry Integration**: Error tracking and monitoring

### Metrics
- **System Metrics**: CPU, memory, disk usage
- **Trading Metrics**: P&L, drawdown, win rate
- **Model Metrics**: Accuracy, precision, recall
- **Performance Metrics**: Response times, throughput

## 🔒 Security

### Authentication
- API key authentication for external access
- Role-based access control
- Secure configuration management

### Data Protection
- Encrypted data storage
- Secure API communication
- Audit logging for all operations

## 🚀 Deployment

### Production Checklist
- [ ] Environment variables configured
- [ ] Database connections tested
- [ ] API keys validated
- [ ] Logging configured
- [ ] Health checks passing
- [ ] Fallback systems tested
- [ ] Performance benchmarks met
- [ ] Security audit completed

### Scaling
- **Horizontal Scaling**: Multiple instances behind load balancer
- **Vertical Scaling**: Resource allocation optimization
- **Database Scaling**: Read replicas and sharding
- **Caching**: Redis for performance optimization

## 📚 API Documentation

### REST API Endpoints
```
GET  /health                    # System health
GET  /forecast/{symbol}         # Generate forecast
POST /strategy/run             # Execute strategy
GET  /portfolio/status         # Portfolio status
GET  /models/performance       # Model performance
POST /backtest/run            # Run backtest
```

### WebSocket Endpoints
```
/ws/forecast                   # Real-time forecasts
/ws/portfolio                  # Portfolio updates
/ws/alerts                     # Trading alerts
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure fallback mechanisms for new components

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [Wiki](https://github.com/Tcooper4/Evolve/wiki)
- **Issues**: [GitHub Issues](https://github.com/Tcooper4/Evolve/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tcooper4/Evolve/discussions)

## 🙏 Acknowledgments

- Built with ❤️ using Python and modern AI/ML technologies
- Inspired by institutional quantitative finance practices
- Leverages state-of-the-art machine learning and deep learning techniques

---

**⭐ Star this repository if you find it helpful!**

**🚀 Ready for production deployment with enterprise-grade reliability and performance.** 