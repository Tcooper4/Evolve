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

### Docker Deployment

```bash
# Build the container
docker build -t evolve-platform .

# Run the application
docker run -p 8501:8501 evolve-platform
```

## 📊 Usage Examples

### Generate Market Forecasts
```python
from trading.models.ensemble_model import EnsembleModel
from trading.data.providers.yfinance_provider import YFinanceProvider

# Initialize data provider and model
provider = YFinanceProvider()
model = EnsembleModel()

# Get forecast for AAPL
forecast = model.predict("AAPL", days=30)
```

### Backtest Trading Strategy
```python
from trading.strategies.rsi_strategy import RSIStrategy
from trading.backtesting.backtester import Backtester

# Create strategy and backtester
strategy = RSIStrategy(period=14, overbought=70, oversold=30)
backtester = Backtester(strategy)

# Run backtest
results = backtester.run("AAPL", start_date="2023-01-01", end_date="2023-12-31")
```

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