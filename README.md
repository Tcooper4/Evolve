# Evolve Clean Trading Dashboard

A comprehensive trading dashboard with AI-powered analysis, portfolio management, and risk assessment capabilities.

## Features

- **AI-Powered Analysis**: Natural language interface for market analysis, strategy generation, and risk assessment
- **Portfolio Management**: Track and optimize your trading portfolio
- **Risk Management**: Comprehensive risk analysis and monitoring
- **Backtesting**: Test trading strategies with historical data
- **Market Analysis**: Real-time market data analysis and visualization
- **ML Models**: Advanced machine learning models for prediction and optimization
- **Streamlit Dashboard**: Modern, interactive web interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd evolve_clean
```

2. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your OpenAI API key (optional)
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run app.py
```

2. Access the dashboard at `http://localhost:8501`

### Dashboard Features

- **AI Assistant**: Ask questions about trading, market analysis, and portfolio management
- **Portfolio Overview**: View portfolio performance, metrics, and recent trades
- **Trading Strategies**: Configure and monitor active trading strategies
- **Risk Management**: Monitor risk metrics and adjust portfolio allocation
- **Backtesting**: Test strategies with historical data
- **Market Analysis**: View market trends and indicators
- **ML Models**: Monitor and update machine learning models
- **Settings**: Configure API keys and other settings

## Project Structure

```
evolve_clean/
├── app.py                 # Main Streamlit application
├── config/               # Configuration files
├── dashboard/           # React dashboard components
├── scripts/             # Management and utility scripts
├── tests/              # Test files
├── trading/            # Core trading modules
│   ├── analysis/       # Market analysis
│   ├── backtesting/    # Backtesting engine
│   ├── evaluation/     # Model evaluation
│   ├── feature_engineering/ # Feature generation
│   ├── llm/           # LLM interface
│   ├── nlp/           # Natural language processing
│   ├── optimization/  # Portfolio optimization
│   ├── portfolio/     # Portfolio management
│   ├── risk/          # Risk management
│   ├── strategies/    # Trading strategies
│   └── utils/         # Utility functions
├── requirements.txt    # Main dependencies
└── requirements_streamlit.txt # Streamlit-specific dependencies
```

## Configuration

The application uses several configuration files:

- `config/app_config.yaml`: Main application configuration
- `config/logging_config.yaml`: Logging configuration
- `config.yaml`: User configuration

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Run pre-commit hooks
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the repository or contact the development team. 