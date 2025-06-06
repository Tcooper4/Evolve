# Trading Dashboard

A comprehensive trading system with advanced models, visualization, and real-time monitoring capabilities.

## Features

- **Advanced Models**
  - LSTM for time series forecasting
  - Transformer models for sequence prediction
  - TCN (Temporal Convolutional Network) for efficient sequence processing
  - Ensemble methods for improved predictions
  - GNN (Graph Neural Networks) for market structure analysis

- **Data Processing**
  - Real-time data ingestion
  - Feature engineering
  - Data validation and cleaning
  - Time series preprocessing

- **Trading Components**
  - Portfolio management
  - Risk assessment
  - Strategy optimization
  - Backtesting framework
  - Execution engine

- **Visualization**
  - Interactive dashboards
  - Real-time monitoring
  - Performance analytics
  - Risk metrics visualization

- **NLP/LLM Integration**
  - News sentiment analysis
  - Market commentary generation
  - Event detection
  - Natural language queries

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/evolve_clean.git
cd evolve_clean

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .[dev]  # For development
# or
pip install -e .  # For production
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Update the configuration in `.env` with your settings:
```env
# API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=your_user
DB_PASSWORD=your_password

# Model Settings
MODEL_PATH=models/
LOG_LEVEL=INFO
```

## Usage

### Starting the Dashboard

```bash
trading-dashboard
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading tests/

# Run specific test file
pytest tests/unit/test_models.py
```

### Development

```bash
# Format code
black trading/
isort trading/

# Type checking
mypy trading/

# Linting
flake8 trading/
```

## Project Structure

```
trading/
├── models/              # Model implementations
│   ├── advanced/       # Advanced models (Transformer, TCN, etc.)
│   └── timeseries/     # Time series specific models
├── data/               # Data handling
├── feature_engineering/# Feature generation
├── visualization/      # Plotting and dashboards
├── web/               # Web interface
├── backtesting/       # Backtesting framework
├── risk/              # Risk management
├── portfolio/         # Portfolio management
└── utils/             # Utility functions
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by various open-source trading projects
- Built with modern Python tools and libraries