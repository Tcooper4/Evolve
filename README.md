# Agentic Trading Forecasting Platform

A modular and autonomous trading platform that combines traditional trading strategies with advanced AI capabilities.

## Project Structure

```
.
├── core/                 # Core AI and routing logic
│   ├── agents/          # Cognitive AI agents
│   ├── router/          # Request routing
│   └── base/            # Base agent logic
├── trading/             # Trading components
│   ├── strategies/      # Trading strategies
│   ├── agents/          # Trading agents
│   ├── signals/         # Signal generation
│   └── optimization/    # Strategy optimization
├── system/              # System infrastructure
│   └── infra/          # System monitoring and automation
├── models/              # ML models and training
├── data/               # Data processing and storage
├── visualization/      # Visualization components
├── evaluation/         # Performance evaluation
├── execution/          # Trade execution
└── tests/              # Test suite
```

## Features

- Modular architecture for easy extension
- AI-powered trading strategies
- Real-time market analysis
- Automated trading execution
- Performance monitoring
- Self-improving agents
- Dynamic strategy optimization

## Installation

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
pip install -r requirements-dev.txt  # For development
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Running the Dashboard

```bash
streamlit run app.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

### Development

```bash
# Lint code
make lint

# Format code
make format

# Type check
make typecheck

# Run all checks
make check
```

## Configuration

- `.env`: Environment variables
- `config.yaml`: Application configuration
- `prometheus.yml`: Monitoring configuration

## Docker Support

Build and run with Docker:

```bash
docker build -t trading-platform .
docker run -p 8501:8501 trading-platform
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

- All sensitive information is stored in environment variables
- API keys and credentials are encrypted
- Regular security audits are performed
- Logs are rotated and monitored

## Monitoring

- Prometheus metrics
- Grafana dashboards
- System health checks
- Performance monitoring
- Error tracking

## Support

For support, please open an issue in the GitHub repository. 