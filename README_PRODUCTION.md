# 🚀 Evolve Trading Platform - Production Deployment Guide

## Overview

The Evolve Trading Platform is a production-ready, autonomous financial forecasting and trading system with institutional-grade capabilities. This guide provides comprehensive instructions for deploying the system in production environments.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   API Gateway   │    │   Data Sources  │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (YFinance,    │
│                 │    │                 │    │    AlphaVantage)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent Hub     │    │   Model Engine  │    │   Risk Manager  │
│   (Routing)     │    │   (Forecasting) │    │   (Position     │
│                 │    │                 │    │    Sizing)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Strategy      │    │   Backtesting   │    │   Portfolio     │
│   Engine        │    │   Engine        │    │   Manager       │
│   (Execution)   │    │   (Validation)  │    │   (Tracking)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM
- 4+ CPU cores
- 50GB+ storage
- Internet connection for data feeds

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/evolve-trading-platform.git
   cd evolve-trading-platform
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.production.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run unified_interface.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret

# Optional Configuration
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/evolve
SLACK_WEBHOOK_URL=your_slack_webhook
EMAIL_PASSWORD=your_email_password

# System Configuration
LOG_LEVEL=INFO
DEBUG_MODE=false
ENABLE_GPU=true
MAX_WORKERS=4
```

### Configuration Files

- `config/system_config.yaml` - Main system configuration
- `config/app_config.yaml` - Application-specific settings
- `.streamlit/config.toml` - Streamlit configuration

## 🏭 Production Deployment

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t evolve-trading-platform .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name evolve-platform \
     -p 8501:8501 \
     --env-file .env \
     evolve-trading-platform
   ```

### Kubernetes Deployment

1. **Apply the deployment**
   ```bash
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   kubectl apply -f kubernetes/ingress.yaml
   ```

2. **Check deployment status**
   ```bash
   kubectl get pods -l app=evolve-trading-platform
   kubectl logs -f deployment/evolve-trading-platform
   ```

### Cloud Deployment

#### Heroku
```bash
heroku create evolve-trading-platform
heroku config:set $(cat .env | xargs)
git push heroku main
```

#### AWS ECS
```bash
aws ecs create-cluster --cluster-name evolve-cluster
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster evolve-cluster --service-name evolve-service --task-definition evolve-trading-platform
```

#### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/evolve-trading-platform
gcloud run deploy evolve-trading-platform --image gcr.io/PROJECT_ID/evolve-trading-platform --platform managed
```

## 📊 Monitoring & Health Checks

### Health Check Endpoints

- **Application Health**: `http://localhost:8501/_stcore/health`
- **System Metrics**: `http://localhost:8501/metrics`
- **API Status**: `http://localhost:8501/api/health`

### Monitoring Dashboard

Access the monitoring dashboard at `http://localhost:8501/system` to view:
- System health status
- Performance metrics
- Error logs
- Resource usage

### Logging

Logs are stored in the `logs/` directory:
- `logs/app.log` - Application logs
- `logs/error.log` - Error logs
- `logs/performance.log` - Performance metrics

## 🔒 Security

### API Key Management

- Store API keys in environment variables
- Use secret management services in production
- Rotate keys regularly
- Monitor API usage

### Access Control

- Implement authentication for production deployments
- Use HTTPS in production
- Configure CORS appropriately
- Enable rate limiting

### Data Security

- Encrypt sensitive data at rest
- Use secure connections for data transmission
- Implement audit logging
- Regular security updates

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading --cov=core --cov-report=html

# Run specific test categories
pytest tests/test_models/
pytest tests/test_strategies/
pytest tests/test_integration/
```

### Test Coverage

The system maintains 80%+ test coverage across:
- Model implementations
- Strategy logic
- API endpoints
- Data processing
- Error handling

## 📈 Performance Optimization

### GPU Acceleration

Enable GPU acceleration for model training:
```python
import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
```

### Caching

Redis caching is enabled by default for:
- Model predictions
- Market data
- Strategy results
- User sessions

### Parallel Processing

Configure parallel processing for:
- Model training
- Backtesting
- Data processing
- Report generation

## 🔄 Backup & Recovery

### Automated Backups

Backups are automatically created:
- Daily at 2 AM
- 30-day retention
- Stored in `backups/` directory

### Manual Backup

```bash
python scripts/backup_system.py
```

### Recovery

```bash
python scripts/restore_system.py --backup-file backup_2024-01-01.tar.gz
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.production.txt --force-reinstall
   ```

2. **API Key Issues**
   - Verify API keys in `.env` file
   - Check API key permissions
   - Monitor API usage limits

3. **Memory Issues**
   - Increase system RAM
   - Enable swap space
   - Optimize model parameters

4. **Performance Issues**
   - Enable GPU acceleration
   - Increase worker processes
   - Optimize database queries

### Debug Mode

Enable debug mode for troubleshooting:
```bash
export DEBUG_MODE=true
streamlit run unified_interface.py
```

### Log Analysis

```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep "ERROR" logs/app.log

# Monitor performance
grep "PERFORMANCE" logs/performance.log
```

## 📚 API Documentation

### Core Endpoints

- `POST /api/forecast` - Generate forecasts
- `POST /api/strategy` - Execute strategies
- `POST /api/backtest` - Run backtests
- `GET /api/health` - Health check
- `GET /api/metrics` - System metrics

### Example API Usage

```python
import requests

# Generate forecast
response = requests.post('http://localhost:8501/api/forecast', json={
    'symbol': 'AAPL',
    'days': 30,
    'model': 'ensemble'
})

# Execute strategy
response = requests.post('http://localhost:8501/api/strategy', json={
    'symbol': 'TSLA',
    'strategy': 'RSI',
    'action': 'BUY'
})
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run linting
flake8 trading/ core/ unified_interface.py

# Run type checking
mypy trading/ core/ unified_interface.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/evolve-trading-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/evolve-trading-platform/discussions)
- **Email**: support@evolve-trading.com

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a complete list of changes and version history.

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software. 