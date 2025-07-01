# Examples & Demos

This directory contains practical examples and demonstrations of the Evolve trading system.

## 📁 Directory Structure

```
examples/
├── README.md                    # This file
├── forecasting_example.py       # Basic forecasting example
├── backtesting_example.py       # RSI strategy backtesting
├── ensemble_example.py          # Ensemble model usage
├── strategy_comparison.py       # Compare multiple strategies
├── portfolio_management.py      # Portfolio allocation example
├── risk_analysis.py            # Risk metrics calculation
└── notebooks/                  # Jupyter notebooks
    ├── 01_quick_start.ipynb    # Quick start guide
    ├── 02_forecasting.ipynb    # Forecasting tutorial
    ├── 03_backtesting.ipynb    # Backtesting tutorial
    └── 04_advanced_features.ipynb # Advanced features
```

## 🚀 Quick Start Examples

### 1. Basic Forecasting

```python
# examples/forecasting_example.py
from trading.models.ensemble_model import EnsembleModel
from trading.data.providers.yfinance_provider import YFinanceProvider

def forecast_aapl():
    """Generate a 30-day forecast for AAPL using ensemble model."""
    
    # Initialize components
    provider = YFinanceProvider()
    model = EnsembleModel()
    
    # Get historical data
    data = provider.fetch("AAPL", "1d", start_date="2023-01-01")
    
    # Generate forecast
    forecast = model.predict(data, horizon=30)
    
    # Display results
    print(f"AAPL Forecast for next 30 days:")
    print(f"Predicted Price: ${forecast['predicted_price']:.2f}")
    print(f"Confidence Interval: ${forecast['lower']:.2f} - ${forecast['upper']:.2f}")
    
    return forecast

if __name__ == "__main__":
    forecast_aapl()
```

### 2. RSI Strategy Backtesting

```python
# examples/backtesting_example.py
from trading.strategies.rsi_strategy import RSIStrategy
from trading.backtesting.backtester import Backtester
from trading.data.providers.yfinance_provider import YFinanceProvider

def backtest_rsi_strategy():
    """Backtest RSI strategy on AAPL for 2023."""
    
    # Initialize components
    provider = YFinanceProvider()
    strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    backtester = Backtester(strategy)
    
    # Get data
    data = provider.fetch("AAPL", "1d", start_date="2023-01-01", end_date="2023-12-31")
    
    # Run backtest
    results = backtester.run(data)
    
    # Display results
    print(f"RSI Strategy Backtest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    
    return results

if __name__ == "__main__":
    backtest_rsi_strategy()
```

### 3. Ensemble Model Usage

```python
# examples/ensemble_example.py
from trading.models.ensemble_model import EnsembleModel
from trading.data.providers.yfinance_provider import YFinanceProvider
import pandas as pd

def ensemble_forecast():
    """Demonstrate ensemble model with multiple tickers."""
    
    # Initialize
    provider = YFinanceProvider()
    model = EnsembleModel()
    
    # List of tickers to forecast
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    results = {}
    
    for ticker in tickers:
        print(f"\nForecasting {ticker}...")
        
        # Get data
        data = provider.fetch(ticker, "1d", start_date="2023-01-01")
        
        # Generate forecast
        forecast = model.predict(data, horizon=30)
        
        results[ticker] = {
            'current_price': data['close'].iloc[-1],
            'predicted_price': forecast['predicted_price'],
            'confidence': forecast['confidence']
        }
        
        print(f"Current: ${results[ticker]['current_price']:.2f}")
        print(f"Predicted: ${results[ticker]['predicted_price']:.2f}")
        print(f"Confidence: {results[ticker]['confidence']:.2%}")
    
    return results

if __name__ == "__main__":
    ensemble_forecast()
```

## 📊 Strategy Comparison Example

```python
# examples/strategy_comparison.py
from trading.strategies.rsi_strategy import RSIStrategy
from trading.strategies.macd_strategy import MACDStrategy
from trading.strategies.bollinger_strategy import BollingerStrategy
from trading.backtesting.backtester import Backtester
from trading.data.providers.yfinance_provider import YFinanceProvider

def compare_strategies():
    """Compare multiple strategies on the same data."""
    
    # Initialize
    provider = YFinanceProvider()
    data = provider.fetch("AAPL", "1d", start_date="2023-01-01", end_date="2023-12-31")
    
    # Define strategies
    strategies = {
        'RSI': RSIStrategy(period=14, overbought=70, oversold=30),
        'MACD': MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        'Bollinger': BollingerStrategy(period=20, std_dev=2)
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\nTesting {name} strategy...")
        
        backtester = Backtester(strategy)
        result = backtester.run(data)
        
        results[name] = result
        
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    
    return results

if __name__ == "__main__":
    compare_strategies()
```

## 📈 Portfolio Management Example

```python
# examples/portfolio_management.py
from trading.portfolio.portfolio_manager import PortfolioManager
from trading.data.providers.yfinance_provider import YFinanceProvider

def portfolio_example():
    """Demonstrate portfolio management features."""
    
    # Initialize
    provider = YFinanceProvider()
    portfolio = PortfolioManager()
    
    # Define portfolio
    positions = {
        'AAPL': 100,
        'GOOGL': 50,
        'MSFT': 75,
        'TSLA': 25
    }
    
    # Get current prices
    current_prices = {}
    for ticker in positions.keys():
        data = provider.fetch(ticker, "1d", start_date="2023-12-01")
        current_prices[ticker] = data['close'].iloc[-1]
    
    # Calculate portfolio value
    total_value = sum(positions[ticker] * current_prices[ticker] for ticker in positions)
    
    # Calculate allocations
    allocations = {}
    for ticker in positions:
        allocations[ticker] = (positions[ticker] * current_prices[ticker]) / total_value
    
    print("Portfolio Analysis:")
    print(f"Total Value: ${total_value:,.2f}")
    print("\nAllocations:")
    for ticker, allocation in allocations.items():
        print(f"{ticker}: {allocation:.2%}")
    
    return {
        'total_value': total_value,
        'allocations': allocations,
        'positions': positions,
        'current_prices': current_prices
    }

if __name__ == "__main__":
    portfolio_example()
```

## 🛡️ Risk Analysis Example

```python
# examples/risk_analysis.py
from trading.risk.risk_manager import RiskManager
from trading.data.providers.yfinance_provider import YFinanceProvider
import numpy as np

def risk_analysis_example():
    """Demonstrate risk analysis features."""
    
    # Initialize
    provider = YFinanceProvider()
    risk_manager = RiskManager()
    
    # Get data for multiple assets
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    returns_data = {}
    
    for ticker in tickers:
        data = provider.fetch(ticker, "1d", start_date="2023-01-01")
        returns_data[ticker] = data['close'].pct_change().dropna()
    
    # Calculate risk metrics
    risk_metrics = {}
    
    for ticker, returns in returns_data.items():
        metrics = risk_manager.calculate_risk_metrics(returns)
        risk_metrics[ticker] = metrics
        
        print(f"\nRisk Metrics for {ticker}:")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"VaR (95%): {metrics['var_95']:.2%}")
        print(f"CVaR (95%): {metrics['cvar_95']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Calculate portfolio risk
    portfolio_returns = np.mean(list(returns_data.values()), axis=0)
    portfolio_metrics = risk_manager.calculate_risk_metrics(portfolio_returns)
    
    print(f"\nPortfolio Risk Metrics:")
    print(f"Portfolio Volatility: {portfolio_metrics['volatility']:.2%}")
    print(f"Portfolio VaR (95%): {portfolio_metrics['var_95']:.2%}")
    
    return risk_metrics

if __name__ == "__main__":
    risk_analysis_example()
```

## 📚 Jupyter Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks:

1. **01_quick_start.ipynb**: Get started with basic forecasting
2. **02_forecasting.ipynb**: Deep dive into forecasting models
3. **03_backtesting.ipynb**: Comprehensive backtesting tutorial
4. **04_advanced_features.ipynb**: Advanced features and customization

## 🎯 Running Examples

```bash
# Run individual examples
python examples/forecasting_example.py
python examples/backtesting_example.py
python examples/ensemble_example.py

# Run all examples
python -m examples

# Run Jupyter notebooks
jupyter notebook examples/notebooks/
```

## 📊 Expected Outputs

Each example generates:
- Console output with results
- Optional charts and visualizations
- Performance metrics and analysis
- Risk assessment reports

## 🔧 Customization

Feel free to modify these examples:
- Change ticker symbols
- Adjust time periods
- Modify strategy parameters
- Add custom indicators
- Integrate with your own data sources 