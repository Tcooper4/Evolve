"""
Basic Forecasting Example

This example demonstrates how to generate market forecasts using the Evolve system.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.models.ensemble_model import EnsembleModel
from trading.data.providers.yfinance_provider import YFinanceProvider
import pandas as pd

def forecast_aapl():
    """Generate a 30-day forecast for AAPL using ensemble model."""
    
    print("🚀 Starting AAPL Forecast Example")
    print("=" * 50)
    
    try:
        # Initialize components
        print("📊 Initializing data provider and model...")
        provider = YFinanceProvider()
        model = EnsembleModel()
        
        # Get historical data
        print("📈 Fetching historical data...")
        data = provider.fetch("AAPL", "1d", start_date="2023-01-01")
        
        if data.empty:
            print("❌ No data received from provider")
            return None
        
        print(f"✅ Retrieved {len(data)} data points")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Generate forecast
        print("🔮 Generating forecast...")
        forecast = model.predict(data, horizon=30)
        
        # Display results
        print("\n📊 FORECAST RESULTS")
        print("=" * 30)
        print(f"Current Price: ${data['close'].iloc[-1]:.2f}")
        print(f"Predicted Price (30d): ${forecast['predicted_price']:.2f}")
        print(f"Price Change: {((forecast['predicted_price'] / data['close'].iloc[-1]) - 1) * 100:.2f}%")
        
        if 'confidence' in forecast:
            print(f"Confidence: {forecast['confidence']:.2%}")
        
        if 'lower' in forecast and 'upper' in forecast:
            print(f"Confidence Interval: ${forecast['lower']:.2f} - ${forecast['upper']:.2f}")
        
        print("\n✅ Forecast completed successfully!")
        return forecast
        
    except Exception as e:
        print(f"❌ Error during forecasting: {e}")
        return None

def forecast_multiple_tickers():
    """Generate forecasts for multiple tickers."""
    
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    results = {}
    
    print(f"\n🔄 Forecasting multiple tickers: {', '.join(tickers)}")
    print("=" * 60)
    
    for ticker in tickers:
        print(f"\n📈 Processing {ticker}...")
        try:
            provider = YFinanceProvider()
            model = EnsembleModel()
            
            data = provider.fetch(ticker, "1d", start_date="2023-01-01")
            forecast = model.predict(data, horizon=30)
            
            results[ticker] = {
                'current_price': data['close'].iloc[-1],
                'predicted_price': forecast['predicted_price'],
                'change_pct': ((forecast['predicted_price'] / data['close'].iloc[-1]) - 1) * 100
            }
            
            print(f"  Current: ${results[ticker]['current_price']:.2f}")
            print(f"  Predicted: ${results[ticker]['predicted_price']:.2f}")
            print(f"  Change: {results[ticker]['change_pct']:+.2f}%")
            
        except Exception as e:
            print(f"  ❌ Error with {ticker}: {e}")
            results[ticker] = None
    
    return results

if __name__ == "__main__":
    print("🎯 EVOLVE FORECASTING EXAMPLE")
    print("=" * 50)
    
    # Single ticker forecast
    forecast_aapl()
    
    # Multiple ticker forecast
    forecast_multiple_tickers()
    
    print("\n🎉 Example completed!") 