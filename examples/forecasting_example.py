"""
Basic Forecasting Example

This example demonstrates how to generate market forecasts using the Evolve system.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.models.ensemble_model import EnsembleModel
from trading.data.providers.yfinance_provider import YFinanceProvider
import pandas as pd

# Configure logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def forecast_aapl():
    """Generate a 30-day forecast for AAPL using ensemble model."""
    
    logger.info("🚀 Starting AAPL Forecast Example")
    logger.info("=" * 50)
    
    try:
        # Initialize components
        logger.info("📊 Initializing data provider and model...")
        provider = YFinanceProvider()
        model = EnsembleModel()
        
        # Get historical data
        logger.info("📈 Fetching historical data...")
        data = provider.fetch("AAPL", "1d", start_date="2023-01-01")
        
        if data.empty:
            logger.error("❌ No data received from provider")
            return None
        
        logger.info(f"✅ Retrieved {len(data)} data points")
        logger.info(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Generate forecast
        logger.info("🔮 Generating forecast...")
        forecast = model.predict(data, horizon=30)
        
        # Display results
        logger.info("\n📊 FORECAST RESULTS")
        logger.info("=" * 30)
        logger.info(f"Current Price: ${data['close'].iloc[-1]:.2f}")
        logger.info(f"Predicted Price (30d): ${forecast['predicted_price']:.2f}")
        logger.info(f"Price Change: {((forecast['predicted_price'] / data['close'].iloc[-1]) - 1) * 100:.2f}%")
        
        if 'confidence' in forecast:
            logger.info(f"Confidence: {forecast['confidence']:.2%}")
        
        if 'lower' in forecast and 'upper' in forecast:
            logger.info(f"Confidence Interval: ${forecast['lower']:.2f} - ${forecast['upper']:.2f}")
        
        logger.info("\n✅ Forecast completed successfully!")
        return forecast
        
    except Exception as e:
        logger.error(f"❌ Error during forecasting: {e}")
        return None

def forecast_multiple_tickers():
    """Generate forecasts for multiple tickers."""
    
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    results = {}
    
    logger.info(f"\n🔄 Forecasting multiple tickers: {', '.join(tickers)}")
    logger.info("=" * 60)
    
    for ticker in tickers:
        logger.info(f"\n📈 Processing {ticker}...")
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
            
            logger.info(f"  Current: ${results[ticker]['current_price']:.2f}")
            logger.info(f"  Predicted: ${results[ticker]['predicted_price']:.2f}")
            logger.info(f"  Change: {results[ticker]['change_pct']:+.2f}%")
            
        except Exception as e:
            logger.error(f"  ❌ Error with {ticker}: {e}")
            results[ticker] = None
    
    return results

if __name__ == "__main__":
    logger.info("🎯 EVOLVE FORECASTING EXAMPLE")
    logger.info("=" * 50)
    
    # Single ticker forecast
    forecast_aapl()
    
    # Multiple ticker forecast
    forecast_multiple_tickers()
    
    logger.info("\n🎉 Example completed!") 