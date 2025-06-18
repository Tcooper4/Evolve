# Market Analyzer

The Market Analyzer is a core component of the trading system that provides comprehensive market analysis capabilities. It combines technical analysis, sentiment analysis, and market microstructure analysis to generate actionable insights.

## Features

### Technical Analysis
- Moving averages (SMA, EMA, VWAP)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume analysis
- Support/Resistance detection
- Trend analysis

### Sentiment Analysis
- News sentiment scoring
- Social media sentiment tracking
- Market sentiment indicators
- Sentiment correlation analysis

### Market Microstructure
- Order book analysis
- Volume profile
- Price impact analysis
- Liquidity analysis
- Market depth visualization

## Usage

```python
from trading.market.market_analyzer import MarketAnalyzer

# Initialize analyzer
analyzer = MarketAnalyzer(
    ticker="AAPL",
    timeframe="1d",
    lookback_period=30
)

# Get technical analysis
tech_analysis = analyzer.get_technical_analysis()

# Get sentiment analysis
sentiment = analyzer.get_sentiment_analysis()

# Get market microstructure
microstructure = analyzer.get_market_microstructure()

# Generate trading signals
signals = analyzer.generate_signals()
```

## Configuration

The Market Analyzer can be configured through environment variables:

```bash
# Technical Analysis
TECH_ANALYSIS_LOOKBACK=30
TECH_ANALYSIS_TIMEFRAME=1d
TECH_ANALYSIS_INDICATORS=SMA,EMA,RSI,MACD

# Sentiment Analysis
SENTIMENT_SOURCES=news,social,market
SENTIMENT_WEIGHTS=0.4,0.3,0.3
SENTIMENT_LOOKBACK=7

# Market Microstructure
MICROSTRUCTURE_DEPTH=10
MICROSTRUCTURE_UPDATE_INTERVAL=1
```

## Integration

The Market Analyzer integrates with:

1. **Data Pipeline**
   - Fetches market data from multiple sources
   - Handles data cleaning and normalization
   - Manages data caching and persistence

2. **Signal Generator**
   - Converts analysis into trading signals
   - Applies signal filters and validation
   - Generates signal metadata

3. **Performance Monitor**
   - Tracks analysis accuracy
   - Monitors signal performance
   - Generates performance reports

## Performance

The Market Analyzer is optimized for performance:

- Asynchronous data fetching
- Parallel processing of indicators
- Efficient memory management
- Caching of intermediate results

## Error Handling

The analyzer implements robust error handling:

- Graceful degradation of features
- Fallback to simpler analysis methods
- Comprehensive error logging
- Automatic recovery mechanisms

## Future Improvements

Planned improvements include:

1. Machine learning integration
2. Real-time analysis capabilities
3. Enhanced visualization tools
4. Custom indicator support
5. Multi-asset correlation analysis 