# Position Sizer

The PositionSizer module provides dynamic position sizing based on risk tolerance, confidence scores, and forecast certainty. It supports multiple sizing strategies and integrates seamlessly with the ExecutionAgent for automated trade execution.

## Features

### Multiple Sizing Strategies

1. **Fixed Percentage** - Simple percentage-based sizing
2. **Kelly Criterion** - Mathematical optimal sizing based on win rate and odds
3. **Volatility-Based** - Adjusts position size based on market volatility
4. **Confidence-Based** - Scales position size with signal confidence
5. **Forecast Certainty** - Uses forecast certainty for sizing decisions
6. **Optimal F** - Ralph Vince's Optimal F formula
7. **Risk Parity** - Equal risk contribution across positions
8. **Martingale** - Increases position size after losses
9. **Anti-Martingale** - Increases position size after wins

### Dynamic Adjustments

- **Risk Adjustment** - Reduces size during portfolio losses
- **Correlation Adjustment** - Adjusts for portfolio correlation
- **Volatility Adjustment** - Adapts to market volatility regimes
- **Liquidity Adjustment** - Considers bid-ask spreads and volume

### Integration Features

- Seamless integration with ExecutionAgent
- Comprehensive logging and history tracking
- Configurable parameters per signal
- Error handling with conservative fallbacks

## Quick Start

### Basic Usage

```python
from trading.portfolio.position_sizer import PositionSizer, SizingStrategy, SizingParameters
from trading.portfolio.position_sizer import MarketContext, SignalContext, PortfolioContext

# Create position sizer
position_sizer = PositionSizer()

# Create contexts
market_context = MarketContext(
    symbol='AAPL',
    current_price=150.0,
    volatility=0.18,
    volume=50000000,
    correlation=0.3,
    liquidity_score=0.9,
    bid_ask_spread=0.0005
)

signal_context = SignalContext(
    confidence=0.8,
    forecast_certainty=0.75,
    strategy_performance=0.12,
    win_rate=0.65,
    avg_win=0.03,
    avg_loss=-0.015,
    sharpe_ratio=1.2,
    max_drawdown=0.08,
    signal_strength=0.85
)

portfolio_context = PortfolioContext(
    total_capital=100000.0,
    available_capital=50000.0,
    current_exposure=0.5,
    open_positions=2,
    daily_pnl=0.02,
    portfolio_volatility=0.15
)

# Calculate position size
position_size, sizing_details = position_sizer.calculate_position_size(
    entry_price=150.0,
    stop_loss_price=147.0,
    market_context=market_context,
    signal_context=signal_context,
    portfolio_context=portfolio_context
)

print(f"Position Size: {position_size:.4f}")
print(f"Risk Percentage: {sizing_details['risk_percentage']:.2%}")
print(f"Strategy Used: {sizing_details['strategy']}")
```

### With ExecutionAgent Integration

```python
from trading.agents.execution_agent import ExecutionAgent, TradeSignal, TradeDirection
from trading.agents.base_agent_interface import AgentConfig

# Create execution agent with position sizing config
config = AgentConfig(
    name="execution_agent",
    agent_type="execution",
    enabled=True,
    custom_config={
        'position_sizing_config': {
            'default_strategy': 'confidence_based',
            'risk_per_trade': 0.02,
            'max_position_size': 0.2,
            'confidence_multiplier': 1.5
        }
    }
)

execution_agent = ExecutionAgent(config)

# Create trade signal with sizing strategy
signal = TradeSignal(
    symbol='AAPL',
    direction=TradeDirection.LONG,
    strategy='momentum_strategy',
    confidence=0.85,
    entry_price=150.0,
    market_data={
        'sizing_strategy': 'kelly_criterion',
        'risk_per_trade': 0.025,
        'kelly_fraction': 0.3
    }
)

# Calculate position size
position_size, sizing_details = execution_agent._calculate_position_size(
    signal, signal.entry_price, market_data
)
```

## Configuration

### PositionSizer Configuration

```python
config = {
    'default_strategy': SizingStrategy.CONFIDENCE_BASED,
    'risk_per_trade': 0.02,  # 2% risk per trade
    'max_position_size': 0.2,  # 20% max position size
    'confidence_multiplier': 1.5,
    'volatility_multiplier': 1.0,
    'kelly_fraction': 0.25,  # Conservative Kelly fraction
    'optimal_f_risk': 0.02,
    'base_position_size': 0.1,
    'enable_risk_adjustment': True,
    'enable_correlation_adjustment': True,
    'enable_volatility_adjustment': True
}

position_sizer = PositionSizer(config)
```

### Per-Signal Configuration

```python
# Signal with custom sizing parameters
signal = TradeSignal(
    symbol='TSLA',
    direction=TradeDirection.LONG,
    strategy='volatility_strategy',
    confidence=0.7,
    entry_price=250.0,
    market_data={
        'sizing_strategy': 'volatility_based',
        'risk_per_trade': 0.015,
        'volatility_multiplier': 0.8,
        'max_position_size': 0.15
    }
)
```

## Sizing Strategies

### 1. Fixed Percentage

Simple percentage-based sizing with configurable risk per trade.

```python
params = SizingParameters(
    strategy=SizingStrategy.FIXED_PERCENTAGE,
    risk_per_trade=0.02  # 2% risk
)
```

### 2. Kelly Criterion

Mathematical optimal sizing based on win rate and odds.

```python
params = SizingParameters(
    strategy=SizingStrategy.KELLY_CRITERION,
    kelly_fraction=0.25  # Conservative fraction
)
```

### 3. Volatility-Based

Adjusts position size inversely to volatility.

```python
params = SizingParameters(
    strategy=SizingStrategy.VOLATILITY_BASED,
    volatility_multiplier=1.0
)
```

### 4. Confidence-Based

Scales position size with signal confidence.

```python
params = SizingParameters(
    strategy=SizingStrategy.CONFIDENCE_BASED,
    confidence_multiplier=1.5
)
```

### 5. Forecast Certainty

Uses forecast certainty for sizing decisions.

```python
params = SizingParameters(
    strategy=SizingStrategy.FORECAST_CERTAINTY,
    confidence_multiplier=1.0
)
```

### 6. Optimal F

Ralph Vince's Optimal F formula for position sizing.

```python
params = SizingParameters(
    strategy=SizingStrategy.OPTIMAL_F,
    optimal_f_risk=0.02
)
```

### 7. Risk Parity

Equal risk contribution across positions.

```python
params = SizingParameters(
    strategy=SizingStrategy.RISK_PARITY,
    risk_per_trade=0.02
)
```

### 8. Martingale

Increases position size after losses.

```python
params = SizingParameters(
    strategy=SizingStrategy.MARTINGALE,
    base_position_size=0.1
)
```

### 9. Anti-Martingale

Increases position size after wins.

```python
params = SizingParameters(
    strategy=SizingStrategy.ANTI_MARTINGALE,
    base_position_size=0.1
)
```

## Dynamic Adjustments

### Risk Adjustment

Automatically reduces position size when:
- Portfolio is experiencing losses
- Signal quality is poor (negative Sharpe ratio)
- Maximum drawdown is high

### Correlation Adjustment

Reduces position size when:
- New position correlates highly with existing positions
- Portfolio diversification is compromised

### Volatility Adjustment

Adjusts position size based on:
- Market volatility regime
- Asset-specific volatility
- Volatility multiplier configuration

### Liquidity Adjustment

Considers:
- Bid-ask spreads
- Trading volume
- Liquidity scores

## API Reference

### PositionSizer Class

#### Methods

- `calculate_position_size(entry_price, stop_loss_price, market_context, signal_context, portfolio_context, sizing_params=None)` - Calculate optimal position size
- `get_sizing_history(limit=100)` - Get sizing decision history
- `get_sizing_summary()` - Get sizing summary statistics

#### Configuration

- `default_strategy` - Default sizing strategy
- `risk_per_trade` - Default risk per trade percentage
- `max_position_size` - Maximum position size percentage
- `confidence_multiplier` - Confidence scaling factor
- `volatility_multiplier` - Volatility adjustment factor
- `kelly_fraction` - Conservative Kelly fraction
- `enable_risk_adjustment` - Enable risk-based adjustments
- `enable_correlation_adjustment` - Enable correlation adjustments
- `enable_volatility_adjustment` - Enable volatility adjustments

### Context Classes

#### MarketContext

- `symbol` - Trading symbol
- `current_price` - Current market price
- `volatility` - Asset volatility
- `volume` - Trading volume
- `market_regime` - Market regime (normal, volatile, etc.)
- `correlation` - Correlation with existing positions
- `liquidity_score` - Liquidity score (0-1)
- `bid_ask_spread` - Bid-ask spread percentage

#### SignalContext

- `confidence` - Signal confidence (0-1)
- `forecast_certainty` - Forecast certainty (0-1)
- `strategy_performance` - Strategy performance
- `win_rate` - Historical win rate
- `avg_win` - Average winning trade
- `avg_loss` - Average losing trade
- `sharpe_ratio` - Strategy Sharpe ratio
- `max_drawdown` - Maximum drawdown
- `signal_strength` - Signal strength (0-1)

#### PortfolioContext

- `total_capital` - Total portfolio capital
- `available_capital` - Available capital for trading
- `current_exposure` - Current portfolio exposure
- `open_positions` - Number of open positions
- `daily_pnl` - Daily profit/loss
- `portfolio_volatility` - Portfolio volatility
- `correlation_matrix` - Position correlation matrix

## Examples

### Conservative Sizing

```python
# Conservative configuration
config = {
    'default_strategy': SizingStrategy.FIXED_PERCENTAGE,
    'risk_per_trade': 0.01,  # 1% risk
    'max_position_size': 0.1,  # 10% max
    'confidence_multiplier': 0.8
}
```

### Aggressive Sizing

```python
# Aggressive configuration
config = {
    'default_strategy': SizingStrategy.CONFIDENCE_BASED,
    'risk_per_trade': 0.04,  # 4% risk
    'max_position_size': 0.3,  # 30% max
    'confidence_multiplier': 2.0
}
```

### Kelly Conservative

```python
# Conservative Kelly
config = {
    'default_strategy': SizingStrategy.KELLY_CRITERION,
    'risk_per_trade': 0.02,
    'kelly_fraction': 0.15,  # Very conservative
    'max_position_size': 0.15
}
```

### Volatility Adjusted

```python
# Volatility-adjusted sizing
config = {
    'default_strategy': SizingStrategy.VOLATILITY_BASED,
    'risk_per_trade': 0.025,
    'volatility_multiplier': 1.5,
    'max_position_size': 0.25
}
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_position_sizer.py -v
```

The test suite covers:
- All sizing strategies
- Edge cases and error handling
- Integration with ExecutionAgent
- Context serialization
- Performance scenarios

## Best Practices

1. **Start Conservative** - Begin with fixed percentage sizing
2. **Monitor Performance** - Track sizing decisions and outcomes
3. **Adjust Gradually** - Modify parameters based on performance
4. **Consider Market Conditions** - Use volatility and correlation adjustments
5. **Test Thoroughly** - Validate sizing logic with backtesting
6. **Log Everything** - Maintain detailed sizing history for analysis

## Integration with Trading System

The PositionSizer integrates seamlessly with the broader trading system:

- **ExecutionAgent** - Automatic position sizing during trade execution
- **PortfolioManager** - Portfolio context and position tracking
- **RiskManager** - Risk limit enforcement
- **Memory System** - Decision logging and performance tracking
- **Reporting System** - Sizing decision analysis and reporting

## Performance Considerations

- Position sizing calculations are optimized for speed
- Context objects are lightweight and efficient
- History tracking is limited to prevent memory bloat
- Conservative fallbacks ensure system stability

## Troubleshooting

### Common Issues

1. **Position size too small** - Check risk_per_trade and max_position_size
2. **Position size too large** - Verify available capital and position limits
3. **Strategy not working** - Ensure proper context data is provided
4. **Integration errors** - Check ExecutionAgent configuration

### Debug Mode

Enable debug logging to troubleshoot sizing decisions:

```python
import logging
logging.getLogger('trading.portfolio.position_sizer').setLevel(logging.DEBUG)
```

## Contributing

When contributing to the PositionSizer:

1. Add tests for new strategies
2. Update documentation
3. Follow the existing code style
4. Validate integration with ExecutionAgent
5. Test with real market data scenarios 