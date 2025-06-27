# ExecutionAgent Risk Controls

The ExecutionAgent now includes comprehensive risk management features to protect capital and ensure disciplined trading. This document describes the risk controls, their configuration, and usage.

## Overview

The risk controls system provides:

- **Stop-loss and take-profit thresholds** (percentage, ATR-based, or fixed)
- **Automatic position exits** when risk limits are breached
- **Portfolio-level risk monitoring** (correlation, exposure, daily loss limits)
- **Detailed exit logging** with reasons and metrics
- **Real-time risk monitoring** during trade execution

## Risk Control Types

### 1. Stop-Loss Controls

Stop-loss controls automatically exit positions when prices move against the position beyond a specified threshold.

#### Percentage-Based Stop-Loss
```python
stop_loss = RiskThreshold(
    threshold_type=RiskThresholdType.PERCENTAGE,
    value=0.02  # 2% below entry price
)
```

#### ATR-Based Stop-Loss
```python
stop_loss = RiskThreshold(
    threshold_type=RiskThresholdType.ATR_BASED,
    value=0.0,
    atr_multiplier=2.0,  # 2x ATR distance
    atr_period=14
)
```

#### Fixed Stop-Loss
```python
stop_loss = RiskThreshold(
    threshold_type=RiskThresholdType.FIXED,
    value=5.0  # $5 below entry price
)
```

### 2. Take-Profit Controls

Take-profit controls automatically exit positions when prices move in favor of the position beyond a specified threshold.

#### Percentage-Based Take-Profit
```python
take_profit = RiskThreshold(
    threshold_type=RiskThresholdType.PERCENTAGE,
    value=0.06  # 6% above entry price
)
```

#### ATR-Based Take-Profit
```python
take_profit = RiskThreshold(
    threshold_type=RiskThresholdType.ATR_BASED,
    value=0.0,
    atr_multiplier=3.0,  # 3x ATR distance
    atr_period=14
)
```

### 3. Portfolio-Level Controls

#### Maximum Position Size
```python
max_position_size=0.2  # Maximum 20% of capital per position
```

#### Maximum Portfolio Risk
```python
max_portfolio_risk=0.05  # Maximum 5% portfolio risk exposure
```

#### Daily Loss Limit
```python
max_daily_loss=0.02  # Maximum 2% daily loss
```

#### Correlation Limit
```python
max_correlation=0.7  # Maximum 70% correlation between positions
```

#### Volatility Limit
```python
volatility_limit=0.5  # Maximum 50% volatility for new positions
```

## Configuration

### Agent Configuration

```python
config = {
    'name': 'execution_agent',
    'enabled': True,
    'custom_config': {
        'execution_mode': 'simulation',
        'risk_monitoring_enabled': True,
        'auto_exit_enabled': True,
        'risk_controls': {
            'stop_loss': {
                'threshold_type': 'percentage',
                'value': 0.02,
                'atr_multiplier': 2.0,
                'atr_period': 14
            },
            'take_profit': {
                'threshold_type': 'percentage',
                'value': 0.06,
                'atr_multiplier': 3.0,
                'atr_period': 14
            },
            'max_position_size': 0.2,
            'max_portfolio_risk': 0.05,
            'max_daily_loss': 0.02,
            'max_correlation': 0.7,
            'volatility_limit': 0.5,
            'trailing_stop': False,
            'trailing_stop_distance': None
        }
    }
}
```

### Signal-Level Risk Controls

You can also specify risk controls per trade signal:

```python
# Create custom risk controls for this signal
risk_controls = RiskControls(
    stop_loss=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.03),
    take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.09),
    max_position_size=0.1,
    volatility_limit=0.3
)

# Create signal with custom risk controls
signal = TradeSignal(
    symbol="AAPL",
    direction=TradeDirection.LONG,
    strategy="bollinger_bands",
    confidence=0.85,
    entry_price=150.25,
    risk_controls=risk_controls  # Custom risk controls
)
```

## Usage Examples

### Basic Risk-Controlled Trading

```python
from trading.agents.execution_agent import ExecutionAgent, TradeSignal, TradeDirection
from trading.agents.base_agent_interface import AgentConfig

# Create agent with risk controls
config = {
    'name': 'risk_controlled_agent',
    'enabled': True,
    'custom_config': {
        'execution_mode': 'simulation',
        'risk_monitoring_enabled': True,
        'auto_exit_enabled': True
    }
}

agent_config = AgentConfig(**config)
agent = ExecutionAgent(agent_config)

# Create trade signal
signal = TradeSignal(
    symbol="AAPL",
    direction=TradeDirection.LONG,
    strategy="rsi_strategy",
    confidence=0.8,
    entry_price=150.00
)

# Execute trade with risk monitoring
market_data = {
    'AAPL': {'price': 150.00, 'volatility': 0.15, 'volume': 1000000}
}

result = await agent.execute(
    signals=[signal],
    market_data=market_data,
    risk_check=True
)
```

### Monitoring Risk Limits

```python
# Monitor existing positions for risk limits
market_data = {
    'AAPL': {'price': 147.00, 'volatility': 0.15},  # Price dropped
    'TSLA': {'price': 260.00, 'volatility': 0.25}   # Price rose
}

# This will automatically check and exit positions if limits are breached
result = await agent.execute(
    signals=[],
    market_data=market_data,
    risk_check=True
)
```

### Viewing Exit Events

```python
# Get all exit events
exit_events = agent.get_exit_events()

for event in exit_events:
    exit_data = event['exit_event']
    print(f"{exit_data['symbol']}: {exit_data['exit_reason']} "
          f"@ ${exit_data['exit_price']:.2f} "
          f"(PnL: ${exit_data['pnl']:.2f})")
```

### Risk Summary

```python
# Get comprehensive risk summary
risk_summary = agent.get_risk_summary()

print(f"Total exits: {risk_summary['total_exits']}")
print(f"Total PnL: ${risk_summary['total_pnl']:.2f}")
print(f"Daily PnL: ${risk_summary['daily_pnl']:.2f}")

print("\nExit reasons breakdown:")
for reason, count in risk_summary['exit_reasons'].items():
    print(f"  {reason}: {count} exits")

print("\nPnL by exit reason:")
for reason, data in risk_summary['pnl_by_reason'].items():
    print(f"  {reason}: ${data['total']:.2f} (avg: ${data['avg']:.2f})")
```

## Exit Reasons

The system tracks various exit reasons:

- **`stop_loss`**: Position exited due to stop-loss trigger
- **`take_profit`**: Position exited due to take-profit trigger
- **`max_holding_period`**: Position exited due to maximum holding period
- **`manual`**: Position manually closed
- **`risk_limit`**: Position exited due to portfolio risk limits
- **`volatility_limit`**: Position exited due to volatility limits
- **`correlation_limit`**: Position exited due to correlation limits

## Risk Metrics

The system calculates and tracks various risk metrics:

### Position-Level Metrics
- Unrealized PnL
- Price change percentage
- Volatility
- Value at Risk (VaR)
- Position size and value

### Portfolio-Level Metrics
- Total unrealized PnL
- Portfolio exposure ratio
- Daily PnL
- Position count
- Portfolio correlation
- Portfolio risk exposure

## Logging and Monitoring

### Exit Event Logging

All exit events are logged with detailed information:

```python
{
    'timestamp': '2024-01-15T10:30:00',
    'symbol': 'AAPL',
    'position_id': '12345',
    'exit_price': 147.00,
    'exit_reason': 'stop_loss',
    'pnl': -30.00,
    'holding_period': 3600,  # seconds
    'risk_metrics': {
        'unrealized_pnl': -30.00,
        'price_change': -0.02,
        'volatility': 0.15,
        'var_95': -0.03
    },
    'market_conditions': {
        'price': 147.00,
        'volume': 1000000,
        'volatility': 0.15,
        'market_regime': 'volatile'
    }
}
```

### Risk Log Files

The system maintains several log files:

- `trading/agents/logs/exit_log.json`: Exit events
- `trading/agents/logs/risk_log.json`: Risk monitoring events
- `trading/agents/logs/trade_log.json`: Trade execution events

## Best Practices

### 1. Set Appropriate Risk Limits

```python
# Conservative risk settings
risk_controls = RiskControls(
    stop_loss=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.015),  # 1.5%
    take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.045),  # 4.5%
    max_position_size=0.1,  # 10% max per position
    max_daily_loss=0.015,  # 1.5% daily loss limit
    volatility_limit=0.3
)

# Aggressive risk settings
risk_controls = RiskControls(
    stop_loss=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.03),  # 3%
    take_profit=RiskThreshold(RiskThresholdType.PERCENTAGE, 0.09),  # 9%
    max_position_size=0.25,  # 25% max per position
    max_daily_loss=0.03,  # 3% daily loss limit
    volatility_limit=0.6
)
```

### 2. Use ATR-Based Thresholds for Volatile Markets

```python
# ATR-based thresholds adapt to market volatility
atr_controls = RiskControls(
    stop_loss=RiskThreshold(RiskThresholdType.ATR_BASED, 0.0, atr_multiplier=2.0),
    take_profit=RiskThreshold(RiskThresholdType.ATR_BASED, 0.0, atr_multiplier=3.0)
)
```

### 3. Monitor Portfolio Correlation

```python
# Set correlation limits to avoid over-concentration
risk_controls = RiskControls(
    # ... other settings ...
    max_correlation=0.5  # Maximum 50% correlation
)
```

### 4. Regular Risk Reviews

```python
# Daily risk summary
risk_summary = agent.get_risk_summary()
print(f"Daily PnL: {risk_summary['daily_pnl']:.2%}")
print(f"Exit breakdown: {risk_summary['exit_reasons']}")

# Weekly risk analysis
exit_events = agent.get_exit_events(
    start_date=datetime.now() - timedelta(days=7)
)
```

## Integration with Other Systems

### QuantGPT Integration

The risk controls are automatically applied when using QuantGPT:

```python
# QuantGPT automatically includes risk controls in trade signals
response = await quantgpt_client.process_query(
    "Buy AAPL with 2% stop loss and 6% take profit"
)
```

### Report Generation

Risk metrics are included in automated reports:

```python
# Risk summary included in trade reports
report_data = {
    'trades': trade_data,
    'risk_summary': agent.get_risk_summary(),
    'exit_events': agent.get_exit_events()
}
```

## Troubleshooting

### Common Issues

1. **Positions not exiting**: Check if `risk_monitoring_enabled` and `auto_exit_enabled` are True
2. **Incorrect stop-loss prices**: Verify threshold type and values in configuration
3. **Missing exit events**: Check log file permissions and paths
4. **High correlation warnings**: Review position diversification

### Debug Mode

Enable debug logging to troubleshoot risk control issues:

```python
import logging
logging.getLogger('trading.agents.execution_agent').setLevel(logging.DEBUG)
```

## Performance Considerations

- Risk monitoring adds minimal overhead to trade execution
- ATR calculations use cached price history for efficiency
- Exit events are logged asynchronously to avoid blocking
- Portfolio correlation calculations are optimized for large position sets

## Future Enhancements

Planned improvements include:

- **Trailing stops**: Dynamic stop-loss adjustment
- **Position sizing**: Kelly criterion and optimal position sizing
- **Risk parity**: Equal risk contribution across positions
- **Stress testing**: Monte Carlo simulation of risk scenarios
- **Real-time alerts**: Webhook notifications for risk events 