# Execution Engine Documentation

## Overview

The Execution Engine is a comprehensive trade execution system that provides both simulation and live trading capabilities. It consists of two main components:

1. **Execution Agent** (`execution/execution_agent.py`) - Core execution logic with simulation and live trading
2. **Broker Adapter** (`execution/broker_adapter.py`) - Unified interface for multiple brokers

## Features

### Execution Agent Features
- **Realistic Simulation**: Market/limit order execution with spread, slippage, and delay
- **Live Trading**: Integration with real brokers for live execution
- **Risk Management**: Position limits, order size limits, daily trade limits
- **Order Book Management**: Comprehensive logging of all orders and executions
- **Performance Tracking**: Real-time metrics and analytics
- **Position Management**: Automatic position tracking and P&L calculation

### Broker Adapter Features
- **Multi-Broker Support**: Alpaca, Interactive Brokers (IBKR), Polygon.io
- **Unified API**: Consistent interface across all brokers
- **Rate Limiting**: Automatic rate limit management
- **Error Handling**: Robust error handling and retry logic
- **Market Data**: Real-time market data access
- **Account Management**: Position and account information

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Execution     │    │   Broker         │    │   External      │
│   Agent         │◄──►│   Adapter        │◄──►│   Brokers       │
│                 │    │                  │    │                 │
│ • Order Mgmt    │    │ • Unified API    │    │ • Alpaca        │
│ • Risk Control  │    │ • Rate Limiting  │    │ • IBKR          │
│ • Simulation    │    │ • Error Handling │    │ • Polygon       │
│ • Performance   │    │ • Market Data    │    │ • Custom        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Usage

```python
import asyncio
from execution.execution_agent import ExecutionAgent, OrderSide, OrderType

async def main():
    # Create execution agent
    agent = ExecutionAgent()
    
    # Start agent
    await agent.start()
    
    # Submit order
    order_id = await agent.submit_order(
        ticker="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    
    # Check status
    execution = agent.get_order_status(order_id)
    print(f"Order status: {execution.status}")
    
    # Stop agent
    await agent.stop()

asyncio.run(main())
```

### Broker Integration

```python
from execution.broker_adapter import create_broker_adapter

# Create broker adapter
adapter = create_broker_adapter("alpaca", {
    "api_key": "your_api_key",
    "secret_key": "your_secret_key",
    "base_url": "https://paper-api.alpaca.markets"
})

# Connect and trade
await adapter.connect()
market_data = await adapter.get_market_data("AAPL")
await adapter.disconnect()
```

## Configuration

### Execution Agent Configuration

The execution agent is configured through `config/app_config.yaml`:

```yaml
execution:
  mode: "simulation"  # simulation, live, paper
  spread_multiplier: 1.0
  slippage_bps: 5
  execution_delay_ms: 100
  commission_rate: 0.001
  min_commission: 1.0
  max_position_size: 0.1
  max_order_size: 10000
  max_daily_trades: 100
  broker:
    type: "alpaca"
    api_key: "your_api_key"
    secret_key: "your_secret_key"
    base_url: "https://paper-api.alpaca.markets"
```

### Broker Configuration

Each broker has specific configuration requirements:

#### Alpaca
```yaml
broker:
  type: "alpaca"
  api_key: "your_api_key"
  secret_key: "your_secret_key"
  base_url: "https://paper-api.alpaca.markets"  # or live URL
```

#### Interactive Brokers
```yaml
broker:
  type: "ibkr"
  host: "127.0.0.1"
  port: 7497  # 7497 for TWS, 4001 for IB Gateway
  client_id: 1
```

#### Polygon.io
```yaml
broker:
  type: "polygon"
  api_key: "your_api_key"
```

## API Reference

### Execution Agent

#### Core Methods

##### `submit_order(ticker, side, order_type, quantity, price=None, stop_price=None, time_in_force="day", client_order_id=None)`
Submit an order for execution.

**Parameters:**
- `ticker` (str): Stock symbol
- `side` (OrderSide): BUY or SELL
- `order_type` (OrderType): MARKET, LIMIT, STOP, etc.
- `quantity` (float): Number of shares
- `price` (float, optional): Limit price
- `stop_price` (float, optional): Stop price
- `time_in_force` (str): Order time in force
- `client_order_id` (str, optional): Client-provided order ID

**Returns:**
- `str`: Order ID

##### `get_order_status(order_id)`
Get the status of an order.

**Parameters:**
- `order_id` (str): Order ID

**Returns:**
- `OrderExecution` or `None`: Order execution details

##### `get_position(ticker)`
Get current position for a ticker.

**Parameters:**
- `ticker` (str): Stock symbol

**Returns:**
- `Position` or `None`: Position details

##### `get_all_positions()`
Get all current positions.

**Returns:**
- `Dict[str, Position]`: All positions

##### `get_performance_metrics()`
Get performance metrics.

**Returns:**
- `Dict[str, Any]`: Performance metrics

##### `cancel_order(order_id)`
Cancel an order.

**Parameters:**
- `order_id` (str): Order ID

**Returns:**
- `bool`: Success status

#### Data Classes

##### `OrderRequest`
Order request structure.

**Attributes:**
- `order_id` (str): Unique order identifier
- `ticker` (str): Stock symbol
- `side` (OrderSide): BUY or SELL
- `order_type` (OrderType): Order type
- `quantity` (float): Number of shares
- `price` (float, optional): Limit price
- `stop_price` (float, optional): Stop price
- `time_in_force` (str): Order time in force
- `client_order_id` (str, optional): Client order ID
- `timestamp` (str): Order timestamp

##### `OrderExecution`
Order execution result.

**Attributes:**
- `order_id` (str): Order ID
- `ticker` (str): Stock symbol
- `side` (OrderSide): Order side
- `order_type` (OrderType): Order type
- `quantity` (float): Requested quantity
- `price` (float): Requested price
- `executed_quantity` (float): Actually executed quantity
- `average_price` (float): Average execution price
- `commission` (float): Commission paid
- `timestamp` (str): Execution timestamp
- `status` (OrderStatus): Order status
- `fills` (List[Dict]): Fill details
- `metadata` (Dict): Additional metadata

##### `Position`
Position information.

**Attributes:**
- `ticker` (str): Stock symbol
- `quantity` (float): Number of shares
- `average_price` (float): Average entry price
- `market_value` (float): Current market value
- `unrealized_pnl` (float): Unrealized profit/loss
- `realized_pnl` (float): Realized profit/loss
- `timestamp` (str): Position timestamp

##### `MarketData`
Market data snapshot.

**Attributes:**
- `ticker` (str): Stock symbol
- `bid` (float): Best bid price
- `ask` (float): Best ask price
- `last` (float): Last traded price
- `volume` (int): Trading volume
- `timestamp` (str): Data timestamp
- `spread` (float): Bid-ask spread
- `volatility` (float): Price volatility

### Broker Adapter

#### Core Methods

##### `connect()`
Connect to the broker.

**Returns:**
- `bool`: Connection success

##### `disconnect()`
Disconnect from the broker.

##### `submit_order(order)`
Submit an order to the broker.

**Parameters:**
- `order` (OrderRequest): Order to submit

**Returns:**
- `OrderExecution`: Order execution result

##### `cancel_order(order_id)`
Cancel an order.

**Parameters:**
- `order_id` (str): Order ID

**Returns:**
- `bool`: Success status

##### `get_order_status(order_id)`
Get order status.

**Parameters:**
- `order_id` (str): Order ID

**Returns:**
- `OrderExecution` or `None`: Order status

##### `get_position(ticker)`
Get position for a ticker.

**Parameters:**
- `ticker` (str): Stock symbol

**Returns:**
- `Position` or `None`: Position details

##### `get_all_positions()`
Get all positions.

**Returns:**
- `Dict[str, Position]`: All positions

##### `get_account_info()`
Get account information.

**Returns:**
- `AccountInfo`: Account details

##### `get_market_data(ticker)`
Get market data for a ticker.

**Parameters:**
- `ticker` (str): Stock symbol

**Returns:**
- `MarketData`: Market data

## Execution Modes

### Simulation Mode
- Realistic order execution simulation
- Configurable spread, slippage, and delay
- No real money involved
- Perfect for testing and development

### Paper Trading Mode
- Real broker connection with paper trading account
- Real market data and order routing
- No real money involved
- Good for strategy validation

### Live Trading Mode
- Real broker connection with live account
- Real money and real risk
- Production trading environment
- Requires proper risk management

## Risk Management

### Position Limits
- Maximum position size as percentage of portfolio
- Prevents over-concentration in single positions

### Order Size Limits
- Maximum order size in dollar terms
- Prevents large orders that could impact market

### Daily Trade Limits
- Maximum number of trades per day
- Prevents excessive trading

### Stop Loss and Take Profit
- Automatic order placement for risk management
- Configurable percentage-based triggers

## Performance Monitoring

### Key Metrics
- **Daily Trades**: Number of trades executed today
- **Daily Volume**: Total dollar volume traded today
- **Daily Commission**: Total commissions paid today
- **Total P&L**: Combined realized and unrealized P&L
- **Position Count**: Number of active positions

### Logging
- All orders logged to `logs/execution/order_book.json`
- All executions logged to `logs/execution/trade_log.json`
- Comprehensive audit trail for compliance

## Error Handling

### Common Errors
- **Rate Limit Exceeded**: Too many requests to broker
- **Insufficient Funds**: Not enough buying power
- **Invalid Order**: Order parameters don't meet broker requirements
- **Connection Lost**: Network or broker connection issues

### Error Recovery
- Automatic retry with exponential backoff
- Graceful degradation to simulation mode
- Comprehensive error logging and reporting

## Best Practices

### Order Management
1. Always check order status after submission
2. Use appropriate order types for your strategy
3. Set reasonable time-in-force values
4. Monitor order execution quality

### Risk Management
1. Set appropriate position and order size limits
2. Monitor daily trading activity
3. Use stop losses for risk control
4. Regularly review performance metrics

### Performance Optimization
1. Use limit orders when possible to reduce slippage
2. Batch orders when appropriate
3. Monitor commission costs
4. Track execution quality metrics

## Examples

### Basic Trading Strategy

```python
async def simple_strategy():
    agent = ExecutionAgent()
    await agent.start()
    
    # Buy signal
    if buy_condition:
        order_id = await agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        # Monitor execution
        execution = agent.get_order_status(order_id)
        if execution.status == OrderStatus.FILLED:
            print(f"Bought {execution.executed_quantity} shares at ${execution.average_price}")
    
    await agent.stop()
```

### Portfolio Rebalancing

```python
async def rebalance_portfolio():
    agent = ExecutionAgent()
    await agent.start()
    
    target_weights = {"AAPL": 0.3, "TSLA": 0.2, "NVDA": 0.5}
    positions = agent.get_all_positions()
    
    for ticker, target_weight in target_weights.items():
        current_position = positions.get(ticker)
        current_weight = current_position.market_value / total_portfolio_value if current_position else 0
        
        if abs(current_weight - target_weight) > 0.02:  # 2% threshold
            # Calculate required trade
            target_value = total_portfolio_value * target_weight
            current_value = current_position.market_value if current_position else 0
            
            if current_value < target_value:
                # Need to buy
                buy_value = target_value - current_value
                market_data = agent.get_market_data(ticker)
                quantity = int(buy_value / market_data.last)
                
                await agent.submit_order(
                    ticker=ticker,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity
                )
    
    await agent.stop()
```

### Risk Management

```python
async def risk_managed_trading():
    agent = ExecutionAgent()
    
    # Set risk limits
    agent.max_position_size = 0.1  # 10% max position
    agent.max_order_size = 5000    # $5K max order
    agent.max_daily_trades = 20    # 20 trades per day
    
    await agent.start()
    
    # Check risk limits before trading
    try:
        order_id = await agent.submit_order(
            ticker="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
    except ValueError as e:
        print(f"Risk limit violation: {e}")
        return
    
    # Monitor performance
    metrics = agent.get_performance_metrics()
    if metrics['daily_trades'] > 15:
        print("Approaching daily trade limit")
    
    await agent.stop()
```

## Troubleshooting

### Common Issues

#### Connection Problems
- Check broker credentials and API keys
- Verify network connectivity
- Check broker service status
- Review rate limits

#### Order Rejections
- Verify order parameters meet broker requirements
- Check account has sufficient funds
- Review position limits
- Check market hours and trading restrictions

#### Performance Issues
- Monitor execution delays
- Check market data quality
- Review commission costs
- Analyze slippage patterns

### Debug Mode
Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('execution').setLevel(logging.DEBUG)
```

## Integration with Evolve Platform

The Execution Engine integrates seamlessly with the Evolve trading platform:

### MetaController Integration
- Automatic execution agent initialization
- Portfolio rebalancing triggers
- Risk management monitoring
- Performance reporting

### Configuration Integration
- Unified configuration through `config/app_config.yaml`
- Trigger thresholds in `config/trigger_thresholds.json`
- Automatic parameter updates

### Monitoring Integration
- Real-time performance metrics
- Risk limit monitoring
- Order execution tracking
- Position management

## Future Enhancements

### Planned Features
- **Advanced Order Types**: Iceberg orders, TWAP, VWAP
- **Algorithmic Trading**: Smart order routing, execution algorithms
- **Multi-Asset Support**: Options, futures, crypto
- **Advanced Risk Management**: VaR, stress testing, scenario analysis
- **Machine Learning**: Execution quality prediction, optimal order sizing

### Extensibility
- **Custom Brokers**: Easy integration of new brokers
- **Custom Strategies**: Strategy-specific execution logic
- **Custom Risk Models**: Advanced risk management frameworks
- **Custom Analytics**: Performance and execution analytics

## Support

For questions, issues, or feature requests:

1. Check the documentation and examples
2. Review the test suite for usage patterns
3. Enable debug logging for troubleshooting
4. Contact the development team

## License

This execution engine is part of the Evolve trading platform and follows the same licensing terms. 