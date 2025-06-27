# Execution Agent

The Execution Agent is responsible for trade execution, position tracking, and portfolio management. It currently operates in simulation mode with hooks for real execution via Alpaca, Interactive Brokers, or Robinhood APIs.

## Features

### 🎯 Trade Execution
- **Simulated Trading**: Execute trades in simulation mode with realistic slippage and fees
- **Real Execution Hooks**: Ready for integration with real brokers (Alpaca, IB, Robinhood)
- **Signal Processing**: Process trade signals from other agents and strategies
- **Risk Management**: Position sizing based on risk parameters and confidence levels

### 📊 Portfolio Tracking
- **Position Management**: Track open and closed positions
- **PnL Calculation**: Real-time profit/loss tracking with unrealized PnL
- **Portfolio State**: Monitor cash, equity, leverage, and available capital
- **Risk Metrics**: Calculate portfolio VaR, volatility, and beta

### 📝 Trade Logging
- **Comprehensive Logging**: Log all trade executions to `trade_log.json`
- **Execution History**: Track execution results, slippage, and fees
- **Export Functionality**: Export trade logs in JSON, CSV, or Excel formats
- **Memory Integration**: Log decisions to agent memory for analysis

## Configuration

### Basic Configuration
```python
from trading.agents.execution_agent import ExecutionAgent, AgentConfig

config = AgentConfig(
    name='execution_agent',
    enabled=True,
    priority=1,
    max_concurrent_runs=1,
    timeout_seconds=300,
    retry_attempts=3,
    custom_config={
        'execution_mode': 'simulation',  # 'simulation', 'alpaca', 'interactive_brokers', 'robinhood'
        'max_positions': 10,
        'min_confidence': 0.7,
        'max_slippage': 0.001,  # 10 bps
        'execution_delay': 1.0,  # seconds
        'risk_per_trade': 0.02,  # 2% per trade
        'max_position_size': 0.2,  # 20% of capital
        'base_fee': 0.001,  # 10 bps
        'min_fee': 1.0
    }
)

agent = ExecutionAgent(config)
```

### Execution Modes

#### Simulation Mode (Default)
- Executes trades in simulation with realistic market conditions
- Calculates slippage based on order size and market volatility
- Applies trading fees and commissions
- Perfect for testing and development

#### Real Execution Modes (Future)
- **Alpaca**: Integration with Alpaca Trading API
- **Interactive Brokers**: Integration with IB API
- **Robinhood**: Integration with Robinhood API

## Usage

### Basic Usage
```python
import asyncio
from trading.agents.execution_agent import ExecutionAgent, TradeSignal, TradeDirection
from trading.agents.base_agent_interface import AgentConfig

# Create agent
config = AgentConfig(
    name='execution_agent',
    custom_config={'execution_mode': 'simulation'}
)
agent = ExecutionAgent(config)

# Create trade signal
signal = TradeSignal(
    symbol="AAPL",
    direction=TradeDirection.LONG,
    strategy="bollinger_bands",
    confidence=0.85,
    entry_price=150.25,
    take_profit=155.00,
    stop_loss=148.00
)

# Execute trade
async def execute_trade():
    result = await agent.execute(
        signals=[signal],
        market_data={'AAPL': {'price': 150.25, 'volatility': 0.15}}
    )
    
    if result.success:
        print("Trade executed successfully!")
        print(f"Portfolio status: {result.data['portfolio_state']}")
    else:
        print(f"Trade execution failed: {result.message}")

asyncio.run(execute_trade())
```

### Advanced Usage
```python
# Multiple signals
signals = [
    TradeSignal(symbol="AAPL", direction=TradeDirection.LONG, strategy="rsi", confidence=0.8, entry_price=150.00),
    TradeSignal(symbol="TSLA", direction=TradeDirection.SHORT, strategy="macd", confidence=0.75, entry_price=245.50)
]

# Market data
market_data = {
    'AAPL': {'price': 150.00, 'volatility': 0.15, 'volume': 1000000},
    'TSLA': {'price': 245.50, 'volatility': 0.25, 'volume': 2000000}
}

# Execute multiple trades
result = await agent.execute(signals=signals, market_data=market_data)
```

## Trade Signal Structure

```python
@dataclass
class TradeSignal:
    symbol: str                    # Trading symbol (e.g., "AAPL")
    direction: TradeDirection      # LONG or SHORT
    strategy: str                  # Strategy name (e.g., "bollinger_bands")
    confidence: float              # Confidence level (0.0-1.0)
    entry_price: float             # Entry price
    size: Optional[float] = None   # Position size (auto-calculated if None)
    take_profit: Optional[float] = None      # Take profit level
    stop_loss: Optional[float] = None        # Stop loss level
    max_holding_period: Optional[timedelta] = None  # Max holding period
    market_data: Optional[Dict[str, Any]] = None    # Market context
    timestamp: datetime = datetime.utcnow()
```

## Execution Result Structure

```python
@dataclass
class ExecutionResult:
    success: bool                  # Whether execution was successful
    signal: TradeSignal           # Original trade signal
    position: Optional[Position]   # Created position (if successful)
    execution_price: Optional[float] = None  # Actual execution price
    slippage: float = 0.0         # Slippage amount
    fees: float = 0.0             # Trading fees
    message: str = ""             # Execution message
    error: Optional[str] = None   # Error message (if failed)
    timestamp: datetime = datetime.utcnow()
```

## Portfolio Management

### Portfolio State
```python
# Get current portfolio status
portfolio_status = agent.get_portfolio_status()

# Access portfolio data
cash = portfolio_status['cash']
equity = portfolio_status['equity']
total_pnl = portfolio_status['total_pnl']
unrealized_pnl = portfolio_status['unrealized_pnl']
open_positions = portfolio_status['open_positions']
```

### Position Tracking
```python
# Get open positions
for position in portfolio_status['open_positions']:
    print(f"{position['symbol']}: {position['size']} shares @ ${position['entry_price']}")
    print(f"Unrealized PnL: ${position['unrealized_pnl']}")
```

## Trade Logging

### View Trade Log
```python
# Get all trade log entries
trade_log = agent.get_trade_log()

# Get entries for specific date range
from datetime import datetime, timedelta
start_date = datetime.now() - timedelta(days=7)
end_date = datetime.now()
recent_trades = agent.get_trade_log(start_date=start_date, end_date=end_date)
```

### Export Trade Log
```python
# Export to different formats
json_file = agent.export_trade_log(format='json')
csv_file = agent.export_trade_log(format='csv')
excel_file = agent.export_trade_log(format='excel')
```

### Clear Trade Log
```python
# Clear the trade log
agent.clear_trade_log()
```

## Execution History

### View Recent Executions
```python
# Get recent execution history
history = agent.get_execution_history(limit=50)

for entry in history:
    signal = entry['signal']
    print(f"{signal['symbol']} {signal['direction']} at {entry['timestamp']}")
    print(f"Success: {entry['success']}, Slippage: {entry['slippage']:.4f}")
```

## Risk Management

### Position Sizing
The agent automatically calculates position sizes based on:
- **Risk per trade**: Percentage of capital to risk (default: 2%)
- **Confidence level**: Higher confidence = larger position
- **Market volatility**: Higher volatility = smaller position
- **Position limits**: Maximum position size (default: 20% of capital)

### Slippage Calculation
Slippage is calculated based on:
- **Base slippage**: 1 basis point
- **Order size**: Larger orders = more slippage
- **Market volatility**: Higher volatility = more slippage

### Fee Calculation
Fees include:
- **Base fee**: 10 basis points
- **Minimum fee**: $1.00 per trade
- **Size-based fees**: Larger trades = higher fees

## Integration with Agent Manager

### Register with Agent Manager
```python
from trading.agents.agent_manager import AgentManager
from trading.agents.execution_agent import ExecutionAgent

# Register execution agent
manager = AgentManager()
manager.register_agent("execution_agent", ExecutionAgent)

# Execute through manager
result = await manager.execute_agent("execution_agent", signals=signals, market_data=market_data)
```

### Configuration File
The execution agent can be configured via the agent configuration file:

```json
{
  "agents": {
    "execution_agent": {
      "enabled": true,
      "priority": 4,
      "max_concurrent_runs": 1,
      "timeout_seconds": 300,
      "retry_attempts": 3,
      "custom_config": {
        "execution_mode": "simulation",
        "max_positions": 10,
        "min_confidence": 0.7,
        "max_slippage": 0.001,
        "execution_delay": 1.0,
        "risk_per_trade": 0.02,
        "max_position_size": 0.2,
        "base_fee": 0.001,
        "min_fee": 1.0
      }
    }
  }
}
```

## Future Enhancements

### Real Execution Integration
- **Alpaca Integration**: Full integration with Alpaca Trading API
- **Interactive Brokers**: Integration with IB API for professional trading
- **Robinhood**: Integration with Robinhood API for retail trading
- **Multi-broker Support**: Execute trades across multiple brokers

### Advanced Features
- **Order Types**: Support for limit orders, stop orders, etc.
- **Market Hours**: Respect market hours and trading sessions
- **Order Routing**: Intelligent order routing to best execution venues
- **Real-time Data**: Integration with real-time market data feeds

### Risk Management
- **Portfolio Limits**: Maximum portfolio exposure limits
- **Sector Limits**: Maximum exposure per sector
- **Correlation Limits**: Maximum correlation between positions
- **Dynamic Position Sizing**: Real-time position size adjustments

## Testing

### Run Tests
```bash
# Test execution agent
python trading/agents/test_execution_agent.py

# Run demo
python trading/agents/demo_execution_agent.py
```

### Test Scenarios
- Signal validation
- Position limit checking
- Execution price calculation
- Trade execution
- Portfolio updates
- Trade logging
- Export functionality

## Error Handling

The execution agent includes comprehensive error handling:
- **Signal validation**: Invalid signals are rejected
- **Position limits**: Exceeds limits are rejected
- **Slippage limits**: Excessive slippage is rejected
- **Execution errors**: Failed executions are logged
- **System errors**: Unexpected errors are handled gracefully

## Monitoring

### Metrics
- Execution success rate
- Average slippage
- Total fees
- Portfolio performance
- Position count
- Risk metrics

### Logging
- All executions are logged to `trade_log.json`
- Execution history is maintained in memory
- Error logs are written to standard logging
- Portfolio updates are tracked

---

**🎯 Execution Agent** - Your gateway to automated trade execution and portfolio management. 