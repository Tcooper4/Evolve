# 🔮 Evolve Unified Interface - Usage Guide

## Quick Start

### 1. Launch the Interface

**Streamlit Web Interface (Recommended):**
```bash
streamlit run app.py
```
Then navigate to "Unified Interface" in the sidebar.

**Terminal Interface:**
```bash
python unified_interface.py --terminal
```

**Direct Command:**
```bash
python unified_interface.py --command "help"
```

### 2. Available Commands

#### Basic Commands
- `help` - Show all available commands
- `status` - Check system status
- `forecast AAPL 7d` - Generate 7-day forecast for AAPL
- `tune model lstm AAPL` - Tune LSTM model for AAPL

#### Strategy Commands
- `strategy list` - List all strategies
- `strategy run bollinger AAPL` - Run Bollinger strategy on AAPL

#### Agent Commands
- `agent list` - List all agents
- `agent status` - Check agent status

#### Portfolio Commands
- `portfolio status` - Get portfolio status
- `portfolio rebalance` - Rebalance portfolio

#### Report Commands
- `report generate AAPL` - Generate report for AAPL

### 3. Natural Language Queries

Ask questions in plain English:

- "What's the best model for TSLA?"
- "Should I buy AAPL now?"
- "Analyze BTCUSDT market conditions"
- "What's the trading signal for NVDA?"

### 4. Interface Features

#### Streamlit Interface
- **Main Interface**: Command input and quick actions
- **QuantGPT**: Natural language queries
- **Forecasting**: Model predictions and analysis
- **Tuning**: Model optimization
- **Strategy**: Strategy management
- **Portfolio**: Portfolio management
- **Agents**: Agent management
- **Reports**: Report generation
- **Help**: Documentation and examples

#### Terminal Interface
- Interactive command prompt
- Command history
- Real-time results
- Error handling

### 5. Examples

#### Quick Examples
```bash
# Get help
python unified_interface.py --command "help"

# Forecast
python unified_interface.py --command "forecast AAPL 7d"

# Ask QuantGPT
python unified_interface.py --command "What's the best model for TSLA?"

# Check status
python unified_interface.py --command "status"
```

#### Advanced Examples
```bash
# Tune model
python unified_interface.py --command "tune model lstm AAPL"

# Run strategy
python unified_interface.py --command "strategy run bollinger AAPL"

# Generate report
python unified_interface.py --command "report generate AAPL"
```

### 6. Configuration

Set environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

### 7. Troubleshooting

**Common Issues:**
1. **Import errors**: Install missing dependencies
2. **QuantGPT not available**: Set OpenAI API key
3. **Redis connection**: Start Redis server
4. **Streamlit not found**: Install with `pip install streamlit`

**Error Messages:**
- `"Interface not initialized"`: Check dependencies
- `"QuantGPT not available"`: Set OpenAI API key
- `"Service not available"`: Start required services

### 8. Integration

The unified interface integrates with:
- All existing Evolve services
- Redis pub/sub communication
- Agent management system
- Report generation system
- Safe execution system
- Reasoning logging system

### 9. Next Steps

1. **Explore Features**: Try different commands and interfaces
2. **Customize**: Configure settings and preferences
3. **Automate**: Create scripts for common tasks
4. **Extend**: Add new commands and features

---

**🔮 Evolve Unified Interface** - Your gateway to intelligent trading automation. 