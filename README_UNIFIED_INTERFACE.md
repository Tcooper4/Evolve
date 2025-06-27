# 🔮 Evolve Unified Interface

The Unified Interface provides access to all Evolve trading system features through multiple access methods:

- **Streamlit Web Interface**: Interactive dashboards and forms
- **Terminal Command Line**: Quick command execution
- **Natural Language**: Ask questions in plain English via QuantGPT
- **Direct Commands**: Execute specific actions with parameters

## 🚀 Quick Start

### Streamlit Web Interface (Recommended)

```bash
# Launch the main Streamlit app
streamlit run app.py

# Or launch the unified interface directly
streamlit run unified_interface.py
```

### Terminal Interface

```bash
# Launch terminal interface
python unified_interface.py --terminal

# Execute a single command
python unified_interface.py --command "forecast AAPL 7d"

# Run demo commands
python unified_interface.py --demo
```

### Natural Language Queries

```bash
# Ask QuantGPT questions
python unified_interface.py --command "What's the best model for TSLA?"

# Get trading signals
python unified_interface.py --command "Should I buy AAPL now?"

# Analyze market conditions
python unified_interface.py --command "Analyze BTCUSDT market conditions"
```

## 📋 Available Features

### 🔮 Forecasting
Generate market predictions using advanced ML models.

**Commands:**
- `forecast AAPL 30d` - Generate 30-day forecast for AAPL
- `predict BTCUSDT 1h` - 1-hour prediction for Bitcoin
- `analyze market NVDA` - Market analysis for NVIDIA

**Natural Language:**
- "What's the best model for TSLA over 90 days?"
- "Give me a forecast for AAPL next week"
- "Predict BTC price movement"

### ⚙️ Model Tuning
Optimize model hyperparameters and strategies.

**Commands:**
- `tune model lstm AAPL` - Tune LSTM model for AAPL
- `optimize strategy rsi` - Optimize RSI strategy
- `hyperparameter search xgboost` - Search XGBoost parameters

**Natural Language:**
- "Optimize the LSTM model for better performance"
- "Find the best parameters for the RSI strategy"
- "Tune the XGBoost model for AAPL"

### 🤖 QuantGPT Natural Language Interface
Ask questions about trading in plain English.

**Example Queries:**
- "What's the best model for NVDA over 90 days?"
- "Should I long TSLA this week?"
- "Analyze BTCUSDT market conditions"
- "What's the trading signal for AAPL?"
- "Find the optimal model for GOOGL on 1h timeframe"

### 🎯 Strategy Management
Manage and execute trading strategies.

**Commands:**
- `strategy list` - List all available strategies
- `strategy run bollinger AAPL` - Run Bollinger strategy on AAPL
- `backtest macd TSLA` - Backtest MACD strategy on TSLA

**Natural Language:**
- "Run the Bollinger Bands strategy on AAPL"
- "What's the performance of the RSI strategy?"
- "Backtest the MACD strategy for the last month"

### 💼 Portfolio Management
Portfolio analysis and management.

**Commands:**
- `portfolio status` - Get current portfolio status
- `rebalance portfolio` - Rebalance portfolio allocation
- `risk analysis` - Analyze portfolio risk

**Natural Language:**
- "What's my current portfolio status?"
- "Should I rebalance my portfolio?"
- "What's the risk level of my current positions?"

### 🤖 Agent Management
Manage autonomous trading agents.

**Commands:**
- `agent list` - List all available agents
- `agent status` - Get agent status
- `start agent model_builder` - Start specific agent

**Natural Language:**
- "What agents are currently running?"
- "Start the model builder agent"
- "Check the status of all agents"

### 📊 Report Generation
Generate comprehensive trading reports.

**Commands:**
- `report generate AAPL` - Generate report for AAPL
- `performance report` - Generate performance report
- `trade log` - Generate trade log

**Natural Language:**
- "Generate a comprehensive report for TSLA"
- "Create a performance report for my portfolio"
- "Show me the trade log for the last week"

## 🖥️ Interface Modes

### Streamlit Web Interface
- **Interactive Dashboards**: Professional-grade visualizations
- **Form-based Input**: Easy parameter configuration
- **Real-time Updates**: Live data and results
- **Multiple Pages**: Organized feature access

**Access:** Navigate to "Unified Interface" in the main app sidebar

### Terminal Command Line
- **Quick Commands**: Fast execution of specific actions
- **Script Integration**: Easy automation and scripting
- **Batch Processing**: Execute multiple commands
- **Remote Access**: SSH-friendly interface

**Usage:**
```bash
python unified_interface.py --terminal
```

### Natural Language Interface
- **Plain English**: Ask questions naturally
- **Context Understanding**: Intelligent query parsing
- **GPT Commentary**: AI-powered analysis and explanations
- **Multi-intent Support**: Handle complex queries

**Usage:**
```bash
python unified_interface.py --command "What's the best model for TSLA?"
```

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI API key for QuantGPT
export OPENAI_API_KEY="your_openai_api_key"

# Redis configuration
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

### Configuration File
Create a `config.json` file:
```json
{
  "redis_host": "localhost",
  "redis_port": 6379,
  "openai_api_key": "your_key_here",
  "default_symbols": ["AAPL", "TSLA", "NVDA"],
  "default_timeframe": "1h"
}
```

## 📚 Examples

### Basic Usage Examples

```bash
# Get help
python unified_interface.py --command "help"

# Quick forecast
python unified_interface.py --command "forecast AAPL 7d"

# Ask QuantGPT
python unified_interface.py --command "What's the trading signal for TSLA?"

# Check system status
python unified_interface.py --command "status"
```

### Advanced Usage Examples

```bash
# Run full analysis with backtest
python unified_interface.py --command "run full analysis AAPL with backtest"

# Optimize portfolio with risk constraints
python unified_interface.py --command "optimize portfolio with risk constraints"

# Generate comprehensive report with heatmap
python unified_interface.py --command "generate comprehensive report with heatmap"
```

### Natural Language Examples

```bash
# Model recommendations
python unified_interface.py --command "What's the best model for NVDA over 90 days?"

# Trading decisions
python unified_interface.py --command "Should I buy AAPL now?"

# Market analysis
python unified_interface.py --command "Analyze BTCUSDT market conditions"

# Strategy questions
python unified_interface.py --command "Which strategy works best for volatile markets?"
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install streamlit redis openai
   ```

2. **QuantGPT Not Available**
   ```bash
   # Set OpenAI API key
   export OPENAI_API_KEY="your_key_here"
   ```

3. **Redis Connection Issues**
   ```bash
   # Start Redis server
   redis-server
   
   # Or use Docker
   docker run -d -p 6379:6379 redis
   ```

### Error Messages

- `"Interface not initialized"`: Check component dependencies
- `"QuantGPT not available"`: Set OpenAI API key
- `"Service not available"`: Start required services

## 🔗 Integration

### With Existing Services
The unified interface integrates with all existing Evolve services:

- **Agent Manager**: Manage autonomous agents
- **Report Generator**: Generate comprehensive reports
- **Safe Executor**: Execute user-defined models safely
- **Reasoning Logger**: Track decision reasoning
- **Service Client**: Communicate with Redis services

### API Access
For programmatic access:

```python
from unified_interface import UnifiedInterface

# Initialize interface
interface = UnifiedInterface()

# Process commands
result = interface.process_command("forecast AAPL 7d")
print(result)
```

## 📈 Performance

### Optimization Tips

1. **Use Terminal for Quick Commands**: Faster than web interface
2. **Batch Commands**: Execute multiple commands in sequence
3. **Cache Results**: Results are cached for repeated queries
4. **Parallel Processing**: Multiple agents can run simultaneously

### Monitoring

- **System Status**: Check `status` command
- **Agent Health**: Monitor agent status
- **Performance Metrics**: Track execution times
- **Error Logs**: Review error messages

## 🚀 Future Enhancements

### Planned Features

- **Voice Interface**: Speech-to-text and text-to-speech
- **Mobile App**: Native mobile application
- **API Gateway**: RESTful API endpoints
- **Plugin System**: Extensible command system
- **Advanced Analytics**: Enhanced visualization options

### Contributing

To add new features:

1. **Extend UnifiedInterface**: Add new command handlers
2. **Update Help System**: Document new features
3. **Add UI Components**: Create Streamlit interfaces
4. **Write Tests**: Ensure reliability

## 📞 Support

For issues and questions:

1. **Check Documentation**: Review this README
2. **Run Help Command**: `python unified_interface.py --command "help"`
3. **Check Logs**: Review error messages
4. **Test Components**: Verify individual services

---

**🔮 Evolve Unified Interface** - Your gateway to intelligent trading automation. 