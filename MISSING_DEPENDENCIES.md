# Missing Dependencies and Mock Data Locations

## Missing Dependencies

Based on the error logs from running the Streamlit app, here are the dependencies that are missing or not properly installed:

### 1. **Optional Dependencies (Warnings, but not critical)**

These are optional and the system will work without them, but some features will be disabled:

- **`pandas_ta`** - Technical analysis library
  - Status: ✅ INSTALLED
  - Impact: Technical indicators work properly
  - Location: Used in `trading/market/market_indicators.py`

- **`autoformer-pytorch`** - Autoformer model for time series
  - Status: ❌ MISSING
  - Impact: Autoformer models disabled
  - Install: `pip install autoformer-pytorch`
  - Location: Used in model generation

- **`sentence-transformers`** - For NLP/sentiment analysis
  - Status: ❌ MISSING
  - Impact: Advanced NLP features disabled
  - Install: `pip install sentence-transformers`
  - Location: Used in sentiment analysis

- **`numba`** - JIT compiler for numerical code
  - Status: ✅ INSTALLED
  - Impact: JIT optimization enabled (faster performance)
  - Location: Used for performance optimization

- **`spacy` English model** - NLP model
  - Status: ❌ MISSING (spaCy package is installed, but model not downloaded)
  - Impact: Some NLP features may not work
  - Install: `python -m spacy download en_core_web_sm`
  - Location: Used in text processing

- **`websocket-client`** - For WebSocket connections
  - Status: ✅ INSTALLED
  - Impact: Polygon WebSocket will work
  - Location: Used in `trading/data/data_listener.py` for Polygon WebSocket

### 2. **Module Import Errors (Non-critical)**

These are modules that don't exist but are handled gracefully:

- **`utils.launch_utils`** - Utility module
  - Status: Referenced but doesn't exist
  - Impact: Some services may not initialize logging properly
  - Note: This is used in many service files but has fallbacks

- **`trading.agents.execution`** - Execution agent module
  - Status: Module path may be incorrect
  - Impact: Task orchestrator may not initialize execution agent
  - Note: Execution agent exists at `execution.execution_agent`

### 3. **Installation Commands**

To install all optional dependencies:

```bash
# Core optional dependencies
pip install pandas-ta autoformer-pytorch sentence-transformers numba websocket-client

# spaCy English model
python -m spacy download en_core_web_sm

# Verify installations
python -c "import pandas_ta; print('pandas_ta OK')"
python -c "import autoformer_pytorch; print('autoformer OK')"
python -c "import sentence_transformers; print('sentence-transformers OK')"
python -c "import numba; print('numba OK')"
python -c "import websocket; print('websocket-client OK')"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spacy model OK')"
```

## Mock Data Locations

**⚠️ NOTE: Mock data generation has been DISABLED in the production system.** 
The system will now fail with clear error messages if real data providers are unavailable, 
ensuring only real market data is used.

The following mock data components remain ONLY for:
- **Simulation mode** (execution simulation, not data generation)
- **Testing** (test fixtures)

### Removed Mock Data Components:
- ❌ `MockDataProvider` - Removed from fallback chain (now raises errors)
- ❌ `_generate_mock_data()` in `fallback/data_feed.py` - Removed (now raises errors)
- ❌ `_create_sample_data()` in `utils/data_loader.py` - Removed (now raises errors)

### Remaining Components (For Simulation/Testing Only):

The system still has several mock/simulated components, but these are for execution simulation and testing, NOT for data generation:

### 1. **Mock Data Provider** (Primary)
- **Location**: `trading/data/providers/fallback_provider.py`
- **Class**: `MockDataProvider`
- **Purpose**: Generates mock OHLCV data when real data providers fail
- **Features**:
  - Deterministic price generation based on symbol hash
  - Realistic price movements with volatility
  - Configurable date ranges
  - OHLCV data structure

### 2. **Fallback Data Feed**
- **Location**: `fallback/data_feed.py`
- **Class**: `FallbackDataFeed`
- **Method**: `_generate_mock_data()`
- **Purpose**: Fallback when yfinance fails
- **Features**:
  - Random walk price generation
  - Base price around 100 with symbol-based variation
  - 2% daily volatility
  - Full OHLCV data

### 3. **Sample Data Generator**
- **Location**: `utils/data_loader.py`
- **Method**: `_create_sample_data()`
- **Purpose**: Creates sample data for testing
- **Features**:
  - Exponential random walk
  - Realistic OHLCV structure
  - Volume generation

### 4. **Shared Utilities Sample Data**
- **Location**: `utils/shared_utilities.py`
- **Functions**: 
  - `create_sample_data()` - Basic market data
  - `create_sample_forecast_data()` - Forecast data
- **Purpose**: Utility functions for creating test data

### 5. **Simulated Execution Engine**
- **Location**: `execution/live_trading_interface.py`
- **Class**: `SimulatedExecutionEngine`
- **Purpose**: Simulates order execution without real broker
- **Features**:
  - Realistic order execution with slippage
  - Market data simulation
  - Position tracking
  - Account management

### 6. **Execution Agent Simulation**
- **Location**: `execution/execution_agent.py`
- **Class**: `ExecutionAgent` (with `ExecutionMode.SIMULATION`)
- **Method**: `_load_historical_data()` - Generates historical market data
- **Purpose**: Simulates trade execution with realistic market conditions
- **Features**:
  - Historical price data generation
  - Market volatility simulation
  - Spread and slippage modeling

### 7. **Simulation Broker Adapter**
- **Location**: `execution/broker_adapter.py`
- **Class**: `SimulationBrokerAdapter`
- **Purpose**: Mock broker adapter for testing
- **Features**:
  - Simulated order placement
  - Mock order status
  - Simulated positions
  - Mock account info

### 8. **Test Fixtures**
- **Location**: `tests/conftest.py`
- **Functions**: 
  - `sample_price_data()` - Test price data
  - `sample_price_data_with_indicators()` - Price data with indicators
- **Purpose**: Pytest fixtures for testing

## How Mock Data Was Used (Now Disabled)

**Previous behavior (now removed):**
1. ~~**Fallback Mechanism**: When real data providers failed, the system would automatically fall back to mock data~~ ❌ REMOVED
2. **Testing**: Test fixtures still use mock data for unit testing (kept for testing)
3. ~~**Development**: Allowed development without API keys or internet connection~~ ❌ REMOVED
4. **Simulation Mode**: The execution engine can still run in simulation mode for backtesting (kept - this is execution simulation, not data generation)

## Key Mock Data Characteristics

- **Deterministic**: Same symbol always generates same data (for testing)
- **Realistic**: Uses random walks and volatility models
- **Complete**: Always includes OHLCV (Open, High, Low, Close, Volume)
- **Configurable**: Date ranges, volatility, base prices can be adjusted

## Notes

- **Mock data generation has been DISABLED** - The system will now fail with clear errors if real data is unavailable
- **Real data providers are REQUIRED** - yfinance, Alpha Vantage, or other configured providers must be working
- **Simulation mode is different** - `SimulatedExecutionEngine` simulates order execution, not market data (this is kept)
- **Test fixtures remain** - Mock data in `tests/conftest.py` is kept for unit testing purposes
- **Error messages are clear** - When data providers fail, you'll get explicit error messages explaining what's wrong

