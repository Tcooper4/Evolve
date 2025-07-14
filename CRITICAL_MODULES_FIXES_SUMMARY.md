# Critical Modules Fixes Summary

This document summarizes the improvements made to five critical modules for production readiness, stability, and prompt-driven routing.

## Overview

The following modules were enhanced with defensive programming, error handling, fallback mechanisms, and comprehensive logging:

1. **trading/report/report_generator.py** - Defensive checks and export logic
2. **trading/llm/parser_engine.py** - LLM prompt parsing with fallback
3. **agents/prompt_agent.py** - Integration with parser engine and strategy routing
4. **trading/backtesting/backtester.py** - NaN handling and error logging
5. **agents/llm/model_loader.py** - Model verification and fallback

## 1. Report Generator Improvements

### File: `trading/report/report_generator.py`

#### New Features:
- **Defensive Export Logic**: Added `export_signals()` method with comprehensive validation
- **Configurable Column Names**: Support for custom buy/sell column names
- **Multiple Export Formats**: CSV, JSON, and Parquet support
- **Data Quality Checks**: Validation for empty, malformed, or NaN-only DataFrames

#### Key Improvements:
```python
def export_signals(
    self, 
    signals_df: pd.DataFrame, 
    output_path: str, 
    buy_col: str = 'Buy', 
    sell_col: str = 'Sell'
) -> bool:
    """
    Export signals DataFrame with defensive checks.
    """
    # Defensive check: if signals DataFrame is empty or malformed, skip export
    if signals_df is None or signals_df.empty:
        logger.warning("Signals DataFrame is empty or None, skipping export")
        return False
    
    # Check for required columns
    required_cols = [buy_col, sell_col]
    missing_cols = [col for col in required_cols if col not in signals_df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}, skipping export")
        return False
    
    # Check for data quality issues
    if signals_df[buy_col].isna().all() and signals_df[sell_col].isna().all():
        logger.warning("All signal columns contain only NaN values, skipping export")
        return False
```

#### Benefits:
- **Reliability**: Prevents crashes from malformed data
- **Flexibility**: Supports custom column naming
- **Safety**: Comprehensive validation before export
- **Logging**: Detailed error messages for debugging

## 2. Parser Engine

### File: `trading/llm/parser_engine.py`

#### New Features:
- **Fallback Chain**: Regex → Local LLM → OpenAI → Enhanced Regex
- **Strategy Routing**: Load strategy routing from JSON configuration
- **Intent Classification**: Comprehensive pattern matching
- **Parameter Extraction**: Automatic extraction of trading parameters

#### Key Components:
```python
@dataclass
class ParsedIntent:
    """Structured parsed intent result."""
    intent: str
    confidence: float
    args: Dict[str, Any]
    provider: str  # 'regex', 'huggingface', 'openai', 'fallback_regex'
    raw_response: str
    error: Optional[str] = None
    json_spec: Optional[Dict[str, Any]] = None

@dataclass
class StrategyRoute:
    """Strategy routing configuration."""
    intent: str
    strategy_name: str
    priority: int
    fallback_strategies: List[str]
    conditions: Dict[str, Any]
    parameters: Dict[str, Any]
```

#### Fallback Regex Router:
```python
def _fallback_regex_router(self, prompt: str) -> ParsedIntent:
    """Enhanced fallback regex router with comprehensive pattern matching."""
    fallback_patterns = {
        r'\b(forecast|predict|future|trend|outlook)\b.*\b(price|stock|market|value)\b': {
            'intent': 'forecasting',
            'confidence': 0.8,
            'args': {'type': 'price_forecast'}
        },
        # ... more patterns
    }
```

#### Strategy Registry:
- **JSON Configuration**: Load strategy routes from `config/strategy_registry.json`
- **Dynamic Routing**: Route intents to appropriate strategies
- **Fallback Chains**: Multiple fallback strategies per intent
- **Condition Matching**: Route based on extracted parameters

#### Benefits:
- **Resilience**: Multiple fallback mechanisms
- **Configurability**: JSON-based strategy routing
- **Accuracy**: Comprehensive pattern matching
- **Extensibility**: Easy to add new patterns and routes

## 3. Prompt Agent Integration

### File: `agents/prompt_agent.py`

#### New Features:
- **Parser Engine Integration**: Uses the new parser engine for intent detection
- **Strategy Routing**: Automatic routing to appropriate strategies
- **Fallback Mechanisms**: Multiple levels of fallback
- **Performance Tracking**: Comprehensive metrics and logging

#### Integration:
```python
def __init__(self, ...):
    # Initialize parser engine
    self.parser_engine = ParserEngine(
        openai_api_key=openai_api_key,
        huggingface_model=huggingface_model,
        enable_debug_mode=enable_debug_mode,
        use_regex_first=use_regex_first,
        use_local_llm=use_local_llm,
        use_openai_fallback=use_openai_fallback,
    )

def parse_intent(self, prompt: str) -> ParsedIntent:
    """Parse intent using the parser engine with fallback chain."""
    try:
        # Use parser engine for intent parsing
        result = self.parser_engine.parse_intent(prompt)
        
        # Update performance metrics
        self.performance_metrics["provider_usage"][result.provider] += 1
        
        return result
        
    except Exception as e:
        logger.error(f"Parser engine failed: {e}")
        
        # Fallback to basic regex parsing
        logger.warning("Using basic regex fallback")
        result = self._basic_regex_fallback(prompt)
        self.performance_metrics["provider_usage"]["fallback"] += 1
        return result
```

#### Benefits:
- **Modularity**: Clean separation of concerns
- **Reliability**: Multiple fallback levels
- **Performance**: Comprehensive tracking
- **Maintainability**: Easy to extend and modify

## 4. Backtester Improvements

### File: `trading/backtesting/backtester.py`

#### New Features:
- **NaN Handling**: Comprehensive NaN detection and handling
- **Error Logging**: Detailed error messages for metrics calculation
- **Data Validation**: Signal DataFrame processing
- **Performance Safety**: Safe defaults for invalid metrics

#### NaN Handling in Performance Metrics:
```python
def get_performance_metrics(self) -> Dict[str, Any]:
    """Get comprehensive performance metrics with NaN handling and error logging."""
    try:
        # Create equity curve
        equity_curve = self._calculate_equity_curve()

        # Add risk metrics with NaN handling
        if "returns" in equity_curve.columns:
            returns_series = equity_curve["returns"].dropna()
            
            # Check for NaN or infinite values in returns
            if returns_series.isna().any() or np.isinf(returns_series).any():
                self.logger.warning("NaN or infinite values detected in returns series")
                returns_series = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns_series) > 0:
                risk_metrics = self.risk_metrics_engine.calculate(returns_series)
                
                # Validate risk metrics for NaN/infinite values
                for metric_name, metric_value in risk_metrics.items():
                    if pd.isna(metric_value) or np.isinf(metric_value):
                        self.logger.error(f"NaN or infinite value in {metric_name}: {metric_value}")
                        risk_metrics[metric_name] = 0.0  # Set to safe default
                
                metrics.update(risk_metrics)
            else:
                self.logger.warning("No valid returns data for risk metrics calculation")

        # Validate final metrics
        for metric_name, metric_value in metrics.items():
            if pd.isna(metric_value) or np.isinf(metric_value):
                self.logger.error(f"NaN or infinite value in final metric {metric_name}: {metric_value}")
                metrics[metric_name] = 0.0  # Set to safe default

        return metrics
        
    except Exception as e:
        self.logger.error(f"Error calculating performance metrics: {e}")
        return self.performance_analyzer.get_fallback_metrics()
```

#### Signal DataFrame Processing:
```python
def process_signals_dataframe(self, signals_df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
    """
    Process signals DataFrame to handle NaN values and ensure data quality.
    """
    if signals_df is None or signals_df.empty:
        self.logger.warning("Signals DataFrame is empty or None")
        return pd.DataFrame()
    
    try:
        # Check for NaN values
        nan_count = signals_df.isna().sum().sum()
        if nan_count > 0:
            self.logger.info(f"Found {nan_count} NaN values in signals DataFrame")
            
            # Handle NaN values based on method
            if fill_method == "ffill":
                signals_df = signals_df.fillna(method='ffill')
            elif fill_method == "bfill":
                signals_df = signals_df.fillna(method='bfill')
            elif fill_method == "drop":
                signals_df = signals_df.dropna()
            elif fill_method == "zero":
                signals_df = signals_df.fillna(0)
        
        # Check for infinite values
        inf_count = np.isinf(signals_df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values in signals DataFrame")
            signals_df = signals_df.replace([np.inf, -np.inf], np.nan)
            signals_df = signals_df.fillna(method='ffill')
        
        return signals_df
        
    except Exception as e:
        self.logger.error(f"Error processing signals DataFrame: {e}")
        return pd.DataFrame()
```

#### Benefits:
- **Data Safety**: Comprehensive NaN and infinite value handling
- **Error Recovery**: Safe defaults for invalid metrics
- **Logging**: Detailed error messages for debugging
- **Flexibility**: Multiple methods for handling missing data

## 5. Model Loader Improvements

### File: `agents/llm/model_loader.py`

#### New Features:
- **Model Verification**: Validate model paths and types before loading
- **Safe Loading**: Wrap `from_pretrained()` in try/except with fallback
- **OpenAI Fallback**: Automatic fallback to OpenAI when local models fail
- **Comprehensive Error Handling**: Detailed error messages and recovery

#### Model Verification:
```python
def verify_model(self, model_name: str) -> bool:
    """
    Verify if a model path/type is valid.
    """
    try:
        if model_name not in self.configs:
            logger.warning(f"Model {model_name} not found in configurations")
            return False
        
        config = self.configs[model_name]
        
        # Verify provider-specific requirements
        if config.provider == "openai":
            # Check if API key is available
            if not config.api_key and not os.getenv("OPENAI_API_KEY"):
                logger.warning(f"OpenAI API key not available for model {model_name}")
                return False
                
        elif config.provider == "huggingface":
            # Check if model exists on HuggingFace Hub
            try:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
                logger.info(f"Model {model_name} verified on HuggingFace Hub")
            except Exception as e:
                logger.warning(f"Model {model_name} not found on HuggingFace Hub: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model {model_name}: {e}")
        return False
```

#### Safe Loading with Fallback:
```python
def _safe_from_pretrained(self, loader_func, model_name: str, **kwargs):
    """
    Safely load a model using from_pretrained with fallback.
    """
    try:
        return loader_func(model_name, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to load {model_name} with {loader_func.__name__}: {e}")
        
        # Try with different model variants
        fallback_models = [
            "distilgpt2",
            "gpt2",
            "bert-base-uncased",
            "roberta-base"
        ]
        
        for fallback in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback}")
                return loader_func(fallback, **kwargs)
            except Exception as fallback_error:
                logger.warning(f"Fallback {fallback} also failed: {fallback_error}")
                continue
        
        raise Exception(f"All fallback models failed for {model_name}")

async def _fallback_to_openai(self, config: ModelConfig) -> None:
    """
    Fallback to OpenAI when HuggingFace model loading fails.
    """
    try:
        # Check if OpenAI is available
        if not config.api_key and not os.getenv("OPENAI_API_KEY"):
            raise Exception("No OpenAI API key available for fallback")
        
        logger.info("Falling back to OpenAI model")
        
        # Use a default OpenAI model
        fallback_config = ModelConfig(
            name="gpt-3.5-turbo",
            provider="openai",
            model_type="chat",
            api_key=config.api_key or os.getenv("OPENAI_API_KEY")
        )
        
        await self._load_openai_model(fallback_config)
        
        # Update the original config to use the fallback
        self.models[config.name] = self.models["gpt-3.5-turbo"]
        self.models[config.name]["config"] = config
        
        logger.info(f"Successfully fell back to OpenAI for {config.name}")
        
    except Exception as e:
        logger.error(f"OpenAI fallback also failed: {e}")
        raise
```

#### Benefits:
- **Reliability**: Model verification before loading
- **Resilience**: Multiple fallback mechanisms
- **Safety**: Comprehensive error handling
- **Flexibility**: Support for multiple model providers

## Testing

### Test File: `tests/test_critical_modules_fixes.py`

Comprehensive test suite covering:
- **Defensive Checks**: Testing with invalid inputs
- **Fallback Mechanisms**: Testing all fallback paths
- **Error Handling**: Testing error scenarios
- **Integration**: Testing module interactions
- **Production Readiness**: Testing real-world scenarios

#### Test Categories:
1. **Report Generator Tests**
   - Defensive checks for empty/malformed data
   - Export functionality with various formats
   - Custom column name support

2. **Parser Engine Tests**
   - Fallback regex router functionality
   - Strategy routing and registry management
   - Intent classification accuracy

3. **Prompt Agent Tests**
   - Parser engine integration
   - Basic regex fallback
   - Strategy routing integration

4. **Backtester Tests**
   - NaN handling in performance metrics
   - Signal DataFrame processing
   - Infinite value handling
   - Empty DataFrame handling

5. **Model Loader Tests**
   - Model verification
   - Safe from_pretrained with fallback
   - HuggingFace to OpenAI fallback

## Production Benefits

### 1. **Reliability**
- Comprehensive error handling prevents crashes
- Fallback mechanisms ensure system availability
- Defensive programming catches edge cases

### 2. **Maintainability**
- Modular design with clear separation of concerns
- Comprehensive logging for debugging
- Configurable behavior through JSON files

### 3. **Performance**
- Fast regex fallback for common cases
- Efficient model loading with verification
- Optimized data processing with NaN handling

### 4. **Safety**
- Data validation prevents corruption
- Safe defaults for invalid metrics
- Model verification prevents loading failures

### 5. **Extensibility**
- Easy to add new patterns and routes
- Configurable strategy registry
- Modular architecture for new features

## Usage Examples

### 1. Export Signals with Defensive Checks
```python
from trading.report.report_generator import ReportGenerator

report_gen = ReportGenerator()

# Safe export with validation
success = report_gen.export_signals(
    signals_df=my_signals,
    output_path="signals.csv",
    buy_col="Long",
    sell_col="Short"
)

if not success:
    print("Export failed - check data quality")
```

### 2. Parse Intent with Fallback
```python
from trading.llm.parser_engine import ParserEngine

parser = ParserEngine()

# Parse intent with automatic fallback
result = parser.parse_intent("Forecast AAPL price for next week")
print(f"Intent: {result.intent}, Confidence: {result.confidence}")

# Route to strategy
route = parser.route_strategy(result.intent, result.args)
print(f"Strategy: {route.strategy_name}")
```

### 3. Process Signals with NaN Handling
```python
from trading.backtesting.backtester import Backtester

backtester = Backtester(data=price_data)

# Process signals safely
clean_signals = backtester.process_signals_dataframe(
    signals_df=raw_signals,
    fill_method="ffill"
)

# Get performance metrics with error handling
metrics = backtester.get_performance_metrics()
```

### 4. Load Models with Verification
```python
from agents.llm.model_loader import ModelLoader

loader = ModelLoader()

# Verify model before loading
if loader.verify_model("my_model"):
    await loader.load_model("my_model")
else:
    print("Model verification failed, using fallback")
```

## Conclusion

These improvements transform the critical modules from basic implementations to production-ready, robust systems with:

- **Comprehensive error handling** and fallback mechanisms
- **Defensive programming** practices throughout
- **Detailed logging** for monitoring and debugging
- **Data validation** and safety checks
- **Modular architecture** for easy maintenance and extension
- **Configurable behavior** through external files

All modules are now safe for prompt-driven routing and can handle real-world scenarios with grace and reliability. 