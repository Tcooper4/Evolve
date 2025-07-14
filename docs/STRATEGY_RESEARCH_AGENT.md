# StrategyResearchAgent Documentation

## Overview

The `StrategyResearchAgent` is an autonomous agent that continuously scans the internet for new trading strategies and models. It discovers strategies from multiple sources, extracts their logic, generates executable code, and integrates them into the Evolve trading platform.

## Features

### 🔍 Multi-Source Discovery
- **arXiv**: Academic papers and research on trading strategies
- **SSRN**: Working papers and preprints
- **GitHub**: Open-source trading repositories and code
- **QuantConnect Forums**: Community strategies and discussions

### 🤖 Intelligent Analysis
- **Strategy Type Detection**: Automatically categorizes strategies (momentum, mean reversion, ML, options, arbitrage)
- **Confidence Scoring**: Rates strategy quality based on content analysis
- **Code Extraction**: Identifies and extracts executable code snippets
- **Parameter Detection**: Automatically finds strategy parameters

### 📊 Strategy Integration
- **Code Generation**: Converts discovered strategies into executable Python classes
- **Backtesting**: Tests strategies using the platform's backtester
- **Performance Evaluation**: Compares strategies against benchmarks
- **Automatic Integration**: Saves promising strategies to the strategy registry

### ⏰ Continuous Monitoring
- **Scheduled Scans**: Periodic discovery runs (configurable intervals)
- **Duplicate Detection**: Prevents re-discovery of existing strategies
- **Progress Tracking**: Maintains history of discoveries and test results

## Architecture

```
StrategyResearchAgent
├── Discovery Engine
│   ├── arXiv Crawler
│   ├── SSRN Crawler
│   ├── GitHub API Client
│   └── QuantConnect Crawler
├── Analysis Engine
│   ├── Strategy Type Classifier
│   ├── Confidence Scorer
│   ├── Code Extractor
│   └── Parameter Detector
├── Code Generator
│   ├── Template Engine
│   ├── Strategy Logic Generator
│   └── Integration Adapter
└── Testing Engine
    ├── Backtest Runner
    ├── Performance Evaluator
    └── Results Analyzer
```

## Installation

### Prerequisites

```bash
pip install requests beautifulsoup4 schedule aiohttp
```

### Optional Dependencies

For enhanced GitHub API access:
```bash
export GITHUB_TOKEN="your_github_token"
```

## Usage

### Basic Usage

```python
from agents.strategy_research_agent import StrategyResearchAgent

# Initialize agent
agent = StrategyResearchAgent()

# Run a single research scan
results = agent.run()
print(f"Found {results['discoveries_found']} new strategies")
```

### Manual Source Search

```python
# Search specific sources
arxiv_strategies = agent.search_arxiv("momentum trading", max_results=10)
github_strategies = agent.search_github("trading strategy", max_results=20)
ssrn_strategies = agent.search_ssrn("quantitative trading", max_results=15)
qc_strategies = agent.search_quantconnect("strategy", max_results=10)
```

### Strategy Testing

```python
# Test a discovered strategy
if arxiv_strategies:
    discovery = arxiv_strategies[0]
    test_results = agent.test_discovered_strategy(discovery)
    
    if 'error' not in test_results:
        print(f"Strategy performance: {test_results}")
    else:
        print(f"Test failed: {test_results['error']}")
```

### Periodic Scanning

```python
# Schedule automatic scans every 12 hours
agent.schedule_periodic_scans(interval_hours=12)

# Keep the agent running
import time
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Stopping agent...")
```

## Configuration

### Agent Configuration

```yaml
# config/app_config.yaml
strategy_research:
  sources:
    - arxiv
    - github
    - ssrn
    - quantconnect
  
  scan_interval: 24  # hours
  max_results_per_source: 50
  confidence_threshold: 0.3
  test_high_confidence_only: true
  
  # GitHub API settings
  github:
    token: ${GITHUB_TOKEN}
    rate_limit: 5000
    
  # Request settings
  requests:
    timeout: 30
    retries: 3
    user_agent: "Evolve-StrategyResearch/1.0"
```

### Source-Specific Settings

```yaml
strategy_research:
  arxiv:
    categories:
      - "q-fin.TR"  # Trading and Market Microstructure
      - "q-fin.CP"  # Computational Finance
    max_results: 50
    
  github:
    min_stars: 10
    languages:
      - Python
      - Jupyter Notebook
    topics:
      - trading
      - finance
      - quantitative
      
  ssrn:
    journals:
      - "Journal of Finance"
      - "Review of Financial Studies"
    max_results: 30
```

## Strategy Discovery Process

### 1. Source Crawling

The agent crawls each configured source:

- **arXiv**: Uses arXiv API to search papers by category and keywords
- **SSRN**: Scrapes SSRN website for relevant papers
- **GitHub**: Uses GitHub API to search repositories
- **QuantConnect**: Scrapes QuantConnect forums for strategies

### 2. Content Analysis

For each discovered item, the agent:

1. **Extracts Metadata**: Title, authors, description, URL
2. **Analyzes Content**: Identifies strategy type and confidence
3. **Extracts Code**: Finds executable code snippets
4. **Detects Parameters**: Identifies strategy parameters
5. **Generates Tags**: Creates relevant tags for categorization

### 3. Strategy Classification

Strategies are classified into types:

- **Momentum**: Trend-following strategies
- **Mean Reversion**: Contrarian strategies
- **ML**: Machine learning approaches
- **Options**: Options trading strategies
- **Arbitrage**: Statistical arbitrage strategies

### 4. Code Generation

The agent generates executable Python code:

```python
class DiscoveredStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.lookback_period = kwargs.get('lookback_period', 20)
        self.threshold = kwargs.get('threshold', 0.5)
    
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Generated strategy logic
        signals = df.copy()
        signals['signal'] = 0
        
        # Strategy implementation
        df['momentum'] = df['close'] / df['close'].shift(self.lookback_period) - 1
        signals['signal'] = np.where(df['momentum'] > self.threshold, 1, 0)
        
        return signals
```

### 5. Testing and Integration

Generated strategies are:

1. **Backtested**: Using historical data
2. **Evaluated**: Performance metrics calculated
3. **Compared**: Against existing strategies
4. **Integrated**: Saved to strategy registry if promising

## API Reference

### StrategyResearchAgent

#### Constructor

```python
StrategyResearchAgent(config_path: str = "config/app_config.yaml")
```

#### Methods

##### `run(**kwargs) -> Dict[str, Any]`

Main execution method that runs a complete research scan.

**Returns:**
```python
{
    'status': 'success' | 'error',
    'discoveries_found': int,
    'strategies_tested': int,
    'summary': Dict,
    'tested_strategies': List[Dict]
}
```

##### `search_arxiv(query: str, max_results: int = 50) -> List[StrategyDiscovery]`

Search arXiv for trading strategy papers.

**Parameters:**
- `query`: Search query string
- `max_results`: Maximum number of results to return

**Returns:** List of `StrategyDiscovery` objects

##### `search_github(query: str, max_results: int = 50) -> List[StrategyDiscovery]`

Search GitHub for trading strategy repositories.

**Parameters:**
- `query`: Search query string
- `max_results`: Maximum number of results to return

**Returns:** List of `StrategyDiscovery` objects

##### `search_ssrn(query: str, max_results: int = 30) -> List[StrategyDiscovery]`

Search SSRN for trading strategy papers.

**Parameters:**
- `query`: Search query string
- `max_results`: Maximum number of results to return

**Returns:** List of `StrategyDiscovery` objects

##### `search_quantconnect(query: str, max_results: int = 30) -> List[StrategyDiscovery]`

Search QuantConnect forums for strategies.

**Parameters:**
- `query`: Search query string
- `max_results`: Maximum number of results to return

**Returns:** List of `StrategyDiscovery` objects

##### `test_discovered_strategy(discovery: StrategyDiscovery) -> Dict[str, Any]`

Test a discovered strategy using the backtester.

**Parameters:**
- `discovery`: StrategyDiscovery object to test

**Returns:** Backtest results dictionary

##### `generate_strategy_code(discovery: StrategyDiscovery) -> str`

Generate executable Python code from a discovery.

**Parameters:**
- `discovery`: StrategyDiscovery object

**Returns:** Generated Python code string

##### `save_discovery(discovery: StrategyDiscovery) -> str`

Save a discovery to disk.

**Parameters:**
- `discovery`: StrategyDiscovery object to save

**Returns:** File path where discovery was saved

##### `schedule_periodic_scans(interval_hours: int = 24)`

Schedule periodic research scans.

**Parameters:**
- `interval_hours`: Hours between scans

##### `get_discovery_summary() -> Dict[str, Any]`

Get summary of all discovered strategies.

**Returns:** Summary statistics dictionary

### StrategyDiscovery

Data class representing a discovered strategy.

```python
@dataclass
class StrategyDiscovery:
    source: str                    # Source (arxiv, github, ssrn, quantconnect)
    title: str                     # Strategy title
    description: str               # Strategy description
    authors: List[str]             # Author names
    url: str                       # Source URL
    discovered_date: str           # ISO format date
    strategy_type: str             # Strategy type
    confidence_score: float        # Confidence score (0-1)
    code_snippets: List[str]       # Extracted code snippets
    parameters: Dict[str, Any]     # Strategy parameters
    requirements: List[str]        # Required dependencies
    tags: List[str]                # Strategy tags
```

## File Structure

```
strategies/discovered/
├── arxiv/
│   ├── momentum_strategy_1234567890.json
│   └── ml_strategy_1234567891.json
├── github/
│   ├── trading_repo_1234567892.json
│   └── quant_strategy_1234567893.json
├── ssrn/
│   └── paper_strategy_1234567894.json
└── quantconnect/
    └── forum_strategy_1234567895.json
```

## Examples

### Complete Workflow Example

```python
from agents.strategy_research_agent import StrategyResearchAgent
import json

# Initialize agent
agent = StrategyResearchAgent()

# Run comprehensive scan
results = agent.run()

# Process results
if results['status'] == 'success':
    print(f"Found {results['discoveries_found']} strategies")
    
    # Test high-confidence strategies
    for strategy_data in results['tested_strategies']:
        discovery = strategy_data['discovery']
        test_results = strategy_data['test_results']
        
        print(f"Strategy: {discovery.title}")
        print(f"Type: {discovery.strategy_type}")
        print(f"Confidence: {discovery.confidence_score:.2f}")
        
        if 'error' not in test_results:
            print(f"Performance: {test_results.get('total_return', 'N/A')}")
        else:
            print(f"Test failed: {test_results['error']}")
```

### Custom Search Example

```python
# Search for specific strategy types
momentum_strategies = agent.search_arxiv("momentum trading strategy")
ml_strategies = agent.search_github("machine learning trading")

# Combine and filter results
all_strategies = momentum_strategies + ml_strategies
high_confidence = [s for s in all_strategies if s.confidence_score > 0.7]

# Test promising strategies
for strategy in high_confidence[:5]:
    print(f"Testing: {strategy.title}")
    results = agent.test_discovered_strategy(strategy)
    print(f"Results: {results}")
```

### Integration with Other Agents

```python
from agents import StrategyResearchAgent, ModelInnovationAgent

# Initialize agents
strategy_agent = StrategyResearchAgent()
model_agent = ModelInnovationAgent()

# Run strategy discovery
strategy_results = strategy_agent.run()

# For each discovered strategy, create corresponding models
for strategy_data in strategy_results['tested_strategies']:
    discovery = strategy_data['discovery']
    
    if discovery.strategy_type == "ml":
        # Use ModelInnovationAgent to create ML models
        model_config = {
            'strategy_type': discovery.strategy_type,
            'parameters': discovery.parameters,
            'requirements': discovery.requirements
        }
        
        model_results = model_agent.run(**model_config)
        print(f"Created models for {discovery.title}")
```

## Best Practices

### 1. Rate Limiting

- Respect API rate limits for external sources
- Use appropriate delays between requests
- Implement exponential backoff for failed requests

### 2. Content Quality

- Set appropriate confidence thresholds
- Validate extracted code before execution
- Test strategies with multiple datasets

### 3. Storage Management

- Regularly clean up old discoveries
- Archive successful strategies
- Maintain discovery history for analysis

### 4. Error Handling

- Handle network failures gracefully
- Log all errors for debugging
- Implement fallback mechanisms

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   ```
   Error: GitHub API rate limit exceeded
   Solution: Use GitHub token or reduce request frequency
   ```

2. **Network Timeouts**
   ```
   Error: Request timeout
   Solution: Increase timeout settings or check network connection
   ```

3. **Code Generation Failures**
   ```
   Error: Failed to generate strategy code
   Solution: Check discovery content quality and parameters
   ```

4. **Backtest Failures**
   ```
   Error: Strategy test failed
   Solution: Validate generated code and check dependencies
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('agents.strategy_research_agent').setLevel(logging.DEBUG)
```

### Performance Monitoring

Monitor agent performance:

```python
# Get performance metrics
summary = agent.get_discovery_summary()
print(f"Total discoveries: {summary['total_discoveries']}")
print(f"Test success rate: {summary['test_results'] / summary['total_discoveries']:.2%}")
```

## Contributing

### Adding New Sources

To add a new discovery source:

1. Create a new search method in `StrategyResearchAgent`
2. Implement content extraction logic
3. Add source-specific configuration
4. Update tests and documentation

### Extending Strategy Types

To support new strategy types:

1. Update `_extract_strategy_from_text()` method
2. Add strategy type to classification logic
3. Create code generation templates
4. Update tests and examples

## License

This agent is part of the Evolve trading platform and follows the same license terms. 