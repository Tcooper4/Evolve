# Portfolio Management Module

## Overview

The Portfolio Management Module provides comprehensive portfolio allocation and risk management capabilities for the Evolve trading platform. It implements modern portfolio theory, risk parity, Kelly criterion, and other advanced allocation strategies with robust risk monitoring and dynamic rebalancing.

## Architecture

```
portfolio/
├── __init__.py              # Module exports and convenience functions
├── allocator.py             # Portfolio allocation strategies
└── risk_manager.py          # Risk management and monitoring
```

## Core Components

### 1. Portfolio Allocator (`allocator.py`)

The allocator implements multiple portfolio allocation strategies:

#### Allocation Strategies

- **Equal Weight**: Simple equal allocation across all assets
- **Minimum Variance**: Optimizes for lowest portfolio volatility
- **Maximum Sharpe**: Maximizes risk-adjusted returns
- **Risk Parity**: Equalizes risk contribution from each asset
- **Kelly Criterion**: Optimal position sizing based on expected returns
- **Black-Litterman**: Incorporates investor views with market equilibrium
- **Mean-Variance**: Traditional Markowitz optimization
- **Maximum Diversification**: Maximizes portfolio diversification ratio

#### Key Classes

```python
class AssetMetrics:
    """Asset-specific metrics for allocation"""
    ticker: str
    expected_return: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation: Dict[str, float]
    market_cap: Optional[float]
    sector: Optional[str]
    sentiment_score: Optional[float]

class AllocationResult:
    """Result of portfolio allocation"""
    strategy: AllocationStrategy
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_contribution: Dict[str, float]
    diversification_ratio: float
    constraints_satisfied: bool
    optimization_status: str
    metadata: Dict[str, Any]
```

### 2. Risk Manager (`risk_manager.py`)

The risk manager provides comprehensive risk monitoring and control:

#### Risk Metrics

- **Value at Risk (VaR)**: Maximum expected loss at confidence level
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Portfolio return standard deviation
- **Beta**: Market sensitivity measure
- **Correlation Risk**: Portfolio concentration risk
- **Exposure Concentration**: Largest position weight
- **Sector Concentration**: Sector-level exposure

#### Risk Limits

```python
class RiskLimits:
    max_drawdown: float = 0.15      # 15% maximum drawdown
    max_exposure: float = 0.3       # 30% maximum single position
    max_leverage: float = 2.0       # 2x maximum leverage
    target_volatility: float = 0.15 # 15% target volatility
    var_limit: float = 0.02         # 2% daily VaR limit
    max_correlation: float = 0.7    # 70% maximum correlation
    sector_limit: float = 0.4       # 40% maximum sector exposure
    liquidity_limit: float = 0.1    # 10% maximum illiquid position
```

## Usage Examples

### Basic Portfolio Allocation

```python
from portfolio import (
    PortfolioAllocator, 
    AllocationStrategy, 
    AssetMetrics,
    create_allocator
)

# Create allocator
allocator = create_allocator()

# Define assets
assets = [
    AssetMetrics(
        ticker="AAPL",
        expected_return=0.12,
        volatility=0.20,
        sharpe_ratio=0.5,
        beta=1.1,
        correlation={"TSLA": 0.3, "NVDA": 0.4},
        market_cap=2000000000000,
        sector="Technology",
        sentiment_score=0.6
    ),
    AssetMetrics(
        ticker="TSLA",
        expected_return=0.18,
        volatility=0.35,
        sharpe_ratio=0.46,
        beta=1.8,
        correlation={"AAPL": 0.3, "NVDA": 0.5},
        market_cap=800000000000,
        sector="Automotive",
        sentiment_score=0.7
    )
]

# Allocate using Maximum Sharpe strategy
result = allocator.allocate_portfolio(assets, AllocationStrategy.MAXIMUM_SHARPE)

print(f"Expected Return: {result.expected_return:.2%}")
print(f"Expected Volatility: {result.expected_volatility:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Weights: {result.weights}")
```

### Risk Management

```python
from portfolio import (
    PortfolioRiskManager,
    create_risk_manager
)

# Create risk manager
risk_manager = create_risk_manager()

# Calculate portfolio risk
positions = {"AAPL": 0.4, "TSLA": 0.3, "NVDA": 0.3}
market_data = {
    "AAPL": pd.DataFrame({"returns": returns_aapl}),
    "TSLA": pd.DataFrame({"returns": returns_tsla}),
    "NVDA": pd.DataFrame({"returns": returns_nvda})
}

risk_metrics = risk_manager.calculate_portfolio_risk(positions, market_data)

# Check risk limits
violations = risk_manager.check_risk_limits(positions, risk_metrics)

for violation in violations:
    print(f"Risk violation: {violation.risk_metric.value}")
    print(f"Current: {violation.current_value:.2%}, Limit: {violation.limit_value:.2%}")
    print(f"Action: {violation.action_required}")
```

### Portfolio Simulation

```python
# Simulate portfolio performance with rebalancing
simulation = risk_manager.simulate_portfolio_returns(
    positions, 
    market_data, 
    rebalancing_freq='monthly'
)

print(f"Final Value: ${simulation['portfolio_value'].iloc[-1]:.2f}")
print(f"Total Return: {simulation['cumulative_return'].iloc[-1]:.2%}")
print(f"Max Drawdown: {simulation['drawdown'].min():.2%}")
```

### Stress Testing

```python
# Define stress scenarios
scenarios = {
    'Market Crash': {
        'AAPL': -0.3, 'TSLA': -0.4, 'NVDA': -0.35
    },
    'Tech Rally': {
        'AAPL': 0.3, 'TSLA': 0.4, 'NVDA': 0.35
    },
    'Volatility Spike': {
        'AAPL': 0.1, 'TSLA': -0.1, 'NVDA': 0.05
    }
}

# Run stress tests
stress_results = risk_manager.stress_test_portfolio(positions, market_data, scenarios)

for scenario, metrics in stress_results.items():
    print(f"{scenario}:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  VaR (95%): {metrics['var_95']:.2%}")
```

### Strategy Comparison

```python
# Compare all allocation strategies
results = allocator.compare_strategies(assets)

for strategy_name, result in results.items():
    print(f"{strategy_name}:")
    print(f"  Return: {result.expected_return:.2%}")
    print(f"  Volatility: {result.expected_volatility:.2%}")
    print(f"  Sharpe: {result.sharpe_ratio:.3f}")

# Find optimal strategy
optimal_strategy, optimal_result = allocator.get_optimal_strategy(assets, 'sharpe')
print(f"Optimal Strategy: {optimal_strategy.value}")
```

## Configuration

### Allocator Configuration

```yaml
# config/app_config.yaml
portfolio:
  max_weight: 0.3              # Maximum position weight
  min_weight: 0.01             # Minimum position weight
  target_volatility: 0.15      # Target portfolio volatility
  risk_free_rate: 0.02         # Risk-free rate for Sharpe calculation
  max_iterations: 1000         # Optimization iterations
  tolerance: 1e-6              # Optimization tolerance
  kelly_fraction: 0.25         # Kelly criterion fraction
  max_kelly_weight: 0.5        # Maximum Kelly weight
```

### Risk Manager Configuration

```yaml
# config/app_config.yaml
risk_management:
  max_drawdown: 0.15           # Maximum drawdown limit
  max_exposure: 0.3            # Maximum single position exposure
  max_leverage: 2.0            # Maximum leverage
  target_volatility: 0.15      # Target volatility
  var_limit: 0.02              # Daily VaR limit
  max_correlation: 0.7         # Maximum correlation
  sector_limit: 0.4            # Maximum sector exposure
  liquidity_limit: 0.1         # Maximum illiquid position
  var_confidence: 0.95         # VaR confidence level
  lookback_period: 252         # Risk calculation lookback
  rebalancing_frequency: daily # Rebalancing frequency
```

## Advanced Features

### Dynamic Rebalancing

The risk manager automatically generates rebalancing actions when risk limits are violated:

```python
# Generate rebalancing actions
actions = risk_manager.generate_rebalancing_actions(
    current_positions, 
    target_positions, 
    risk_violations
)

for action in actions:
    print(f"{action.action_type.upper()} {action.ticker}")
    print(f"  {action.current_weight:.3f} → {action.target_weight:.3f}")
    print(f"  Priority: {action.priority}, Reason: {action.reason}")
```

### Risk Attribution

Analyze how each position contributes to portfolio risk:

```python
attribution = risk_manager.calculate_risk_attribution(positions, risk_metrics)

for ticker, metrics in attribution.items():
    print(f"{ticker}:")
    print(f"  Weight: {metrics['weight']:.3f}")
    print(f"  Volatility Contribution: {metrics['volatility_contribution']:.4f}")
    print(f"  VaR Contribution: {metrics['var_contribution']:.4f}")
```

### Position Sizing Optimization

Optimize position sizes to achieve target volatility:

```python
optimized_positions = risk_manager.optimize_position_sizing(
    positions, 
    risk_metrics, 
    target_volatility=0.12
)
```

## Integration with Trading Platform

### Meta Controller Integration

The portfolio management module integrates with the MetaControllerAgent:

```python
from meta.meta_controller import MetaControllerAgent

# Meta controller can trigger portfolio rebalancing
meta_controller = MetaControllerAgent()
meta_controller.trigger_portfolio_rebalancing()
```

### Agent Integration

Portfolio allocation can be integrated with trading agents:

```python
from agents.model_innovation_agent import ModelInnovationAgent
from agents.strategy_research_agent import StrategyResearchAgent

# Agents can use portfolio allocation for position sizing
model_agent = ModelInnovationAgent()
strategy_agent = StrategyResearchAgent()

# Get allocation recommendations
allocation = model_agent.get_portfolio_allocation()
risk_metrics = strategy_agent.assess_portfolio_risk()
```

## Best Practices

### 1. Asset Selection

- Use diverse assets across different sectors and geographies
- Include both growth and value assets
- Consider liquidity and trading costs
- Monitor correlation changes over time

### 2. Risk Management

- Set appropriate risk limits based on investment objectives
- Monitor risk metrics continuously
- Implement dynamic rebalancing
- Use stress testing for scenario planning

### 3. Allocation Strategy Selection

- **Conservative**: Minimum Variance or Risk Parity
- **Balanced**: Maximum Sharpe or Mean-Variance
- **Aggressive**: Kelly Criterion or Maximum Diversification
- **Simple**: Equal Weight for passive management

### 4. Rebalancing

- Rebalance regularly (monthly/quarterly)
- Consider transaction costs
- Use threshold-based rebalancing
- Monitor tax implications

### 5. Performance Monitoring

- Track risk-adjusted returns
- Monitor drawdowns and recovery periods
- Compare against benchmarks
- Analyze attribution and factor exposures

## Testing

### Unit Tests

```bash
# Run portfolio module tests
python -m pytest tests/test_portfolio_modules.py -v
```

### Integration Tests

```bash
# Run complete portfolio analysis
python examples/portfolio_management_example.py
```

## Performance Considerations

### Optimization

- Use efficient optimization algorithms (SLSQP, SCS)
- Implement caching for repeated calculations
- Vectorize operations where possible
- Use parallel processing for large portfolios

### Memory Management

- Clean up large DataFrames after use
- Use generators for large datasets
- Implement lazy loading for market data
- Monitor memory usage during simulations

### Scalability

- Support for 100+ asset portfolios
- Efficient correlation matrix calculations
- Parallel stress testing
- Distributed optimization for large-scale problems

## Troubleshooting

### Common Issues

1. **Optimization Failures**
   - Check constraint feasibility
   - Adjust tolerance parameters
   - Verify input data quality

2. **Risk Limit Violations**
   - Review position sizing
   - Check correlation assumptions
   - Adjust risk limits if appropriate

3. **Performance Issues**
   - Use caching for repeated calculations
   - Optimize data structures
   - Consider parallel processing

### Debug Mode

Enable debug logging for detailed analysis:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug allocation process
allocator = PortfolioAllocator()
result = allocator.allocate_portfolio(assets, strategy, debug=True)
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - ML-based return forecasting
   - Dynamic correlation modeling
   - Automated strategy selection

2. **Advanced Risk Models**
   - Regime-switching models
   - Tail risk modeling
   - Factor risk decomposition

3. **Real-time Monitoring**
   - Live risk dashboard
   - Automated alerts
   - Real-time rebalancing

4. **Multi-Asset Support**
   - Fixed income allocation
   - Alternative investments
   - Currency hedging

### Contributing

To contribute to the portfolio management module:

1. Follow the existing code style
2. Add comprehensive tests
3. Update documentation
4. Include performance benchmarks
5. Submit pull requests with detailed descriptions

## References

- Markowitz, H. (1952). Portfolio Selection
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization
- Kelly, J. L. (1956). A New Interpretation of Information Rate
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios 