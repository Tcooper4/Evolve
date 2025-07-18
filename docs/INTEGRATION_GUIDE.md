# Evolve Trading System - Integration Guide

## Quick Start

The upgraded Evolve trading system is now ready for production use. Here's how to integrate and use all the new components:

## 🚀 Getting Started

### 1. System Initialization

```python
# Initialize the main system
from trading.agents.agent_manager import AgentManager
from trading.portfolio.portfolio_manager import PortfolioManager

# Create agent manager
agent_manager = AgentManager()

# Initialize portfolio manager
portfolio_manager = PortfolioManager()
```

### 2. Core Agent Integration

```python
# Model selection for forecasting
from trading.agents.model_selector_agent import ModelSelectorAgent, ForecastingHorizon, MarketRegime

model_selector = ModelSelectorAgent()
selected_model, confidence = model_selector.select_model(
    horizon=ForecastingHorizon.MEDIUM_TERM,
    market_regime=MarketRegime.TRENDING_UP,
    data_length=1000
)

# Strategy selection
from trading.agents.strategy_selector_agent import StrategySelectorAgent

strategy_selector = StrategySelectorAgent()
recommendation = strategy_selector.select_strategy(
    market_data=price_data,
    asset_symbol="AAPL",
    forecast_horizon=14,
    risk_tolerance="medium"
)
```

### 3. Advanced Strategy Engines

```python
# Pairs trading
from trading.strategies.pairs_trading_engine import PairsTradingEngine

pairs_engine = PairsTradingEngine()
cointegrated_pairs = pairs_engine.find_cointegrated_pairs(price_data, symbols)
signals = pairs_engine.generate_signals(price_data, cointegrated_pairs)

# Breakout strategy
from trading.strategies.breakout_strategy_engine import BreakoutStrategyEngine

breakout_engine = BreakoutStrategyEngine()
consolidation_ranges = breakout_engine.detect_consolidation_ranges(data, "AAPL")
breakout_signals = breakout_engine.detect_breakouts(data, "AAPL")
```

### 4. Portfolio Management

```python
# Portfolio optimization
from trading.portfolio.portfolio_simulator import PortfolioSimulator, OptimizationMethod

portfolio_sim = PortfolioSimulator()
result = portfolio_sim.optimize_portfolio(
    returns_data=returns_df,
    method=OptimizationMethod.BLACK_LITTERMAN,
    constraints=constraints,
    views=market_views
)

# Trade execution
from trading.execution.trade_execution_simulator import TradeExecutionSimulator, OrderType

execution_sim = TradeExecutionSimulator()
order_id = execution_sim.place_order(
    symbol="AAPL",
    order_type=OrderType.MARKET,
    side="buy",
    quantity=100
)
```

### 5. Self-Evolving Components

```python
# Meta-learning feedback
from trading.agents.meta_learning_feedback_agent import MetaLearningFeedbackAgent

feedback_agent = MetaLearningFeedbackAgent()
await feedback_agent.process_model_feedback(model_feedback)

# Self-tuning optimization
from trading.agents.self_tuning_optimizer_agent import SelfTuningOptimizerAgent

optimizer = SelfTuningOptimizerAgent()
triggers = await optimizer.check_optimization_triggers(
    strategy_name="rsi_strategy",
    strategy_performance=performance_series,
    market_data=market_data
)

# Meta-research
from trading.agents.meta_research_agent import MetaResearchAgent

research_agent = MetaResearchAgent()
papers = await research_agent.discover_research_papers()
evaluations = await research_agent.evaluate_models(papers)
```

### 6. Analytics and Commentary

```python
# Alpha attribution
from trading.analytics.alpha_attribution_engine import AlphaAttributionEngine

attribution_engine = AlphaAttributionEngine()
attribution_result = attribution_engine.perform_attribution_analysis(
    portfolio_returns=portfolio_returns,
    strategy_returns=strategy_returns,
    benchmark_returns=benchmark_returns
)

# Commentary generation
from trading.llm.quant_gpt_commentary_agent import QuantGPTCommentaryAgent, CommentaryType

commentary_agent = QuantGPTCommentaryAgent()
commentary = await commentary_agent.generate_commentary(
    CommentaryType.TRADE_EXPLANATION,
    trade_data=trade_info,
    market_data=market_data
)
```

## 🔧 Configuration

### Main Configuration (`config.yaml`)

```yaml
# Agent configuration
agents:
  model_selector:
    learning_rate: 0.01
    exploration_rate: 0.1
    performance_window: 30
  
  strategy_selector:
    optimization_frequency: "weekly"
    min_performance_threshold: 0.5
  
  data_quality:
    anomaly_thresholds:
      z_score_threshold: 3.0
      volume_spike_threshold: 5.0

# Portfolio configuration
portfolio:
  optimization_method: "black_litterman"
  risk_free_rate: 0.02
  transaction_cost_rate: 0.001

# Execution configuration
execution:
  base_commission: 0.005
  base_slippage: 0.0001
  market_hours_only: true
```

### Environment Variables

```bash
# Data sources
export ALPHA_VANTAGE_API_KEY="your_key"
export POLYGON_API_KEY="your_key"

# LLM configuration
export OPENAI_API_KEY="your_key"

# System configuration
export LOG_LEVEL="INFO"
export USE_GPU="true"
```

## 📊 Monitoring and Analytics

### Real-Time Dashboard

```python
# Launch the dashboard
python app.py

# Access via browser: http://localhost:8501
```

### Performance Monitoring

```python
# Get system status
from trading.agents.agent_manager import AgentManager

agent_manager = AgentManager()
status = agent_manager.get_system_status()

# Get performance summary
from trading.analytics.alpha_attribution_engine import AlphaAttributionEngine

attribution_engine = AlphaAttributionEngine()
summary = attribution_engine.get_attribution_summary()
```

### Risk Monitoring

```python
# Check risk metrics
from trading.risk.risk_analyzer import RiskAnalyzer

risk_analyzer = RiskAnalyzer()
risk_metrics = risk_analyzer.calculate_portfolio_risk(portfolio_data)
```

## 🔄 Automation Workflows

### Daily Trading Workflow

```python
async def daily_trading_workflow():
    # 1. Data quality check
    data_quality_agent = DataQualityAgent()
    quality_report = await data_quality_agent.assess_data_quality(market_data, "AAPL")
    
    # 2. Market regime detection
    market_regime = model_selector.detect_market_regime(market_data)
    
    # 3. Model selection
    selected_model, confidence = model_selector.select_model(
        horizon=ForecastingHorizon.SHORT_TERM,
        market_regime=market_regime,
        data_length=len(market_data)
    )
    
    # 4. Strategy selection
    strategy_recommendation = strategy_selector.select_strategy(
        market_data=market_data,
        asset_symbol="AAPL",
        forecast_horizon=5
    )
    
    # 5. Portfolio optimization
    portfolio_result = portfolio_sim.optimize_portfolio(
        returns_data=returns_data,
        method=OptimizationMethod.MEAN_VARIANCE
    )
    
    # 6. Generate signals
    signals = generate_trading_signals(market_data, strategy_recommendation)
    
    # 7. Execute trades
    for signal in signals:
        order_id = execution_sim.place_order(
            symbol=signal.symbol,
            order_type=OrderType.MARKET,
            side=signal.side,
            quantity=signal.quantity
        )
    
    # 8. Generate commentary
    commentary = await commentary_agent.generate_commentary(
        CommentaryType.TRADE_EXPLANATION,
        trade_data=trade_summary
    )
    
    # 9. Update feedback
    await feedback_agent.process_model_feedback(model_performance)
```

### Weekly Optimization Workflow

```python
async def weekly_optimization_workflow():
    # 1. Check optimization triggers
    triggers = await optimizer.check_optimization_triggers(
        strategy_name="all_strategies",
        strategy_performance=performance_data,
        market_data=market_data
    )
    
    # 2. Optimize parameters
    if triggers:
        optimization_result = await optimizer.optimize_strategy_parameters(
            strategy_name="rsi_strategy",
            current_parameters=current_params,
            strategy_performance=performance_data,
            market_data=market_data,
            triggers=triggers
        )
    
    # 3. Research new models
    papers = await research_agent.discover_research_papers()
    evaluations = await research_agent.evaluate_models(papers)
    
    # 4. Implement top models
    implemented = await research_agent.auto_implement_top_models(evaluations)
    
    # 5. Update ensemble weights
    await feedback_agent.update_ensemble_weights()
```

## 🛡️ Error Handling and Resilience

### Graceful Degradation

```python
try:
    # Try primary data source
    data = primary_provider.get_data(symbol)
except Exception as e:
    # Fallback to backup provider
    data = backup_provider.get_data(symbol)
    logger.warning(f"Using backup data provider: {e}")
```

### System Health Monitoring

```python
# Check system health
def check_system_health():
    health_status = {
        'data_quality': data_quality_agent.get_quality_summary(),
        'model_performance': model_selector.get_model_recommendations(),
        'strategy_performance': strategy_selector.get_strategy_recommendations(),
        'portfolio_health': portfolio_manager.get_portfolio_health(),
        'execution_status': execution_sim.get_execution_summary()
    }
    
    # Alert if any component is unhealthy
    for component, status in health_status.items():
        if status.get('status') != 'healthy':
            send_alert(f"Component {component} is unhealthy: {status}")
    
    return health_status
```

## 📈 Performance Optimization

### Memory Management

```python
# Clear old data periodically
def cleanup_old_data():
    cutoff_date = datetime.now() - timedelta(days=30)
    
    # Clean up old performance data
    for agent in [model_selector, strategy_selector, feedback_agent]:
        agent.cleanup_old_data(cutoff_date)
    
    # Clean up old market data
    data_loader.cleanup_old_data(cutoff_date)
```

### Caching

```python
# Cache frequently used data
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_market_regime(market_data_hash):
    return model_selector.detect_market_regime(market_data)
```

## 🎯 Best Practices

### 1. Start Small
- Begin with a few strategies and models
- Gradually add complexity as you understand the system
- Monitor performance closely during initial deployment

### 2. Monitor Everything
- Set up comprehensive logging
- Monitor all agent performance
- Track system health metrics
- Set up alerts for anomalies

### 3. Validate Results
- Always backtest new strategies
- Use out-of-sample testing
- Monitor for overfitting
- Validate model assumptions

### 4. Manage Risk
- Set appropriate position limits
- Use stop-losses and take-profits
- Monitor portfolio concentration
- Diversify across strategies

### 5. Continuous Improvement
- Regularly review and update models
- Incorporate new research findings
- Optimize parameters based on performance
- Stay updated with market conditions

## 🚨 Troubleshooting

### Common Issues

1. **Data Quality Issues**
   ```python
   # Check data quality
   quality_report = data_quality_agent.assess_data_quality(data, symbol)
   if quality_report.quality_level == "poor":
       # Use backup data source
       data = backup_provider.get_data(symbol)
   ```

2. **Model Performance Decline**
   ```python
   # Check for overfitting
   overfitting_analysis = commentary_agent.detect_overfitting(model_data)
   if overfitting_analysis.score > 0.8:
       # Retrain model or adjust parameters
       await feedback_agent.retune_hyperparameters(model_name)
   ```

3. **Strategy Underperformance**
   ```python
   # Check strategy performance
   strategy_performance = strategy_selector.get_strategy_performance()
   if strategy_performance.sharpe_ratio < 0.5:
       # Optimize strategy parameters
       await optimizer.optimize_strategy_parameters(strategy_name)
   ```

### Getting Help

- Check the logs in `logs/` directory
- Review the system status in the dashboard
- Use the agent leaderboard to identify underperforming components
- Consult the comprehensive documentation

## 🎉 Success Metrics

Monitor these key metrics to ensure the system is performing well:

1. **Performance Metrics**
   - Sharpe Ratio > 1.0
   - Maximum Drawdown < 20%
   - Win Rate > 50%
   - Profit Factor > 1.5

2. **System Metrics**
   - Data Quality Score > 0.8
   - Model Confidence > 0.7
   - Strategy Performance > 0.5
   - Execution Success Rate > 95%

3. **Operational Metrics**
   - System Uptime > 99%
   - Response Time < 1 second
   - Error Rate < 1%
   - Optimization Frequency (weekly)

The upgraded Evolve trading system is now ready for production use with comprehensive automation, monitoring, and self-improvement capabilities!
