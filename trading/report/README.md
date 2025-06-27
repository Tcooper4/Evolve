# Report Generation System

A comprehensive reporting system for the Evolve trading platform that generates detailed reports after each forecast and strategy execution, including trade reports, model performance metrics, and strategy reasoning.

## Features

- **Trade Reports**: PnL analysis, win rate, average gains/losses, Sharpe ratio, drawdown
- **Model Reports**: MSE, MAE, RMSE, accuracy, precision, recall, F1 score
- **Strategy Reasoning**: GPT-powered analysis of why actions were taken
- **Multiple Formats**: PDF, Markdown, and HTML output
- **Integrations**: Slack, Notion, and email notifications
- **Automated Service**: Redis pub/sub service for automatic report generation
- **Visualizations**: Equity curves, prediction vs actual charts, PnL distributions
- **Profitability Heatmap**: Visual heatmap of trade profitability over time (day vs. hour)
- **Model Summary**: Highlights most and least successful models (by PnL)
- **Per-Trade Execution Log**: Table of all trades with time, action, outcome, and model
- **Configurable Chart Types**: Toggle chart/report sections via `report_config`

## New Report Sections

- **🔥 Trade Profitability Heatmap**: Shows a heatmap of trade PnL by day and hour (if timestamps are available in trade data).
- **🏆 Model Summary**: Lists the most and least successful models by total PnL, trade count, and average PnL.
- **🧾 Per-Trade Execution Log**: Table of all trades with columns for time, action, PnL, result, and model ID.

## Chart/Section Configuration

You can control which charts and sections appear in the report by passing a `report_config` dictionary to `ReportGenerator`:

```python
report_config = {
    'equity_curve': True,
    'predictions': True,
    'pnl_distribution': True,
    'heatmap': True,  # Show profitability heatmap
    'model_summary': True,  # Show model summary
    'trade_log': True  # Show per-trade log
}
generator = ReportGenerator(report_config=report_config)
```

## Quick Start

### Basic Usage

```python
from trading.report.report_generator import generate_quick_report

# Sample data
trade_data = {
    'trades': [
        {'pnl': 100, 'duration': 3600, 'timestamp': '2024-06-20T10:00:00', 'action': 'BUY', 'result': 'WIN', 'model_id': 'lstm_v1'},
        {'pnl': -50, 'duration': 1800, 'timestamp': '2024-06-20T11:00:00', 'action': 'SELL', 'result': 'LOSS', 'model_id': 'lstm_v1'},
        {'pnl': 200, 'duration': 7200, 'timestamp': '2024-06-21T09:00:00', 'action': 'BUY', 'result': 'WIN', 'model_id': 'xgb_v2'}
    ]
}

model_data = {
    'predictions': [100, 102, 98, 105, 103],
    'actuals': [100, 101, 99, 104, 102]
}

strategy_data = {
    'strategy_name': 'RSI Strategy',
    'symbol': 'AAPL',
    'timeframe': '1h',
    'signals': ['BUY', 'SELL', 'BUY'],
    'market_conditions': {'trend': 'bullish'},
    'performance': {'total_return': 0.15},
    'parameters': {'rsi_period': 14}
}

# Generate report
report_data = generate_quick_report(
    trade_data=trade_data,
    model_data=model_data,
    strategy_data=strategy_data,
    symbol='AAPL',
    timeframe='1h',
    period='7d'
)

print(f"Report generated: {report_data['report_id']}")
print(f"Files: {report_data['files']}")
```

### Service Integration

```python
from trading.report.report_client import ReportClient

# Initialize client
client = ReportClient()

# Trigger automated report generation
event_id = client.trigger_strategy_report(
    strategy_data=strategy_data,
    trade_data=trade_data,
    model_data=model_data,
    symbol='AAPL',
    timeframe='1h',
    period='7d'
)

# Wait for completion
report_result = client.wait_for_report(event_id, timeout=60)
if report_result:
    print(f"Report completed: {report_result['report_id']}")
```

## Installation

### Dependencies

```bash
pip install redis openai jinja2 matplotlib seaborn pandas numpy requests
```

### Environment Variables

```bash
# OpenAI for GPT reasoning
export OPENAI_API_KEY="your_openai_api_key"

# Redis for service communication
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"

# Integrations (optional)
export NOTION_TOKEN="your_notion_token"
export SLACK_WEBHOOK="your_slack_webhook_url"

# Email configuration (optional)
export EMAIL_SMTP_SERVER="smtp.gmail.com"
export EMAIL_SMTP_PORT="587"
export EMAIL_USERNAME="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
export EMAIL_FROM="your_email@gmail.com"
export EMAIL_TO="recipient@example.com"
```

## Components

### ReportGenerator

The core report generation engine that calculates metrics and creates reports.

```python
from trading.report.report_generator import ReportGenerator

generator = ReportGenerator(
    openai_api_key="your_key",
    notion_token="your_token",
    slack_webhook="your_webhook",
    email_config=email_config,
    output_dir="reports"
)

report_data = generator.generate_comprehensive_report(
    trade_data=trade_data,
    model_data=model_data,
    strategy_data=strategy_data,
    symbol='AAPL',
    timeframe='1h',
    period='7d'
)
```

### ReportService

Redis pub/sub service for automated report generation.

```bash
# Start the service
python trading/report/launch_report_service.py

# Or with custom configuration
python trading/report/launch_report_service.py \
    --redis-host localhost \
    --redis-port 6379 \
    --service-name report_service \
    --output-dir reports
```

### ReportClient

Client for interacting with the report service.

```python
from trading.report.report_client import ReportClient

client = ReportClient()

# Check service status
status = client.get_service_status()
print(f"Service running: {status['running']}")

# List available reports
reports = client.list_available_reports()
for report in reports:
    print(f"{report['report_id']} ({report['format']})")
```

## Report Types

### Trade Report

Comprehensive trading performance analysis:

- **Total Trades**: Number of executed trades
- **Win Rate**: Percentage of profitable trades
- **Total PnL**: Net profit/loss
- **Average Gain/Loss**: Mean profit and loss per trade
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Profit Factor**: Ratio of gross profit to gross loss

### Model Report

Model performance evaluation:

- **MSE/MAE/RMSE**: Error metrics
- **Accuracy**: Correct prediction percentage
- **Precision/Recall/F1**: Classification metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Volatility**: Price movement variability
- **Max Drawdown**: Model performance decline

### Strategy Reasoning

AI-powered analysis of strategy decisions:

- **Summary**: Executive summary of actions taken
- **Key Factors**: Main drivers of decisions
- **Risk Assessment**: Risk analysis and mitigation
- **Confidence Level**: AI confidence in analysis
- **Recommendations**: Future action suggestions
- **Market Conditions**: Market environment analysis

## Output Formats

### Markdown

Clean, readable format suitable for documentation:

```markdown
# Trading Report - AAPL

**Report ID:** report_1234567890  
**Generated:** 2024-01-01T12:00:00  
**Symbol:** AAPL  
**Timeframe:** 1h  
**Period:** 7d

## 📊 Trade Performance

### Summary
- **Total Trades:** 5
- **Win Rate:** 60.0%
- **Total PnL:** $300.00
- **Sharpe Ratio:** 1.25
```

### HTML

Rich, interactive format with styling and charts:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Trading Report - AAPL</title>
    <style>
        /* Professional styling */
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading Report - AAPL</h1>
        </div>
        <div class="metrics-grid">
            <!-- Interactive metrics cards -->
        </div>
        <div class="charts">
            <!-- Embedded charts -->
        </div>
    </div>
</body>
</html>
```

### PDF

Professional format for sharing and archiving:

- High-quality charts and graphics
- Consistent formatting
- Print-friendly layout
- Professional appearance

## Integrations

### Slack

Automated notifications with rich formatting:

```python
# Configure in ReportGenerator
generator = ReportGenerator(
    slack_webhook="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
)
```

### Notion

Direct integration with Notion databases:

```python
# Configure in ReportGenerator
generator = ReportGenerator(
    notion_token="your_notion_integration_token"
)
```

### Email

Automated email delivery with attachments:

```python
# Configure in ReportGenerator
email_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_app_password',
    'from_email': 'your_email@gmail.com',
    'to_email': 'recipient@example.com'
}

generator = ReportGenerator(email_config=email_config)
```

## Configuration

### Report Templates

Customize report appearance by modifying templates:

```python
# Custom template
template_str = """
# Custom Trading Report

## Performance Summary
- Symbol: {{ symbol }}
- Total PnL: ${{ "%.2f"|format(trade_metrics.total_pnl) }}
- Win Rate: {{ "%.1f"|format(trade_metrics.win_rate * 100) }}%
"""

# Use custom template
generator = ReportGenerator()
generator._get_markdown_template = lambda: Template(template_str)
```

### Chart Customization

Modify chart appearance and types:

```python
def custom_charts(self, trade_data, model_data, symbol):
    """Generate custom charts."""
    charts = {}
    
    # Custom equity curve
    plt.figure(figsize=(12, 8))
    # ... custom plotting code ...
    
    return charts

# Override chart generation
generator._generate_charts = custom_charts
```

## API Reference

### ReportGenerator

#### Methods

- `generate_comprehensive_report()`: Generate complete report
- `_calculate_trade_metrics()`: Calculate trading performance metrics
- `_calculate_model_metrics()`: Calculate model performance metrics
- `_generate_strategy_reasoning()`: Generate AI-powered strategy analysis
- `_generate_charts()`: Create visualizations
- `_generate_markdown_report()`: Create Markdown format
- `_generate_html_report()`: Create HTML format
- `_generate_pdf_report()`: Create PDF format

### ReportClient

#### Methods

- `generate_report()`: Generate report directly
- `trigger_forecast_report()`: Trigger forecast report generation
- `trigger_strategy_report()`: Trigger strategy report generation
- `trigger_backtest_report()`: Trigger backtest report generation
- `wait_for_report()`: Wait for report completion
- `get_recent_reports()`: Get recent reports
- `list_available_reports()`: List all available reports
- `get_report_files()`: Get report file paths
- `delete_report()`: Delete report and files
- `get_service_status()`: Get service status

### ReportService

#### Methods

- `start()`: Start the service
- `stop()`: Stop the service
- `get_status()`: Get service status

## Examples

### Integration with Trading System

```python
# After strategy execution
def on_strategy_completed(strategy_result, trade_data, model_data):
    """Handle strategy completion."""
    
    # Generate report
    report_data = generate_quick_report(
        trade_data=trade_data,
        model_data=model_data,
        strategy_data=strategy_result,
        symbol=strategy_result['symbol'],
        timeframe=strategy_result['timeframe'],
        period=strategy_result['period']
    )
    
    # Send to integrations
    if report_data['trade_metrics'].total_pnl > 0:
        # Positive PnL - send to Slack
        send_to_slack(report_data)
    
    # Archive to Notion
    archive_to_notion(report_data)
    
    return report_data
```

### Custom Metrics

```python
def custom_trade_metrics(trade_data):
    """Calculate custom trade metrics."""
    trades = trade_data['trades']
    
    # Custom metrics
    custom_metrics = {
        'total_trades': len(trades),
        'avg_trade_duration': np.mean([t['duration'] for t in trades]),
        'best_trade': max([t['pnl'] for t in trades]),
        'worst_trade': min([t['pnl'] for t in trades]),
        'profit_factor': sum([t['pnl'] for t in trades if t['pnl'] > 0]) / 
                        abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
    }
    
    return custom_metrics
```

### Automated Reporting

```python
# Set up automated reporting
def setup_automated_reporting():
    """Setup automated report generation."""
    
    # Start report service
    service = ReportService()
    service.start()
    
    # Configure triggers
    def on_forecast_complete(forecast_data):
        """Trigger report on forecast completion."""
        client = ReportClient()
        client.trigger_forecast_report(
            forecast_data=forecast_data,
            symbol=forecast_data['symbol'],
            timeframe=forecast_data['timeframe'],
            period=forecast_data['period']
        )
    
    def on_strategy_complete(strategy_data):
        """Trigger report on strategy completion."""
        client = ReportClient()
        client.trigger_strategy_report(
            strategy_data=strategy_data,
            trade_data=strategy_data['trades'],
            model_data=strategy_data['model'],
            symbol=strategy_data['symbol'],
            timeframe=strategy_data['timeframe'],
            period=strategy_data['period']
        )
    
    return on_forecast_complete, on_strategy_complete
```

## Testing

Run the test suite:

```bash
# Run all tests
python trading/report/test_report_system.py

# Run specific test class
python -m unittest trading.report.test_report_system.TestReportGenerator

# Run with coverage
coverage run trading/report/test_report_system.py
coverage report
```

## Demo

Run the demo to see the system in action:

```bash
python trading/report/demo_report_generation.py
```

This will:
1. Generate sample trading data
2. Create comprehensive reports
3. Demonstrate service integration
4. Show different output formats
5. Test integrations

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```
   Error: Redis connection failed
   Solution: Ensure Redis is running and accessible
   ```

2. **OpenAI API Error**
   ```
   Error: OpenAI API key not configured
   Solution: Set OPENAI_API_KEY environment variable
   ```

3. **Chart Generation Error**
   ```
   Error: Matplotlib backend not available
   Solution: Install required dependencies: pip install matplotlib seaborn
   ```

4. **File Permission Error**
   ```
   Error: Cannot write to output directory
   Solution: Ensure write permissions to reports directory
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific components
logging.getLogger('trading.report').setLevel(logging.DEBUG)
```

### Service Monitoring

Monitor service health:

```python
client = ReportClient()
status = client.get_service_status()

if not status['running']:
    print("Service is not running")
    print(f"Last heartbeat: {status['last_heartbeat']}")
    print(f"Redis connected: {status['redis_connected']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 