# Visualization Module

The visualization module provides interactive dashboards and data visualization capabilities for the trading platform.

## Structure

```
visualization/
├── dashboards/      # Interactive dashboards
├── charts/          # Chart components
├── widgets/         # UI widgets
└── utils/          # Visualization utilities
```

## Components

### Dashboards

The `dashboards` directory contains:
- Trading dashboard
- Performance dashboard
- System dashboard
- Analytics dashboard
- Custom dashboards

### Charts

The `charts` directory contains:
- Price charts
- Technical indicators
- Performance metrics
- System metrics
- Custom charts

### Widgets

The `widgets` directory contains:
- Control panels
- Data tables
- Status indicators
- Input forms
- Custom widgets

### Utilities

The `utils` directory contains:
- Theme management
- Layout utilities
- Data formatting
- Event handlers
- Helper functions

## Usage

```python
from visualization.dashboards import TradingDashboard
from visualization.charts import PriceChart
from visualization.widgets import ControlPanel
from visualization.utils import ThemeManager

# Create dashboard
dashboard = TradingDashboard()
dashboard.show()

# Add chart
chart = PriceChart()
dashboard.add_chart(chart)

# Add widget
panel = ControlPanel()
dashboard.add_widget(panel)
```

## Testing

```bash
# Run visualization tests
pytest tests/unit/visualization/

# Run specific component tests
pytest tests/unit/visualization/dashboards/
pytest tests/unit/visualization/charts/
```

## Configuration

The visualization module can be configured through:
- Theme settings
- Layout options
- Chart parameters
- Widget properties

## Dependencies

- streamlit
- plotly
- dash
- bokeh
- matplotlib

## Features

- Real-time updates
- Interactive charts
- Custom themes
- Responsive layouts
- Data filtering
- Export capabilities

## Contributing

1. Follow the coding style guide
2. Write unit tests for new features
3. Update documentation
4. Submit a pull request

## License

This module is part of the main project and is licensed under the MIT License. 