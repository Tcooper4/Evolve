# Market Analysis System

A comprehensive market analysis system for processing, analyzing, and visualizing market data. The system includes utilities for data validation, preprocessing, analysis, and visualization.

## Features

- **Data Validation**: Robust validation of market data structure and quality
- **Data Pipeline**: End-to-end pipeline for data loading, preprocessing, and analysis
- **Market Analysis**: Technical analysis, market regime detection, and condition analysis
- **Visualization**: Comprehensive plotting capabilities for market data and analysis results
- **Configuration Management**: Flexible configuration system for customizing analysis parameters

## Project Structure

```
market_analysis/
├── config/
│   └── market_analysis_config.yaml  # Configuration file
├── data/
│   └── sample_market_data.csv      # Sample market data
├── examples/
│   └── market_analysis_example.py  # Example usage script
├── src/
│   ├── analysis/
│   │   └── market_analysis.py      # Market analysis module
│   ├── features/
│   │   └── feature_engineering.py  # Feature engineering module
│   └── utils/
│       ├── data_validation.py      # Data validation utilities
│       ├── data_pipeline.py        # Data pipeline utilities
│       ├── visualization.py        # Visualization utilities
│       └── config_manager.py       # Configuration management utilities
└── tests/
    ├── analysis/
    │   └── test_market_analysis.py # Market analysis tests
    └── features/
        └── test_feature_engineering.py  # Feature engineering tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market_analysis.git
cd market_analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (includes optional visualization and reporting packages):
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Prepare your market data in CSV format with the following columns:
   - date
   - open
   - high
   - low
   - close
   - volume

2. Configure your analysis parameters in `config/market_analysis_config.yaml`

3. Run the example script:
```bash
python examples/market_analysis_example.py
```

### Custom Analysis

```python
from src.utils.data_pipeline import DataPipeline
from src.utils.config_manager import ConfigManager
from src.analysis.market_analysis import MarketAnalysis

# Initialize configuration
config_manager = ConfigManager('config/market_analysis_config.yaml')

# Initialize pipeline
pipeline = DataPipeline(config_manager.get_pipeline_settings())

# Load and process data
pipeline.run_pipeline('path/to/your/data.csv')

# Get processed data
processed_data = pipeline.get_processed_data()

# Perform analysis
market_analysis = MarketAnalysis()
analysis_results = market_analysis.analyze_market(processed_data)
```

## Configuration

The system is highly configurable through the `market_analysis_config.yaml` file. Key configuration sections include:

- **Market Conditions**: Thresholds for trend detection, volatility, and volume
- **Analysis Settings**: Parameters for technical indicators and analysis methods
- **Visualization Settings**: Plot styles, colors, and output formats
- **Pipeline Settings**: Data loading, preprocessing, and validation parameters

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by various market analysis methodologies
- Built with Python's data science ecosystem 
