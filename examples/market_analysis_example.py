"""
Example script demonstrating the usage of market analysis utilities.
"""
import sys
from pathlib import Path
import pandas as pd
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.data_validation import DataValidator
from src.utils.data_pipeline import DataPipeline
from src.utils.visualization import MarketVisualizer
from src.utils.config_manager import ConfigManager
from src.analysis.market_analysis import MarketAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating the usage of market analysis utilities."""
    try:
        # Initialize configuration manager
        config_manager = ConfigManager(project_root / 'config' / 'market_analysis_config.yaml')
        
        # Validate configuration
        is_valid, error_message = config_manager.validate_config()
        if not is_valid:
            logger.error(f"Configuration validation failed: {error_message}")
            return
        
        # Initialize data pipeline
        pipeline = DataPipeline(config_manager.get_pipeline_settings())
        
        # Load and process data
        data_file = project_root / 'data' / 'sample_market_data.csv'
        if not pipeline.run_pipeline(data_file):
            logger.error("Failed to run data pipeline")
            return
        
        # Get processed data
        processed_data = pipeline.get_processed_data()
        if processed_data is None:
            logger.error("No processed data available")
            return
        
        # Initialize market analysis
        market_analysis = MarketAnalysis()
        
        # Perform market analysis
        analysis_results = market_analysis.analyze_market(processed_data)
        
        # Initialize visualizer
        visualizer = MarketVisualizer(
            style=config_manager.get_visualization_settings().get('style', 'seaborn')
        )
        
        # Create output directory for plots
        plots_dir = project_root / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        # 1. Price data plot
        visualizer.plot_price_data(
            processed_data,
            title="Market Price Data",
            save_path=plots_dir / 'price_data.png'
        )
        
        # 2. Technical indicators plot
        indicators = ['sma_20', 'sma_50', 'sma_200', 'rsi']
        visualizer.plot_technical_indicators(
            processed_data,
            indicators,
            title="Technical Indicators",
            save_path=plots_dir / 'technical_indicators.png'
        )
        
        # 3. Market regime plot
        visualizer.plot_market_regime(
            processed_data,
            analysis_results['regime'],
            title="Market Regime Analysis",
            save_path=plots_dir / 'market_regime.png'
        )
        
        # 4. Correlation matrix
        visualizer.plot_correlation_matrix(
            processed_data,
            columns=['close', 'volume', 'returns', 'volatility'],
            title="Market Data Correlations",
            save_path=plots_dir / 'correlation_matrix.png'
        )
        
        # 5. Distribution plots
        visualizer.plot_distribution(
            processed_data,
            'returns',
            title="Returns Distribution",
            save_path=plots_dir / 'returns_distribution.png'
        )
        
        # 6. Rolling statistics
        visualizer.plot_rolling_statistics(
            processed_data,
            'close',
            window=20,
            title="Price Rolling Statistics",
            save_path=plots_dir / 'rolling_statistics.png'
        )
        
        logger.info("Market analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in market analysis: {str(e)}")

if __name__ == '__main__':
    main() 