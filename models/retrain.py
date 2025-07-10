"""Model retraining script for maintaining and updating forecasting models.

This script handles the retraining of forecasting models based on:
1. Performance degradation
2. Scheduled intervals
3. New data availability
4. Model drift detection

It supports multiple model types and implements a flexible retraining pipeline
that can be customized for different use cases.

Example:
    ```python
    python retrain.py --model lstm --data data/stock_data.csv --force
    ```
"""

import argparse
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from models.forecast_router import ForecastRouter
from trading.utils.data_utils import load_data, prepare_forecast_data
from trading.utils.logging import setup_logging
from trading.config.settings import get_config_value

logger = setup_logging(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrain forecasting models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["arima", "lstm", "xgboost", "prophet", "autoformer", "all"],
        default="all",
        help="Model type to retrain"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of conditions"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of data to use"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Performance degradation threshold"
    )
    return parser.parse_args()

def check_retraining_needed(model_type: str,
                          performance_history: pd.DataFrame,
                          threshold: float) -> bool:
    """Check if model needs retraining."""
    if performance_history.empty:
        return True
    # Get recent performance
    recent = performance_history[
        performance_history['model'] == model_type
    ].tail(10)
    if recent.empty:
        return True
    # Calculate performance trend
    mse_trend = recent['mse'].pct_change().mean()
    return mse_trend > threshold

def retrain_model(model_type: str,
                 data: pd.DataFrame,
                 router: ForecastRouter,
                 **kwargs) -> Dict[str, Any]:
    """Retrain a specific model."""
    try:
        logger.info(f"Retraining {model_type} model")
        # Prepare data
        prepared_data = prepare_forecast_data(data)
        # Get forecast to trigger retraining
        result = router.get_forecast(
            data=prepared_data,
            model_type=model_type,
            **kwargs
        )
        return {
            'model': model_type,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'metadata': result.get('metadata', {})
        }
    except Exception as e:
        logger.error(f"Error retraining {model_type}: {str(e)}")
        return {
            'model': model_type,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main entry point for retraining script."""
    args = parse_args()
    try:
        # Load data
        data = load_data(args.data)
        if data is None:
            logger.error("Failed to load data")
            return
        # Initialize router
        router = ForecastRouter()
        # Get models to retrain
        models = [args.model] if args.model != "all" else router.get_available_models()
        results = []
        for model_type in models:
            # Check if retraining is needed
            if not args.force:
                performance = router.get_model_performance(model_type)
                if not check_retraining_needed(model_type, performance, args.threshold):
                    logger.info(f"Skipping {model_type} - no retraining needed")
                    continue
            # Retrain model
            result = retrain_model(
                model_type=model_type,
                data=data,
                router=router,
                days=args.days
            )
            results.append(result)
        # Log results
        logger.info(f"Retraining completed: {results}")
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        raise

def trigger_retraining_if_needed(weights: Dict[str, float], threshold: float = 0.1) -> bool:
    """Trigger model retraining if needed."""
    try:
        # Simple logic: trigger retraining if any weight is below threshold
        for model, weight in weights.items():
            if weight < threshold:
                logger.info(f"Triggering retraining for {model} (weight: {weight})")
                return True
        return False
    except Exception as e:
        logger.error(f"Error in retraining trigger: {e}")
        return False

if __name__ == "__main__":
    main()

"""
Model retraining utilities.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def trigger_retraining_if_needed(weights: Dict[str, float], threshold: float = 0.1) -> bool:
    """
    Trigger model retraining if needed.
    
    Args:
        weights: Model weights dictionary
        threshold: Threshold for triggering retraining
        
    Returns:
        True if retraining was triggered, False otherwise
    """
    try:
        # Simple logic: trigger retraining if any weight is below threshold
        for model, weight in weights.items():
            if weight < threshold:
                logger.info(f"Triggering retraining for {model} (weight: {weight})")
                return True
                
        return False
        
    except Exception as e:
        logger.error(f"Error in retraining trigger: {e}")
        return False 