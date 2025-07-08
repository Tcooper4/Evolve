"""
Performance weights management for trading models.
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def export_weights_to_file(ticker: str, strategy: str = "balanced") -> Dict[str, float]:
    """
    Export model weights to file.
    
    Args:
        ticker: Trading symbol
        strategy: Strategy type
        
    Returns:
        Dictionary of model weights
    """
    try:
        # Default weights for demonstration
        default_weights = {
            "lstm": 0.3,
            "xgboost": 0.25,
            "prophet": 0.2,
            "ensemble": 0.15,
            "tcn": 0.1
        }
        
        try:
            os.makedirs("memory", exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create memory directory: {e}")
        
        # Save weights to file
        weights_file = f"memory/{ticker}_weights.json"
        weights_data = {
            "ticker": ticker,
            "strategy": strategy,
            "weights": default_weights,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f, indent=2)
            
        return default_weights
        
    except Exception as e:
        logger.error(f"Error exporting weights: {e}")
        return {"lstm": 1.0}  # Fallback to single model

def get_latest_weights(ticker: str = "AAPL") -> Dict[str, float]:
    """
    Get the latest performance weights for a ticker.
    
    Args:
        ticker: Trading symbol
        
    Returns:
        Dictionary of model weights
    """
    try:
        weights_file = f"memory/{ticker}_weights.json"
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
                return {'success': True, 'result': weights_data.get("weights", {}), 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        else:
            # Return default weights if file doesn't exist
            return {
                "lstm": 0.3,
                "xgboost": 0.25,
                "prophet": 0.2,
                "ensemble": 0.15,
                "tcn": 0.1
            }
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
        return {"lstm": 1.0}  # Fallback to single model 