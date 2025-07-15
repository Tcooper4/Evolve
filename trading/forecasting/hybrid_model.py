import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

class HybridModel:
    """
    Hybrid ensemble model that tracks model performance, auto-updates weights, and persists state.
    """
    def __init__(self, model_dict: Dict[str, Any], weight_file: str = "hybrid_weights.json", perf_file: str = "hybrid_performance.json"):
        """
        Args:
            model_dict: Dictionary of model_name: model_instance
            weight_file: Path to save/load ensemble weights
            perf_file: Path to save/load model performance
        """
        self.models = model_dict
        self.weight_file = weight_file
        self.perf_file = perf_file
        self.weights = {name: 1.0 / len(model_dict) for name in model_dict}
        self.performance = {name: [] for name in model_dict}  # List of recent MSEs
        self.load_state()

    def fit(self, data: pd.DataFrame, window: int = 50):
        """Fit all models and update performance tracking."""
        for name, model in self.models.items():
            try:
                model.fit(data)
                preds = model.predict(data)
                actual = data["close"].values[-len(preds):]
                mse = float(np.mean((actual - preds) ** 2))
                self.performance[name].append({
                    "timestamp": datetime.now().isoformat(),
                    "mse": mse
                })
                # Keep only trailing window
                self.performance[name] = self.performance[name][-window:]
            except Exception as e:
                logger.warning(f"Model {name} failed to fit or predict: {e}")
                self.performance[name].append({
                    "timestamp": datetime.now().isoformat(),
                    "mse": float('inf')
                })
        self.save_state()
        self.update_weights()

    def update_weights(self):
        """Auto-update ensemble weights based on trailing MSE performance."""
        avg_mse = {name: np.mean([entry["mse"] for entry in perf if np.isfinite(entry["mse"])])
                   for name, perf in self.performance.items()}
        # Inverse MSE weighting
        inv_mse = {name: 1.0 / mse if mse > 0 else 0.0 for name, mse in avg_mse.items()}
        total = sum(inv_mse.values())
        if total > 0:
            self.weights = {name: val / total for name, val in inv_mse.items()}
        else:
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
        self.save_state()

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble prediction with fallback for None/mismatched forecasts."""
        valid_preds = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(data)
                
                # Validate prediction
                if pred is None:
                    logger.warning(f"Model {name} returned None prediction, skipping")
                    continue
                    
                # Convert to numpy array if needed
                if not isinstance(pred, np.ndarray):
                    pred = np.array(pred)
                
                # Check for valid values
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    logger.warning(f"Model {name} returned invalid values (NaN/Inf), skipping")
                    continue
                
                # Check for reasonable prediction range
                if np.any(pred < -1e6) or np.any(pred > 1e6):
                    logger.warning(f"Model {name} returned extreme values, skipping")
                    continue
                
                valid_preds.append((name, pred))
                
            except Exception as e:
                logger.warning(f"Model {name} failed to predict: {e}")
                continue
        
        # Handle case where no valid predictions
        if not valid_preds:
            logger.error("No valid predictions from any model, returning fallback")
            # Return simple moving average as fallback
            if len(data) > 0 and "close" in data.columns:
                fallback_pred = np.full(len(data), data["close"].mean())
                return fallback_pred
            else:
                return np.array([])
        
        # Align predictions to same length
        min_len = min(len(p) for _, p in valid_preds)
        if min_len == 0:
            logger.error("All predictions have zero length")
            return np.array([])
        
        # Truncate all predictions to minimum length
        aligned_preds = []
        for name, pred in valid_preds:
            if len(pred) > min_len:
                # Take the last min_len values
                aligned_pred = pred[-min_len:]
            else:
                aligned_pred = pred
            aligned_preds.append((name, aligned_pred))
        
        # Calculate weighted ensemble
        weighted = np.zeros(min_len)
        total_weight = 0.0
        
        for name, pred in aligned_preds:
            weight = self.weights.get(name, 0.0)
            if weight > 0:
                weighted += weight * pred
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted = weighted / total_weight
        else:
            # Equal weighting if no valid weights
            weighted = np.mean([pred for _, pred in aligned_preds], axis=0)
        
        return weighted

    def save_state(self):
        """Save weights and performance to disk (JSON and joblib)."""
        try:
            with open(self.weight_file, "w") as f:
                json.dump(self.weights, f, indent=2)
            with open(self.perf_file, "w") as f:
                json.dump(self.performance, f, indent=2)
            joblib.dump(self.weights, self.weight_file + ".joblib")
            joblib.dump(self.performance, self.perf_file + ".joblib")
        except Exception as e:
            logger.error(f"Failed to save hybrid model state: {e}")

    def load_state(self):
        """Load weights and performance from disk if available."""
        try:
            if os.path.exists(self.weight_file):
                with open(self.weight_file, "r") as f:
                    self.weights = json.load(f)
            if os.path.exists(self.perf_file):
                with open(self.perf_file, "r") as f:
                    self.performance = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load hybrid model state: {e}")
        # Try joblib as fallback
        try:
            if os.path.exists(self.weight_file + ".joblib"):
                self.weights = joblib.load(self.weight_file + ".joblib")
            if os.path.exists(self.perf_file + ".joblib"):
                self.performance = joblib.load(self.perf_file + ".joblib")
        except Exception as e:
            logger.warning(f"Failed to load hybrid model state from joblib: {e}") 