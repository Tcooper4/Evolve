"""
Performance Checker Agent

This agent is responsible for:
1. Monitoring and logging model performance metrics over time
2. Detecting performance drift or degradation
3. Triggering model retraining when necessary
4. Managing performance logs and thresholds

The agent maintains a rolling window of performance metrics and uses
statistical methods to detect significant changes in model behavior.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from scipy import stats
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for model performance metrics"""
    model_name: str
    timestamp: str
    mse: float
    sharpe: float
    drawdown: float
    prediction_latency: float
    data_freshness: float
    confidence_score: float

class PerformanceChecker:
    def __init__(self, config_path: str = "config/performance_checker_config.json"):
        """
        Initialize the PerformanceChecker agent.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.performance_log = self.load_log()
        self.log_dir = Path(self.config.get("log_dir", "logs/performance"))
        self.thresholds = self.config.get("thresholds", {
            "mse_increase": 0.2,  # 20% increase in MSE
            "sharpe_decrease": 0.15,  # 15% decrease in Sharpe
            "drawdown_increase": 0.1,  # 10% increase in drawdown
            "confidence_threshold": 0.7,  # Minimum confidence score
            "drift_threshold": 0.05  # Statistical significance level
        })
        self.window_size = self.config.get("window_size", 30)  # Days
        self.retrain_cooldown = self.config.get("retrain_cooldown", 7)  # Days
        
        # Create necessary directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file watcher for performance logs
        self.observer = Observer()
        self.observer.schedule(
            PerformanceLogHandler(self),
            str(self.log_dir),
            recursive=False
        )
        self.observer.start()
        
        logger.info("Initialized PerformanceChecker agent")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}

    def load_log(self) -> Dict[str, List[PerformanceMetrics]]:
        """
        Load performance metrics from log files.
        
        Returns:
            Dictionary mapping model names to lists of their performance metrics
        """
        try:
            log_data = {}
            for log_file in self.log_dir.glob("*.json"):
                model_name = log_file.stem
                with open(log_file, 'r') as f:
                    raw_metrics = json.load(f)
                    log_data[model_name] = [
                        PerformanceMetrics(**metrics)
                        for metrics in raw_metrics
                    ]
            return log_data
        except Exception as e:
            logger.error(f"Error loading performance logs: {e}")
            return {}

    def detect_drift(self, model_name: str, new_metrics: PerformanceMetrics) -> Tuple[bool, Dict]:
        """
        Detect performance drift by comparing new metrics to historical data.
        
        Args:
            model_name: Name of the model
            new_metrics: Latest performance metrics
            
        Returns:
            Tuple of (drift_detected, drift_details)
        """
        try:
            if model_name not in self.performance_log:
                return False, {"reason": "No historical data available"}
            
            # Get recent metrics
            recent_metrics = self.performance_log[model_name][-self.window_size:]
            if not recent_metrics:
                return False, {"reason": "Insufficient historical data"}
            
            # Calculate rolling statistics
            recent_mse = [m.mse for m in recent_metrics]
            recent_sharpe = [m.sharpe for m in recent_metrics]
            recent_drawdown = [m.drawdown for m in recent_metrics]
            
            # Check for significant changes
            drift_details = {
                "mse_change": self._calculate_change(new_metrics.mse, recent_mse),
                "sharpe_change": self._calculate_change(new_metrics.sharpe, recent_sharpe),
                "drawdown_change": self._calculate_change(new_metrics.drawdown, recent_drawdown),
                "confidence_low": new_metrics.confidence_score < self.thresholds["confidence_threshold"]
            }
            
            # Perform statistical tests
            drift_detected = (
                drift_details["mse_change"] > self.thresholds["mse_increase"] or
                drift_details["sharpe_change"] < -self.thresholds["sharpe_decrease"] or
                drift_details["drawdown_change"] > self.thresholds["drawdown_increase"] or
                drift_details["confidence_low"]
            )
            
            if drift_detected:
                logger.warning(f"Drift detected for model {model_name}: {drift_details}")
            
            return drift_detected, drift_details
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return False, {"error": str(e)}

    def _calculate_change(self, new_value: float, historical_values: List[float]) -> float:
        """Calculate relative change compared to historical values"""
        try:
            if not historical_values:
                return 0.0
            historical_mean = np.mean(historical_values)
            if historical_mean == 0:
                return 0.0
            return (new_value - historical_mean) / abs(historical_mean)
        except Exception as e:
            logger.error(f"Error calculating change: {e}")
            return 0.0

    async def trigger_retrain(self, model_name: str, drift_details: Dict) -> bool:
        """
        Trigger model retraining if conditions are met.
        
        Args:
            model_name: Name of the model to retrain
            drift_details: Details about the detected drift
            
        Returns:
            True if retraining was triggered, False otherwise
        """
        try:
            # Check retrain cooldown
            last_retrain = self._get_last_retrain_time(model_name)
            if last_retrain and (datetime.now() - last_retrain).days < self.retrain_cooldown:
                logger.info(f"Retraining skipped for {model_name} due to cooldown period")
                return False
            
            # Prepare retraining signal
            signal = {
                "model_name": model_name,
                "drift_details": drift_details,
                "timestamp": datetime.now().isoformat(),
                "priority": self._calculate_retrain_priority(drift_details)
            }
            
            # Send signal to ModelBuilder
            signal_path = self.log_dir / f"{model_name}_retrain_signal.json"
            with open(signal_path, 'w') as f:
                json.dump(signal, f, indent=2)
            
            logger.info(f"Retraining triggered for model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering retrain: {e}")
            return False

    def _get_last_retrain_time(self, model_name: str) -> Optional[datetime]:
        """Get the timestamp of the last retraining for a model"""
        try:
            signal_files = list(self.log_dir.glob(f"{model_name}_retrain_signal.json"))
            if not signal_files:
                return None
            latest_file = max(signal_files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                signal = json.load(f)
                return datetime.fromisoformat(signal["timestamp"])
        except Exception as e:
            logger.error(f"Error getting last retrain time: {e}")
            return None

    def _calculate_retrain_priority(self, drift_details: Dict) -> int:
        """Calculate retraining priority based on drift severity"""
        try:
            priority = 0
            if drift_details["mse_change"] > self.thresholds["mse_increase"] * 2:
                priority += 3
            elif drift_details["mse_change"] > self.thresholds["mse_increase"]:
                priority += 2
                
            if drift_details["sharpe_change"] < -self.thresholds["sharpe_decrease"] * 2:
                priority += 3
            elif drift_details["sharpe_change"] < -self.thresholds["sharpe_decrease"]:
                priority += 2
                
            if drift_details["drawdown_change"] > self.thresholds["drawdown_increase"] * 2:
                priority += 3
            elif drift_details["drawdown_change"] > self.thresholds["drawdown_increase"]:
                priority += 2
                
            if drift_details["confidence_low"]:
                priority += 1
                
            return min(priority, 10)  # Cap at 10
            
        except Exception as e:
            logger.error(f"Error calculating retrain priority: {e}")
            return 5  # Default priority

    def update_log(self, model_name: str, new_metrics: PerformanceMetrics) -> None:
        """
        Update performance log with new metrics.
        
        Args:
            model_name: Name of the model
            new_metrics: New performance metrics
        """
        try:
            # Update in-memory log
            if model_name not in self.performance_log:
                self.performance_log[model_name] = []
            self.performance_log[model_name].append(new_metrics)
            
            # Keep only recent metrics
            self.performance_log[model_name] = self.performance_log[model_name][-self.window_size:]
            
            # Save to disk
            log_path = self.log_dir / f"{model_name}.json"
            with open(log_path, 'w') as f:
                json.dump(
                    [vars(m) for m in self.performance_log[model_name]],
                    f,
                    indent=2
                )
            
            logger.info(f"Updated performance log for model {model_name}")
            
        except Exception as e:
            logger.error(f"Error updating performance log: {e}")

    async def run(self) -> None:
        """Main execution loop for the PerformanceChecker agent"""
        try:
            while True:
                # Check for new performance metrics
                for model_name, metrics in self.performance_log.items():
                    if not metrics:
                        continue
                        
                    latest_metrics = metrics[-1]
                    
                    # Detect drift
                    drift_detected, drift_details = self.detect_drift(
                        model_name,
                        latest_metrics
                    )
                    
                    if drift_detected:
                        # Trigger retraining if necessary
                        await self.trigger_retrain(model_name, drift_details)
                
                # Wait before next check
                await asyncio.sleep(self.config.get("check_interval", 3600))  # Default: 1 hour
                
        except KeyboardInterrupt:
            logger.info("PerformanceChecker agent stopped by user")
        except Exception as e:
            logger.error(f"Error in PerformanceChecker agent: {e}")
        finally:
            self.observer.stop()
            self.observer.join()

class PerformanceLogHandler(FileSystemEventHandler):
    """Handler for monitoring performance log files"""
    
    def __init__(self, checker: PerformanceChecker):
        self.checker = checker
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.json'):
            logger.info(f"Performance log updated: {event.src_path}")
            # Reload the log file
            self.checker.performance_log = self.checker.load_log()

if __name__ == "__main__":
    checker = PerformanceChecker()
    asyncio.run(checker.run()) 