"""
Metrics Service

This module implements metrics collection and aggregation functionality.

Note: This module was adapted from the legacy automation/services/automation_metrics.py file.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import yaml
import time
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import psutil
import numpy as np
from collections import deque

class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    """Metric configuration."""
    name: str
    type: MetricType
    description: str
    labels: List[str]
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None

class MetricsService:
    """Manages metrics collection and aggregation."""
    
    def __init__(self, config_path: str):
        """Initialize metrics service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_database()
        self.metrics: Dict[str, Metric] = {}
        self.metrics_queue = Queue()
        self.running = False
        self.initialize_metrics()
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/metrics")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "metrics_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    self.config = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def setup_database(self) -> None:
        """Set up metrics database."""
        try:
            db_path = Path(self.config['database']['path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(str(db_path))
            self.cursor = self.conn.cursor()
            
            # Create metrics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    labels TEXT,
                    metric_type TEXT NOT NULL
                )
            ''')
            
            # Create index on timestamp and metric_name
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            ''')
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_name 
                ON metrics(metric_name)
            ''')
            
            self.conn.commit()
            self.logger.info("Database setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def initialize_metrics(self) -> None:
        """Initialize metrics from configuration."""
        try:
            for metric_config in self.config['metrics']:
                metric = Metric(
                    name=metric_config['name'],
                    type=MetricType(metric_config['type']),
                    description=metric_config['description'],
                    labels=metric_config.get('labels', []),
                    buckets=metric_config.get('buckets'),
                    quantiles=metric_config.get('quantiles')
                )
                self.metrics[metric.name] = metric
        except Exception as e:
            self.logger.error(f"Error initializing metrics: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        try:
            if name not in self.metrics:
                raise ValueError(f"Unknown metric: {name}")
            
            metric = self.metrics[name]
            if metric.type != MetricType.COUNTER:
                raise ValueError(f"Metric {name} is not a counter")
            
            self.metrics_queue.put({
                'name': name,
                'value': value,
                'labels': labels,
                'type': MetricType.COUNTER.value,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error recording counter: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        try:
            if name not in self.metrics:
                raise ValueError(f"Unknown metric: {name}")
            
            metric = self.metrics[name]
            if metric.type != MetricType.GAUGE:
                raise ValueError(f"Metric {name} is not a gauge")
            
            self.metrics_queue.put({
                'name': name,
                'value': value,
                'labels': labels,
                'type': MetricType.GAUGE.value,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error recording gauge: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        try:
            if name not in self.metrics:
                raise ValueError(f"Unknown metric: {name}")
            
            metric = self.metrics[name]
            if metric.type != MetricType.HISTOGRAM:
                raise ValueError(f"Metric {name} is not a histogram")
            
            if not metric.buckets:
                raise ValueError(f"Histogram {name} has no buckets defined")
            
            # Record value in each bucket
            for bucket in metric.buckets:
                bucket_value = 1.0 if value <= bucket else 0.0
                self.metrics_queue.put({
                    'name': f"{name}_bucket",
                    'value': bucket_value,
                    'labels': {**(labels or {}), 'le': str(bucket)},
                    'type': MetricType.HISTOGRAM.value,
                    'timestamp': datetime.now()
                })
            
            # Record sum
            self.metrics_queue.put({
                'name': f"{name}_sum",
                'value': value,
                'labels': labels,
                'type': MetricType.HISTOGRAM.value,
                'timestamp': datetime.now()
            })
            
            # Record count
            self.metrics_queue.put({
                'name': f"{name}_count",
                'value': 1.0,
                'labels': labels,
                'type': MetricType.HISTOGRAM.value,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error recording histogram: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def record_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a summary metric."""
        try:
            if name not in self.metrics:
                raise ValueError(f"Unknown metric: {name}")
            
            metric = self.metrics[name]
            if metric.type != MetricType.SUMMARY:
                raise ValueError(f"Metric {name} is not a summary")
            
            if not metric.quantiles:
                raise ValueError(f"Summary {name} has no quantiles defined")
            
            # Record value in each quantile
            for quantile in metric.quantiles:
                self.metrics_queue.put({
                    'name': f"{name}_quantile",
                    'value': value,
                    'labels': {**(labels or {}), 'quantile': str(quantile)},
                    'type': MetricType.SUMMARY.value,
                    'timestamp': datetime.now()
                })
            
            # Record sum
            self.metrics_queue.put({
                'name': f"{name}_sum",
                'value': value,
                'labels': labels,
                'type': MetricType.SUMMARY.value,
                'timestamp': datetime.now()
            })
            
            # Record count
            self.metrics_queue.put({
                'name': f"{name}_count",
                'value': 1.0,
                'labels': labels,
                'type': MetricType.SUMMARY.value,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error recording summary: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def process_metrics_queue(self) -> None:
        """Process metrics from the queue."""
        try:
            while self.running:
                try:
                    metric = self.metrics_queue.get(timeout=1)
                    self.cursor.execute('''
                        INSERT INTO metrics (timestamp, metric_name, metric_value, labels, metric_type)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        metric['timestamp'],
                        metric['name'],
                        metric['value'],
                        json.dumps(metric['labels']) if metric['labels'] else None,
                        metric['type']
                    ))
                    self.conn.commit()
                except Queue.Empty:
                    continue
        except Exception as e:
            self.logger.error(f"Error processing metrics queue: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def get_metric(self, name: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get metric values for a specific time range."""
        try:
            self.cursor.execute('''
                SELECT timestamp, metric_value, labels
                FROM metrics
                WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (name, start_time, end_time))
            
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'value': row[1],
                    'labels': json.loads(row[2]) if row[2] else None
                })
            
            return {'success': True, 'result': results, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error getting metric: {str(e)}")
            raise
    
    def get_metric_names(self) -> List[str]:
        """Get list of all metric names."""
        try:
            self.cursor.execute('''
                SELECT DISTINCT metric_name
                FROM metrics
                ORDER BY metric_name
            ''')
            
            return {'success': True, 'result': [row[0] for row in self.cursor.fetchall()], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error getting metric names: {str(e)}")
            raise
    
    def cleanup_old_metrics(self, days: int) -> None:
        """Clean up metrics older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            self.cursor.execute('''
                DELETE FROM metrics
                WHERE timestamp < ?
            ''', (cutoff_date,))
            
            self.conn.commit()
            self.logger.info(f"Cleaned up metrics older than {days} days")
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    async def start(self) -> None:
        """Start metrics collection."""
        try:
            self.running = True
            queue_thread = threading.Thread(target=self.process_metrics_queue)
            queue_thread.start()
            
            while self.running:
                # Collect system metrics
                self.record_gauge('system_cpu_usage', psutil.cpu_percent())
                self.record_gauge('system_memory_usage', psutil.virtual_memory().percent)
                self.record_gauge('system_disk_usage', psutil.disk_usage('/').percent)
                
                # Collect process metrics
                for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        self.record_gauge(
                            'process_cpu_usage',
                            process.info['cpu_percent'],
                            {'pid': str(process.info['pid']), 'name': process.info['name']}
                        )
                        self.record_gauge(
                            'process_memory_usage',
                            process.info['memory_percent'],
                            {'pid': str(process.info['pid']), 'name': process.info['name']}
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                await asyncio.sleep(self.config['collection']['interval'])
        except Exception as e:
            self.logger.error(f"Error in metrics collection: {str(e)}")
            raise
        finally:
            self.running = False
            queue_thread.join()
    
    def stop(self) -> None:
        """Stop metrics collection."""
        self.running = False

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect system metrics')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--cleanup', type=int, help='Clean up metrics older than specified days')
    args = parser.parse_args()
    
    try:
        service = MetricsService(args.config)
        
        if args.cleanup:
            service.cleanup_old_metrics(args.cleanup)
        else:
            asyncio.run(service.start())
    except KeyboardInterrupt:
        logging.info("Metrics collection interrupted")
    except Exception as e:
        logging.error(f"Error collecting metrics: {str(e)}")
        raise

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == '__main__':
    main() 