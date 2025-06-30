"""
Monitoring Service

Implements system monitoring and metrics collection functionality.
Adapted from legacy automation/services/automation_monitoring.py.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import psutil
import sqlite3
from pydantic import BaseModel
import aiohttp
import ssl
import socket
from dataclasses import dataclass
from enum import Enum

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
    labels: Dict[str, str] = None
    buckets: List[float] = None
    quantiles: List[float] = None

class MonitoringService:
    """Manages system monitoring and metrics collection."""
    
    def __init__(self, config_path: str = "config/monitoring.json"):
        """Initialize monitoring service."""
        self.config_path = config_path
        self.setup_logging()
        self.load_config()
        self.setup_database()
        self.initialize_metrics()
        self.metrics_queue = asyncio.Queue()
        self.running = False
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/monitoring")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "monitoring_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
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
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    value REAL NOT NULL,
                    labels TEXT,
                    timestamp DATETIME NOT NULL
                )
            ''')
            
            # Create index on timestamp
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON metrics(timestamp)
            ''')
            
            self.conn.commit()
            self.logger.info("Database setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    def initialize_metrics(self) -> None:
        """Initialize system metrics."""
        self.metrics = {
            # CPU metrics
            "cpu_usage": Metric(
                name="cpu_usage",
                type=MetricType.GAUGE,
                description="CPU usage percentage"
            ),
            "cpu_count": Metric(
                name="cpu_count",
                type=MetricType.GAUGE,
                description="Number of CPU cores"
            ),
            
            # Memory metrics
            "memory_usage": Metric(
                name="memory_usage",
                type=MetricType.GAUGE,
                description="Memory usage percentage"
            ),
            "memory_total": Metric(
                name="memory_total",
                type=MetricType.GAUGE,
                description="Total memory in bytes"
            ),
            "memory_available": Metric(
                name="memory_available",
                type=MetricType.GAUGE,
                description="Available memory in bytes"
            ),
            
            # Disk metrics
            "disk_usage": Metric(
                name="disk_usage",
                type=MetricType.GAUGE,
                description="Disk usage percentage"
            ),
            "disk_total": Metric(
                name="disk_total",
                type=MetricType.GAUGE,
                description="Total disk space in bytes"
            ),
            "disk_free": Metric(
                name="disk_free",
                type=MetricType.GAUGE,
                description="Free disk space in bytes"
            ),
            
            # Network metrics
            "network_bytes_sent": Metric(
                name="network_bytes_sent",
                type=MetricType.COUNTER,
                description="Total bytes sent"
            ),
            "network_bytes_recv": Metric(
                name="network_bytes_recv",
                type=MetricType.COUNTER,
                description="Total bytes received"
            ),
            
            # Process metrics
            "process_count": Metric(
                name="process_count",
                type=MetricType.GAUGE,
                description="Number of processes"
            ),
            "process_threads": Metric(
                name="process_threads",
                type=MetricType.GAUGE,
                description="Number of threads"
            )
        }
    
        return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    async def collect_system_metrics(self) -> None:
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            await self.record_metric("cpu_usage", cpu_percent)
            await self.record_metric("cpu_count", cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_metric("memory_usage", memory.percent)
            await self.record_metric("memory_total", memory.total)
            await self.record_metric("memory_available", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self.record_metric("disk_usage", disk.percent)
            await self.record_metric("disk_total", disk.total)
            await self.record_metric("disk_free", disk.free)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            await self.record_metric("network_bytes_sent", net_io.bytes_sent)
            await self.record_metric("network_bytes_recv", net_io.bytes_recv)
            
            # Process metrics
            process_count = len(psutil.pids())
            thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']))
            await self.record_metric("process_count", process_count)
            await self.record_metric("process_threads", thread_count)
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            raise
    
    async def record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a metric value."""
        try:
            if name not in self.metrics:
                raise ValueError(f"Unknown metric: {name}")
            
            metric = self.metrics[name]
            
            # Add to queue
            await self.metrics_queue.put({
                "name": name,
                "type": metric.type.value,
                "value": value,
                "labels": json.dumps(labels) if labels else None,
                "timestamp": datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error recording metric: {str(e)}")
            raise
    
    async def process_metrics_queue(self) -> None:
        """Process metrics queue."""
        while self.running:
            try:
                metric = await self.metrics_queue.get()
                
                # Save to database
                self.cursor.execute('''
                    INSERT INTO metrics (
                        name, type, value, labels, timestamp
                    )
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    metric["name"],
                    metric["type"],
                    metric["value"],
                    metric["labels"],
                    metric["timestamp"]
                ))
                
                self.conn.commit()
                self.metrics_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing metric: {str(e)}")
    
    async def query_metrics(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        labels: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """Query metrics from database."""
        try:
            query = '''
                SELECT * FROM metrics
                WHERE name = ?
                AND timestamp BETWEEN ? AND ?
            '''
            params = [name, start_time, end_time]
            
            if labels:
                query += ' AND labels = ?'
                params.append(json.dumps(labels))
            
            query += ' ORDER BY timestamp'
            
            self.cursor.execute(query, params)
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    'id': row[0],
                    'name': row[1],
                    'type': row[2],
                    'value': row[3],
                    'labels': json.loads(row[4]) if row[4] else None,
                    'timestamp': row[5]
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Error querying metrics: {str(e)}")
            raise
    
    async def cleanup_metrics(self, retention_days: int) -> None:
        """Clean up old metrics."""
        try:
            cutoff = datetime.now() - timedelta(days=retention_days)
            
            self.cursor.execute('''
                DELETE FROM metrics
                WHERE timestamp < ?
            ''', (cutoff,))
            
            self.conn.commit()
            self.logger.info(f"Cleaned up metrics older than {retention_days} days")
        except Exception as e:
            self.logger.error(f"Error cleaning up metrics: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start monitoring service."""
        try:
            self.running = True
            
            # Start queue processor
            processor = asyncio.create_task(self.process_metrics_queue())
            
            # Start metrics collection
            while self.running:
                await self.collect_system_metrics()
                await asyncio.sleep(self.config['collection_interval'])
            
            # Cleanup
            processor.cancel()
            try:
                await processor
            except asyncio.CancelledError:
                pass
        except Exception as e:
            self.logger.error(f"Error in monitoring service: {str(e)}")
            raise
        finally:
            self.running = False
    
    def stop(self) -> None:
        """Stop monitoring service."""
        self.running = False

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitoring service')
    parser.add_argument('--config', default="config/monitoring.json", help='Path to config file')
    args = parser.parse_args()
    
    try:
        service = MonitoringService(args.config)
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logging.info("Monitoring service interrupted")
    except Exception as e:
        logging.error(f"Error in monitoring service: {str(e)}")
        raise

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
if __name__ == '__main__':
    main() 