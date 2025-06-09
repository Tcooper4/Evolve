import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import redis
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary
import json

from ..config.config import load_config

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and exposes system metrics."""
    
    def __init__(self, config_path: str = "automation/config/config.json"):
        self.config = load_config(config_path)
        self.setup_logging()
        self.setup_metrics()
        self.redis_client = None
        self.running = False
        
    def setup_logging(self):
        """Setup logging for the metrics collector."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('automation/logs/metrics_collector.log'),
                logging.StreamHandler()
            ]
        )
        
    def setup_metrics(self):
        """Setup Prometheus metrics."""
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage', 'Disk usage percentage')
        self.network_io = Gauge('system_network_io', 'Network I/O bytes')
        
        # Task metrics
        self.tasks_total = Counter('tasks_total', 'Total number of tasks processed')
        self.tasks_failed = Counter('tasks_failed', 'Number of failed tasks')
        self.task_duration = Histogram('task_duration_seconds', 'Task processing duration')
        
        # Agent metrics
        self.active_agents = Gauge('active_agents', 'Number of active agents')
        self.agent_heartbeats = Gauge('agent_heartbeats', 'Agent heartbeat status')
        
 #       # Redis metrics
 #       self.redis_connected = Gauge('redis_connected', 'Redis connection status')
 #       self.redis_memory_usage = Gauge('redis_memory_usage', 'Redis memory usage')
        
        # Model metrics
        self.model_predictions = Counter('model_predictions_total', 'Total number of model predictions')
        self.model_accuracy = Gauge('model_accuracy', 'Model prediction accuracy')
        self.model_latency = Histogram('model_latency_seconds', 'Model prediction latency')
        
        # New metrics
        self.task_execution_time = Histogram(
            'task_execution_time_seconds',
            'Task execution time in seconds',
            ['task_type']
        )
        self.task_success = Counter(
            'task_success_total',
            'Total successful tasks',
            ['task_type']
        )
        self.task_failure = Counter(
            'task_failure_total',
            'Total failed tasks',
            ['task_type']
        )
        self.api_request_duration = Summary(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['endpoint']
        )
        
    async def initialize(self) -> None:
        """Initialize Redis connection and start metrics server."""
        try:
            # Setup Redis
            redis_config = self.config["redis"]
            self.redis_client = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                decode_responses=True
            )
            
            # Start Prometheus metrics server
            metrics_config = self.config["monitoring"]
            start_http_server(metrics_config["metrics_port"])
            
            logger.info("Metrics collector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {str(e)}")
            raise
            
    async def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage.set(disk.percent)
            
            # Network usage
            net_io = psutil.net_io_counters()
            self.network_io.set(net_io.bytes_sent + net_io.bytes_recv)
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "network_io": net_io.bytes_sent + net_io.bytes_recv
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            raise
            
    async def collect_task_metrics(self) -> Dict[str, Any]:
        """Collect task-related metrics."""
        try:
            # Get task statistics from Redis
            task_stats = await self.redis_client.hgetall("task_stats")
            
            # Update metrics
            self.tasks_total.inc(int(task_stats.get("total", 0)))
            self.tasks_failed.inc(int(task_stats.get("failed", 0)))
            
            return {
                "total_tasks": int(task_stats.get("total", 0)),
                "failed_tasks": int(task_stats.get("failed", 0)),
                "success_rate": float(task_stats.get("success_rate", 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to collect task metrics: {str(e)}")
            raise
            
    async def collect_agent_metrics(self) -> Dict[str, Any]:
        """Collect agent-related metrics."""
        try:
            # Get agent statistics from Redis
            agents = await self.redis_client.hgetall("agents")
            active_count = sum(1 for agent in agents.values() if agent.get("status") == "running")
            
            # Update metrics
            self.active_agents.set(active_count)
            
            # Check agent heartbeats
            for agent_id, agent_info in agents.items():
                last_heartbeat = datetime.fromisoformat(agent_info.get("last_heartbeat", datetime.min.isoformat()))
                is_alive = (datetime.now() - last_heartbeat).total_seconds() < 300  # 5 minutes
                self.agent_heartbeats.labels(agent_id=agent_id).set(1 if is_alive else 0)
                
            return {
                "active_agents": active_count,
                "total_agents": len(agents)
            }
            
        except Exception as e:
            logger.error(f"Failed to collect agent metrics: {str(e)}")
            raise
            
    async def collect_model_metrics(self) -> Dict[str, Any]:
        """Collect model-related metrics."""
        try:
            # Get model statistics from Redis
            model_stats = await self.redis_client.hgetall("model_stats")
            
            # Update metrics
            self.model_predictions.inc(int(model_stats.get("predictions", 0)))
            self.model_accuracy.set(float(model_stats.get("accuracy", 0)))
            
            return {
                "predictions": int(model_stats.get("predictions", 0)),
                "accuracy": float(model_stats.get("accuracy", 0)),
                "latency": float(model_stats.get("latency", 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to collect model metrics: {str(e)}")
            raise
            
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all system metrics."""
        try:
            metrics = {
                "system": await self.collect_system_metrics(),
                "tasks": await self.collect_task_metrics(),
                "agents": await self.collect_agent_metrics(),
                "models": await self.collect_model_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store metrics in Redis for historical tracking
            await self.redis_client.hset(
                "metrics_history",
                datetime.now().strftime("%Y%m%d_%H%M%S"),
                json.dumps(metrics)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect all metrics: {str(e)}")
            raise
            
    async def start(self) -> None:
        """Start the metrics collector."""
        try:
            self.running = True
            await self.initialize()
            logger.info("Metrics collector started")
            
            while self.running:
                await self.collect_all_metrics()
                await asyncio.sleep(self.config["monitoring"]["collection_interval"])
                
        except Exception as e:
            logger.error(f"Metrics collector failed: {str(e)}")
            raise
        finally:
            self.running = False
            
    async def stop(self) -> None:
        """Stop the metrics collector."""
        self.running = False
        logger.info("Metrics collector stopped")

    async def start_collection(self):
        """Start metrics collection."""
        try:
            while True:
                await self.collect_metrics()
                await asyncio.sleep(self.config["monitoring"]["collection_interval"])
                
        except Exception as e:
            logger.error(f"Error in metrics collection: {str(e)}")
            raise
            
    async def collect_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.disk_usage.set(disk.percent)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.network_io.set(net_io.bytes_sent + net_io.bytes_recv)
            
            # Store metrics in Redis
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "network_io": net_io.bytes_sent + net_io.bytes_recv
            }
            
            await self.redis_client.lpush(
                "system_metrics",
                json.dumps(metrics)
            )
            
            # Trim metrics to last 24 hours
            await self.redis_client.ltrim(
                "system_metrics",
                0,
                int(24 * 3600 / self.config["monitoring"]["collection_interval"])
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            
    async def record_task_metrics(
        self,
        task_type: str,
        execution_time: float,
        success: bool
    ):
        """Record task execution metrics."""
        try:
            # Record execution time
            self.task_execution_time.labels(task_type).observe(execution_time)
            
            # Record success/failure
            if success:
                self.task_success.labels(task_type).inc()
            else:
                self.task_failure.labels(task_type).inc()
                
            # Store in Redis
            task_metrics = {
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
                "execution_time": execution_time,
                "success": success
            }
            
            await self.redis_client.lpush(
                "task_metrics",
                json.dumps(task_metrics)
            )
            
            # Trim metrics to last 24 hours
            await self.redis_client.ltrim(
                "task_metrics",
                0,
                999  # Keep last 1000 task metrics
            )
            
        except Exception as e:
            logger.error(f"Error recording task metrics: {str(e)}")
            
    async def record_api_metrics(
        self,
        endpoint: str,
        duration: float
    ):
        """Record API request metrics."""
        try:
            # Record duration
            self.api_request_duration.labels(endpoint).observe(duration)
            
            # Store in Redis
            api_metrics = {
                "timestamp": datetime.now().isoformat(),
                "endpoint": endpoint,
                "duration": duration
            }
            
            await self.redis_client.lpush(
                "api_metrics",
                json.dumps(api_metrics)
            )
            
            # Trim metrics to last 24 hours
            await self.redis_client.ltrim(
                "api_metrics",
                0,
                999  # Keep last 1000 API metrics
            )
            
        except Exception as e:
            logger.error(f"Error recording API metrics: {str(e)}")
            
    async def get_metrics(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get metrics with optional time filtering."""
        try:
            # Get metrics from Redis
            metrics = await self.redis_client.lrange(
                f"{metric_type}_metrics",
                0,
                limit - 1
            )
            
            # Parse and filter metrics
            filtered_metrics = []
            for metric in metrics:
                metric_data = json.loads(metric)
                metric_time = datetime.fromisoformat(metric_data["timestamp"])
                
                # Apply time filters
                if start_time and metric_time < start_time:
                    continue
                if end_time and metric_time > end_time:
                    continue
                    
                filtered_metrics.append(metric_data)
                
            return filtered_metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return []
            
    async def get_metric_summary(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """Get metric summary statistics."""
        try:
            # Get metrics
            metrics = await self.get_metrics(
                metric_type,
                start_time,
                end_time,
                limit=1000
            )
            
            if not metrics:
                return {}
                
            # Calculate statistics
            values = [m["value"] for m in metrics]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values)
            }
            
        except Exception as e:
            logger.error(f"Error getting metric summary: {str(e)}")
            return {} 