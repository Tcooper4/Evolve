import asyncio
import json
import logging
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import aiohttp
import dns.resolver
import psutil
import pymongo
import redis
from cachetools import TTLCache
from elasticsearch import Elasticsearch
from prometheus_client import Gauge, start_http_server
from pydantic import BaseModel, Field
from ratelimit import limits, sleep_and_retry
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class HealthConfig(BaseModel):
    """Configuration for health checks."""

    check_interval: int = Field(default=60)
    timeout: int = Field(default=10)
    retries: int = Field(default=3)
    retry_delay: int = Field(default=5)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)
    metrics_port: int = Field(default=9090)
    alert_threshold: float = Field(default=0.9)
    services: Dict[str, Dict[str, Any]] = {}


class HealthStatus(BaseModel):
    """Health status model."""

    service: str
    status: str
    message: str
    timestamp: datetime
    metrics: Dict[str, float] = {}
    details: Dict[str, Any] = {}


class AutomationHealth:
    """Health monitoring functionality."""

    def __init__(self, config_path: str = "automation/config/health.json"):
        """Initialize health monitoring."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_metrics()
        self.setup_cache()
        self.status: Dict[str, HealthStatus] = {}
        self.lock = asyncio.Lock()

    def _load_config(self, config_path: str) -> HealthConfig:
        """Load health configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return HealthConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load health config: {str(e)}")
            raise

    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_path / "health.log"), logging.StreamHandler()],
        )

    def setup_metrics(self):
        """Setup health metrics."""
        try:
            # System metrics
            self.cpu_usage = Gauge("system_cpu_usage", "CPU usage percentage")
            self.memory_usage = Gauge("system_memory_usage", "Memory usage percentage")
            self.disk_usage = Gauge("system_disk_usage", "Disk usage percentage")

            # Service metrics
            self.service_status = Gauge("service_status", "Service status (1=healthy, 0=unhealthy)", ["service"])
            self.service_latency = Gauge("service_latency_seconds", "Service response latency", ["service"])

            # Start metrics server
            start_http_server(self.config.metrics_port)

        except Exception as e:
            logger.error(f"Failed to setup metrics: {str(e)}")
            raise

    def setup_cache(self):
        """Setup health caching."""
        self.cache = TTLCache(maxsize=self.config.cache_size, ttl=self.config.cache_ttl)

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def check_system_health(self) -> HealthStatus:
        """Check system health."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Update metrics
            self.cpu_usage.set(cpu_percent)
            self.memory_usage.set(memory.percent)
            self.disk_usage.set(disk.percent)

            # Check thresholds
            is_healthy = all(
                [
                    cpu_percent < self.config.alert_threshold * 100,
                    memory.percent < self.config.alert_threshold * 100,
                    disk.percent < self.config.alert_threshold * 100,
                ]
            )

            status = HealthStatus(
                service="system",
                status="healthy" if is_healthy else "unhealthy",
                message="System is healthy" if is_healthy else "System resources critical",
                timestamp=datetime.now(),
                metrics={"cpu_usage": cpu_percent, "memory_usage": memory.percent, "disk_usage": disk.percent},
                details={
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": memory.total,
                    "memory_available": memory.available,
                    "disk_total": disk.total,
                    "disk_free": disk.free,
                },
            )

            self.status["system"] = status
            return status

        except Exception as e:
            logger.error(f"Failed to check system health: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def check_service_health(self, service: str, config: Dict[str, Any]) -> HealthStatus:
        """Check service health."""
        try:
            start_time = time.time()

            # Check based on service type
            if config["type"] == "http":
                status = await self._check_http_health(
                    service,
                    config["url"],
                    config.get("method", "GET"),
                    config.get("headers", {}),
                    config.get("timeout", self.config.timeout),
                )
            elif config["type"] == "tcp":
                status = await self._check_tcp_health(
                    service, config["host"], config["port"], config.get("timeout", self.config.timeout)
                )
            elif config["type"] == "dns":
                status = await self._check_dns_health(
                    service,
                    config["domain"],
                    config.get("record_type", "A"),
                    config.get("timeout", self.config.timeout),
                )
            elif config["type"] == "redis":
                status = await self._check_redis_health(
                    service,
                    config["host"],
                    config["port"],
                    config.get("password"),
                    config.get("timeout", self.config.timeout),
                )
            elif config["type"] == "mongodb":
                status = await self._check_mongodb_health(
                    service, config["uri"], config.get("timeout", self.config.timeout)
                )
            elif config["type"] == "postgresql":
                status = await self._check_postgresql_health(
                    service, config["uri"], config.get("timeout", self.config.timeout)
                )
            elif config["type"] == "elasticsearch":
                status = await self._check_elasticsearch_health(
                    service, config["host"], config["port"], config.get("timeout", self.config.timeout)
                )
            else:
                raise ValueError(f"Unsupported service type: {config['type']}")

            # Calculate latency
            latency = time.time() - start_time

            # Update metrics
            self.service_status.labels(service=service).set(1 if status.status == "healthy" else 0)
            self.service_latency.labels(service=service).set(latency)

            # Add latency to metrics
            status.metrics["latency"] = latency

            self.status[service] = status
            return status

        except Exception as e:
            logger.error(f"Failed to check service health: {str(e)}")
            raise

    async def _check_http_health(
        self, service: str, url: str, method: str = "GET", headers: Dict[str, str] = {}, timeout: int = 10
    ) -> HealthStatus:
        """Check HTTP service health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, timeout=timeout) as response:
                    is_healthy = response.status < 400

                    return HealthStatus(
                        service=service,
                        status="healthy" if is_healthy else "unhealthy",
                        message=f"HTTP {response.status}",
                        timestamp=datetime.now(),
                        metrics={"status_code": response.status, "response_time": response.elapsed.total_seconds()},
                    )

        except Exception as e:
            return HealthStatus(service=service, status="unhealthy", message=str(e), timestamp=datetime.now())

    async def _check_tcp_health(self, service: str, host: str, port: int, timeout: int = 10) -> HealthStatus:
        """Check TCP service health."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            is_healthy = result == 0

            return HealthStatus(
                service=service,
                status="healthy" if is_healthy else "unhealthy",
                message=f"TCP {result}",
                timestamp=datetime.now(),
            )

        except Exception as e:
            return HealthStatus(service=service, status="unhealthy", message=str(e), timestamp=datetime.now())

    async def _check_dns_health(
        self, service: str, domain: str, record_type: str = "A", timeout: int = 10
    ) -> HealthStatus:
        """Check DNS service health."""
        try:
            resolver = dns.resolver.Resolver()
            resolver.timeout = timeout
            resolver.lifetime = timeout

            answers = resolver.resolve(domain, record_type)

            return HealthStatus(
                service=service,
                status="healthy",
                message=f"DNS {record_type} records found",
                timestamp=datetime.now(),
                metrics={"record_count": len(answers)},
            )

        except Exception as e:
            return HealthStatus(service=service, status="unhealthy", message=str(e), timestamp=datetime.now())

    async def _check_redis_health(
        self, service: str, host: str, port: int, password: Optional[str] = None, timeout: int = 10
    ) -> HealthStatus:
        """Check Redis service health."""
        try:
            client = redis.Redis(host=host, port=port, password=password, socket_timeout=timeout)

            # Test connection
            client.ping()

            # Get info
            info = client.info()

            return HealthStatus(
                service=service,
                status="healthy",
                message="Redis connected",
                timestamp=datetime.now(),
                metrics={
                    "connected_clients": info["connected_clients"],
                    "used_memory": info["used_memory"],
                    "total_connections_received": info["total_connections_received"],
                },
            )

        except Exception as e:
            return HealthStatus(service=service, status="unhealthy", message=str(e), timestamp=datetime.now())

    async def _check_mongodb_health(self, service: str, uri: str, timeout: int = 10) -> HealthStatus:
        """Check MongoDB service health."""
        try:
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=timeout * 1000)

            # Test connection
            client.admin.command("ping")

            # Get status
            status = client.admin.command("serverStatus")

            return HealthStatus(
                service=service,
                status="healthy",
                message="MongoDB connected",
                timestamp=datetime.now(),
                metrics={
                    "connections": status["connections"]["current"],
                    "opcounters": status["opcounters"],
                    "mem": status["mem"],
                },
            )

        except Exception as e:
            return HealthStatus(service=service, status="unhealthy", message=str(e), timestamp=datetime.now())

    async def _check_postgresql_health(self, service: str, uri: str, timeout: int = 10) -> HealthStatus:
        """Check PostgreSQL service health."""
        try:
            engine = create_engine(uri, connect_args={"connect_timeout": timeout})

            # Test connection
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")

            # Get status
            with engine.connect() as conn:
                result = conn.execute(
                    """
                    SELECT
                        numbackends as connections,
                        xact_commit as transactions,
                        blks_read as blocks_read,
                        blks_hit as blocks_hit
                    FROM pg_stat_database
                    WHERE datname = current_database()
                """
                )
                stats = result.fetchone()

            return HealthStatus(
                service=service,
                status="healthy",
                message="PostgreSQL connected",
                timestamp=datetime.now(),
                metrics={
                    "connections": stats[0],
                    "transactions": stats[1],
                    "blocks_read": stats[2],
                    "blocks_hit": stats[3],
                },
            )

        except Exception as e:
            return HealthStatus(service=service, status="unhealthy", message=str(e), timestamp=datetime.now())

    async def _check_elasticsearch_health(self, service: str, host: str, port: int, timeout: int = 10) -> HealthStatus:
        """Check Elasticsearch service health."""
        try:
            client = Elasticsearch([f"http://{host}:{port}"], timeout=timeout)

            # Test connection
            health = client.cluster.health()

            return HealthStatus(
                service=service,
                status="healthy",
                message=f"Elasticsearch {health['status']}",
                timestamp=datetime.now(),
                metrics={
                    "number_of_nodes": health["number_of_nodes"],
                    "number_of_data_nodes": health["number_of_data_nodes"],
                    "active_shards": health["active_shards"],
                    "relocating_shards": health["relocating_shards"],
                    "initializing_shards": health["initializing_shards"],
                    "unassigned_shards": health["unassigned_shards"],
                },
            )

        except Exception as e:
            return HealthStatus(service=service, status="unhealthy", message=str(e), timestamp=datetime.now())

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def check_all_health(self) -> Dict[str, HealthStatus]:
        """Check health of all services."""
        try:
            # Check system health
            await self.check_system_health()

            # Check service health
            for service, config in self.config.services.items():
                await self.check_service_health(service, config)

            return self.status

        except Exception as e:
            logger.error(f"Failed to check all health: {str(e)}")
            raise

    async def get_health_status(self, service: Optional[str] = None) -> Union[HealthStatus, Dict[str, HealthStatus]]:
        """Get health status."""
        try:
            if service:
                return self.status.get(service)
            return self.status

        except Exception as e:
            logger.error(f"Failed to get health status: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.cache.clear()
            self.status.clear()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
