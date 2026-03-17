"""
Redundant Broker Manager with Failover Support

Provides broker redundancy and automatic failover for high-availability trading.
Supports multiple brokers with health monitoring and automatic switching.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from execution.broker_adapter import (
    BaseBrokerAdapter,
    BrokerAdapter,
    BrokerType,
    OrderExecution,
    OrderRequest,
    AccountInfo,
    Position,
    MarketData,
)

logger = logging.getLogger(__name__)


class BrokerHealth(Enum):
    """Broker health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BrokerConfig:
    """Configuration for a broker in the redundant setup."""
    broker_type: BrokerType
    config: Dict[str, Any]
    priority: int  # Lower number = higher priority
    enabled: bool = True
    name: Optional[str] = None


@dataclass
class BrokerStatus:
    """Status information for a broker."""
    broker_id: str
    broker_type: BrokerType
    health: BrokerHealth
    is_connected: bool
    last_successful_request: Optional[datetime]
    last_failure: Optional[datetime]
    failure_count: int
    success_count: int
    average_response_time: float
    last_health_check: Optional[datetime]
    error_message: Optional[str] = None


class RedundantBrokerManager:
    """
    Manages multiple broker connections with automatic failover.
    
    Features:
    - Multiple broker support with priority ordering
    - Automatic failover on broker failure
    - Health monitoring and status tracking
    - Load balancing (optional)
    - Connection retry with exponential backoff
    """
    
    def __init__(
        self,
        broker_configs: List[BrokerConfig],
        failover_enabled: bool = True,
        health_check_interval: float = 30.0,
        max_failures_before_switch: int = 3,
        response_timeout: float = 10.0,
    ):
        """
        Initialize redundant broker manager.
        
        Args:
            broker_configs: List of broker configurations (ordered by priority)
            failover_enabled: Enable automatic failover
            health_check_interval: Interval between health checks (seconds)
            max_failures_before_switch: Number of failures before switching brokers
            response_timeout: Timeout for broker operations (seconds)
        """
        self.broker_configs = sorted(broker_configs, key=lambda x: x.priority)
        self.failover_enabled = failover_enabled
        self.health_check_interval = health_check_interval
        self.max_failures_before_switch = max_failures_before_switch
        self.response_timeout = response_timeout
        
        # Broker adapters
        self.adapters: Dict[str, BrokerAdapter] = {}
        self.broker_statuses: Dict[str, BrokerStatus] = {}
        
        # Current active broker
        self.active_broker_id: Optional[str] = None
        
        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "failover_count": 0,
            "last_failover": None,
        }
        
        # Initialize brokers
        self._initialize_brokers()
        
        logger.info(
            f"Initialized RedundantBrokerManager with {len(self.adapters)} brokers"
        )
    
    def _initialize_brokers(self):
        """Initialize all broker adapters."""
        for i, broker_config in enumerate(self.broker_configs):
            if not broker_config.enabled:
                continue
            
            broker_id = broker_config.name or f"{broker_config.broker_type.value}_{i}"
            
            try:
                adapter = BrokerAdapter(
                    broker_type=broker_config.broker_type.value,
                    config=broker_config.config
                )
                self.adapters[broker_id] = adapter
                
                # Initialize status
                self.broker_statuses[broker_id] = BrokerStatus(
                    broker_id=broker_id,
                    broker_type=broker_config.broker_type,
                    health=BrokerHealth.UNKNOWN,
                    is_connected=False,
                    last_successful_request=None,
                    last_failure=None,
                    failure_count=0,
                    success_count=0,
                    average_response_time=0.0,
                    last_health_check=None,
                )
                
                logger.info(f"Initialized broker: {broker_id} ({broker_config.broker_type.value})")
            
            except Exception as e:
                logger.error(f"Failed to initialize broker {broker_id}: {e}")
                # Mark as unhealthy
                if broker_id in self.broker_statuses:
                    self.broker_statuses[broker_id].health = BrokerHealth.UNHEALTHY
                    self.broker_statuses[broker_id].error_message = str(e)
    
    async def start(self):
        """Start the broker manager and connect to brokers."""
        if self.is_running:
            logger.warning("Broker manager already running")
            return
        
        self.is_running = True
        
        # Connect to all brokers
        for broker_id, adapter in self.adapters.items():
            try:
                connected = await asyncio.wait_for(
                    adapter.connect(),
                    timeout=self.response_timeout
                )
                
                if connected:
                    self.broker_statuses[broker_id].is_connected = True
                    self.broker_statuses[broker_id].health = BrokerHealth.HEALTHY
                    logger.info(f"Connected to broker: {broker_id}")
                else:
                    self.broker_statuses[broker_id].health = BrokerHealth.UNHEALTHY
                    logger.warning(f"Failed to connect to broker: {broker_id}")
            
            except asyncio.TimeoutError:
                logger.error(f"Connection timeout for broker: {broker_id}")
                self.broker_statuses[broker_id].health = BrokerHealth.UNHEALTHY
                self.broker_statuses[broker_id].error_message = "Connection timeout"
            
            except Exception as e:
                logger.error(f"Error connecting to broker {broker_id}: {e}")
                self.broker_statuses[broker_id].health = BrokerHealth.UNHEALTHY
                self.broker_statuses[broker_id].error_message = str(e)
        
        # Set active broker (first healthy one)
        self._select_active_broker()
        
        # Start health check task
        if self.failover_enabled:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Broker manager started. Active broker: {self.active_broker_id}")
    
    async def stop(self):
        """Stop the broker manager and disconnect from all brokers."""
        self.is_running = False
        
        # Stop health check
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect from all brokers
        for broker_id, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                self.broker_statuses[broker_id].is_connected = False
                logger.info(f"Disconnected from broker: {broker_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from broker {broker_id}: {e}")
        
        logger.info("Broker manager stopped")
    
    def _select_active_broker(self):
        """Select the active broker based on health and priority."""
        # Find first healthy broker
        for broker_config in self.broker_configs:
            broker_id = broker_config.name or f"{broker_config.broker_type.value}_0"
            
            if broker_id not in self.broker_statuses:
                continue
            
            status = self.broker_statuses[broker_id]
            
            if (
                status.health in [BrokerHealth.HEALTHY, BrokerHealth.DEGRADED]
                and status.is_connected
            ):
                if self.active_broker_id != broker_id:
                    if self.active_broker_id:
                        logger.info(f"Switching active broker: {self.active_broker_id} -> {broker_id}")
                        self.stats["failover_count"] += 1
                        self.stats["last_failover"] = datetime.now().isoformat()
                    else:
                        logger.info(f"Selected active broker: {broker_id}")
                    
                    self.active_broker_id = broker_id
                    return
        
        # No healthy broker found
        if self.active_broker_id:
            logger.warning(f"No healthy broker available. Previous active: {self.active_broker_id}")
        else:
            logger.error("No healthy broker available")
        
        self.active_broker_id = None
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_brokers_health()
                
                # Re-select active broker if needed
                if self.failover_enabled:
                    current_status = self.broker_statuses.get(self.active_broker_id)
                    if (
                        not current_status
                        or current_status.health == BrokerHealth.UNHEALTHY
                        or not current_status.is_connected
                    ):
                        self._select_active_broker()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _check_all_brokers_health(self):
        """Check health of all brokers."""
        for broker_id, adapter in self.adapters.items():
            try:
                start_time = time.time()
                
                # Try to get account info as health check
                account_info = await asyncio.wait_for(
                    adapter.get_account_info(),
                    timeout=self.response_timeout
                )
                
                response_time = time.time() - start_time
                status = self.broker_statuses[broker_id]
                
                # Update status
                status.last_health_check = datetime.now()
                status.last_successful_request = datetime.now()
                status.success_count += 1
                status.is_connected = True
                status.error_message = None
                
                # Update average response time
                if status.average_response_time == 0:
                    status.average_response_time = response_time
                else:
                    status.average_response_time = (
                        status.average_response_time * 0.9 + response_time * 0.1
                    )
                
                # Determine health
                if response_time > self.response_timeout * 0.8:
                    status.health = BrokerHealth.DEGRADED
                elif status.failure_count > 0:
                    status.health = BrokerHealth.DEGRADED
                else:
                    status.health = BrokerHealth.HEALTHY
                
                # Reset failure count on success
                if status.failure_count > 0:
                    status.failure_count = max(0, status.failure_count - 1)
            
            except asyncio.TimeoutError:
                self._update_broker_failure(broker_id, "Health check timeout")
            
            except Exception as e:
                self._update_broker_failure(broker_id, str(e))
    
    def _update_broker_failure(self, broker_id: str, error_message: str):
        """Update broker status on failure."""
        if broker_id not in self.broker_statuses:
            return
        
        status = self.broker_statuses[broker_id]
        status.last_failure = datetime.now()
        status.failure_count += 1
        status.error_message = error_message
        
        # Mark as unhealthy if too many failures
        if status.failure_count >= self.max_failures_before_switch:
            status.health = BrokerHealth.UNHEALTHY
            status.is_connected = False
            logger.warning(
                f"Broker {broker_id} marked as unhealthy after {status.failure_count} failures"
            )
        else:
            status.health = BrokerHealth.DEGRADED
        
        self.stats["failed_requests"] += 1
    
    async def _execute_with_failover(
        self,
        operation: str,
        func,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with automatic failover.
        
        Args:
            operation: Operation name for logging
            func: Function to execute (must be a method of BrokerAdapter)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Result of the operation
        """
        if not self.active_broker_id:
            raise Exception("No active broker available")
        
        last_error = None
        attempted_brokers = []
        
        # Try active broker first
        broker_ids_to_try = [self.active_broker_id]
        
        # Add other healthy brokers as fallback
        for broker_config in self.broker_configs:
            broker_id = broker_config.name or f"{broker_config.broker_type.value}_0"
            if (
                broker_id != self.active_broker_id
                and broker_id in self.adapters
                and self.broker_statuses[broker_id].health != BrokerHealth.UNHEALTHY
            ):
                broker_ids_to_try.append(broker_id)
        
        # Try each broker
        for broker_id in broker_ids_to_try:
            if broker_id in attempted_brokers:
                continue
            
            attempted_brokers.append(broker_id)
            adapter = self.adapters[broker_id]
            status = self.broker_statuses[broker_id]
            
            try:
                self.stats["total_requests"] += 1
                start_time = time.time()
                
                # Execute operation with timeout
                result = await asyncio.wait_for(
                    func(adapter, *args, **kwargs),
                    timeout=self.response_timeout
                )
                
                response_time = time.time() - start_time
                
                # Update success metrics
                status.last_successful_request = datetime.now()
                status.success_count += 1
                status.is_connected = True
                status.error_message = None
                
                # Update average response time
                if status.average_response_time == 0:
                    status.average_response_time = response_time
                else:
                    status.average_response_time = (
                        status.average_response_time * 0.9 + response_time * 0.1
                    )
                
                # Reset failure count
                if status.failure_count > 0:
                    status.failure_count = max(0, status.failure_count - 1)
                
                # Update health
                if response_time > self.response_timeout * 0.8:
                    status.health = BrokerHealth.DEGRADED
                else:
                    status.health = BrokerHealth.HEALTHY
                
                # Switch to this broker if it's not the active one and it's healthy
                if broker_id != self.active_broker_id and status.health == BrokerHealth.HEALTHY:
                    self.active_broker_id = broker_id
                    logger.info(f"Switched to broker: {broker_id} (better performance)")
                
                self.stats["successful_requests"] += 1
                return result
            
            except asyncio.TimeoutError:
                error_msg = f"{operation} timeout on {broker_id}"
                logger.warning(error_msg)
                self._update_broker_failure(broker_id, error_msg)
                last_error = TimeoutError(error_msg)
            
            except Exception as e:
                error_msg = f"{operation} failed on {broker_id}: {str(e)}"
                logger.warning(error_msg)
                self._update_broker_failure(broker_id, error_msg)
                last_error = e
        
        # All brokers failed
        self.stats["failed_requests"] += 1
        
        # Try to select a new active broker
        if self.failover_enabled:
            self._select_active_broker()
        
        if last_error:
            raise last_error
        else:
            raise Exception(f"{operation} failed on all brokers")
    
    # Broker operation methods with failover
    
    async def submit_order(self, order: OrderRequest) -> OrderExecution:
        """Submit order with failover support."""
        async def _submit(adapter, order):
            return await adapter.submit_order(order)
        
        return await self._execute_with_failover("submit_order", _submit, order)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order with failover support."""
        async def _cancel(adapter, order_id):
            return await adapter.cancel_order(order_id)
        
        return await self._execute_with_failover("cancel_order", _cancel, order_id)
    
    async def get_order_status(self, order_id: str) -> Optional[OrderExecution]:
        """Get order status with failover support."""
        async def _get_status(adapter, order_id):
            return await adapter.get_order_status(order_id)
        
        return await self._execute_with_failover("get_order_status", _get_status, order_id)
    
    async def get_position(self, ticker: str) -> Optional[Position]:
        """Get position with failover support."""
        async def _get_position(adapter, ticker):
            return await adapter.get_position(ticker)
        
        return await self._execute_with_failover("get_position", _get_position, ticker)
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions with failover support."""
        async def _get_all(adapter):
            return await adapter.get_all_positions()
        
        return await self._execute_with_failover("get_all_positions", _get_all)
    
    async def get_account_info(self) -> AccountInfo:
        """Get account info with failover support."""
        async def _get_account(adapter):
            return await adapter.get_account_info()
        
        return await self._execute_with_failover("get_account_info", _get_account)
    
    async def get_market_data(self, ticker: str) -> MarketData:
        """Get market data with failover support."""
        async def _get_market(adapter, ticker):
            return await adapter.get_market_data(ticker)
        
        return await self._execute_with_failover("get_market_data", _get_market, ticker)
    
    def get_broker_status(self, broker_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a specific broker or all brokers."""
        if broker_id:
            if broker_id in self.broker_statuses:
                status = self.broker_statuses[broker_id]
                return {
                    "broker_id": status.broker_id,
                    "broker_type": status.broker_type.value,
                    "health": status.health.value,
                    "is_connected": status.is_connected,
                    "is_active": broker_id == self.active_broker_id,
                    "last_successful_request": (
                        status.last_successful_request.isoformat()
                        if status.last_successful_request
                        else None
                    ),
                    "last_failure": (
                        status.last_failure.isoformat()
                        if status.last_failure
                        else None
                    ),
                    "failure_count": status.failure_count,
                    "success_count": status.success_count,
                    "average_response_time": status.average_response_time,
                    "error_message": status.error_message,
                }
            else:
                return {"error": f"Broker {broker_id} not found"}
        else:
            return {
                "active_broker": self.active_broker_id,
                "brokers": {
                    bid: self.get_broker_status(bid)
                    for bid in self.broker_statuses.keys()
                },
                "statistics": self.stats,
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            "active_broker": self.active_broker_id,
            "total_brokers": len(self.adapters),
            "healthy_brokers": sum(
                1
                for status in self.broker_statuses.values()
                if status.health == BrokerHealth.HEALTHY
            ),
        }

