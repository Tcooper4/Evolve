"""
Advanced Order Executor

Implements advanced order types: TWAP, VWAP, and Iceberg orders.
These orders require special execution logic to split orders over time or volume.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from execution.broker_adapter import (
    BaseBrokerAdapter,
    OrderExecution,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


@dataclass
class OrderSlice:
    """Represents a slice of an advanced order."""
    slice_id: str
    parent_order_id: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    execute_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    execution: Optional[OrderExecution] = None


class AdvancedOrderExecutor:
    """
    Executes advanced order types (TWAP, VWAP, Iceberg).
    
    Features:
    - TWAP: Time-weighted average price execution
    - VWAP: Volume-weighted average price execution
    - Iceberg: Hidden quantity orders
    """
    
    def __init__(self, broker_adapter: BaseBrokerAdapter):
        """
        Initialize advanced order executor.
        
        Args:
            broker_adapter: Broker adapter for executing child orders
        """
        self.broker_adapter = broker_adapter
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.order_slices: Dict[str, List[OrderSlice]] = {}
        self.is_running = False
        self.execution_task: Optional[asyncio.Task] = None
        
        logger.info("Advanced Order Executor initialized")
    
    async def start(self):
        """Start the advanced order executor."""
        if self.is_running:
            return
        
        self.is_running = True
        self.execution_task = asyncio.create_task(self._execution_loop())
        logger.info("Advanced Order Executor started")
    
    async def stop(self):
        """Stop the advanced order executor."""
        self.is_running = False
        
        if self.execution_task:
            self.execution_task.cancel()
            try:
                await self.execution_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced Order Executor stopped")
    
    async def submit_advanced_order(self, order: OrderRequest) -> str:
        """
        Submit an advanced order (TWAP, VWAP, or Iceberg).
        
        Args:
            order: Order request with advanced order parameters
        
        Returns:
            Order ID
        """
        if order.order_type == OrderType.TWAP:
            return await self._submit_twap_order(order)
        elif order.order_type == OrderType.VWAP:
            return await self._submit_vwap_order(order)
        elif order.order_type == OrderType.ICEBERG:
            return await self._submit_iceberg_order(order)
        else:
            raise ValueError(f"Not an advanced order type: {order.order_type}")
    
    async def _submit_twap_order(self, order: OrderRequest) -> str:
        """
        Submit a TWAP (Time-Weighted Average Price) order.
        
        TWAP orders are split into equal slices over a time period.
        """
        logger.info(
            f"Submitting TWAP order: {order.order_id} - {order.quantity} {order.ticker}"
        )
        
        # Calculate slice parameters
        duration = order.twap_duration_seconds or 3600  # Default 1 hour
        slice_count = order.twap_slice_count or 10  # Default 10 slices
        
        slice_quantity = order.quantity / slice_count
        slice_interval = duration / slice_count
        
        # Create order slices
        slices = []
        start_time = datetime.now()
        
        for i in range(slice_count):
            slice_id = f"{order.order_id}_slice_{i}"
            execute_at = start_time + timedelta(seconds=i * slice_interval)
            
            slice = OrderSlice(
                slice_id=slice_id,
                parent_order_id=order.order_id,
                quantity=slice_quantity,
                price=order.price,  # Use limit price if provided
                execute_at=execute_at,
            )
            slices.append(slice)
        
        # Store order and slices
        self.active_orders[order.order_id] = {
            "order": order,
            "order_type": "TWAP",
            "total_quantity": order.quantity,
            "executed_quantity": 0.0,
            "status": OrderStatus.PENDING,
            "start_time": start_time,
            "end_time": start_time + timedelta(seconds=duration),
        }
        self.order_slices[order.order_id] = slices
        
        logger.info(
            f"TWAP order {order.order_id} split into {slice_count} slices "
            f"over {duration} seconds"
        )
        
        return order.order_id
    
    async def _submit_vwap_order(self, order: OrderRequest) -> str:
        """
        Submit a VWAP (Volume-Weighted Average Price) order.
        
        VWAP orders are split based on historical volume profile.
        """
        logger.info(
            f"Submitting VWAP order: {order.order_id} - {order.quantity} {order.ticker}"
        )
        
        # Get historical volume data for volume profile
        try:
            # Try to get market data to estimate volume profile
            # In practice, this would use historical volume data
            market_data = await self.broker_adapter.get_market_data(order.ticker)
            
            # Estimate volume profile (simplified - would use real historical data)
            # Default: assume uniform volume distribution
            duration = 3600  # 1 hour default
            if order.vwap_start_time and order.vwap_end_time:
                start = datetime.fromisoformat(order.vwap_start_time)
                end = datetime.fromisoformat(order.vwap_end_time)
                duration = int((end - start).total_seconds())
            
            slice_count = 10  # Default 10 slices
            slice_quantity = order.quantity / slice_count
            slice_interval = duration / slice_count
            
            # Create order slices (uniform for now, would be volume-weighted)
            slices = []
            start_time = datetime.now()
            
            for i in range(slice_count):
                slice_id = f"{order.order_id}_slice_{i}"
                execute_at = start_time + timedelta(seconds=i * slice_interval)
                
                slice = OrderSlice(
                    slice_id=slice_id,
                    parent_order_id=order.order_id,
                    quantity=slice_quantity,
                    price=order.price,
                    execute_at=execute_at,
                )
                slices.append(slice)
            
            # Store order and slices
            self.active_orders[order.order_id] = {
                "order": order,
                "order_type": "VWAP",
                "total_quantity": order.quantity,
                "executed_quantity": 0.0,
                "status": OrderStatus.PENDING,
                "start_time": start_time,
                "end_time": start_time + timedelta(seconds=duration),
            }
            self.order_slices[order.order_id] = slices
            
            logger.info(
                f"VWAP order {order.order_id} split into {slice_count} slices "
                f"over {duration} seconds"
            )
            
            return order.order_id
        
        except Exception as e:
            logger.error(f"Failed to submit VWAP order: {e}")
            raise
    
    async def _submit_iceberg_order(self, order: OrderRequest) -> str:
        """
        Submit an Iceberg order.
        
        Iceberg orders show only a small visible quantity, revealing more as it fills.
        """
        logger.info(
            f"Submitting Iceberg order: {order.order_id} - {order.quantity} {order.ticker}"
        )
        
        # Calculate visible and hidden quantities
        visible_qty = order.iceberg_visible_quantity or (order.quantity * 0.1)  # Default 10%
        reveal_qty = order.iceberg_reveal_quantity or visible_qty  # Reveal same amount
        
        # Create initial visible slice
        slices = []
        slice_id = f"{order.order_id}_slice_0"
        
        slice = OrderSlice(
            slice_id=slice_id,
            parent_order_id=order.order_id,
            quantity=min(visible_qty, order.quantity),
            price=order.price,
            execute_at=datetime.now(),
        )
        slices.append(slice)
        
        # Calculate remaining hidden slices
        remaining_qty = order.quantity - visible_qty
        slice_index = 1
        
        while remaining_qty > 0:
            slice_qty = min(reveal_qty, remaining_qty)
            slice_id = f"{order.order_id}_slice_{slice_index}"
            
            # These slices will be revealed as previous ones fill
            slice = OrderSlice(
                slice_id=slice_id,
                parent_order_id=order.order_id,
                quantity=slice_qty,
                price=order.price,
                execute_at=None,  # Will be set when previous slice fills
            )
            slices.append(slice)
            
            remaining_qty -= slice_qty
            slice_index += 1
        
        # Store order and slices
        self.active_orders[order.order_id] = {
            "order": order,
            "order_type": "ICEBERG",
            "total_quantity": order.quantity,
            "executed_quantity": 0.0,
            "status": OrderStatus.PENDING,
            "visible_quantity": visible_qty,
            "reveal_quantity": reveal_qty,
            "current_slice_index": 0,
        }
        self.order_slices[order.order_id] = slices
        
        logger.info(
            f"Iceberg order {order.order_id} created with {len(slices)} slices, "
            f"visible: {visible_qty}"
        )
        
        return order.order_id
    
    async def _execution_loop(self):
        """Main execution loop for advanced orders."""
        while self.is_running:
            try:
                await self._process_pending_slices()
                await asyncio.sleep(1)  # Check every second
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_pending_slices(self):
        """Process pending order slices."""
        current_time = datetime.now()
        
        for order_id, order_info in list(self.active_orders.items()):
            if order_info["status"] in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                continue
            
            slices = self.order_slices.get(order_id, [])
            
            for slice in slices:
                if slice.status == OrderStatus.PENDING:
                    # Check if it's time to execute this slice
                    if slice.execute_at and current_time >= slice.execute_at:
                        await self._execute_slice(order_id, slice)
                    elif not slice.execute_at and order_info["order_type"] == "ICEBERG":
                        # Iceberg: execute when previous slice is filled
                        slice_index = slices.index(slice)
                        if slice_index > 0:
                            prev_slice = slices[slice_index - 1]
                            if prev_slice.status == OrderStatus.FILLED:
                                slice.execute_at = current_time
                                await self._execute_slice(order_id, slice)
            
            # Check if all slices are complete
            all_filled = all(s.status == OrderStatus.FILLED for s in slices)
            if all_filled and slices:
                order_info["status"] = OrderStatus.FILLED
                logger.info(f"Advanced order {order_id} fully executed")
    
    async def _execute_slice(
        self, order_id: str, slice: OrderSlice
    ) -> Optional[OrderExecution]:
        """Execute a single order slice."""
        if slice.status != OrderStatus.PENDING:
            return None
        
        try:
            order_info = self.active_orders[order_id]
            parent_order = order_info["order"]
            
            # Create child order request
            child_order = OrderRequest(
                order_id=slice.slice_id,
                ticker=parent_order.ticker,
                side=parent_order.side,
                order_type=OrderType.LIMIT if slice.price else OrderType.MARKET,
                quantity=slice.quantity,
                price=slice.price,
                stop_price=slice.stop_price,
                time_in_force=parent_order.time_in_force,
            )
            
            # Submit child order
            logger.info(
                f"Executing slice {slice.slice_id}: {slice.quantity} {parent_order.ticker}"
            )
            
            execution = await self.broker_adapter.submit_order(child_order)
            slice.execution = execution
            slice.status = execution.status
            
            # Update parent order
            if execution.status == OrderStatus.FILLED:
                order_info["executed_quantity"] += execution.executed_quantity
                logger.info(
                    f"Slice {slice.slice_id} filled: {execution.executed_quantity} @ "
                    f"{execution.average_price:.2f}"
                )
            
            return execution
        
        except Exception as e:
            logger.error(f"Failed to execute slice {slice.slice_id}: {e}")
            slice.status = OrderStatus.REJECTED
            return None
    
    async def cancel_advanced_order(self, order_id: str) -> bool:
        """Cancel an advanced order and all its slices."""
        if order_id not in self.active_orders:
            return False
        
        try:
            order_info = self.active_orders[order_id]
            slices = self.order_slices.get(order_id, [])
            
            # Cancel all pending slices
            cancelled_count = 0
            for slice in slices:
                if slice.status == OrderStatus.PENDING:
                    try:
                        # Cancel the child order if it was submitted
                        if slice.execution:
                            await self.broker_adapter.cancel_order(slice.slice_id)
                        slice.status = OrderStatus.CANCELLED
                        cancelled_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to cancel slice {slice.slice_id}: {e}")
            
            order_info["status"] = OrderStatus.CANCELLED
            
            logger.info(
                f"Cancelled advanced order {order_id} ({cancelled_count} slices cancelled)"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel advanced order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an advanced order."""
        if order_id not in self.active_orders:
            return None
        
        order_info = self.active_orders[order_id]
        slices = self.order_slices.get(order_id, [])
        
        # Calculate aggregate statistics
        total_executed = sum(
            s.execution.executed_quantity
            for s in slices
            if s.execution and s.execution.status == OrderStatus.FILLED
        )
        
        total_cost = sum(
            s.execution.executed_quantity * s.execution.average_price
            for s in slices
            if s.execution and s.execution.status == OrderStatus.FILLED
        )
        
        avg_price = total_cost / total_executed if total_executed > 0 else 0.0
        
        return {
            "order_id": order_id,
            "order_type": order_info["order_type"],
            "ticker": order_info["order"].ticker,
            "total_quantity": order_info["total_quantity"],
            "executed_quantity": total_executed,
            "remaining_quantity": order_info["total_quantity"] - total_executed,
            "average_price": avg_price,
            "status": order_info["status"].value,
            "slice_count": len(slices),
            "filled_slices": sum(1 for s in slices if s.status == OrderStatus.FILLED),
            "pending_slices": sum(1 for s in slices if s.status == OrderStatus.PENDING),
            "slices": [
                {
                    "slice_id": s.slice_id,
                    "quantity": s.quantity,
                    "status": s.status.value,
                    "executed_quantity": (
                        s.execution.executed_quantity if s.execution else 0.0
                    ),
                    "average_price": (
                        s.execution.average_price if s.execution else 0.0
                    ),
                }
                for s in slices
            ],
        }
    
    def get_all_active_orders(self) -> List[Dict[str, Any]]:
        """Get status of all active advanced orders."""
        return [
            self.get_order_status(order_id)
            for order_id in self.active_orders.keys()
            if self.active_orders[order_id]["status"] not in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
            ]
        ]

