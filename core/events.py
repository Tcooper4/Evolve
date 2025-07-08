"""
Event-Driven Architecture System

This module provides an event bus for loose coupling between system components.
Components can publish events and subscribe to events without direct dependencies.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Set
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time

from .interfaces import IEventBus, EventType, SystemEvent

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class EventHandler:
    """Event handler registration."""
    handler: Callable[[SystemEvent], None]
    priority: EventPriority = EventPriority.NORMAL
    async_handler: bool = False
    filter_func: Optional[Callable[[SystemEvent], bool]] = None

class EventBus(IEventBus):
    """
    Event bus implementation for system-wide event communication.
    
    Features:
    - Synchronous and asynchronous event handling
    - Event filtering and priority handling
    - Event history and replay
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, max_history: int = 1000, enable_async: bool = True):
        """
        Initialize the event bus.
        
        Args:
            max_history: Maximum number of events to keep in history
            enable_async: Whether to enable asynchronous event processing
        """
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._event_history: List[SystemEvent] = []
        self._max_history = max_history
        self._enable_async = enable_async
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_queue: queue.Queue = queue.Queue()
        self._processing_thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'handlers_executed': 0,
            'errors': 0
        }
        
        if self._enable_async:
            self._start_async_processing()
    
    def _start_async_processing(self) -> None:
        """Start the asynchronous event processing thread."""
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_events_async, daemon=True)
        self._processing_thread.start()
        logger.info("Event bus async processing started")
    
    def _process_events_async(self) -> None:
        """Process events asynchronously in a separate thread."""
        try:
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            
            while self._running:
                try:
                    # Get event from queue with timeout
                    event = self._event_queue.get(timeout=1.0)
                    self._async_loop.run_until_complete(self._handle_event_async(event))
                    self._event_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing async event: {e}")
                    self._stats['errors'] += 1
        except Exception as e:
            logger.error(f"Async event processing thread error: {e}")
        finally:
            if self._async_loop:
                self._async_loop.close()
    
    async def _handle_event_async(self, event: SystemEvent) -> None:
        """Handle an event asynchronously."""
        handlers = self._handlers.get(event.event_type, [])
        
        # Sort handlers by priority
        handlers.sort(key=lambda h: h.priority.value, reverse=True)
        
        for handler_info in handlers:
            if handler_info.async_handler:
                try:
                    await handler_info.handler(event)
                    self._stats['handlers_executed'] += 1
                except Exception as e:
                    logger.error(f"Async handler error: {e}")
                    self._stats['errors'] += 1
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[SystemEvent], None],
        priority: EventPriority = EventPriority.NORMAL,
        async_handler: bool = False,
        filter_func: Optional[Callable[[SystemEvent], bool]] = None
    ) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: The event type to subscribe to
            handler: The handler function to call
            priority: Handler priority
            async_handler: Whether this is an async handler
            filter_func: Optional filter function
        """
        handler_info = EventHandler(
            handler=handler,
            priority=priority,
            async_handler=async_handler,
            filter_func=filter_func
        )
        
        self._handlers[event_type].append(handler_info)
        logger.debug(f"Subscribed to {event_type.value} with priority {priority.value}")
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[SystemEvent], None]
    ) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: The event type to unsubscribe from
            handler: The handler function to remove
        """
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type]
                if h.handler != handler
            ]
            logger.debug(f"Unsubscribed from {event_type.value}")
    
    def publish(self, event: SystemEvent) -> None:
        """
        Publish an event.
        
        Args:
            event: The event to publish
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        self._stats['events_published'] += 1
        
        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])
        
        if not handlers:
            logger.debug(f"No handlers for event type: {event.event_type.value}")
            return
        
        # Sort handlers by priority
        handlers.sort(key=lambda h: h.priority.value, reverse=True)
        
        # Process synchronous handlers
        sync_handlers = [h for h in handlers if not h.async_handler]
        for handler_info in sync_handlers:
            try:
                # Apply filter if present
                if handler_info.filter_func and not handler_info.filter_func(event):
                    continue
                
                handler_info.handler(event)
                self._stats['handlers_executed'] += 1
            except Exception as e:
                logger.error(f"Handler error: {e}")
                self._stats['errors'] += 1
        
        # Queue async handlers
        async_handlers = [h for h in handlers if h.async_handler]
        if async_handlers and self._enable_async:
            self._event_queue.put(event)
        
        logger.debug(f"Published event: {event.event_type.value} from {event.source}")
    
    def publish_sync(self, event: SystemEvent) -> None:
        """Publish an event synchronously (all handlers run in current thread)."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        self._stats['events_published'] += 1
        
        handlers = self._handlers.get(event.event_type, [])
        handlers.sort(key=lambda h: h.priority.value, reverse=True)
        
        for handler_info in handlers:
            try:
                if handler_info.filter_func and not handler_info.filter_func(event):
                    continue
                
                handler_info.handler(event)
                self._stats['handlers_executed'] += 1
            except Exception as e:
                logger.error(f"Handler error: {e}")
                self._stats['errors'] += 1
        
        logger.debug(f"Published sync event: {event.event_type.value}")
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[SystemEvent]:
        """
        Get event history.
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        if event_type:
            events = [e for e in self._event_history if e.event_type == event_type]
        else:
            events = self._event_history.copy()
        
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            'active_handlers': sum(len(handlers) for handlers in self._handlers.values()),
            'event_types': len(self._handlers),
            'history_size': len(self._event_history),
            'queue_size': self._event_queue.qsize() if self._enable_async else 0
        }
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")
    
    def shutdown(self) -> None:
        """Shutdown the event bus."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        logger.info("Event bus shutdown complete")

# Convenience functions for common events
def publish_data_loaded(symbol: str, data: Any, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Publish a data loaded event."""
    event = SystemEvent(
        event_type=EventType.DATA_LOADED,
        timestamp=datetime.now(),
        data={'symbol': symbol, 'data': data},
        source=source,
        metadata=metadata
    )
    get_event_bus().publish(event)

def publish_model_trained(model_name: str, metrics: Dict[str, float], source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Publish a model trained event."""
    event = SystemEvent(
        event_type=EventType.MODEL_TRAINED,
        timestamp=datetime.now(),
        data={'model_name': model_name, 'metrics': metrics},
        source=source,
        metadata=metadata
    )
    get_event_bus().publish(event)

def publish_signal_generated(symbol: str, signal_type: str, strategy: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Publish a signal generated event."""
    event = SystemEvent(
        event_type=EventType.SIGNAL_GENERATED,
        timestamp=datetime.now(),
        data={'symbol': symbol, 'signal_type': signal_type, 'strategy': strategy},
        source=source,
        metadata=metadata
    )
    get_event_bus().publish(event)

def publish_trade_executed(symbol: str, trade_type: str, quantity: float, price: float, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Publish a trade executed event."""
    event = SystemEvent(
        event_type=EventType.TRADE_EXECUTED,
        timestamp=datetime.now(),
        data={'symbol': symbol, 'trade_type': trade_type, 'quantity': quantity, 'price': price},
        source=source,
        metadata=metadata
    )
    get_event_bus().publish(event)

def publish_risk_alert(alert_type: str, message: str, severity: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Publish a risk alert event."""
    event = SystemEvent(
        event_type=EventType.RISK_ALERT,
        timestamp=datetime.now(),
        data={'alert_type': alert_type, 'message': message, 'severity': severity},
        source=source,
        metadata=metadata
    )
    get_event_bus().publish(event)

def publish_system_error(error_type: str, message: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Publish a system error event."""
    event = SystemEvent(
        event_type=EventType.SYSTEM_ERROR,
        timestamp=datetime.now(),
        data={'error_type': error_type, 'message': message},
        source=source,
        metadata=metadata
    )
    get_event_bus().publish(event)

# Global event bus instance
_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

def set_event_bus(event_bus: EventBus) -> None:
    """Set the global event bus instance."""
    global _event_bus
    _event_bus = event_bus

__all__ = [
    'EventBus', 'EventPriority', 'EventHandler',
    'publish_data_loaded', 'publish_model_trained', 'publish_signal_generated',
    'publish_trade_executed', 'publish_risk_alert', 'publish_system_error',
    'get_event_bus', 'set_event_bus'
] 