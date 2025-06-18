# -*- coding: utf-8 -*-
"""
Unified routing engine for the trading system.

This module provides a unified interface for both cognitive and operational routing,
handling both agent selection and infrastructure-level task routing.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime

from ..config.settings import ROUTING_CONFIG
from ..utils.error_handling import handle_routing_errors

logger = logging.getLogger(__name__)

class RouteType(str, Enum):
    """Types of routing operations."""
    COGNITIVE = "cognitive"  # Agent selection and prompt interpretation
    OPERATIONAL = "operational"  # Infrastructure and scheduling tasks

@dataclass
class RouteResult:
    """Result of a routing operation."""
    success: bool
    route_type: RouteType
    handler: Optional[Callable] = None
    confidence: float = 0.0
    message: str = ""
    metadata: Optional[Dict[str, Any]] = None

class RoutingEngine:
    """Unified routing engine for cognitive and operational tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the routing engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or ROUTING_CONFIG
        self.cognitive_handlers: Dict[str, Callable] = {}
        self.operational_handlers: Dict[str, Callable] = {}
        self._setup_routing()
        
    def _setup_routing(self):
        """Setup routing configuration."""
        # Load cognitive routing handlers
        for handler in self.config.get("cognitive_handlers", []):
            self.register_handler(handler, RouteType.COGNITIVE)
            
        # Load operational routing handlers
        for handler in self.config.get("operational_handlers", []):
            self.register_handler(handler, RouteType.OPERATIONAL)
            
    def register_handler(self, handler: Callable, route_type: RouteType):
        """Register a new routing handler.
        
        Args:
            handler: Handler function to register
            route_type: Type of routing this handler supports
        """
        handler_name = handler.__name__
        if route_type == RouteType.COGNITIVE:
            self.cognitive_handlers[handler_name] = handler
        else:
            self.operational_handlers[handler_name] = handler
        logger.info(f"Registered {route_type.value} handler: {handler_name}")
        
    @handle_routing_errors
    def route(self, task: Dict[str, Any]) -> RouteResult:
        """Route a task to the appropriate handler.
        
        Args:
            task: Task to route
            
        Returns:
            RouteResult containing routing information
        """
        # Determine route type
        route_type = self._determine_route_type(task)
        
        # Get appropriate handlers
        handlers = (self.cognitive_handlers if route_type == RouteType.COGNITIVE 
                   else self.operational_handlers)
        
        # Find best matching handler
        best_handler = None
        best_confidence = 0.0
        
        for handler in handlers.values():
            confidence = self._calculate_confidence(handler, task)
            if confidence > best_confidence:
                best_handler = handler
                best_confidence = confidence
                
        if best_handler and best_confidence >= self.config.get("min_confidence", 0.5):
            return RouteResult(
                success=True,
                route_type=route_type,
                handler=best_handler,
                confidence=best_confidence,
                message=f"Routed to {best_handler.__name__}"
            )
            
        return RouteResult(
            success=False,
            route_type=route_type,
            message="No suitable handler found"
        )
        
    def _determine_route_type(self, task: Dict[str, Any]) -> RouteType:
        """Determine the type of routing needed for a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            RouteType indicating the type of routing needed
        """
        # Check for operational task indicators
        if any(key in task for key in ["schedule", "monitor", "trigger"]):
            return RouteType.OPERATIONAL
            
        # Default to cognitive routing
        return RouteType.COGNITIVE
        
    def _calculate_confidence(self, handler: Callable, task: Dict[str, Any]) -> float:
        """Calculate confidence score for a handler.
        
        Args:
            handler: Handler to evaluate
            task: Task to route
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get handler metadata
        metadata = getattr(handler, "_routing_metadata", {})
        
        # Check task type match
        if metadata.get("task_type") == task.get("type"):
            return 1.0
            
        # Check keyword matches
        keywords = metadata.get("keywords", [])
        task_text = str(task).lower()
        matches = sum(1 for kw in keywords if kw.lower() in task_text)
        
        return min(1.0, matches / max(1, len(keywords)))
        
    def get_route_info(self) -> Dict[str, Any]:
        """Get information about available routes.
        
        Returns:
            Dictionary containing routing information
        """
        return {
            "cognitive_handlers": list(self.cognitive_handlers.keys()),
            "operational_handlers": list(self.operational_handlers.keys()),
            "config": self.config
        } 