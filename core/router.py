"""Unified routing module that combines cognitive and operational routing."""

from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path

from core.agents.router import AgentRouter as CognitiveRouter
from system.infra.agents.infra_router import InfraRouter as OperationalRouter

logger = logging.getLogger(__name__)

class UnifiedRouter:
    """Combines cognitive and operational routing for comprehensive task handling."""
    
    def __init__(self):
        """Initialize the unified router with both cognitive and operational components."""
        self.cognitive_router = CognitiveRouter()
        self.operational_router = OperationalRouter()
        self.logger = logging.getLogger(__name__)
        
    def route_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route a task to either cognitive or operational handling.
        
        Args:
            task: Task specification including type, priority, and requirements
            
        Returns:
            Dict containing routing result and execution plan
        """
        try:
            # Determine if task is cognitive or operational
            task_type = task.get('type', '').lower()
            
            if task_type in ['planning', 'decision', 'learning', 'goal']:
                # Route to cognitive system
                result = self.cognitive_router.route(task)
                self.logger.info(f"Cognitive task routed: {task_type}")
                return {
                    'status': 'success',
                    'router': 'cognitive',
                    'result': result
                }
            elif task_type in ['schedule', 'monitor', 'automate', 'maintain']:
                # Route to operational system
                result = self.operational_router.route(task)
                self.logger.info(f"Operational task routed: {task_type}")
                return {
                    'status': 'success',
                    'router': 'operational',
                    'result': result
                }
            else:
                # Try both routers if type is ambiguous
                cognitive_result = self.cognitive_router.route(task)
                operational_result = self.operational_router.route(task)
                
                # Choose the best match based on confidence
                if cognitive_result.get('confidence', 0) > operational_result.get('confidence', 0):
                    return {
                        'status': 'success',
                        'router': 'cognitive',
                        'result': cognitive_result
                    }
                else:
                    return {
                        'status': 'success',
                        'router': 'operational',
                        'result': operational_result
                    }
                    
        except Exception as e:
            self.logger.error(f"Error routing task: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_route_info(self, task_type: str) -> Dict[str, Any]:
        """Get information about how a task type would be routed.
        
        Args:
            task_type: Type of task to get routing information for
            
        Returns:
            Dict containing routing information
        """
        return {
            'cognitive_handlers': self.cognitive_router.get_handlers(),
            'operational_handlers': self.operational_router.get_handlers(),
            'suggested_router': 'cognitive' if task_type in ['planning', 'decision', 'learning', 'goal']
                              else 'operational' if task_type in ['schedule', 'monitor', 'automate', 'maintain']
                              else 'both'
        }
        
    def register_handler(self, task_type: str, handler: callable, router: str = 'auto') -> None:
        """Register a new task handler.
        
        Args:
            task_type: Type of task the handler can process
            handler: Callable that processes the task
            router: Which router to register with ('cognitive', 'operational', or 'auto')
        """
        if router == 'auto':
            router = 'cognitive' if task_type in ['planning', 'decision', 'learning', 'goal'] else 'operational'
            
        if router == 'cognitive':
            self.cognitive_router.register_handler(task_type, handler)
        else:
            self.operational_router.register_handler(task_type, handler)
            
        self.logger.info(f"Registered handler for {task_type} with {router} router") 